"""
End-to-end test: DuckDBSnapshotLoader + DuckDBUniverseResolver against the
``build_mini_duckdb`` fixture.

Drop location
-------------
    quantengine/tests/test_duckdb_loader_roundtrip.py

What it verifies
----------------
1.  **PIT no-leak**: query at t returns only rows with session_date <= t.
2.  **Survivorship safety**:
    * TSLA (delisted 2026-01-20) IS in universe on 2026-01-19.
    * TSLA is NOT in universe on 2026-01-20.
    * NVDA (IPO 2026-01-15) is NOT in universe on 2026-01-14.
    * NVDA IS in universe on 2026-01-15.
3.  **Stale-tolerance**: GE price at as_of=2026-01-10 returns the most-recent
    2026-01-09 close (gap logic handled by pit_filter).
4.  **Parity with DataFrame backend**: the DuckDB query path produces the
    same PIT snapshot as ``pit_filter`` applied to the full prices DataFrame.
5.  **Universe intersection**: snapshot is intersected with membership —
    a ticker with a price row on D but member=False on D must be dropped.

Runbook (assumes quantengine venv with duckdb + pandas + pyarrow installed)
--------------------------------------------------------------------------
    pytest tests/test_duckdb_loader_roundtrip.py -v

Limits
------
* Fixture prices are deterministic, not adjusted for corp actions. Reader
  paths that enforce corp_actions must be tested separately.
* Test assumes ``pit_filter`` default ``price_col="close"`` and date column
  ``"session_date"``. If the reader's defaults drift, override explicitly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantengine.data.snapshot import DuckDBSnapshotLoader, pit_filter  # type: ignore
from quantengine.data.universe import DuckDBUniverseResolver  # type: ignore

from tests.fixtures.build_mini_duckdb import (
    build_mini_duckdb,
    build_price_frame,
    build_universe_frame,
)


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mini_duckdb_path():
    tmp = Path(tempfile.mkdtemp(prefix="qe_mini_duckdb_"))
    db = tmp / "mini.duckdb"
    build_mini_duckdb(db)
    yield db


@pytest.fixture(scope="module")
def prices_df():
    return build_price_frame()


@pytest.fixture(scope="module")
def universe_df():
    return build_universe_frame()


# ---------------------------------------------------------------------------
# Universe resolver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "as_of, expected",
    [
        ("2026-01-14", {"AAPL", "MSFT", "TSLA", "GE"}),  # NVDA not yet
        ("2026-01-15", {"AAPL", "MSFT", "TSLA", "NVDA", "GE"}),
        ("2026-01-19", {"AAPL", "MSFT", "TSLA", "NVDA", "GE"}),
        ("2026-01-20", {"AAPL", "MSFT", "NVDA", "GE"}),  # TSLA delisted
        ("2026-01-30", {"AAPL", "MSFT", "NVDA", "GE"}),
    ],
)
def test_universe_resolver_handles_entries_exits(
    mini_duckdb_path,
    as_of,
    expected,
):
    resolver = DuckDBUniverseResolver(db_path=str(mini_duckdb_path))
    members = set(resolver.resolve(pd.Timestamp(as_of)))
    assert members == expected, f"as_of={as_of}: expected {expected}, got {members}"


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------
def test_snapshot_loader_no_future_leak(mini_duckdb_path, prices_df):
    loader = DuckDBSnapshotLoader(
        db_path=str(mini_duckdb_path),
        price_table="daily_bars_adj",
        price_field="close",
    )
    as_of = pd.Timestamp("2026-01-15")
    tickers = ("AAPL", "MSFT", "GE")
    snap = loader.load(as_of=as_of, universe=tickers)

    # No row dated after as_of
    max_date = snap["session_date"].max()
    assert max_date <= as_of.date(), f"future leak: {max_date} > {as_of.date()}"

    # One row per ticker
    assert set(snap["ticker"]) == set(tickers)
    assert len(snap) == len(tickers)


def test_snapshot_parity_with_pandas_pit_filter(mini_duckdb_path, prices_df):
    loader = DuckDBSnapshotLoader(
        db_path=str(mini_duckdb_path),
        price_table="daily_bars_adj",
        price_field="close",
    )
    as_of = pd.Timestamp("2026-01-15")
    tickers = ("AAPL", "MSFT", "GE", "NVDA")

    snap_duck = loader.load(as_of=as_of, universe=tickers).sort_values("ticker")
    snap_pandas = pit_filter(
        prices_df,
        as_of=as_of,
        universe=tickers,
        price_col="close",
        ticker_col="ticker",
        date_col="session_date",
    ).sort_values("ticker")

    # Compare latest close per ticker across backends
    for col in ("ticker", "close"):
        assert list(snap_duck[col]) == list(snap_pandas[col]), f"mismatch on column {col!r}"


def test_ge_gap_handled_by_stale_tolerance(mini_duckdb_path):
    """GE has no bar 2026-01-10..12; as_of=2026-01-10 should serve 01-09 close."""
    loader = DuckDBSnapshotLoader(
        db_path=str(mini_duckdb_path),
        price_table="daily_bars_adj",
        price_field="close",
    )
    as_of = pd.Timestamp("2026-01-10")
    snap = loader.load(as_of=as_of, universe=("GE",))
    assert len(snap) == 1
    # Must be 2026-01-09 (the last business day before the gap)
    assert str(snap["session_date"].iloc[0]) == "2026-01-09"


def test_delisted_ticker_excluded_from_intersected_snapshot(mini_duckdb_path):
    """
    Combining snapshot + universe membership at 2026-01-20 must drop TSLA
    even though its prior prices are in the bars table.
    """
    resolver = DuckDBUniverseResolver(db_path=str(mini_duckdb_path))
    loader = DuckDBSnapshotLoader(
        db_path=str(mini_duckdb_path),
        price_table="daily_bars_adj",
        price_field="close",
    )
    as_of = pd.Timestamp("2026-01-20")
    active = resolver.resolve(as_of)
    snap = loader.load(as_of=as_of, universe=tuple(active))
    assert "TSLA" not in set(snap["ticker"])
    assert {"AAPL", "MSFT", "NVDA", "GE"}.issubset(set(snap["ticker"]))


def test_pre_ipo_ticker_absent(mini_duckdb_path):
    loader = DuckDBSnapshotLoader(
        db_path=str(mini_duckdb_path),
        price_table="daily_bars_adj",
        price_field="close",
    )
    as_of = pd.Timestamp("2026-01-14")
    snap = loader.load(as_of=as_of, universe=("NVDA",))
    # Either empty frame, or explicit ValueError — both acceptable contracts.
    assert len(snap) == 0 or "NVDA" not in set(snap["ticker"])


# ---------------------------------------------------------------------------
# Sanity: pit_filter alone (reference backend) honours as_of
# ---------------------------------------------------------------------------
def test_pit_filter_reference_no_future_leak(prices_df):
    as_of = pd.Timestamp("2026-01-12")
    out = pit_filter(
        prices_df,
        as_of=as_of,
        universe=("AAPL", "MSFT"),
        price_col="close",
        ticker_col="ticker",
        date_col="session_date",
    )
    assert (pd.to_datetime(out["session_date"]) <= as_of).all()
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
