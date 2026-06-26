"""S27 PR1 — Validate the realistic DJ30 panel loader.

Covers AC1.1–AC1.7:

* AC1.1 deterministic — two calls return frame-equal DataFrames.
* AC1.2 schema       — columns ``{ticker, session_date, price}`` with
                       pinned dtypes (object/string, datetime64[ns],
                       float64) for direct ``DataFrameSnapshotLoader``
                       consumption.
* AC1.3 universe     — unique tickers equal the sorted DJ30 set exactly
                       (no silent drop, no silent introduction).
* AC1.4 window       — ``session_date`` within ``[START_DATE, END_DATE]``.
* AC1.5 finite-data  — no NaN/inf/non-positive prices, no duplicate
                       ``(ticker, session_date)`` pairs.
* AC1.6 PIT alignment — per-ticker ``session_date`` strictly increasing,
                       no row outside the pinned window.
* AC1.7 no-mutation  — ``quantdata/quant.duckdb`` mtime unchanged across
                       the loader call (read-only invariant).

Run:
  uv run --directory quantstrat pytest tests/research/test_realistic_panel.py -x
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research._realistic_panel import (
    DUCKDB_PATH,
    END_DATE,
    START_DATE,
    _load_dj30_tickers,
    load_dj30_panel,
    pivot_to_wide,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not DUCKDB_PATH.exists():
        pytest.fail(f"data-lake fixture missing: catalog not found at {DUCKDB_PATH}")
    return load_dj30_panel()


# ─── AC1.1 — deterministic ─────────────────────────────────────────
def test_ac1_1_deterministic_repeated_call() -> None:
    a = load_dj30_panel()
    b = load_dj30_panel()
    assert_frame_equal(a, b, check_exact=True)


# ─── AC1.2 — schema ────────────────────────────────────────────────
def test_ac1_2_schema_columns_and_dtypes(panel: pd.DataFrame) -> None:
    assert set(panel.columns) == {"ticker", "session_date", "price"}
    # ticker: object/string-like
    assert panel["ticker"].dtype == np.dtype("O")
    assert all(isinstance(t, str) for t in panel["ticker"].unique())
    # session_date: datetime64[ns]
    assert panel["session_date"].dtype == np.dtype("datetime64[ns]")
    # price: float64
    assert panel["price"].dtype == np.dtype("float64")


# ─── AC1.3 — universe ──────────────────────────────────────────────
def test_ac1_3_universe_equals_sorted_dj30(panel: pd.DataFrame) -> None:
    expected = sorted(_load_dj30_tickers())
    actual = sorted(panel["ticker"].unique().tolist())
    assert actual == expected, (
        f"universe mismatch: "
        f"missing={sorted(set(expected) - set(actual))}, "
        f"extra={sorted(set(actual) - set(expected))}"
    )


# ─── AC1.4 — window ────────────────────────────────────────────────
def test_ac1_4_window_within_pinned_range(panel: pd.DataFrame) -> None:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    assert panel["session_date"].min() >= start
    assert panel["session_date"].max() <= end


# ─── AC1.5 — finite-data ──────────────────────────────────────────
def test_ac1_5_no_nan_inf_or_nonpositive(panel: pd.DataFrame) -> None:
    price = panel["price"].to_numpy()
    assert not panel["price"].isna().any()
    assert np.isfinite(price).all()
    assert (price > 0).all()


def test_ac1_5_no_duplicate_pairs(panel: pd.DataFrame) -> None:
    assert not panel.duplicated(subset=["ticker", "session_date"]).any()


def test_ac1_5_loader_rejects_injected_nan() -> None:
    """The loader's invariant is asserted by `_validate_panel`. Smoke-check
    that the validator (the loader's last step) raises on a corrupted frame.
    """
    from research._realistic_panel import _validate_panel

    bad = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "session_date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "price": [100.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="non-finite prices"):
        _validate_panel(bad, expected_tickers=["AAPL"])

    bad_neg = bad.copy()
    bad_neg["price"] = [100.0, -1.0]
    with pytest.raises(ValueError, match="non-positive prices"):
        _validate_panel(bad_neg, expected_tickers=["AAPL"])

    bad_dup = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "session_date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "price": [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_panel(bad_dup, expected_tickers=["AAPL"])


# ─── AC1.6 — PIT / monotonic ──────────────────────────────────────
def test_ac1_6_per_ticker_strictly_increasing(panel: pd.DataFrame) -> None:
    for ticker, group in panel.groupby("ticker", sort=False):
        dates = group["session_date"].to_numpy()
        diffs = np.diff(dates)
        # all positive (strictly increasing); diffs are timedelta64[ns]
        assert (diffs > np.timedelta64(0, "ns")).all(), f"non-monotonic session_date for {ticker!r}"


def test_ac1_6_no_row_outside_pinned_window(panel: pd.DataFrame) -> None:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    outside = panel[(panel["session_date"] < start) | (panel["session_date"] > end)]
    assert outside.empty, f"{len(outside)} rows outside [{START_DATE}, {END_DATE}]"


# ─── AC1.7 — no mutation (read-only) ──────────────────────────────
def test_ac1_7_duckdb_mtime_unchanged() -> None:
    before = DUCKDB_PATH.stat().st_mtime
    _ = load_dj30_panel()
    after = DUCKDB_PATH.stat().st_mtime
    assert before == after, f"quant.duckdb mtime changed across loader call: {before} -> {after}"


# ─── pivot helper smoke ──────────────────────────────────────────
def test_pivot_to_wide_shape_and_index(panel: pd.DataFrame) -> None:
    wide = pivot_to_wide(panel)
    assert wide.index.name == "session_date"
    assert wide.columns.name == "ticker"
    expected_tickers = sorted(_load_dj30_tickers())
    assert sorted(wide.columns.tolist()) == expected_tickers
