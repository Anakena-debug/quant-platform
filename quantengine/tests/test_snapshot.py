"""Smoke tests for quantengine.data.snapshot.

The DataFrame backend shares ``pit_filter`` with the DuckDB backend, so
these tests prove the correctness invariants once for both.

Covers:
    - PIT leak: no row with session_date > as_of survives.
    - Latest-row selection within the PIT window.
    - Universe intersection: extra tickers in data dropped; missing
      tickers silently dropped.
    - Non-positive / NaN prices filtered.
    - Empty universe → ValueError.
    - Empty result set (no tickers priced) → ValueError.
    - MarketSnapshot dtype and ordering invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantengine.data.snapshot import (
    DataFrameSnapshotLoader,
    pit_filter,
)


def _toy_prices() -> pd.DataFrame:
    rows = [
        # (ticker, session_date, price)
        ("AAPL", "2026-04-14", 198.0),
        ("AAPL", "2026-04-15", 199.0),
        ("AAPL", "2026-04-16", 200.0),  # latest <= as_of
        ("AAPL", "2026-04-17", 201.0),  # future wrt some as_of
        ("MSFT", "2026-04-15", 395.0),
        ("MSFT", "2026-04-16", 400.0),
        ("NVDA", "2026-04-10", 98.0),
        ("NVDA", "2026-04-16", 100.0),
        ("SPY", "2026-04-16", 500.0),
        # Edge cases we must drop.
        ("DEAD", "2026-04-16", 0.0),  # non-positive price
        ("NANP", "2026-04-16", float("nan")),  # NaN
    ]
    return pd.DataFrame(rows, columns=["ticker", "session_date", "price"])


# ---------------------------------------------------------------------------
# pit_filter contract
# ---------------------------------------------------------------------------
def test_pit_filter_drops_future_rows():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("AAPL", "MSFT"))
    assert (out["session_date"] <= as_of).all()
    # AAPL 2026-04-17 must NOT appear.
    aapl = out[out["ticker"] == "AAPL"]
    assert aapl["session_date"].iloc[0] == pd.Timestamp("2026-04-16")


def test_pit_filter_picks_latest_in_window():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("AAPL",))
    assert len(out) == 1
    assert out["price"].iloc[0] == 200.0  # not 198 or 199.


def test_pit_filter_drops_non_positive_and_nan():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("DEAD", "NANP", "AAPL"))
    assert set(out["ticker"]) == {"AAPL"}


def test_pit_filter_drops_tickers_not_in_universe():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("AAPL",))
    assert set(out["ticker"]) == {"AAPL"}


def test_pit_filter_drops_universe_tickers_without_data():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("AAPL", "UNLISTED"))
    # UNLISTED silently dropped, no error.
    assert set(out["ticker"]) == {"AAPL"}


def test_pit_filter_output_sorted_and_unique():
    as_of = pd.Timestamp("2026-04-16")
    out = pit_filter(_toy_prices(), as_of=as_of, universe=("SPY", "AAPL", "NVDA", "MSFT"))
    # Tickers sorted lexicographically.
    assert list(out["ticker"]) == ["AAPL", "MSFT", "NVDA", "SPY"]
    # Unique tickers.
    assert out["ticker"].is_unique


# ---------------------------------------------------------------------------
# DataFrameSnapshotLoader
# ---------------------------------------------------------------------------
def test_dataframe_loader_returns_market_snapshot():
    loader = DataFrameSnapshotLoader(prices=_toy_prices())
    ms = loader.load(pd.Timestamp("2026-04-16"), ("AAPL", "MSFT", "NVDA", "SPY"))
    assert ms.tickers == ("AAPL", "MSFT", "NVDA", "SPY")
    assert np.all(ms.prices > 0)
    assert ms.prices.dtype == np.float64
    # Metadata pins provenance.
    assert ms.metadata["source"] == "DataFrame"
    assert ms.metadata["n_tickers"] == 4


def test_dataframe_loader_empty_universe_raises():
    loader = DataFrameSnapshotLoader(prices=_toy_prices())
    raised = False
    try:
        loader.load(pd.Timestamp("2026-04-16"), ())
    except ValueError:
        raised = True
    assert raised


def test_dataframe_loader_no_priced_tickers_raises():
    """If the PIT window is before all data, we must fail loudly."""
    loader = DataFrameSnapshotLoader(prices=_toy_prices())
    raised = False
    try:
        loader.load(pd.Timestamp("1999-01-01"), ("AAPL",))
    except ValueError:
        raised = True
    assert raised


def test_dataframe_loader_prices_match_latest_rows():
    loader = DataFrameSnapshotLoader(prices=_toy_prices())
    ms = loader.load(pd.Timestamp("2026-04-16"), ("AAPL", "MSFT", "NVDA", "SPY"))
    expected = {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0, "SPY": 500.0}
    for t, p in zip(ms.tickers, ms.prices):
        assert p == expected[t]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_pit_filter_drops_future_rows,
        test_pit_filter_picks_latest_in_window,
        test_pit_filter_drops_non_positive_and_nan,
        test_pit_filter_drops_tickers_not_in_universe,
        test_pit_filter_drops_universe_tickers_without_data,
        test_pit_filter_output_sorted_and_unique,
        test_dataframe_loader_returns_market_snapshot,
        test_dataframe_loader_empty_universe_raises,
        test_dataframe_loader_no_priced_tickers_raises,
        test_dataframe_loader_prices_match_latest_rows,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\ndata.snapshot: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
