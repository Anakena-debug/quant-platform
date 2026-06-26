"""quantdata — unit tests for the survivorship-free PIT panel builder (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlake.store.panel import build_panel, delisting_metadata, validate_panel


def _raw_with_split_and_delisting() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=12)
    # instrument 1: SURVIVOR with a 2:1 split mid-series (open gaps, flat intraday, vol jumps)
    c1 = [100, 101, 99, 100, 50, 51, 49, 50, 51, 50, 52, 51]
    o1 = [100, 101, 99, 100, 50, 51, 49, 50, 51, 50, 52, 51]
    o1[4] = 50  # split day opens at the split-adjusted level (gap from prev close 100)
    v1 = [1000] * 12
    v1[4] = 2200  # share volume ~doubles on the 2:1 split
    surv = pd.DataFrame(
        {
            "instrument_id": 1,
            "raw_symbol": "SURV",
            "date": dates,
            "open": o1,
            "high": c1,
            "low": c1,
            "close": c1,
            "volume": v1,
        }
    )
    # instrument 2: DELISTED — bars stop at day 4 (survivorship-free requires retaining it)
    dead = pd.DataFrame(
        {
            "instrument_id": 2,
            "raw_symbol": "DEAD",
            "date": dates[:4],
            "open": 20.0,
            "high": 20.0,
            "low": 20.0,
            "close": 20.0,
            "volume": 500.0,
        }
    )
    return pd.concat([surv, dead], ignore_index=True)


def test_build_panel_contract_and_adjustment():
    panel = build_panel(_raw_with_split_and_delisting())
    # s59 column contract
    assert {
        "ticker",
        "raw_symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
    } <= set(panel.columns)
    assert (panel["dividends"] == 0.0).all() and (panel["stock_splits"] == 0.0).all()
    # the 2:1 split on instrument 1 is adjusted away (no ~-50% day in adjusted close)
    surv = panel[panel["ticker"] == 1].sort_values("date")
    ret = surv["close"].pct_change().abs()
    assert ret.max() < 0.10
    # OHLC adjusted consistently: pre-split close 100 -> 50 (divided by the 2x future split)
    assert np.isclose(surv["close"].iloc[0], 50.0)
    assert np.isclose(surv["open"].iloc[0], 50.0)  # open adjusted by the same factor


def test_delisting_metadata_flags_delisted():
    panel = build_panel(_raw_with_split_and_delisting())
    meta = delisting_metadata(panel, gap_days=2).set_index("ticker")
    assert meta.loc[2, "delisted"]  # DEAD stopped early -> delisted
    assert not meta.loc[1, "delisted"]  # SURV runs to the end
    assert meta.loc[2, "last_date"] < meta.loc[1, "last_date"]


def test_validate_panel_survivorship_free():
    panel = build_panel(_raw_with_split_and_delisting())
    rep = validate_panel(panel, gap_days=2)  # short synthetic window
    assert rep["survivorship_free"] is True  # delisted name retained
    assert rep["n_delisted"] >= 1
    assert rep["all_prices_positive"] is True
    assert rep["no_future_dates"] is True
    assert rep["unique_ticker_date"] is True
