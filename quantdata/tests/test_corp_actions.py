"""quantdata — unit tests for corporate-action split detection + back-adjustment (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlake.store.corp_actions import adjust_splits, detect_splits, residual_extreme_return_rate


def test_detect_clean_forward_split():
    # 4 days ~1000, then a 10:1 split: open gaps to ~100, trades flat intraday, volume jumps 10x.
    close = [1000, 1010, 990, 1000, 101, 100, 102]
    open_ = [1000, 1010, 990, 1000, 100, 101, 100]
    vol = [1000, 1000, 1000, 1000, 10000, 9000, 9500]
    ev = detect_splits(np.array(open_, float), np.array(close, float), np.array(vol, float))
    assert ev == [(4, 10.0)]


def test_detect_ignores_intraday_crash():
    # a 50% INTRADAY crash (open flat, close halves) is NOT a split (it moves intraday).
    close = [1000, 1010, 990, 1000, 500, 510, 495]
    open_ = [1000, 1010, 990, 1000, 1000, 505, 500]
    vol = [1000.0] * 7
    assert detect_splits(np.array(open_, float), np.array(close, float), np.array(vol, float)) == []


def test_detect_reverse_split():
    # 1:10 reverse: price RISES ~10x at the open, trades flat intraday.
    close = [10, 11, 9, 10, 101, 100, 102]
    open_ = [10, 11, 9, 10, 100, 101, 100]
    vol = [1000.0] * 7
    ev = detect_splits(np.array(open_, float), np.array(close, float), np.array(vol, float))
    assert ev == [(4, 0.1)]  # ratio 1/10


def test_adjust_splits_back_adjustment_convention():
    close = [1000, 1010, 990, 1000, 101, 100, 102]
    open_ = [1000, 1010, 990, 1000, 100, 101, 100]
    vol = [1000, 1000, 1000, 1000, 10000, 9000, 9500]
    df = pd.DataFrame(
        {
            "instrument_id": 1,
            "date": pd.bdate_range("2024-01-01", periods=7),
            "open": open_,
            "high": close,
            "low": close,
            "close": close,
            "volume": vol,
        }
    )
    adj = adjust_splits(df).sort_values("date")
    ret = adj["adj_close"].pct_change().to_numpy()
    assert abs(ret[4]) < 0.05  # the -90% raw split-day return is adjusted away
    assert np.isclose(adj["adj_close"].iloc[-1], 102.0)  # recent unchanged
    assert np.isclose(adj["adj_close"].iloc[0], 100.0)  # 1000 / 10 (one future split)
    assert np.isclose(adj["adj_factor"].iloc[0], 10.0)  # pre-split divisor
    assert np.isclose(adj["adj_factor"].iloc[-1], 1.0)  # recent factor is 1


def test_residual_extreme_rate_counts_unadjusted_moves():
    # a genuine -60% crash (not a clean split ratio) stays in adj_close -> counted as residual.
    df = pd.DataFrame(
        {
            "instrument_id": 1,
            "date": pd.bdate_range("2024-01-01", periods=5),
            "adj_close": [100.0, 100.0, 100.0, 40.0, 40.0],  # -60% on day 3
        }
    )
    rate = residual_extreme_return_rate(df, key="instrument_id", threshold=0.5)
    assert rate == 0.25  # 1 of 4 daily returns exceeds 50%
