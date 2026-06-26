"""s79 — API-level migration parity (note 2).

Pins the corp-actions behavior as an IDENTICAL PER-NAME RESIDUAL SET (the exact (index, ratio)
detections + back-adjustment factors), NOT a <=0.22% aggregate — a different heuristic could hit the
same aggregate residual while flipping individual names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlake.store.corp_actions import adjust_splits, detect_splits, residual_extreme_return_rate
from quantlake.store.panel import build_panel


def _two_for_one_fixture() -> pd.DataFrame:
    # Instrument A: a clean 2:1 split at index 5 (gap 100->50 at the open, flat intraday, vol confirms).
    # Instrument B: no split (flat 100). raw, split-unadjusted.
    a_close = [100.0] * 5 + [50.0] * 5
    a = pd.DataFrame(
        {
            "instrument_id": 1,
            "raw_symbol": "AAA",
            "date": pd.bdate_range("2024-01-02", periods=10),
            "open": a_close,
            "high": a_close,
            "low": a_close,
            "close": a_close,
            "volume": [1000.0] * 10,
        }
    )
    b = a.copy()
    b["instrument_id"] = 2
    b["raw_symbol"] = "BBB"
    b[["open", "high", "low", "close"]] = 100.0  # no split
    return pd.concat([a, b], ignore_index=True)


def test_detect_splits_identical_residual_set():
    panel = _two_for_one_fixture()
    a = panel[panel["instrument_id"] == 1]
    b = panel[panel["instrument_id"] == 2]
    # EXACT per-name detection set — not an aggregate rate.
    assert detect_splits(a["open"].to_numpy(), a["close"].to_numpy(), a["volume"].to_numpy()) == [
        (5, 2.0)
    ]
    assert detect_splits(b["open"].to_numpy(), b["close"].to_numpy(), b["volume"].to_numpy()) == []


def test_adjust_splits_back_adjustment_exact():
    adj = adjust_splits(_two_for_one_fixture())
    a = adj[adj["instrument_id"] == 1].sort_values("date")
    # back-adjustment: 2x before the split, 1x after; adj_close flat at 50 (recent unchanged).
    np.testing.assert_array_equal(a["adj_factor"].to_numpy(), [2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(a["adj_close"].to_numpy(), [50.0] * 10)
    assert a["split_ratio"].to_numpy()[5] == 2.0


def test_panel_and_residual_rate_exact():
    panel = build_panel(_two_for_one_fixture())
    # adjusted close is flat -> zero extreme adjusted returns (the residual-error bound).
    rate = residual_extreme_return_rate(panel.assign(adj_close=panel["close"]), key="ticker")
    assert rate == 0.0
    assert (panel["close"] > 0).all() and not panel.duplicated(subset=["ticker", "date"]).any()
