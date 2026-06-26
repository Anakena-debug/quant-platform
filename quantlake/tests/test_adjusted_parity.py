"""s80 D3 — adjusted-column parity: pinned to the EXACT per-name factor set, not an aggregate bound.

The store's ``adjusted_prices`` view applies stored corp_actions (kd<=as_of). On a known 2:1 split the
per-instrument ``adj_factor`` vector is exact; prices stay byte-identical (s78). This is the adjusted half
of item 10a — a different heuristic could hit the same aggregate residual while flipping per-name factors,
so the per-name vector is pinned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlake.store.bitemporal import BitemporalStore


def _store_with_split() -> BitemporalStore:
    s = BitemporalStore()
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = [100.0] * 5 + [50.0] * 5  # 2:1 split at index 5
    s.append(
        "prices",
        pd.DataFrame(
            {
                "quantlake_id": 1,
                "event_date": dates,
                "knowledge_date": dates,  # session-close availability
                "close": close,
            }
        ),
    )
    s.append(
        "corp_actions",
        pd.DataFrame(
            [
                {
                    "quantlake_id": 1,
                    "type": "split",
                    "event_date": dates[5],  # ex-date
                    "knowledge_date": dates[5],
                    "raw_factor": 2.0,
                }
            ]
        ),
    )
    return s


def test_adjusted_prices_exact_per_name_factor_set():
    s = _store_with_split()
    adj = s.adjusted_prices(pd.Timestamp("2024-01-16")).sort_values("event_date")
    # pre-split (event_date < ex) divided down; post-split unchanged -> adj_close flat, factor exact.
    np.testing.assert_array_equal(
        adj["adj_factor"].to_numpy(), [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(adj["adj_close"].to_numpy(), [50.0] * 10)


def test_raw_prices_byte_identical_through_store():
    s = _store_with_split()
    got = s.as_of("prices", pd.Timestamp("2024-01-16")).sort_values("event_date")
    np.testing.assert_array_equal(
        got["close"].to_numpy(), [100.0] * 5 + [50.0] * 5
    )  # raw unchanged
