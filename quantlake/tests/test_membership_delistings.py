"""Item 3 — announcement vs effective; delisting_return nullable-with-flag; PIT liquidity universe (B4)."""

from __future__ import annotations

import math

import pandas as pd

from quantlake.store.bitemporal import BitemporalStore
from quantlake.universe.membership import LIQUIDITY_LAG, Universe, build_liquidity_universe


def test_membership_announcement_distinct_from_effective():
    u = Universe(BitemporalStore())
    u.add_membership(1, "liq500", effective_date="2024-03-01", announcement_date="2024-02-15")
    row = u.store.as_of("universe_membership", "2024-03-15").iloc[0]
    assert pd.Timestamp(row["announcement_date"]) == pd.Timestamp("2024-02-15")
    assert pd.Timestamp(row["effective_date"]) == pd.Timestamp("2024-03-01")
    assert row["announcement_date"] != row["effective_date"]
    # known at announcement (kd), not only at effective
    assert not u.store.as_of("universe_membership", "2024-02-20").empty


def test_delisting_return_nullable_with_flag():
    u = Universe(BitemporalStore())
    u.set_status(7, "delisted", effective_date="2023-09-01", announcement_date="2023-08-25")
    row = u.store.as_of("instrument_status", "2023-10-01").iloc[0]
    assert row["status"] == "delisted"
    assert row["delisting_return"] is None or (
        isinstance(row["delisting_return"], float) and math.isnan(row["delisting_return"])
    )
    assert row["_source_flag"] == "unsourced"  # flagged, not silently absent


def test_pit_liquidity_universe_window_ends_before_effective():
    dates = pd.bdate_range("2023-01-02", periods=200)
    rows = []
    for qid, vol in [(1, 100.0), (2, 50.0), (3, 10.0)]:  # 1 most liquid, 3 least
        for d in dates:
            rows.append({"quantlake_id": qid, "event_date": d, "close": 10.0, "volume": vol})
    prices = pd.DataFrame(rows)
    eff = dates[-1] + pd.tseries.offsets.BDay(1)  # effective AFTER all sampled sessions
    uni = build_liquidity_universe(prices, eff, top_n=2, window=126, lag=LIQUIDITY_LAG)
    assert uni.window_end < uni.effective_date  # B4: strictly PIT
    # lag sessions of gap: window_end is the (lag+1)-th-from-last eligible session
    eligible = dates[dates < eff]
    assert uni.window_end == eligible[-(LIQUIDITY_LAG + 1)]
    assert uni.members == (1, 2)  # top-2 by trailing median dollar-volume
