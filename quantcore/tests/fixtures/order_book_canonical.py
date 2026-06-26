"""Hand-built canonical L3 stream for S33 §3.AC8.

Ten events covering ADD / CANCEL / FILL on both sides plus a TRADE
no-op. The expected final ``BookSnapshot`` is hand-computed below.
"""

from __future__ import annotations

import numpy as np

from quantcore.data import (
    Action,
    BaseEvent,
    BookSnapshot,
    OrderEvent,
    Side,
    TradeEvent,
)

INSTRUMENT_ID = 42

# Trace of state after each event (for debugging if assertions fire):
#  1. ADD bid @ 99.0  size 10.0  oid=1            bids: {99.0: {1: 10}}
#  2. ADD bid @ 98.0  size 20.0  oid=2            bids: {99: {1:10}, 98:{2:20}}
#  3. ADD ask @ 101.0 size 15.0  oid=3            asks: {101: {3:15}}
#  4. ADD ask @ 102.0 size 25.0  oid=4            asks: {101:{3:15}, 102:{4:25}}
#  5. ADD bid @ 99.0  size  5.0  oid=5            bids 99: {1:10, 5:5}
#  6. TRADE @ 99.0    size  3.0       (no-op; _last_ts unchanged at ts=5)
#  7. FILL  oid=1     size  4.0       bids 99: {1: 6, 5:5}
#  8. FILL  oid=5     size  5.0       bids 99: {1: 6}                (full fill)
#  9. CANCEL oid=4                    asks: {101: {3:15}}
# 10. ADD ask @ 103.0 size  8.0  oid=6 asks: {101: {3:15}, 103: {6:8}}
#
# Final book:
#   Bids:  99.0 -> 6.0  (oid=1 only)
#          98.0 -> 20.0 (oid=2)
#   Asks: 101.0 -> 15.0 (oid=3)
#         103.0 -> 8.0  (oid=6)
#   _last_ts = 10  (TradeEvent at ts=6 does NOT advance it; AC7 P5)


CANONICAL_EVENTS: list[BaseEvent] = [
    OrderEvent(1, INSTRUMENT_ID, 1, Action.ADD, Side.BID, 1, 99.0, 10.0),
    OrderEvent(2, INSTRUMENT_ID, 2, Action.ADD, Side.BID, 2, 98.0, 20.0),
    OrderEvent(3, INSTRUMENT_ID, 3, Action.ADD, Side.ASK, 3, 101.0, 15.0),
    OrderEvent(4, INSTRUMENT_ID, 4, Action.ADD, Side.ASK, 4, 102.0, 25.0),
    OrderEvent(5, INSTRUMENT_ID, 5, Action.ADD, Side.BID, 5, 99.0, 5.0),
    TradeEvent(6, INSTRUMENT_ID, 6, 99.0, 3.0, Side.BID),
    OrderEvent(7, INSTRUMENT_ID, 7, Action.FILL, Side.BID, 1, 99.0, 4.0),
    OrderEvent(8, INSTRUMENT_ID, 8, Action.FILL, Side.BID, 5, 99.0, 5.0),
    OrderEvent(9, INSTRUMENT_ID, 9, Action.CANCEL, Side.ASK, 4, 102.0, 25.0),
    OrderEvent(10, INSTRUMENT_ID, 10, Action.ADD, Side.ASK, 6, 103.0, 8.0),
]


EXPECTED_FINAL_SNAPSHOT = BookSnapshot(
    ts_event=10,
    bid_px=np.array([99.0, 98.0], dtype=np.float64),
    bid_sz=np.array([6.0, 20.0], dtype=np.float64),
    ask_px=np.array([101.0, 103.0], dtype=np.float64),
    ask_sz=np.array([15.0, 8.0], dtype=np.float64),
)
