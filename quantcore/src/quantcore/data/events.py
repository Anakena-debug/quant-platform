"""Typed market-event hierarchy for the L3 / OOP refactor (S33 §3.AC1).

Pure frozen+slotted data shape. No validation, no methods. Per S33
§5.D1, ABCs go on machines with behaviour to enforce — not on data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Side(IntEnum):
    """Order/trade side. ``int(Side.BID)`` == +1, ``int(Side.ASK)`` == -1."""

    BID = 1
    ASK = -1


class Action(IntEnum):
    """L3 / MBO event action codes."""

    ADD = 1
    CANCEL = 2
    MODIFY = 3
    TRADE = 4
    FILL = 5
    CLEAR = 6


@dataclass(frozen=True, slots=True)
class BaseEvent:
    ts_event: int
    instrument_id: int
    sequence: int


@dataclass(frozen=True, slots=True)
class TradeEvent(BaseEvent):
    price: float
    size: float
    aggressor_side: Side
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    bid_sz: float = float("nan")
    ask_sz: float = float("nan")


@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    action: Action
    side: Side
    order_id: int
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    ts_event: int
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
