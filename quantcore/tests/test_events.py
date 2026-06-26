"""Tests for ``quantcore.data.events`` (S33 AC1, AC6).

Verifies:
- Side / Action IntEnum values
- Construction of all four dataclasses
- ``frozen=True`` rejects mutation
- ``slots=True`` prevents __dict__ allocation
- ``isinstance`` dispatch works across BaseEvent / TradeEvent / OrderEvent
- AC4 demo: TradeEvent and OrderEvent reachable from the same
  ``quantcore.data`` namespace (one-line swap)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

# AC4: single import line; swap TradeEvent <-> OrderEvent by changing
# only the constructor used in user code.
from quantcore.data import (
    Action,
    BaseEvent,
    BookSnapshot,
    OrderEvent,
    Side,
    TradeEvent,
)


def test_side_enum_values() -> None:
    assert int(Side.BID) == 1
    assert int(Side.ASK) == -1


def test_action_enum_values() -> None:
    assert int(Action.ADD) == 1
    assert int(Action.CANCEL) == 2
    assert int(Action.MODIFY) == 3
    assert int(Action.TRADE) == 4
    assert int(Action.FILL) == 5
    assert int(Action.CLEAR) == 6


def test_trade_event_construction() -> None:
    t = TradeEvent(
        ts_event=1_000_000_000,
        instrument_id=42,
        sequence=1,
        price=100.5,
        size=10.0,
        aggressor_side=Side.BID,
    )
    assert t.ts_event == 1_000_000_000
    assert t.instrument_id == 42
    assert t.sequence == 1
    assert t.price == 100.5
    assert t.size == 10.0
    assert t.aggressor_side == Side.BID


def test_order_event_construction() -> None:
    o = OrderEvent(
        ts_event=1,
        instrument_id=42,
        sequence=1,
        action=Action.ADD,
        side=Side.ASK,
        order_id=7,
        price=100.0,
        size=5.0,
    )
    assert o.action == Action.ADD
    assert o.side == Side.ASK
    assert o.order_id == 7


def test_book_snapshot_construction() -> None:
    snap = BookSnapshot(
        ts_event=1,
        bid_px=np.array([100.0, 99.0], dtype=np.float64),
        bid_sz=np.array([10.0, 20.0], dtype=np.float64),
        ask_px=np.array([101.0, 102.0], dtype=np.float64),
        ask_sz=np.array([5.0, 15.0], dtype=np.float64),
    )
    assert snap.ts_event == 1
    assert snap.bid_px.shape == (2,)


def test_frozen_trade_event_rejects_mutation() -> None:
    t = TradeEvent(1, 1, 1, 100.0, 1.0, Side.BID)
    with pytest.raises(FrozenInstanceError):
        t.price = 200.0  # type: ignore[misc]


def test_frozen_order_event_rejects_mutation() -> None:
    o = OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 100.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        o.size = 99.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "cls, args",
    [
        (BaseEvent, (1, 1, 1)),
        (TradeEvent, (1, 1, 1, 100.0, 1.0, Side.BID)),
        (
            OrderEvent,
            (1, 1, 1, Action.ADD, Side.BID, 1, 100.0, 1.0),
        ),
    ],
)
def test_dataclasses_use_slots(cls: type, args: tuple) -> None:
    """AC6 — frozen+slots enforced. No __dict__ on instances."""
    inst = cls(*args)
    assert hasattr(cls, "__slots__"), f"{cls.__name__} missing __slots__"
    assert not hasattr(inst, "__dict__"), f"{cls.__name__} instance has __dict__ — slots dropped"


def test_book_snapshot_uses_slots() -> None:
    snap = BookSnapshot(
        ts_event=1,
        bid_px=np.array([100.0], dtype=np.float64),
        bid_sz=np.array([10.0], dtype=np.float64),
        ask_px=np.array([101.0], dtype=np.float64),
        ask_sz=np.array([5.0], dtype=np.float64),
    )
    assert hasattr(BookSnapshot, "__slots__")
    assert not hasattr(snap, "__dict__")


def test_isinstance_dispatch() -> None:
    """AC3 readiness — isinstance() narrows BaseEvent into the right child."""
    t = TradeEvent(1, 1, 1, 100.0, 1.0, Side.BID)
    o = OrderEvent(1, 1, 1, Action.ADD, Side.BID, 1, 100.0, 1.0)
    b = BaseEvent(1, 1, 1)
    assert isinstance(t, BaseEvent)
    assert isinstance(o, BaseEvent)
    assert isinstance(b, BaseEvent)
    assert isinstance(t, TradeEvent)
    assert not isinstance(t, OrderEvent)
    assert isinstance(o, OrderEvent)
    assert not isinstance(o, TradeEvent)


def test_baseevent_is_not_abc() -> None:
    """AC2 / §5.D1 — BaseEvent is data, not an ABC."""
    import abc

    assert abc.ABC not in BaseEvent.__mro__
    assert type(BaseEvent) is type
