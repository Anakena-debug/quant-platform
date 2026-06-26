"""Tests for quantcore.data.bars (S34 AC1, AC6, AC9)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quantcore.data import Bar, BarKind, BaseEvent, TradeEvent


def _make_bar(**overrides) -> Bar:
    fields = dict(
        ts_event=10,
        instrument_id=1,
        sequence=1,
        ts_open=5,
        kind=BarKind.TICK,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        vwap=100.2,
        tick_count=3,
        dollar_volume=1002.0,
        # The microstructure fields default to NaN; the fixture fills them with finite
        # values so two identically-built bars compare equal (NaN != NaN would break ==,
        # though hashability is unaffected). See test_bar_hashable.
        spread_mean=0.02,
        spread_last=0.02,
        spread_std=0.005,
        imbalance_mean=0.1,
        imbalance_last=0.1,
        imbalance_std=0.05,
        microprice_dev_mean=0.001,
        microprice_dev_last=0.001,
        microprice_dev_std=0.0005,
        signed_volume_sum=2.0,
        signed_dollar_sum=200.4,
        signed_tick_imbalance=0.3,
    )
    fields.update(overrides)
    return Bar(**fields)


def test_barkind_values() -> None:
    assert int(BarKind.TICK) == 1
    assert int(BarKind.VOLUME) == 2
    assert int(BarKind.DOLLAR) == 3


def test_bar_construction() -> None:
    b = _make_bar()
    assert b.kind == BarKind.TICK
    assert b.open == 100.0
    assert b.high == 101.0
    assert b.low == 99.0
    assert b.close == 100.5
    assert b.tick_count == 3


def test_bar_is_base_event() -> None:
    """AC9 — Bar inherits BaseEvent so it flows through on_event(BaseEvent)."""
    b = _make_bar()
    assert isinstance(b, BaseEvent)
    assert b.ts_event == 10
    assert b.instrument_id == 1
    assert b.sequence == 1


def test_bar_frozen() -> None:
    b = _make_bar()
    with pytest.raises(FrozenInstanceError):
        b.close = 999.0  # type: ignore[misc]


def test_bar_slots() -> None:
    b = _make_bar()
    assert hasattr(Bar, "__slots__")
    assert not hasattr(b, "__dict__")


def test_bar_not_a_trade_event() -> None:
    b = _make_bar()
    assert not isinstance(b, TradeEvent)


def test_bar_hashable() -> None:
    """Frozen dataclasses with only scalars + IntEnum are hashable."""
    b1 = _make_bar()
    b2 = _make_bar()
    assert hash(b1) == hash(b2)
    assert b1 == b2
