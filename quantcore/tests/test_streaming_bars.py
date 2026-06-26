"""Unit tests for streaming BarBuilder subclasses (S34 §3.AC3, AC4)."""

from __future__ import annotations

import pytest

from quantcore.bars import (
    BarBuilder,
    DollarBarBuilder,
    DollarImbalanceBarBuilder,
    DollarRunsBarBuilder,
    ImbalanceConfig,
    RunsConfig,
    TickBarBuilder,
    TickImbalanceBarBuilder,
    TickRunsBarBuilder,
    VolumeBarBuilder,
    VolumeImbalanceBarBuilder,
    VolumeRunsBarBuilder,
)
from quantcore.data import Bar, BarKind, Side, TradeEvent


def _t(i: int, price: float, size: float = 1.0, instrument_id: int = 1) -> TradeEvent:
    return TradeEvent(
        ts_event=i,
        instrument_id=instrument_id,
        sequence=i,
        price=price,
        size=size,
        aggressor_side=Side.BID,
    )


ICFG = ImbalanceConfig(exp_imbalance_init=0.0)
RCFG = RunsConfig(exp_prob_buy_init=0.5, exp_w_buy_init=1.0, exp_w_sell_init=1.0)


def test_all_nine_are_bar_builders() -> None:
    plain = [
        TickBarBuilder(threshold=10),
        VolumeBarBuilder(threshold=10.0),
        DollarBarBuilder(threshold=10.0),
    ]
    imb = [
        TickImbalanceBarBuilder(config=ICFG),
        VolumeImbalanceBarBuilder(config=ICFG),
        DollarImbalanceBarBuilder(config=ICFG),
    ]
    runs = [
        TickRunsBarBuilder(config=RCFG),
        VolumeRunsBarBuilder(config=RCFG),
        DollarRunsBarBuilder(config=RCFG),
    ]
    for b in plain + imb + runs:
        assert isinstance(b, BarBuilder)


def test_tick_bar_emits_on_threshold() -> None:
    b = TickBarBuilder(threshold=3)
    assert b.on_event(_t(0, 100.0)) is None
    assert b.on_event(_t(1, 101.0)) is None
    bar = b.on_event(_t(2, 102.0))
    assert bar is not None
    assert bar.tick_count == 3
    assert bar.open == 100.0
    assert bar.close == 102.0
    assert bar.high == 102.0
    assert bar.low == 100.0
    assert bar.kind == BarKind.TICK
    assert bar.ts_open == 0
    assert bar.ts_event == 2
    assert bar.sequence == 2


def test_volume_bar_emits_on_cumulative_volume() -> None:
    b = VolumeBarBuilder(threshold=5.0)
    assert b.on_event(_t(0, 100.0, size=2.0)) is None
    assert b.on_event(_t(1, 100.0, size=2.0)) is None
    bar = b.on_event(_t(2, 100.0, size=2.0))
    assert bar is not None
    assert bar.volume == 6.0


def test_dollar_bar_emits_on_cumulative_notional() -> None:
    b = DollarBarBuilder(threshold=300.0)
    bar = None
    for i in range(5):
        bar = b.on_event(_t(i, 100.0, size=1.0))
        if bar is not None:
            break
    assert bar is not None
    assert bar.dollar_volume >= 300.0


def test_non_trade_events_are_no_op() -> None:
    b = TickBarBuilder(threshold=2)
    fake_bar = Bar(
        ts_event=99,
        instrument_id=1,
        sequence=99,
        ts_open=99,
        kind=BarKind.TICK,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=0.0,
        vwap=1.0,
        tick_count=1,
        dollar_volume=0.0,
    )
    assert b.on_event(fake_bar) is None
    assert b.on_event(_t(0, 100.0)) is None  # only one trade so far


def test_instrument_id_mismatch_raises() -> None:
    b = TickBarBuilder(threshold=2)
    b.on_event(_t(0, 100.0, instrument_id=1))
    with pytest.raises(ValueError, match="instrument_id"):
        b.on_event(_t(1, 100.0, instrument_id=2))


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threshold must be > 0"):
        TickBarBuilder(threshold=0)


def test_flush_returns_partial_bar() -> None:
    b = TickBarBuilder(threshold=10)
    b.on_event(_t(0, 100.0))
    b.on_event(_t(1, 101.0))
    bar = b.flush()
    assert bar is not None
    assert bar.tick_count == 2
    assert b.flush() is None  # idempotent


def test_flush_on_empty_builder_returns_none() -> None:
    b = TickBarBuilder(threshold=10)
    assert b.flush() is None


def test_imbalance_requires_explicit_init() -> None:
    with pytest.raises(ValueError, match="exp_imbalance_init"):
        TickImbalanceBarBuilder(config=ImbalanceConfig())


def test_runs_requires_explicit_init_prob() -> None:
    with pytest.raises(ValueError, match="exp_prob_buy_init"):
        TickRunsBarBuilder(config=RunsConfig(exp_w_buy_init=1.0, exp_w_sell_init=1.0))


def test_runs_requires_explicit_init_w_buy() -> None:
    with pytest.raises(ValueError, match="exp_w_buy_init"):
        TickRunsBarBuilder(config=RunsConfig(exp_prob_buy_init=0.5, exp_w_sell_init=1.0))


def test_runs_requires_explicit_init_w_sell() -> None:
    with pytest.raises(ValueError, match="exp_w_sell_init"):
        TickRunsBarBuilder(config=RunsConfig(exp_prob_buy_init=0.5, exp_w_buy_init=1.0))
