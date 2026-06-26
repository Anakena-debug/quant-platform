"""Regression tests for TBBO-aware streaming bar builders.

Covers:
- Duck-typed event acceptance (_is_trade_event TypeGuard)
- Aggressor-side priority in _direction (exchange > tick-rule)
- Microstructure field population from BBO
- NaN-BBO graceful degradation
- Dollar threshold arithmetic unchanged for both event shapes
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from quantcore.bars import (
    DollarBarBuilder,
    DollarImbalanceBarBuilder,
    ImbalanceConfig,
)
from quantcore.bars.streaming import (
    _TickRule,
    _is_trade_event,
)
from quantcore.data import Bar, BarKind, Side, TradeEvent


# ---------------------------------------------------------------------------
# Fixture: Databento-shaped event (no quantcore inheritance)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _DbnLikeEvent:
    """Structurally satisfies _TradeEventLike without inheriting BaseEvent."""

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    bid_sz: float = float("nan")
    ask_sz: float = float("nan")


def _dbn(
    i: int,
    price: float,
    size: float = 1.0,
    side: int = 1,
    bid: float = float("nan"),
    ask: float = float("nan"),
    bid_sz: float = float("nan"),
    ask_sz: float = float("nan"),
) -> _DbnLikeEvent:
    return _DbnLikeEvent(
        ts_event=i,
        instrument_id=1,
        sequence=i,
        price=price,
        size=size,
        aggressor_side=side,
        bid_px=bid,
        ask_px=ask,
        bid_sz=bid_sz,
        ask_sz=ask_sz,
    )


def _qc(
    i: int,
    price: float,
    size: float = 1.0,
    side: Side = Side.BID,
    bid: float = float("nan"),
    ask: float = float("nan"),
    bid_sz: float = float("nan"),
    ask_sz: float = float("nan"),
) -> TradeEvent:
    return TradeEvent(
        ts_event=i,
        instrument_id=1,
        sequence=i,
        price=price,
        size=size,
        aggressor_side=side,
        bid_px=bid,
        ask_px=ask,
        bid_sz=bid_sz,
        ask_sz=ask_sz,
    )


# ===================================================================
# _is_trade_event
# ===================================================================
class TestIsTradeEvent:
    def test_accepts_quantcore_trade_event(self) -> None:
        assert _is_trade_event(_qc(0, 100.0)) is True

    def test_accepts_dbn_like_event(self) -> None:
        assert _is_trade_event(_dbn(0, 100.0)) is True

    def test_rejects_missing_sequence(self) -> None:
        @dataclass(frozen=True)
        class _NoSeq:
            ts_event: int = 0
            instrument_id: int = 1
            price: float = 100.0
            size: float = 1.0

        assert _is_trade_event(_NoSeq()) is False

    def test_rejects_missing_ts_event(self) -> None:
        @dataclass(frozen=True)
        class _NoTs:
            instrument_id: int = 1
            sequence: int = 0
            price: float = 100.0
            size: float = 1.0

        assert _is_trade_event(_NoTs()) is False

    def test_rejects_bar(self) -> None:
        bar = Bar(
            ts_event=0,
            instrument_id=1,
            sequence=0,
            ts_open=0,
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
        assert _is_trade_event(bar) is False


# ===================================================================
# _TickRule._direction — aggressor-side priority
# ===================================================================
class TestDirectionPriority:
    def test_aggressor_buy_overrides_tick_downtick(self) -> None:
        """aggressor_side=+1, price decreases → direction must be +1."""
        tr = _TickRule()
        tr._tick_direction(100.0)
        d = tr._direction(99.0, aggressor_side=1)
        assert d == 1

    def test_aggressor_sell_overrides_tick_uptick(self) -> None:
        """aggressor_side=-1, price increases → direction must be -1."""
        tr = _TickRule()
        tr._tick_direction(100.0)
        d = tr._direction(101.0, aggressor_side=-1)
        assert d == -1

    def test_zero_side_falls_back_to_tick_rule(self) -> None:
        """aggressor_side=0 → tick-rule direction."""
        tr = _TickRule()
        tr._tick_direction(100.0)
        d = tr._direction(99.0, aggressor_side=0)
        assert d == -1

    def test_tick_state_updated_even_when_aggressor_used(self) -> None:
        """Tick-rule internal state must advance regardless of override.

        Sequence: 100 → 99 (side=+1, override to +1, but tick state
        records downtick) → 99 (side=0, zero-diff → carry forward the
        downtick recorded at previous step).
        """
        tr = _TickRule()
        tr._tick_direction(100.0)
        tr._direction(99.0, aggressor_side=1)
        assert tr._last_direction == -1
        d = tr._direction(99.0, aggressor_side=0)
        assert d == -1

    def test_default_no_aggressor_is_tick_rule(self) -> None:
        """Omitting aggressor_side defaults to tick rule."""
        tr = _TickRule()
        d = tr._direction(100.0)
        assert d == 1


# ===================================================================
# Dollar bar threshold arithmetic — both event shapes
# ===================================================================
class TestDollarThresholdBothShapes:
    """Dollar bar must emit at the same tick regardless of event origin."""

    THRESHOLD = 300.0

    def _feed_sequence(self, builder: DollarBarBuilder) -> list[Bar | None]:
        """Feed 4 events at price=100 size=1 (cum: 100, 200, 300, 400).

        The bar should close on the third event (cum 300 >= 300).
        """
        results: list[Bar | None] = []
        for i in range(4):
            bar = builder.on_event(_dbn(i, price=100.0, size=1.0))
            results.append(bar)
        return results

    def test_dbn_event_dollar_bar_triggers_at_threshold(self) -> None:
        b = DollarBarBuilder(threshold=self.THRESHOLD)
        results = self._feed_sequence(b)
        assert results[0] is None
        assert results[1] is None
        bar = results[2]
        assert bar is not None
        assert bar.dollar_volume == pytest.approx(300.0)
        assert bar.tick_count == 3

    def test_quantcore_event_same_trigger(self) -> None:
        b = DollarBarBuilder(threshold=self.THRESHOLD)
        results = []
        for i in range(4):
            results.append(b.on_event(_qc(i, 100.0, size=1.0)))
        assert results[0] is None
        assert results[1] is None
        bar = results[2]
        assert bar is not None
        assert bar.dollar_volume == pytest.approx(300.0)
        assert bar.tick_count == 3

    def test_weight_is_price_times_size(self) -> None:
        """price=50 size=3 → weight=150 per tick. Two ticks >= 300."""
        b = DollarBarBuilder(threshold=self.THRESHOLD)
        assert b.on_event(_dbn(0, price=50.0, size=3.0)) is None
        bar = b.on_event(_dbn(1, price=50.0, size=3.0))
        assert bar is not None
        assert bar.dollar_volume == pytest.approx(300.0)


# ===================================================================
# TBBO microstructure in emitted bars
# ===================================================================
class TestTBBOMicrostructure:
    """Verify finite BBO → finite microstructure fields in the bar."""

    BID = 99.95
    ASK = 100.05
    BID_SZ = 200.0
    ASK_SZ = 100.0

    def _tbbo_events(self, n: int, side: int = 1) -> list[_DbnLikeEvent]:
        return [
            _dbn(
                i,
                price=100.0,
                size=1.0,
                side=side,
                bid=self.BID,
                ask=self.ASK,
                bid_sz=self.BID_SZ,
                ask_sz=self.ASK_SZ,
            )
            for i in range(n)
        ]

    def test_dollar_bar_spread_mean_finite(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3):
            bar = b.on_event(ev)
        assert bar is not None
        assert math.isfinite(bar.spread_mean)
        assert bar.spread_mean == pytest.approx(self.ASK - self.BID)

    def test_dollar_bar_imbalance_mean_finite(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3):
            bar = b.on_event(ev)
        assert bar is not None
        expected = (self.BID_SZ - self.ASK_SZ) / (self.BID_SZ + self.ASK_SZ)
        assert math.isfinite(bar.imbalance_mean)
        assert bar.imbalance_mean == pytest.approx(expected)

    def test_dollar_bar_microprice_dev_mean_finite(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3):
            bar = b.on_event(ev)
        assert bar is not None
        mid = (self.BID + self.ASK) / 2.0
        total = self.BID_SZ + self.ASK_SZ
        microprice = (self.ASK * self.BID_SZ + self.BID * self.ASK_SZ) / total
        expected = (microprice - mid) / mid
        assert math.isfinite(bar.microprice_dev_mean)
        assert bar.microprice_dev_mean == pytest.approx(expected)

    def test_signed_volume_sum_nonzero(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3, side=1):
            bar = b.on_event(ev)
        assert bar is not None
        assert bar.signed_volume_sum == pytest.approx(3.0)

    def test_signed_dollar_sum_nonzero(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3, side=-1):
            bar = b.on_event(ev)
        assert bar is not None
        assert bar.signed_dollar_sum == pytest.approx(-300.0)

    def test_signed_tick_imbalance_buy_only(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for ev in self._tbbo_events(3, side=1):
            bar = b.on_event(ev)
        assert bar is not None
        assert bar.signed_tick_imbalance == pytest.approx(1.0)


# ===================================================================
# NaN-BBO graceful degradation (trades-only / synthetic)
# ===================================================================
class TestNanBBODegradation:
    def test_no_bbo_emits_valid_ohlcv(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        for i in range(3):
            bar = b.on_event(_dbn(i, price=100.0, size=1.0, side=0))
        assert bar is not None
        assert bar.open == 100.0
        assert bar.close == 100.0
        assert bar.volume == pytest.approx(3.0)
        assert bar.dollar_volume == pytest.approx(300.0)

    def test_no_bbo_microstructure_is_nan(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        bar = None
        for i in range(3):
            bar = b.on_event(_dbn(i, price=100.0, size=1.0))
        assert bar is not None
        assert math.isnan(bar.spread_mean)
        assert math.isnan(bar.imbalance_mean)
        assert math.isnan(bar.microprice_dev_mean)

    def test_no_bbo_no_exception(self) -> None:
        b = DollarBarBuilder(threshold=300.0)
        for i in range(3):
            b.on_event(_dbn(i, price=100.0, size=1.0))


# ===================================================================
# Imbalance bars — duck-typed acceptance + TBBO micro + direction
# ===================================================================
ICFG = ImbalanceConfig(
    exp_imbalance_init=0.5,
    exp_num_ticks_init=3.0,
    exp_num_ticks_min=1.0,
    exp_num_ticks_max=1000.0,
)


class TestImbalanceBarsTBBO:
    """Imbalance bars must accept Databento-like events and populate
    microstructure from BBO."""

    BID = 99.95
    ASK = 100.05
    BID_SZ = 200.0
    ASK_SZ = 100.0

    def _tbbo_buy(self, i: int, price: float = 100.0) -> _DbnLikeEvent:
        return _dbn(
            i,
            price=price,
            size=1.0,
            side=1,
            bid=self.BID,
            ask=self.ASK,
            bid_sz=self.BID_SZ,
            ask_sz=self.ASK_SZ,
        )

    def test_dbn_event_accepted(self) -> None:
        b = DollarImbalanceBarBuilder(config=ICFG)
        result = b.on_event(self._tbbo_buy(0))
        assert result is None or isinstance(result, Bar)

    def test_emits_bar_with_finite_micro(self) -> None:
        b = DollarImbalanceBarBuilder(config=ICFG)
        bar = None
        for i in range(200):
            result = b.on_event(self._tbbo_buy(i, price=100.0 + i * 0.01))
            if result is not None:
                bar = result
                break
        if bar is None:
            bar = b.flush()
        assert bar is not None
        assert math.isfinite(bar.spread_mean)
        assert bar.spread_mean == pytest.approx(self.ASK - self.BID)
        assert math.isfinite(bar.imbalance_mean)
        assert math.isfinite(bar.microprice_dev_mean)

    def test_aggressor_side_used_for_imbalance_direction(self) -> None:
        """Feed events with aggressor_side=+1 but declining prices.

        Without the aggressor-side fix, the tick rule would classify
        these as sells (price decreasing). With the fix, direction is
        +1 (buy) because aggressor_side=+1 takes priority. The theta
        accumulator should reflect positive (buy-side) imbalance.
        """
        cfg = ImbalanceConfig(
            exp_imbalance_init=0.01,
            exp_num_ticks_init=5.0,
            exp_num_ticks_min=1.0,
            exp_num_ticks_max=1000.0,
        )
        b = DollarImbalanceBarBuilder(config=cfg)

        bar = None
        for i in range(200):
            ev = _dbn(i, price=100.0 - i * 0.01, size=1.0, side=1)
            result = b.on_event(ev)
            if result is not None:
                bar = result
                break
        if bar is None:
            bar = b.flush()
        assert bar is not None
        assert bar.signed_volume_sum > 0.0

    def test_nan_bbo_still_emits(self) -> None:
        b = DollarImbalanceBarBuilder(config=ICFG)
        bar = None
        for i in range(200):
            result = b.on_event(_dbn(i, price=100.0 + i * 0.01, size=1.0, side=0))
            if result is not None:
                bar = result
                break
        if bar is None:
            bar = b.flush()
        assert bar is not None
        assert math.isnan(bar.spread_mean)
        assert bar.open > 0.0
