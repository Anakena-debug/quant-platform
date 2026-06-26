"""Tests for quantengine.runtime.streaming.protocols.

Pins the protocol surfaces (StreamingStrategy, AsyncBrokerProtocol,
SyncBrokerFacade, DataFeedProtocol, Clock, BarBuilderProtocol,
OnlineCUSUMFilterProtocol, OnlineEWMAVolatilityProtocol) and their
reference implementations (EventClock, WallClock) plus support types
(CUSUMEvent, StreamContext, BrokerTimeoutError, TradeEventLike, BarLike).

This test file does not import from quantcore. The runtime engine
consumes structural protocols (``TradeEventLike``, ``BarLike``,
``BarBuilderProtocol``, ``OnlineCUSUMFilterProtocol``,
``OnlineEWMAVolatilityProtocol``) which quantcore's concrete
dataclasses satisfy; this preserves the quantcore (L3 research) /
quantengine (L1 deployment) boundary established in S33 §5.D5.

Local test fixtures (``_FakeTradeEvent``, ``_FakeBar``, ``_Stub*``)
construct minimal dataclasses that satisfy the same protocols, so
the protocol contract is exercised end-to-end without any cross-
package import.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming import (
    AsyncBrokerProtocol,
    BarBuilderProtocol,
    BarLike,
    BrokerTimeoutError,
    Clock,
    CUSUMEvent,
    DataFeedProtocol,
    EventClock,
    OnlineCUSUMFilterProtocol,
    OnlineEWMAVolatilityProtocol,
    StreamContext,
    StreamingStrategy,
    SyncBrokerFacade,
    TradeEventLike,
    WallClock,
)


# ---------------------------------------------------------------------------
# Local fixtures — minimal dataclasses that satisfy the structural Protocols.
# quantcore's ``TradeEvent`` (quantcore.data.events) and ``Bar``
# (quantcore.data.bars) satisfy the same Protocols at runtime via duck
# typing; quantengine never imports them.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _FakeTradeEvent:
    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int


@dataclass(frozen=True, slots=True)
class _FakeBar:
    ts_event: int
    instrument_id: int
    sequence: int
    ts_open: int
    kind: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    dollar_volume: float


def _make_trade(i: int) -> _FakeTradeEvent:
    return _FakeTradeEvent(
        ts_event=i * 1_000_000,
        instrument_id=1,
        sequence=i,
        price=100.0 + i,
        size=10.0,
        aggressor_side=1,
    )


def _make_bar(ts: int = 100, close: float = 100.5) -> _FakeBar:
    return _FakeBar(
        ts_event=ts,
        instrument_id=1,
        sequence=0,
        ts_open=ts - 100,
        kind=1,
        open=100.0,
        high=101.0,
        low=99.0,
        close=close,
        volume=1000.0,
        vwap=100.25,
        tick_count=42,
        dollar_volume=100_250.0,
    )


# ---------------------------------------------------------------------------
# Structural shapes
# ---------------------------------------------------------------------------
def test_fake_trade_event_satisfies_trade_event_like() -> None:
    assert isinstance(_make_trade(0), TradeEventLike)


def test_fake_bar_satisfies_bar_like() -> None:
    assert isinstance(_make_bar(), BarLike)


def test_objects_missing_attributes_fail_structural_check() -> None:
    """Sanity: a dataclass with the wrong shape must NOT satisfy the
    structural protocol."""

    @dataclass(frozen=True, slots=True)
    class _MissingPrice:
        ts_event: int
        instrument_id: int
        sequence: int
        size: float
        aggressor_side: int

    assert not isinstance(_MissingPrice(0, 1, 0, 10.0, 1), TradeEventLike)


# ---------------------------------------------------------------------------
# Clock implementations
# ---------------------------------------------------------------------------
def test_event_clock_starts_at_zero_then_advances() -> None:
    clock = EventClock()
    assert clock.now_ns() == 0
    clock.tick(1_000_000_000)
    assert clock.now_ns() == 1_000_000_000
    clock.tick(2_500_000_000)
    assert clock.now_ns() == 2_500_000_000


def test_event_clock_does_not_enforce_monotonicity() -> None:
    """Engine owns ordering; clock just records."""
    clock = EventClock()
    clock.tick(100)
    clock.tick(50)
    assert clock.now_ns() == 50


def test_wall_clock_returns_monotonic_ns() -> None:
    wc = WallClock()
    t0 = wc.now_ns()
    for _ in range(1000):
        t1 = wc.now_ns()
        if t1 > t0:
            break
    assert wc.now_ns() >= t0


def test_event_clock_and_wall_clock_satisfy_clock_protocol() -> None:
    assert isinstance(EventClock(), Clock)
    assert isinstance(WallClock(), Clock)


def test_a_plain_object_does_not_satisfy_clock() -> None:
    class NotAClock:
        pass

    assert not isinstance(NotAClock(), Clock)


# ---------------------------------------------------------------------------
# Support types
# ---------------------------------------------------------------------------
def test_cusum_event_is_frozen_slotted_dataclass() -> None:
    e = CUSUMEvent(ts_event=123, instrument_id=42, bar_close=100.5)
    assert e.ts_event == 123
    assert e.instrument_id == 42
    assert e.bar_close == 100.5
    with pytest.raises((AttributeError, Exception)):
        e.ts_event = 999  # type: ignore[misc]


def test_stream_context_is_frozen() -> None:
    ctx = StreamContext(sequence=10, clock_ns=1_000_000, instrument_id=7)
    assert ctx.sequence == 10
    assert ctx.clock_ns == 1_000_000
    assert ctx.instrument_id == 7
    with pytest.raises((AttributeError, Exception)):
        ctx.sequence = 11  # type: ignore[misc]


def test_broker_timeout_error_subclasses_timeout_error() -> None:
    exc = BrokerTimeoutError("timed out")
    assert isinstance(exc, TimeoutError)
    with pytest.raises(TimeoutError):
        raise BrokerTimeoutError("re-raise via base class")


# ---------------------------------------------------------------------------
# AsyncBrokerProtocol
# ---------------------------------------------------------------------------
class _MinimalAsyncBroker:
    """Smallest possible implementation satisfying AsyncBrokerProtocol."""

    async def submit_order(self, order: Order) -> list[Fill]:
        return []

    async def cancel_order(self, order_id: UUID) -> bool:
        return True

    async def get_position(self, ticker: str) -> Position | None:
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=0.0, positions={})


def test_minimal_async_broker_satisfies_protocol() -> None:
    assert isinstance(_MinimalAsyncBroker(), AsyncBrokerProtocol)


def test_async_broker_methods_are_awaitable() -> None:
    broker = _MinimalAsyncBroker()
    order = Order(
        order_id=uuid4(),
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )

    async def _drive() -> None:
        fills = await broker.submit_order(order)
        assert fills == []
        cancelled = await broker.cancel_order(order.order_id)
        assert cancelled is True
        pos = await broker.get_position("AAPL")
        assert pos is None
        state = await broker.get_account_state()
        assert isinstance(state, PortfolioState)

    asyncio.run(_drive())


def test_non_broker_object_does_not_satisfy_async_broker_protocol() -> None:
    class HalfImplemented:
        async def submit_order(self, order: Order) -> list[Fill]:
            return []

        # Missing cancel_order / get_position / get_account_state.

    assert not isinstance(HalfImplemented(), AsyncBrokerProtocol)


# ---------------------------------------------------------------------------
# SyncBrokerFacade
# ---------------------------------------------------------------------------
class _MinimalSyncBroker:
    def submit_order(self, order: Order, timeout: float | None = None) -> list[Fill]:
        return []

    def cancel_order(self, order_id: UUID, timeout: float | None = None) -> bool:
        return True

    def get_position(self, ticker: str, timeout: float | None = None) -> Position | None:
        return None

    def get_account_state(self, timeout: float | None = None) -> PortfolioState:
        return PortfolioState(cash=0.0, positions={})


def test_minimal_sync_broker_satisfies_facade() -> None:
    assert isinstance(_MinimalSyncBroker(), SyncBrokerFacade)


# ---------------------------------------------------------------------------
# DataFeedProtocol
# ---------------------------------------------------------------------------
class _MinimalFeed:
    def __init__(self, events: list[_FakeTradeEvent]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self) -> "_MinimalFeed":
        return self

    async def __anext__(self) -> _FakeTradeEvent:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        out = self._events[self._index]
        self._index += 1
        return out


def test_minimal_feed_yields_expected_sequence() -> None:
    events = [_make_trade(i) for i in range(3)]
    feed = _MinimalFeed(events)

    async def _drain() -> list[_FakeTradeEvent]:
        return [e async for e in feed]

    out = asyncio.run(_drain())
    assert out == events


def test_feed_satisfies_data_feed_protocol() -> None:
    """DataFeedProtocol is structural — pin the minimal feed type-
    checks against an explicit annotation."""
    feed: DataFeedProtocol = _MinimalFeed([_make_trade(0)])

    async def _exhaust() -> None:
        async for _ in feed:
            pass

    asyncio.run(_exhaust())


# ---------------------------------------------------------------------------
# Pipeline primitive protocols (BarBuilder / CUSUM / EWMA-vol)
# ---------------------------------------------------------------------------
class _StubBarBuilder:
    """Emits one bar after N events, then None."""

    def __init__(self, emit_every: int = 3) -> None:
        self._n = 0
        self._emit_every = emit_every

    def on_event(self, event: TradeEventLike) -> BarLike | None:
        self._n += 1
        if self._n % self._emit_every == 0:
            return _make_bar(ts=event.ts_event, close=event.price)
        return None

    def flush(self) -> BarLike | None:
        return None


class _StubCUSUM:
    """Fires every other bar."""

    def __init__(self) -> None:
        self._n = 0

    def on_event(self, event: BarLike) -> int | None:
        self._n += 1
        if self._n % 2 == 0:
            return event.ts_event
        return None

    def reset(self) -> None:
        self._n = 0


class _StubEWMAVol:
    """Returns a constant after first bar."""

    def __init__(self) -> None:
        self._seen_one = False

    def on_event(self, event: BarLike) -> float | None:
        if not self._seen_one:
            self._seen_one = True
            return None
        return 0.0125

    def reset(self) -> None:
        self._seen_one = False


def test_bar_builder_protocol_satisfied_structurally() -> None:
    assert isinstance(_StubBarBuilder(), BarBuilderProtocol)


def test_cusum_filter_protocol_satisfied_structurally() -> None:
    assert isinstance(_StubCUSUM(), OnlineCUSUMFilterProtocol)


def test_ewma_vol_protocol_satisfied_structurally() -> None:
    assert isinstance(_StubEWMAVol(), OnlineEWMAVolatilityProtocol)


def test_stub_bar_builder_emits_bar_then_none() -> None:
    b = _StubBarBuilder(emit_every=3)
    assert b.on_event(_make_trade(0)) is None
    assert b.on_event(_make_trade(1)) is None
    out = b.on_event(_make_trade(2))
    assert out is not None
    assert isinstance(out, BarLike)


# ---------------------------------------------------------------------------
# StreamingStrategy
# ---------------------------------------------------------------------------
class _RecordingStrategy:
    """Records every callback invocation for assertion."""

    def __init__(self) -> None:
        self.bar_calls: list[tuple[int, BarLike, StreamContext]] = []
        self.cusum_calls: list[tuple[int, CUSUMEvent, StreamContext]] = []
        self.vol_calls: list[tuple[int, float, StreamContext]] = []

    def on_bar(
        self,
        ts: int,
        bar: BarLike,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        self.bar_calls.append((ts, bar, ctx))

    def on_cusum(
        self,
        ts: int,
        event: CUSUMEvent,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        self.cusum_calls.append((ts, event, ctx))

    def on_vol(
        self,
        ts: int,
        sigma: float,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        self.vol_calls.append((ts, sigma, ctx))


def test_recording_strategy_satisfies_protocol() -> None:
    assert isinstance(_RecordingStrategy(), StreamingStrategy)


def test_strategy_callbacks_have_expected_signature_and_record() -> None:
    strat = _RecordingStrategy()
    broker = _MinimalSyncBroker()
    ctx = StreamContext(sequence=0, clock_ns=0, instrument_id=1)
    bar = _make_bar(ts=100, close=100.5)

    strat.on_bar(100, bar, ctx, broker)
    strat.on_cusum(100, CUSUMEvent(100, 1, 100.5), ctx, broker)
    strat.on_vol(100, 0.012, ctx, broker)

    assert len(strat.bar_calls) == 1
    assert len(strat.cusum_calls) == 1
    assert len(strat.vol_calls) == 1
    assert strat.bar_calls[0][1] is bar
    assert strat.cusum_calls[0][1].bar_close == 100.5
    assert strat.vol_calls[0][1] == 0.012


def test_half_strategy_does_not_satisfy_protocol() -> None:
    class HalfStrat:
        def on_bar(
            self,
            ts: int,
            bar: BarLike,
            ctx: StreamContext,
            broker: SyncBrokerFacade,
        ) -> None:
            pass

        # Missing on_cusum / on_vol.

    assert not isinstance(HalfStrat(), StreamingStrategy)
