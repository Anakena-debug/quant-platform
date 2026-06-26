"""Tests for quantengine.runtime.streaming.engine.StreamingEngine.

Pins the engine contract (S35 D8, D11; AC4, AC10, AC12, AC13):

- ``asyncio.to_thread`` is the strategy-dispatch mechanism (AC4 grep is
  structural; pipeline tests pin the resulting behavior).
- Bounded queue drops newest under backpressure with counter
  increment (D8 / AC12: test_backpressure_drops_counter).
- Position atomicity: after ``broker.submit_order``, ``broker.get_position``
  reflects the fill (AC10: test_position_atomicity_after_submit).
- Shutdown DRAIN completes within timeout (D11 / AC13:
  test_shutdown_drain_within_timeout).

quantcore-independence: no quantcore imports; local stubs satisfy
the pipeline primitive protocols.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Iterator
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.portfolio.state import Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming import (
    BarLike,
    CUSUMEvent,
    DemoBroker,
    EngineConfig,
    EventClock,
    SafeBroker,
    ShutdownMode,
    StreamContext,
    StreamingEngine,
    SyncBrokerFacade,
    SyntheticTradeFeed,
    ThreadSafeBrokerWrapper,
    TradeEventLike,
    WrapperTimeouts,
)


# ---------------------------------------------------------------------------
# Local stub pipeline primitives (quantcore concrete classes also work)
# ---------------------------------------------------------------------------
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


class _EveryNBarBuilder:
    """Emits one bar every N events; useful for predictable test counts."""

    def __init__(self, every: int = 5) -> None:
        self._every = every
        self._n = 0
        self._open: float | None = None
        self._high: float = float("-inf")
        self._low: float = float("inf")
        self._vol: float = 0.0
        self._dol: float = 0.0
        self._tcount: int = 0
        self._start_ts: int = 0

    def on_event(self, event: TradeEventLike) -> BarLike | None:
        if self._open is None:
            self._open = event.price
            self._start_ts = event.ts_event
        self._high = max(self._high, event.price)
        self._low = min(self._low, event.price)
        self._vol += event.size
        self._dol += event.size * event.price
        self._tcount += 1
        self._n += 1
        if self._n % self._every == 0:
            bar = _FakeBar(
                ts_event=event.ts_event,
                instrument_id=event.instrument_id,
                sequence=self._n,
                ts_open=self._start_ts,
                kind=1,
                open=self._open,
                high=self._high,
                low=self._low,
                close=event.price,
                volume=self._vol,
                vwap=self._dol / max(self._vol, 1e-12),
                tick_count=self._tcount,
                dollar_volume=self._dol,
            )
            self._open = None
            self._high = float("-inf")
            self._low = float("inf")
            self._vol = 0.0
            self._dol = 0.0
            self._tcount = 0
            return bar
        return None

    def flush(self) -> BarLike | None:
        return None


class _AlwaysCUSUM:
    """Fires CUSUM on every bar — useful for exercising the cusum path."""

    def on_event(self, event: BarLike) -> int | None:
        return event.ts_event

    def reset(self) -> None:
        pass


class _AlwaysVol:
    """Returns a constant sigma after first bar."""

    def __init__(self) -> None:
        self._seen_one = False

    def on_event(self, event: BarLike) -> float | None:
        if not self._seen_one:
            self._seen_one = True
            return None
        return 0.01

    def reset(self) -> None:
        self._seen_one = False


# ---------------------------------------------------------------------------
# Recording strategy
# ---------------------------------------------------------------------------
class _RecordingStrategy:
    def __init__(self) -> None:
        self.bars: list[BarLike] = []
        self.cusums: list[CUSUMEvent] = []
        self.vols: list[float] = []

    def on_bar(self, ts: int, bar: BarLike, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        self.bars.append(bar)

    def on_cusum(
        self, ts: int, event: CUSUMEvent, ctx: StreamContext, broker: SyncBrokerFacade
    ) -> None:
        self.cusums.append(event)

    def on_vol(self, ts: int, sigma: float, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        self.vols.append(sigma)


class _SubmitOnBarStrategy:
    """Calls broker.submit_order on the first bar; records broker.get_position
    immediately after the call returns. Exercises AC10 atomicity."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.positions_after_submit: list[Position | None] = []
        self.fills_history: list[int] = []  # signed_quantity of each fill returned
        self.fired = False

    def on_bar(self, ts: int, bar: BarLike, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        if self.fired:
            return
        self.fired = True
        order = Order(
            order_id=uuid4(),
            ticker=self.ticker,
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )
        fills = broker.submit_order(order, timeout=2.0)
        self.fills_history.extend(f.signed_quantity for f in fills)
        # Immediately query position; must reflect the fill (AC10).
        pos = broker.get_position(self.ticker, timeout=2.0)
        self.positions_after_submit.append(pos)

    def on_cusum(self, ts, event, ctx, broker) -> None:
        pass

    def on_vol(self, ts, sigma, ctx, broker) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers for spinning up the event loop on a worker thread
# ---------------------------------------------------------------------------
@pytest.fixture
def loop_thread() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_runner, name="engine-loop", daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0), "loop thread failed to start"
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def _run_on_loop(loop: asyncio.AbstractEventLoop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=10.0)


# ---------------------------------------------------------------------------
# Basic pipeline: feed -> bars -> strategy receives them
# ---------------------------------------------------------------------------
def test_engine_runs_feed_to_completion_dispatches_bars(
    loop_thread: asyncio.AbstractEventLoop, tmp_path
) -> None:
    feed = SyntheticTradeFeed(seed=1, instrument_id=1, n_events=20)
    builder = _EveryNBarBuilder(every=5)
    cusum = _AlwaysCUSUM()
    vol = _AlwaysVol()
    strat = _RecordingStrategy()
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities(max_order_notional=1_000_000.0)
    journal = tmp_path / "journal.jsonl"
    sb = SafeBroker(
        demo,
        gate,
        journal,
        price_provider=lambda: {"AAPL": 100.0},
        state_provider=lambda: demo.state,
        clock=EventClock(),
    )
    wrapper = ThreadSafeBrokerWrapper(sb, loop_thread)
    config = EngineConfig(instrument_id=1, ticker="AAPL", queue_maxsize=1000)
    engine = StreamingEngine(feed, builder, cusum, vol, strat, wrapper, EventClock(), config)

    _run_on_loop(loop_thread, engine.run())

    assert engine.metrics.events_processed == 20
    assert engine.metrics.bars_emitted == 4  # 20 events / every=5
    assert len(strat.bars) == 4
    # CUSUM fires on every bar; first vol is None, then 3 sigma values.
    assert len(strat.cusums) == 4
    assert len(strat.vols) == 3
    assert engine.metrics.backpressure_drops == 0


# ---------------------------------------------------------------------------
# AC10 — position atomicity after broker.submit_order
# ---------------------------------------------------------------------------
def test_position_atomicity_after_submit(loop_thread: asyncio.AbstractEventLoop, tmp_path) -> None:
    """Strategy calls submit_order then get_position; position MUST
    reflect the fill before submit returns (AC10).
    """
    feed = SyntheticTradeFeed(seed=7, instrument_id=1, n_events=15)
    builder = _EveryNBarBuilder(every=5)
    cusum = _AlwaysCUSUM()
    vol = _AlwaysVol()
    strat = _SubmitOnBarStrategy(ticker="AAPL")
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities(max_order_notional=1_000_000.0)
    journal = tmp_path / "atomicity.jsonl"
    sb = SafeBroker(
        demo,
        gate,
        journal,
        price_provider=lambda: {"AAPL": 100.0},
        state_provider=lambda: demo.state,
        clock=EventClock(),
    )
    wrapper = ThreadSafeBrokerWrapper(sb, loop_thread, timeouts=WrapperTimeouts(submit_order_s=3.0))
    config = EngineConfig(instrument_id=1, ticker="AAPL", queue_maxsize=1000)
    engine = StreamingEngine(feed, builder, cusum, vol, strat, wrapper, EventClock(), config)

    _run_on_loop(loop_thread, engine.run())

    # Strategy fired once on first bar.
    assert strat.fired is True
    # submit_order returned a non-empty fill list...
    assert strat.fills_history == [10]
    # ...AND get_position immediately afterwards saw quantity == 10.
    assert len(strat.positions_after_submit) == 1
    pos = strat.positions_after_submit[0]
    assert pos is not None, "AC10 violation: position was None after fill"
    assert pos.quantity == 10, f"AC10 violation: quantity={pos.quantity}, expected 10"


# ---------------------------------------------------------------------------
# AC12 — backpressure_drops counter increments under queue full
# ---------------------------------------------------------------------------
def test_backpressure_drops_counter(loop_thread: asyncio.AbstractEventLoop, tmp_path) -> None:
    """Tiny queue + a feed that produces many events synchronously
    triggers backpressure; counter must increment (D8 / AC12)."""

    class _FloodFeed:
        """Synchronous feed: yields 500 events as fast as possible
        WITHOUT awaiting (no `await asyncio.sleep(0)`) so the
        consumer cannot drain between puts. This forces the queue
        to fill before the consumer runs."""

        def __init__(self) -> None:
            self._i = 0

        def __aiter__(self) -> "_FloodFeed":
            return self

        async def __anext__(self):
            if self._i >= 500:
                raise StopAsyncIteration
            # No `await asyncio.sleep(0)` — block-yield only when queue.put_nowait
            # raises QueueFull (which it doesn't — engine drops instead).
            ev = _SyntheticEvent(
                ts_event=self._i,
                instrument_id=1,
                sequence=self._i,
                price=100.0,
                size=1.0,
                aggressor_side=1,
            )
            self._i += 1
            return ev

    feed = _FloodFeed()
    builder = _EveryNBarBuilder(every=1000)  # never emits
    cusum = _AlwaysCUSUM()
    vol = _AlwaysVol()
    strat = _RecordingStrategy()
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities()
    sb = SafeBroker(
        demo,
        gate,
        tmp_path / "j.jsonl",
        price_provider=lambda: {"AAPL": 100.0},
        state_provider=lambda: demo.state,
        clock=EventClock(),
    )
    wrapper = ThreadSafeBrokerWrapper(sb, loop_thread)
    # queue_maxsize=2 + 500-event burst -> drops expected.
    config = EngineConfig(
        instrument_id=1,
        ticker="AAPL",
        queue_maxsize=2,
        consumer_poll_timeout_s=0.5,
    )
    engine = StreamingEngine(feed, builder, cusum, vol, strat, wrapper, EventClock(), config)

    _run_on_loop(loop_thread, engine.run())

    assert engine.metrics.backpressure_drops > 0, (
        f"AC12 violation: backpressure_drops=0 after flood; "
        f"events_processed={engine.metrics.events_processed}"
    )


# ---------------------------------------------------------------------------
# AC13 — shutdown DRAIN completes within timeout
# ---------------------------------------------------------------------------
def test_shutdown_drain_within_timeout(loop_thread: asyncio.AbstractEventLoop, tmp_path) -> None:
    """Spawn an engine on a long synthetic feed; call shutdown(DRAIN,
    timeout_s=2.0) midway; verify both shutdown and run() complete
    within the timeout window (D11)."""
    feed = SyntheticTradeFeed(seed=3, instrument_id=1, n_events=10_000)
    builder = _EveryNBarBuilder(every=100)
    cusum = _AlwaysCUSUM()
    vol = _AlwaysVol()
    strat = _RecordingStrategy()
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities()
    sb = SafeBroker(
        demo,
        gate,
        tmp_path / "j.jsonl",
        price_provider=lambda: {"AAPL": 100.0},
        state_provider=lambda: demo.state,
        clock=EventClock(),
    )
    wrapper = ThreadSafeBrokerWrapper(sb, loop_thread)
    config = EngineConfig(instrument_id=1, ticker="AAPL", queue_maxsize=100)
    engine = StreamingEngine(feed, builder, cusum, vol, strat, wrapper, EventClock(), config)

    # Start run() as a task on the loop.
    run_future = asyncio.run_coroutine_threadsafe(engine.run(), loop_thread)
    # Let it process a bit.
    import time as _time

    _time.sleep(0.2)
    # Trigger shutdown.
    shutdown_future = asyncio.run_coroutine_threadsafe(
        engine.shutdown(ShutdownMode.DRAIN, timeout_s=2.0), loop_thread
    )
    shutdown_future.result(timeout=3.0)
    # And wait for run() to wrap up.
    run_future.result(timeout=3.0)

    # Some events processed but not all.
    assert engine.metrics.events_processed > 0
    assert engine.metrics.events_processed < 10_000


# ---------------------------------------------------------------------------
# Local synthetic-event for the AC12 flood test (also TradeEventLike)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SyntheticEvent:
    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int
