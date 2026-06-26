"""S36b PR4: watchdog detection tests.

Coverage:

- ``TestFeedSilence``: silence > threshold triggers an alert; updated
  ``_last_event_received_at_ns`` resets the deadline.
- ``TestHealthProbe``: probe returning False triggers an alert; probe
  raising is also an alert (probe is untrusted code).
- ``TestSteadyState``: events flowing + healthy probe → zero alerts.
- ``TestCombined``: feed silence AND probe-fail in the same tick →
  two increments (load-bearing for ops-dashboard accuracy).

The watchdog runs as an `asyncio.create_task`; tests await briefly,
cancel, then inspect `engine.metrics.watchdog_alerts_total`. To keep
tests fast we override `watchdog_interval_s` and `watchdog_feed_silence_s`
to 10–50 ms. The defaults (1.0s interval, 30s silence) are production
values; the watchdog logic is interval-independent.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming import (
    EngineConfig,
    EventClock,
    StreamingEngine,
)
from quantengine.runtime.streaming.protocols import (
    BarLike,
    CUSUMEvent,
    StreamContext,
    SyncBrokerFacade,
    TradeEventLike,
)

# ---------------------------------------------------------------------------
# Minimal stubs — enough to construct StreamingEngine without external deps
# ---------------------------------------------------------------------------


class _EmptyFeed:
    """Yields nothing immediately. Lets us isolate the watchdog from feed
    activity."""

    def __aiter__(self) -> _EmptyFeed:
        return self

    async def __anext__(self) -> TradeEventLike:
        await asyncio.sleep(60.0)  # block; the engine will be cancelled first
        raise StopAsyncIteration


class _NoopBarBuilder:
    def on_event(self, event: TradeEventLike) -> BarLike | None:
        del event
        return None

    def flush(self) -> BarLike | None:
        return None


class _NoCUSUM:
    def on_event(self, event: BarLike) -> int | None:
        del event
        return None

    def reset(self) -> None:
        pass


class _NoVol:
    def on_event(self, event: BarLike) -> float | None:
        del event
        return None

    def reset(self) -> None:
        pass


class _NoopStrategy:
    def on_bar(self, ts: int, bar: BarLike, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        del ts, bar, ctx, broker

    def on_cusum(
        self,
        ts: int,
        event: CUSUMEvent,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        del ts, event, ctx, broker

    def on_vol(
        self,
        ts: int,
        sigma: float,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        del ts, sigma, ctx, broker


class _NoopBroker:
    def submit_order(self, order: Order, timeout: float | None = None) -> list[Fill]:
        del order, timeout
        return []

    def cancel_order(self, order_id: Any, timeout: float | None = None) -> bool:
        del order_id, timeout
        return False

    def get_position(self, ticker: str, timeout: float | None = None) -> Position | None:
        del ticker, timeout
        return None

    def get_account_state(self, timeout: float | None = None) -> PortfolioState:
        del timeout
        return PortfolioState(cash=0.0, positions={})


def _make_engine(
    *,
    watchdog_interval_s: float = 0.01,
    watchdog_feed_silence_s: float = 0.05,
    health_probe: Any = None,
) -> StreamingEngine:
    """Construct an engine wired with all-noop stubs for watchdog isolation."""
    config = EngineConfig(
        instrument_id=1,
        ticker="WATCH",
        watchdog_interval_s=watchdog_interval_s,
        watchdog_feed_silence_s=watchdog_feed_silence_s,
        health_probe=health_probe,
    )
    return StreamingEngine(
        feed=_EmptyFeed(),
        bar_builder=_NoopBarBuilder(),
        cusum=_NoCUSUM(),
        vol=_NoVol(),
        strategy=_NoopStrategy(),
        broker=_NoopBroker(),
        clock=EventClock(),
        config=config,
    )


async def _drive_watchdog(engine: StreamingEngine, duration_s: float) -> None:
    """Run only the watchdog (not the full engine) for a bounded duration."""
    watchdog = asyncio.create_task(engine._watchdog())
    try:
        await asyncio.sleep(duration_s)
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Feed-silence detection
# ---------------------------------------------------------------------------


class TestFeedSilence:
    def test_silence_beyond_threshold_triggers_alert(self) -> None:
        async def run() -> None:
            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=0.05,
            )
            # Force "already silent" state: last receipt was 1s ago.
            engine._last_event_received_at_ns = time.monotonic_ns() - 1_000_000_000
            await _drive_watchdog(engine, duration_s=0.05)
            assert engine.metrics.watchdog_alerts_total >= 1

        asyncio.run(run())

    def test_recent_event_no_alert(self) -> None:
        async def run() -> None:
            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=0.5,  # generous
            )
            # Receipt time is "now"; silence threshold is 500ms; we
            # only run for 50ms — well under threshold.
            engine._last_event_received_at_ns = time.monotonic_ns()
            await _drive_watchdog(engine, duration_s=0.05)
            assert engine.metrics.watchdog_alerts_total == 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Broker health probe
# ---------------------------------------------------------------------------


class TestHealthProbe:
    def test_probe_returns_false_triggers_alert(self) -> None:
        async def run() -> None:
            async def unhealthy() -> bool:
                return False

            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=1.0,  # silence disabled (long threshold)
                health_probe=unhealthy,
            )
            engine._last_event_received_at_ns = time.monotonic_ns()  # no silence
            await _drive_watchdog(engine, duration_s=0.05)
            assert engine.metrics.watchdog_alerts_total >= 1

        asyncio.run(run())

    def test_probe_returns_true_no_alert(self) -> None:
        async def run() -> None:
            async def healthy() -> bool:
                return True

            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=1.0,
                health_probe=healthy,
            )
            engine._last_event_received_at_ns = time.monotonic_ns()
            await _drive_watchdog(engine, duration_s=0.05)
            assert engine.metrics.watchdog_alerts_total == 0

        asyncio.run(run())

    def test_probe_raising_treated_as_alert(self) -> None:
        async def run() -> None:
            async def raising() -> bool:
                raise RuntimeError("simulated broker socket error")

            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=1.0,
                health_probe=raising,
            )
            engine._last_event_received_at_ns = time.monotonic_ns()
            await _drive_watchdog(engine, duration_s=0.05)
            # Raising probe counted as alert (untrusted code; surface
            # the failure rather than silently swallow).
            assert engine.metrics.watchdog_alerts_total >= 1

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Steady state
# ---------------------------------------------------------------------------


class TestSteadyState:
    def test_active_feed_plus_healthy_probe_no_alerts(self) -> None:
        async def run() -> None:
            async def healthy() -> bool:
                return True

            engine = _make_engine(
                watchdog_interval_s=0.01,
                watchdog_feed_silence_s=0.1,
                health_probe=healthy,
            )

            # Simulate active feed: keep updating the receipt timestamp.
            async def fake_feed_pump() -> None:
                while True:
                    engine._last_event_received_at_ns = time.monotonic_ns()
                    await asyncio.sleep(0.005)

            pump = asyncio.create_task(fake_feed_pump())
            try:
                await _drive_watchdog(engine, duration_s=0.05)
            finally:
                pump.cancel()
                try:
                    await pump
                except asyncio.CancelledError:
                    pass
            assert engine.metrics.watchdog_alerts_total == 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Combined alerts (silence + probe-fail in same tick)
# ---------------------------------------------------------------------------


class TestCombined:
    def test_silence_and_probe_fail_count_independently(self) -> None:
        async def run() -> None:
            async def unhealthy() -> bool:
                return False

            engine = _make_engine(
                watchdog_interval_s=0.02,
                watchdog_feed_silence_s=0.01,
                health_probe=unhealthy,
            )
            # Force silence state
            engine._last_event_received_at_ns = time.monotonic_ns() - 1_000_000_000

            # Single tick → both arms fire → +2
            await _drive_watchdog(engine, duration_s=0.025)
            # Each tick contributes 2; we expect at least 2 (one tick).
            # Upper bound is loose because timing varies.
            assert engine.metrics.watchdog_alerts_total >= 2

        asyncio.run(run())
