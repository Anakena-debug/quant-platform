"""StreamingEngine — async event loop orchestrating the S35 pipeline.

Consumes ``TradeEventLike`` from a ``DataFeedProtocol`` through a
bounded ``asyncio.Queue`` (S35 D8 backpressure: drop newest, increment
counter, no silent loss). Routes each event through:

    feed -> queue -> consumer
                       |
                       v
                  BarBuilder.on_event(event) -> bar | None
                       |
                       v  (only on bar emission)
                  CUSUM.on_event(bar) -> ts | None
                  Vol.on_event(bar)   -> sigma | None
                       |
                       v
                  dispatch_executor -> strategy.on_bar(...)
                  dispatch_executor -> strategy.on_cusum(...)  if cusum_ts
                  dispatch_executor -> strategy.on_vol(...)    if sigma

The engine dispatches strategy callbacks via a dedicated single-worker
``ThreadPoolExecutor`` (``thread_name_prefix='engine-strategy'``,
``max_workers=1``) so the user-authored sync code runs off the loop
thread without blocking it. Per S36b D2 this replaces S35's default-
pool dispatch — the load-bearing invariant of running off the loop
thread (required by ``ThreadSafeBrokerWrapper.run_coroutine_threadsafe``)
is preserved; the savings come from one warm thread reused across
dispatches instead of default-pool thread acquisition per call.
Strategy code calls back into the broker via the ``SyncBrokerFacade``
(``ThreadSafeBrokerWrapper`` in production) which bridges back to the
same loop via ``asyncio.run_coroutine_threadsafe``.

Shutdown semantics (S35 D11):

- ``DRAIN`` (default): stop reading the feed, finish queue, cancel
  open orders if requested, exit. Times out to ``CANCEL`` after
  ``timeout_s``.
- ``CANCEL``: stop everything immediately, cancel open orders at
  broker.
- ``HALT``: stop everything, leave open orders untouched.

The engine does NOT own portfolio state. State lives in the broker
(``DemoBroker`` in S35 maintains its own ``VirtualPortfolio``;
``AsyncIBKRBroker`` in S36 queries TWS). SafeBroker queries the
broker via its ``state_provider`` callable for pre-trade risk checks.

quantcore-independence pattern: this module does not import quantcore.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from quantengine.runtime.streaming.protocols import (
    BarBuilderProtocol,
    CUSUMEvent,
    Clock,
    DataFeedProtocol,
    EventClock,
    OnlineCUSUMFilterProtocol,
    OnlineEWMAVolatilityProtocol,
    StreamContext,
    StreamingStrategy,
    SyncBrokerFacade,
    TradeEventLike,
)
from quantengine.runtime.streaming.spsc_queue import SpscQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shutdown semantics (S35 D11)
# ---------------------------------------------------------------------------
class ShutdownMode(Enum):
    """How the engine stops.

    DRAIN  — finish queued events, then exit; default.
    CANCEL — stop immediately, abandon queued events.
    HALT   — stop immediately, leave broker state untouched (no cancels).
    """

    DRAIN = "drain"
    CANCEL = "cancel"
    HALT = "halt"


# ---------------------------------------------------------------------------
# Engine metrics (S35 D8 backpressure counter + general observability)
# ---------------------------------------------------------------------------
@dataclass
class EngineMetrics:
    """Mutable counters exposed to tests, watchdog, and external monitoring."""

    events_processed: int = 0
    backpressure_drops: int = 0  # D8 / AC12
    bars_emitted: int = 0
    cusum_events: int = 0
    vol_updates: int = 0
    callback_errors: int = 0
    last_event_ts_ns: int = 0
    # S36b D4 — watchdog alert count. Increments once per detected
    # condition per watchdog tick (feed-silence and health-probe
    # fail are counted independently if they fire in the same tick).
    watchdog_alerts_total: int = 0


# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineConfig:
    """Single-instrument MVP configuration (S35 scope).

    Multi-instrument scaling lives in S36+. The engine treats
    ``instrument_id`` as the single channel of interest; events for
    other instrument_ids are still routed through the pipeline (the
    pipeline primitives decide whether to act).
    """

    instrument_id: int
    ticker: str
    queue_maxsize: int = 1000  # D8
    watchdog_interval_s: float = 1.0
    consumer_poll_timeout_s: float = 0.1
    # S36b D4 — feed-silence detection threshold. If wall-clock time
    # since the last successfully-enqueued event exceeds this value,
    # the watchdog logs a warning and increments
    # `metrics.watchdog_alerts_total`. Default 30s tolerates quiet
    # midday equity sessions; tune lower for futures / FX.
    watchdog_feed_silence_s: float = 30.0
    # S36b D4 — externally-injected broker health probe.
    # AsyncBrokerProtocol does NOT declare is_connected (locked per
    # D7); the engine instead accepts a probe callable wired in by
    # the CLI bootstrap (e.g., wrapping AsyncIBKRBroker's underlying
    # connection.is_connected()). Probe returning False ⇒ alert.
    # None ⇒ no probe, watchdog only checks feed silence.
    health_probe: Callable[[], Awaitable[bool]] | None = field(
        default=None, repr=False, compare=False
    )
    # S39 D4 (REC-004) — externally-injected watchdog-alert callback. Invoked
    # synchronously with a short reason string on EVERY watchdog alert
    # (feed-silence or health-probe failure) via _fire_watchdog_alert. The CLI
    # bootstrap wires this to SafeBroker.trip_kill_switch so a stuck/disconnected
    # feed auto-trips the kill-switch (no new orders) rather than only logging.
    # None ⇒ log-only (pre-S39 behaviour preserved). The engine does NOT import
    # SafeBroker — the dependency stays one-directional (CLI owns the wiring).
    on_watchdog_alert: Callable[[str], None] | None = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# StreamingEngine
# ---------------------------------------------------------------------------
class StreamingEngine:
    """Async event-loop runtime for the S35 streaming pipeline.

    Lifecycle:

        engine = StreamingEngine(feed, builder, cusum, vol, strat, broker, clock, cfg)
        await engine.run()  # blocks until shutdown or feed exhausted
        # OR
        task = asyncio.create_task(engine.run())
        ...
        await engine.shutdown(ShutdownMode.DRAIN, timeout_s=10.0)
        await task
    """

    def __init__(
        self,
        feed: DataFeedProtocol,
        bar_builder: BarBuilderProtocol,
        cusum: OnlineCUSUMFilterProtocol,
        vol: OnlineEWMAVolatilityProtocol,
        strategy: StreamingStrategy,
        broker: SyncBrokerFacade,
        clock: Clock,
        config: EngineConfig,
    ) -> None:
        self._feed = feed
        self._bar_builder = bar_builder
        self._cusum = cusum
        self._vol = vol
        self._strategy = strategy
        self._broker = broker
        self._clock = clock
        self._config = config

        # S36b D3 — SpscQueue replaces asyncio.Queue. The engine has one
        # producer (_feed_loop) and one consumer (_consumer_loop);
        # asyncio.Queue's per-put/get Future allocation + multi-waiter
        # machinery is ~8µs/event of unused overhead vs SpscQueue's
        # deque + single asyncio.Event signal. The
        # put_nowait/get/qsize/empty surface is drop-in compatible with
        # asyncio.Queue, and put_nowait still raises asyncio.QueueFull
        # so the existing backpressure-drops branch below works
        # unchanged.
        self._queue: SpscQueue[TradeEventLike] = SpscQueue(maxsize=config.queue_maxsize)
        self._sequence: int = 0
        self.metrics: EngineMetrics = EngineMetrics()

        self._stop_event = asyncio.Event()
        self._feed_done_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []

        # S36b D4 — wall-clock receipt time of the most recent
        # successfully-enqueued event (monotonic ns). Initialised to
        # the engine-construction time so feed silence isn't detected
        # before the first event lands. Updated in _feed_loop after
        # each successful put_nowait.
        self._last_event_received_at_ns: int = time.monotonic_ns()

        # S36b D2 — dedicated single-worker executor for strategy
        # dispatch. Replaces S35's `asyncio.to_thread` which acquired
        # from the default pool (~min(32, cpu+4) workers) per call.
        # max_workers=1 makes the executor a FIFO serializer matching
        # the strategy callback's implicit ordering invariant; multi-
        # worker would un-order on_bar callbacks under load. The thread
        # name prefix `engine-strategy` is asserted by the regression
        # test suite — a future "modernization" that swaps back to
        # `asyncio.to_thread` would deadlock per
        # ThreadSafeBrokerWrapper's caller-on-non-loop-thread invariant
        # AND would fail the prefix-check.
        self._dispatch_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="engine-strategy",
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Run until the feed exhausts or :meth:`shutdown` is called.

        Spawns three tasks: feed pump, consumer pipeline, watchdog.
        Returns when the consumer drains the queue after feed completion
        OR :meth:`shutdown` has fired.
        """
        feed_task = asyncio.create_task(self._feed_loop(), name="se-feed")
        consumer_task = asyncio.create_task(self._consumer_loop(), name="se-consumer")
        watchdog_task = asyncio.create_task(self._watchdog(), name="se-watchdog")
        self._tasks = [feed_task, consumer_task, watchdog_task]

        try:
            # Consumer is the work-doing task. When it ends (queue drained
            # after feed done OR stop_event fired), we tear down.
            await consumer_task
        finally:
            for t in self._tasks:
                if not t.done():
                    t.cancel()
            # Surface task exceptions (other than CancelledError).
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []
            # S36b D2 — tear down the strategy-dispatch executor.
            # wait=True so any in-flight callback completes before the
            # engine returns. cancel_futures=False because a partially
            # executed strategy callback may have already submitted an
            # order; cancelling its future would leave the broker in
            # an ambiguous state.
            self._dispatch_executor.shutdown(wait=True, cancel_futures=False)

    # ------------------------------------------------------------------
    # Feed pump
    # ------------------------------------------------------------------
    async def _feed_loop(self) -> None:
        """Pull from feed, push onto bounded queue. Backpressure: drop newest."""
        try:
            async for event in self._feed:
                if self._stop_event.is_set():
                    break
                try:
                    self._queue.put_nowait(event)
                    # S36b D4 — wall-clock receipt for feed-silence
                    # detection. Only updated on successful put: a
                    # backpressure-drop is NOT a healthy event for
                    # watchdog purposes (we're already alerting on
                    # drops via the counter above).
                    self._last_event_received_at_ns = time.monotonic_ns()
                except asyncio.QueueFull:
                    # D8: drop newest, increment counter, warn.
                    self.metrics.backpressure_drops += 1
                    logger.warning(
                        "streaming engine backpressure drop: seq=%d ts=%d (drops=%d)",
                        getattr(event, "sequence", -1),
                        getattr(event, "ts_event", -1),
                        self.metrics.backpressure_drops,
                    )
        finally:
            self._feed_done_event.set()

    # ------------------------------------------------------------------
    # Consumer pipeline
    # ------------------------------------------------------------------
    async def _consumer_loop(self) -> None:
        """Drain queue; route each event through pipeline + dispatch."""
        while True:
            # Termination: both feed done AND queue empty.
            if self._feed_done_event.is_set() and self._queue.empty():
                return
            # Hard stop: stop_event AND queue empty (DRAIN finished early).
            if self._stop_event.is_set() and self._queue.empty():
                return
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._config.consumer_poll_timeout_s,
                )
            except asyncio.TimeoutError:
                continue
            await self._process_event(event)

    async def _process_event(self, event: TradeEventLike) -> None:
        """One event through the pipeline; dispatch to strategy."""
        if isinstance(self._clock, EventClock):
            self._clock.tick(event.ts_event)
        self.metrics.last_event_ts_ns = self._clock.now_ns()
        self._sequence += 1
        self.metrics.events_processed += 1

        bar = self._bar_builder.on_event(event)
        if bar is None:
            return
        self.metrics.bars_emitted += 1
        cusum_ts = self._cusum.on_event(bar)
        sigma = self._vol.on_event(bar)

        ctx = StreamContext(
            sequence=self._sequence,
            clock_ns=self._clock.now_ns(),
            instrument_id=event.instrument_id,
        )
        await self._safe_dispatch(self._strategy.on_bar, self._clock.now_ns(), bar, ctx)

        if cusum_ts is not None:
            self.metrics.cusum_events += 1
            ce = CUSUMEvent(
                ts_event=cusum_ts,
                instrument_id=event.instrument_id,
                bar_close=bar.close,
            )
            await self._safe_dispatch(self._strategy.on_cusum, self._clock.now_ns(), ce, ctx)

        if sigma is not None:
            self.metrics.vol_updates += 1
            await self._safe_dispatch(self._strategy.on_vol, self._clock.now_ns(), sigma, ctx)

    async def _safe_dispatch(
        self,
        callback: Any,
        ts: int,
        payload: Any,
        ctx: StreamContext,
    ) -> None:
        """Dispatch a strategy callback via the dedicated single-worker executor.

        Per S36b D2: ``loop.run_in_executor(self._dispatch_executor, ...)``
        replaces S35's ``asyncio.to_thread`` default-pool dispatch. The
        thread-bridge invariant (callback runs off the loop thread so the
        broker's ``run_coroutine_threadsafe`` doesn't deadlock) is
        preserved; the saving is one warm reused thread vs default-pool
        thread acquisition per call.

        Strategy exceptions are caught, counted, and logged — the engine
        does not crash on a misbehaving strategy. The exception detail
        lands in the log; structured handling is the strategy author's
        responsibility.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._dispatch_executor, callback, ts, payload, ctx, self._broker
            )
        except Exception as e:  # broad on purpose: strategy is untrusted
            self.metrics.callback_errors += 1
            logger.exception("strategy callback %s raised: %s", callback.__name__, e)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------
    async def _watchdog(self) -> None:
        """Periodic background check (S36b D4 — feed silence + broker probe).

        Two detection arms:

        1. **Feed silence** — alert if wall-clock time since the last
           successful enqueue exceeds `config.watchdog_feed_silence_s`.
           Compares `time.monotonic_ns()` against the engine's
           `_last_event_received_at_ns` which `_feed_loop` updates on
           every successful put.
        2. **Broker health** — if `config.health_probe` is set, await
           it each tick; a False return ⇒ alert. The probe is
           externally injected (CLI bootstrap wires it; the engine
           does NOT call `is_connected()` on the broker directly
           because `AsyncBrokerProtocol` does not declare it — locked
           per D7).

        Each detection increments `metrics.watchdog_alerts_total` and
        logs a structured `logger.warning`. The watchdog does NOT
        auto-shutdown on alerts in S36b — auto-shutdown policy is an
        S37+ runbook decision. The alert is operator-visible; the
        operational response (cancel orders, halt engine, etc.) lives
        outside the engine.
        """
        feed_silence_deadline_ns = int(self._config.watchdog_feed_silence_s * 1e9)
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._config.watchdog_interval_s)
            except asyncio.CancelledError:
                return

            # Arm 1: feed silence.
            now_ns = time.monotonic_ns()
            silence_ns = now_ns - self._last_event_received_at_ns
            if silence_ns > feed_silence_deadline_ns:
                self.metrics.watchdog_alerts_total += 1
                logger.warning(
                    "streaming engine watchdog: feed silence detected "
                    "(silent_s=%.3f, threshold_s=%.3f, last_event_ts_ns=%d)",
                    silence_ns / 1e9,
                    self._config.watchdog_feed_silence_s,
                    self.metrics.last_event_ts_ns,
                )
                self._fire_watchdog_alert(f"feed_silence_{silence_ns / 1e9:.1f}s")

            # Arm 2: broker health probe (if injected).
            probe = self._config.health_probe
            if probe is not None:
                try:
                    healthy = await probe()
                except Exception as e:  # broad: probe is untrusted
                    self.metrics.watchdog_alerts_total += 1
                    logger.warning(
                        "streaming engine watchdog: health_probe raised %s: %s",
                        type(e).__name__,
                        e,
                    )
                    self._fire_watchdog_alert(f"health_probe_raised_{type(e).__name__}")
                else:
                    if not healthy:
                        self.metrics.watchdog_alerts_total += 1
                        logger.warning("streaming engine watchdog: health_probe returned False")
                        self._fire_watchdog_alert("health_probe_false")

    def _fire_watchdog_alert(self, reason: str) -> None:
        """Invoke the injected watchdog-alert callback, if any (S39 D4).

        Swallows callback exceptions (broad by design): the callback is
        untrusted operator wiring, and a watchdog tick must never crash the
        background task. A failure to trip is logged, not raised.
        """
        cb = self._config.on_watchdog_alert
        if cb is None:
            return
        try:
            cb(reason)
        except Exception as e:  # callback is untrusted; never kill the watchdog
            logger.warning(
                "streaming engine watchdog: on_watchdog_alert callback raised %s: %s",
                type(e).__name__,
                e,
            )

    # ------------------------------------------------------------------
    # Shutdown (S35 D11)
    # ------------------------------------------------------------------
    async def shutdown(
        self,
        mode: ShutdownMode = ShutdownMode.DRAIN,
        timeout_s: float = 10.0,
    ) -> None:
        """Stop the engine per ``mode``.

        DRAIN waits for the consumer to finish queued events (up to
        ``timeout_s``), then falls back to CANCEL behavior.
        """
        if mode is ShutdownMode.DRAIN:
            # Signal feed to stop reading; consumer keeps draining.
            self._stop_event.set()
            # Consumer exits when (feed_done AND queue empty) OR
            # (stop_event AND queue empty). Since stop_event is set, the
            # condition reduces to "queue empty". Wait for that, with
            # timeout fallback.
            try:
                await asyncio.wait_for(self._await_drain(), timeout=timeout_s)
            except asyncio.TimeoutError:
                logger.warning("shutdown DRAIN exceeded %.2fs; falling back to CANCEL", timeout_s)
                # Fall through to cancel tasks below.
        elif mode is ShutdownMode.CANCEL:
            self._stop_event.set()
            # Cancel tasks immediately — drain not required.
        elif mode is ShutdownMode.HALT:
            self._stop_event.set()
            # Same as CANCEL but documented intent: no broker cleanup.
        else:
            raise ValueError(f"unknown ShutdownMode: {mode!r}")

        # Always cancel watchdog (and any other still-running tasks).
        for t in self._tasks:
            if not t.done():
                t.cancel()

    async def _await_drain(self) -> None:
        """Wait until the consumer has emptied the queue."""
        while not self._queue.empty():
            await asyncio.sleep(0.01)


__all__ = [
    "EngineConfig",
    "EngineMetrics",
    "ShutdownMode",
    "StreamingEngine",
]
