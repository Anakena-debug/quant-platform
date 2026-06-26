"""StreamingReconciler — event-driven position reconciliation (S37 D1+D2).

Compares the in-process ``VirtualPortfolio`` (engine-side cumulative
state via SafeBroker fills) against the broker's authoritative
position snapshot, on every Fill, with a debounce that absorbs
position-event-after-fill latency. On quantity divergence exceeding
a configured tolerance, the reconciler triggers an async
``halt_callback`` (typically `engine.shutdown(ShutdownMode.CANCEL)`).

Per D1 — event-driven join with debounce:

  1. On local Fill (from SafeBroker.on_fill), arm or extend a 100 ms
     per-ticker timer.
  2. On broker positionEvent for the same ticker, cancel the timer
     and reconcile using the broker-pushed position (no extra
     get_position round-trip).
  3. If the timer fires before a positionEvent arrives, fall back to
     one broker.get_position(ticker) call.
  4. Additional Fills for the same ticker inside the debounce window
     RESET the timer (single reconciliation per fill-burst, not per
     Fill — reduces broker-side load and avoids redundant compares).

Per D2 — quantity-only comparison; 1-share tolerance; avg-price as
WARNING not halt:

  - Compare VirtualPortfolio.positions[ticker].quantity vs
    broker_position.quantity (signed; None on either side ⇒ 0).
  - abs(diff) <= tolerance ⇒ silent pass.
  - abs(diff) > tolerance ⇒ trigger halt_callback with a structured
    reason. After halt, subsequent fills are ignored (the engine is
    shutting down).
  - Default tolerance = 1 share (absorbs signal-side rounding +
    in-flight events between the local Fill callback and the broker
    positionEvent).
  - Average price (cost basis) divergence between VirtualPortfolio's
    weighted average and the broker's avgCost is logged as a
    structured WARNING but NEVER halts. Cost-basis methodology
    differs between accounting systems (FIFO vs LIFO vs weighted);
    halting on every fill against an existing position would be
    useless. The journal preserves the divergence record for
    post-hoc analysis.

S37 boundaries:

  - Reconciler does NOT enable real-money trading. Real-money cutover
    is a follow-up sprint with its own plan-commit (D6 cutover-gate
    runbook in S37 PR5 documents the precondition).
  - Reconciler does NOT modify the engine. Halt signalling is via
    the injected async ``halt_callback``; the engine + SafeBroker +
    wrappers stay locked at the S35 contract surface.
  - Reconciler does NOT subscribe broker positionEvent itself. The
    CLI / wiring layer subscribes the broker's position-push and
    forwards to ``StreamingReconciler.on_position_event``.
    AsyncBrokerProtocol does NOT declare a position-event surface
    (it's locked); position-push wiring lives in the operator's
    integration code per D1 step 2.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final

from quantengine.runtime.streaming.protocols import AsyncBrokerProtocol

if TYPE_CHECKING:
    from quantengine.contracts.orders import Fill
    from quantengine.portfolio.state import PortfolioState, Position

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE: Final[int] = 1
_DEFAULT_DEBOUNCE_S: Final[float] = 0.1
# Avg-price WARNING threshold. Cost-basis divergence below this is
# numerical noise; above is structural divergence worth flagging.
# 0.01 = 1 cent on equity prices.
_DEFAULT_AVG_PRICE_WARN_THRESHOLD: Final[float] = 0.01


VirtualPortfolioProvider = Callable[[], "PortfolioState"]
"""Callable returning the current engine-side PortfolioState snapshot.

Mirrors the SafeBroker state_provider pattern from S35: the reconciler
reads the state at reconciliation time so a series of fills accumulates
correctly before the comparison fires.
"""

HaltCallback = Callable[[str], Awaitable[None]]
"""Async callback the reconciler awaits when divergence exceeds tolerance.

Production wiring: ``lambda reason: engine.shutdown(ShutdownMode.CANCEL,
timeout_s=10.0)``. Operator runbooks consume the reason string for
post-incident forensics.
"""


class StreamingReconciler:
    """Event-driven position reconciler with debounce.

    Construct once per engine; wire ``on_fill`` to SafeBroker's fill
    stream and ``on_position_event`` to the broker's position-push
    (operator wiring per D1 step 2). The reconciler does NOT poll —
    it runs only when an event arrives.

    After the halt_callback fires once, the reconciler enters a
    sticky halted state: further events are ignored. This avoids
    re-triggering the halt path during the engine's CANCEL drain.
    """

    def __init__(
        self,
        broker: AsyncBrokerProtocol,
        virtual_portfolio_provider: VirtualPortfolioProvider,
        halt_callback: HaltCallback,
        *,
        tolerance: int = _DEFAULT_TOLERANCE,
        debounce_s: float = _DEFAULT_DEBOUNCE_S,
        avg_price_warn_threshold: float = _DEFAULT_AVG_PRICE_WARN_THRESHOLD,
    ) -> None:
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0; got {tolerance}")
        if debounce_s <= 0:
            raise ValueError(f"debounce_s must be > 0; got {debounce_s}")
        self._broker = broker
        self._vp_provider = virtual_portfolio_provider
        self._halt_callback = halt_callback
        self._tolerance = tolerance
        self._debounce_s = debounce_s
        self._avg_price_warn_threshold = avg_price_warn_threshold
        self._pending_timers: dict[str, asyncio.TimerHandle] = {}
        self._pending_tasks: dict[str, asyncio.Task[None]] = {}
        self._halted = False

    @property
    def halted(self) -> bool:
        """True once the halt_callback has been invoked. Sticky."""
        return self._halted

    @property
    def tolerance(self) -> int:
        """Per-ticker quantity divergence tolerance in shares. Config-tunable."""
        return self._tolerance

    def on_fill(self, fill: Fill) -> None:
        """Arm (or extend) the debounce timer for this ticker.

        Burst-of-fills behaviour: subsequent fills for the same
        ticker before the timer fires CANCEL the existing timer and
        re-arm it, so reconciliation runs once after the burst
        stabilises (D1 step 4).
        """
        if self._halted:
            return
        ticker = fill.ticker
        # Cancel any in-flight timer for this ticker — burst-extends the debounce.
        existing = self._pending_timers.pop(ticker, None)
        if existing is not None:
            existing.cancel()
        loop = asyncio.get_running_loop()
        self._pending_timers[ticker] = loop.call_later(
            self._debounce_s,
            self._on_debounce_fire,
            ticker,
        )

    def _on_debounce_fire(self, ticker: str) -> None:
        """Timer-thread callback: schedule the async fallback reconcile.

        ``loop.call_later`` callbacks are sync; we need to invoke the
        async broker.get_position. Schedule it via create_task and
        track for cleanup.
        """
        self._pending_timers.pop(ticker, None)
        if self._halted:
            return
        task = asyncio.create_task(
            self._reconcile_via_broker_get(ticker),
            name=f"reconciler-fallback-{ticker}",
        )
        self._pending_tasks[ticker] = task
        task.add_done_callback(lambda t: self._pending_tasks.pop(ticker, None))

    async def _reconcile_via_broker_get(self, ticker: str) -> None:
        """Fallback path: query broker position via AsyncBrokerProtocol."""
        if self._halted:
            return
        broker_position = await self._broker.get_position(ticker)
        await self._reconcile(ticker, broker_position)

    async def on_position_event(self, ticker: str, broker_position: Position | None) -> None:
        """Broker pushed a position update — fast path reconcile + cancel timer.

        If no pending Fill triggered a timer for this ticker, the
        position-event is unsolicited (broker pushed without our
        having traded recently); we ignore it. The reconciler is
        Fill-driven by design (D1).
        """
        if self._halted:
            return
        timer = self._pending_timers.pop(ticker, None)
        if timer is None:
            # No pending Fill for this ticker — uncorrelated push, ignore.
            return
        timer.cancel()
        # Cancel any in-flight fallback task too (shouldn't normally
        # exist; defensive).
        task = self._pending_tasks.pop(ticker, None)
        if task is not None and not task.done():
            task.cancel()
        await self._reconcile(ticker, broker_position)

    async def _reconcile(self, ticker: str, broker_position: Position | None) -> None:
        """Compare VirtualPortfolio vs broker. Quantity-only halt; avg-price WARN."""
        if self._halted:
            return
        vp_state = self._vp_provider()
        vp_position = vp_state.positions.get(ticker)
        vp_qty = vp_position.quantity if vp_position is not None else 0
        broker_qty = broker_position.quantity if broker_position is not None else 0
        diff = abs(vp_qty - broker_qty)

        if diff > self._tolerance:
            self._halted = True
            reason = (
                f"position divergence exceeds tolerance: ticker={ticker!r} "
                f"vp_qty={vp_qty} broker_qty={broker_qty} diff={diff} "
                f"tolerance={self._tolerance}"
            )
            logger.error("reconciler halt: %s", reason)
            await self._halt_callback(reason)
            return

        # Within tolerance — quantity OK. Check avg-price for WARNING.
        if vp_position is not None and broker_position is not None:
            vp_avg = vp_position.avg_cost
            broker_avg = broker_position.avg_cost
            avg_diff = abs(vp_avg - broker_avg)
            if avg_diff > self._avg_price_warn_threshold:
                logger.warning(
                    "reconciler avg-price divergence: ticker=%r "
                    "vp_avg=%.4f broker_avg=%.4f diff=%.4f (within quantity tolerance)",
                    ticker,
                    vp_avg,
                    broker_avg,
                    avg_diff,
                )

    async def aclose(self) -> None:
        """Graceful teardown: cancel pending timers + tasks."""
        for timer in self._pending_timers.values():
            timer.cancel()
        self._pending_timers.clear()
        # Cancel any in-flight fallback tasks
        for task in list(self._pending_tasks.values()):
            if not task.done():
                task.cancel()
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks.values(), return_exceptions=True)
        self._pending_tasks.clear()


__all__ = [
    "HaltCallback",
    "StreamingReconciler",
    "VirtualPortfolioProvider",
]
