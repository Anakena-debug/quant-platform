"""IBKRBroker — the AbstractBroker implementation against ib_async.

PAPER ONLY. The broker depends on ``IBKRConnection``, which validates
paper-only ports and paper-prefix accounts at config-construction
time and cross-checks ``ib.managedAccounts()`` at connect time.

Sync-with-timeout boundary: ``submit_orders`` is synchronous
(preserving the ``AbstractBroker`` contract); internally drives the
asyncio loop via ``connection.ib.sleep(_LOOP_SLICE_SECONDS)`` until
every submitted order reaches a terminal status or one of two
timeouts fires:

- per-order: cancel that order at ``timeouts.per_order_seconds``;
- batch ceiling: cancel any remaining orders at
  ``timeouts.batch_ceiling_seconds``.

The naïve alternative loop primitive (the one ib_insync used to
expose for "wait at most N seconds for an update") is forbidden in
this module — verified broken on ib_async 2.x on 2026-05-07 (returns
in 0.000s regardless of timeout). ``IB.sleep`` is
the verified working primitive (returns in ~0.501s for a 0.5s sleep,
processes callbacks during the sleep).

PHASE 3 LIMITATION — hard-fail on mid-cycle disconnect. The
``OrderTracker`` is in-process; if the IBKR socket drops mid-cycle,
the in-process tracker and the IBKR server-side state diverge. S22
policy: propagate the exception, halt the cycle, restart manually.
Phase 4 will add ``OrderTracker`` persistence + replay-on-reconnect.

Order IDs are IB-assigned via ``IB.placeOrder`` (which calls
``client.getReqId()`` internally) — never client-generated, which
would cause duplicate-order rejections at the IBKR side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.ibkr.config import TimeoutPolicy
from quantengine.execution.ibkr.connection import IBKRConnection
from quantengine.execution.ibkr.order_mapping import (
    ib_trade_to_fill,
    order_to_ib_order,
)
from quantengine.execution.order_state import OrderTracker, is_terminal

if TYPE_CHECKING:
    from ib_async import Trade

# How long ``IB.sleep`` blocks per loop iteration. Smaller = more
# responsive timeout cancellation; larger = less loop overhead. 0.25s
# is a balance — at p99 paper-fill latency of ~2s it gives ~8 chances
# to observe terminal status before per_order_seconds default of 60s
# fires.
_LOOP_SLICE_SECONDS: float = 0.25


@dataclass
class IBKRBroker(AbstractBroker):
    """PAPER-only IBKR adapter implementing ``AbstractBroker``.

    Construction is cheap and side-effect-free. The caller is
    responsible for ``connection.connect(config)`` and
    ``assert_paper_account(connection.ib, config.account)`` before
    invoking ``submit_orders``.
    """

    connection: IBKRConnection
    tracker: OrderTracker | None = None
    timeouts: TimeoutPolicy = field(default_factory=TimeoutPolicy)

    def submit_orders(self, orders: Sequence[Order], market: MarketSnapshot) -> list[Fill]:
        """Submit orders, drive the loop, return collected fills.

        Empty order list returns ``[]`` without touching the broker.

        Each order is placed via ``connection.ib.placeOrder``, which
        auto-assigns ``orderId`` via ``client.getReqId()`` — never
        set client-side.

        Two clocks bound the wait:

        - ``timeouts.per_order_seconds``: request cancellation of
          that order via ``ib.cancelOrder``; the tracker stays
          non-terminal until IBKR confirms with a Cancelled /
          ApiCancelled status.
        - ``timeouts.batch_ceiling_seconds``: request cancellation
          of all remaining orders the same way.

        The loop drives the asyncio event loop via
        ``connection.ib.sleep(_LOOP_SLICE_SECONDS)`` and continues
        until every Trade reports ``isDone()``. Callbacks
        (fillEvent / statusEvent) wired in ``_wire_trade_callbacks``
        route into ``OrderTracker.ack`` / ``on_fill`` / ``cancel`` /
        ``reject``.

        Race safety on cancellation: timeout never marks the tracker
        terminal directly. Doing so would silently drop any fill that
        arrives after we issued ``cancelOrder`` but before IBKR
        confirms the cancellation — the fill callback would see a
        terminal tracker and skip ``tracker.on_fill``, leaving the
        ledger and the returned fills list divergent. Instead we
        record the local reason in ``pending_cancel_reasons`` and let
        the broker's Cancelled / ApiCancelled status callback perform
        the actual ``tracker.cancel`` transition (using the local
        reason if present, else ``whyHeld``).

        Post-loop reconciliation: once every Trade is done, every
        submitted order MUST be in a tracker-terminal state. If any
        is not, the function raises ``RuntimeError`` — this catches
        unmapped IBKR statuses or missed callbacks loudly rather
        than silently leaving the ledger inconsistent.
        """
        if not orders:
            return []

        if not self.connection.is_connected():
            raise RuntimeError(
                "IBKRBroker.submit_orders called before "
                "IBKRConnection.connect() — refusing to proceed"
            )

        ib = self.connection.ib
        per_order = self.timeouts.per_order_seconds
        batch_ceiling = self.timeouts.batch_ceiling_seconds

        collected_fills: list[Fill] = []

        # Submit all orders + wire callbacks.
        trades: list[Trade] = []
        order_by_trade: dict[int, Order] = {}
        deadlines: dict[int, float] = {}
        # Reason chosen by the broker (timeout reason); consulted by the
        # statusEvent callback when the broker confirms cancellation.
        # Keyed by id(trade); never reused across submit_orders calls.
        pending_cancel_reasons: dict[int, str] = {}

        for order in orders:
            contract, ib_order = order_to_ib_order(order)
            # IB.placeOrder auto-assigns orderId via client.getReqId()
            trade = ib.placeOrder(contract, ib_order)
            trades.append(trade)
            order_by_trade[id(trade)] = order
            deadlines[id(trade)] = time.monotonic() + per_order

            # Pre-track via OrderTracker (PENDING → SUBMITTED).
            if self.tracker is not None:
                self.tracker.submit(order, market.timestamp)

            self._wire_trade_callbacks(
                trade, order, collected_fills, market, pending_cancel_reasons
            )

        batch_deadline = time.monotonic() + batch_ceiling
        batch_cancel_issued = False

        # Drive loop via IB.sleep until all terminal or timeouts fire.
        while any(not t.isDone() for t in trades):
            now = time.monotonic()

            # Timeout means "request cancellation", NOT "confirmed
            # cancellation". The tracker stays non-terminal until the
            # broker emits Cancelled / ApiCancelled (handled in
            # on_status_cb). Doing the terminal transition here would
            # race with any execution fill that lands between this
            # cancelOrder call and the broker's confirmation, silently
            # dropping that fill from the tracker/ledger while still
            # appending it to collected_fills.
            #
            # We do NOT break out of the loop. The cancelOrder bytes
            # have to flush through the asyncio socket and the
            # Cancelled callbacks have to land before each Trade.isDone
            # flips True. The batch_cancel_issued flag prevents
            # re-issuing cancellations on subsequent iterations.
            if now >= batch_deadline and not batch_cancel_issued:
                for t in trades:
                    if not t.isDone():
                        pending_cancel_reasons.setdefault(id(t), "batch_ceiling_timeout")
                        ib.cancelOrder(t.order)
                batch_cancel_issued = True

            for t in trades:
                if not t.isDone() and now >= deadlines[id(t)]:
                    pending_cancel_reasons.setdefault(id(t), "per_order_timeout")
                    ib.cancelOrder(t.order)
                    deadlines[id(t)] = float("inf")  # don't re-cancel

            # Drive the asyncio loop one slice at a time.
            ib.sleep(_LOOP_SLICE_SECONDS)

        # Post-loop reconciliation: every submitted order MUST be in
        # a tracker-terminal state. A non-terminal tracker after
        # isDone() == True indicates an unmapped IBKR status (e.g. a
        # new status string we don't handle in on_status_cb) or a
        # missed callback. Surface loudly rather than silently
        # leaving the ledger inconsistent with the broker.
        if self.tracker is not None:
            for t in trades:
                order = order_by_trade[id(t)]
                current = self.tracker.status(order.order_id)
                if not is_terminal(current):
                    ib_status = str(t.orderStatus.status)
                    raise RuntimeError(
                        "IBKR Trade reached done state but OrderTracker "
                        "remains non-terminal: "
                        f"order_id={order.order_id}, "
                        f"ib_order_id={getattr(t.order, 'orderId', None)}, "
                        f"ib_status={ib_status!r}, "
                        f"tracker_status={current.value!r}. "
                        "Likely an unmapped IBKR terminal status — extend "
                        "on_status_cb in IBKRBroker._wire_trade_callbacks."
                    )

        return collected_fills

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled.

        Calls ``ib.reqGlobalCancel()`` and waits one slice for the
        cancels to register, then diffs ``len(openTrades())``
        before-vs-after.
        """
        if not self.connection.is_connected():
            return 0
        ib = self.connection.ib
        before = len(list(ib.openTrades()))
        ib.reqGlobalCancel()
        ib.sleep(_LOOP_SLICE_SECONDS)
        after = len(list(ib.openTrades()))
        return max(0, before - after)

    def open_orders(self) -> Iterable[Order]:
        """Return open orders. Phase 3 returns empty.

        ib_async exposes ``openTrades()`` directly but the inverse
        mapping from ``ib_async.Order`` back to our ``Order`` dataclass
        is lossy (no reverse lookup stored). Phase 4 will track this
        via ``OrderTracker`` persistence.
        """
        return ()

    def _wire_trade_callbacks(
        self,
        trade: Trade,
        order: Order,
        collected_fills: list[Fill],
        market: MarketSnapshot,
        pending_cancel_reasons: dict[int, str],
    ) -> None:
        """Wire ``fillEvent`` + ``statusEvent`` → tracker calls + fill list.

        Closures capture ``order``, ``market``, and the broker-scoped
        ``pending_cancel_reasons`` map for each trade. The ``seen_ack``
        flag (per-trade) defends against the ack arriving AFTER the
        first fill (sync brokers / paper sim collapse the order); the
        ``OrderTracker`` allows ``SUBMITTED → FILLED`` directly, but we
        also call ``ack`` before the first ``on_fill`` to ensure the
        audit chain captures the WORKING state when the broker did
        ack separately.

        Fill ordering: the tracker invariants run BEFORE the fill is
        appended to ``collected_fills``. A fill that arrives after the
        tracker reaches a terminal state — which after the timeout
        race fix should only happen on a true broker-side anomaly
        (IBKR filled an order it told us was cancelled) — raises
        ``RuntimeError`` and is NOT appended. This keeps the returned
        fills list and the ledger consistent: every entry in the
        returned list is also in the ledger via ``tracker.on_fill``.

        Status mapping:

        - ``Submitted`` / ``PreSubmitted`` → ``tracker.ack`` once.
        - ``Cancelled`` / ``ApiCancelled`` → ``tracker.cancel`` with
          the local timeout reason (from ``pending_cancel_reasons``)
          if a cancel was requested by us, else ``whyHeld`` text from
          the broker.
        - ``Inactive`` → terminal mapping. From ``SUBMITTED`` we route
          to ``tracker.reject`` (legal pre-trade refusal). From
          ``WORKING`` / ``PARTIALLY_FILLED`` we route to
          ``tracker.cancel`` because the legal transition table
          forbids ``WORKING → REJECTED``.

        Idempotency on terminal state: ``on_status_cb`` short-circuits
        if the tracker is already terminal so a duplicate Cancelled
        callback after we've already routed the cancel is a no-op.
        """
        seen_ack = [False]

        def on_fill_cb(_t: Trade, fill_event: Any) -> None:
            fill = ib_trade_to_fill(_t, order, fill_event)

            if self.tracker is None:
                collected_fills.append(fill)
                return

            current = self.tracker.status(order.order_id)
            if is_terminal(current):
                # True broker-side anomaly: IBKR delivered a fill for
                # an order we believe is in a terminal state. Halting
                # the cycle is safer than silently growing the
                # collected_fills/ledger divergence the previous code
                # did.
                raise RuntimeError(
                    "IBKR fill arrived after local terminal state for "
                    f"order_id={order.order_id}, "
                    f"tracker_status={current.value!r}. "
                    "Halt cycle and reconcile manually."
                )

            # Defensive ack: route SUBMITTED → WORKING if not already done.
            if not seen_ack[0] and current.value == "SUBMITTED":
                self.tracker.ack(order.order_id, market.timestamp)
                seen_ack[0] = True

            self.tracker.on_fill(fill)
            collected_fills.append(fill)

        def on_status_cb(_t: Trade) -> None:
            if self.tracker is None:
                return

            status = str(_t.orderStatus.status)
            current = self.tracker.status(order.order_id)
            if is_terminal(current):
                return

            if status in ("Submitted", "PreSubmitted") and not seen_ack[0]:
                self.tracker.ack(order.order_id, market.timestamp)
                seen_ack[0] = True
                return

            if status in ("Cancelled", "ApiCancelled"):
                broker_reason = str(_t.orderStatus.whyHeld or "cancelled")
                reason = pending_cancel_reasons.get(id(_t), broker_reason)
                self.tracker.cancel(order.order_id, market.timestamp, reason=reason)
                return

            if status == "Inactive":
                reason_parts = ["inactive"]
                why_held = str(_t.orderStatus.whyHeld or "")
                if why_held:
                    reason_parts.append(f"whyHeld={why_held}")
                advanced_error = str(getattr(_t, "advancedError", "") or "")
                if advanced_error:
                    reason_parts.append(f"advancedError={advanced_error}")
                reason = "; ".join(reason_parts)

                # SUBMITTED → REJECTED is legal (pre-trade refusal).
                # From any other non-terminal state the legal table
                # forbids REJECTED, so route to CANCELLED to preserve
                # the audit trail without crashing the loop.
                if current.value == "SUBMITTED":
                    self.tracker.reject(order.order_id, market.timestamp, reason=reason)
                else:
                    self.tracker.cancel(order.order_id, market.timestamp, reason=reason)
                return

        trade.fillEvent += on_fill_cb
        trade.statusEvent += on_status_cb


__all__ = ["IBKRBroker"]
