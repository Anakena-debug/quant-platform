"""Order lifecycle state machine.

Problem
-------
Phase-1 ``PaperBroker`` emits fills synchronously and ``Runner`` appends
``ORDER_SUBMITTED`` / ``ORDER_FILLED`` to the ``Ledger`` directly. That is
fine for batch replay, but it carries three production-critical gaps:

    (i)   There is no single source of truth for the *current* status of
          an in-flight order (PENDING vs SUBMITTED vs WORKING vs …).
    (ii)  Partial fills are representable (``Fill.signed_quantity`` is a
          slice) but nothing enforces that the sum of fills never exceeds
          the parent order's ``quantity``.
    (iii) Illegal transitions (e.g. ``FILLED → FILLED``, or a fill arriving
          for an unknown order_id) go unnoticed.

The ``OrderTracker`` closes these gaps with a tiny finite-state machine
while remaining fully backward-compatible: when the ``Runner`` is
constructed without a tracker, the Phase-1 ledger path is unchanged.

Transition table (US-equities paper/live)
-----------------------------------------
The legal transition relation :math:`\\mathcal{T} \\subseteq S \\times S`
is enumerated in ``_LEGAL`` below. In summary:

    PENDING           ──▶ SUBMITTED | REJECTED
    SUBMITTED         ──▶ WORKING | PARTIALLY_FILLED | FILLED | REJECTED | CANCELLED
    WORKING           ──▶ PARTIALLY_FILLED | FILLED | CANCELLED
    PARTIALLY_FILLED  ──▶ PARTIALLY_FILLED | FILLED | CANCELLED
    FILLED            ──▶ ∅     (terminal)
    CANCELLED         ──▶ ∅     (terminal)
    REJECTED          ──▶ ∅     (terminal)

``SUBMITTED → FILLED`` is permitted as a convenience for synchronous paper
brokers that never emit a separate acknowledgment. The asynchronous IBKR
adapter (Phase 3) will always traverse ``SUBMITTED → WORKING → …``.

Conservation laws enforced
--------------------------
For every tracked order :math:`o` with signed target :math:`q^\\star`,
and cumulative signed fill :math:`Q_t`:

    1. :math:`\\operatorname{sign}(f) = \\operatorname{sign}(q^\\star)`
       for every incoming fill :math:`f` (no flip-through-zero within a
       single order).
    2. :math:`|Q_t + f| \\le |q^\\star|` for every incoming fill (no
       overfill).
    3. :math:`|Q_t| = |q^\\star| \\Longleftrightarrow \\text{status} =
       \\mathrm{FILLED}` (cumulative matches ⇔ terminal-filled).

Violations raise ``OrderStateError`` and leave the tracker unchanged —
the error surfaces loudly to the event loop; no partial silent updates.

Integration points
------------------
- ``Runner.step`` routes through ``OrderTracker.submit`` and
  ``OrderTracker.on_fill`` when a tracker is attached. All ledger writes
  remain append-only; the tracker is the only path that emits
  ``ORDER_ACKED`` / ``ORDER_REJECTED`` / ``ORDER_WORKING`` / partial /
  terminal ``ORDER_FILLED`` events with the right state annotations.
- Phase 3 IBKR adapter will call the same API from its async callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from uuid import UUID

from quantengine.contracts.orders import Fill, Order, OrderStatus
from quantengine.portfolio.ledger import Ledger


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class OrderStateError(RuntimeError):
    """Raised when a state-machine invariant is violated.

    Subclass of ``RuntimeError`` rather than ``ValueError`` because these
    violations indicate *logic* bugs (wrong caller, wrong broker, race
    condition), not merely bad user input.
    """


# ---------------------------------------------------------------------------
# Legal transition relation
# ---------------------------------------------------------------------------
_LEGAL: Mapping[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset(
        {
            OrderStatus.SUBMITTED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,  # sync brokers: fill arrives before any ack
            OrderStatus.FILLED,  # sync brokers: full fill in one step
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.WORKING: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,  # more partials allowed
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def is_terminal(status: OrderStatus) -> bool:
    return not _LEGAL[status]


def is_legal_transition(src: OrderStatus, dst: OrderStatus) -> bool:
    return dst in _LEGAL[src]


# ---------------------------------------------------------------------------
# Per-order record
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _OrderRecord:
    order: Order
    status: OrderStatus = OrderStatus.PENDING
    cumulative_filled: int = 0  # signed shares filled so far

    @property
    def remaining(self) -> int:
        """Signed remaining shares to fill (target − cumulative)."""
        return self.order.signed_quantity - self.cumulative_filled

    @property
    def target(self) -> int:
        return self.order.signed_quantity


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
@dataclass
class OrderTracker:
    """Finite-state machine over the lifecycle of every submitted order.

    All mutators are synchronous and thread-*hostile* by design — the
    Phase-1 runner is single-threaded, and Phase-3 IBKR callbacks should
    marshal into the main event loop before calling these methods.

    Every state transition emits an event to the attached ``Ledger`` so
    the audit chain (``quantengine.audit.journal``) sees the full
    lifecycle, not just submit/fill.
    """

    ledger: Ledger
    _orders: dict[UUID, _OrderRecord] = field(default_factory=dict)

    # ---------------- introspection --------------------------------------
    def status(self, order_id: UUID) -> OrderStatus:
        return self._require(order_id).status

    def cumulative_filled(self, order_id: UUID) -> int:
        return self._require(order_id).cumulative_filled

    def remaining(self, order_id: UUID) -> int:
        return self._require(order_id).remaining

    def open_orders(self) -> tuple[UUID, ...]:
        return tuple(oid for oid, r in self._orders.items() if not is_terminal(r.status))

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    # ---------------- transitions ----------------------------------------
    def submit(self, order: Order, timestamp: str) -> None:
        """Register a newly submitted order: PENDING → SUBMITTED.

        The caller (``Runner``) has already computed the order via the
        ``RebalanceEngine``; this method asserts the order is fresh and
        writes ``ORDER_SUBMITTED`` to the ledger.
        """
        if order.order_id in self._orders:
            raise OrderStateError(
                f"Duplicate order_id {order.order_id}; tracker already has this order."
            )
        rec = _OrderRecord(order=order, status=OrderStatus.PENDING)
        self._transition(rec, OrderStatus.SUBMITTED)
        self._orders[order.order_id] = rec
        self.ledger.append(timestamp, "ORDER_SUBMITTED", order)

    def ack(self, order_id: UUID, timestamp: str) -> None:
        """Broker acknowledged: SUBMITTED → WORKING.

        Emits a distinct ``ORDER_ACKED`` event so downstream audit can
        measure submit→ack latency (a key microstructure metric).
        """
        rec = self._require(order_id)
        self._transition(rec, OrderStatus.WORKING)
        self.ledger.append(
            timestamp,
            "ORDER_ACKED",
            {"order_id": str(order_id), "new_status": OrderStatus.WORKING.value},
        )

    def on_fill(self, fill: Fill) -> OrderStatus:
        """Ingest a Fill, enforce conservation laws, advance the state.

        Returns the new ``OrderStatus`` after this fill. Semantics:

        - a fill that does *not* close out the order → PARTIALLY_FILLED
          (self-loop from PARTIALLY_FILLED allowed);
        - a fill whose cumulative matches the target → FILLED.

        The ledger event is always ``ORDER_FILLED`` (terminal vs partial
        is discoverable from status via ``cumulative_filled``); this
        preserves backward-compatibility with the Phase-1 schema.
        """
        rec = self._require(fill.order_id)
        if rec.order.ticker != fill.ticker:
            raise OrderStateError(
                f"Fill ticker {fill.ticker!r} does not match "
                f"order ticker {rec.order.ticker!r} for order_id={fill.order_id}."
            )
        if fill.signed_quantity == 0:
            raise OrderStateError(
                f"Zero-quantity fill for order_id={fill.order_id} is not allowed."
            )
        # Invariant 1: same side as the target.
        if (fill.signed_quantity > 0) != (rec.target > 0):
            raise OrderStateError(
                f"Fill direction (signed_quantity={fill.signed_quantity}) "
                f"disagrees with order target ({rec.target}) for "
                f"order_id={fill.order_id}."
            )
        # Invariant 2: no overfill (|cum+fill| <= |target|).
        new_cum = rec.cumulative_filled + fill.signed_quantity
        if abs(new_cum) > abs(rec.target):
            raise OrderStateError(
                f"Overfill: target={rec.target}, cumulative_before="
                f"{rec.cumulative_filled}, fill={fill.signed_quantity}. "
                f"Resulting |{new_cum}| > |{rec.target}|."
            )

        fully_filled = new_cum == rec.target
        dst = OrderStatus.FILLED if fully_filled else OrderStatus.PARTIALLY_FILLED
        self._transition(rec, dst)
        rec.cumulative_filled = new_cum
        self.ledger.append(fill.timestamp, "ORDER_FILLED", fill)
        return dst

    def cancel(self, order_id: UUID, timestamp: str, reason: str = "") -> None:
        """Cancel an open order. Terminal state; no further fills accepted."""
        rec = self._require(order_id)
        self._transition(rec, OrderStatus.CANCELLED)
        self.ledger.append(
            timestamp,
            "ORDER_CANCELLED",
            {
                "order_id": str(order_id),
                "reason": reason,
                "cumulative_filled": int(rec.cumulative_filled),
                "remaining": int(rec.remaining),
            },
        )

    def reject(self, order_id: UUID, timestamp: str, reason: str = "") -> None:
        """Reject an order (broker/pre-trade gate). Terminal."""
        rec = self._require(order_id)
        self._transition(rec, OrderStatus.REJECTED)
        self.ledger.append(
            timestamp,
            "ORDER_REJECTED",
            {
                "order_id": str(order_id),
                "reason": reason,
            },
        )

    # ---------------- internals ------------------------------------------
    def _require(self, order_id: UUID) -> _OrderRecord:
        try:
            return self._orders[order_id]
        except KeyError:
            raise OrderStateError(f"Unknown order_id={order_id}; no such order tracked.") from None

    @staticmethod
    def _transition(rec: _OrderRecord, dst: OrderStatus) -> None:
        if not is_legal_transition(rec.status, dst):
            raise OrderStateError(
                f"Illegal transition {rec.status.value} → {dst.value} "
                f"for order_id={rec.order.order_id}."
            )
        rec.status = dst


__all__ = [
    "OrderStateError",
    "OrderTracker",
    "is_legal_transition",
    "is_terminal",
]
