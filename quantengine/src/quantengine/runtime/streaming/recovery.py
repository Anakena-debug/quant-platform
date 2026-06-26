"""RecoveryCoordinator — engine restart-recovery sequence (S37 D4 + D5).

On engine startup after a crash or graceful shutdown, RecoveryCoordinator:

  1. Reads the JSONL journal sequentially from start.
  2. For each record, dispatches to OrderTracker (state transitions)
     or VirtualPortfolio (Fill application).
  3. Tolerates a truncated last line (crash mid-fsync): JSON parse
     failure on the final line is logged + skipped. Parse failures
     mid-file raise — they indicate real corruption.
  4. For each order in non-terminal state (PENDING / SUBMITTED /
     WORKING / PARTIALLY_FILLED) after replay, queries the broker
     for that order's current state via the injected
     ``order_state_probe`` callable. Resolves to one of:
       - Filled    → applies fills to VP + transitions tracker
       - Cancelled → transitions tracker to CANCELLED
       - Rejected  → transitions tracker to REJECTED
       - Working   → leave non-terminal (carry forward)
       - Unknown   → recorded as unresolved; halt at cross-check time
  5. Queries broker for live positions and compares against the
     rebuilt VirtualPortfolio with the configured tolerance.
     Within-tolerance ⇒ recovery clean ⇒ engine may resume.
     Exceeds ⇒ halted=True ⇒ operator intervention required.

Per D7 (locked at plan-write): ``AsyncBrokerProtocol`` does NOT
declare a per-order lookup method, and ``protocols.py`` is locked by
S37 forbidden_actions. The recovery code accepts an injected
``order_state_probe`` callable (same pattern as S36b D4's
``health_probe``); operator wiring layer translates broker-specific
order-by-id queries (e.g., ib_async's ``trades()`` filter or REST
order-status endpoint) into the probe's contract.

VirtualPortfolio aliases PortfolioState in this module for semantic
clarity — the engine-side cumulative state is conceptually distinct
from the broker's authoritative truth even though they share the
same Python class. Reconciler module uses the same naming
convention.

S37 boundaries:

  - No real-money cutover (S38+).
  - Does not modify OrderTracker (S22) or PortfolioState reducer
    (quantcore-side reducer untouched).
  - Does not write to the journal — read-only at startup.
  - Does not start the engine; reports a ``RecoveryResult`` for the
    operator runbook to act on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from quantengine.contracts.orders import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quantengine.execution.order_state import OrderTracker, is_terminal
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState as VirtualPortfolio
from quantengine.runtime.streaming.protocols import AsyncBrokerProtocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE: Final[int] = 1


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedOrderState:
    """What the broker reports for a non-terminal order during recovery.

    The injected ``order_state_probe`` returns one of these per
    queried order_id. ``status`` is one of OrderStatus values
    (FILLED / CANCELLED / REJECTED / WORKING) — UNKNOWN is signalled
    by the probe returning ``None`` to keep the contract simple.
    """

    order_id: UUID
    status: OrderStatus
    fills: tuple[Fill, ...] = ()  # any fills the broker has on record


OrderStateProbe = Callable[[UUID], Awaitable["ResolvedOrderState | None"]]
"""Operator-injected callable. Returns ``None`` for UNKNOWN orders.

The probe is broker-specific; integration code (CLI / runbook) wires
it to the underlying broker's per-order-id lookup surface.
"""


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Outcome of a recovery pass.

    ``halted=True`` means the engine SHOULD NOT resume — operator
    must intervene. The ``halt_reason`` string is suitable for
    structured logging + runbook triage.
    """

    order_tracker: OrderTracker
    virtual_portfolio: VirtualPortfolio
    unresolved_orders: tuple[UUID, ...] = ()
    halted: bool = False
    halt_reason: str | None = None
    records_replayed: int = 0
    truncated_last_line: bool = False
    divergences: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class RecoveryCoordinator:
    """Reads journal, rebuilds state, resolves pending, cross-checks positions.

    Construct with the journal path + broker (for position queries)
    + optional order_state_probe (operator-injected per-order
    resolver). If the journal does not exist, ``replay()`` returns
    a clean RecoveryResult with empty state (first-launch scenario).
    """

    def __init__(
        self,
        journal_path: Path | str,
        broker: AsyncBrokerProtocol,
        *,
        order_state_probe: OrderStateProbe | None = None,
        tolerance: int = _DEFAULT_TOLERANCE,
    ) -> None:
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0; got {tolerance}")
        self._journal_path = Path(journal_path)
        self._broker = broker
        self._order_state_probe = order_state_probe
        self._tolerance = tolerance

    async def replay(self) -> RecoveryResult:
        """End-to-end recovery: journal → resolve pending → cross-check.

        Returns a RecoveryResult. Caller (operator runbook / engine
        bootstrap) inspects ``halted`` + ``unresolved_orders`` and
        decides whether to resume the engine.
        """
        order_tracker, vp, records, truncated, applied_fill_ids = self._replay_journal_to_state()

        unresolved: list[UUID] = []
        if self._order_state_probe is not None:
            vp, unresolved = await self._resolve_pending_orders(order_tracker, vp, applied_fill_ids)

        halted, reason, divergences = await self._cross_check_positions(vp)
        # If we had unresolved orders, halt regardless of position cross-check
        if unresolved and not halted:
            halted = True
            reason = f"unresolved orders after broker probe: {[str(u) for u in unresolved]}"

        return RecoveryResult(
            order_tracker=order_tracker,
            virtual_portfolio=vp,
            unresolved_orders=tuple(unresolved),
            halted=halted,
            halt_reason=reason,
            records_replayed=records,
            truncated_last_line=truncated,
            divergences=tuple(divergences),
        )

    def _replay_journal_to_state(
        self,
    ) -> tuple[OrderTracker, VirtualPortfolio, int, bool, set[UUID]]:
        """Read JSONL journal; rebuild OrderTracker + VirtualPortfolio.

        Returns (tracker, vp, n_records_replayed, truncated_last_line,
        applied_fill_ids). ``applied_fill_ids`` is the set of every
        ``fill_id`` already applied from the journal — threaded into
        ``_resolve_pending_orders`` so a broker probe that returns an
        order's FULL fill history does not re-apply fills the journal
        already recorded (which would Overfill the OrderTracker and
        double-count the VirtualPortfolio). If the journal does not
        exist, returns empty state with 0 records (first-launch
        scenario, no crash to recover from).
        """
        ledger = Ledger()
        tracker = OrderTracker(ledger=ledger)
        vp = VirtualPortfolio(cash=0.0, positions={})
        applied_fill_ids: set[UUID] = set()

        if not self._journal_path.exists():
            return tracker, vp, 0, False, applied_fill_ids

        lines = self._journal_path.read_text(encoding="utf-8").splitlines()
        records_replayed = 0
        truncated = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if i == len(lines) - 1:
                    # Crash mid-fsync — tolerate truncated last line.
                    logger.warning(
                        "recovery: truncated last line in journal %s (line %d): %s",
                        self._journal_path,
                        i + 1,
                        e,
                    )
                    truncated = True
                    break
                # Mid-file corruption — fail loudly.
                raise RecoveryError(
                    f"journal {self._journal_path} corrupted at line {i + 1}: {e}"
                ) from e
            vp = self._dispatch_record(record, tracker, vp, applied_fill_ids)
            records_replayed += 1
        return tracker, vp, records_replayed, truncated, applied_fill_ids

    def _dispatch_record(
        self,
        record: dict[str, Any],
        tracker: OrderTracker,
        vp: VirtualPortfolio,
        applied_fill_ids: set[UUID],
    ) -> VirtualPortfolio:
        """Dispatch one SafeBroker journal record to OrderTracker / VP.

        Reads the ACTUAL ``SafeBroker`` journal schema (REC-001 fix, S39):
        each record is ``{"ts_ns": int, "event_type": submit|fill|reject|
        cancel, **payload}`` (safe_broker.py). The pre-S39 code dispatched on
        ``record["type"]`` ∈ OrderSubmit/... with ``record["timestamp"]`` — a
        schema no writer emits — so every real record fell through to the skip
        branch and recovery rebuilt EMPTY state.
        """
        event_type = record.get("event_type")
        ts = str(record.get("ts_ns", ""))
        if event_type == "submit":
            tracker.submit(_order_from_record(record), ts)
        elif event_type == "fill":
            fill = _fill_from_record(record)
            tracker.on_fill(fill)
            vp = vp.apply(fill)
            applied_fill_ids.add(fill.fill_id)
        elif event_type == "cancel":
            # SafeBroker journals cancel with an ``ack`` flag; only transition
            # when the broker actually accepted the cancel and the order is
            # still open (a fill may have landed first).
            if record.get("ack", True):
                order_id = UUID(str(record["order_id"]))
                if order_id in tracker and not is_terminal(tracker.status(order_id)):
                    tracker.cancel(order_id, ts, reason="journal_cancel")
        elif event_type == "reject":
            # SafeBroker journals reject when the pre-trade RiskGate blocks the
            # order BEFORE submission — it never entered the tracker lifecycle,
            # so there is nothing to transition (tracker.reject would raise
            # OrderStateError on the unknown id). Counted, not applied.
            pass
        else:
            logger.warning("recovery: unknown event_type %r — skipping", event_type)
        return vp

    async def _resolve_pending_orders(
        self,
        tracker: OrderTracker,
        vp: VirtualPortfolio,
        applied_fill_ids: set[UUID],
    ) -> tuple[VirtualPortfolio, list[UUID]]:
        """Query the order_state_probe for each non-terminal order.

        Apply resolutions to tracker + vp. ``applied_fill_ids`` carries
        the fills already applied during journal replay so probe-returned
        fills are de-duplicated. Return (updated_vp, unresolved_order_ids).
        """
        assert self._order_state_probe is not None
        unresolved: list[UUID] = []
        for order_id in tuple(tracker.open_orders()):
            resolved = await self._order_state_probe(order_id)
            if resolved is None:
                unresolved.append(order_id)
                logger.warning(
                    "recovery: order %s unknown to broker — operator review needed",
                    order_id,
                )
                continue
            vp = self._apply_resolution(resolved, tracker, vp, applied_fill_ids)
        return vp, unresolved

    def _apply_fill_once(
        self,
        fill: Fill,
        tracker: OrderTracker,
        vp: VirtualPortfolio,
        applied_fill_ids: set[UUID],
    ) -> VirtualPortfolio:
        """Apply a probe-returned fill iff its ``fill_id`` is new.

        A real broker probe (e.g. ib_async ``trades().fills``) returns an
        order's FULL fill history, including partials the journal already
        recorded. Re-applying those would raise ``OrderStateError``
        ("Overfill") in ``OrderTracker.on_fill`` and double-count the
        VirtualPortfolio. Skipping by ``fill_id`` makes resolution
        idempotent w.r.t. the journal.
        """
        if fill.fill_id in applied_fill_ids:
            logger.info(
                "recovery: probe fill %s for order %s already applied from journal — skipping",
                fill.fill_id,
                fill.order_id,
            )
            return vp
        tracker.on_fill(fill)
        vp = vp.apply(fill)
        applied_fill_ids.add(fill.fill_id)
        return vp

    def _apply_resolution(
        self,
        resolved: ResolvedOrderState,
        tracker: OrderTracker,
        vp: VirtualPortfolio,
        applied_fill_ids: set[UUID],
    ) -> VirtualPortfolio:
        """Apply a single broker-probe resolution to the rebuilt state."""
        order_id = resolved.order_id
        if resolved.status == OrderStatus.FILLED:
            for fill in resolved.fills:
                vp = self._apply_fill_once(fill, tracker, vp, applied_fill_ids)
        elif resolved.status == OrderStatus.CANCELLED:
            # Apply any partial fills first, then cancel
            for fill in resolved.fills:
                vp = self._apply_fill_once(fill, tracker, vp, applied_fill_ids)
            current = tracker.status(order_id)
            if not is_terminal(current):
                tracker.cancel(order_id, "recovery", reason="broker_probe_resolved")
        elif resolved.status == OrderStatus.REJECTED:
            tracker.reject(order_id, "recovery", reason="broker_probe_resolved")
        elif resolved.status == OrderStatus.WORKING:
            # Leave non-terminal; broker still working the order
            logger.info(
                "recovery: order %s still WORKING at broker — carrying forward",
                order_id,
            )
        else:
            logger.warning(
                "recovery: unexpected resolution status %r for order %s",
                resolved.status,
                order_id,
            )
        return vp

    async def _cross_check_positions(
        self, vp: VirtualPortfolio
    ) -> tuple[bool, str | None, list[str]]:
        """Compare rebuilt VP vs broker.get_position per ticker.

        Returns (halted, halt_reason, list_of_divergence_strings).
        Within tolerance ⇒ (False, None, []).
        Exceeds tolerance on ANY ticker ⇒ halt with reason listing
        all divergent tickers.
        """
        divergences: list[str] = []
        for ticker, vp_pos in vp.positions.items():
            broker_pos = await self._broker.get_position(ticker)
            broker_qty = broker_pos.quantity if broker_pos is not None else 0
            diff = abs(vp_pos.quantity - broker_qty)
            if diff > self._tolerance:
                divergences.append(
                    f"{ticker}: vp_qty={vp_pos.quantity} broker_qty={broker_qty} diff={diff}"
                )
        if divergences:
            reason = (
                f"position divergence on {len(divergences)} ticker(s) "
                f"exceeds tolerance={self._tolerance}: " + "; ".join(divergences)
            )
            return True, reason, divergences
        return False, None, divergences


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RecoveryError(RuntimeError):
    """Raised on unrecoverable journal corruption mid-file."""


# ---------------------------------------------------------------------------
# Record → typed-object reconstruction
# ---------------------------------------------------------------------------


def _order_from_record(record: dict[str, Any]) -> Order:
    """Reconstruct an Order from an OrderSubmit journal record."""
    return Order(
        order_id=UUID(str(record["order_id"])),
        ticker=str(record["ticker"]),
        side=OrderSide(str(record["side"])),
        quantity=int(record["quantity"]),
        order_type=OrderType(str(record["order_type"])),
        limit_price=(
            float(record["limit_price"]) if record.get("limit_price") is not None else None
        ),
        # s73: restore stop/trail trigger fields so a resting STOP/STOP_LIMIT/TRAIL/
        # TRAIL_LIMIT order survives recovery (else Order.__post_init__ rejects it). All
        # use record.get(...) so pre-s73 journal records (which lack the keys) recover as
        # None — safe, since no stop-family order was ever journaled before s73.
        stop_price=(float(record["stop_price"]) if record.get("stop_price") is not None else None),
        trail_amount=(
            float(record["trail_amount"]) if record.get("trail_amount") is not None else None
        ),
        trail_percent=(
            float(record["trail_percent"]) if record.get("trail_percent") is not None else None
        ),
        limit_offset=(
            float(record["limit_offset"]) if record.get("limit_offset") is not None else None
        ),
        timestamp=str(record["timestamp"]) if record.get("timestamp") else None,
    )


def _fill_from_record(record: dict[str, Any]) -> Fill:
    """Reconstruct a Fill from an OrderFill journal record."""
    return Fill(
        fill_id=UUID(str(record["fill_id"])),
        order_id=UUID(str(record["order_id"])),
        ticker=str(record["ticker"]),
        signed_quantity=int(record["signed_quantity"]),
        price=float(record["price"]),
        commission=float(record["commission"]),
        timestamp=str(record["timestamp"]),
    )


__all__ = [
    "OrderStateProbe",
    "RecoveryCoordinator",
    "RecoveryError",
    "RecoveryResult",
    "ResolvedOrderState",
    "VirtualPortfolio",
]
