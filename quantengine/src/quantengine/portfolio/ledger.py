"""Append-only event ledger.

Every state-changing event (Order submitted, Fill received, cash sweep, etc.)
is written here. The audit journal (`audit/journal.py`) wraps this with
hash-chaining for tamper evidence.

For Phase 1 this is an in-memory list; Phase 2 persists to DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Union

from quantengine.contracts.orders import Fill, Order

EventKind = Literal[
    "ORDER_SUBMITTED",
    "ORDER_ACKED",  # broker accepted (SUBMITTED → WORKING)
    "ORDER_WORKING",  # alias for ACKED when explicit live-state transitions matter
    "ORDER_FILLED",  # one fill slice (partial or terminal — see OrderTracker)
    "ORDER_CANCELLED",
    "ORDER_REJECTED",  # broker/engine refused the order
    "CORP_ACTION",  # split / dividend / spin-off applied to state
    "CASH_ADJ",  # manual cash adjustment (deposit, fee, interest)
    "RECONCILE",  # session-boundary broker-vs-internal state diff
]


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    seq: int
    timestamp: str
    kind: EventKind
    payload: Union[Order, Fill, dict]

    def summary(self) -> str:
        if isinstance(self.payload, Order):
            return f"{self.kind} {self.payload.ticker} {self.payload.side.value} {self.payload.quantity}"
        if isinstance(self.payload, Fill):
            return (
                f"{self.kind} {self.payload.ticker} "
                f"{self.payload.signed_quantity:+d}@{self.payload.price:.4f}"
            )
        return f"{self.kind} {self.payload}"


@dataclass
class Ledger:
    """Append-only log. Mutating operations return the new sequence number."""

    _events: list[LedgerEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[LedgerEvent]:
        return iter(self._events)

    def append(self, timestamp: str, kind: EventKind, payload: Union[Order, Fill, dict]) -> int:
        seq = len(self._events)
        self._events.append(LedgerEvent(seq=seq, timestamp=timestamp, kind=kind, payload=payload))
        return seq

    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    def fills(self) -> tuple[Fill, ...]:
        return tuple(
            e.payload
            for e in self._events
            if e.kind == "ORDER_FILLED" and isinstance(e.payload, Fill)
        )

    def orders(self) -> tuple[Order, ...]:
        return tuple(
            e.payload
            for e in self._events
            if e.kind == "ORDER_SUBMITTED" and isinstance(e.payload, Order)
        )
