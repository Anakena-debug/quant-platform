"""SafeBroker — async broker decorator with pre-trade risk + JSONL journal (S35 D4).

Wraps any ``AsyncBrokerProtocol`` and adds two responsibilities the
plain broker adapters do not handle:

1. **Pre-trade risk** — delegates to the existing
   ``quantengine.risk.gate.RiskGate`` (AC3). The gate is the same one
   used by ``runtime.daily_cycle`` for batch; SafeBroker calls it
   per-order with a single-order ``Sequence`` and a per-call snapshot
   built from injected providers (S35 D5: the engine owns price + state;
   SafeBroker only peeks). Orders that fail any check are journaled
   with the rejection reason and ``submit_order`` returns an empty
   ``list[Fill]`` (matching the ``AbstractBroker`` CI-1 contract:
   zero-fill is the broker-decided-not-to-fill case).

2. **JSONL forensic journal** — every state-mutating call (submit,
   fill, reject, cancel) appends one JSON line to the journal with a
   nanosecond timestamp from the injected ``Clock``, then ``fsync`` so
   the line survives SIGKILL. Read-only queries (``get_position``,
   ``get_account_state``) are NOT journaled.

The SafeBroker itself satisfies ``AsyncBrokerProtocol`` so it nests
inside ``ThreadSafeBrokerWrapper`` exactly like the underlying
broker. quantcore-independence pattern: this module does not import
``quantcore``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from uuid import UUID

import numpy as np

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming.protocols import (
    AsyncBrokerProtocol,
    Clock,
)


PriceProvider = Callable[[], Mapping[str, float]]
"""Returns the current price map (ticker -> reference price).

The engine owns the underlying state and updates it on each ``TradeEvent``;
SafeBroker calls the provider once per ``submit_order`` to take a snapshot
for ``RiskGate.validate``. Empty mapping is legal (SafeBroker will reject
any order it sees, because ``known_ticker_check`` will not find a price)."""

StateProvider = Callable[[], PortfolioState]
"""Returns the engine's current ``PortfolioState`` (cash + positions).

The engine maintains a ``VirtualPortfolio``-style state by applying every
fill it receives; SafeBroker reads via this provider on each
``submit_order``."""


_INFLIGHT_SENTINEL_TS = "inflight-synthetic"
"""Timestamp on synthetic in-flight Fills (see SafeBroker._state_with_inflight).

These Fills are transient — folded into a copy of the validation state for the
RiskGate projection and then discarded — so the field is never journaled, read,
or compared. A labelled sentinel keeps Fill.timestamp's non-optional str
contract without implying a real wall-clock instant."""


class SafeBrokerJournalError(IOError):
    """Raised when the JSONL journal cannot be written.

    Treated as fatal at the engine level: an unjournaled fill is an
    audit-trail gap. Operator must address before the runtime can
    resume.
    """


@dataclass(frozen=True, slots=True)
class JournalRecord:
    """One line in the JSONL journal."""

    ts_ns: int
    event_type: str  # one of: submit | fill | reject | cancel
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"ts_ns": self.ts_ns, "event_type": self.event_type, **self.payload}


def _write_and_fsync(fh: IO[bytes], data: bytes) -> None:
    """Append bytes to ``fh`` and fsync."""
    fh.write(data)
    fh.flush()
    os.fsync(fh.fileno())


def _ns_to_isoz(ns: int) -> str:
    """Convert nanosecond epoch to ISO-8601 UTC string."""
    if ns <= 0:
        return "1970-01-01T00:00:00Z"
    seconds, _nanos = divmod(ns, 1_000_000_000)
    return (
        _dt.datetime.fromtimestamp(seconds, tz=_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class SafeBroker:
    """Async decorator over ``AsyncBrokerProtocol``: pre-trade risk + journal."""

    def __init__(
        self,
        broker: AsyncBrokerProtocol,
        risk_gate: RiskGate,
        journal_path: Path | str,
        *,
        price_provider: PriceProvider,
        state_provider: StateProvider,
        clock: Clock,
        on_fill: Callable[[Fill], None] | None = None,
    ) -> None:
        self._broker = broker
        self._risk_gate = risk_gate
        self._journal_path = Path(journal_path)
        self._price_provider = price_provider
        self._state_provider = state_provider
        self._clock = clock
        # REC-003 (S39): optional fill subscriber invoked AFTER each fill is
        # durably journaled, so a StreamingReconciler only ever sees fills
        # that survived fsync. Default None preserves prior behaviour.
        self._on_fill = on_fill
        # REC-004 (S39): kill-switch. When tripped, submit_order rejects every
        # order pre-submit (the broker is never called) until the process is
        # restarted. Set via trip_kill_switch() (signal handler / watchdog).
        self._killed = False
        self._kill_reason = ""
        self._journal_lock = asyncio.Lock()
        self._journal_file: IO[bytes] | None = None
        # In-flight exposure accumulator: orders this SafeBroker has accepted
        # (passed risk + journaled "submit") but whose fills have not yet been
        # applied to the state the ``state_provider`` reads. Each subsequent
        # ``submit_order`` validates the NEW order against a state that already
        # reflects these — so a burst of orders sharing the same pre-fill state
        # snapshot cannot collectively breach the cash / gross-leverage /
        # per-name caps that a per-order check would each individually pass.
        #
        # Under the CURRENT runtime (single-worker dispatch executor +
        # ThreadSafeBrokerWrapper's blocking sync submit, and fills applied to
        # state before submit_order returns) order submission is serialized and
        # this map is empty at every validation, so behaviour is unchanged.
        # The accumulator is forward-insurance: it makes the cap invariant hold
        # if dispatch ever becomes concurrent or fills become async-deferred.
        # Exact in-flight↔realized atomicity under TRUE concurrency would need a
        # lock around register→submit→apply→deregister; the serialized model
        # makes that unnecessary today (no await interleaves a sibling between
        # this order's broker return and its deregister).
        self._inflight: dict[UUID, Order] = {}

    def trip_kill_switch(self, reason: str = "manual") -> None:
        """Trip the kill-switch: all subsequent submits are rejected pre-broker.

        Idempotent and synchronous so a signal handler or the engine watchdog
        can call it from any context. Sticky for the process lifetime.
        """
        self._killed = True
        self._kill_reason = reason

    @property
    def killed(self) -> bool:
        """True once the kill-switch has been tripped. Sticky."""
        return self._killed

    def set_fill_subscriber(self, on_fill: Callable[[Fill], None] | None) -> None:
        """Attach (or clear) the post-journal fill subscriber after construction.

        Needed by the CLI bootstrap, where the SafeBroker, the engine, and the
        StreamingReconciler form a construction cycle (the reconciler's
        halt_callback needs the engine, the engine wraps the broker): build the
        SafeBroker first, then wire ``reconciler.on_fill`` once both exist.
        """
        self._on_fill = on_fill

    def _ensure_journal_open(self) -> None:
        """Open the journal file for append if not already open."""
        if self._journal_file is not None:
            return
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal_file = open(self._journal_path, "ab")

    async def _journal(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one JSONL record. fsyncs before returning."""
        record = JournalRecord(
            ts_ns=self._clock.now_ns(),
            event_type=event_type,
            payload=payload,
        )
        line = (json.dumps(record.to_dict(), default=str) + "\n").encode("utf-8")
        async with self._journal_lock:
            try:
                self._ensure_journal_open()
                assert self._journal_file is not None
                fh = self._journal_file
                await asyncio.to_thread(_write_and_fsync, fh, line)
            except OSError as e:
                raise SafeBrokerJournalError(
                    f"failed to write SafeBroker journal at {self._journal_path}: {e}"
                ) from e

    async def aclose(self) -> None:
        """Flush and close the journal file. Idempotent."""
        async with self._journal_lock:
            if self._journal_file is not None:
                await asyncio.to_thread(self._journal_file.close)
                self._journal_file = None

    def _state_with_inflight(
        self, state: PortfolioState, prices: Mapping[str, float]
    ) -> PortfolioState:
        """Return ``state`` with every in-flight order's exposure applied.

        Each in-flight order is folded in as a synthetic commission-free
        ``Fill`` at the current reference price via ``PortfolioState.apply``
        (the same reducer the live book uses), so the resulting state reflects
        positions + cash AS IF the in-flight orders had filled. Orders whose
        ticker is unpriced are skipped — they would be caught by the gate's
        ``known_ticker`` check anyway. Commission is omitted to match the
        gate's notional-only cash projection (see ``risk.gate._project_cash``).

        Fast path: no in-flight orders ⇒ return ``state`` unchanged (the
        common case under serialized dispatch — zero overhead).
        """
        if not self._inflight:
            return state
        adjusted = state
        for order in self._inflight.values():
            px = prices.get(order.ticker)
            if px is None or px <= 0:
                continue
            adjusted = adjusted.apply(
                Fill(
                    fill_id=order.order_id,  # synthetic; never journaled/persisted
                    order_id=order.order_id,
                    ticker=order.ticker,
                    signed_quantity=order.signed_quantity,
                    price=float(px),
                    commission=0.0,
                    # Fill.timestamp is non-optional str; this synthetic fill is
                    # discarded after .apply (cash/position math only), so the
                    # value is never read — use a labelled sentinel, not None.
                    timestamp=_INFLIGHT_SENTINEL_TS,
                )
            )
        return adjusted

    def _build_market_snapshot(self, prices: Mapping[str, float]) -> MarketSnapshot:
        """Construct a ``MarketSnapshot`` from the current price map."""
        good_pairs = [(t, float(p)) for t, p in prices.items() if p > 0]
        if not good_pairs:
            return MarketSnapshot(
                timestamp=_ns_to_isoz(self._clock.now_ns()),
                tickers=(),
                prices=np.array([], dtype=np.float64),
            )
        tickers = tuple(t for t, _ in good_pairs)
        arr = np.array([p for _, p in good_pairs], dtype=np.float64)
        return MarketSnapshot(
            timestamp=_ns_to_isoz(self._clock.now_ns()),
            tickers=tickers,
            prices=arr,
        )

    async def submit_order(self, order: Order) -> list[Fill]:
        # REC-004: kill-switch short-circuits BEFORE risk/broker — journal the
        # rejection for the audit trail and return zero fills.
        if self._killed:
            await self._journal(
                "reject",
                {
                    "order_id": str(order.order_id),
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value,
                    "check": "kill_switch",
                    "reason": self._kill_reason or "kill_switch_tripped",
                },
            )
            return []

        prices = self._price_provider()
        state = self._state_provider()
        # Fold in any accepted-but-unfilled siblings so this order is validated
        # against the exposure they already represent, not a stale snapshot.
        state = self._state_with_inflight(state, prices)
        market = self._build_market_snapshot(prices)
        accepted, rejected = self._risk_gate.validate([order], state, market)

        if rejected:
            rj = rejected[0]
            await self._journal(
                "reject",
                {
                    "order_id": str(order.order_id),
                    "ticker": order.ticker,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value,
                    "check": rj.check,
                    "reason": rj.reason,
                },
            )
            return []

        await self._journal(
            "submit",
            {
                "order_id": str(order.order_id),
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "order_type": order.order_type.value,
                # REC-002 (S39): include limit_price + an order timestamp so
                # RecoveryCoordinator can reconstruct in-flight LIMIT orders on
                # restart (market orders recovered without these; limit orders
                # need the price). limit_price is null for market orders.
                "limit_price": order.limit_price,
                # s73: stop/trail trigger fields so a resting STOP/STOP_LIMIT/TRAIL/
                # TRAIL_LIMIT in-flight order reconstructs on restart (null otherwise).
                "stop_price": order.stop_price,
                "trail_amount": order.trail_amount,
                "trail_percent": order.trail_percent,
                "limit_offset": order.limit_offset,
                "timestamp": _ns_to_isoz(self._clock.now_ns()),
            },
        )
        # Register as in-flight BEFORE awaiting the broker so a sibling that
        # validates while this order is working counts its exposure. Removed in
        # ``finally`` once its fills are applied to the state source (via
        # ``_on_fill`` below / the broker's own bookkeeping), so it is never
        # double-counted (in-flight AND realized) once control returns here.
        accepted_order = accepted[0]
        self._inflight[accepted_order.order_id] = accepted_order
        try:
            fills = await self._broker.submit_order(accepted_order)
            for fill in fills:
                await self._journal(
                    "fill",
                    {
                        "fill_id": str(fill.fill_id),
                        "order_id": str(fill.order_id),
                        "ticker": fill.ticker,
                        "signed_quantity": fill.signed_quantity,
                        "price": fill.price,
                        "commission": fill.commission,
                        "timestamp": fill.timestamp,
                    },
                )
                # REC-003: notify the reconciler ONLY after the fill is durable.
                if self._on_fill is not None:
                    self._on_fill(fill)
        finally:
            self._inflight.pop(accepted_order.order_id, None)
        return fills

    async def cancel_order(self, order_id: UUID) -> bool:
        ack = await self._broker.cancel_order(order_id)
        await self._journal(
            "cancel",
            {"order_id": str(order_id), "ack": ack},
        )
        return ack

    async def get_position(self, ticker: str) -> Position | None:
        return await self._broker.get_position(ticker)

    async def get_account_state(self) -> PortfolioState:
        return await self._broker.get_account_state()


__all__ = [
    "JournalRecord",
    "PriceProvider",
    "SafeBroker",
    "SafeBrokerJournalError",
    "StateProvider",
]
