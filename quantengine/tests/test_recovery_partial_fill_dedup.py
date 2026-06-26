"""Regression: RecoveryCoordinator must not re-apply journal fills the
broker probe also returns (s47 audit fix).

A real ``order_state_probe`` (e.g. ib_async ``trades().fills``) returns an
order's FULL fill history. When a partial fill was already journaled and
replayed, re-applying it during pending-order resolution:

  - raises ``OrderStateError`` ("Overfill") inside ``OrderTracker.on_fill``
    once cumulative exceeds the order target, aborting recovery entirely; and
  - double-counts the VirtualPortfolio position.

The fix threads the set of journal-applied ``fill_id``s into resolution and
skips any probe fill already applied. Each test below FAILS on the pre-fix
code (recovery raises / position doubles) and PASSES on the fix.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from quantengine.contracts.orders import Fill, OrderStatus
from quantengine.portfolio.state import Position
from quantengine.runtime.streaming.recovery import (
    RecoveryCoordinator,
    ResolvedOrderState,
)

_TS0 = 1_700_000_000_000_000_000


def _write_journal(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _submit_record(order_id: UUID, ticker: str = "AAPL", qty: int = 10) -> dict[str, object]:
    return {
        "ts_ns": _TS0,
        "event_type": "submit",
        "order_id": str(order_id),
        "ticker": ticker,
        "side": "BUY" if qty > 0 else "SELL",
        "quantity": abs(qty),
        "order_type": "MARKET",
    }


def _fill_record(
    order_id: UUID,
    fill_id: UUID,
    ticker: str = "AAPL",
    signed_qty: int = 5,
    price: float = 100.0,
) -> dict[str, object]:
    """A SafeBroker ``fill`` record with an EXPLICIT fill_id so the journal
    fill and the probe-returned fill can share identity."""
    return {
        "ts_ns": _TS0 + 2,
        "event_type": "fill",
        "fill_id": str(fill_id),
        "order_id": str(order_id),
        "ticker": ticker,
        "signed_quantity": signed_qty,
        "price": price,
        "commission": 0.0,
        "timestamp": "2026-05-31T15:00:02Z",
    }


def _fill(order_id: UUID, fill_id: UUID, signed_qty: int, price: float = 100.0) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        ticker="AAPL",
        signed_quantity=signed_qty,
        price=price,
        commission=0.0,
        timestamp="2026-05-31T15:01:00Z",
    )


def test_probe_full_history_does_not_double_apply_journal_partial(tmp_path: Path) -> None:
    """Journal has partial F1 (5/10); probe returns FILLED history (F1, F2).
    F1 must be skipped (already applied) — no Overfill, position == 10."""

    async def run() -> None:
        journal = tmp_path / "j.jsonl"
        oid, f1, f2 = uuid4(), uuid4(), uuid4()
        _write_journal(
            journal,
            [_submit_record(oid, "AAPL", 10), _fill_record(oid, f1, signed_qty=5)],
        )
        probe = AsyncMock()
        probe.return_value = ResolvedOrderState(
            order_id=oid,
            status=OrderStatus.FILLED,
            fills=(_fill(oid, f1, 5), _fill(oid, f2, 5)),  # FULL history, F1 repeated
        )
        broker = AsyncMock()
        broker.get_position.return_value = Position("AAPL", 10, 100.0)
        coord = RecoveryCoordinator(journal_path=journal, broker=broker, order_state_probe=probe)

        result = await coord.replay()  # pre-fix: raises OrderStateError("Overfill")

        assert result.halted is False
        assert result.order_tracker.status(oid) == OrderStatus.FILLED
        assert result.virtual_portfolio.positions["AAPL"].quantity == 10  # not 15

    asyncio.run(run())


def test_probe_returns_only_new_fill_still_completes(tmp_path: Path) -> None:
    """Probe that returns ONLY the post-journal fill (F2) still resolves to
    FILLED with the correct position (dedup is a no-op here)."""

    async def run() -> None:
        journal = tmp_path / "j.jsonl"
        oid, f1, f2 = uuid4(), uuid4(), uuid4()
        _write_journal(
            journal,
            [_submit_record(oid, "AAPL", 10), _fill_record(oid, f1, signed_qty=5)],
        )
        probe = AsyncMock()
        probe.return_value = ResolvedOrderState(
            order_id=oid, status=OrderStatus.FILLED, fills=(_fill(oid, f2, 5),)
        )
        broker = AsyncMock()
        broker.get_position.return_value = Position("AAPL", 10, 100.0)
        coord = RecoveryCoordinator(journal_path=journal, broker=broker, order_state_probe=probe)

        result = await coord.replay()

        assert result.order_tracker.status(oid) == OrderStatus.FILLED
        assert result.virtual_portfolio.positions["AAPL"].quantity == 10

    asyncio.run(run())


def test_cancelled_with_journal_partial_does_not_double_count(tmp_path: Path) -> None:
    """Journal has partial F1 (5/10); probe reports CANCELLED with the same
    F1 in its history. F1 skipped → position stays 5, order CANCELLED."""

    async def run() -> None:
        journal = tmp_path / "j.jsonl"
        oid, f1 = uuid4(), uuid4()
        _write_journal(
            journal,
            [_submit_record(oid, "AAPL", 10), _fill_record(oid, f1, signed_qty=5)],
        )
        probe = AsyncMock()
        probe.return_value = ResolvedOrderState(
            order_id=oid, status=OrderStatus.CANCELLED, fills=(_fill(oid, f1, 5),)
        )
        broker = AsyncMock()
        broker.get_position.return_value = Position("AAPL", 5, 100.0)
        coord = RecoveryCoordinator(journal_path=journal, broker=broker, order_state_probe=probe)

        result = await coord.replay()

        assert result.order_tracker.status(oid) == OrderStatus.CANCELLED
        assert result.virtual_portfolio.positions["AAPL"].quantity == 5  # not 10

    asyncio.run(run())
