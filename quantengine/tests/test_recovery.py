"""Tests for RecoveryCoordinator (S37 PR3; schema updated S39 REC-001).

S39 NOTE: these tests previously fabricated a journal schema
(``{"type": "OrderSubmit", ..., "timestamp": ...}`` + a separate
``OrderAck`` record) that NO production writer emits. The only
broker-event journal is ``SafeBroker``'s — ``{"ts_ns": int,
"event_type": submit|fill|reject|cancel, **payload}`` — and it emits no
ack (the OrderTracker FSM permits ``SUBMITTED → FILLED`` for sync
brokers). The helpers below now produce the REAL schema; see
``test_recovery_safebroker_roundtrip.py`` for the end-to-end pin that
journals THROUGH SafeBroker.

Coverage:

- ``TestEmptyJournal``: nonexistent + empty journal yield clean state.
- ``TestBasicReplay``: submit → fill rebuilds OrderTracker + VirtualPortfolio.
- ``TestTruncatedLastLine``: crash mid-fsync ⇒ last-line JSON parse
  failure tolerated; mid-file corruption raises ``RecoveryError``.
- ``TestPendingOrderResolution``: probe returns Filled / Cancelled /
  Rejected / Working / None (Unknown); each applied correctly;
  Unknown adds to unresolved_orders.
- ``TestCrossCheckPositions``: within tolerance → not halted;
  exceeds tolerance → halted with reason naming the divergent tickers.
- ``TestReplayOrchestrator``: end-to-end replay() integrates journal
  replay + pending resolution + cross-check.
- ``TestUnresolvedHalts``: unresolved order ids halt the engine even
  when position cross-check passes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from quantengine.contracts.orders import Fill, OrderStatus
from quantengine.portfolio.state import Position
from quantengine.runtime.streaming.recovery import (
    RecoveryCoordinator,
    RecoveryError,
    ResolvedOrderState,
)

_TS0 = 1_700_000_000_000_000_000  # nanosecond epoch base for deterministic ts_ns


def _write_journal(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSON records as a JSONL file (one JSON dict per line)."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _submit_record(order_id: UUID, ticker: str = "AAPL", qty: int = 10) -> dict[str, object]:
    """The real SafeBroker ``submit`` record (event_type + ts_ns; no ack)."""
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
    order_id: UUID, ticker: str = "AAPL", signed_qty: int = 10, price: float = 100.0
) -> dict[str, object]:
    """The real SafeBroker ``fill`` record."""
    return {
        "ts_ns": _TS0 + 2,
        "event_type": "fill",
        "fill_id": str(uuid4()),
        "order_id": str(order_id),
        "ticker": ticker,
        "signed_quantity": signed_qty,
        "price": price,
        "commission": 0.0,
        "timestamp": "2026-05-22T15:00:02Z",
    }


def _cancel_record(order_id: UUID, ack: bool = True) -> dict[str, object]:
    """The real SafeBroker ``cancel`` record (carries an ``ack`` flag)."""
    return {
        "ts_ns": _TS0 + 3,
        "event_type": "cancel",
        "order_id": str(order_id),
        "ack": ack,
    }


# ---------------------------------------------------------------------------
# Empty journal
# ---------------------------------------------------------------------------


class TestEmptyJournal:
    def test_nonexistent_journal_returns_clean_state(self, tmp_path: Path) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=tmp_path / "missing.jsonl", broker=broker)
            result = await coord.replay()
            assert result.records_replayed == 0
            assert result.halted is False
            assert len(result.order_tracker) == 0
            assert dict(result.virtual_portfolio.positions) == {}

        asyncio.run(run())

    def test_empty_journal_file_is_clean(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "empty.jsonl"
            journal.write_text("", encoding="utf-8")
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.records_replayed == 0
            assert result.halted is False

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Basic replay
# ---------------------------------------------------------------------------


class TestBasicReplay:
    def test_submit_fill_round_trip(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(
                journal,
                [
                    _submit_record(oid, "AAPL", 10),
                    _fill_record(oid, "AAPL", 10, 100.0),
                ],
            )
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.records_replayed == 2
            assert result.halted is False
            # OrderTracker has the order in FILLED state (SUBMITTED → FILLED).
            assert oid in result.order_tracker
            assert result.order_tracker.status(oid) == OrderStatus.FILLED
            # VP has the AAPL position
            pos = result.virtual_portfolio.positions.get("AAPL")
            assert pos is not None
            assert pos.quantity == 10
            assert pos.avg_cost == 100.0

        asyncio.run(run())

    def test_cancel_record(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid), _cancel_record(oid, ack=True)])
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.order_tracker.status(oid) == OrderStatus.CANCELLED

        asyncio.run(run())

    def test_cancel_not_acked_leaves_order_open(self, tmp_path: Path) -> None:
        """A cancel the broker did NOT accept (ack=False) must not transition."""

        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid), _cancel_record(oid, ack=False)])
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.order_tracker.status(oid) == OrderStatus.SUBMITTED

        asyncio.run(run())

    def test_reject_record_not_applied(self, tmp_path: Path) -> None:
        """A pre-submit RiskGate reject never entered the tracker; recovery
        counts it but does not try to transition an unknown order."""

        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            reject = {
                "ts_ns": _TS0,
                "event_type": "reject",
                "order_id": str(oid),
                "ticker": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "check": "buying_power",
                "reason": "insufficient_cash",
            }
            _write_journal(journal, [reject])
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.records_replayed == 1
            assert len(result.order_tracker) == 0  # never tracked
            assert result.halted is False

        asyncio.run(run())

    def test_unknown_event_type_warned_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            _write_journal(journal, [{"ts_ns": _TS0, "event_type": "future_kind", "data": "abc"}])
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            with caplog.at_level("WARNING"):
                result = await coord.replay()
            assert result.records_replayed == 1
            assert any("unknown event_type" in r.message for r in caplog.records)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Truncated last line
# ---------------------------------------------------------------------------


class TestTruncatedLastLine:
    def test_truncated_last_line_tolerated(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            good = json.dumps(_submit_record(oid))
            # Truncated record: open-brace + partial content, no closing brace
            truncated = '{"event_type": "fill", "order_'
            journal.write_text(good + "\n" + truncated, encoding="utf-8")
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            result = await coord.replay()
            assert result.truncated_last_line is True
            assert result.records_replayed == 1
            # First record applied; truncated second skipped
            assert oid in result.order_tracker

        asyncio.run(run())

    def test_mid_file_corruption_raises(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            good = json.dumps(_submit_record(oid))
            corrupt = '{"event_type": "fill", garbage}'
            tail = json.dumps(_fill_record(oid))
            journal.write_text(good + "\n" + corrupt + "\n" + tail + "\n", encoding="utf-8")
            broker = AsyncMock()
            coord = RecoveryCoordinator(journal_path=journal, broker=broker)
            with pytest.raises(RecoveryError, match="corrupted"):
                await coord.replay()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Pending order resolution
# ---------------------------------------------------------------------------


class TestPendingOrderResolution:
    def test_probe_resolves_to_filled(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            # Pending fill that the broker reports
            broker_fill = Fill(
                fill_id=uuid4(),
                order_id=oid,
                ticker="AAPL",
                signed_quantity=10,
                price=101.0,
                commission=0.0,
                timestamp="2026-05-22T15:01:00Z",
            )
            probe = AsyncMock()
            probe.return_value = ResolvedOrderState(
                order_id=oid, status=OrderStatus.FILLED, fills=(broker_fill,)
            )
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 101.0)
            coord = RecoveryCoordinator(
                journal_path=journal, broker=broker, order_state_probe=probe
            )
            result = await coord.replay()
            assert result.halted is False
            assert result.order_tracker.status(oid) == OrderStatus.FILLED
            assert result.virtual_portfolio.positions["AAPL"].quantity == 10
            assert result.unresolved_orders == ()

        asyncio.run(run())

    def test_probe_resolves_to_cancelled(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            probe = AsyncMock()
            probe.return_value = ResolvedOrderState(order_id=oid, status=OrderStatus.CANCELLED)
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(
                journal_path=journal, broker=broker, order_state_probe=probe
            )
            result = await coord.replay()
            assert result.halted is False
            assert result.order_tracker.status(oid) == OrderStatus.CANCELLED

        asyncio.run(run())

    def test_probe_returns_none_means_unknown(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            probe = AsyncMock()
            probe.return_value = None  # broker doesn't know this order
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(
                journal_path=journal, broker=broker, order_state_probe=probe
            )
            result = await coord.replay()
            assert oid in result.unresolved_orders
            assert result.halted is True
            assert result.halt_reason is not None
            assert "unresolved orders" in result.halt_reason

        asyncio.run(run())

    def test_probe_working_keeps_order_non_terminal(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            probe = AsyncMock()
            probe.return_value = ResolvedOrderState(order_id=oid, status=OrderStatus.WORKING)
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(
                journal_path=journal, broker=broker, order_state_probe=probe
            )
            result = await coord.replay()
            # Carried forward as non-terminal; the WORKING resolution does not
            # transition SUBMITTED → WORKING (recovery leaves it open).
            from quantengine.execution.order_state import is_terminal

            assert not is_terminal(result.order_tracker.status(oid))
            assert oid not in result.unresolved_orders
            assert result.halted is False

        asyncio.run(run())

    def test_no_probe_skips_resolution(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            broker = AsyncMock()
            broker.get_position.return_value = None
            coord = RecoveryCoordinator(journal_path=journal, broker=broker, order_state_probe=None)
            result = await coord.replay()
            # Without a probe, the submitted order carries forward open.
            assert result.unresolved_orders == ()
            assert result.halted is False
            assert result.order_tracker.status(oid) == OrderStatus.SUBMITTED

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Cross-check positions
# ---------------------------------------------------------------------------


class TestCrossCheckPositions:
    def test_within_tolerance_not_halted(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(
                journal,
                [_submit_record(oid, "AAPL", 10), _fill_record(oid, "AAPL", 10, 100.0)],
            )
            broker = AsyncMock()
            # broker reports 11 — diff=1, within tolerance=1
            broker.get_position.return_value = Position("AAPL", 11, 100.0)
            coord = RecoveryCoordinator(journal_path=journal, broker=broker, tolerance=1)
            result = await coord.replay()
            assert result.halted is False
            assert result.divergences == ()

        asyncio.run(run())

    def test_exceeds_tolerance_halts(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(
                journal,
                [_submit_record(oid, "AAPL", 10), _fill_record(oid, "AAPL", 10, 100.0)],
            )
            broker = AsyncMock()
            # broker reports 5 — diff=5, exceeds tolerance=1
            broker.get_position.return_value = Position("AAPL", 5, 100.0)
            coord = RecoveryCoordinator(journal_path=journal, broker=broker, tolerance=1)
            result = await coord.replay()
            assert result.halted is True
            assert result.halt_reason is not None
            assert "AAPL" in result.halt_reason
            assert "diff=5" in result.halt_reason

        asyncio.run(run())


# ---------------------------------------------------------------------------
# End-to-end orchestrator integration
# ---------------------------------------------------------------------------


class TestReplayOrchestrator:
    def test_replay_integrates_all_phases(self, tmp_path: Path) -> None:
        """Full path: journal → resolve pending → cross-check positions."""

        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid_filled = uuid4()
            oid_pending = uuid4()
            _write_journal(
                journal,
                [
                    _submit_record(oid_filled, "AAPL", 10),
                    _fill_record(oid_filled, "AAPL", 10, 100.0),
                    _submit_record(oid_pending, "MSFT", 5),
                ],
            )
            # Probe resolves the pending order to Filled
            broker_fill = Fill(
                fill_id=uuid4(),
                order_id=oid_pending,
                ticker="MSFT",
                signed_quantity=5,
                price=300.0,
                commission=0.0,
                timestamp="2026-05-22T15:02:00Z",
            )
            probe = AsyncMock(
                return_value=ResolvedOrderState(
                    order_id=oid_pending, status=OrderStatus.FILLED, fills=(broker_fill,)
                )
            )
            broker = AsyncMock()

            def _broker_position(ticker: str) -> Position | None:
                if ticker == "AAPL":
                    return Position("AAPL", 10, 100.0)
                if ticker == "MSFT":
                    return Position("MSFT", 5, 300.0)
                return None

            broker.get_position.side_effect = _broker_position
            coord = RecoveryCoordinator(
                journal_path=journal,
                broker=broker,
                order_state_probe=probe,
                tolerance=1,
            )
            result = await coord.replay()
            assert result.halted is False
            assert result.unresolved_orders == ()
            assert result.virtual_portfolio.positions["AAPL"].quantity == 10
            assert result.virtual_portfolio.positions["MSFT"].quantity == 5

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Unresolved orders halt even if positions cross-check passes
# ---------------------------------------------------------------------------


class TestUnresolvedHalts:
    def test_unresolved_orders_halt_engine(self, tmp_path: Path) -> None:
        async def run() -> None:
            journal = tmp_path / "j.jsonl"
            oid = uuid4()
            _write_journal(journal, [_submit_record(oid, "AAPL", 10)])
            probe = AsyncMock(return_value=None)  # Unknown
            broker = AsyncMock()
            broker.get_position.return_value = None  # No position
            coord = RecoveryCoordinator(
                journal_path=journal, broker=broker, order_state_probe=probe
            )
            result = await coord.replay()
            # Position cross-check would PASS (no position diff), but the
            # unresolved order itself halts.
            assert result.halted is True
            assert oid in result.unresolved_orders
            assert "unresolved" in (result.halt_reason or "")

        asyncio.run(run())
