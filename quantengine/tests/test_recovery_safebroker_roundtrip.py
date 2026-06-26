"""REC-001 end-to-end regression (S39).

Journals THROUGH the real ``SafeBroker`` (not a hand-fabricated dict) and
replays THROUGH ``RecoveryCoordinator``. This is the pin that the pre-S39
recovery code could not satisfy: it dispatched on ``record["type"]`` ∈
OrderSubmit/... while SafeBroker writes ``event_type`` ∈ submit/fill/...,
so every real record was skipped and recovery rebuilt an EMPTY
VirtualPortfolio. On the fixed code the journal replays into the correct
non-empty state.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from quantengine.contracts.orders import Fill, Order, OrderSide, OrderStatus, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming.protocols import EventClock
from quantengine.runtime.streaming.recovery import RecoveryCoordinator, _order_from_record
from quantengine.runtime.streaming.safe_broker import SafeBroker


class _FillingBroker:
    """Async broker that fully fills every order at a fixed price."""

    def __init__(self, price: float = 150.0) -> None:
        self._price = price

    async def submit_order(self, order: Order) -> list[Fill]:
        return [
            Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=self._price,
                commission=0.0,
                timestamp="2026-05-21T16:00:00Z",
            )
        ]

    async def cancel_order(self, order_id: UUID) -> bool:
        return True

    async def get_position(self, ticker: str) -> Position | None:
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=1_000_000.0, positions={})


def _safe_broker(journal_path: Path) -> SafeBroker:
    clock = EventClock()
    clock.tick(1_700_000_000_000_000_000)
    gate = RiskGate.default_us_equities(
        max_order_notional=200_000.0,
        max_gross_leverage=2.0,
        max_position_weight=0.25,
        min_cash=0.0,
    )
    return SafeBroker(
        _FillingBroker(),
        gate,
        journal_path,
        price_provider=lambda: {"AAPL": 150.0, "MSFT": 300.0},
        state_provider=lambda: PortfolioState(cash=1_000_000.0, positions={}),
        clock=clock,
    )


def _market_order(ticker: str, qty: int) -> Order:
    return Order(
        order_id=uuid4(),
        ticker=ticker,
        side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
        quantity=abs(qty),
        order_type=OrderType.MARKET,
    )


def test_safebroker_journal_replays_into_nonempty_state(tmp_path: Path) -> None:
    """The journal SafeBroker ACTUALLY writes must rebuild correct, non-empty
    state (fails on the pre-S39 schema mismatch, which rebuilt empty)."""
    journal = tmp_path / "safe_broker.jsonl"

    async def run() -> None:
        sb = _safe_broker(journal)
        o_aapl = _market_order("AAPL", 10)
        o_msft = _market_order("MSFT", 5)
        await sb.submit_order(o_aapl)
        await sb.submit_order(o_msft)
        await sb.aclose()

        # The journal is the real SafeBroker schema (event_type, not type).
        recs = [json.loads(line) for line in journal.read_text().splitlines() if line]
        assert {r["event_type"] for r in recs} == {"submit", "fill"}
        assert all("ts_ns" in r for r in recs)

        # Recovery cross-check broker reports positions matching the fills.
        broker = AsyncMock()

        def _pos(ticker: str) -> Position | None:
            return {
                "AAPL": Position("AAPL", 10, 150.0),
                "MSFT": Position("MSFT", 5, 300.0),
            }.get(ticker)

        broker.get_position.side_effect = _pos

        coord = RecoveryCoordinator(journal_path=journal, broker=broker, tolerance=1)
        result = await coord.replay()

        # The REC-001 fix: state rebuilt NON-empty, clean cross-check.
        assert result.records_replayed == 4  # 2 submit + 2 fill
        assert result.halted is False
        assert result.virtual_portfolio.positions["AAPL"].quantity == 10
        assert result.virtual_portfolio.positions["MSFT"].quantity == 5
        assert result.order_tracker.status(o_aapl.order_id) == OrderStatus.FILLED
        assert result.order_tracker.status(o_msft.order_id) == OrderStatus.FILLED

    asyncio.run(run())


def test_pre_fix_schema_would_rebuild_empty(tmp_path: Path) -> None:
    """Documents the bug: the OLD fabricated schema (record['type']) skips on
    the fixed dispatcher (which keys on event_type), proving the two schemas
    are genuinely different — i.e. the pre-S39 code, keyed on 'type', saw the
    SafeBroker journal as all-unknown and rebuilt empty."""
    journal = tmp_path / "old_schema.jsonl"
    oid = uuid4()
    old_records = [
        {
            "type": "OrderSubmit",
            "order_id": str(oid),
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "order_type": "MARKET",
            "timestamp": "2026-05-22T15:00:00Z",
        },
        {
            "type": "OrderFill",
            "fill_id": str(uuid4()),
            "order_id": str(oid),
            "ticker": "AAPL",
            "signed_quantity": 10,
            "price": 150.0,
            "commission": 0.0,
            "timestamp": "2026-05-22T15:00:02Z",
        },
    ]
    journal.write_text("\n".join(json.dumps(r) for r in old_records) + "\n", encoding="utf-8")

    async def run() -> None:
        broker = AsyncMock()
        broker.get_position.return_value = None
        coord = RecoveryCoordinator(journal_path=journal, broker=broker)
        result = await coord.replay()
        # Records are read but, lacking event_type, dispatch to no-op → empty.
        assert result.records_replayed == 2
        assert len(result.order_tracker) == 0
        assert dict(result.virtual_portfolio.positions) == {}

    asyncio.run(run())


def test_limit_order_submit_journals_limit_price(tmp_path: Path) -> None:
    """REC-002 (S39): the submit record carries limit_price + timestamp so an
    in-flight LIMIT order can be reconstructed on restart."""
    journal = tmp_path / "limit.jsonl"

    async def run() -> None:
        sb = _safe_broker(journal)
        order = Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=149.0,
        )
        await sb.submit_order(order)
        await sb.aclose()
        recs = [json.loads(line) for line in journal.read_text().splitlines() if line]
        submit = next(r for r in recs if r["event_type"] == "submit")
        assert submit["limit_price"] == 149.0
        assert submit.get("timestamp")

    asyncio.run(run())


def test_stop_family_submit_journals_trigger_fields(tmp_path: Path) -> None:
    """s73 (REC-002 extension): the submit record carries stop_price + trail fields
    so an in-flight STOP/TRAIL order reconstructs on restart instead of being rejected
    by Order.__post_init__ with the trigger fields None."""
    journal = tmp_path / "stop_trail.jsonl"

    async def run() -> None:
        sb = _safe_broker(journal)
        stop = Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.STOP,
            stop_price=151.0,
        )
        trail = Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.TRAIL,
            trail_percent=2.5,
        )
        await sb.submit_order(stop)
        await sb.submit_order(trail)
        await sb.aclose()

        recs = [json.loads(line) for line in journal.read_text().splitlines() if line]
        by_type = {r["order_type"]: r for r in recs if r.get("event_type") == "submit"}
        assert by_type["STOP"]["stop_price"] == 151.0 and by_type["STOP"]["trail_percent"] is None
        assert by_type["TRAIL"]["trail_percent"] == 2.5 and by_type["TRAIL"]["stop_price"] is None
        # the recovery reader reconstructs both without raising (the crash-fix path)
        assert _order_from_record(by_type["STOP"]).stop_price == 151.0
        assert _order_from_record(by_type["TRAIL"]).trail_percent == 2.5

    asyncio.run(run())
