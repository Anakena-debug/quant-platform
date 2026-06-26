"""Tests for quantengine.runtime.streaming.safe_broker.SafeBroker.

Pins the SafeBroker contract (S35 D4):

- Pre-trade ``RiskGate`` integration (AC3 import grep is structural;
  these tests pin the *behavior*: rejected orders never reach the
  underlying broker, and the rejection lands in the journal).
- JSONL journal records submit + fill + reject + cancel; read-only
  queries (get_position, get_account_state) are NOT journaled.
- fsync per write (smoke-tested via successful round-trip; the actual
  fsync syscall is mock-coverable but flakey across OS).

quantcore-independence: no quantcore imports.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from quantengine.contracts.orders import (
    Fill,
    Order,
    OrderSide,
    OrderType,
)
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming.protocols import (
    AsyncBrokerProtocol,
    EventClock,
)
from quantengine.runtime.streaming.safe_broker import (
    JournalRecord,
    SafeBroker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _MockBroker:
    """Records every call; returns one fill per submit."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.cancelled: list[UUID] = []
        self.next_exception: BaseException | None = None

    async def submit_order(self, order: Order) -> list[Fill]:
        if self.next_exception is not None:
            exc, self.next_exception = self.next_exception, None
            raise exc
        self.submitted.append(order)
        return [
            Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=150.0,
                commission=1.0,
                timestamp="2026-05-21T16:00:00Z",
            )
        ]

    async def cancel_order(self, order_id: UUID) -> bool:
        self.cancelled.append(order_id)
        return True

    async def get_position(self, ticker: str) -> Position | None:
        if ticker == "AAPL":
            return Position(ticker="AAPL", quantity=42, avg_cost=148.0)
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=500_000.0, positions={})


def _make_order(ticker: str = "AAPL", qty: int = 10) -> Order:
    return Order(
        order_id=uuid4(),
        ticker=ticker,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
    )


def _read_journal(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def journal_path(tmp_path: Path) -> Path:
    return tmp_path / "safe_broker.jsonl"


@pytest.fixture
def clock() -> EventClock:
    c = EventClock()
    c.tick(1_700_000_000_000_000_000)  # arbitrary fixed instant
    return c


@pytest.fixture
def prices() -> dict[str, float]:
    return {"AAPL": 150.0, "MSFT": 300.0}


@pytest.fixture
def state() -> PortfolioState:
    return PortfolioState(cash=1_000_000.0, positions={})


@pytest.fixture
def risk_gate() -> RiskGate:
    return RiskGate.default_us_equities(
        max_order_notional=200_000.0,
        max_gross_leverage=2.0,
        max_position_weight=0.25,
        min_cash=0.0,
    )


@pytest.fixture
def safe_broker(
    journal_path: Path,
    clock: EventClock,
    prices: dict[str, float],
    state: PortfolioState,
    risk_gate: RiskGate,
) -> tuple[SafeBroker, _MockBroker]:
    mock = _MockBroker()
    sb = SafeBroker(
        mock,
        risk_gate,
        journal_path,
        price_provider=lambda: prices,
        state_provider=lambda: state,
        clock=clock,
    )
    return sb, mock


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
def test_safe_broker_satisfies_async_broker_protocol(
    safe_broker: tuple[SafeBroker, _MockBroker],
) -> None:
    sb, _ = safe_broker
    assert isinstance(sb, AsyncBrokerProtocol)


def test_mock_broker_satisfies_async_broker_protocol() -> None:
    assert isinstance(_MockBroker(), AsyncBrokerProtocol)


# ---------------------------------------------------------------------------
# Happy path: order passes risk -> submit + fill journaled
# ---------------------------------------------------------------------------
def test_submit_order_passes_risk_then_fills(
    safe_broker: tuple[SafeBroker, _MockBroker], journal_path: Path
) -> None:
    sb, mock = safe_broker
    order = _make_order("AAPL", 10)
    fills = asyncio.run(sb.submit_order(order))
    asyncio.run(sb.aclose())

    assert len(fills) == 1
    assert fills[0].order_id == order.order_id
    assert mock.submitted == [order]

    records = _read_journal(journal_path)
    # Expect: one submit + one fill record.
    event_types = [r["event_type"] for r in records]
    assert event_types == ["submit", "fill"], event_types
    assert records[0]["order_id"] == str(order.order_id)
    assert records[1]["order_id"] == str(order.order_id)
    assert records[1]["price"] == 150.0


# ---------------------------------------------------------------------------
# Reject path: order fails risk -> empty fills, journal reject, broker untouched
# ---------------------------------------------------------------------------
def test_submit_order_rejected_by_unknown_ticker(
    journal_path: Path,
    clock: EventClock,
    prices: dict[str, float],
    state: PortfolioState,
    risk_gate: RiskGate,
) -> None:
    """Order on ticker absent from price map -> known_ticker_check rejects."""
    mock = _MockBroker()
    sb = SafeBroker(
        mock,
        risk_gate,
        journal_path,
        price_provider=lambda: prices,
        state_provider=lambda: state,
        clock=clock,
    )
    order = _make_order("ZZZZ", 5)
    fills = asyncio.run(sb.submit_order(order))
    asyncio.run(sb.aclose())

    assert fills == []
    assert mock.submitted == []  # underlying broker NOT called

    records = _read_journal(journal_path)
    assert len(records) == 1
    assert records[0]["event_type"] == "reject"
    assert records[0]["check"] == "known_ticker"
    assert records[0]["ticker"] == "ZZZZ"


def test_submit_order_rejected_by_fat_finger(
    journal_path: Path,
    clock: EventClock,
    prices: dict[str, float],
    state: PortfolioState,
) -> None:
    """Cap at 200k; order for 100k*150 = 15M -> max_order_notional rejects."""
    gate = RiskGate.default_us_equities(max_order_notional=200_000.0)
    mock = _MockBroker()
    sb = SafeBroker(
        mock,
        gate,
        journal_path,
        price_provider=lambda: prices,
        state_provider=lambda: state,
        clock=clock,
    )
    order = _make_order("AAPL", 100_000)
    fills = asyncio.run(sb.submit_order(order))
    asyncio.run(sb.aclose())

    assert fills == []
    assert mock.submitted == []

    records = _read_journal(journal_path)
    assert records[0]["event_type"] == "reject"
    assert records[0]["check"] == "max_order_notional"


# ---------------------------------------------------------------------------
# Cancel — journaled with ack
# ---------------------------------------------------------------------------
def test_cancel_order_journaled(
    safe_broker: tuple[SafeBroker, _MockBroker], journal_path: Path
) -> None:
    sb, mock = safe_broker
    oid = uuid4()
    ack = asyncio.run(sb.cancel_order(oid))
    asyncio.run(sb.aclose())

    assert ack is True
    assert mock.cancelled == [oid]

    records = _read_journal(journal_path)
    assert len(records) == 1
    assert records[0]["event_type"] == "cancel"
    assert records[0]["order_id"] == str(oid)
    assert records[0]["ack"] is True


# ---------------------------------------------------------------------------
# Read-only queries pass through without journal
# ---------------------------------------------------------------------------
def test_get_position_passes_through_no_journal(
    safe_broker: tuple[SafeBroker, _MockBroker], journal_path: Path
) -> None:
    sb, _ = safe_broker
    pos = asyncio.run(sb.get_position("AAPL"))
    asyncio.run(sb.aclose())
    assert pos is not None
    assert pos.quantity == 42
    # Journal file may not even be created.
    assert _read_journal(journal_path) == []


def test_get_account_state_passes_through_no_journal(
    safe_broker: tuple[SafeBroker, _MockBroker], journal_path: Path
) -> None:
    sb, _ = safe_broker
    st = asyncio.run(sb.get_account_state())
    asyncio.run(sb.aclose())
    assert isinstance(st, PortfolioState)
    assert st.cash == 500_000.0
    assert _read_journal(journal_path) == []


# ---------------------------------------------------------------------------
# Journal record shape
# ---------------------------------------------------------------------------
def test_journal_record_to_dict_flat_keys() -> None:
    """JournalRecord.to_dict flattens payload alongside ts_ns/event_type
    so JSONL consumers can grep `event_type` without nested parsing."""
    r = JournalRecord(ts_ns=42, event_type="cancel", payload={"order_id": "abc"})
    d = r.to_dict()
    assert d["ts_ns"] == 42
    assert d["event_type"] == "cancel"
    assert d["order_id"] == "abc"


def test_journal_lines_are_valid_json(
    safe_broker: tuple[SafeBroker, _MockBroker], journal_path: Path
) -> None:
    sb, _ = safe_broker
    asyncio.run(sb.submit_order(_make_order("AAPL", 5)))
    asyncio.run(sb.submit_order(_make_order("ZZZZ", 1)))  # reject
    asyncio.run(sb.cancel_order(uuid4()))
    asyncio.run(sb.aclose())

    text = journal_path.read_text()
    lines = [line for line in text.splitlines() if line]
    for line in lines:
        # Must not raise.
        record = json.loads(line)
        assert "ts_ns" in record
        assert "event_type" in record


# ---------------------------------------------------------------------------
# aclose is idempotent
# ---------------------------------------------------------------------------
def test_aclose_is_idempotent(safe_broker: tuple[SafeBroker, _MockBroker]) -> None:
    sb, _ = safe_broker
    asyncio.run(sb.cancel_order(uuid4()))  # opens journal
    asyncio.run(sb.aclose())
    asyncio.run(sb.aclose())  # second close: no-op, no exception


# ---------------------------------------------------------------------------
# Non-timeout exception from underlying broker propagates
# ---------------------------------------------------------------------------
def test_broker_exception_propagates(safe_broker: tuple[SafeBroker, _MockBroker]) -> None:
    sb, mock = safe_broker
    mock.next_exception = RuntimeError("simulated network error")
    with pytest.raises(RuntimeError, match="simulated network error"):
        asyncio.run(sb.submit_order(_make_order("AAPL", 5)))
    asyncio.run(sb.aclose())


# ---------------------------------------------------------------------------
# Empty-price-map case: every order rejected by known_ticker_check
# ---------------------------------------------------------------------------
def test_empty_price_map_rejects_all_orders(
    journal_path: Path,
    clock: EventClock,
    state: PortfolioState,
    risk_gate: RiskGate,
) -> None:
    mock = _MockBroker()
    sb = SafeBroker(
        mock,
        risk_gate,
        journal_path,
        price_provider=lambda: {},  # empty market
        state_provider=lambda: state,
        clock=clock,
    )
    fills = asyncio.run(sb.submit_order(_make_order("AAPL", 5)))
    asyncio.run(sb.aclose())

    assert fills == []
    assert mock.submitted == []
    records = _read_journal(journal_path)
    assert records[0]["event_type"] == "reject"
    assert records[0]["check"] == "known_ticker"
