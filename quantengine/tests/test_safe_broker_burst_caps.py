"""SafeBroker in-flight exposure accumulator (s48).

A burst of orders that each individually pass the RiskGate caps must not
collectively breach them. The per-order ``RiskGate.validate([order], state, ...)``
in ``SafeBroker.submit_order`` only sees the state snapshot at submit time; if
that snapshot does not yet reflect already-accepted siblings (because dispatch
became concurrent, or fills are async-deferred), two orders that each fit under
the gross-leverage / cash cap could together exceed it.

The accumulator folds accepted-but-unfilled orders into the validation state, so
the gate sees siblings' exposure. Under the CURRENT serialized runtime the
in-flight map is empty at every validation (each submit_order returns — and its
fill is applied to state — before the next begins), so behaviour is unchanged;
these tests pin both that no-regression property AND the concurrency-insurance
behaviour (simulated by a broker that blocks mid-submit).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from quantengine.contracts.orders import Fill, Order, OrderType
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming.protocols import Clock
from quantengine.runtime.streaming.safe_broker import SafeBroker

_PRICE = 100.0


class _StubClock(Clock):
    def __init__(self, start_ns: int = 1_700_000_000_000_000_000) -> None:
        self._t = start_ns

    def now_ns(self) -> int:
        t = self._t
        self._t += 1
        return t

    def now_iso(self) -> str:
        return "2026-05-31T16:00:00Z"


class _BlockingBroker:
    """submit_order blocks on an event so the order stays in-flight while a
    sibling is validated — simulates concurrent / deferred-fill dispatch."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.submitted: list[Order] = []

    async def submit_order(self, order: Order) -> list[Fill]:
        self.submitted.append(order)
        await self.release.wait()
        return [
            Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=_PRICE,
                commission=0.0,
                timestamp="2026-05-31T16:00:00Z",
            )
        ]

    async def cancel_order(self, order_id: UUID) -> bool:
        return True

    async def get_position(self, ticker: str) -> None:
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=0.0, positions={})


class _RaisingBroker:
    async def submit_order(self, order: Order) -> list[Fill]:
        raise RuntimeError("broker boom")

    async def cancel_order(self, order_id: UUID) -> bool:
        return True

    async def get_position(self, ticker: str) -> None:
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=0.0, positions={})


def _gate() -> RiskGate:
    # Gross-leverage cap 1.0 on a 1,000-NAV book ⇒ one 500-notional order fits
    # (0.5x), two together (1,000 notional) would be 1.0x — at the edge — and a
    # third tips over. Use cash cap as the hard stop: min_cash=0 with 1,000 cash
    # ⇒ two 600-notional buys (1,200) breach; one (600) is fine.
    return RiskGate.default_us_equities(
        max_order_notional=1_000_000.0,
        max_gross_leverage=10.0,
        max_position_weight=1.0,
        min_cash=0.0,
    )


def _order(ticker: str, qty: int) -> Order:
    return Order.new(ticker, qty, OrderType.MARKET)


def test_inflight_sibling_exposure_rejects_overcommitting_burst(tmp_path: Path) -> None:
    """Two 6-share AAPL buys at $100 = $600 each. Cash = $1,000. Each alone
    passes the non_negative_cash check; together ($1,200) they breach it. With
    the broker blocked (order 1 still in-flight), order 2 must be rejected
    because the accumulator folds order 1's $600 exposure into the state."""

    async def run() -> None:
        broker = _BlockingBroker()
        state = PortfolioState(cash=1_000.0, positions={})
        sb = SafeBroker(
            broker,
            _gate(),
            tmp_path / "j.jsonl",
            price_provider=lambda: {"AAPL": _PRICE},
            state_provider=lambda: state,  # FIXED — never updates (concurrent sim)
            clock=_StubClock(),
        )
        o1, o2 = _order("AAPL", 6), _order("AAPL", 6)

        t1 = asyncio.create_task(sb.submit_order(o1))
        # Wait until o1 is registered in-flight (after its journal-submit, at the
        # broker block) so o2 validates against state+inflight.
        while o1.order_id not in sb._inflight:
            await asyncio.sleep(0.001)

        fills2 = await sb.submit_order(o2)
        assert fills2 == []  # rejected: $600 (o1 in-flight) + $600 (o2) > $1,000

        broker.release.set()
        fills1 = await t1
        assert len(fills1) == 1  # o1 itself filled
        assert sb._inflight == {}  # both deregistered

    asyncio.run(run())


def test_serialized_burst_unchanged_when_state_reflects_fills(tmp_path: Path) -> None:
    """No-regression: under serialized dispatch each order returns and its fill
    is applied to state before the next submit, so the in-flight map is empty at
    validation and two within-budget orders both succeed."""

    async def run() -> None:
        # Mutable single-element holder so the broker can update the state the
        # provider reads, mirroring DemoBroker's apply-before-return.
        holder = [PortfolioState(cash=10_000.0, positions={})]

        class _ApplyingBroker:
            def __init__(self) -> None:
                self.submitted: list[Order] = []

            async def submit_order(self, order: Order) -> list[Fill]:
                self.submitted.append(order)
                fill = Fill(
                    fill_id=uuid4(),
                    order_id=order.order_id,
                    ticker=order.ticker,
                    signed_quantity=order.signed_quantity,
                    price=_PRICE,
                    commission=0.0,
                    timestamp="2026-05-31T16:00:00Z",
                )
                holder[0] = holder[0].apply(fill)
                return [fill]

            async def cancel_order(self, order_id: UUID) -> bool:
                return True

            async def get_position(self, ticker: str) -> None:
                return None

            async def get_account_state(self) -> PortfolioState:
                return holder[0]

        broker = _ApplyingBroker()
        sb = SafeBroker(
            broker,
            _gate(),
            tmp_path / "j.jsonl",
            price_provider=lambda: {"AAPL": _PRICE},
            state_provider=lambda: holder[0],
            clock=_StubClock(),
        )
        f1 = await sb.submit_order(_order("AAPL", 6))
        assert sb._inflight == {}  # deregistered before return
        f2 = await sb.submit_order(_order("AAPL", 6))
        assert len(f1) == 1 and len(f2) == 1
        assert len(broker.submitted) == 2
        assert sb._inflight == {}

    asyncio.run(run())


def test_inflight_cleared_on_broker_exception(tmp_path: Path) -> None:
    """A broker exception must not leak the order into the in-flight map (the
    finally-pop runs), or every future validation would over-count it forever."""

    async def run() -> None:
        sb = SafeBroker(
            _RaisingBroker(),
            _gate(),
            tmp_path / "j.jsonl",
            price_provider=lambda: {"AAPL": _PRICE},
            state_provider=lambda: PortfolioState(cash=10_000.0, positions={}),
            clock=_StubClock(),
        )
        raised = False
        try:
            await sb.submit_order(_order("AAPL", 6))
        except RuntimeError:
            raised = True
        assert raised
        assert sb._inflight == {}

    asyncio.run(run())
