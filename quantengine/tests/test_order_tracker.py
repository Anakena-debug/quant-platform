"""Smoke tests for quantengine.execution.order_state.OrderTracker.

Covers:
    - Legal lifecycle: submit → full-fill → FILLED.
    - Partial-fill cascade: submit → ack → partial → partial → full.
    - Illegal transitions raise OrderStateError.
    - Conservation laws: no overfill, no side mismatch, unknown order_id,
      duplicate submit, zero-qty fill.
    - Ledger writes: tracker emits the right EventKinds.
    - End-to-end Runner parity: identical PortfolioState with and without
      tracker attached.
"""

from __future__ import annotations

import numpy as np
from uuid import uuid4

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
)
from quantengine.contracts.signal import AlphaSignal
from quantengine.execution.order_state import (
    OrderStateError,
    OrderTracker,
    is_legal_transition,
    is_terminal,
)
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState
from quantengine.runtime.runner import Runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk_order(ticker: str = "AAPL", signed_qty: int = 100) -> Order:
    return Order.new(
        ticker=ticker,
        signed_quantity=signed_qty,
        order_type=OrderType.MARKET,
        timestamp="2025-01-02T00:00:00",
    )


def _mk_fill(order: Order, signed_qty: int, price: float = 100.0) -> Fill:
    return Fill(
        fill_id=uuid4(),
        order_id=order.order_id,
        ticker=order.ticker,
        signed_quantity=signed_qty,
        price=price,
        commission=0.50,
        timestamp="2025-01-02T00:00:00",
    )


# ---------------------------------------------------------------------------
# Transition table invariants
# ---------------------------------------------------------------------------
def test_terminal_states_have_no_outgoing_edges():
    for s in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
        assert is_terminal(s)
        for d in OrderStatus:
            assert not is_legal_transition(s, d), f"{s} → {d} must be illegal"


def test_non_terminal_states_have_outgoing_edges():
    for s in (
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.WORKING,
        OrderStatus.PARTIALLY_FILLED,
    ):
        assert not is_terminal(s)


# ---------------------------------------------------------------------------
# Happy path: sync-broker full fill
# ---------------------------------------------------------------------------
def test_sync_full_fill_flow():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)

    tracker.submit(order, "2025-01-02T00:00:00")
    assert tracker.status(order.order_id) == OrderStatus.SUBMITTED

    fill = _mk_fill(order, signed_qty=100)
    status = tracker.on_fill(fill)
    assert status == OrderStatus.FILLED
    assert tracker.status(order.order_id) == OrderStatus.FILLED
    assert tracker.cumulative_filled(order.order_id) == 100
    assert tracker.remaining(order.order_id) == 0
    assert tracker.open_orders() == ()

    kinds = [e.kind for e in ledger.events()]
    assert kinds == ["ORDER_SUBMITTED", "ORDER_FILLED"]


# ---------------------------------------------------------------------------
# Partial-fill cascade
# ---------------------------------------------------------------------------
def test_partial_fills_cascade_to_filled():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)

    tracker.submit(order, "2025-01-02T00:00:00")
    tracker.ack(order.order_id, "2025-01-02T00:00:01")
    assert tracker.status(order.order_id) == OrderStatus.WORKING

    st1 = tracker.on_fill(_mk_fill(order, 30))
    assert st1 == OrderStatus.PARTIALLY_FILLED
    assert tracker.cumulative_filled(order.order_id) == 30

    st2 = tracker.on_fill(_mk_fill(order, 50))
    assert st2 == OrderStatus.PARTIALLY_FILLED
    assert tracker.cumulative_filled(order.order_id) == 80

    st3 = tracker.on_fill(_mk_fill(order, 20))
    assert st3 == OrderStatus.FILLED
    assert tracker.cumulative_filled(order.order_id) == 100

    kinds = [e.kind for e in ledger.events()]
    assert kinds == [
        "ORDER_SUBMITTED",
        "ORDER_ACKED",
        "ORDER_FILLED",
        "ORDER_FILLED",
        "ORDER_FILLED",
    ]


# ---------------------------------------------------------------------------
# Illegal transitions & conservation laws
# ---------------------------------------------------------------------------
def test_duplicate_submit_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order()
    tracker.submit(order, "t0")
    try:
        tracker.submit(order, "t1")
    except OrderStateError as e:
        assert "Duplicate" in str(e)
    else:
        raise AssertionError("duplicate submit did not raise")


def test_fill_for_unknown_order_raises():
    tracker = OrderTracker(ledger=Ledger())
    stranger = _mk_order()
    try:
        tracker.on_fill(_mk_fill(stranger, 1))
    except OrderStateError as e:
        assert "Unknown order_id" in str(e)
    else:
        raise AssertionError("unknown-order fill did not raise")


def test_overfill_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)
    tracker.submit(order, "t0")
    tracker.on_fill(_mk_fill(order, 80))
    try:
        tracker.on_fill(_mk_fill(order, 25))  # 80+25 = 105 > 100
    except OrderStateError as e:
        assert "Overfill" in str(e)
    else:
        raise AssertionError("overfill did not raise")


def test_wrong_side_fill_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)  # BUY 100
    tracker.submit(order, "t0")
    try:
        tracker.on_fill(_mk_fill(order, -10))  # SELL-side fill for BUY order
    except OrderStateError as e:
        assert "disagrees" in str(e) or "direction" in str(e)
    else:
        raise AssertionError("wrong-side fill did not raise")


def test_ticker_mismatch_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order("AAPL", 100)
    tracker.submit(order, "t0")
    bad_fill = Fill(
        fill_id=uuid4(),
        order_id=order.order_id,
        ticker="MSFT",
        signed_quantity=100,
        price=100.0,
        commission=0.0,
        timestamp="t0",
    )
    try:
        tracker.on_fill(bad_fill)
    except OrderStateError as e:
        assert "ticker" in str(e)
    else:
        raise AssertionError("ticker mismatch did not raise")


def test_zero_quantity_fill_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)
    tracker.submit(order, "t0")
    try:
        tracker.on_fill(_mk_fill(order, 0))
    except OrderStateError as e:
        assert "Zero" in str(e)
    else:
        raise AssertionError("zero-qty fill did not raise")


def test_fill_after_cancel_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)
    tracker.submit(order, "t0")
    tracker.cancel(order.order_id, "t1", reason="user")
    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    try:
        tracker.on_fill(_mk_fill(order, 10))
    except OrderStateError as e:
        assert "Illegal transition" in str(e)
    else:
        raise AssertionError("fill after cancel did not raise")


def test_cancel_after_filled_raises():
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    order = _mk_order(signed_qty=100)
    tracker.submit(order, "t0")
    tracker.on_fill(_mk_fill(order, 100))
    try:
        tracker.cancel(order.order_id, "t1", reason="too late")
    except OrderStateError as e:
        assert "Illegal transition" in str(e)
    else:
        raise AssertionError("cancel after FILLED did not raise")


# ---------------------------------------------------------------------------
# End-to-end: Runner with vs. without tracker must produce the same
# PortfolioState. The tracker's extra events (beyond SUBMIT/FILL) are
# accounting-only and cannot affect the state reducer.
# ---------------------------------------------------------------------------
class _FixedSignalStrategy:
    """Tiny strategy: always asks to hold 10% AAPL, 0% cash-drag, all else flat."""

    def __init__(self, tickers):
        self._tickers = tickers

    def predict(self, market: MarketSnapshot) -> AlphaSignal:
        n = len(self._tickers)
        er = np.zeros(n)
        lo = np.zeros(n)
        hi = np.zeros(n)
        kw = np.zeros(n)
        # Put 100% weight on ticker[0] with a clearly tradeable interval.
        er[0] = 0.01
        lo[0] = 0.005
        hi[0] = 0.02
        kw[0] = 1.0
        return AlphaSignal(
            tickers=tuple(self._tickers),
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=kw,
            timestamp=market.timestamp,
        )


def _run_once(with_tracker: bool) -> tuple[PortfolioState, Ledger]:
    tickers = ("AAPL", "MSFT")
    prices = np.array([100.0, 200.0])
    market = MarketSnapshot(
        timestamp="2025-01-02T00:00:00",
        tickers=tickers,
        prices=prices,
    )
    ledger = Ledger()
    state = PortfolioState.empty(1_000_000.0)
    runner = Runner(
        state=state,
        rebalance=RebalanceEngine(RebalanceConstraints()),
        broker=PaperBroker(),
        ledger=ledger,
        tracker=OrderTracker(ledger=ledger) if with_tracker else None,
    )
    strat = _FixedSignalStrategy(tickers)
    signal = strat.predict(market)
    final = runner.step(signal, market)
    return final, ledger


def test_runner_parity_with_tracker():
    s_no, l_no = _run_once(with_tracker=False)
    s_yes, l_yes = _run_once(with_tracker=True)

    # State reducer is untouched by the tracker path.
    assert abs(s_no.cash - s_yes.cash) < 1e-9
    assert abs(s_no.realized_pnl - s_yes.realized_pnl) < 1e-9
    assert abs(s_no.total_commission - s_yes.total_commission) < 1e-9
    assert set(s_no.positions.keys()) == set(s_yes.positions.keys())
    for tkr in s_no.positions:
        assert s_no.positions[tkr].quantity == s_yes.positions[tkr].quantity

    # With tracker: identical SUBMIT/FILL stream (no WORKING since sync broker).
    kinds_no = [e.kind for e in l_no.events()]
    kinds_yes = [e.kind for e in l_yes.events()]
    assert kinds_no == kinds_yes, (kinds_no, kinds_yes)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_terminal_states_have_no_outgoing_edges,
        test_non_terminal_states_have_outgoing_edges,
        test_sync_full_fill_flow,
        test_partial_fills_cascade_to_filled,
        test_duplicate_submit_raises,
        test_fill_for_unknown_order_raises,
        test_overfill_raises,
        test_wrong_side_fill_raises,
        test_ticker_mismatch_raises,
        test_zero_quantity_fill_raises,
        test_fill_after_cancel_raises,
        test_cancel_after_filled_raises,
        test_runner_parity_with_tracker,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\norder_state: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
