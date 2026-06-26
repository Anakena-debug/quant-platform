"""S25 PR1 — PaperBroker submit-and-fill lifecycle smoke (offline).

Extends the S24 lifecycle smoke across the broker boundary using
``quantengine.execution.paper.PaperBroker`` — deterministic,
synchronous, no external dependencies. This is the
replay/paper-first half of S25; the IBKR-paper half (PR2-3) requires
TWS/Gateway + paper credentials + RTH and lives in
``quantengine/tests/test_s25_ibkr_paper_smoke.py``.

Lifts S24's AC5 prohibition specifically for paper-broker submission;
the test imports ``PaperBroker`` (which would have been an AC5
violation under S24's structural assertion). S25 PR1 makes a
narrower AC5 claim: no IBKR / live broker class is imported.

PR1 is mostly composition — it chains S24 fixtures (orders,
portfolio state, market snapshot) through PaperBroker to fills, and
asserts the bookkeeping identities. PR1 surfaces no new modules in
``quantengine`` source.

Locked PR1 decisions (per sprint plan §5 PR1 + this offline-WIP
extension):
  - Broker: ``PaperBroker`` with default ``LinearCostModel``
    (1 bp slippage, default commission). Cost-model parameters are
    NOT pinned at the AC4 invariant level — bookkeeping identities
    must hold for any cost model.
  - Account states: BOTH (empty + pre-staged), inherited from S24.
  - AC4 framing: bookkeeping identities, exact float math
    (PaperBroker computes Fill from order × price; the test
    re-derives the same expression).
  - AC5 framing: structural — every submitted order produces
    exactly one fill (no orphans, no inventions); no IBKR / live
    broker class imported.

Run:
  uv run --directory quantstrat pytest tests/test_s25_paper_broker_smoke.py -v
"""

from __future__ import annotations

import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.state import PortfolioState

# ─── PR1 fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def paper_broker() -> PaperBroker:
    """Default PaperBroker (LinearCostModel: 1 bp slippage)."""
    return PaperBroker()


@pytest.fixture
def fills_empty(
    paper_broker: PaperBroker,
    orders_empty: list[Order],
    market_snapshot: MarketSnapshot,
) -> list[Fill]:
    return paper_broker.submit_orders(orders_empty, market_snapshot)


@pytest.fixture
def fills_pre_staged(
    paper_broker: PaperBroker,
    orders_pre_staged: list[Order],
    market_snapshot: MarketSnapshot,
) -> list[Fill]:
    return paper_broker.submit_orders(orders_pre_staged, market_snapshot)


# ─── AC1 structural — fill count + order traceability ────────────────


def test_fill_count_matches_order_count_empty(
    fills_empty: list[Fill], orders_empty: list[Order]
) -> None:
    assert len(fills_empty) == len(orders_empty)


def test_fill_count_matches_order_count_pre_staged(
    fills_pre_staged: list[Fill], orders_pre_staged: list[Order]
) -> None:
    assert len(fills_pre_staged) == len(orders_pre_staged)


def test_every_fill_traces_to_a_submitted_order_empty(
    fills_empty: list[Fill], orders_empty: list[Order]
) -> None:
    """No orphan fills, no inventions: each fill's order_id matches a
    submitted order, with consistent ticker + signed_quantity."""
    by_id = {o.order_id: o for o in orders_empty}
    assert {f.order_id for f in fills_empty} == set(by_id)
    for f in fills_empty:
        o = by_id[f.order_id]
        assert f.ticker == o.ticker
        assert f.signed_quantity == o.signed_quantity


def test_every_fill_traces_to_a_submitted_order_pre_staged(
    fills_pre_staged: list[Fill], orders_pre_staged: list[Order]
) -> None:
    by_id = {o.order_id: o for o in orders_pre_staged}
    assert {f.order_id for f in fills_pre_staged} == set(by_id)
    for f in fills_pre_staged:
        o = by_id[f.order_id]
        assert f.ticker == o.ticker
        assert f.signed_quantity == o.signed_quantity


def test_no_open_orders_after_submit_empty(
    paper_broker: PaperBroker, fills_empty: list[Fill]
) -> None:
    """All DJ30 tickers are priced in the snapshot, so no order is
    queued unfilled — every order produces a fill."""
    assert list(paper_broker.open_orders()) == []


# ─── AC4 post-fill arithmetic — bookkeeping identities ───────────────


def test_fill_cash_delta_identity_empty(fills_empty: list[Fill]) -> None:
    """Fill.cash_delta = -(signed_quantity × price) - commission."""
    for f in fills_empty:
        expected = -(f.signed_quantity * f.price) - f.commission
        assert f.cash_delta == pytest.approx(expected, abs=1e-9)


def test_state_cash_invariant_empty(
    fills_empty: list[Fill], portfolio_state_empty: PortfolioState
) -> None:
    """Sum of fill.cash_delta = post-state cash − pre-state cash."""
    pre_cash = portfolio_state_empty.cash
    expected_delta = sum(f.cash_delta for f in fills_empty)
    state = portfolio_state_empty
    for f in fills_empty:
        state = state.apply(f)
    assert state.cash == pytest.approx(pre_cash + expected_delta, abs=1e-6)


def test_state_position_invariant_empty(
    fills_empty: list[Fill], portfolio_state_empty: PortfolioState
) -> None:
    """Per-ticker post quantity = pre quantity + Σ signed fills."""
    state = portfolio_state_empty
    for f in fills_empty:
        state = state.apply(f)
    expected_qty: dict[str, int] = {}
    for f in fills_empty:
        expected_qty[f.ticker] = expected_qty.get(f.ticker, 0) + f.signed_quantity
    for ticker, qty in expected_qty.items():
        if qty == 0:
            assert ticker not in state.positions, (
                f"net-zero ticker {ticker} should not appear in positions"
            )
        else:
            assert ticker in state.positions, f"missing position for {ticker}"
            assert state.positions[ticker].quantity == qty, (
                f"qty mismatch for {ticker}: expected {qty}, got {state.positions[ticker].quantity}"
            )


def test_state_cash_invariant_pre_staged(
    fills_pre_staged: list[Fill], portfolio_state_pre_staged: PortfolioState
) -> None:
    pre_cash = portfolio_state_pre_staged.cash
    expected_delta = sum(f.cash_delta for f in fills_pre_staged)
    state = portfolio_state_pre_staged
    for f in fills_pre_staged:
        state = state.apply(f)
    assert state.cash == pytest.approx(pre_cash + expected_delta, abs=1e-6)


def test_state_position_invariant_pre_staged(
    fills_pre_staged: list[Fill],
    portfolio_state_pre_staged: PortfolioState,
) -> None:
    pre_qty = {t: p.quantity for t, p in portfolio_state_pre_staged.positions.items()}
    state = portfolio_state_pre_staged
    for f in fills_pre_staged:
        state = state.apply(f)
    delta_qty: dict[str, int] = {}
    for f in fills_pre_staged:
        delta_qty[f.ticker] = delta_qty.get(f.ticker, 0) + f.signed_quantity
    all_tickers = set(pre_qty) | set(delta_qty)
    for t in all_tickers:
        post = pre_qty.get(t, 0) + delta_qty.get(t, 0)
        if post == 0:
            assert t not in state.positions, f"net-zero ticker {t} should not appear in positions"
        else:
            assert t in state.positions, f"missing position for {t}"
            assert state.positions[t].quantity == post, (
                f"qty mismatch for {t}: expected {post}, got {state.positions[t].quantity}"
            )


# ─── AC5 dry-run discipline (PaperBroker variant) ────────────────────
#
# S24's AC5 was "no broker class imported anywhere." S25 PR1 lifts
# that specifically for PaperBroker (intentional: this IS the broker
# wiring). The narrower AC5 here: no IBKR / live broker imported.


def test_no_ibkr_or_live_broker_imported_in_test_module() -> None:
    import sys

    test_module = sys.modules[__name__]
    forbidden = ["IBKRBroker", "AbstractLiveBroker"]
    in_scope = [n for n in forbidden if hasattr(test_module, n)]
    assert not in_scope, f"AC5 violation: {in_scope} present in test scope"


def test_paper_broker_is_paper_by_class_name(paper_broker: PaperBroker) -> None:
    """Defensive: structural check that we instantiated the paper class."""
    assert type(paper_broker).__name__ == "PaperBroker"


# ─── AC4 aggregate sanity — sum-over-orders bookkeeping ──────────────


def test_aggregate_cash_consistent_with_sum_of_notionals_and_commissions_empty(
    fills_empty: list[Fill],
) -> None:
    """Σ cash_delta = -Σ notional - Σ commission."""
    sum_cash_delta = sum(f.cash_delta for f in fills_empty)
    sum_notional = sum(f.signed_quantity * f.price for f in fills_empty)
    sum_commission = sum(f.commission for f in fills_empty)
    assert sum_cash_delta == pytest.approx(-sum_notional - sum_commission, abs=1e-6)
