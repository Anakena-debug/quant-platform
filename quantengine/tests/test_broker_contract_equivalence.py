"""Cross-broker contract equivalence (S25 Phase A.3 — option c).

Exercises both ``PaperBroker`` and ``IBKRBroker`` against assertions
restricted to the cross-implementation invariants (CI-1) … (CI-6)
documented on ``AbstractBroker`` in
``quantengine/execution/broker.py``. Implementation-specific behaviour
(exact fill price, exact commission magnitude, return ordering of
fills) is NOT asserted here — those checks belong in
``test_paper_broker.py`` and ``test_ibkr_broker.py`` respectively.

PaperBroker scenarios are unconditional. IBKRBroker scenarios are
gated by ``IBKR_PAPER_SMOKE=1`` (matching the S22 ``test_ibkr_paper_smoke.py``
gating convention); RTH + booted TWS/Gateway are the operator's
responsibility when opting in.

Scenario: BUY 1 AAPL + SELL 1 AAPL submitted as a single batch via
``submit_orders``. Net-flat after the round-trip so the paper account
doesn't accumulate across runs. Per-trade quantity is intentionally
the minimum (1 share) to keep the AC6 max-notional-cap risk negligible
during S25 development.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.execution.paper import PaperBroker

# ─── Shared scenario ─────────────────────────────────────────────────


def _make_round_trip_scenario() -> tuple[list[Order], MarketSnapshot]:
    """BUY 1 AAPL + SELL 1 AAPL → single-batch net-flat scenario.

    The MarketSnapshot price (1.00) is a placeholder for PaperBroker's
    cost-model math; IBKRBroker ignores it (real fill price comes from
    the live market). PaperBroker requires a positive price; 1.00 is
    the minimum viable value that keeps fixture commission < $1
    minimum-floor risk identical across both implementations.
    """
    orders = [
        Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        ),
        Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.SELL,
            quantity=1,
            order_type=OrderType.MARKET,
        ),
    ]
    market = MarketSnapshot(
        timestamp="2026-05-11T15:00:00+00:00",
        tickers=("AAPL",),
        prices=np.array([1.0]),
    )
    return orders, market


# ─── Invariant helpers (CI-1, CI-3, CI-4, CI-6) ──────────────────────


def _assert_ci_1_execution_level_fills(orders: list[Order], fills: list[Fill]) -> None:
    """(CI-1) execution-level: zero or more fills per order; aggregate matches target.

    The contract (per AbstractBroker docstring CI-1) does NOT promise
    one fill per order — IBKR may emit multiple execution-level Fill
    objects for a single parent order on partial fills. Callers
    aggregate by ``Fill.order_id``.

    For this scenario (MARKET round-trip against liquid AAPL in a
    paper account), every submitted order is expected to terminally
    fill, so the aggregate signed_quantity per order_id equals the
    parent order's signed_quantity. Assertions:

    1. Every fill's ``order_id`` is in the submitted set (no orphans).
    2. Every submitted order received at least one fill (this scenario
       expects terminal-FILLED for every order; a violation is a real
       finding, not a contract relaxation).
    3. ``sum(f.signed_quantity for f in fills_of(o))`` == ``o.signed_quantity``
       (aggregate matches target ⇒ terminal-FILLED).
    """
    submitted_ids = {o.order_id for o in orders}
    by_id = {o.order_id: o for o in orders}

    fills_by_order: dict[Any, list[Fill]] = {}
    for f in fills:
        assert f.order_id in submitted_ids, (
            f"(CI-1) orphan fill {f.fill_id}: order_id {f.order_id} not in submitted"
        )
        fills_by_order.setdefault(f.order_id, []).append(f)

    for order_id in submitted_ids:
        order_fills = fills_by_order.get(order_id, [])
        assert order_fills, (
            f"(CI-1) submitted order {order_id} ({by_id[order_id].ticker} "
            f"{by_id[order_id].signed_quantity}) received no fills"
        )
        agg = sum(f.signed_quantity for f in order_fills)
        target = by_id[order_id].signed_quantity
        assert agg == target, (
            f"(CI-1) order {order_id} ({by_id[order_id].ticker}): "
            f"aggregate signed_quantity {agg} != target {target}"
        )


def _assert_ci_3_commission_nonneg(fills: list[Fill]) -> None:
    """(CI-3) Fill.commission >= 0. Magnitudes NOT compared across implementations."""
    for f in fills:
        assert f.commission >= 0.0, (
            f"(CI-3) negative commission on fill {f.fill_id}: {f.commission}"
        )


def _assert_ci_4_cash_delta_identity(fills: list[Fill]) -> None:
    """(CI-4) Fill.cash_delta == -(signed_quantity × price) - commission.

    Holds for both implementations by construction (the property is
    computed on the Fill itself). Byte-equal float math expected.
    """
    for f in fills:
        expected = -(f.signed_quantity * f.price) - f.commission
        assert f.cash_delta == pytest.approx(expected, abs=1e-9), (
            f"(CI-4) cash_delta identity broken on {f.fill_id}: "
            f"cash_delta={f.cash_delta}, expected={expected}"
        )


def _assert_ci_6_order_id_preserved(orders: list[Order], fills: list[Fill]) -> None:
    """(CI-6) Fill.order_id mirrors Order.order_id (UUID); ticker + signed_qty match."""
    by_id = {o.order_id: o for o in orders}
    for f in fills:
        assert f.order_id in by_id, (
            f"(CI-6) orphan fill {f.fill_id}: order_id {f.order_id} not in submitted orders"
        )
        o = by_id[f.order_id]
        assert f.ticker == o.ticker, f"(CI-6) ticker mismatch: fill={f.ticker} order={o.ticker}"
        assert f.signed_quantity == o.signed_quantity, (
            f"(CI-6) signed_quantity mismatch on {f.order_id}: "
            f"fill={f.signed_quantity} order={o.signed_quantity}"
        )


# ─── PaperBroker — unconditional ─────────────────────────────────────


@pytest.fixture
def paper_broker() -> PaperBroker:
    """Default PaperBroker (LinearCostModel: 1bp slippage, $0.005/share, $1 min)."""
    return PaperBroker()


@pytest.fixture
def paper_broker_scenario(
    paper_broker: PaperBroker,
) -> tuple[list[Order], list[Fill]]:
    """PaperBroker scenario: BUY+SELL in one batch (PaperBroker has no
    wash-trade gate; batch is fine). Asserts non-empty fills before
    yielding so vacuous-pass on empty-fills regression cannot hide
    a real failure on the count gate."""
    orders, market = _make_round_trip_scenario()
    fills = paper_broker.submit_orders(orders, market)
    assert len(fills) > 0, (
        "PaperBroker fixture produced zero fills — refusing to yield a vacuous scenario."
    )
    return orders, fills


def test_paper_broker_ci_1_one_fill_per_order(
    paper_broker_scenario: tuple[list[Order], list[Fill]],
) -> None:
    orders, fills = paper_broker_scenario
    _assert_ci_1_execution_level_fills(orders, fills)


def test_paper_broker_ci_3_commission_nonneg(
    paper_broker_scenario: tuple[list[Order], list[Fill]],
) -> None:
    _, fills = paper_broker_scenario
    _assert_ci_3_commission_nonneg(fills)


def test_paper_broker_ci_4_cash_delta_identity(
    paper_broker_scenario: tuple[list[Order], list[Fill]],
) -> None:
    _, fills = paper_broker_scenario
    _assert_ci_4_cash_delta_identity(fills)


def test_paper_broker_ci_6_order_id_preserved(
    paper_broker_scenario: tuple[list[Order], list[Fill]],
) -> None:
    orders, fills = paper_broker_scenario
    _assert_ci_6_order_id_preserved(orders, fills)


def test_paper_broker_ci_5_open_orders_truthful_after_priced_submit(
    paper_broker: PaperBroker,
    paper_broker_scenario: tuple[list[Order], list[Fill]],
) -> None:
    """(CI-5 positive side) PaperBroker's open_orders is truthful: after
    submitting orders whose tickers are priced in the market snapshot,
    nothing is queued. This is the contract caller can rely on for
    PaperBroker (PaperBroker only)."""
    # Scenario fixture has already run submit_orders.
    assert list(paper_broker.open_orders()) == [], (
        "(CI-5) PaperBroker.open_orders should be empty after submit of fully-priced orders"
    )


# ─── IBKRBroker — opt-in via IBKR_PAPER_SMOKE=1 ──────────────────────


_IBKR_SKIP_REASON = (
    "opt-in IBKR contract equivalence; set IBKR_PAPER_SMOKE=1 (also "
    "ensure TWS/Gateway is up + RTH window). Same gating convention as "
    "test_ibkr_paper_smoke.py."
)


@pytest.fixture
def ibkr_broker_scenario():
    """Live IBKRBroker round-trip scenario.

    Yields (orders, fills, broker) where ``broker`` is the IBKRBroker
    used to submit. Connection is opened on fixture entry and
    disconnected unconditionally on fixture teardown.

    Submission pattern: **BUY in its own batch, then SELL in its own
    batch** (two ``submit_orders`` calls, each with one order) —
    mirrors S22's working pattern. Submitting BUY+SELL of the same
    ticker as a single batch triggers IBKR Error 10349 (wash-trade
    gate); runtime-verified 2026-05-11. Production rebalance pattern
    (one direction per ticker per cycle) is unaffected and uses the
    multi-ticker batch path.

    Imports are inside the fixture body so the IBKR import path
    (which transitively imports ``ib_async``) is NOT exercised when
    the test is skipped — matching the S22 paper-smoke convention.
    """
    from quantengine.execution.ibkr.broker import IBKRBroker
    from quantengine.execution.ibkr.config import IBKRConfig
    from quantengine.execution.ibkr.connection import (
        IBKRConnection,
        assert_paper_account,
    )

    cfg = IBKRConfig.from_env()
    connection = IBKRConnection()
    connection.connect(cfg)
    try:
        assert_paper_account(connection.ib, cfg.account)
        broker = IBKRBroker(connection=connection)
        market = MarketSnapshot(
            timestamp="2026-05-11T15:00:00+00:00",
            tickers=("AAPL",),
            prices=np.array([1.0]),
        )
        # Sequential BUY-then-SELL — each in its own batch — to avoid
        # IBKR's wash-trade gate (Error 10349).
        buy_order = Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )
        sell_order = Order(
            order_id=uuid4(),
            ticker="AAPL",
            side=OrderSide.SELL,
            quantity=1,
            order_type=OrderType.MARKET,
        )
        buy_fills = broker.submit_orders([buy_order], market)
        sell_fills = broker.submit_orders([sell_order], market)
        orders = [buy_order, sell_order]
        fills = buy_fills + sell_fills
        assert len(fills) > 0, (
            "IBKR fixture produced zero fills — refusing to yield a "
            "vacuous scenario. Check IBKR rejection reason in captured "
            "log; common causes include wash-trade gate (Error 10349), "
            "market closed, or order-preset rules in TWS."
        )
        yield orders, fills, broker
    finally:
        connection.disconnect()


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_1_one_fill_per_order(ibkr_broker_scenario) -> None:
    orders, fills, _broker = ibkr_broker_scenario
    _assert_ci_1_execution_level_fills(orders, fills)


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_3_commission_nonneg(ibkr_broker_scenario) -> None:
    _, fills, _broker = ibkr_broker_scenario
    _assert_ci_3_commission_nonneg(fills)


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_4_cash_delta_identity(ibkr_broker_scenario) -> None:
    _, fills, _broker = ibkr_broker_scenario
    _assert_ci_4_cash_delta_identity(fills)


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_6_order_id_preserved(ibkr_broker_scenario) -> None:
    orders, fills, _broker = ibkr_broker_scenario
    _assert_ci_6_order_id_preserved(orders, fills)


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_5_open_orders_phase_3_placeholder(
    ibkr_broker_scenario,
) -> None:
    """(CI-5 negative side) IBKRBroker.open_orders returns () as a Phase 3
    placeholder; callers MUST NOT rely on it as a source of truth.
    See ``quantengine/execution/ibkr/broker.py:181-189``."""
    _, _, broker = ibkr_broker_scenario
    assert tuple(broker.open_orders()) == (), (
        "(CI-5) IBKRBroker.open_orders is documented as a Phase 3 "
        "placeholder returning () regardless of actual open trades. "
        "If this assertion starts failing, Phase 4 has landed and the "
        "contract docstring needs updating."
    )


@pytest.mark.ibkr_paper
@pytest.mark.skipif(
    os.environ.get("IBKR_PAPER_SMOKE") != "1",
    reason=_IBKR_SKIP_REASON,
)
def test_ibkr_broker_ci_6_ib_order_id_recorded_in_metadata(
    ibkr_broker_scenario,
) -> None:
    """(CI-6 IBKR side) the IBKR-assigned int orderId is recorded in
    ``Fill.metadata["ib_order_id"]`` as a separate identity from our
    client-side ``order_id`` (UUID). PaperBroker has no equivalent
    metadata field; this assertion is IBKR-specific."""
    _, fills, _broker = ibkr_broker_scenario
    for f in fills:
        assert "ib_order_id" in f.metadata, (
            f"(CI-6) missing 'ib_order_id' in fill metadata: {f.metadata}"
        )
        assert isinstance(f.metadata["ib_order_id"], int), (
            f"(CI-6) ib_order_id is not an int: {f.metadata['ib_order_id']!r}"
        )
