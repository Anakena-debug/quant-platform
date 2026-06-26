"""S26 PR3 — Offline IBKR mapping integration / normalized cross-broker equivalence.

Submits the SAME ``list[Order]`` to two brokers in parallel:

  * ``PaperBroker`` with a shared ``LinearCostModel`` instance.
  * A test-local ``MockIBKRBroker`` (defined inline below) that invokes
    the real
    ``quantengine.execution.ibkr.order_mapping.order_to_ib_order`` on
    every order before synthesizing a ``Fill`` against the SAME
    ``LinearCostModel`` instance.

Then asserts that:

  * the real ``order_to_ib_order`` was invoked once per submitted order
    on the mock-IBKR path (AC3.1);
  * neither ``IBKRConnection`` nor ``ib_async.IB`` is constructed (AC3.2);
  * final ``PortfolioState.positions`` quantities match across the two
    paths (AC3.3);
  * final ``PortfolioState.cash`` matches across the two paths, holding
    because both brokers share the SAME ``LinearCostModel`` instance
    (AC3.4);
  * the normalized fill projection
    ``sorted((f.ticker, f.signed_quantity, f.price) for f in fills)``
    is equal across the two paths (AC3.5);
  * the ``IBKR_PAPER_SMOKE`` env var is unset or != ``"1"``; if observed
    at module-load time, the module refuses to import (AC3.6).

Hard constraints:

* No live or paper IBKR submission.
* No ``IBKR_PAPER_SMOKE``.
* No ``IBKRConnection`` construction.
* No ``ib_async.IB`` construction.
* No socket. The test must pass on a fresh checkout with pytest alone
  — no env vars, no TWS, no network.

Note (cf. plan §7.4): ``order_to_ib_order`` performs a function-body
``from ib_async import Order, Stock`` of pure data classes — those
imports do NOT open a socket. ``ib_async.IB`` is the loop-owning class
that constructs the asyncio session; that one is the load-bearing
negative.

Run:

    uv run --directory quantengine pytest tests/test_s26_cross_broker_mapping_equivalence.py -x
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order, OrderType
from quantengine.contracts.signal import build_alpha_signal
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.cost_model import CostModel, LinearCostModel
from quantengine.execution.ibkr import order_mapping
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState

# ─── AC3.6 module-load env guard ────────────────────────────────────────
# Fails import if a developer left IBKR_PAPER_SMOKE='1' in their shell.
# Per plan §3 AC3.6: "fail loud (pre-construction) rather than silently
# entering a different mode. This is intentional — the test should never
# be friendly to a misconfigured shell."
if os.environ.get("IBKR_PAPER_SMOKE") == "1":
    raise RuntimeError(
        "S26 PR3 (cross-broker mapping equivalence) must NOT run with "
        "IBKR_PAPER_SMOKE='1'. Detected env override; refusing to load "
        "this module to prevent silently running in IBKR paper-broker "
        "smoke mode. Unset IBKR_PAPER_SMOKE (or set it to anything "
        "other than '1') and rerun."
    )


# ─── AC3.2 forbidden-pattern table ──────────────────────────────────────
# Regexes (not plain substrings) so:
#   1. word boundaries (``\b``) prevent test-local ``Mock``-prefixed
#      class constructions from false-matching the live-broker negative;
#   2. env-var WRITES are distinguished from READS — reads via
#      ``os.environ.get(...)`` are allowed (used by the module-load
#      guard at the top of this file); only assignments (``= ...``)
#      and ``setenv`` calls are banned.
#
# Each regex pattern is built via string concatenation so the table
# source itself does not contain a contiguous match against its own
# regex (avoids a self-trigger when the AC3.2 test greps this file's
# bytes).
_AC3_2_FORBIDDEN_REGEXES: tuple[str, ...] = (
    # Live-broker / connection class construction.
    r"\b" + "IBKR" + r"Connection\(",
    r"\b" + "IBKR" + r"Broker\(",
    # ib_async event-loop session construction. Data classes Order /
    # Stock are imported transitively by order_to_ib_order; the
    # loop-owning class is the negative target.
    r"\b" + "ib_async" + r"\.IB\(",
    r"from " + "ib_async" + r" import IB\b",
    # IBKR paper-smoke env-var WRITES (reads via ``.get()`` are allowed).
    r"os\.environ\[[\"']" + "IBKR_PAPER_SMOKE" + r"[\"']\]\s*=",
    r"monkeypatch\.setenv\([\"']" + "IBKR_PAPER_SMOKE" + r"[\"']",
    r"\bsetenv\([\"']" + "IBKR_PAPER_SMOKE" + r"[\"']",
)


# ─── Test-local MockIBKRBroker ──────────────────────────────────────────
class MockIBKRBroker(AbstractBroker):
    """Offline IBKR-broker stand-in for the mapping-integration test.

    For every order in ``submit_orders``:

      1. Invokes the **real** ``order_to_ib_order(order)`` to exercise
         the IBKR-side mapping function (AC3.1). The returned
         ``(Contract, IBOrder)`` pair is discarded — it would otherwise
         feed into ``IB.placeOrder``, which PR3 forbids.

      2. Synthesizes a ``Fill`` using the same ``cost_model`` instance
         that the comparison ``PaperBroker`` uses, so AC3.3 / AC3.4 /
         AC3.5 hold by construction.

    The class never opens a socket, never constructs ``ib_async.IB``,
    and never touches an ``IBKRConnection``. The mapping function does
    a function-body ``from ib_async import Order, Stock`` (pure data
    classes; no network) — explicitly permitted by the plan §7.4.

    Why the attribute-lookup call form ``order_mapping.order_to_ib_order``
    (rather than ``from ... import order_to_ib_order``): so the AC3.1
    test can spy on invocations via
    ``patch.object(order_mapping, "order_to_ib_order", wraps=...)``.
    A ``from`` import would create a local binding in this module
    that ``patch.object`` could not intercept.
    """

    def __init__(self, cost_model: CostModel) -> None:
        self.cost_model: CostModel = cost_model
        self._open: list[Order] = []

    def submit_orders(self, orders: Sequence[Order], market: MarketSnapshot) -> list[Fill]:
        fills: list[Fill] = []
        price_map = {t: float(p) for t, p in zip(market.tickers, market.prices)}
        for order in orders:
            # AC3.1 integration point — call via attribute lookup so the
            # spy patches correctly.
            order_mapping.order_to_ib_order(order)
            ref = price_map.get(order.ticker)
            if ref is None:
                self._open.append(order)
                continue
            fp = self.cost_model.fill_price(order, ref)
            comm = self.cost_model.commission(order, fp)
            fills.append(
                Fill(
                    fill_id=uuid4(),
                    order_id=order.order_id,
                    ticker=order.ticker,
                    signed_quantity=order.signed_quantity,
                    price=fp,
                    commission=comm,
                    timestamp=market.timestamp,
                    metadata={
                        "reference_price": ref,
                        "ib_mapping_invoked": True,
                    },
                )
            )
        return fills

    def cancel_all(self) -> int:
        n = len(self._open)
        self._open.clear()
        return n

    def open_orders(self) -> Iterable[Order]:
        return tuple(self._open)


# ─── Shared scenario ────────────────────────────────────────────────────
@pytest.fixture
def cost_model() -> LinearCostModel:
    """Single ``LinearCostModel`` instance, shared by both brokers so
    AC3.4 (exact-cash equality) holds. Default parameters are fine —
    any ``LinearCostModel`` will do as long as ``PaperBroker`` and
    ``MockIBKRBroker`` receive the SAME instance."""
    return LinearCostModel()


@pytest.fixture
def market() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-05-11T16:00:00+00:00",
        tickers=("AAPL", "MSFT", "NVDA", "SPY"),
        prices=np.array([150.0, 300.0, 600.0, 500.0], dtype=np.float64),
    )


@pytest.fixture
def initial_state() -> PortfolioState:
    return PortfolioState.empty(initial_cash=1_000_000.0)


@pytest.fixture
def orders(
    market: MarketSnapshot,
    initial_state: PortfolioState,
) -> list[Order]:
    """Build a single ``list[Order]`` via ``RebalanceEngine`` over a
    hand-crafted ``AlphaSignal``.

    Per plan §5 PR3: PR3 is deliberately independent of PR1's toy AFML
    chain so a drift in the toy generator cannot silently break this
    cross-broker test. The signal here is small (4 tickers, all
    tradeable, gross 0.90) and the empty portfolio guarantees a
    non-vacuous order list (RebalanceEngine emits one ``BUY`` per
    ticker).

    The SAME ``Order`` list — including the same ``order_id`` UUIDs —
    is submitted to both brokers so per-order identity matches across
    the two paths. (Building one list per broker would regenerate UUIDs
    and complicate AC3.1's per-order tracing.)
    """
    sig = build_alpha_signal(
        tickers=market.tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, 0.002, 0.01, 0.001],
        upper=[0.04, 0.020, 0.05, 0.010],
        alpha=0.10,
        kelly_weights=[0.25, 0.20, 0.30, 0.15],  # gross = 0.90
    )
    engine = RebalanceEngine()
    return engine.rebalance(sig, initial_state, market, order_type=OrderType.MARKET)


@pytest.fixture
def paper_broker(cost_model: LinearCostModel) -> PaperBroker:
    return PaperBroker(cost_model=cost_model)


@pytest.fixture
def mock_ibkr_broker(cost_model: LinearCostModel) -> MockIBKRBroker:
    return MockIBKRBroker(cost_model=cost_model)


@pytest.fixture
def paper_run(
    paper_broker: PaperBroker,
    orders: list[Order],
    market: MarketSnapshot,
    initial_state: PortfolioState,
) -> tuple[list[Fill], PortfolioState]:
    fills = paper_broker.submit_orders(orders, market)
    state = initial_state
    for f in fills:
        state = state.apply(f)
    return fills, state


@pytest.fixture
def mock_ibkr_run(
    mock_ibkr_broker: MockIBKRBroker,
    orders: list[Order],
    market: MarketSnapshot,
    initial_state: PortfolioState,
) -> tuple[list[Fill], PortfolioState, int]:
    """Submit the SAME ``orders`` list through ``MockIBKRBroker`` with a
    ``patch.object`` spy wrapping ``order_to_ib_order`` so AC3.1 can
    count invocations without altering the function's behaviour.

    Returns ``(fills, final_state, mapping_call_count)``.
    """
    with patch.object(
        order_mapping,
        "order_to_ib_order",
        wraps=order_mapping.order_to_ib_order,
    ) as spy:
        fills = mock_ibkr_broker.submit_orders(orders, market)
        call_count = spy.call_count
    state = initial_state
    for f in fills:
        state = state.apply(f)
    return fills, state, call_count


# ─── AC3.1 — order_to_ib_order invoked for every Order ──────────────────
def test_ac3_1_order_to_ib_order_called_per_submitted_order(
    orders: list[Order],
    mock_ibkr_run: tuple[list[Fill], PortfolioState, int],
) -> None:
    """The real ``order_to_ib_order`` is invoked exactly once per
    submitted order on the MockIBKRBroker path. Counts come from
    ``patch.object(..., wraps=order_to_ib_order)``; the function still
    runs (``wraps`` delegates), so the assertion also exercises the
    real mapping body (Contract construction, type-table lookup, etc.).
    """
    assert len(orders) >= 1, (
        "fixture vacuous: RebalanceEngine returned no orders — check the "
        "hand-crafted AlphaSignal / NAV / min_trade_notional sizing."
    )
    _, _, calls = mock_ibkr_run
    assert calls == len(orders), (
        f"AC3.1 violation: order_to_ib_order called {calls} times, "
        f"expected {len(orders)} (one per submitted order)."
    )


# ─── AC3.2 — no live broker construction or env-var leak ────────────────
def test_ac3_2_no_live_broker_construction_or_env_leak() -> None:
    """Structural negatives. The module-load guard (top of this file)
    already fails the import if ``IBKR_PAPER_SMOKE='1'`` at collection
    time; this test re-asserts the same negative at run time (against
    a fixture mutating the env mid-run) and pins source-level negatives
    on ``IBKRConnection`` / ``IBKRBroker`` / ``ib_async.IB``
    construction.
    """
    # 1) Module-scope: no live-broker / connection names bound here.
    mod = sys.modules[__name__]
    forbidden_names = ("IBKR" + "Connection", "IBKR" + "Broker")
    in_scope = [n for n in forbidden_names if hasattr(mod, n)]
    assert not in_scope, f"AC3.2 violation: {in_scope} present in module scope"

    # 2) Source-grep: no construction expressions, no env-var mutations.
    src = Path(__file__).read_text()
    hits = [p for p in _AC3_2_FORBIDDEN_REGEXES if re.search(p, src)]
    assert not hits, f"AC3.2 violation: source matches forbidden regexes {hits}"

    # 3) Run-time AC3.6 cross-check: env var still not '1'.
    assert os.environ.get("IBKR_PAPER_SMOKE") != "1", (
        "AC3.6 violation: IBKR_PAPER_SMOKE='1' observed at test runtime"
    )


# ─── AC3.3 — final positions equal across brokers ───────────────────────
def test_ac3_3_final_positions_equal(
    paper_run: tuple[list[Fill], PortfolioState],
    mock_ibkr_run: tuple[list[Fill], PortfolioState, int],
) -> None:
    """``{ticker: signed_quantity}`` is identical on both paths. The
    plan AC3.3 explicitly scopes the equality to the quantity mapping
    (not the full ``Position`` object) — ``avg_cost`` is derived from
    ``fill_price`` and is also equal under a shared cost model, but the
    quantity contract is the load-bearing identity here."""
    _, paper_state = paper_run
    _, mock_state, _ = mock_ibkr_run
    paper_qty = {t: p.quantity for t, p in paper_state.positions.items()}
    mock_qty = {t: p.quantity for t, p in mock_state.positions.items()}
    assert paper_qty == mock_qty, (
        f"AC3.3 violation: paper positions {paper_qty} != mock IBKR positions {mock_qty}"
    )


# ─── AC3.4 — cash equal under shared cost model ─────────────────────────
def test_ac3_4_final_cash_equal_under_shared_cost_model(
    paper_run: tuple[list[Fill], PortfolioState],
    mock_ibkr_run: tuple[list[Fill], PortfolioState, int],
) -> None:
    """Cash equality holds because both brokers share the SAME
    ``LinearCostModel`` instance (same slippage, same commission). The
    1e-6 absolute tolerance accommodates float64 accumulation across
    the fill-application loop; PaperBroker and MockIBKRBroker produce
    identical fill-price and commission values per fill so the
    arithmetic agrees exactly in practice."""
    _, paper_state = paper_run
    _, mock_state, _ = mock_ibkr_run
    assert paper_state.cash == pytest.approx(mock_state.cash, abs=1e-6), (
        f"AC3.4 violation: paper cash {paper_state.cash} != mock IBKR cash {mock_state.cash}"
    )


# ─── AC3.5 — normalized fill projection equal ───────────────────────────
def test_ac3_5_normalized_fill_projection_equal(
    paper_run: tuple[list[Fill], PortfolioState],
    mock_ibkr_run: tuple[list[Fill], PortfolioState, int],
) -> None:
    """``sorted((ticker, signed_quantity, price) for f in fills)``
    equal across both paths. ``Fill.timestamp`` and ``Fill.commission``
    are excluded from the projection (matches AbstractBroker's CI-2 /
    CI-3 documented invariants); ``Fill.fill_id`` and ``Fill.metadata``
    are also implementation-specific and excluded. Sorting handles any
    fill-ordering difference between the two implementations."""
    paper_fills, _ = paper_run
    mock_fills, _, _ = mock_ibkr_run
    paper_proj = sorted((f.ticker, f.signed_quantity, f.price) for f in paper_fills)
    mock_proj = sorted((f.ticker, f.signed_quantity, f.price) for f in mock_fills)
    assert paper_proj == mock_proj, (
        f"AC3.5 violation: normalized fill projections differ.\n"
        f"  paper:    {paper_proj}\n"
        f"  mock IBKR: {mock_proj}"
    )


# ─── AC3.6 — env-var negative (positive side) ───────────────────────────
def test_ac3_6_ibkr_paper_smoke_not_one() -> None:
    """The module-load guard at the top of this file fails import when
    ``IBKR_PAPER_SMOKE=='1'`` at collection time. This test re-asserts
    the same negative once the module has loaded — defence-in-depth
    against a fixture or earlier test mutating the env mid-run."""
    assert os.environ.get("IBKR_PAPER_SMOKE") != "1"
