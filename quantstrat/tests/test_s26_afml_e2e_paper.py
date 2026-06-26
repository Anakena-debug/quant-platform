"""S26 PR2 — SignalArtifact-mediated PaperBroker E2E (offline wiring proof).

Routes the toy AFML chain's output through the canonical SignalArtifact
disk handoff into ``run_daily_cycle`` + ``PaperBroker``, proving the
full chain end-to-end without touching IBKR:

    toy AFML SignalArtifact
        -> SignalArtifact.read()
        -> run_daily_cycle(...)
            -> RebalanceEngine -> list[Order]
                -> PaperBroker -> list[Fill]
                    -> PortfolioState.apply
                        -> ledger / chain_digest

Framing:

* **Wiring proof, not alpha research.** The toy chain is deterministic
  and single-config; no parameter matrix, no claim about Sharpe / IC /
  hit-rate / any alpha-quality metric.
* **Offline.** No live or paper IBKR submission. PR2 must not import
  the IBKR broker / connection classes, must not construct an
  ``ib_async`` ``IB`` session, and must not set the IBKR paper-smoke
  env var. AC2.6 below makes that contract explicit.

Acceptance criteria covered (AC2.1–AC2.6 in §3 of the plan):

* AC2.1 — in-memory ``AlphaSignal`` is byte-equal to disk read,
          modulo reader-added metadata keys.
* AC2.2 — ``run_daily_cycle`` produces >= 1 ``Order`` on the toy universe;
          every order has nonzero signed quantity and trades a ticker in
          the toy universe.
* AC2.3 — every accepted order produces exactly one fill; every
          ``Fill.order_id`` traces to a submitted/accepted Order.
* AC2.4 — ``Fill.cash_delta == -(signed_quantity * price) - commission``
          for every fill; ``Sum Fill.cash_delta == final_state.cash -
          initial_state.cash`` to 1e-6 absolute tolerance; final
          positions consistent with aggregated signed quantities.
* AC2.5 — ``chain_digest(events).digest`` is a 64-char lowercase hex
          string; ``verify_chain(events, digest)`` returns True.
* AC2.6 — module does not import the live-broker classes, does not
          construct an ``ib_async`` ``IB`` session, and does not mutate
          the IBKR paper-smoke env var. Asserted by inspecting this
          module's source code (the plan's §3 AC2.6 makes the claim
          behavioural — *not* on module reachability — since indirect
          package-level imports may appear in ``sys.modules`` from
          unrelated test modules).

Run:

    uv run --directory quantstrat pytest tests/test_s26_afml_e2e_paper.py -x
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research._toy_afml import (
    CONFORMAL_ALPHA,
    N_TICKERS,
    SEED,
    TICKERS,
    run_toy_afml,
)
from quantcore.signals.producer import write_alpha_signal
from quantengine.audit.journal import chain_digest, verify_chain
from quantengine.contracts.signal import AlphaSignal
from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import DataFrameSnapshotLoader
from quantengine.execution.order_state import OrderTracker
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.daily_cycle import PaperCycleResult, run_daily_cycle

# ─── Pinned configuration ───────────────────────────────────────────────
AS_OF_ISO: str = "2026-05-11T16:00:00Z"
INITIAL_CASH: float = 100_000.0


# ─── AC2.6 forbidden-pattern table ──────────────────────────────────────
# Built via string concatenation so the table's source text does not
# itself match any pattern (avoids a self-trigger when the AC2.6 test
# greps this file's bytes). The Python parser concatenates the literals
# at compile time; the on-disk source still shows them split across the
# ``+`` operators, so a substring search against the source text cannot
# match the runtime-constructed needle.
_AC2_6_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    # Live-broker class construction
    "IBKR" + "Broker(",
    "IBKR" + "Connection(",
    # ib_async event-loop session construction (data classes Order / Stock
    # are PR3-only; ``IB`` is the loop-owning class forbidden in PR2)
    "ib_async" + ".IB(",
    "from " + "ib_async" + " import IB",
    # IBKR paper-smoke env-var mutation
    'os.environ["' + "IBKR_PAPER_SMOKE" + '"]',
    "os.environ['" + "IBKR_PAPER_SMOKE" + "']",
    'monkeypatch.setenv("' + "IBKR_PAPER_SMOKE" + '"',
    "monkeypatch.setenv('" + "IBKR_PAPER_SMOKE" + "'",
    'setenv("' + "IBKR_PAPER_SMOKE" + '"',
    "setenv('" + "IBKR_PAPER_SMOKE" + "'",
)


# ─── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def toy_output() -> dict[str, np.ndarray]:
    """Deterministic toy AFML output for the pinned seed.

    The chain is rerun per test invocation; PR1's AC1.1 already pins
    that the output is byte-equal across consecutive calls in the same
    process.
    """
    return run_toy_afml(seed=SEED)


@pytest.fixture
def signal_artifact_dir(toy_output: dict[str, np.ndarray], tmp_path: Path) -> Path:
    """Write the toy ``AlphaSignal`` to disk via the canonical producer.

    ``fmt='json'`` gives bit-exact float64 round-trip (mirrors PR1's
    AC1.2 choice; see ``quantcore.signals.producer.write_alpha_signal``
    and ``quantengine.data.signal.SignalArtifact.read``). Parquet would
    require ``pyarrow`` and risk a precision quirk under AC2.1's
    byte-equality assertion.
    """
    out_dir = tmp_path / "signals" / "as_of=2026-05-11"
    write_alpha_signal(
        tickers=TICKERS,
        expected_return=toy_output["expected_return"],
        lower=toy_output["lower"],
        upper=toy_output["upper"],
        alpha=CONFORMAL_ALPHA,
        kelly_weights=toy_output["kelly_weights"],
        as_of=AS_OF_ISO,
        out_dir=out_dir,
        run_id="toy-afml-s26-pr2",
        model_sha="ridge-split-conformal-toy-v1",
        fmt="json",
    )
    return out_dir


@pytest.fixture
def signal_artifact(signal_artifact_dir: Path) -> SignalArtifact:
    return SignalArtifact(path=signal_artifact_dir, fmt="json")


@pytest.fixture
def snapshot_source() -> DataFrameSnapshotLoader:
    """In-memory PIT source for the toy universe at ``AS_OF_ISO``.

    One row per ticker at ``session_date == as_of.normalize()`` (midnight
    on the as-of date), with nominal prices in ``[50, 200]`` USD so the
    toy chain's per-ticker kelly weight of ``0.1`` and a $100K NAV map
    to an order with notional well above the default $100
    ``min_trade_notional`` floor.

    The PIT filter compares ``session_date <= as_of``; the equality
    branch holds, so every ticker is included.
    """
    as_of_ts = pd.Timestamp(AS_OF_ISO)
    prices = np.linspace(50.0, 200.0, N_TICKERS, dtype=np.float64)
    df = pd.DataFrame(
        {
            "ticker": list(TICKERS),
            "session_date": [as_of_ts.normalize()] * N_TICKERS,
            "price": prices,
        }
    )
    return DataFrameSnapshotLoader(prices=df)


@pytest.fixture
def constraints() -> RebalanceConstraints:
    """Test-local ``RebalanceConstraints`` pinned inline.

    The toy chain emits ``kelly_weights in {-0.1, 0, +0.1}`` (only the
    naturally tradeable tickers are non-zero); with NAV = $100K, a
    tradeable long produces ~$10K notional, comfortably above the $100
    ``min_trade_notional`` floor. ``allow_short=False`` (default) clips
    short legs to zero — the long-only toy book matches PR1's framing.

    Pinned inline rather than calling the package default factory so the
    test's contract is local to this file and not coupled to any future
    drift in default knobs.
    """
    return RebalanceConstraints(
        cash_buffer=0.02,
        max_gross_leverage=1.0,
        max_turnover=1.0,
        min_trade_notional=100.0,
        allow_short=False,
        lot_size=1,
    )


@pytest.fixture
def risk_gate() -> RiskGate:
    """Empty gate — no checks.

    The wiring proof asserts behaviour through the rebalance engine,
    not through gate-side rejections. The default-US-equities factory
    would admit every toy order today, but pinning *no* checks here
    isolates AC2.2 / AC2.3 from any future tightening of the default
    cap configuration.
    """
    return RiskGate()


@pytest.fixture
def initial_state() -> PortfolioState:
    return PortfolioState.empty(initial_cash=INITIAL_CASH)


@pytest.fixture
def ledger() -> Ledger:
    return Ledger()


@pytest.fixture
def tracker(ledger: Ledger) -> OrderTracker:
    return OrderTracker(ledger=ledger)


@pytest.fixture
def paper_broker() -> PaperBroker:
    return PaperBroker()


@pytest.fixture
def cycle_result(
    signal_artifact: SignalArtifact,
    snapshot_source: DataFrameSnapshotLoader,
    initial_state: PortfolioState,
    constraints: RebalanceConstraints,
    risk_gate: RiskGate,
    paper_broker: PaperBroker,
    tracker: OrderTracker,
) -> PaperCycleResult:
    """Drive one daily cycle end-to-end. Single offline call; no reconcile."""
    return run_daily_cycle(
        pd.Timestamp(AS_OF_ISO),
        snapshot_source=snapshot_source,
        signal_artifact=signal_artifact,
        state=initial_state,
        constraints=constraints,
        gate=risk_gate,
        broker=paper_broker,
        tracker=tracker,
        pull_broker_snapshot=None,
    )


# ─── AC2.1 — round-trip identity ────────────────────────────────────────
def test_ac2_1_signal_artifact_round_trip_identity(
    toy_output: dict[str, np.ndarray], signal_artifact: SignalArtifact
) -> None:
    """In-memory ``AlphaSignal`` is byte-equal to the disk-read instance,
    modulo reader-added ``metadata`` keys (``run_id``, ``model_sha``)."""
    sig = signal_artifact.read()
    assert isinstance(sig, AlphaSignal)
    assert sig.tickers == TICKERS
    assert sig.n == N_TICKERS
    np.testing.assert_array_equal(sig.expected_return, toy_output["expected_return"])
    np.testing.assert_array_equal(sig.lower, toy_output["lower"])
    np.testing.assert_array_equal(sig.upper, toy_output["upper"])
    assert sig.kelly_weights is not None
    np.testing.assert_array_equal(sig.kelly_weights, toy_output["kelly_weights"])
    assert sig.alpha == CONFORMAL_ALPHA


# ─── AC2.2 — orders emitted ─────────────────────────────────────────────
def test_ac2_2_orders_emitted_on_toy_universe(cycle_result: PaperCycleResult) -> None:
    """``run_daily_cycle`` produces >= 1 ``Order``; every order trades a
    ticker in the toy universe and has nonzero signed quantity."""
    orders = cycle_result.orders_built
    assert len(orders) >= 1, "rebalance engine returned no orders"
    toy_set = set(TICKERS)
    for o in orders:
        assert o.ticker in toy_set, f"order on non-toy ticker {o.ticker!r}"
        assert o.quantity > 0
        assert o.signed_quantity != 0


# ─── AC2.3 — fills landed ───────────────────────────────────────────────
def test_ac2_3_fills_match_accepted_orders(cycle_result: PaperCycleResult) -> None:
    """Every accepted order yields exactly one ``Fill``; every fill
    traces to a submitted/accepted ``Order`` (no orphans, no inventions)."""
    fills = cycle_result.fills
    accepted = cycle_result.orders_accepted
    assert len(fills) >= 1, "PaperBroker emitted no fills"
    assert len(fills) == len(accepted), (
        f"fill / accepted mismatch: len(fills)={len(fills)} len(orders_accepted)={len(accepted)}"
    )
    by_id = {o.order_id: o for o in accepted}
    assert {f.order_id for f in fills} == set(by_id)
    for f in fills:
        o = by_id[f.order_id]
        assert f.ticker == o.ticker
        assert f.signed_quantity == o.signed_quantity


# ─── AC2.4 — bookkeeping identities ─────────────────────────────────────
def test_ac2_4_fill_cash_delta_identity(cycle_result: PaperCycleResult) -> None:
    """``Fill.cash_delta == -(signed_quantity * price) - commission``."""
    for f in cycle_result.fills:
        expected = -(f.signed_quantity * f.price) - f.commission
        assert f.cash_delta == pytest.approx(expected, abs=1e-9)


def test_ac2_4_state_cash_invariant(cycle_result: PaperCycleResult) -> None:
    """After applying all fills, ``final_state.cash - initial_state.cash
    == sum(Fill.cash_delta)`` to 1e-6 absolute tolerance on float64 cash."""
    expected_delta = sum(f.cash_delta for f in cycle_result.fills)
    actual_delta = cycle_result.final_state.cash - cycle_result.initial_state.cash
    assert actual_delta == pytest.approx(expected_delta, abs=1e-6)


def test_ac2_4_position_consistency(cycle_result: PaperCycleResult) -> None:
    """Final positions equal pre-trade positions (empty here) plus the
    aggregated signed fills per ticker. Net-zero tickers must not appear
    in the positions map."""
    expected_qty: dict[str, int] = {}
    for f in cycle_result.fills:
        expected_qty[f.ticker] = expected_qty.get(f.ticker, 0) + f.signed_quantity
    for ticker, qty in expected_qty.items():
        if qty == 0:
            assert ticker not in cycle_result.final_state.positions, (
                f"net-zero ticker {ticker} should not appear in positions"
            )
        else:
            assert ticker in cycle_result.final_state.positions, f"missing position for {ticker}"
            assert cycle_result.final_state.positions[ticker].quantity == qty, (
                f"qty mismatch for {ticker}: "
                f"expected {qty}, got {cycle_result.final_state.positions[ticker].quantity}"
            )


# ─── AC2.5 — journal closes ─────────────────────────────────────────────
def test_ac2_5_chain_digest_is_lowercase_hex(
    cycle_result: PaperCycleResult, tracker: OrderTracker
) -> None:
    """``chain_digest(events).digest`` is a 64-char lowercase hex string
    matching the cycle result's ``journal_digest``, and ``verify_chain``
    returns ``True`` against those same events."""
    digest = cycle_result.journal_digest
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", digest), f"non-lowercase-hex digest: {digest!r}"

    events = tracker.ledger.events()
    recomputed = chain_digest(events).digest
    assert recomputed == digest
    assert verify_chain(events, digest) is True


# ─── AC2.6 — no live broker use ─────────────────────────────────────────
def test_ac2_6_no_live_broker_construction_or_env_mutation() -> None:
    """Structural negatives, asserted by inspecting this module's source.

    AC2.6 — the test must
    not (a) import the live-broker classes, (b) construct an event-loop
    session over the IBKR adapter library, (c) mutate the paper-smoke
    env var. The plan explicitly says the assertion is on *behaviour*
    (no construction, no env mutation), not on ``sys.modules``
    reachability, since unrelated tests in the same pytest session may
    leave package-level imports of pure data classes in ``sys.modules``.

    Two complementary checks:
      1. The current module's namespace contains no broker / connection
         names — they were never imported here.
      2. The source file itself contains no construction expression or
         env-var assignment matching the forbidden patterns.
    """
    # 1) Module-scope check
    mod = sys.modules[__name__]
    forbidden_names = ("IBKR" + "Broker", "IBKR" + "Connection")
    in_scope = [n for n in forbidden_names if hasattr(mod, n)]
    assert not in_scope, f"AC2.6 violation: {in_scope} present in module scope"

    # 2) Source-grep
    src = Path(__file__).read_text()
    hits = [s for s in _AC2_6_FORBIDDEN_SUBSTRINGS if s in src]
    assert not hits, f"AC2.6 violation: source contains forbidden substrings {hits}"


def test_ac2_6_broker_used_is_paper(paper_broker: PaperBroker) -> None:
    """Positive sanity check: the broker actually instantiated and passed
    to ``run_daily_cycle`` is a ``PaperBroker`` instance (no live broker
    substituted upstream of this test)."""
    assert type(paper_broker).__name__ == "PaperBroker"
