"""S27 PR3 — Realistic SignalArtifact → PaperBroker *no-trade* E2E.

Routes the realistic AFML chain's `SignalArtifact` (S27 PR2, commit
`93318bc`) through the canonical disk handoff into `run_daily_cycle` +
`PaperBroker`, asserting the **no-trade** runtime contract:

    realistic SignalArtifact
        -> SignalArtifact.read()
        -> run_daily_cycle(...)
            -> RebalanceEngine        -> zero Orders
                -> PaperBroker        -> zero Fills
                    -> PortfolioState (unchanged)
                        -> ledger / chain_digest (still valid)

Why no-trade?

The pinned realistic chain (Ridge(alpha=1.0) + pooled
SplitConformalRegressor(alpha=0.20, calibration_fraction=0.25, seed=0)
over `mom5/z20/vol20`, 1-day forward log-return label,
tradeable rule = `(lower > 0) | (upper < 0)`) was empirically observed
on 2026-05-11 to emit 0 tradeable tickers across the entire fallback
as_of sequence on DJ30 (0/30) and across the same dates on S&P 500
(0/~484–490). The median conformal PI half-width (~0.02–0.03) dominates
the max |expected_return| (~0.001–0.004) by 5×–20× independent of
universe size; breadth is not the bottleneck. S27 PR3 therefore proves
the runtime correctly handles the no-trade case rather than
manufacturing a trade. Trade-forcing — changing alpha, horizon,
features, model, conformal method, tradeability rule, or rewriting
`lower`/`upper` post-`predict()` — is forbidden by §4 and the §8
amendment; trade-producing realistic alpha calibration is deferred
to a later sprint.

Acceptance criteria covered (AC3.1–AC3.7 in §3 of the amended plan):

* AC3.1 — in-memory `AlphaSignal` is byte-equal to the disk-read
          instance on `expected_return`, `lower`, `upper`,
          `kelly_weights`, modulo reader-added `metadata` keys.
* AC3.2 — `AlphaSignal.tradeable.sum() == 0` for the pinned as_of
          under the unmodified PR2 chain (positive no-trade assertion).
* AC3.3 — `PaperCycleResult.orders_built` is empty.
* AC3.4 — `PaperCycleResult.fills` is empty.
* AC3.5 — final_state equals initial_state: same cash, no positions.
* AC3.6 — `chain_digest(events).digest` is 64-char lowercase hex and
          `verify_chain(events, digest)` returns True.
* AC3.7 — structural negative — module does not import or construct
          the live-broker classes; no env-var mutation of
          `IBKR_PAPER_SMOKE`.

Run:

    uv run --directory quantstrat pytest tests/test_s27_realistic_e2e_paper.py -x
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from research._realistic_panel import load_dj30_panel
from research.test_realistic_afml_signal_producer import (
    AS_OF,
    CONFORMAL_ALPHA,
    MODEL_SHA,
    N_DJ30,
    RUN_ID,
    run_realistic_afml,
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

# ─── Pinned configuration (sprint §6) ────────────────────────────────
INITIAL_CASH: float = 100_000.0


# ─── AC3.7 forbidden-pattern table ──────────────────────────────────
# Constructed via string concatenation so the table's source text does
# not match any pattern itself — mirrors S26 PR2's AC2.6 idiom and lets
# the source-grep run against the on-disk file without self-tripping.
_AC3_7_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "IBKR" + "Broker(",
    "IBKR" + "Connection(",
    "ib_async" + ".IB(",
    "from " + "ib_async" + " import IB",
    'os.environ["' + "IBKR_PAPER_SMOKE" + '"]',
    "os.environ['" + "IBKR_PAPER_SMOKE" + "']",
    'monkeypatch.setenv("' + "IBKR_PAPER_SMOKE" + '"',
    "monkeypatch.setenv('" + "IBKR_PAPER_SMOKE" + "'",
    'setenv("' + "IBKR_PAPER_SMOKE" + '"',
    "setenv('" + "IBKR_PAPER_SMOKE" + "'",
)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def realistic_output() -> dict[str, object]:
    """Deterministic realistic AFML output for the pinned as_of+seed."""
    return run_realistic_afml()


@pytest.fixture
def signal_artifact_dir(realistic_output: dict[str, object], tmp_path: Path) -> Path:
    """Write the realistic AlphaSignal to disk via the canonical producer.

    `fmt='json'` mirrors S26 PR2 and S27 PR2 — bit-exact float64
    round-trip; parquet would risk sub-ULP drift under AC3.1's
    byte-equality assertion.
    """
    out_dir = tmp_path / "signals" / "as_of=2024-12-31"
    write_alpha_signal(
        tickers=realistic_output["tickers"],  # type: ignore[arg-type]
        expected_return=realistic_output["expected_return"],  # type: ignore[arg-type]
        lower=realistic_output["lower"],  # type: ignore[arg-type]
        upper=realistic_output["upper"],  # type: ignore[arg-type]
        alpha=CONFORMAL_ALPHA,
        kelly_weights=realistic_output["kelly_weights"],  # type: ignore[arg-type]
        as_of=AS_OF,
        out_dir=out_dir,
        run_id=RUN_ID,
        model_sha=MODEL_SHA,
        fmt="json",
    )
    return out_dir


@pytest.fixture
def signal_artifact(signal_artifact_dir: Path) -> SignalArtifact:
    return SignalArtifact(path=signal_artifact_dir, fmt="json")


@pytest.fixture(scope="module")
def snapshot_source() -> DataFrameSnapshotLoader:
    """PIT-safe snapshot source backed by PR1's realistic DJ30 panel.

    Feeding the long-form `(ticker, session_date, price)` panel directly
    into `DataFrameSnapshotLoader` — no rename needed, the panel's
    column names already match the loader's defaults.
    """
    panel = load_dj30_panel()
    return DataFrameSnapshotLoader(prices=panel)


@pytest.fixture
def constraints() -> RebalanceConstraints:
    """Test-local `RebalanceConstraints` pinned inline per §6 (verbatim
    from S26 PR2, same `min_trade_notional=$100` floor)."""
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
    """Empty gate per §6 — keeps the no-trade contract isolated from
    default-cap drift."""
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
        AS_OF,
        snapshot_source=snapshot_source,
        signal_artifact=signal_artifact,
        state=initial_state,
        constraints=constraints,
        gate=risk_gate,
        broker=paper_broker,
        tracker=tracker,
        pull_broker_snapshot=None,
    )


# ─── AC3.1 — round-trip identity (realistic) ────────────────────────
def test_ac3_1_signal_artifact_round_trip_identity(
    realistic_output: dict[str, object], signal_artifact: SignalArtifact
) -> None:
    """In-memory AlphaSignal is byte-equal to disk-read; modulo
    reader-added metadata keys (run_id, model_sha)."""
    sig = signal_artifact.read()
    assert isinstance(sig, AlphaSignal)
    assert sig.tickers == realistic_output["tickers"]
    assert sig.n == N_DJ30
    np.testing.assert_array_equal(sig.expected_return, realistic_output["expected_return"])
    np.testing.assert_array_equal(sig.lower, realistic_output["lower"])
    np.testing.assert_array_equal(sig.upper, realistic_output["upper"])
    assert sig.kelly_weights is not None
    np.testing.assert_array_equal(sig.kelly_weights, realistic_output["kelly_weights"])
    assert sig.alpha == CONFORMAL_ALPHA


# ─── AC3.2 — no-trade signal contract ───────────────────────────────
def test_ac3_2_realistic_signal_is_no_trade(signal_artifact: SignalArtifact) -> None:
    """For the pinned as_of, the realistic chain produces zero tradeable
    tickers. This is the empirical finding from 2026-05-11 (see §8
    landed amendment); the test pins that finding as the contract."""
    sig = signal_artifact.read()
    n_tradeable = int(sig.tradeable.sum())
    assert n_tradeable == 0, (
        f"S27 PR3 pinned as_of {AS_OF.date()} produced {n_tradeable} tradeable tickers. "
        "Under the amended plan, the realistic chain is expected to be no-trade. "
        "If a future plan amendment delivers a trade-producing chain, AC3 must be re-amended; "
        "do NOT silently change alpha/features/model/horizon/conformal/tradeability to absorb a tradeable signal."
    )
    # Kelly weights must therefore all be zero — no per-ticker sizing leaked through.
    assert sig.kelly_weights is not None
    np.testing.assert_array_equal(sig.kelly_weights, np.zeros_like(sig.kelly_weights))


# ─── AC3.3 — zero orders emitted ────────────────────────────────────
def test_ac3_3_rebalance_emits_zero_orders(cycle_result: PaperCycleResult) -> None:
    """RebalanceEngine produces no Orders for a tradeable-empty signal."""
    assert cycle_result.n_orders == 0
    assert len(cycle_result.orders_built) == 0
    assert len(cycle_result.orders_accepted) == 0
    assert len(cycle_result.rejections) == 0


# ─── AC3.4 — zero fills landed ──────────────────────────────────────
def test_ac3_4_paper_broker_emits_zero_fills(cycle_result: PaperCycleResult) -> None:
    """PaperBroker receives zero orders; PaperCycleResult.fills is empty."""
    assert cycle_result.n_fills == 0
    assert len(cycle_result.fills) == 0


# ─── AC3.5 — PortfolioState unchanged ───────────────────────────────
def test_ac3_5_state_cash_unchanged(cycle_result: PaperCycleResult) -> None:
    """No fills means no cash deltas; final cash matches initial cash bit-exactly."""
    assert cycle_result.final_state.cash == cycle_result.initial_state.cash
    assert cycle_result.final_state.cash == INITIAL_CASH


def test_ac3_5_state_positions_unchanged(cycle_result: PaperCycleResult) -> None:
    """No fills means no position changes; both maps are empty here."""
    assert dict(cycle_result.final_state.positions) == dict(cycle_result.initial_state.positions)
    assert len(cycle_result.final_state.positions) == 0


# ─── AC3.6 — journal closes on no-trade ─────────────────────────────
def test_ac3_6_chain_digest_is_lowercase_hex(
    cycle_result: PaperCycleResult, tracker: OrderTracker
) -> None:
    """chain_digest is well-formed (64-char lowercase hex) and validates
    via verify_chain even on the no-trade path."""
    digest = cycle_result.journal_digest
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", digest), f"non-lowercase-hex digest: {digest!r}"

    events = tracker.ledger.events()
    recomputed = chain_digest(events).digest
    assert recomputed == digest
    assert verify_chain(events, digest) is True


# ─── AC3.7 — no live broker use ─────────────────────────────────────
def test_ac3_7_no_live_broker_construction_or_env_mutation() -> None:
    """Structural negatives, asserted by inspecting this module's source.

    Two complementary checks:
      1. No live-broker name leaked into module scope (no import).
      2. The on-disk source contains no construction expression or env-var
         mutation matching the forbidden patterns.
    """
    mod = sys.modules[__name__]
    forbidden_names = ("IBKR" + "Broker", "IBKR" + "Connection")
    in_scope = [n for n in forbidden_names if hasattr(mod, n)]
    assert not in_scope, f"AC3.7 violation: {in_scope} present in module scope"

    src = Path(__file__).read_text()
    hits = [s for s in _AC3_7_FORBIDDEN_SUBSTRINGS if s in src]
    assert not hits, f"AC3.7 violation: source contains forbidden substrings {hits}"


def test_ac3_7_broker_used_is_paper(paper_broker: PaperBroker) -> None:
    """Positive sanity check: the broker passed to run_daily_cycle is a
    PaperBroker instance."""
    assert type(paper_broker).__name__ == "PaperBroker"
