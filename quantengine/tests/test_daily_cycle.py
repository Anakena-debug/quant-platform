"""End-to-end smoke test for quantengine.runtime.daily_cycle.run_paper_cycle.

Exercises the full plane:
    SnapshotLoader → SignalArtifact → RebalanceEngine → RiskGate →
    PaperBroker → OrderTracker → ledger/journal

Without quantdata or pyarrow. Uses DataFrame-based snapshot + JSON-format
signal artifact so it runs in the minimal sandbox.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from quantengine.audit.journal import verify_chain
from quantengine.contracts.signal import build_alpha_signal
from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import DataFrameSnapshotLoader
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.order_state import OrderTracker
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.daily_cycle import PaperCycleResult, run_paper_cycle


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("AAPL", "2026-04-15", 199.0),
            ("AAPL", "2026-04-16", 200.0),
            ("AAPL", "2026-04-17", 201.0),
            ("MSFT", "2026-04-15", 395.0),
            ("MSFT", "2026-04-16", 400.0),
            ("MSFT", "2026-04-17", 402.0),
            ("NVDA", "2026-04-15", 98.0),
            ("NVDA", "2026-04-16", 100.0),
            ("NVDA", "2026-04-17", 102.0),
            ("SPY", "2026-04-15", 498.0),
            ("SPY", "2026-04-16", 500.0),
            ("SPY", "2026-04-17", 503.0),
        ],
        columns=["ticker", "session_date", "price"],
    )


def _write_signal(tmp_path: Path) -> SignalArtifact:
    sig = build_alpha_signal(
        tickers=("AAPL", "MSFT", "NVDA", "SPY"),
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, 0.002, 0.01, 0.001],
        upper=[0.04, 0.02, 0.05, 0.01],
        alpha=0.10,
        kelly_weights=[0.25, 0.20, 0.30, 0.15],
        timestamp="2026-04-17T16:00:00Z",
    )
    art = SignalArtifact(path=tmp_path / "sig-2026-04-17", fmt="json")
    art.write(sig, run_id="r-e2e-001", model_sha="e2e-test")
    return art


def _make_stack(state: PortfolioState, ledger: Ledger):
    snap_source = DataFrameSnapshotLoader(prices=_prices_frame())
    broker = PaperBroker(cost_model=LinearCostModel())
    tracker = OrderTracker(ledger=ledger)
    gate = RiskGate.default_us_equities(
        max_order_notional=500_000,
        max_gross_leverage=1.5,
        max_position_weight=0.40,
    )
    return snap_source, broker, tracker, gate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_single_cycle_produces_fills_and_updates_state():
    with tempfile.TemporaryDirectory() as td:
        artifact = _write_signal(Path(td))
        state = PortfolioState.empty(1_000_000.0)
        ledger = Ledger()
        snap_source, broker, tracker, gate = _make_stack(state, ledger)

        result = run_paper_cycle(
            pd.Timestamp("2026-04-17"),
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
        )

    assert isinstance(result, PaperCycleResult)
    assert result.n_orders > 0
    assert result.n_fills == len(result.orders_accepted)
    assert result.final_state is not state  # new immutable state
    # Journal verify on the full ledger hash chain.
    assert verify_chain(ledger.events(), result.journal_digest)


def test_two_cycle_continuity_of_state():
    """Day 1 opens long; Day 2 reuses final_state as opening book."""
    with tempfile.TemporaryDirectory() as td:
        artifact = _write_signal(Path(td))
        state = PortfolioState.empty(1_000_000.0)
        ledger = Ledger()
        snap_source, broker, tracker, gate = _make_stack(state, ledger)

        r1 = run_paper_cycle(
            pd.Timestamp("2026-04-16"),
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
        )
        r2 = run_paper_cycle(
            pd.Timestamp("2026-04-17"),
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=r1.final_state,  # carry over
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,  # same tracker → chained journal
        )

    # Ledger grew across both cycles.
    assert len(ledger) > 0
    # Day 2's final state descends from day 1's.
    assert r2.initial_state is r1.final_state
    # Hash chain remains valid after cycle 2.
    assert verify_chain(ledger.events(), r2.journal_digest)


def test_risk_gate_rejections_are_logged():
    """A crazy-tight cap forces rejections; these must land in the ledger."""
    with tempfile.TemporaryDirectory() as td:
        artifact = _write_signal(Path(td))
        state = PortfolioState.empty(1_000_000.0)
        ledger = Ledger()
        snap_source = DataFrameSnapshotLoader(prices=_prices_frame())
        broker = PaperBroker(cost_model=LinearCostModel())
        tracker = OrderTracker(ledger=ledger)
        # Absurdly tight fat-finger cap — every order should be rejected.
        gate = RiskGate.default_us_equities(
            max_order_notional=1.0,
            max_gross_leverage=1.5,
            max_position_weight=0.40,
        )
        result = run_paper_cycle(
            pd.Timestamp("2026-04-17"),
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
        )
    assert len(result.rejections) > 0
    assert result.n_fills == 0
    kinds = {e.kind for e in ledger.events()}
    # Lifecycle FSM: rejections must appear as SUBMITTED then REJECTED.
    assert "ORDER_SUBMITTED" in kinds
    assert "ORDER_REJECTED" in kinds


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_single_cycle_produces_fills_and_updates_state,
        test_two_cycle_continuity_of_state,
        test_risk_gate_rejections_are_logged,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nruntime.daily_cycle: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
