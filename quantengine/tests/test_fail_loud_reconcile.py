"""s89 — fail-closed must also fail-LOUD (F24 third lesson).

A pre-trade reconcile refusal must persist a FAILED run row (with the reason and the
already-journaled RECONCILE drift event) so absence-of-trades is forever distinguishable from
absence-of-attempts. The 2026-06-05 phase-C refusal left ZERO rows — these tests pin that gap
shut, including the s79 lesson (close -> reopen) and state-neutrality of the FAILED row.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantengine.contracts.signal import build_alpha_signal
from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import DataFrameSnapshotLoader
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.order_state import OrderTracker
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.daily_cycle import run_daily_cycle
from quantengine.runtime.reconcile import BrokerSnapshot, ReconciliationError
from quantengine.runtime.state_store import DuckDBStore


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("AAPL", "2026-04-16", 200.0),
            ("AAPL", "2026-04-17", 201.0),
            ("MSFT", "2026-04-16", 400.0),
            ("MSFT", "2026-04-17", 402.0),
        ],
        columns=["ticker", "session_date", "price"],
    )


def _write_signal(tmp_path: Path) -> SignalArtifact:
    sig = build_alpha_signal(
        tickers=("AAPL", "MSFT"),
        expected_return=[0.02, 0.01],
        lower=[0.005, 0.002],
        upper=[0.04, 0.02],
        alpha=0.10,
        kelly_weights=[0.25, 0.20],
        timestamp="2026-04-17T16:00:00Z",
    )
    art = SignalArtifact(path=tmp_path / "sig", fmt="json")
    art.write(sig, run_id="r-s89-001", model_sha="s89-test")
    return art


def _stack(ledger: Ledger):
    return (
        DataFrameSnapshotLoader(prices=_prices_frame()),
        PaperBroker(cost_model=LinearCostModel()),
        OrderTracker(ledger=ledger),
        RiskGate.default_us_equities(
            max_order_notional=500_000,
            max_gross_leverage=1.5,
            max_position_weight=0.40,
        ),
    )


def _run(tmp: Path, state: PortfolioState, store, snapshot: BrokerSnapshot | None):
    ledger = Ledger()
    snap_source, broker, tracker, gate = _stack(ledger)
    return run_daily_cycle(
        pd.Timestamp("2026-04-17"),
        snapshot_source=snap_source,
        signal_artifact=_write_signal(tmp),
        state=state,
        constraints=RebalanceConstraints(),
        gate=gate,
        broker=broker,
        tracker=tracker,
        pull_broker_snapshot=(lambda: snapshot) if snapshot is not None else None,
        store=store,
        metadata={"stage": "s89-test"},
    )


def test_pre_trade_refusal_persists_failed_row_state_neutral():
    opening = PortfolioState(
        cash=1_000_000.0,
        positions={"AAPL": Position("AAPL", 10, 100.0)},
        realized_pnl=0.0,
        total_commission=0.0,
    )
    drifting = BrokerSnapshot(as_of="2026-04-17T09:30:00", cash=900_000.0, positions=())
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "book.duckdb"
        store = DuckDBStore(path=str(db), read_only=False)
        with pytest.raises(ReconciliationError):
            _run(Path(td), opening, store, drifting)

        runs = store.list_runs()
        assert len(runs) == 1
        md = json.loads(runs.iloc[0]["metadata"])
        assert md["status"] == "FAILED_PRE_TRADE_RECONCILE"
        assert "DRIFT" in md["reason"]
        assert md["stage"] == "s89-test"  # caller metadata preserved on the FAILED row
        # state-neutral: the row carries the UNCHANGED opening book
        assert runs.iloc[0]["final_cash"] == pytest.approx(1_000_000.0)
        pos = store.load_run(runs.iloc[0]["run_id"])["positions"]
        assert len(pos) == 1
        assert pos.iloc[0]["ticker"] == "AAPL"
        assert int(pos.iloc[0]["quantity"]) == 10
        # the journaled RECONCILE drift event persisted (n_events > 0, digest present)
        assert int(runs.iloc[0]["n_events"]) > 0
        assert runs.iloc[0]["journal_digest"]

        # s79 lesson: the record survives close -> reopen
        store.close()
        store2 = DuckDBStore(path=str(db), read_only=True)
        runs2 = store2.list_runs()
        assert len(runs2) == 1
        assert json.loads(runs2.iloc[0]["metadata"])["status"] == "FAILED_PRE_TRADE_RECONCILE"
        store2.close()


def test_success_path_rows_carry_status_ok():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "book.duckdb"
        store = DuckDBStore(path=str(db), read_only=False)
        result = _run(Path(td), PortfolioState.empty(1_000_000.0), store, snapshot=None)
        assert result.n_fills > 0
        runs = store.list_runs()
        assert len(runs) == 1
        assert json.loads(runs.iloc[0]["metadata"])["status"] == "OK"


# NOTE: a "matching static snapshot trades through" variant is deliberately absent — the
# cycle re-pulls the snapshot POST-trade, so a static mock that matched pre-trade drifts
# after fills by construction. Post-trade reconcile semantics are out of s89 scope; the
# trading path itself is covered by test_daily_cycle.py.
