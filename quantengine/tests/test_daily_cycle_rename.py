"""Tests for the PR4.0 rename refactor of ``run_paper_cycle`` →
``run_daily_cycle`` with optional pre-/post-trade reconcile.

The existing ``test_daily_cycle.py`` covers back-compat semantics
(``run_paper_cycle`` continues to work unchanged). This file pins the
NEW invariants introduced by PR4.0:

1. ``run_paper_cycle`` is a back-compat wrapper that delegates to
   ``run_daily_cycle`` with ``pull_broker_snapshot=None``.
2. ``run_daily_cycle`` with a non-None ``pull_broker_snapshot``
   triggers pre- AND post-trade reconcile (two RECONCILE events).
3. ``run_daily_cycle`` with ``pull_broker_snapshot=None`` emits
   ZERO RECONCILE events (matching pre-S22 ``run_paper_cycle``
   semantics).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

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
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime import daily_cycle as daily_cycle_mod
from quantengine.runtime.daily_cycle import (
    PaperCycleResult,
    run_daily_cycle,
    run_paper_cycle,
)
from quantengine.runtime.reconcile import BrokerPosition, BrokerSnapshot


# ---- Fixtures --------------------------------------------------------


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("AAPL", "2026-04-17", 200.0),
            ("MSFT", "2026-04-17", 400.0),
        ],
        columns=["ticker", "session_date", "price"],
    )


def _no_fill_artifact(tmp_path: Path) -> SignalArtifact:
    """Build a signal where every ticker is NOT tradeable (interval
    contains zero), so ``RebalanceEngine`` produces zero orders and
    the cycle's initial_state == final_state. Keeps the pre-/post-
    reconcile snapshots identical, simplifying tests 2 + 3.
    """
    sig = build_alpha_signal(
        tickers=("AAPL", "MSFT"),
        expected_return=[0.005, 0.005],
        lower=[-0.01, -0.01],  # interval contains 0 → not tradeable
        upper=[0.02, 0.02],
        alpha=0.10,
        kelly_weights=[0.0, 0.0],
        timestamp="2026-04-17T16:00:00Z",
    )
    art = SignalArtifact(path=tmp_path / "no-fill-sig", fmt="json")
    art.write(sig, run_id="r-rename-001", model_sha="rename-test")
    return art


def _state_to_snapshot(state: PortfolioState, as_of: str) -> BrokerSnapshot:
    return BrokerSnapshot(
        as_of=as_of,
        cash=state.cash,
        positions=tuple(
            BrokerPosition(ticker=p.ticker, quantity=p.quantity, avg_cost=p.avg_cost)
            for p in state.positions.values()
        ),
    )


def _make_stack(
    state: PortfolioState, ledger: Ledger
) -> tuple[DataFrameSnapshotLoader, PaperBroker, OrderTracker, RiskGate]:
    snap_source = DataFrameSnapshotLoader(prices=_prices_frame())
    broker = PaperBroker(cost_model=LinearCostModel())
    tracker = OrderTracker(ledger=ledger)
    gate = RiskGate.default_us_equities(
        max_order_notional=500_000,
        max_gross_leverage=1.5,
        max_position_weight=0.40,
    )
    return snap_source, broker, tracker, gate


_AS_OF = cast(pd.Timestamp, pd.Timestamp("2026-04-17"))


# ---- Test 1: back-compat wrapper delegates ---------------------------


def test_run_paper_cycle_back_compat_delegates_to_run_daily_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_paper_cycle`` delegates to ``run_daily_cycle`` with
    ``pull_broker_snapshot=None``.

    Monkeypatches ``run_daily_cycle`` in the daily_cycle module
    namespace; calls ``run_paper_cycle`` with a representative kwarg
    set; asserts the patched function was called once with the same
    kwargs plus ``pull_broker_snapshot=None``.
    """
    sentinel_result = object()

    captured_kwargs: dict[str, object] = {}

    def fake_run_daily_cycle(as_of, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["__as_of__"] = as_of
        return sentinel_result

    monkeypatch.setattr(daily_cycle_mod, "run_daily_cycle", fake_run_daily_cycle)

    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    snap_source, broker, tracker, gate = _make_stack(state, ledger)

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        result = run_paper_cycle(
            _AS_OF,
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
        )

    # Wrapper returned whatever the patched run_daily_cycle returned.
    assert result is sentinel_result
    # And the patched function was called with pull_broker_snapshot=None.
    assert captured_kwargs.get("pull_broker_snapshot") is None
    # All other kwargs flowed through.
    assert captured_kwargs.get("snapshot_source") is snap_source
    assert captured_kwargs.get("broker") is broker
    assert captured_kwargs.get("tracker") is tracker


# ---- Test 2: provider triggers pre + post reconcile ------------------


def test_run_daily_cycle_optional_snapshot_source() -> None:
    """A non-None ``pull_broker_snapshot`` triggers reconcile twice
    (pre-trade + post-trade) and the ledger records two RECONCILE
    events.
    """
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    snap_source, broker, tracker, gate = _make_stack(state, ledger)

    # No-fill cycle keeps initial_state == final_state, so the same
    # snapshot reconciles correctly both pre and post.
    snapshot = _state_to_snapshot(state, as_of="2026-04-17T16:00:00Z")
    provider_calls = [0]

    def provider() -> BrokerSnapshot:
        provider_calls[0] += 1
        return snapshot

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        result = run_daily_cycle(
            _AS_OF,
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
            pull_broker_snapshot=provider,
        )

    assert isinstance(result, PaperCycleResult)
    # No-fill semantic: initial_state == final_state.
    assert result.n_fills == 0
    # Provider was called twice: pre-trade + post-trade.
    assert provider_calls[0] == 2
    # Two RECONCILE events landed in the ledger.
    reconcile_events = [e for e in ledger.events() if e.kind == "RECONCILE"]
    assert len(reconcile_events) == 2


# ---- Test 3: no provider → zero reconcile ---------------------------


def test_run_daily_cycle_no_snapshot_source_skips_reconcile() -> None:
    """``pull_broker_snapshot=None`` → no reconcile, no RECONCILE events.

    Matches pre-S22 ``run_paper_cycle`` semantics exactly.
    """
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    snap_source, broker, tracker, gate = _make_stack(state, ledger)

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        result = run_daily_cycle(
            _AS_OF,
            snapshot_source=snap_source,
            signal_artifact=artifact,
            state=state,
            constraints=RebalanceConstraints(),
            gate=gate,
            broker=broker,
            tracker=tracker,
            pull_broker_snapshot=None,
        )

    assert isinstance(result, PaperCycleResult)
    # Zero RECONCILE events.
    reconcile_events = [e for e in ledger.events() if e.kind == "RECONCILE"]
    assert len(reconcile_events) == 0
