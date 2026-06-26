"""Tests for ``run_ibkr_paper_cycle`` (PR4.1).

Covers the IBKR-specific glue around ``run_daily_cycle``:
- RTH gate (outside-RTH and holiday paths)
- Layer-2 paper-account assertion at entry (via ``managedAccounts()``)
- Bracketed pre-/post-trade reconcile via the snapshot provider
- Connection scope (disconnect on exit, even on exception)
- Hard-fail-on-mid-cycle-disconnect (Phase 3 limitation)
- Persistence + journal-digest reproducibility
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock
from uuid import UUID

import pandas as pd
import pytest

from quantengine.contracts.signal import build_alpha_signal
from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import DataFrameSnapshotLoader, SnapshotSource
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.ibkr.broker import IBKRBroker
from quantengine.execution.ibkr.config import TimeoutPolicy
from quantengine.execution.ibkr.connection import IBKRConnection
from quantengine.execution.order_state import OrderTracker
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.daily_cycle import PaperCycleResult
from quantengine.runtime.ibkr_daily_cycle import run_ibkr_paper_cycle
from quantengine.runtime.reconcile import ReconciliationError


# ---- Test fixtures ---------------------------------------------------


@dataclass
class _FakeAV:
    tag: str
    value: str
    currency: str = "USD"


@dataclass
class _FakeContract:
    symbol: str
    secType: str = "STK"
    currency: str = "USD"


@dataclass
class _FakePortfolioItem:
    contract: _FakeContract
    position: float
    averageCost: float


def _good_account_values(cash: float = 1_000_000.0) -> list[_FakeAV]:
    return [
        _FakeAV(tag="TotalCashValue", value=str(cash)),
        _FakeAV(tag="NetLiquidation", value=str(cash)),
        _FakeAV(tag="BuyingPower", value=str(cash * 2)),
    ]


def _make_connection_with_mock_ib(
    *,
    managed_accounts: tuple[str, ...] = ("DU123",),
    account_values: list[_FakeAV] | None = None,
    portfolio_items: list[_FakePortfolioItem] | None = None,
) -> tuple[IBKRConnection, MagicMock]:
    """Build an IBKRConnection wrapping a mocked IB. Skip the connect()
    flow by directly setting the internal state — tests exercise only
    the cycle, not the connect path.
    """
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True
    mock_ib.managedAccounts.return_value = list(managed_accounts)
    mock_ib.accountValues.return_value = (
        account_values if account_values is not None else _good_account_values()
    )
    mock_ib.portfolio.return_value = portfolio_items if portfolio_items is not None else []

    conn = IBKRConnection()
    conn._ib = mock_ib
    conn._connected = True
    return conn, mock_ib


def _prices_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("AAPL", "2026-11-23", 200.0),
            ("MSFT", "2026-11-23", 400.0),
            ("AAPL", "2026-04-17", 200.0),
            ("MSFT", "2026-04-17", 400.0),
        ],
        columns=["ticker", "session_date", "price"],
    )


def _no_fill_artifact(tmp_path: Path) -> SignalArtifact:
    """Not-tradeable signal → no orders → no fills → state unchanged.

    Allows pre/post snapshots to match without dry-run computation.
    """
    sig = build_alpha_signal(
        tickers=("AAPL", "MSFT"),
        expected_return=[0.005, 0.005],
        lower=[-0.01, -0.01],  # interval contains 0 → not tradeable
        upper=[0.02, 0.02],
        alpha=0.10,
        kelly_weights=[0.0, 0.0],
        timestamp="2026-11-23T16:00:00Z",
    )
    art = SignalArtifact(path=tmp_path / "no-fill", fmt="json")
    art.write(sig, run_id="r-pr4-1", model_sha="pr4-1-test")
    return art


def _tradeable_artifact(tmp_path: Path) -> SignalArtifact:
    """All-tradeable signal → produces orders + fills."""
    sig = build_alpha_signal(
        tickers=("AAPL", "MSFT"),
        expected_return=[0.02, 0.01],
        lower=[0.005, 0.002],
        upper=[0.04, 0.02],
        alpha=0.10,
        kelly_weights=[0.25, 0.20],
        timestamp="2026-11-23T16:00:00Z",
    )
    art = SignalArtifact(path=tmp_path / "tradeable", fmt="json")
    art.write(sig, run_id="r-pr4-1-trd", model_sha="pr4-1-test")
    return art


class _CycleKwargs(TypedDict, total=False):
    snapshot_source: SnapshotSource
    signal_artifact: SignalArtifact
    state: PortfolioState
    constraints: RebalanceConstraints
    gate: RiskGate
    broker: AbstractBroker
    tracker: OrderTracker
    connection: IBKRConnection
    account: str
    store: Any


def _make_default_kwargs(
    *,
    state: PortfolioState,
    ledger: Ledger,
    artifact: SignalArtifact,
    broker_cls: type = PaperBroker,
    connection: IBKRConnection,
    account: str = "DU123",
    timeouts: TimeoutPolicy | None = None,
) -> _CycleKwargs:
    snap_source = DataFrameSnapshotLoader(prices=_prices_frame())
    broker: AbstractBroker
    if broker_cls is PaperBroker:
        broker = PaperBroker(cost_model=LinearCostModel())
    elif broker_cls is IBKRBroker:
        broker = IBKRBroker(connection=connection, timeouts=timeouts or TimeoutPolicy())
    else:
        raise ValueError(f"unsupported broker_cls: {broker_cls}")

    tracker = OrderTracker(ledger=ledger)
    gate = RiskGate.default_us_equities(
        max_order_notional=500_000,
        max_gross_leverage=1.5,
        max_position_weight=0.40,
    )
    return _CycleKwargs(
        snapshot_source=snap_source,
        signal_artifact=artifact,
        state=state,
        constraints=RebalanceConstraints(),
        gate=gate,
        broker=broker,
        tracker=tracker,
        connection=connection,
        account=account,
    )


# A clean RTH timestamp on a known session day (Monday 2026-11-23, 15:00 ET).
_RTH_AS_OF = cast(pd.Timestamp, pd.Timestamp("2026-11-23 15:00:00"))


# ---- Test 1: reconciles before AND after -----------------------------


def test_ibkr_daily_cycle_reconciles_before_and_after() -> None:
    """Happy path: pre AND post reconcile pass; ledger has 2 RECONCILE events."""
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    conn, _mock_ib = _make_connection_with_mock_ib()

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state, ledger=ledger, artifact=artifact, connection=conn
        )
        result = run_ibkr_paper_cycle(_RTH_AS_OF, **kwargs)
    assert isinstance(result, PaperCycleResult)
    assert result.n_fills == 0  # not-tradeable signal
    reconcile_events = [e for e in ledger.events() if e.kind == "RECONCILE"]
    assert len(reconcile_events) == 2  # pre + post


# ---- Test 2: aborts on pre-trade drift ------------------------------


def test_ibkr_daily_cycle_aborts_on_pre_trade_drift() -> None:
    """Pre-trade snapshot drifts from internal state → ReconciliationError.

    No orders are submitted — the cycle aborts before order construction.
    """
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    # Mock: broker reports 100 AAPL but internal state has none.
    drift_items = [
        _FakePortfolioItem(_FakeContract("AAPL"), position=100, averageCost=150.0),
    ]
    conn, mock_ib = _make_connection_with_mock_ib(portfolio_items=drift_items)

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state, ledger=ledger, artifact=artifact, connection=conn
        )
        with pytest.raises(ReconciliationError):
            run_ibkr_paper_cycle(_RTH_AS_OF, **kwargs)
    # No broker orders submitted: PaperBroker.submit_orders not exercised
    # because the cycle raised inside the pre-trade reconcile, before
    # any orders were built. The mock IB's portfolio() was called once
    # (for the pre-trade snapshot).
    assert mock_ib.portfolio.call_count == 1


# ---- Test 3: aborts outside RTH --------------------------------------


def test_ibkr_daily_cycle_aborts_outside_rth() -> None:
    """as_of at 03:00 ET on a session day → RuntimeError before any IB call."""
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    conn, mock_ib = _make_connection_with_mock_ib()

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state, ledger=ledger, artifact=artifact, connection=conn
        )
        # 03:00 ET on Monday 2026-11-23 — before market open.
        too_early = cast(pd.Timestamp, pd.Timestamp("2026-11-23 03:00:00"))
        with pytest.raises(RuntimeError, match="outside RTH"):
            run_ibkr_paper_cycle(too_early, **kwargs)
    # No IB call: cycle aborted at the RTH gate before connection scope.
    mock_ib.managedAccounts.assert_not_called()
    mock_ib.accountValues.assert_not_called()
    mock_ib.portfolio.assert_not_called()


# ---- Test 4: aborts on holiday ---------------------------------------


def test_ibkr_daily_cycle_aborts_on_holiday() -> None:
    """as_of on Thanksgiving Thursday 2026-11-26 → RuntimeError before any IB call.

    Avoids 2026-07-04 (Saturday in 2026), which would exercise the
    weekend path rather than the holiday path.
    """
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    conn, mock_ib = _make_connection_with_mock_ib()

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state, ledger=ledger, artifact=artifact, connection=conn
        )
        # Thanksgiving Thursday — unambiguous weekday holiday.
        thanksgiving = cast(pd.Timestamp, pd.Timestamp("2026-11-26 15:00:00"))
        with pytest.raises(RuntimeError, match="not a NYSE trading session"):
            run_ibkr_paper_cycle(thanksgiving, **kwargs)
    # No IB call: cycle aborted at the RTH gate before connection scope.
    mock_ib.managedAccounts.assert_not_called()


# ---- Test 5: journal digest reproducibility --------------------------


def test_ibkr_daily_cycle_writes_journal_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive runs with byte-identical inputs produce the same digest.

    Implementation: ``uuid.uuid4`` is monkeypatched to a deterministic
    counter so that Order/Fill IDs are reproducible across runs.
    Without that, each ``uuid4()`` call returns a different random
    value and digests trivially differ.
    """

    def make_uuid_factory(start: int = 1):
        counter = [start]

        def fake() -> UUID:
            counter[0] += 1
            return UUID(int=counter[0])

        return fake

    def run_one() -> str:
        state = PortfolioState.empty(1_000_000.0)
        ledger = Ledger()
        conn, _ = _make_connection_with_mock_ib()
        with tempfile.TemporaryDirectory() as td:
            artifact = _no_fill_artifact(Path(td))
            kwargs = _make_default_kwargs(
                state=state, ledger=ledger, artifact=artifact, connection=conn
            )
            result = run_ibkr_paper_cycle(_RTH_AS_OF, **kwargs)
        return result.journal_digest

    # Run 1.
    monkeypatch.setattr(uuid, "uuid4", make_uuid_factory())
    digest_1 = run_one()
    assert isinstance(digest_1, str)
    assert len(digest_1) == 64  # SHA-256 hex.

    # Run 2 with a fresh deterministic counter — identical sequence.
    monkeypatch.setattr(uuid, "uuid4", make_uuid_factory())
    digest_2 = run_one()

    assert digest_1 == digest_2


# ---- Test 6: persists to store ---------------------------------------


def test_ibkr_daily_cycle_persists_to_store_when_supplied() -> None:
    """``store.save_run`` is called once when ``store`` is provided."""
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    conn, _ = _make_connection_with_mock_ib()
    mock_store = MagicMock()

    with tempfile.TemporaryDirectory() as td:
        artifact = _no_fill_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state, ledger=ledger, artifact=artifact, connection=conn
        )
        kwargs["store"] = mock_store
        result = run_ibkr_paper_cycle(_RTH_AS_OF, **kwargs)
    assert isinstance(result, PaperCycleResult)
    mock_store.save_run.assert_called_once()
    call_kwargs = mock_store.save_run.call_args.kwargs
    # Metadata propagated through.
    assert "metadata" in call_kwargs


# ---- Test 7: disconnect mid-cycle hard-fails ------------------------


def test_ibkr_daily_cycle_disconnect_mid_cycle_hard_fails() -> None:
    """``ib.placeOrder`` raises ``ConnectionError`` mid-cycle → propagates.

    Uses ``IBKRBroker`` (not ``PaperBroker``) so the placeOrder code
    path is exercised. The ``with connection:`` scope ensures
    ``disconnect()`` runs in cleanup before the exception bubbles up.
    """
    state = PortfolioState.empty(1_000_000.0)
    ledger = Ledger()
    conn, mock_ib = _make_connection_with_mock_ib()
    # ib.placeOrder raises mid-cycle (after pre-trade reconcile passes).
    mock_ib.placeOrder.side_effect = ConnectionError("simulated mid-cycle socket drop")

    with tempfile.TemporaryDirectory() as td:
        artifact = _tradeable_artifact(Path(td))
        kwargs = _make_default_kwargs(
            state=state,
            ledger=ledger,
            artifact=artifact,
            broker_cls=IBKRBroker,
            connection=conn,
        )
        with pytest.raises(ConnectionError, match="mid-cycle socket drop"):
            run_ibkr_paper_cycle(_RTH_AS_OF, **kwargs)
    # Connection's __exit__ called disconnect on the underlying IB.
    mock_ib.disconnect.assert_called_once()
    # Pre-trade reconcile DID happen (1 RECONCILE event).
    reconcile_events = [e for e in ledger.events() if e.kind == "RECONCILE"]
    assert len(reconcile_events) == 1
