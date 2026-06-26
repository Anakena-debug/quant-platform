"""Smoke tests for quantengine.runtime.reconcile.

Covers:
    - No drift when internal == broker.
    - Cash drift above / below tolerance.
    - Quantity drift (same ticker, mismatched shares).
    - Cost drift (same qty, different avg cost).
    - UNKNOWN_TO_INTERNAL / UNKNOWN_TO_BROKER classification.
    - ignore_tickers whitelist.
    - assert_reconciled raises + logs RECONCILE event.
    - RECONCILE event always logged, even on success.
"""

from __future__ import annotations

from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.reconcile import (
    BrokerPosition,
    BrokerSnapshot,
    ReconciliationError,
    assert_reconciled,
    reconcile,
)


def _internal() -> PortfolioState:
    return PortfolioState(
        cash=10_000.0,
        positions={
            "AAPL": Position("AAPL", 100, 150.0),
            "MSFT": Position("MSFT", 50, 400.0),
        },
    )


def _broker_matching() -> BrokerSnapshot:
    return BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=10_000.0,
        positions=(
            BrokerPosition("AAPL", 100, 150.0),
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )


# ---------------------------------------------------------------------------
def test_no_drift_reports_ok():
    r = reconcile(_internal(), _broker_matching())
    assert r.ok is True
    assert r.position_drifts == ()
    assert abs(r.cash_drift_usd) < 1e-9


def test_cash_drift_within_tolerance_ok():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=10_000.50,  # 50¢ drift
        positions=(
            BrokerPosition("AAPL", 100, 150.0),
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    r = reconcile(internal, broker, tol_cash_usd=1.0)
    assert r.ok is True


def test_cash_drift_above_tolerance_flags():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=8_500.0,  # $1_500 drift
        positions=(
            BrokerPosition("AAPL", 100, 150.0),
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    r = reconcile(internal, broker, tol_cash_usd=1.0)
    assert r.ok is False
    assert abs(r.cash_drift_usd - (-1500.0)) < 1e-9


def test_quantity_drift_flagged():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=10_000.0,
        positions=(
            BrokerPosition("AAPL", 99, 150.0),  # off by one share
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    r = reconcile(internal, broker)
    assert not r.ok
    kinds = [d.kind for d in r.position_drifts]
    assert "QUANTITY" in kinds


def test_cost_drift_flagged():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=10_000.0,
        positions=(
            BrokerPosition("AAPL", 100, 150.50),  # 50¢ off per share
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    r = reconcile(internal, broker, tol_cost_usd=0.01)
    kinds = [d.kind for d in r.position_drifts]
    assert "COST" in kinds


def test_unknown_ticker_classification():
    internal = PortfolioState(
        cash=0.0,
        positions={"AAPL": Position("AAPL", 100, 150.0)},
    )
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=0.0,
        positions=(BrokerPosition("MSFT", 50, 400.0),),
    )
    r = reconcile(internal, broker)
    kinds_by_ticker = {d.ticker: d.kind for d in r.position_drifts}
    assert kinds_by_ticker["AAPL"] == "UNKNOWN_TO_BROKER"
    assert kinds_by_ticker["MSFT"] == "UNKNOWN_TO_INTERNAL"


def test_ignore_tickers_excludes_from_check():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=10_000.0,
        positions=(
            BrokerPosition("AAPL", 99, 150.0),  # drift, but whitelisted
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    r = reconcile(internal, broker, ignore_tickers=["AAPL"])
    assert r.ok is True
    assert "AAPL" in r.ignored_tickers


def test_assert_reconciled_raises_on_drift_and_logs():
    internal = _internal()
    broker = BrokerSnapshot(
        as_of="2026-04-17T16:30:00Z",
        cash=7_000.0,
        positions=(
            BrokerPosition("AAPL", 100, 150.0),
            BrokerPosition("MSFT", 50, 400.0),
        ),
    )
    ledger = Ledger()
    raised = False
    try:
        assert_reconciled(internal, broker, ledger=ledger, tol_cash_usd=1.0)
    except ReconciliationError:
        raised = True
    assert raised
    # RECONCILE event always logged.
    kinds = [e.kind for e in ledger.events()]
    assert "RECONCILE" in kinds
    reconcile_event = next(e for e in ledger.events() if e.kind == "RECONCILE")
    assert reconcile_event.payload["ok"] is False


def test_assert_reconciled_success_logs_ok_event():
    ledger = Ledger()
    result = assert_reconciled(_internal(), _broker_matching(), ledger=ledger)
    assert result.ok is True
    assert len(ledger) == 1
    assert ledger.events()[0].payload["ok"] is True


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_no_drift_reports_ok,
        test_cash_drift_within_tolerance_ok,
        test_cash_drift_above_tolerance_flags,
        test_quantity_drift_flagged,
        test_cost_drift_flagged,
        test_unknown_ticker_classification,
        test_ignore_tickers_excludes_from_check,
        test_assert_reconciled_raises_on_drift_and_logs,
        test_assert_reconciled_success_logs_ok_event,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nruntime.reconcile: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
