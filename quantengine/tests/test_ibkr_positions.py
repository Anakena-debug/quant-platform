"""Tests for ``pull_broker_snapshot`` — the IBKR position adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from quantengine.execution.ibkr.positions import pull_broker_snapshot
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import Position, PortfolioState
from quantengine.runtime.reconcile import (
    BrokerSnapshot,
    ReconciliationError,
    assert_reconciled,
)


# ---- Test fixtures ---------------------------------------------------


@dataclass
class _FakeAV:
    """Stand-in for ib_async ``AccountValue``."""

    tag: str
    value: str
    currency: str = "USD"


@dataclass
class _FakeContract:
    """Stand-in for ib_async ``Contract``."""

    symbol: str
    secType: str = "STK"
    currency: str = "USD"


@dataclass
class _FakePortfolioItem:
    """Stand-in for ib_async ``PortfolioItem``."""

    contract: _FakeContract
    position: float
    averageCost: float


def _good_account_values(
    cash: float = 100_000.0,
    nl: float = 105_000.0,
    bp: float = 200_000.0,
) -> list[_FakeAV]:
    return [
        _FakeAV(tag="TotalCashValue", value=str(cash)),
        _FakeAV(tag="NetLiquidation", value=str(nl)),
        _FakeAV(tag="BuyingPower", value=str(bp)),
    ]


def _make_mock_ib(
    account_values: list[_FakeAV] | None = None,
    portfolio_items: list[Any] | None = None,
) -> MagicMock:
    mock_ib = MagicMock()
    mock_ib.accountValues.return_value = (
        account_values if account_values is not None else _good_account_values()
    )
    mock_ib.portfolio.return_value = portfolio_items if portfolio_items is not None else []
    return mock_ib


# ---- accountValues → cash --------------------------------------------


def test_pull_broker_snapshot_maps_account_values_to_cash() -> None:
    """``TotalCashValue`` (USD) is read into ``BrokerSnapshot.cash``."""
    mock_ib = _make_mock_ib(account_values=_good_account_values(cash=12345.67))
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")
    assert isinstance(snap, BrokerSnapshot)
    assert snap.cash == 12345.67
    mock_ib.accountValues.assert_called_once_with("DU123")


# ---- portfolio() → BrokerPosition tuple -----------------------------


def test_pull_broker_snapshot_maps_portfolio_items_to_positions() -> None:
    """3 STK + USD portfolio items → 3 BrokerPosition entries with right fields."""
    items = [
        _FakePortfolioItem(_FakeContract("AAPL"), position=10, averageCost=150.0),
        _FakePortfolioItem(_FakeContract("MSFT"), position=20, averageCost=300.0),
        _FakePortfolioItem(_FakeContract("NVDA"), position=5, averageCost=600.0),
    ]
    mock_ib = _make_mock_ib(portfolio_items=items)
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")
    assert len(snap.positions) == 3
    by_ticker = {p.ticker: p for p in snap.positions}
    assert by_ticker["AAPL"].quantity == 10
    assert by_ticker["AAPL"].avg_cost == 150.0
    assert by_ticker["MSFT"].quantity == 20
    assert by_ticker["MSFT"].avg_cost == 300.0
    assert by_ticker["NVDA"].quantity == 5
    assert by_ticker["NVDA"].avg_cost == 600.0
    mock_ib.portfolio.assert_called_once_with("DU123")


# ---- Sign convention -------------------------------------------------


def test_pull_broker_snapshot_short_position_negative_quantity_positive_avg_cost() -> None:
    """Short position: ``quantity`` is negative; ``avg_cost`` STAYS POSITIVE.

    Defends against silent firmware sign-flips on ``averageCost``
    that would let reconciliation drift go undetected. Per IBKR
    firmware (verified against state.py:17-21 + the ``Position``
    dataclass convention): ``position`` is signed by direction;
    ``averageCost`` is always positive (it's the magnitude of the
    weighted cost basis per share).
    """
    items = [
        _FakePortfolioItem(_FakeContract("AAPL"), position=-100, averageCost=190.50),
    ]
    mock_ib = _make_mock_ib(portfolio_items=items)
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")
    assert len(snap.positions) == 1
    pos = snap.positions[0]
    assert pos.quantity == -100  # signed (short → negative)
    assert pos.avg_cost == 190.50  # POSITIVE, NOT -190.50
    assert pos.avg_cost > 0


# ---- as_of timestamp -------------------------------------------------


def test_pull_broker_snapshot_records_as_of_timestamp() -> None:
    """The ``as_of`` argument is preserved verbatim in the snapshot."""
    mock_ib = _make_mock_ib()
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-11-26T15:00:00-05:00")
    assert snap.as_of == "2026-11-26T15:00:00-05:00"


# ---- accountValues sanity --------------------------------------------


def test_pull_broker_snapshot_records_account_values_sanity() -> None:
    """``NetLiquidation`` + ``BuyingPower`` (USD) are required for the call to succeed.

    Defends against silent API-field renames in IBKR firmware updates.
    All three tags (``TotalCashValue``, ``NetLiquidation``,
    ``BuyingPower``) must be present and finite; missing any one
    raises ``ValueError`` naming the missing tag.
    """
    # All three present + finite → success.
    mock_ib = _make_mock_ib(account_values=_good_account_values(cash=100.0, nl=200.0, bp=400.0))
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")
    assert snap.cash == 100.0

    # Missing NetLiquidation → ValueError naming it.
    bad_no_nl = [
        _FakeAV(tag="TotalCashValue", value="100"),
        _FakeAV(tag="BuyingPower", value="400"),
    ]
    mock_ib_no_nl = _make_mock_ib(account_values=bad_no_nl)
    with pytest.raises(ValueError, match="NetLiquidation"):
        pull_broker_snapshot(mock_ib_no_nl, "DU123", as_of="...")

    # Missing BuyingPower → ValueError naming it.
    bad_no_bp = [
        _FakeAV(tag="TotalCashValue", value="100"),
        _FakeAV(tag="NetLiquidation", value="200"),
    ]
    mock_ib_no_bp = _make_mock_ib(account_values=bad_no_bp)
    with pytest.raises(ValueError, match="BuyingPower"):
        pull_broker_snapshot(mock_ib_no_bp, "DU123", as_of="...")

    # Non-finite NetLiquidation → ValueError.
    bad_nl_nan = [
        _FakeAV(tag="TotalCashValue", value="100"),
        _FakeAV(tag="NetLiquidation", value="nan"),
        _FakeAV(tag="BuyingPower", value="400"),
    ]
    mock_ib_nan = _make_mock_ib(account_values=bad_nl_nan)
    with pytest.raises(ValueError, match="non-finite"):
        pull_broker_snapshot(mock_ib_nan, "DU123", as_of="...")


# ---- Filter non-STK / non-USD ---------------------------------------


def test_pull_broker_snapshot_filters_non_stk_securities() -> None:
    """OPT, FUT, CASH (forex) items are silently dropped; STK+USD is kept."""
    items = [
        _FakePortfolioItem(
            _FakeContract("AAPL", secType="STK"),
            position=10,
            averageCost=150.0,
        ),
        _FakePortfolioItem(
            _FakeContract("AAPL_OPT", secType="OPT"),
            position=2,
            averageCost=5.0,
        ),
        _FakePortfolioItem(
            _FakeContract("ESH26", secType="FUT"),
            position=1,
            averageCost=4500.0,
        ),
        _FakePortfolioItem(
            _FakeContract("EUR", secType="CASH", currency="USD"),
            position=10000,
            averageCost=1.08,
        ),
        # Non-USD STK also filtered.
        _FakePortfolioItem(
            _FakeContract("BMW", secType="STK", currency="EUR"),
            position=50,
            averageCost=120.0,
        ),
    ]
    mock_ib = _make_mock_ib(portfolio_items=items)
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="...")

    # Only the STK + USD item survives.
    assert len(snap.positions) == 1
    assert snap.positions[0].ticker == "AAPL"


# ---- Base-currency-aware sanity-tag filter --------------------------


def test_pull_broker_snapshot_accepts_eur_base_account_when_no_usd() -> None:
    """An EUR-base paper account (no USD sanity-tag rows) is accepted.

    Regression for the load-bearing PR5 smoke against DUM268500 (EUR-
    base): the original USD-only filter raised ValueError because no
    USD rows existed for TotalCashValue / NetLiquidation / BuyingPower.
    The fix preserves the field-rename defense (all three tags must be
    present and finite) while allowing a non-USD currency when no USD
    set is complete.
    """
    eur_values = [
        _FakeAV(tag="TotalCashValue", value="1013695.60", currency="EUR"),
        _FakeAV(tag="NetLiquidation", value="1013874.30", currency="EUR"),
        _FakeAV(tag="BuyingPower", value="6759162.00", currency="EUR"),
    ]
    mock_ib = _make_mock_ib(account_values=eur_values)
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")
    assert snap.cash == 1013695.60


def test_pull_broker_snapshot_prefers_usd_when_both_present() -> None:
    """When both USD and EUR rows are complete, USD is selected.

    Defends against silently switching currencies on accounts that
    happen to receive a complete EUR set in addition to USD.
    """
    mixed = [
        _FakeAV(tag="TotalCashValue", value="50000", currency="USD"),
        _FakeAV(tag="NetLiquidation", value="60000", currency="USD"),
        _FakeAV(tag="BuyingPower", value="120000", currency="USD"),
        _FakeAV(tag="TotalCashValue", value="9999999", currency="EUR"),
        _FakeAV(tag="NetLiquidation", value="9999999", currency="EUR"),
        _FakeAV(tag="BuyingPower", value="9999999", currency="EUR"),
    ]
    mock_ib = _make_mock_ib(account_values=mixed)
    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="...")
    assert snap.cash == 50000.0  # USD row chosen, not EUR


# ---- Round-trip through reconcile -----------------------------------


def test_pull_broker_snapshot_round_trip_through_reconcile() -> None:
    """End-to-end: synthetic IBKR portfolio → snapshot → assert_reconciled.

    Builds matching internal ``PortfolioState``; ``assert_reconciled``
    returns ``ok=True``. Then drifts one ticker by 1 share →
    ``ReconciliationError``.
    """
    items = [
        _FakePortfolioItem(_FakeContract("AAPL"), position=100, averageCost=150.0),
        _FakePortfolioItem(_FakeContract("MSFT"), position=50, averageCost=300.0),
    ]
    mock_ib = _make_mock_ib(
        account_values=_good_account_values(cash=10_000.0),
        portfolio_items=items,
    )

    snap = pull_broker_snapshot(mock_ib, "DU123", as_of="2026-05-07T16:00:00Z")

    # Internal state matching the IBKR side.
    state = PortfolioState(
        cash=10_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=100, avg_cost=150.0),
            "MSFT": Position(ticker="MSFT", quantity=50, avg_cost=300.0),
        },
    )
    ledger = Ledger()
    report = assert_reconciled(state, snap, ledger=ledger)
    assert report.ok
    assert report.cash_drift_usd == 0.0
    assert len(report.position_drifts) == 0

    # Drift one ticker by 1 share → ReconciliationError raised.
    drifted_state = PortfolioState(
        cash=10_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=99, avg_cost=150.0),  # off by 1
            "MSFT": Position(ticker="MSFT", quantity=50, avg_cost=300.0),
        },
    )
    with pytest.raises(ReconciliationError):
        assert_reconciled(drifted_state, snap, ledger=Ledger())
