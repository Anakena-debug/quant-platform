"""Tests for ``order_to_ib_order`` and ``ib_trade_to_fill``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quantengine.contracts.orders import Order, OrderType
from quantengine.execution.ibkr.order_mapping import (
    ib_trade_to_fill,
    order_to_ib_order,
)


def test_order_mapping_market_order_roundtrip() -> None:
    """BUY 100 AAPL MARKET → ib_async Stock + IBOrder MKT, orderId=0."""
    order = Order.new(ticker="AAPL", signed_quantity=100, order_type=OrderType.MARKET)
    contract, ib_order = order_to_ib_order(order)

    assert contract.symbol == "AAPL"
    assert contract.secType == "STK"
    assert contract.currency == "USD"
    assert contract.exchange == "SMART"
    assert ib_order.action == "BUY"
    assert ib_order.totalQuantity == 100
    assert ib_order.orderType == "MKT"
    # orderId left as 0 — IB.placeOrder assigns via client.getReqId().
    assert ib_order.orderId == 0


def test_order_mapping_limit_order_carries_limit_price() -> None:
    """LIMIT order → orderType=LMT and lmtPrice set."""
    order = Order.new(
        ticker="AAPL",
        signed_quantity=50,
        order_type=OrderType.LIMIT,
        limit_price=152.50,
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "LMT"
    assert ib_order.lmtPrice == 152.50

    # LIMIT without limit_price would fail at Order construction (post-init).
    # If somehow constructed without one, order_to_ib_order also rejects.
    with pytest.raises(ValueError, match="LIMIT order requires limit_price"):
        # Bypass Order.__post_init__ by direct construction with None.
        from uuid import uuid4

        from quantengine.contracts.orders import OrderSide

        bad = Order.__new__(Order)
        object.__setattr__(bad, "order_id", uuid4())
        object.__setattr__(bad, "ticker", "AAPL")
        object.__setattr__(bad, "side", OrderSide.BUY)
        object.__setattr__(bad, "quantity", 10)
        object.__setattr__(bad, "order_type", OrderType.LIMIT)
        object.__setattr__(bad, "limit_price", None)
        object.__setattr__(bad, "timestamp", None)
        object.__setattr__(bad, "parent_signal_ts", None)
        object.__setattr__(bad, "metadata", {})
        order_to_ib_order(bad)


def test_order_mapping_moc_loo_aux_types() -> None:
    """MOC and LOO map to correct orderType + tif."""
    moc = Order.new(ticker="AAPL", signed_quantity=10, order_type=OrderType.MOC)
    _, ib_moc = order_to_ib_order(moc)
    assert ib_moc.orderType == "MOC"

    loo = Order.new(
        ticker="AAPL",
        signed_quantity=10,
        order_type=OrderType.LOO,
        limit_price=150.0,
    )
    _, ib_loo = order_to_ib_order(loo)
    # LOO is encoded as orderType="LMT" + tif="OPG"; the literal "LOO"
    # is NOT a valid IBKR orderType and the gateway rejects it.
    assert ib_loo.orderType == "LMT"
    assert ib_loo.tif == "OPG"
    assert ib_loo.lmtPrice == 150.0


def test_order_mapping_stop_carries_aux_price() -> None:
    """STOP → orderType=STP with auxPrice set to the stop trigger."""
    order = Order.new(
        ticker="AAPL", signed_quantity=-100, order_type=OrderType.STOP, stop_price=145.0
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "STP"
    assert ib_order.auxPrice == 145.0
    assert ib_order.action == "SELL"  # sell-stop (stop-loss)


def test_order_mapping_stop_limit_carries_both_prices() -> None:
    """STOP_LIMIT → orderType='STP LMT' with auxPrice (stop) + lmtPrice (post-trigger limit)."""
    order = Order.new(
        ticker="AAPL",
        signed_quantity=100,
        order_type=OrderType.STOP_LIMIT,
        stop_price=150.0,
        limit_price=151.0,
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "STP LMT"
    assert ib_order.auxPrice == 150.0  # trigger
    assert ib_order.lmtPrice == 151.0  # post-trigger limit


def test_order_mapping_trail_amount_sets_aux_price() -> None:
    """TRAIL with an absolute distance → orderType='TRAIL' with auxPrice set."""
    order = Order.new(
        ticker="AAPL", signed_quantity=-100, order_type=OrderType.TRAIL, trail_amount=2.5
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "TRAIL"
    assert ib_order.auxPrice == 2.5
    assert ib_order.action == "SELL"


def test_order_mapping_trail_percent_sets_trailing_percent() -> None:
    """TRAIL with a percent → orderType='TRAIL' with trailingPercent set (IBKR convention)."""
    order = Order.new(
        ticker="AAPL", signed_quantity=-100, order_type=OrderType.TRAIL, trail_percent=3.0
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "TRAIL"
    assert ib_order.trailingPercent == 3.0


def test_order_mapping_trail_limit_sets_offset() -> None:
    """TRAIL_LIMIT → orderType='TRAIL LIMIT' with the trail distance + lmtPriceOffset."""
    order = Order.new(
        ticker="AAPL",
        signed_quantity=-100,
        order_type=OrderType.TRAIL_LIMIT,
        trail_amount=2.0,
        limit_offset=0.5,
    )
    _, ib_order = order_to_ib_order(order)
    assert ib_order.orderType == "TRAIL LIMIT"
    assert ib_order.auxPrice == 2.0
    assert ib_order.lmtPriceOffset == 0.5


def test_order_mapping_buy_sell_side() -> None:
    """signed_quantity > 0 → BUY, < 0 → SELL, abs preserved."""
    buy = Order.new(ticker="AAPL", signed_quantity=100)
    _, ib_buy = order_to_ib_order(buy)
    assert ib_buy.action == "BUY"
    assert ib_buy.totalQuantity == 100

    sell = Order.new(ticker="AAPL", signed_quantity=-100)
    _, ib_sell = order_to_ib_order(sell)
    assert ib_sell.action == "SELL"
    assert ib_sell.totalQuantity == 100  # abs preserved (always positive)


def test_order_mapping_fill_preserves_signed_quantity() -> None:
    """SELL fill → ``Fill.signed_quantity`` is negative; metadata records IB IDs."""
    order = Order.new(ticker="AAPL", signed_quantity=-100)

    # Mock ib_async.Fill: has .execution and .commissionReport.
    mock_fill_event = MagicMock()
    mock_fill_event.execution.shares = 100  # IBKR always positive
    mock_fill_event.execution.price = 152.50
    mock_fill_event.execution.permId = 12345
    mock_fill_event.execution.execId = "abc.001"
    mock_fill_event.execution.time = "2026-05-07T16:00:00Z"
    mock_fill_event.commissionReport.commission = 1.0

    mock_trade = MagicMock()
    mock_trade.order.orderId = 42

    fill = ib_trade_to_fill(mock_trade, order, mock_fill_event)

    assert fill.ticker == "AAPL"
    assert fill.signed_quantity == -100  # SELL → negative
    assert fill.price == 152.50
    assert fill.commission == 1.0
    assert fill.metadata["ib_perm_id"] == 12345
    assert fill.metadata["ib_exec_id"] == "abc.001"
    assert fill.metadata["ib_order_id"] == 42

    # BUY fill: signed_quantity stays positive.
    buy_order = Order.new(ticker="AAPL", signed_quantity=50)
    mock_fill_event.execution.shares = 50
    fill_buy = ib_trade_to_fill(mock_trade, buy_order, mock_fill_event)
    assert fill_buy.signed_quantity == 50  # BUY → positive
