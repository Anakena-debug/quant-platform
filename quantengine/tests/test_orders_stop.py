"""quantengine — STOP / STOP_LIMIT Order contract validation (s71)."""

from __future__ import annotations

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType


def test_stop_order_requires_stop_price():
    with pytest.raises(ValueError, match="STOP order requires stop_price"):
        Order.new("AAA", 100, OrderType.STOP)  # no stop_price
    o = Order.new("AAA", 100, OrderType.STOP, stop_price=95.0)
    assert o.order_type == OrderType.STOP and o.stop_price == 95.0
    assert o.side == OrderSide.BUY and o.signed_quantity == 100


def test_stop_limit_requires_both_prices():
    with pytest.raises(ValueError, match="STOP_LIMIT order requires both"):
        Order.new("AAA", 100, OrderType.STOP_LIMIT, stop_price=95.0)  # missing limit
    with pytest.raises(ValueError, match="STOP_LIMIT order requires both"):
        Order.new("AAA", 100, OrderType.STOP_LIMIT, limit_price=94.0)  # missing stop
    o = Order.new("AAA", -100, OrderType.STOP_LIMIT, stop_price=95.0, limit_price=94.5)
    assert o.stop_price == 95.0 and o.limit_price == 94.5
    assert o.side == OrderSide.SELL and o.signed_quantity == -100


def test_non_stop_orders_default_stop_price_none():
    # stop_price defaults to None and is not required for non-stop types
    assert Order.new("AAA", 100, OrderType.MARKET).stop_price is None
    assert Order.new("AAA", 100, OrderType.LIMIT, limit_price=100.0).stop_price is None
    # a stray stop_price on a non-stop order is tolerated (ignored), no validation error
    assert Order.new("AAA", 100, OrderType.MARKET, stop_price=50.0).order_type == OrderType.MARKET
