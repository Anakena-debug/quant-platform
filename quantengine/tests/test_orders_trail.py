"""quantengine — TRAIL / TRAIL_LIMIT Order contract validation (s72)."""

from __future__ import annotations

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType


def test_trail_requires_exactly_one_of_amount_or_percent():
    # neither → reject
    with pytest.raises(ValueError, match="exactly one of trail_amount or trail_percent"):
        Order.new("AAA", -100, OrderType.TRAIL)
    # both → reject
    with pytest.raises(ValueError, match="exactly one of trail_amount or trail_percent"):
        Order.new("AAA", -100, OrderType.TRAIL, trail_amount=1.0, trail_percent=2.0)
    # amount alone → ok
    o = Order.new("AAA", -100, OrderType.TRAIL, trail_amount=1.5)
    assert o.order_type == OrderType.TRAIL and o.trail_amount == 1.5
    assert o.side == OrderSide.SELL and o.signed_quantity == -100
    # percent alone → ok
    o2 = Order.new("AAA", -100, OrderType.TRAIL, trail_percent=2.0)
    assert o2.trail_percent == 2.0 and o2.trail_amount is None


def test_trail_non_positive_distance_rejected():
    with pytest.raises(ValueError, match="trail_amount must be > 0"):
        Order.new("AAA", -100, OrderType.TRAIL, trail_amount=0.0)
    with pytest.raises(ValueError, match="trail_percent must be > 0"):
        Order.new("AAA", -100, OrderType.TRAIL, trail_percent=-1.0)


def test_trail_limit_requires_limit_offset():
    # the exactly-one-trail rule still applies to the limit variant
    with pytest.raises(ValueError, match="exactly one of trail_amount or trail_percent"):
        Order.new("AAA", -100, OrderType.TRAIL_LIMIT, limit_offset=0.5)
    # trail param present but no offset → reject
    with pytest.raises(ValueError, match="TRAIL_LIMIT order requires limit_offset"):
        Order.new("AAA", -100, OrderType.TRAIL_LIMIT, trail_amount=1.0)
    # negative offset → reject
    with pytest.raises(ValueError, match="TRAIL_LIMIT order requires limit_offset"):
        Order.new("AAA", -100, OrderType.TRAIL_LIMIT, trail_amount=1.0, limit_offset=-0.1)
    # offset == 0 is allowed (limit sits exactly at the trigger)
    o = Order.new("AAA", -100, OrderType.TRAIL_LIMIT, trail_percent=2.0, limit_offset=0.0)
    assert o.order_type == OrderType.TRAIL_LIMIT and o.limit_offset == 0.0
    o2 = Order.new("AAA", 100, OrderType.TRAIL_LIMIT, trail_amount=1.0, limit_offset=0.25)
    assert o2.side == OrderSide.BUY and o2.limit_offset == 0.25


def test_non_trail_orders_default_trail_fields_none():
    m = Order.new("AAA", 100, OrderType.MARKET)
    assert m.trail_amount is None and m.trail_percent is None and m.limit_offset is None
    # STOP is unaffected by the new fields
    s = Order.new("AAA", -100, OrderType.STOP, stop_price=95.0)
    assert s.trail_amount is None and s.stop_price == 95.0
    # a stray trail field on a non-trail order is tolerated (ignored), no validation error
    assert Order.new("AAA", 100, OrderType.MARKET, trail_amount=5.0).order_type == OrderType.MARKET


def test_exactly_one_error_names_the_actual_order_type():
    # A misconfigured TRAIL_LIMIT must name TRAIL_LIMIT (not "TRAIL"), so the developer
    # debugs the right order type; the plain TRAIL still names TRAIL.
    with pytest.raises(ValueError, match="TRAIL_LIMIT order requires exactly one"):
        Order.new("AAA", -100, OrderType.TRAIL_LIMIT, limit_offset=0.5)
    with pytest.raises(ValueError, match="TRAIL order requires exactly one"):
        Order.new("AAA", -100, OrderType.TRAIL)
