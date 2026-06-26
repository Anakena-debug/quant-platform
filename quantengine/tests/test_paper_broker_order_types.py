"""quantengine — PaperBroker order-type-aware fill semantics (s66)."""

from __future__ import annotations

import numpy as np

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order, OrderType
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.paper import PaperBroker


def _market(ref: float = 100.0, ticker: str = "AAA") -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-01-02T16:00:00", tickers=(ticker,), prices=np.array([ref])
    )


def _broker(slippage_bps: float = 0.0) -> PaperBroker:
    return PaperBroker(cost_model=LinearCostModel(slippage_bps=slippage_bps))


def test_market_fills_unconditionally_at_ref():
    b = _broker(0.0)
    fills = b.submit_orders([Order.new("AAA", 100, OrderType.MARKET)], _market(100.0))
    assert len(fills) == 1 and fills[0].price == 100.0
    assert b.open_orders() == ()


def test_limit_buy_marketable_fills_capped_at_limit():
    # ref 100 <= limit 100.5, 100bps slippage -> cost_fill 101 -> capped to the 100.5 limit.
    b = _broker(100.0)
    fills = b.submit_orders(
        [Order.new("AAA", 100, OrderType.LIMIT, limit_price=100.5)], _market(100.0)
    )
    assert len(fills) == 1 and np.isclose(fills[0].price, 100.5)  # never pay above the limit


def test_limit_buy_unmarketable_rests():
    b = _broker(0.0)
    fills = b.submit_orders(
        [Order.new("AAA", 100, OrderType.LIMIT, limit_price=99.0)], _market(100.0)
    )
    assert fills == []  # ref 100 > limit 99 -> not marketable
    assert len(tuple(b.open_orders())) == 1  # order rested


def test_limit_sell_marketable_fills_capped_at_limit():
    # ref 100 >= limit 99.5, 100bps slippage -> cost_fill 99 -> capped UP to the 99.5 limit.
    b = _broker(100.0)
    fills = b.submit_orders(
        [Order.new("AAA", -100, OrderType.LIMIT, limit_price=99.5)], _market(100.0)
    )
    assert len(fills) == 1 and np.isclose(fills[0].price, 99.5)  # never sell below the limit


def test_limit_sell_unmarketable_rests():
    b = _broker(0.0)
    fills = b.submit_orders(
        [Order.new("AAA", -100, OrderType.LIMIT, limit_price=101.0)], _market(100.0)
    )
    assert fills == []  # ref 100 < limit 101 -> not marketable
    assert len(tuple(b.open_orders())) == 1


def test_moc_and_loo_no_limit_fill_like_market():
    b = _broker(0.0)
    moc = b.submit_orders([Order.new("AAA", 100, OrderType.MOC)], _market(100.0))
    loo = b.submit_orders([Order.new("AAA", 100, OrderType.LOO)], _market(100.0))  # no limit_price
    assert len(moc) == 1 and moc[0].price == 100.0
    assert len(loo) == 1 and loo[0].price == 100.0


def test_buy_stop_triggers_at_or_above_stop_else_rests():
    # breakout buy-stop at 101: triggers when ref >= 101, then fills market-style.
    b = _broker(0.0)
    fills = b.submit_orders(
        [Order.new("AAA", 100, OrderType.STOP, stop_price=101.0)], _market(102.0)
    )
    assert len(fills) == 1 and fills[0].price == 102.0  # market-style fill at ref
    # below the stop → rests
    b2 = _broker(0.0)
    assert (
        b2.submit_orders([Order.new("AAA", 100, OrderType.STOP, stop_price=101.0)], _market(100.0))
        == []
    )
    assert len(tuple(b2.open_orders())) == 1


def test_sell_stop_triggers_at_or_below_stop_else_rests():
    # stop-loss sell-stop at 99: triggers when ref <= 99.
    b = _broker(0.0)
    fills = b.submit_orders(
        [Order.new("AAA", -100, OrderType.STOP, stop_price=99.0)], _market(98.0)
    )
    assert len(fills) == 1 and fills[0].price == 98.0
    b2 = _broker(0.0)
    assert (
        b2.submit_orders([Order.new("AAA", -100, OrderType.STOP, stop_price=99.0)], _market(100.0))
        == []
    )
    assert len(tuple(b2.open_orders())) == 1


def _buy_stop_limit() -> Order:
    return Order.new("AAA", 100, OrderType.STOP_LIMIT, stop_price=101.0, limit_price=102.0)


def _sell_stop_limit() -> Order:
    return Order.new("AAA", -100, OrderType.STOP_LIMIT, stop_price=99.0, limit_price=98.0)


def test_buy_stop_limit_trigger_plus_marketable():
    # ref 101.5: triggered (>=101) AND marketable (<=102) -> fills capped at min(cost, 102)
    assert _broker(0.0).submit_orders([_buy_stop_limit()], _market(101.5))[0].price == 101.5
    # ref 100: not triggered -> rest
    b = _broker(0.0)
    assert b.submit_orders([_buy_stop_limit()], _market(100.0)) == []
    assert len(tuple(b.open_orders())) == 1
    # ref 103: triggered (>=101) but NOT marketable (103 > limit 102) -> rest
    b2 = _broker(0.0)
    assert b2.submit_orders([_buy_stop_limit()], _market(103.0)) == []
    assert len(tuple(b2.open_orders())) == 1


def test_sell_stop_limit_trigger_plus_marketable():
    # ref 98.5: triggered (<=99) AND marketable (>=98) -> fills
    assert _broker(0.0).submit_orders([_sell_stop_limit()], _market(98.5))[0].price == 98.5
    # ref 97: triggered (<=99) but NOT marketable (97 < limit 98, price gapped through) -> rest
    b = _broker(0.0)
    assert b.submit_orders([_sell_stop_limit()], _market(97.0)) == []
    assert len(tuple(b.open_orders())) == 1


# --- s72: resting-order re-evaluation sweep + trailing stops ---------------------


def test_resting_stop_refires_on_a_later_triggering_bar():
    """A sell-stop that rests on its submit bar triggers on a LATER bar (the re-evaluation sweep)."""
    b = _broker(0.0)
    stop = Order.new("AAA", -100, OrderType.STOP, stop_price=99.0)
    # bar 1: ref 100 > stop 99 -> rests, no fill
    assert b.submit_orders([stop], _market(100.0)) == []
    assert len(tuple(b.open_orders())) == 1
    # bar 2: ref 98 <= stop 99, no fresh orders -> the sweep fires the resting stop
    fills = b.submit_orders([], _market(98.0))
    assert len(fills) == 1 and fills[0].price == 98.0
    assert b.open_orders() == ()  # dropped from the resting book


def test_sell_trail_ratchets_up_then_triggers_on_pullback():
    """SELL trail tracks the running HIGH; triggers when ref falls trail_amount below the peak."""
    b = _broker(0.0)
    trail = Order.new("AAA", -100, OrderType.TRAIL, trail_amount=5.0)
    # submit at 100 -> rests, water-mark 100, effective stop 95
    assert b.submit_orders([trail], _market(100.0)) == []
    # rises to 110 -> water-mark ratchets to 110, stop 105, 110 > 105 no trigger
    assert b.submit_orders([], _market(110.0)) == []
    assert len(tuple(b.open_orders())) == 1
    # pullback to 108 > 105 -> still no trigger (the mark does NOT ratchet back down)
    assert b.submit_orders([], _market(108.0)) == []
    # pullback to 104 <= 105 -> triggers, market-style fill at 104
    fills = b.submit_orders([], _market(104.0))
    assert len(fills) == 1 and fills[0].price == 104.0
    assert b.open_orders() == ()


def test_buy_trail_ratchets_down_then_triggers_on_bounce():
    """BUY trail (cover a short) tracks the running LOW; triggers when ref rises above low+amount."""
    b = _broker(0.0)
    trail = Order.new("AAA", 100, OrderType.TRAIL, trail_amount=5.0)
    # submit at 100 -> water-mark 100, stop 105, 100 < 105 no trigger
    assert b.submit_orders([trail], _market(100.0)) == []
    # falls to 90 -> water-mark 90, stop 95
    assert b.submit_orders([], _market(90.0)) == []
    # bounce to 94 < 95 -> no trigger
    assert b.submit_orders([], _market(94.0)) == []
    # bounce to 96 >= 95 -> triggers
    fills = b.submit_orders([], _market(96.0))
    assert len(fills) == 1 and fills[0].price == 96.0


def test_sell_trail_percent_triggers():
    """SELL trail with a percent distance: stop = high * (1 - pct/100)."""
    b = _broker(0.0)
    trail = Order.new("AAA", -100, OrderType.TRAIL, trail_percent=10.0)
    assert b.submit_orders([trail], _market(100.0)) == []  # stop 90
    assert b.submit_orders([], _market(120.0)) == []  # ratchet: stop 108
    assert b.submit_orders([], _market(109.0)) == []  # 109 > 108 no trigger
    fills = b.submit_orders([], _market(107.0))  # 107 <= 108 triggers
    assert len(fills) == 1 and fills[0].price == 107.0


def test_trail_limit_trigger_marketable_vs_gapped():
    """TRAIL_LIMIT triggers like a trail, then the post-trigger LIMIT can gap through and rest."""
    # SELL trail_limit, amount 5, offset 1 -> on trigger, limit = effective_stop - 1
    b = _broker(0.0)
    tl = Order.new("AAA", -100, OrderType.TRAIL_LIMIT, trail_amount=5.0, limit_offset=1.0)
    assert b.submit_orders([tl], _market(100.0)) == []  # water-mark 100, stop 95
    # drop to 94: triggered (94 <= 95); limit = 95 - 1 = 94; ref 94 >= 94 marketable -> fills at 94
    fills = b.submit_orders([], _market(94.0))
    assert len(fills) == 1 and fills[0].price == 94.0

    # gapped: drop straight through the limit -> rests
    b2 = _broker(0.0)
    tl2 = Order.new("AAA", -100, OrderType.TRAIL_LIMIT, trail_amount=5.0, limit_offset=1.0)
    assert b2.submit_orders([tl2], _market(100.0)) == []  # stop 95, limit-on-trigger 94
    # drop to 93: triggered (93 <= 95) but ref 93 < limit 94 -> not marketable -> rests
    assert b2.submit_orders([], _market(93.0)) == []
    assert len(tuple(b2.open_orders())) == 1


def test_limit_and_loo_resting_orders_are_not_swept():
    """s66 byte-parity guard: resting LIMIT/LOO are submit-once — a later bar never fills them."""
    b = _broker(0.0)
    # unmarketable buy-limit (ref 100 > limit 99) rests
    assert (
        b.submit_orders([Order.new("AAA", 100, OrderType.LIMIT, limit_price=99.0)], _market(100.0))
        == []
    )
    assert len(tuple(b.open_orders())) == 1
    # a later bar where it WOULD be marketable as a fresh order (ref 98 <= 99) must NOT fill it
    assert b.submit_orders([], _market(98.0)) == []
    assert len(tuple(b.open_orders())) == 1  # still resting, untouched by the sweep


def test_cancel_all_clears_trail_state():
    """cancel_all drops resting trails so a later bar can't fire a cancelled order."""
    b = _broker(0.0)
    b.submit_orders([Order.new("AAA", -100, OrderType.TRAIL, trail_amount=5.0)], _market(100.0))
    assert b.cancel_all() == 1
    # even a steep drop produces no fill — the trail (and its water-mark) is gone
    assert b.submit_orders([], _market(50.0)) == []
    assert b.open_orders() == ()


def test_cancel_order_removes_one_resting_order_and_its_trail_mark():
    """F2: cancel_order retires a single resting order (and its trail water-mark),
    leaving the rest — so per-bar protective_stops regeneration can be idempotent."""
    b = _broker(0.0)
    stop = Order.new("AAA", 100, OrderType.STOP, stop_price=101.0)  # buy-stop above 100 → rests
    trail = Order.new("BBB", -100, OrderType.TRAIL, trail_amount=5.0)  # rests w/ water-mark
    mkt = MarketSnapshot(timestamp="t", tickers=("AAA", "BBB"), prices=np.array([100.0, 100.0]))
    assert b.submit_orders([stop, trail], mkt) == []  # both rest
    assert {o.order_id for o in b.open_orders()} == {stop.order_id, trail.order_id}
    assert trail.order_id in b._trail_state

    assert b.cancel_order(trail.order_id) is True
    assert {o.order_id for o in b.open_orders()} == {stop.order_id}  # the stop still rests
    assert trail.order_id not in b._trail_state  # its water-mark was dropped
    assert b.cancel_order(trail.order_id) is False  # already gone → no-op


def test_misconfigured_stop_family_rests_not_market_fills():
    """F6: a stop-family order with a None trigger (only reachable by bypassing the
    frozen __post_init__ — e.g. a recovery path that skips validation) must REST, not
    fire a market exit. Resting is fail-closed; a market exit is the worst default."""
    b = _broker(0.0)
    stop = Order.new("AAA", -100, OrderType.STOP, stop_price=95.0)
    object.__setattr__(stop, "stop_price", None)  # corrupt the trigger post-construction
    assert b.submit_orders([stop], _market(50.0)) == []  # 50 is far through a real stop
    assert len(tuple(b.open_orders())) == 1  # rested, NOT market-filled

    b2 = _broker(0.0)
    trail = Order.new("AAA", -100, OrderType.TRAIL, trail_percent=3.0)
    object.__setattr__(trail, "trail_percent", None)  # neither trail distance set
    assert b2.submit_orders([trail], _market(50.0)) == []
    assert len(tuple(b2.open_orders())) == 1
