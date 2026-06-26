import numpy as np
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import OrderSide, OrderType
from quantengine.contracts.signal import build_alpha_signal
from quantengine.portfolio.constraints import NoTradePolicy, RebalanceConstraints
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState


# ------------------------------------------------------------------
# Core behaviour
# ------------------------------------------------------------------
def test_zero_signal_produces_no_orders(empty_state, market, tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.0] * 4,
        lower=[-0.01] * 4,
        upper=[0.01] * 4,
        alpha=0.10,
        kelly_weights=[0.0] * 4,
    )
    engine = RebalanceEngine()
    orders = engine.rebalance(sig, empty_state, market)
    assert orders == []


def test_untradeable_names_are_skipped_on_empty_book(empty_state, market, partial_signal):
    engine = RebalanceEngine()
    orders = engine.rebalance(partial_signal, empty_state, market)
    # Only AAPL + NVDA should trade (MSFT and SPY intervals contain zero)
    traded = {o.ticker for o in orders}
    assert "AAPL" in traded and "NVDA" in traded
    assert "MSFT" not in traded and "SPY" not in traded


def test_all_tradeable_generates_buys(empty_state, market, tradeable_signal):
    engine = RebalanceEngine()
    orders = engine.rebalance(tradeable_signal, empty_state, market)
    assert len(orders) == 4
    assert all(o.side == OrderSide.BUY for o in orders)
    assert all(o.order_type == OrderType.MARKET for o in orders)


# ------------------------------------------------------------------
# Invariants
# ------------------------------------------------------------------
def test_post_trade_gross_within_leverage_and_cash_buffer(empty_state, market, tradeable_signal):
    c = RebalanceConstraints(cash_buffer=0.05, max_gross_leverage=1.0)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(tradeable_signal, empty_state, market)

    notional = sum(o.signed_quantity * market.price_of(o.ticker) for o in orders)
    gross = sum(abs(o.signed_quantity) * market.price_of(o.ticker) for o in orders)
    nav = empty_state.cash  # empty book
    # Gross <= (1 - cash_buffer) * NAV  (rounding slack)
    assert gross <= (1 - c.cash_buffer) * nav + 1e3
    # Post-trade cash >= cash_buffer * NAV  (pre-commission, reference prices)
    assert empty_state.cash - notional >= c.cash_buffer * nav - 1e3


def test_turnover_cap_is_respected(empty_state, market, tradeable_signal):
    c = RebalanceConstraints(max_turnover=0.20, cash_buffer=0.0)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(tradeable_signal, empty_state, market)
    gross = sum(abs(o.signed_quantity) * market.price_of(o.ticker) for o in orders)
    nav = empty_state.cash
    # Gross turnover <= T * NAV + small rounding slack
    assert gross <= c.max_turnover * nav + 1e3


def test_min_trade_notional_filters_dust(empty_state, market, tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.01, 0.01, 0.01, 0.01],
        lower=[0.001, 0.001, 0.001, 0.001],
        upper=[0.02, 0.02, 0.02, 0.02],
        alpha=0.10,
        kelly_weights=[1e-6, 1e-6, 1e-6, 1e-6],  # dust
    )
    c = RebalanceConstraints(min_trade_notional=1000.0)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(sig, empty_state, market)
    for o in orders:
        assert o.quantity * market.price_of(o.ticker) >= c.min_trade_notional


# ------------------------------------------------------------------
# No-trade policies
# ------------------------------------------------------------------
def test_no_trade_policy_flatten_closes_incumbent(market, tickers):
    # Start with existing MSFT position; signal says MSFT is untradeable.
    from uuid import uuid4
    from quantengine.contracts.orders import Fill

    state = PortfolioState.empty(500_000.0)
    state = state.apply(
        Fill(
            fill_id=uuid4(),
            order_id=uuid4(),
            ticker="MSFT",
            signed_quantity=+200,
            price=300.0,
            commission=0.0,
            timestamp="2026-04-16",
        )
    )
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, -0.005, 0.01, -0.002],  # MSFT interval contains 0
        upper=[0.04, 0.020, 0.05, 0.010],
        alpha=0.10,
        kelly_weights=[0.30, 0.0, 0.20, 0.0],
    )
    c = RebalanceConstraints(no_trade_policy=NoTradePolicy.FLATTEN)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(sig, state, market)
    msft_orders = [o for o in orders if o.ticker == "MSFT"]
    assert len(msft_orders) == 1
    assert msft_orders[0].side == OrderSide.SELL
    assert msft_orders[0].quantity == 200


def test_no_trade_policy_hold_leaves_incumbent_alone(market, tickers):
    from uuid import uuid4
    from quantengine.contracts.orders import Fill

    state = PortfolioState.empty(500_000.0)
    state = state.apply(
        Fill(
            fill_id=uuid4(),
            order_id=uuid4(),
            ticker="MSFT",
            signed_quantity=+200,
            price=300.0,
            commission=0.0,
            timestamp="2026-04-16",
        )
    )
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, -0.005, 0.01, -0.002],
        upper=[0.04, 0.020, 0.05, 0.010],
        alpha=0.10,
        kelly_weights=[0.30, 0.0, 0.20, 0.0],
    )
    c = RebalanceConstraints(no_trade_policy=NoTradePolicy.HOLD)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(sig, state, market)
    msft_orders = [o for o in orders if o.ticker == "MSFT"]
    assert msft_orders == []


# ------------------------------------------------------------------
# Short-sale policy
# ------------------------------------------------------------------
def test_allow_short_false_clips_negative_weights(empty_state, market, tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, -0.02, 0.03, -0.01],
        lower=[0.005, -0.04, 0.01, -0.03],
        upper=[0.04, -0.005, 0.05, -0.001],
        alpha=0.10,
        kelly_weights=[0.30, -0.20, 0.25, -0.10],
    )
    engine = RebalanceEngine(RebalanceConstraints(allow_short=False))
    orders = engine.rebalance(sig, empty_state, market)
    # No SELL orders on an empty book when shorts are disabled
    assert all(o.side == OrderSide.BUY for o in orders)


def test_allow_short_true_opens_shorts(empty_state, market, tickers):
    sig = build_alpha_signal(
        tickers=tickers,
        expected_return=[0.02, -0.02, 0.03, -0.01],
        lower=[0.005, -0.04, 0.01, -0.03],
        upper=[0.04, -0.005, 0.05, -0.001],
        alpha=0.10,
        kelly_weights=[0.30, -0.20, 0.25, -0.10],
    )
    c = RebalanceConstraints(allow_short=True, cash_buffer=0.05)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(sig, empty_state, market)
    sides = {o.ticker: o.side for o in orders}
    assert sides["MSFT"] == OrderSide.SELL
    assert sides["SPY"] == OrderSide.SELL


# ------------------------------------------------------------------
# Misalignment guard
# ------------------------------------------------------------------
def test_misaligned_tickers_raise(empty_state, tradeable_signal):
    wrong = MarketSnapshot(
        timestamp="2026-04-17",
        tickers=("AAPL", "MSFT", "NVDA", "TSLA"),
        prices=np.array([150.0, 300.0, 600.0, 250.0]),
    )
    engine = RebalanceEngine()
    with pytest.raises(ValueError):
        engine.rebalance(tradeable_signal, empty_state, wrong)


# ------------------------------------------------------------------
# Lot size > 1
# ------------------------------------------------------------------
def test_lot_size_rounds_to_multiples(empty_state, market, tradeable_signal):
    c = RebalanceConstraints(lot_size=10, cash_buffer=0.02)
    engine = RebalanceEngine(c)
    orders = engine.rebalance(tradeable_signal, empty_state, market)
    for o in orders:
        assert o.quantity % 10 == 0


# ------------------------------------------------------------------
# Reference-price sanity gate (audit: garbage-order money-path)
# ------------------------------------------------------------------
def test_nan_reference_price_raises(empty_state, tickers, tradeable_signal):
    # A halted / no-quote symbol can carry a NaN reference price. NaN slips past
    # MarketSnapshot.__post_init__ (NaN <= 0 is False), so without the rebalance
    # guard `q_star = hat_w * NAV / NaN` -> NaN -> int64 sentinel share count.
    prices = np.array([150.0, np.nan, 600.0, 500.0])
    bad_market = MarketSnapshot(timestamp="2026-04-17", tickers=tickers, prices=prices)
    engine = RebalanceEngine()
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        engine.rebalance(tradeable_signal, empty_state, bad_market)


def test_inf_reference_price_raises(empty_state, tickers, tradeable_signal):
    prices = np.array([150.0, 300.0, np.inf, 500.0])
    bad_market = MarketSnapshot(timestamp="2026-04-17", tickers=tickers, prices=prices)
    engine = RebalanceEngine()
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        engine.rebalance(tradeable_signal, empty_state, bad_market)


def test_valid_prices_still_size_normally(empty_state, market, tradeable_signal):
    # Guard must not regress the happy path: finite positive prices -> orders.
    engine = RebalanceEngine()
    orders = engine.rebalance(tradeable_signal, empty_state, market)
    assert len(orders) == 4
    assert all(np.isfinite(o.signed_quantity) for o in orders)


# ------------------------------------------------------------------
# Protective stops (s72)
# ------------------------------------------------------------------
def _state_with(positions: dict[str, int]) -> PortfolioState:
    """Build a PortfolioState holding the given signed positions (opened at $100)."""
    from uuid import uuid4

    from quantengine.contracts.orders import Fill

    state = PortfolioState.empty(1_000_000.0)
    for tkr, qty in positions.items():
        state = state.apply(
            Fill(
                fill_id=uuid4(),
                order_id=uuid4(),
                ticker=tkr,
                signed_quantity=qty,
                price=100.0,
                commission=0.0,
                timestamp="2026-04-16",
            )
        )
    return state


def test_protective_stops_long_emits_sell_stop_below(market):
    state = _state_with({"AAPL": 100})  # long
    orders = RebalanceEngine().protective_stops(state, market, stop_loss_pct=0.05)
    assert len(orders) == 1
    o = orders[0]
    assert o.ticker == "AAPL" and o.side == OrderSide.SELL and o.quantity == 100
    assert o.order_type == OrderType.STOP
    assert np.isclose(o.stop_price, 150.0 * 0.95)  # 5% below the 150 ref


def test_protective_stops_short_emits_buy_stop_above(market):
    state = _state_with({"MSFT": -50})  # short
    orders = RebalanceEngine().protective_stops(state, market, stop_loss_pct=0.05)
    assert len(orders) == 1
    o = orders[0]
    assert o.ticker == "MSFT" and o.side == OrderSide.BUY and o.quantity == 50
    assert o.order_type == OrderType.STOP
    assert np.isclose(o.stop_price, 300.0 * 1.05)  # 5% above the 300 ref


def test_protective_stops_trail_variant(market):
    state = _state_with({"AAPL": 100})
    orders = RebalanceEngine().protective_stops(state, market, trail_percent=3.0)
    assert len(orders) == 1
    o = orders[0]
    assert o.order_type == OrderType.TRAIL and o.trail_percent == 3.0
    assert o.side == OrderSide.SELL and o.stop_price is None


def test_protective_stops_skips_unpriced_positions():
    # market prices only AAPL; a held name absent from market cannot be levelled -> skipped
    mkt = MarketSnapshot(
        timestamp="2026-04-17T16:00:00Z", tickers=("AAPL",), prices=np.array([150.0])
    )
    state = _state_with({"AAPL": 100, "TSLA": 20})
    orders = RebalanceEngine().protective_stops(state, mkt, stop_loss_pct=0.05)
    assert {o.ticker for o in orders} == {"AAPL"}  # TSLA skipped (unpriced)


def test_protective_stops_requires_exactly_one_param(market):
    state = _state_with({"AAPL": 100})
    engine = RebalanceEngine()
    with pytest.raises(ValueError, match="exactly one of stop_loss_pct or trail_percent"):
        engine.protective_stops(state, market)  # neither
    with pytest.raises(ValueError, match="exactly one of stop_loss_pct or trail_percent"):
        engine.protective_stops(state, market, stop_loss_pct=0.05, trail_percent=3.0)  # both


def test_protective_stops_rejects_out_of_range_pct(market):
    # The XOR check is a None-identity test, so 0.0 slips past it; and a stop_loss_pct >= 1
    # yields a non-positive SELL stop that ref > 0 can never reach (silent zero protection).
    # Both must be rejected loudly at the boundary.
    state = _state_with({"AAPL": 100})
    engine = RebalanceEngine()
    for bad in (0.0, 1.0, 1.5):  # 0 → fires on submit bar; >=1 → never protects
        with pytest.raises(ValueError, match=r"stop_loss_pct must be in \(0, 1\)"):
            engine.protective_stops(state, market, stop_loss_pct=bad)
    for bad in (0.0, 100.0):  # percent in (0, 100); not deep inside Order.__post_init__
        with pytest.raises(ValueError, match=r"trail_percent must be in \(0, 100\)"):
            engine.protective_stops(state, market, trail_percent=bad)
