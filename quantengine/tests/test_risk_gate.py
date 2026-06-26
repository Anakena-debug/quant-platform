"""Smoke tests for quantengine.risk.gate.

Covers:
    - each built-in check in isolation
    - RiskGate composition order / greedy drop
    - reason/check attribution on rejections
    - edge cases: unknown tickers, insolvent NAV, priced-out orders
"""

from __future__ import annotations

import numpy as np

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import (
    RiskGate,
    known_ticker_check,
    max_gross_leverage_check,
    max_order_notional_check,
    max_position_weight_check,
    non_negative_cash_check,
)
from uuid import uuid4


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-04-17T16:00:00Z",
        tickers=("AAPL", "MSFT", "NVDA", "SPY"),
        prices=np.array([200.0, 400.0, 100.0, 500.0]),
    )


def _order(ticker: str, signed_qty: int) -> Order:
    side = OrderSide.BUY if signed_qty > 0 else OrderSide.SELL
    return Order(
        order_id=uuid4(),
        ticker=ticker,
        side=side,
        quantity=abs(signed_qty),
        order_type=OrderType.MARKET,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def test_known_ticker_rejects_unpriced():
    chk = known_ticker_check()
    state = PortfolioState.empty(1_000_000.0)
    orders = [_order("AAPL", 10), _order("FOOBAR", 10)]
    rej = chk(orders, state, _market())
    assert len(rej) == 1
    assert rej[0].order.ticker == "FOOBAR"
    assert rej[0].check == "known_ticker"


def test_max_order_notional_rejects_fat_finger():
    chk = max_order_notional_check(max_notional=50_000.0)
    state = PortfolioState.empty(1_000_000.0)
    # 300 * 200 = 60_000 > 50_000 cap; 100 * 200 = 20_000 ok.
    orders = [_order("AAPL", 300), _order("AAPL", 100)]
    rej = chk(orders, state, _market())
    assert len(rej) == 1
    assert rej[0].order.quantity == 300
    assert "cap" in rej[0].reason


def test_non_negative_cash_drops_biggest_buy_first():
    chk = non_negative_cash_check(min_cash=0.0)
    state = PortfolioState.empty(50_000.0)
    # Buys: MSFT 100*400=40_000, NVDA 200*100=20_000; both together would eat 60k > 50k.
    orders = [_order("MSFT", 100), _order("NVDA", 200)]
    rej = chk(orders, state, _market())
    # Greedy drops largest first (MSFT notional 40k).
    assert len(rej) == 1
    assert rej[0].order.ticker == "MSFT"


def test_non_negative_cash_permits_sells():
    chk = non_negative_cash_check(min_cash=0.0)
    state = PortfolioState(
        cash=1000.0,
        positions={"AAPL": Position("AAPL", 500, 150.0)},
    )
    # Single SELL raises cash; never rejected.
    orders = [_order("AAPL", -100)]
    rej = chk(orders, state, _market())
    assert rej == []


def test_max_gross_leverage_drops_to_cap():
    chk = max_gross_leverage_check(max_leverage=1.0)
    state = PortfolioState(
        cash=100_000.0,
        positions={"AAPL": Position("AAPL", 500, 200.0)},  # $100k equity
    )
    # NAV = 200k. Add long 500 MSFT (200k more) → gross 300k, gross/NAV=1.0.
    # Further add NVDA 1000 (100k notional) → gross 400k/NAV 200k=2.0 → must drop.
    orders = [_order("MSFT", 500), _order("NVDA", 1000)]
    rej = chk(orders, state, _market())
    assert len(rej) >= 1
    # Largest notional dropped first.
    assert any(r.order.ticker == "MSFT" for r in rej)


def test_max_position_weight_rejects_concentration():
    chk = max_position_weight_check(max_weight=0.10)
    state = PortfolioState.empty(1_000_000.0)
    # 1000 * 500 = 500k on SPY; NAV≈1M → 50% weight > 10% cap.
    orders = [_order("SPY", 1000)]
    rej = chk(orders, state, _market())
    assert len(rej) == 1
    assert rej[0].order.ticker == "SPY"


def test_max_order_notional_rejects_invalid_cap():
    raised = False
    try:
        max_order_notional_check(max_notional=0.0)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Gate composition
# ---------------------------------------------------------------------------
def test_gate_default_factory_composes_checks():
    gate = RiskGate.default_us_equities(
        max_order_notional=500_000,
        max_gross_leverage=1.5,
        max_position_weight=0.10,
    )
    # 5 built-in checks in the factory.
    names = {c.name for c in gate.checks}
    assert names == {
        "known_ticker",
        "max_order_notional",
        "non_negative_cash",
        "max_gross_leverage",
        "max_position_weight",
    }


def test_gate_later_checks_see_only_survivors():
    """Greedy drop: unknown-ticker rejection in check 1 excludes order from check 2."""
    gate = RiskGate(
        checks=[
            known_ticker_check(),
            max_order_notional_check(1.0),  # ridiculous cap — would reject everything
        ]
    )
    state = PortfolioState.empty(1_000_000.0)
    orders = [_order("FOOBAR", 10), _order("AAPL", 10)]
    accepted, rejected = gate.validate(orders, state, _market())
    # FOOBAR rejected by check 1; AAPL rejected by check 2.
    assert len(rejected) == 2
    foobar_rej = [r for r in rejected if r.order.ticker == "FOOBAR"]
    aapl_rej = [r for r in rejected if r.order.ticker == "AAPL"]
    assert len(foobar_rej) == 1 and foobar_rej[0].check == "known_ticker"
    assert len(aapl_rej) == 1 and aapl_rej[0].check == "max_order_notional"
    assert accepted == []


def test_gate_clean_orders_pass_through():
    gate = RiskGate.default_us_equities(
        max_order_notional=500_000,
        max_gross_leverage=2.0,
        max_position_weight=0.20,
    )
    state = PortfolioState.empty(1_000_000.0)
    # 100 AAPL @ 200 = 20k notional; well under all caps.
    orders = [_order("AAPL", 100)]
    accepted, rejected = gate.validate(orders, state, _market())
    assert len(accepted) == 1 and rejected == []


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_known_ticker_rejects_unpriced,
        test_max_order_notional_rejects_fat_finger,
        test_non_negative_cash_drops_biggest_buy_first,
        test_non_negative_cash_permits_sells,
        test_max_gross_leverage_drops_to_cap,
        test_max_position_weight_rejects_concentration,
        test_max_order_notional_rejects_invalid_cap,
        test_gate_default_factory_composes_checks,
        test_gate_later_checks_see_only_survivors,
        test_gate_clean_orders_pass_through,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nrisk.gate: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
