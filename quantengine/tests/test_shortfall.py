"""Smoke tests for quantengine.analytics.shortfall.

Covers:
    - Price-impact bucket = Σ q_k (p_k - p0).
    - Commission bucket sums across fills.
    - Missed-trade bucket = (q* - Q)(p_T - p0).
    - Total $ and bps computations.
    - decision_prices_from_metadata extracts first-fill reference price.
    - Orders without a decision price are skipped.
"""

from __future__ import annotations

from uuid import uuid4

from quantengine.analytics.shortfall import (
    compute_shortfall,
    decision_prices_from_metadata,
)
from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.portfolio.ledger import Ledger


def _submit_order(ledger: Ledger, ticker: str, signed_qty: int) -> Order:
    side = OrderSide.BUY if signed_qty > 0 else OrderSide.SELL
    order = Order(
        order_id=uuid4(),
        ticker=ticker,
        side=side,
        quantity=abs(signed_qty),
        order_type=OrderType.MARKET,
    )
    ledger.append("2026-04-17T16:00:00Z", "ORDER_SUBMITTED", order)
    return order


def _fill(
    order: Order, signed_qty: int, price: float, commission: float = 0.0, ref: float | None = None
) -> Fill:
    metadata: dict = {}
    if ref is not None:
        metadata["reference_price"] = ref
    return Fill(
        fill_id=uuid4(),
        order_id=order.order_id,
        ticker=order.ticker,
        signed_quantity=signed_qty,
        price=price,
        commission=commission,
        timestamp="2026-04-17T16:00:00Z",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def test_price_impact_zero_at_decision_price():
    """If every fill prints at p0, price_impact = 0."""
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=200.0, commission=0.0)
    )
    reports = compute_shortfall(ledger, decision_prices={o.order_id: 200.0})
    assert len(reports) == 1
    r = reports[0]
    assert abs(r.price_impact_usd) < 1e-9
    assert r.filled_quantity == 100


def test_price_impact_buy_above_decision_is_positive_cost():
    """BUY @ 201 after decision @ 200: paid $1/sh above → +100 USD cost."""
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=201.0, commission=0.0)
    )
    r = compute_shortfall(ledger, decision_prices={o.order_id: 200.0})[0]
    assert abs(r.price_impact_usd - 100.0) < 1e-9


def test_commission_sums_across_fills():
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 200)
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=200.0, commission=0.50)
    )
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=200.0, commission=0.75)
    )
    r = compute_shortfall(ledger, decision_prices={o.order_id: 200.0})[0]
    assert abs(r.commission_usd - 1.25) < 1e-9


def test_missed_trade_bucket_with_partial_fill():
    """Target 100, fill 40, p0=200, p_T=210: missed = (100-40) * (210-200) = 600."""
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    ledger.append("2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 40, price=200.0, commission=0.0))
    r = compute_shortfall(
        ledger,
        decision_prices={o.order_id: 200.0},
        final_prices={"AAPL": 210.0},
    )[0]
    assert abs(r.missed_usd - 600.0) < 1e-9
    assert r.target_quantity == 100
    assert r.filled_quantity == 40


def test_bps_computation():
    """Total $ / (|target| * p0) * 1e4."""
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    # 100 shares @ 201 vs p0=200 → price_impact = 100.
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=201.0, commission=0.0)
    )
    r = compute_shortfall(ledger, decision_prices={o.order_id: 200.0})[0]
    expected_bps = 10_000.0 * 100.0 / (100 * 200.0)  # = 50 bps
    assert abs(r.total_bps - expected_bps) < 1e-6


def test_decision_prices_from_metadata():
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 50, price=201.0, commission=0.0, ref=200.0)
    )
    # Second fill has a different ref — first-fill-wins.
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 50, price=202.0, commission=0.0, ref=999.0)
    )
    dp = decision_prices_from_metadata(ledger)
    assert dp[o.order_id] == 200.0


def test_order_without_decision_price_is_skipped():
    ledger = Ledger()
    o = _submit_order(ledger, "AAPL", 100)
    ledger.append(
        "2026-04-17T16:00:00Z", "ORDER_FILLED", _fill(o, 100, price=200.0, commission=0.0)
    )
    reports = compute_shortfall(ledger, decision_prices={})
    assert reports == []


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_price_impact_zero_at_decision_price,
        test_price_impact_buy_above_decision_is_positive_cost,
        test_commission_sums_across_fills,
        test_missed_trade_bucket_with_partial_fill,
        test_bps_computation,
        test_decision_prices_from_metadata,
        test_order_without_decision_price_is_skipped,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nanalytics.shortfall: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
