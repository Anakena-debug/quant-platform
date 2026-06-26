"""max_participation_check (s91) — %ADV capacity as a gated RiskCheck."""

from __future__ import annotations

import numpy as np
import pytest
from uuid import uuid4

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate, max_participation_check


def _order(ticker: str, qty: int) -> Order:
    return Order(
        order_id=uuid4(),
        ticker=ticker,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
    )


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-06-11T16:00:00",
        tickers=("LIQ", "THIN", "NOADV"),
        prices=np.array([100.0, 100.0, 100.0]),
    )


def test_participation_check_rejects_oversize_and_blind_names():
    adv = {"LIQ": 100_000_000.0, "THIN": 100_000.0}
    gate = RiskGate(checks=[max_participation_check(adv, 0.01)])
    orders = [
        _order("LIQ", 1_000),  # $100k = 0.1% of ADV -> passes
        _order("THIN", 100),  # $10k = 10% of ADV -> rejected
        _order("NOADV", 1),  # unknown liquidity -> rejected (fail-closed)
    ]
    accepted, rejected = gate.validate(orders, PortfolioState.empty(1e9), _market())
    assert [o.ticker for o in accepted] == ["LIQ"]
    reasons = {r.order.ticker: r.reason for r in rejected}
    assert "ADV" in reasons["THIN"] and "cap" in reasons["THIN"]
    assert "refusing to size blind" in reasons["NOADV"]


def test_participation_bounds_validated():
    with pytest.raises(ValueError):
        max_participation_check({}, 0.0)
