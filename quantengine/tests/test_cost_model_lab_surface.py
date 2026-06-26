"""LinearCostModel.from_lab_surface (s91 cost spine) — the measured surface, not 1bp."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.execution.cost_model import LinearCostModel

SURFACE = {
    "standing_cost_for_gates_bps": {
        "at_open_total_one_way_median": 6.347732179248889,
        "intraday_typical_total_one_way_median": 3.0237979772366854,
    }
}


@pytest.fixture()
def surface_path(tmp_path):
    p = tmp_path / "cost_model.json"
    p.write_text(json.dumps(SURFACE))
    return p


def _order(side: OrderSide) -> Order:
    return Order(
        order_id=uuid4(),
        ticker="AAPL",
        side=side,
        quantity=100,
        order_type=OrderType.MARKET,
    )


def test_at_open_discipline_sets_the_standing_one_way_total(surface_path):
    cm = LinearCostModel.from_lab_surface(surface_path)
    assert cm.slippage_bps == SURFACE["standing_cost_for_gates_bps"]["at_open_total_one_way_median"]
    # the IBKR commission defaults survive calibration (surface is price impact only)
    assert cm.commission_per_share == 0.005 and cm.commission_min == 1.00


def test_intraday_discipline(surface_path):
    cm = LinearCostModel.from_lab_surface(surface_path, discipline="intraday_typical")
    assert (
        cm.slippage_bps
        == SURFACE["standing_cost_for_gates_bps"]["intraday_typical_total_one_way_median"]
    )


def test_unknown_discipline_raises(surface_path):
    with pytest.raises(ValueError, match="discipline"):
        LinearCostModel.from_lab_surface(surface_path, discipline="vwap")


def test_fill_prices_carry_the_surface_both_sides(surface_path):
    cm = LinearCostModel.from_lab_surface(surface_path)
    ref = 100.0
    bump = ref * cm.slippage_bps / 1e4
    assert cm.fill_price(_order(OrderSide.BUY), ref) == pytest.approx(ref + bump)
    assert cm.fill_price(_order(OrderSide.SELL), ref) == pytest.approx(ref - bump)
