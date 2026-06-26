import pytest

from quantengine.contracts.orders import Order
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.paper import PaperBroker
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.runtime.runner import Runner


def test_paper_broker_fills_at_adjusted_price(market):
    broker = PaperBroker(cost_model=LinearCostModel(slippage_bps=5.0, commission_min=1.0))
    order = Order.new(ticker="AAPL", signed_quantity=100)
    fills = broker.submit_orders([order], market)
    assert len(fills) == 1
    f = fills[0]
    # Buy: price bumped up by 5 bps
    assert f.price == pytest.approx(150.0 * (1 + 5e-4))
    assert f.commission >= 1.0


def test_runner_step_updates_state_and_ledger(empty_state, market, tradeable_signal):
    runner = Runner(
        state=empty_state,
        rebalance=RebalanceEngine(),
        broker=PaperBroker(),
        ledger=Ledger(),
    )
    state = runner.step(tradeable_signal, market)
    assert state.cash < empty_state.cash
    # Positions opened
    assert len(state.positions) >= 1
    # Ledger has ORDER_SUBMITTED + ORDER_FILLED entries
    kinds = [e.kind for e in runner.ledger.events()]
    assert "ORDER_SUBMITTED" in kinds
    assert "ORDER_FILLED" in kinds
