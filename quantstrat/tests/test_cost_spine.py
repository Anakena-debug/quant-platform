"""s91 cost spine through the handoff harness: run_backtest(broker=, risk_gate=).

Pins (1) backtest fills price at the MEASURED surface when the calibrated broker is passed
(the deployment-handoff form replacing the 1bp default), and (2) the attached RiskGate is
the same rejection stage the live loop runs — a gate that rejects everything produces a
fill-free run, recorded, not silently skipped.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantengine.contracts.signal import build_alpha_signal
from quantengine.execution.cost_model import LinearCostModel
from quantengine.execution.paper import PaperBroker
from quantengine.risk.gate import RiskGate, max_order_notional_check
from quantengine.strategies.base import Strategy

from quantstrat.backtest import run_backtest

STANDING_AT_OPEN = 6.347732179248889  # bps; the s86 measured one-way total


class _AllInOne(Strategy):
    """Buys one name with 50% of capital — a deterministic single-fill probe."""

    def predict(self, market):  # noqa: ANN001
        n = len(market.tickers)
        w = [0.5] + [0.0] * (n - 1)
        return build_alpha_signal(
            tickers=market.tickers,
            expected_return=[1e-6] * n,
            lower=[1e-6 if x > 0 else -1e-6 for x in w],
            upper=[1.0 if x > 0 else 1e-6 for x in w],
            alpha=0.1,
            kelly_weights=w,
            timestamp=market.timestamp,
        )


def _flat_panel(n_days: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=dates, dtype=np.float64)


@pytest.fixture()
def calibrated_broker(tmp_path) -> PaperBroker:
    p = tmp_path / "cost_model.json"
    p.write_text(
        json.dumps(
            {"standing_cost_for_gates_bps": {"at_open_total_one_way_median": STANDING_AT_OPEN}}
        )
    )
    return PaperBroker(cost_model=LinearCostModel.from_lab_surface(p))


def test_backtest_fills_at_the_measured_surface(calibrated_broker):
    res = run_backtest(_AllInOne(), _flat_panel(), broker=calibrated_broker)
    fills = res.run_frames["fills"]
    buys = fills[fills["signed_quantity"] > 0]
    assert len(buys) > 0
    # handoff pin: fill price == ref * (1 + standing/1e4), byte-derived from the JSON
    expected = 100.0 * (1.0 + STANDING_AT_OPEN / 1e4)
    assert buys["price"].iloc[0] == pytest.approx(expected, rel=1e-12)


def test_backtest_default_broker_is_still_the_uncalibrated_1bp():
    # Opt into the optimistic default explicitly — this test pins the 1bp default pricing,
    # which the fail-closed guard otherwise refuses.
    res = run_backtest(_AllInOne(), _flat_panel(), allow_optimistic_costs=True)
    fills = res.run_frames["fills"]
    buys = fills[fills["signed_quantity"] > 0]
    assert buys["price"].iloc[0] == pytest.approx(100.0 * (1.0 + 1.0 / 1e4), rel=1e-12)


def test_risk_gate_attaches_to_the_replay_loop():
    gate = RiskGate(checks=[max_order_notional_check(1.0)])  # rejects every real order
    res = run_backtest(_AllInOne(), _flat_panel(), risk_gate=gate, allow_optimistic_costs=True)
    assert len(res.run_frames["fills"]) == 0
    # the rejection stage RAN (events journaled), it did not silently vanish
    lifecycle = res.run_frames["lifecycle"]
    assert (lifecycle["kind"] == "ORDER_REJECTED").any()
