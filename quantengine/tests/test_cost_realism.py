"""quantengine.execution.cost_model — the cost-realism guard (realistic preset + optimism flag)."""

from __future__ import annotations

from quantengine.execution.cost_model import LinearCostModel


def test_default_cost_model_is_optimistic():
    # The bare 1bp default under-charges for real execution and must be flagged.
    assert LinearCostModel().is_optimistic()


def test_realistic_preset_is_not_optimistic():
    m = LinearCostModel.realistic()
    assert m.slippage_bps == 5.0 and not m.is_optimistic()


def test_optimism_floor_is_tunable():
    cm = LinearCostModel(slippage_bps=3.0)
    assert not cm.is_optimistic()  # 3bp clears the default 2bp floor
    assert cm.is_optimistic(slippage_floor_bps=5.0)  # but not a stricter 5bp floor


def test_assumptions_dict_round_trips_the_params():
    assert LinearCostModel.realistic().assumptions() == {
        "slippage_bps": 5.0,
        "commission_per_share": 0.005,
        "commission_min": 1.00,
    }


def test_realistic_keeps_commission_defaults_but_raises_slippage():
    default = LinearCostModel()
    realistic = LinearCostModel.realistic()
    assert realistic.slippage_bps > default.slippage_bps
    assert realistic.commission_per_share == default.commission_per_share
