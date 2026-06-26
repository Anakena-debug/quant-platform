"""quantstrat — richer performance metrics (s67): IR, rolling vol, contribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstrat.metrics.performance import (
    contribution_to_return,
    information_ratio,
    rolling_volatility,
)


def test_information_ratio_positive_when_beating_benchmark():
    rng = np.random.default_rng(0)
    bench = rng.normal(0.0003, 0.01, 252)
    active = rng.normal(0.0006, 0.003, 252)  # positive mean + nonzero tracking error
    ir = information_ratio(bench + active, bench)
    assert ir > 0
    # a CONSTANT active edge -> zero tracking error -> IR undefined (degenerate)
    with pytest.raises(ValueError):
        information_ratio(bench + 0.0004, bench)


def test_information_ratio_alignment_and_min_obs():
    with pytest.raises(ValueError):
        information_ratio(np.zeros(5), np.zeros(4))  # shape mismatch
    with pytest.raises(ValueError):
        information_ratio(np.array([0.01]), np.array([0.0]))  # < 2 obs


def test_rolling_volatility_shape_and_positivity():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
    rv = rolling_volatility(r, window=20)
    assert isinstance(rv, pd.Series) and len(rv) == 100
    assert rv.iloc[:19].isna().all()  # warmup
    assert (rv.dropna() >= 0).all()


def test_contribution_to_return():
    dates = pd.bdate_range("2024-01-01", periods=5)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
    rets = pd.DataFrame({"A": 0.02, "B": 0.0}, index=dates)  # only A earns
    contrib = contribution_to_return(weights, rets)
    # A held at 0.5 going into each of 4 post-lag days, earning 2% -> 0.5*0.02*4 = 0.04
    assert np.isclose(contrib["A"], 0.04)
    assert np.isclose(contrib["B"], 0.0)
    assert contrib["A"] > contrib["B"]
