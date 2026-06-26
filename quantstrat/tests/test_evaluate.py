"""quantstrat — deflated/multiple-testing evaluation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantstrat.backtest.evaluate import compare_strategies, deflated_evaluation


def _returns(mu: float, sigma: float, n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def test_deflated_evaluation_basic_fields():
    ev = deflated_evaluation(_returns(0.0008, 0.01), n_trials=1, n_windows=4)
    assert ev.sharpe > 0
    assert 0.0 <= ev.psr <= 1.0
    assert 0.0 <= ev.dsr <= 1.0
    assert len(ev.window_sharpes) == 4  # per-window stability breakdown
    assert isinstance(ev.is_significant, bool)


def test_more_trials_deflate_significance():
    # same returns; more trials considered -> DSR must not exceed the n_trials=1 case.
    r = _returns(0.0008, 0.01)
    dsr_1 = deflated_evaluation(r, n_trials=1).dsr
    dsr_many = deflated_evaluation(r, n_trials=50, sr_std_cross_trial=0.5).dsr
    assert dsr_many <= dsr_1  # deflation penalises multiple testing


def test_short_series_raises():
    import pytest

    with pytest.raises(ValueError):
        deflated_evaluation(pd.Series([0.01, 0.02, -0.01]), n_trials=1)  # < 4 obs


def test_compare_strategies_ranks_by_dsr():
    rets = {
        "strong": _returns(0.0012, 0.008, seed=1),  # high SR
        "weak": _returns(0.0002, 0.012, seed=2),  # low SR
        "noise": _returns(0.00005, 0.015, seed=3),  # ~zero SR
    }
    df = compare_strategies(rets)
    assert list(df.columns) >= ["strategy", "sharpe", "psr", "dsr", "n_trials", "is_significant"]
    assert (df["n_trials"] == 3).all()  # K strategies = K trials
    assert df.iloc[0]["strategy"] == "strong"  # ranked by DSR desc
    assert df.iloc[0]["dsr"] >= df.iloc[-1]["dsr"]
