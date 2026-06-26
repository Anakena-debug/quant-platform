"""quantstrat — risk & tail analytics (s68): tail_metrics + drawdown_table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantstrat.metrics.performance import (
    TailMetrics,
    drawdown_table,
    max_drawdown,
    tail_metrics,
)

# ---------------------------------------------------------------------------
# tail_metrics
# ---------------------------------------------------------------------------


def test_tail_metrics_var_cvar_historical_and_signed():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 2000)
    tm = tail_metrics(x, level=0.95)
    assert isinstance(tm, TailMetrics)
    assert tm.level == 0.95
    # VaR is the 5th-percentile empirical return; CVaR is the mean of the tail below it
    assert np.isclose(tm.var, np.quantile(x, 0.05))
    assert np.isclose(tm.cvar, x[x <= tm.var].mean())
    assert tm.cvar <= tm.var < 0.0  # expected shortfall at least as extreme; real left tail
    assert tm.best == x.max() and tm.worst == x.min()
    assert 0.0 <= tm.hit_rate <= 1.0


def test_tail_metrics_skew_kurtosis_match_scipy_and_pandas():
    rng = np.random.default_rng(7)
    x = rng.standard_t(5, size=1500) * 0.01  # heavy-tailed, mild sample asymmetry
    tm = tail_metrics(x)
    assert np.isclose(tm.skew, stats.skew(x, bias=False))
    assert np.isclose(tm.excess_kurtosis, stats.kurtosis(x, fisher=True, bias=False))
    s = pd.Series(x)
    assert np.isclose(tm.skew, s.skew())
    assert np.isclose(tm.excess_kurtosis, s.kurt())


def test_tail_metrics_degenerate_and_short_series():
    const = tail_metrics(np.full(50, 0.001))
    assert np.isnan(const.skew) and np.isnan(const.excess_kurtosis)  # ~zero dispersion
    assert const.best == const.worst == pytest.approx(0.001)
    assert const.hit_rate == 1.0
    assert const.var == pytest.approx(0.001)  # var/cvar still defined on a constant series

    two = tail_metrics(np.array([0.01, -0.02]))
    assert np.isnan(two.skew) and np.isnan(two.excess_kurtosis)  # < 3 / < 4 obs
    assert two.worst == -0.02 and two.best == 0.01

    three = tail_metrics(np.array([0.01, -0.02, 0.0]))
    assert np.isfinite(three.skew)  # G1 defined at n == 3
    assert np.isnan(three.excess_kurtosis)  # G2 still needs n >= 4


def test_tail_metrics_guards():
    with pytest.raises(ValueError):
        tail_metrics(np.array([]))  # empty
    for bad_level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            tail_metrics(np.zeros(10), level=bad_level)


# ---------------------------------------------------------------------------
# drawdown_table
# ---------------------------------------------------------------------------

_COLUMNS = ["magnitude", "peak", "trough", "recovery", "depth_periods", "recovery_periods"]


def test_drawdown_table_ranks_and_decomposes_episodes():
    # Two distinct, recovered drawdowns: a shallow early one then a deeper later one.
    r = np.array(
        [
            0.10,  # 0: peak A
            -0.05,  # 1
            -0.05,  # 2: trough of DD1
            0.11,  # 3: recover above A -> new high
            0.10,  # 4: peak B
            -0.20,  # 5
            -0.10,  # 6: trough of DD2 (deepest)
            0.05,  # 7
            0.40,  # 8: full recovery above B
        ]
    )
    tbl = drawdown_table(r, top_n=5)
    assert list(tbl.columns) == _COLUMNS
    assert len(tbl) == 2
    assert tbl["magnitude"].iloc[0] < tbl["magnitude"].iloc[1] < 0.0  # worst first

    worst = tbl.iloc[0]
    assert worst["peak"] == 4 and worst["trough"] == 6 and worst["recovery"] == 8
    assert worst["depth_periods"] == 2 and worst["recovery_periods"] == 2.0

    # worst row agrees with the scalar max_drawdown primitive
    mdd = max_drawdown(r)
    assert np.isclose(worst["magnitude"], mdd.magnitude)
    assert worst["trough"] == mdd.trough_index


def test_drawdown_table_unrecovered_tail_and_series_labels():
    dates = pd.bdate_range("2024-01-01", periods=5)
    r = pd.Series([0.05, 0.05, -0.10, -0.10, -0.05], index=dates)  # falls, never recovers
    tbl = drawdown_table(r)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["peak"] == dates[1]  # high-water at idx 1
    assert isinstance(row["trough"], pd.Timestamp)
    assert row["recovery"] is None and np.isnan(row["recovery_periods"])


def test_drawdown_table_no_drawdown_is_empty():
    up = np.array([0.01, 0.02, 0.01, 0.03])  # monotone-up equity — never underwater
    tbl = drawdown_table(up)
    assert tbl.empty and list(tbl.columns) == _COLUMNS


def test_drawdown_table_top_n_caps_to_worst_rows():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.02, 500)
    full = drawdown_table(r, top_n=10_000)
    capped = drawdown_table(r, top_n=3)
    assert len(capped) == min(3, len(full))
    assert list(capped["magnitude"]) == list(full["magnitude"].iloc[:3])  # the worst three
    with pytest.raises(ValueError):
        drawdown_table(r, top_n=0)
