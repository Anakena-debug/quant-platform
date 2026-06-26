"""quantstrat — multi-factor regression attribution (s75).

Pins ``factor_attribution`` as a strict generalisation of the single-benchmark
``relative_metrics`` beta/alpha: it recovers known loadings, matches ``relative_metrics``
exactly for one factor at ``rf=0``, degrades gracefully (NaN) on a rank-deficient design,
and aligns a factor frame the same way ``relative_metrics_from_series`` aligns a benchmark.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantstrat.metrics.performance import (
    FactorAttribution,
    factor_attribution,
    factor_attribution_from_frame,
    relative_metrics,
)


def test_recovers_known_betas_alpha_rsquared_noiseless():
    rng = np.random.default_rng(1)
    f1, f2 = rng.normal(size=500), rng.normal(size=500)
    y = 0.001 + 1.5 * f1 - 0.5 * f2  # exact linear combo, alpha = 0.001 / bar
    fa = factor_attribution(y, {"f1": f1, "f2": f2})
    assert isinstance(fa, FactorAttribution)
    assert np.isclose(fa.betas["f1"], 1.5) and np.isclose(fa.betas["f2"], -0.5)
    assert np.isclose(fa.alpha, 0.001 * 252)
    assert fa.r_squared > 0.99999 and fa.residual_volatility < 1e-6
    assert fa.n_obs == 500
    assert list(fa.betas) == ["f1", "f2"]  # key order mirrors the factor order


def test_single_factor_matches_relative_metrics_at_rf0():
    # The invariant: factor_attribution generalises relative_metrics — for ONE factor at rf=0
    # the OLS slope/intercept equal the cov/var beta and Jensen alpha (single source of truth).
    rng = np.random.default_rng(2)
    b = rng.normal(size=400)
    r = 0.9 * b + 0.015 * rng.normal(size=400)
    fa = factor_attribution(r, {"bench": b}, rf=0.0)
    rm = relative_metrics(r, b, rf=0.0)
    assert np.isclose(fa.betas["bench"], rm.beta)
    assert np.isclose(fa.alpha, rm.alpha)


def test_noisy_recovers_betas_rsquared_below_one_finite_tstats():
    rng = np.random.default_rng(3)
    f1, f2 = rng.normal(size=800), rng.normal(size=800)
    y = 1.2 * f1 - 0.7 * f2 + 0.5 * rng.normal(size=800)
    fa = factor_attribution(y, {"f1": f1, "f2": f2})
    assert np.isclose(fa.betas["f1"], 1.2, atol=0.05)
    assert np.isclose(fa.betas["f2"], -0.7, atol=0.05)
    assert 0.0 < fa.r_squared < 1.0 and fa.residual_volatility > 0.0
    assert np.isfinite(fa.alpha_tstat) and np.isfinite(fa.beta_tstats["f1"])
    assert abs(fa.beta_tstats["f1"]) > 10.0  # a strong loading is highly significant


def test_collinear_and_constant_factors_yield_nan():
    rng = np.random.default_rng(4)
    f1 = rng.normal(size=300)
    y = f1 + 0.1 * rng.normal(size=300)
    # a duplicated (collinear) factor → rank-deficient → attribution undefined, not a crash
    fc = factor_attribution(y, {"f1": f1, "dup": 3.0 * f1})
    assert math.isnan(fc.betas["f1"]) and math.isnan(fc.betas["dup"])
    assert math.isnan(fc.r_squared) and math.isnan(fc.alpha)
    # a constant factor is collinear with the intercept → also rank-deficient
    fk = factor_attribution(y, {"f1": f1, "const": np.full(300, 2.0)})
    assert math.isnan(fk.betas["const"]) and math.isnan(fk.r_squared)


def test_guards():
    f = np.arange(50.0)
    with pytest.raises(ValueError, match="at least one factor"):
        factor_attribution(f, {})
    with pytest.raises(ValueError, match="must align"):
        factor_attribution(f, {"a": np.arange(49.0)})
    with pytest.raises(ValueError, match=r"need >= K\+2"):  # n=3, K=2 → need 4
        factor_attribution(np.arange(3.0), {"a": np.arange(3.0), "b": np.arange(3.0)})


def test_from_frame_aligns_drops_nonfinite_and_threshold():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-01-01", periods=300, freq="D")
    f1, f2 = rng.normal(size=300), rng.normal(size=300)
    returns = pd.Series(0.5 * f1 + 0.25 * f2, index=idx)
    factors = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)
    # a non-finite factor cell and a missing return → both rows dropped, betas still exact
    factors.iloc[10, 0] = np.nan
    returns.iloc[20] = np.nan
    fa = factor_attribution_from_frame(returns, factors)
    assert fa is not None
    assert np.isclose(fa.betas["f1"], 0.5) and np.isclose(fa.betas["f2"], 0.25)
    assert fa.n_obs == 298  # 300 - 2 dropped rows

    # below K+2 aligned rows → None (mirrors relative_metrics_from_series)
    assert factor_attribution_from_frame(returns.iloc[:3], factors.iloc[:3]) is None
    # type guard: an ndarray is not a Series (intentional wrong type → runtime TypeError)
    with pytest.raises(TypeError):
        factor_attribution_from_frame(returns.to_numpy(), factors)  # pyright: ignore[reportArgumentType]
