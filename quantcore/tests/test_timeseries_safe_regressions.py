"""Regression tests for uncertainty/conformal/timeseries_safe.py P0 fixes.

Covered
-------
*   Parameter validation.
*   Asymmetric signed-residual interval under skewed noise.
*   Studentized-abs-residual narrows conditional-coverage gap under
    heteroskedasticity.
*   CQR non-crossing enforcement.
*   Rolling-window ACI score-buffer cap.
*   update() accepts 1-D ndarray / pd.Series / pd.DataFrame.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from quantcore.uncertainty.conformal.timeseries_safe import TimeSeriesConformal, ACIRegressor


class RidgeLite:
    """Minimal self-contained ridge estimator (no sklearn dependency)."""

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        k = X.shape[1]
        self.beta_ = np.linalg.solve(X.T @ X + self.lam * np.eye(k), X.T @ y)
        resid = y - X @ self.beta_
        self.sigma_ = float(np.std(resid))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.beta_

    def predict_scale(self, X):
        return np.full(len(X), self.sigma_)


class HeteroRidge(RidgeLite):
    def predict_scale(self, X):
        return 1.0 + np.abs(np.asarray(X, dtype=float)[:, 0])


class CrossingQR(RidgeLite):
    def predict_quantiles(self, X):
        base = self.predict(X)
        lo = base - 1.0
        hi = base + 1.0
        # Swap on even rows to force crossing
        swap = np.arange(len(X)) % 2 == 0
        q_lo = np.where(swap, hi, lo)
        q_hi = np.where(swap, lo, hi)
        return q_lo, q_hi


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(alpha=-0.1), "alpha"),
        (dict(alpha=1.5), "alpha"),
        (dict(method="unknown"), "method"),
        (dict(method="rolling", window=1), "window"),
        (dict(method="block", block_size=1), "block_size"),
        (dict(method="aci", aci_gamma=0.0), "aci_gamma"),
        (dict(score="garbage"), "score"),
    ],
)
def test_parameter_validation(kwargs, msg):
    defaults = dict(alpha=0.1, method="split")
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        TimeSeriesConformal(estimator=RidgeLite(), **defaults)


def test_signed_residual_is_asymmetric_under_skew():
    rng = np.random.default_rng(42)
    T = 2000
    X = rng.standard_normal((T, 3))
    y = X @ np.array([0.5, -0.3, 0.2]) + (rng.exponential(1.0, T) - 1.0)
    cp = TimeSeriesConformal(
        estimator=RidgeLite(),
        alpha=0.1,
        method="split",
        score="signed_residual",
    )
    cp.fit(X[:1500], y[:1500], cal_size=400)
    assert cp.q_hi_score_ != -cp.q_lo_score_  # asymmetric
    ratio = abs(cp.q_hi_score_ / cp.q_lo_score_)
    assert abs(ratio - 1.0) > 0.1  # meaningfully asymmetric
    _, lo, hi = cp.predict(X[1500:])
    cov = ((y[1500:] >= lo) & (y[1500:] <= hi)).mean()
    assert cov >= 0.85


def test_studentized_reduces_conditional_coverage_gap():
    rng = np.random.default_rng(7)
    T = 3000
    X = rng.standard_normal((T, 3))
    vol = 0.3 + 1.5 * np.abs(X[:, 0])
    y = X @ np.array([0.5, -0.3, 0.2]) + vol * rng.standard_normal(T)

    cp_plain = TimeSeriesConformal(
        estimator=HeteroRidge(),
        alpha=0.1,
        method="split",
        score="abs_residual",
    )
    cp_plain.fit(X[:2000], y[:2000], cal_size=500)
    _, lo_p, hi_p = cp_plain.predict(X[2000:])

    cp_stud = TimeSeriesConformal(
        estimator=HeteroRidge(),
        alpha=0.1,
        method="split",
        score="studentized_abs_residual",
    )
    cp_stud.fit(X[:2000], y[:2000], cal_size=500)
    _, lo_s, hi_s = cp_stud.predict(X[2000:])

    vol_test = 0.3 + 1.5 * np.abs(X[2000:, 0])
    hi_v = vol_test > np.median(vol_test)
    lo_v = ~hi_v
    cov_p_lo = ((y[2000:][lo_v] >= lo_p[lo_v]) & (y[2000:][lo_v] <= hi_p[lo_v])).mean()
    cov_p_hi = ((y[2000:][hi_v] >= lo_p[hi_v]) & (y[2000:][hi_v] <= hi_p[hi_v])).mean()
    cov_s_lo = ((y[2000:][lo_v] >= lo_s[lo_v]) & (y[2000:][lo_v] <= hi_s[lo_v])).mean()
    cov_s_hi = ((y[2000:][hi_v] >= lo_s[hi_v]) & (y[2000:][hi_v] <= hi_s[hi_v])).mean()

    gap_plain = abs(cov_p_hi - cov_p_lo)
    gap_stud = abs(cov_s_hi - cov_s_lo)
    # Studentized should not be worse; tolerate small finite-sample noise
    assert gap_stud <= gap_plain + 0.03


def test_cqr_non_crossing_enforced():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 3))
    y = X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(400) * 0.5
    cp = TimeSeriesConformal(
        estimator=CrossingQR(),
        alpha=0.1,
        method="split",
        score="cqr",
    )
    cp.fit(X[:300], y[:300], cal_size=80)
    _, lo, hi = cp.predict(X[300:])
    assert (hi >= lo).all()


def test_rolling_aci_caps_buffer():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((2500, 3))
    y = X @ np.array([0.5, -0.3, 0.2]) + 0.3 * rng.standard_normal(2500)

    aci = ACIRegressor(RidgeLite(), alpha=0.1, gamma=0.02, aci_window=200)
    aci.fit(X[:2000], y[:2000], cal_size=500)
    assert aci.aci_window == 200
    for t in range(2000, 2400):
        _, _, _ = aci.predict(X[t : t + 1])
        aci.update(X[t : t + 1], y[t])
    slice_ = aci._slice_for_method()
    assert slice_.size == 200
    assert aci.scores_.size > 200


def test_update_accepts_multiple_shapes():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((1200, 3))
    y = X @ np.array([0.5, -0.3, 0.2]) + 0.3 * rng.standard_normal(1200)
    cp = ACIRegressor(RidgeLite(), alpha=0.1, gamma=0.02)
    cp.fit(X[:1000], y[:1000], cal_size=200)

    n0 = cp.n_cal_
    # 1-D ndarray
    cp.update(X[1000], y[1000])
    # pd.Series
    cp.update(pd.Series(X[1001], index=[f"f{i}" for i in range(3)]), y[1001])
    # 1-row pd.DataFrame
    cp.update(
        pd.DataFrame([X[1002]], columns=[f"f{i}" for i in range(3)]),
        y[1002],
    )
    # 2-D ndarray
    cp.update(X[1003:1004], y[1003])
    assert cp.n_cal_ == n0 + 4
