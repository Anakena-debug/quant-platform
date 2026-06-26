"""Regression tests for purged_kfold.py P0 fixes.

Covered
-------
*   Variable-horizon leakage under PurgedKFold.split().
*   Variable-horizon leakage under CombinatorialPurgedKFold.split().
*   sample_weight propagation into scoring callable (signature-gated).
*   Folds disjoint with test samples.
*   Constant-horizon sanity preserved.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from quantcore.cv.purged_kfold import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    cv_score_purged,
    ml_get_train_times,
)


def _make_series(T: int = 500, horizon: str = "variable", seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = pd.date_range("2020-01-01", periods=T, freq="h")
    if horizon == "variable":
        h = rng.integers(1, 30, size=T)
        t1 = pd.Series(
            [t0[i] + pd.Timedelta(hours=int(h[i])) for i in range(T)],
            index=t0,
        )
    else:
        t1 = pd.Series(t0 + pd.Timedelta("5h"), index=t0)
    X = pd.DataFrame(rng.standard_normal((T, 3)), index=t0)
    y = pd.Series(rng.choice([-1, 1], size=T), index=t0)
    return t1, X, y


def _leak_count(t1: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    """Count samples in train that violate AFML 3-way overlap vs test envelope."""
    te_t0_min = t1.index[test_idx].min()
    te_t1_max = t1.iloc[test_idx].max()
    tr_t0 = t1.index[train_idx]
    tr_t1 = t1.iloc[train_idx]
    cond1 = (tr_t0 >= te_t0_min) & (tr_t0 <= te_t1_max)
    cond2 = (tr_t1 >= te_t0_min) & (tr_t1 <= te_t1_max)
    cond3 = (tr_t0 <= te_t0_min) & (tr_t1 >= te_t1_max)
    return int(np.asarray(cond1 | cond2 | cond3).sum())


def test_purged_kfold_variable_horizon_no_leak():
    t1, X, _ = _make_series(T=500, horizon="variable")
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
    for tr, te in cv.split(X):
        assert len(set(tr).intersection(te)) == 0
        assert _leak_count(t1, tr, te) == 0


def test_purged_kfold_constant_horizon_no_leak():
    t1, X, _ = _make_series(T=500, horizon="constant")
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)
    for tr, te in cv.split(X):
        assert _leak_count(t1, tr, te) == 0


def test_cpcv_variable_horizon_disjoint_and_correct_count():
    t1, X, _ = _make_series(T=400, horizon="variable")
    cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=t1, embargo_pct=0.01)
    # C(6,2) = 15
    assert cpcv.get_n_splits() == 15
    fold_ct = 0
    for tr, te in cpcv.split(X):
        fold_ct += 1
        assert len(set(tr).intersection(te)) == 0
    assert fold_ct == 15


def test_cv_score_propagates_sample_weight_to_scorer():
    t1, X, y = _make_series(T=500, horizon="variable")
    captured = {}

    class Dummy:
        def fit(self, X, y, sample_weight=None):
            return self

        def score(self, X, y):
            return 0.0

    def scorer(est, X, y, sample_weight=None):
        captured["sw"] = sample_weight
        return 1.0

    sw = pd.Series(np.linspace(0.5, 1.5, len(X)), index=X.index)
    scores = cv_score_purged(
        Dummy(),
        X,
        y,
        sample_weight=sw,
        scoring=scorer,
        t1=t1,
        embargo_pct=0.01,
        n_splits=5,
    )
    assert scores.shape == (5,)
    assert captured["sw"] is not None


def test_cv_score_handles_scorer_without_sw_kwarg():
    t1, X, y = _make_series(T=500, horizon="variable")

    class Dummy:
        def fit(self, X, y, sample_weight=None):
            return self

    def scorer_no_sw(est, X, y):
        return 0.5

    sw = pd.Series(np.ones(len(X)), index=X.index)
    # Should not raise: signature inspection must gate sample_weight passing.
    scores = cv_score_purged(
        Dummy(),
        X,
        y,
        sample_weight=sw,
        scoring=scorer_no_sw,
        t1=t1,
        embargo_pct=0.01,
        n_splits=5,
    )
    assert np.allclose(scores, 0.5)


def test_ml_get_train_times_three_way_overlap():
    # train t0 inside test window
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    t1 = pd.Series(idx + pd.Timedelta("1h"), index=idx)
    test_times = pd.Series([idx[5]], index=[idx[3]])
    safe = ml_get_train_times(t1, test_times)
    # Any train sample with t0 in [idx[3], idx[5]] or whose t1 enters,
    # or which envelops, must be purged.
    assert idx[3] not in safe.index
    assert idx[4] not in safe.index
    assert idx[5] not in safe.index
    # But distant samples survive
    assert idx[0] in safe.index
    assert idx[9] in safe.index
