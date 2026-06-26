"""sample_weight threading for MDA + SFI (closes CV-FI-001/002, S38).

Pre-S38, ``feature_importance_mda`` and ``feature_importance_sfi`` silently
dropped ``sample_weight``: the CV fit and scoring were unweighted, so AFML
§4.10 observation-uniqueness weights (which differ per sample under
overlapping triple-barrier labels) never reached the estimator. These tests
pin that (a) non-uniform weights now MATERIALLY change both importances
(weight actually flows into fit + scorer), and (b) the default
``sample_weight=None`` path is byte-identical to the pre-S38 behaviour
(existing pins in test_importance_mda_sem.py / test_importance_sfi_baseline.py
are the backward-compat guard; here we add a direct uniform≈None check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from quantcore.importance import feature_importance_mda, feature_importance_sfi

FEATURES = ["signal", "noise1", "noise2"]


def _fixture(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "signal": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
        }
    )
    prob = 1.0 / (1.0 + np.exp(-2.0 * X["signal"].to_numpy()))
    y = pd.Series((rng.random(n) < prob).astype(int))
    return X, y


def _model():
    return RandomForestClassifier(n_estimators=60, random_state=0)


def _cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def _imbalanced_weight(X: pd.DataFrame) -> pd.Series:
    w = np.ones(len(X), dtype=np.float64)
    w[X["signal"].to_numpy() > 0.0] = 5.0
    return pd.Series(w, index=X.index)


# -----------------------------------------------------------------------------
# MDA
# -----------------------------------------------------------------------------


def test_mda_imbalanced_weight_changes_result() -> None:
    X, y = _fixture()
    w = _imbalanced_weight(X)
    none = feature_importance_mda(_model(), X, y, _cv(), n_repeats=5, random_state=1)
    weighted = feature_importance_mda(
        _model(), X, y, _cv(), n_repeats=5, random_state=1, sample_weight=w
    )
    a = none["mean"].reindex(FEATURES).to_numpy()
    b = weighted["mean"].reindex(FEATURES).to_numpy()
    assert not np.allclose(a, b, atol=1e-6), (
        "MDA importances are identical with vs without imbalanced "
        "sample_weight — weight is being dropped (CV-FI-001 regressed)."
    )


def test_mda_uniform_weight_matches_none() -> None:
    """Uniform weights must not materially change MDA (sanity: the weighted
    path is correct, not just 'different')."""
    X, y = _fixture()
    w = pd.Series(np.ones(len(X)), index=X.index)
    none = feature_importance_mda(_model(), X, y, _cv(), n_repeats=5, random_state=1)
    uniform = feature_importance_mda(
        _model(), X, y, _cv(), n_repeats=5, random_state=1, sample_weight=w
    )
    a = none["mean"].reindex(FEATURES).to_numpy()
    b = uniform["mean"].reindex(FEATURES).to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_mda_accepts_ndarray_weight() -> None:
    X, y = _fixture()
    w = _imbalanced_weight(X).to_numpy()  # ndarray, not Series
    res = feature_importance_mda(
        _model(), X, y, _cv(), n_repeats=3, random_state=1, sample_weight=w
    )
    assert set(res.index) == set(FEATURES)


# -----------------------------------------------------------------------------
# SFI
# -----------------------------------------------------------------------------


def test_sfi_imbalanced_weight_changes_result() -> None:
    X, y = _fixture()
    w = _imbalanced_weight(X)
    none = feature_importance_sfi(_model(), X, y, _cv())
    weighted = feature_importance_sfi(_model(), X, y, _cv(), sample_weight=w)
    a = none["mean"].reindex(FEATURES).to_numpy()
    b = weighted["mean"].reindex(FEATURES).to_numpy()
    assert not np.allclose(a, b, atol=1e-6), (
        "SFI importances are identical with vs without imbalanced "
        "sample_weight — weight is being dropped (CV-FI-002 regressed)."
    )


def test_sfi_none_path_unchanged() -> None:
    """The default (sample_weight=None) keeps the cross_val_score path: two
    None calls are byte-identical, and the columns/shape are preserved."""
    X, y = _fixture()
    r1 = feature_importance_sfi(_model(), X, y, _cv())
    r2 = feature_importance_sfi(_model(), X, y, _cv(), sample_weight=None)
    assert list(r1.columns) == ["mean", "std", "mean_raw", "std_raw"]
    np.testing.assert_allclose(
        r1["mean"].reindex(FEATURES).to_numpy(),
        r2["mean"].reindex(FEATURES).to_numpy(),
        atol=1e-12,
    )


@pytest.mark.parametrize("baseline", ["prior", None, -0.5])
def test_sfi_weighted_runs_for_all_baselines(baseline) -> None:
    X, y = _fixture()
    w = _imbalanced_weight(X)
    res = feature_importance_sfi(_model(), X, y, _cv(), baseline=baseline, sample_weight=w)
    assert set(res.index) == set(FEATURES)
    assert res["std"].to_numpy().min() >= 0.0
