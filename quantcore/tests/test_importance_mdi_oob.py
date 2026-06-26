"""Regression tests for per-node OOB-evaluated MDI (S4 / P3.1, Loecher 2022).

Scope of pins (5 seeds per parametrized invariant; calibration spike
2026-04-23 confirmed stability across N in {1000, 2000, 5000}):

Primary fixture — Li-§4-style, N=2000, max_features="sqrt":
  A. Debiasing ratio: mdi_oob_raw[null] / mdi_gini[null] < 0.20
     (pre-sprint plateau ~0.144; 13σ margin from 5-seed worst case.)
  B. Null relative to OOB max: mdi_oob_raw[null] / max(mdi_oob_raw) < 0.06
     (pre-sprint ~0.033; 4.2σ margin — watch for CI flap; relax to 0.08
      if numpy/BLAS drift flags, but do NOT weaken A.)
  C. Gini bias sanity: mdi_gini[null] / max(mdi_gini) > 0.04
     (pins that the baseline IS biased; ~0.099 observed.)
  D. Top-1 rank preserved between methods (x_solo wins in both).

Secondary fixture — Strobl-pure, N=2000, y deterministic from x_solo:
  E. Null ceiling: mdi_oob[null] / mdi_oob[solo] < 0.010
     (deterministic-signal regime; ~0.003 observed.)

Schema + error-contract pins: required-X/y, bootstrap=True,
max_samples=None, estimators_ present; 4-column schema stable across
methods; sklearn_gini back-compat golden.

References — see docstring on feature_importance_mdi in
quantcore.importance.importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from quantcore.importance import feature_importance_mdi


SEEDS = [0, 1, 2, 42, 20260423]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _li_fixture(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Li-§4-style fixture: pure-noise null + rho=0.9 correlated
    informative pair + one solo informative feature. Binary y via a
    logistic-like cutoff on ``x_solo + x_corr_1 + 0.3*noise``.
    """
    rng = np.random.default_rng(seed)
    x_null = rng.standard_normal(n)
    x_corr_1 = rng.standard_normal(n)
    x_corr_2 = 0.9 * x_corr_1 + np.sqrt(0.19) * rng.standard_normal(n)
    x_solo = rng.standard_normal(n)
    logits = x_solo + x_corr_1 + 0.3 * rng.standard_normal(n)
    y = (logits > 0).astype(int)
    X = pd.DataFrame(
        {
            "x_null": x_null,
            "x_corr_1": x_corr_1,
            "x_corr_2": x_corr_2,
            "x_solo": x_solo,
        }
    )
    return X, pd.Series(y, name="y")


def _strobl_fixture(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Strobl-style pure-signal fixture: y deterministic from x_solo;
    x_null is independent N(0,1)."""
    rng = np.random.default_rng(seed)
    x_solo = rng.standard_normal(n)
    x_null = rng.standard_normal(n)
    y = (x_solo > 0).astype(int)
    X = pd.DataFrame({"x_null": x_null, "x_solo": x_solo})
    return X, pd.Series(y, name="y")


def _fit_rf(X: pd.DataFrame, y: pd.Series, seed: int) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        bootstrap=True,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    rf.fit(X, y)
    return rf


# -----------------------------------------------------------------------------
# PIN A — debiasing ratio
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pinA_debiasing_ratio_li_fixture(seed: int) -> None:
    """``mdi_oob_raw[null] / mdi_gini[null] < 0.20`` — per-node OOB
    reduces the null-feature importance vs raw sklearn Gini. Pre-sprint
    spike measured 0.137–0.148 across 5 seeds (σ≈0.004). Pin at 0.20
    leaves ~13σ margin.

    Compares raw OOB impurity decrease (``mean_raw``) to sklearn's
    normalized Gini importance (``mean``). Mixed units — the ratio
    number itself is not directly interpretable, but its stability
    across N and seeds IS the debiasing signal.
    """
    X, y = _li_fixture(2000, seed)
    rf = _fit_rf(X, y, seed)
    oob = feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")
    gini = feature_importance_mdi(rf, list(X.columns), method="sklearn_gini")
    ratio = float(oob.loc["x_null", "mean_raw"]) / float(gini.loc["x_null", "mean"])
    assert ratio < 0.20, f"seed={seed}: pin A ratio={ratio:.4f} ≥ 0.20"


# -----------------------------------------------------------------------------
# PIN B — null / max (OOB side)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pinB_oob_null_relative_to_max(seed: int) -> None:
    """``mdi_oob_raw[null] / max(mdi_oob_raw) < 0.06`` — under OOB
    evaluation, the null feature takes less than 6% of the most-
    important feature's raw impurity share. Pre-sprint: 0.027–0.039,
    σ≈0.005, 4.2σ margin. Watch for CI flap; see plan S4-p3.1 for
    relax-to-0.08 fallback.
    """
    X, y = _li_fixture(2000, seed)
    rf = _fit_rf(X, y, seed)
    oob = feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")
    null_raw = float(oob.loc["x_null", "mean_raw"])
    max_raw = float(oob["mean_raw"].max())
    ratio = null_raw / max_raw
    assert ratio < 0.06, f"seed={seed}: pin B ratio={ratio:.4f} ≥ 0.06"


# -----------------------------------------------------------------------------
# PIN C — gini bias sanity (confirms the baseline IS biased)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pinC_gini_baseline_is_biased(seed: int) -> None:
    """``mdi_gini[null] / max(mdi_gini) > 0.04`` — raw sklearn Gini
    allocates >4% of max importance to the pure-noise null feature,
    confirming the cardinality bias we're correcting. Pre-sprint:
    0.085–0.113, σ≈0.011, 4.1σ margin.
    """
    X, y = _li_fixture(2000, seed)
    rf = _fit_rf(X, y, seed)
    gini = feature_importance_mdi(rf, list(X.columns), method="sklearn_gini")
    null = float(gini.loc["x_null", "mean"])
    max_imp = float(gini["mean"].max())
    ratio = null / max_imp
    assert ratio > 0.04, (
        f"seed={seed}: pin C ratio={ratio:.4f} ≤ 0.04 (bias not detectable — fixture drift?)"
    )


# -----------------------------------------------------------------------------
# PIN D — ranking preserved
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pinD_top1_rank_preserved(seed: int) -> None:
    """Both methods rank ``x_solo`` (the strongest independent informative
    feature) as #1. Ranking-preservation is the primary downstream use
    case — absolute calibration of null importance is secondary.
    """
    X, y = _li_fixture(2000, seed)
    rf = _fit_rf(X, y, seed)
    gini = feature_importance_mdi(rf, list(X.columns), method="sklearn_gini")
    oob = feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")
    assert gini.index[0] == oob.index[0] == "x_solo", (
        f"seed={seed}: top-1 mismatch — gini={gini.index[0]}, oob={oob.index[0]}; expected x_solo."
    )


# -----------------------------------------------------------------------------
# PIN E — Strobl deterministic-signal null ceiling (secondary fixture)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pinE_strobl_null_ceiling(seed: int) -> None:
    """On the deterministic-signal fixture (``y = (x_solo > 0)``), OOB
    pins the null feature to <1% of the signal feature's importance.
    Pre-sprint: 0.002–0.003 across seeds; ~233% margin to the 0.010
    threshold. This is the strongest-signal regime; loose margin is
    expected because there's nothing for noise splits to latch onto.
    """
    X, y = _strobl_fixture(2000, seed)
    rf = _fit_rf(X, y, seed)
    oob = feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")
    ratio = float(oob.loc["x_null", "mean"]) / float(oob.loc["x_solo", "mean"])
    assert ratio < 0.010, f"seed={seed}: pin E ratio={ratio:.4f} ≥ 0.010"


# -----------------------------------------------------------------------------
# Back-compat golden — sklearn_gini path matches pre-sprint behaviour
# -----------------------------------------------------------------------------


def test_sklearn_gini_back_compat_golden() -> None:
    """On the ``sklearn_gini`` path, ``mean`` equals the bag-mean of
    ``tree.feature_importances_`` and ``std`` equals SEM (ddof=1).
    Pins the pre-sprint arithmetic so legacy callers (if any ever exist)
    read the same numbers.
    """
    X, y = _li_fixture(2000, 42)
    rf = _fit_rf(X, y, 42)
    imp = np.asarray([t.feature_importances_ for t in rf.estimators_], dtype=np.float64)
    expected_mean = imp.mean(axis=0)
    expected_std = imp.std(axis=0, ddof=1) / np.sqrt(imp.shape[0])
    df = feature_importance_mdi(rf, list(X.columns), method="sklearn_gini")
    aligned = df.loc[list(X.columns)]
    np.testing.assert_allclose(aligned["mean"].to_numpy(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(aligned["std"].to_numpy(), expected_std, atol=1e-12)


# -----------------------------------------------------------------------------
# Schema contract
# -----------------------------------------------------------------------------


def test_return_schema_stable_across_methods() -> None:
    """Both methods return the same 4-column schema: mean, std, mean_raw,
    std_raw. ``sklearn_gini`` path emits NaN on the raw columns (raw
    impurity not recoverable from ``tree.feature_importances_``).
    """
    X, y = _li_fixture(400, 0)
    rf = _fit_rf(X, y, 0)
    gini = feature_importance_mdi(rf, list(X.columns), method="sklearn_gini")
    oob = feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")
    assert list(gini.columns) == ["mean", "std", "mean_raw", "std_raw"]
    assert list(oob.columns) == ["mean", "std", "mean_raw", "std_raw"]
    # sklearn_gini emits NaN on raw columns per schema contract.
    assert gini["mean_raw"].isna().all()
    assert gini["std_raw"].isna().all()
    # oob_corrected emits numeric raw columns.
    assert not oob["mean_raw"].isna().any()
    assert not oob["std_raw"].isna().any()


# -----------------------------------------------------------------------------
# Error contract — preconditions on oob_corrected path
# -----------------------------------------------------------------------------


def test_oob_requires_X_and_y_both() -> None:
    """Missing X OR y raises ValueError naming the missing input(s)."""
    X, y = _li_fixture(200, 0)
    rf = _fit_rf(X, y, 0)
    with pytest.raises(ValueError, match=r"requires X "):
        feature_importance_mdi(rf, list(X.columns), X=None, y=y, method="oob_corrected")
    with pytest.raises(ValueError, match=r"requires y "):
        feature_importance_mdi(rf, list(X.columns), X=X, y=None, method="oob_corrected")
    with pytest.raises(ValueError, match=r"requires X and y"):
        feature_importance_mdi(rf, list(X.columns), X=None, y=None, method="oob_corrected")


def test_oob_requires_bootstrap_true() -> None:
    """Forest with bootstrap=False raises ValueError."""
    X, y = _li_fixture(200, 0)
    rf = RandomForestClassifier(n_estimators=50, bootstrap=False, random_state=0)
    rf.fit(X, y)
    with pytest.raises(ValueError, match=r"bootstrap=True"):
        feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")


def test_oob_rejects_nonnull_max_samples() -> None:
    """Custom max_samples is not supported; raises ValueError."""
    X, y = _li_fixture(200, 0)
    rf = RandomForestClassifier(n_estimators=50, bootstrap=True, max_samples=0.5, random_state=0)
    rf.fit(X, y)
    with pytest.raises(ValueError, match=r"max_samples=None"):
        feature_importance_mdi(rf, list(X.columns), X=X, y=y, method="oob_corrected")


def test_oob_requires_estimators_attr() -> None:
    """Non-bagged estimator raises TypeError naming estimators_."""
    X, y = _li_fixture(200, 0)
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    with pytest.raises(TypeError, match=r"estimators_"):
        feature_importance_mdi(clf, list(X.columns), X=X, y=y, method="oob_corrected")


def test_unknown_method_raises() -> None:
    """Unknown method values raise ValueError listing valid options."""
    X, y = _li_fixture(200, 0)
    rf = _fit_rf(X, y, 0)
    with pytest.raises(ValueError, match=r"sklearn_gini.*oob_corrected"):
        feature_importance_mdi(
            rf,
            list(X.columns),
            X=X,
            y=y,
            method="unknown",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# s83 F22 — binary-label guard on the OOB path
# ----------------------------------------------------------------------


class TestOobBinaryLabelGuard:
    """s83 F22: _gini_binary's p = mean(y) is a probability only for {0,1}
    labels. Pre-s83, this codebase's {-1,+1} triple-barrier bins flowed
    through raw — a 75/25 split scored 0.5 ('maximal'), minority-positive
    nodes went NEGATIVE, and a {-1,0,1} mix scored 0.0 ('pure'). Two-class
    codings are now auto-binarized (Gini is label-swap symmetric, so this
    is the correct math, not a workaround); >2 classes are rejected."""

    def _forest(self, y):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(len(y), 3)), columns=["a", "b", "c"])
        model = RandomForestClassifier(n_estimators=5, bootstrap=True, random_state=0)
        model.fit(X, y)
        return model, X

    def test_signed_labels_match_relabeled_01(self) -> None:
        """{-1,+1} must produce EXACTLY the importances of the same data
        relabeled {0,1} — the discriminating pin: pre-s83 the ±1 path used
        corrupted impurities and diverged."""
        rng = np.random.default_rng(1)
        y_signed = pd.Series(rng.choice([-1, 1], size=80, p=[0.3, 0.7]))
        y_01 = ((y_signed + 1) // 2).astype(int)
        model_s, X = self._forest(y_signed)
        model_b, _ = self._forest(y_01)
        out_s = feature_importance_mdi(
            model_s, ["a", "b", "c"], X=X, y=y_signed, method="oob_corrected"
        )
        out_b = feature_importance_mdi(
            model_b, ["a", "b", "c"], X=X, y=y_01, method="oob_corrected"
        )
        pd.testing.assert_frame_equal(out_s, out_b)

    def test_multiclass_labels_rejected(self) -> None:
        y = pd.Series(np.resize([-1, 0, 1], 42))
        model, X = self._forest(y)
        with pytest.raises(ValueError, match="binary classification only"):
            feature_importance_mdi(model, ["a", "b", "c"], X=X, y=y, method="oob_corrected")

    def test_binary_01_labels_accepted(self) -> None:
        y = pd.Series(np.resize([0, 1], 40))
        model, X = self._forest(y)
        out = feature_importance_mdi(model, ["a", "b", "c"], X=X, y=y, method="oob_corrected")
        assert set(out.index) == {"a", "b", "c"}
