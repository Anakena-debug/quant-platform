"""Regression tests for P3.2 — unified dispersion estimator (SEM with
ddof=1) across MDI / MDA / SFI.

The law of total variance applied to MDA:

    Var(Î) = Var_f(E_r[Î_{f,r}])     between-fold
           + E_f[Var_r(Î_{f,r} | f)]  within-fold permutation noise

Pre-sprint, MDA's ``np.std(flat_list)`` over ``n_folds × n_repeats``
collapsed these two components and treated ``n = F·R`` as the sample
count. That underestimates the true dispersion whenever within-fold
permutation noise is non-trivial, and leaves ``importance_gate`` with
an inconsistent effective CI across MDI / MDA / SFI.

Post-P3.2:
  - MDA collapses within-fold repeats first, then reports SEM across
    folds with ``ddof=1``.
  - SFI reports SEM across folds with ``ddof=1`` (was pooled std).
  - MDI (sklearn_gini path, already set in P3.1) reports SEM across
    trees with ``ddof=1``.

Pins numeric SEM formulas directly against independent recomputation
and confirms ``mean`` is unchanged under the restructure (fold-mean-
of-means equals flat-list mean when repeats per fold are balanced).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from quantcore.importance import (
    feature_importance_mda,
    feature_importance_sfi,
    importance_gate,
)


def _fixture(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
        }
    )
    # `a` is informative; `b`, `c` are pure noise.
    y = pd.Series((X["a"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# -----------------------------------------------------------------------------
# MDA — fold-mean SEM with ddof=1
# -----------------------------------------------------------------------------


def test_mda_sem_on_fold_means_ddof1() -> None:
    """``std`` column equals ``std_{folds}(fold_mean, ddof=1) / sqrt(n_folds)``.

    Collapses repeats first (within-fold mean) and computes unbiased
    SEM across folds. Reproduces the fold-mean pipeline independently
    and pins within atol=1e-12.
    """
    X, y = _fixture(300, 0)
    n_folds, n_repeats = 5, 3
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    model = DecisionTreeClassifier(random_state=0)

    df = feature_importance_mda(
        model,
        X,
        y,
        cv=cv,
        n_repeats=n_repeats,
        random_state=0,
    )

    # Independent recomputation of fold-means from the same seed.
    scorer = get_scorer("neg_log_loss")
    rng = np.random.default_rng(0)
    fold_means: dict[str, list[float]] = {c: [] for c in X.columns}
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = scorer(m, X.iloc[te], y.iloc[te])
        for c in X.columns:
            repeats: list[float] = []
            for _ in range(n_repeats):
                Xp = X.iloc[te].copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                repeats.append(float(base - scorer(m, Xp, y.iloc[te])))
            fold_means[c].append(float(np.mean(repeats)))

    for c in X.columns:
        expected_mean = float(np.mean(fold_means[c]))
        expected_sem = float(np.std(fold_means[c], ddof=1) / np.sqrt(n_folds))
        assert df.loc[c, "mean"] == pytest.approx(expected_mean, abs=1e-12), (
            f"[{c}] MDA mean mismatch: got {df.loc[c, 'mean']}, expected {expected_mean}"
        )
        assert df.loc[c, "std"] == pytest.approx(expected_sem, abs=1e-12), (
            f"[{c}] MDA SEM mismatch: got {df.loc[c, 'std']}, expected {expected_sem}"
        )


def test_mda_mean_invariant_under_restructure() -> None:
    """The fold-mean-of-means equals the flat-list mean — the restructure
    changes only the dispersion computation, not the central-tendency
    estimate. Pins that mean is unchanged vs the pre-sprint flat list.
    """
    X, y = _fixture(300, 1)
    n_folds, n_repeats = 5, 3
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    model = DecisionTreeClassifier(random_state=1)

    df = feature_importance_mda(
        model,
        X,
        y,
        cv=cv,
        n_repeats=n_repeats,
        random_state=1,
    )

    # Pre-sprint flat-list mean computation:
    scorer = get_scorer("neg_log_loss")
    rng = np.random.default_rng(1)
    flat: dict[str, list[float]] = {c: [] for c in X.columns}
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = scorer(m, X.iloc[te], y.iloc[te])
        for c in X.columns:
            for _ in range(n_repeats):
                Xp = X.iloc[te].copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                flat[c].append(float(base - scorer(m, Xp, y.iloc[te])))
    for c in X.columns:
        flat_mean = float(np.mean(flat[c]))
        assert df.loc[c, "mean"] == pytest.approx(flat_mean, abs=1e-12), (
            f"[{c}] mean should be invariant to fold-vs-flat restructure"
        )


def test_mda_sem_exceeds_pre_sprint_flat_sem() -> None:
    """On a fixture with non-trivial within-fold permutation noise, the
    fold-mean SEM (post-sprint) is ≥ the pre-sprint flat-list SEM (the
    buggy one). This pins that the fix is in the *conservative*
    direction — the old SEM was too small.
    """
    X, y = _fixture(200, 2)
    n_folds, n_repeats = 5, 3
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=2)
    model = DecisionTreeClassifier(random_state=2)

    df = feature_importance_mda(
        model,
        X,
        y,
        cv=cv,
        n_repeats=n_repeats,
        random_state=2,
    )

    # Compute the pre-sprint (buggy) flat-list SEM for comparison.
    scorer = get_scorer("neg_log_loss")
    rng = np.random.default_rng(2)
    flat: dict[str, list[float]] = {c: [] for c in X.columns}
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        base = scorer(m, X.iloc[te], y.iloc[te])
        for c in X.columns:
            for _ in range(n_repeats):
                Xp = X.iloc[te].copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                flat[c].append(float(base - scorer(m, Xp, y.iloc[te])))

    # At least one feature should show the fold-mean SEM being larger
    # than the flat-list SEM. (Strict equality is possible on fixtures
    # with zero within-fold variance but that's pathological.)
    larger_count = 0
    for c in X.columns:
        flat_sem = float(np.std(flat[c], ddof=0) / np.sqrt(len(flat[c])))
        if df.loc[c, "std"] > flat_sem - 1e-12:
            larger_count += 1
    assert larger_count >= 2, (
        "Fold-mean SEM should meet or exceed flat-list SEM on most "
        "features (law-of-total-variance: fold-mean SEM ≥ pooled-flat "
        "SEM whenever within-fold variance > 0)."
    )


# -----------------------------------------------------------------------------
# SFI — SEM with ddof=1
# -----------------------------------------------------------------------------


def test_sfi_sem_with_ddof1() -> None:
    """SFI ``std`` column equals ``std(fold_scores, ddof=1) / sqrt(n_folds)``.

    Was pre-P3.2 ``s.std()`` (ddof=0, no SEM division). Pins the P3.2
    fix on the SEM arithmetic.

    Post-S7, SFI's ``mean`` is baseline-adjusted (``mean_raw − baseline``)
    so the raw-CV-score arithmetic pin reads from the new ``mean_raw``
    column instead. ``std`` is invariant under the baseline subtraction
    by the constant-shift identity (``std(X − c) = std(X)``).
    """
    X, y = _fixture(300, 0)
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    model = DecisionTreeClassifier(random_state=0)

    df = feature_importance_sfi(model, X, y, cv=cv)

    for c in X.columns:
        s = cross_val_score(clone(model), X[[c]], y, cv=cv, scoring="neg_log_loss")
        expected_mean = float(s.mean())
        expected_sem = float(s.std(ddof=1) / np.sqrt(len(s)))
        assert df.loc[c, "mean_raw"] == pytest.approx(expected_mean, abs=1e-12)
        assert df.loc[c, "std"] == pytest.approx(expected_sem, abs=1e-12)


# -----------------------------------------------------------------------------
# importance_gate — smoke under unified SEM
# -----------------------------------------------------------------------------


def test_importance_gate_flags_informative_under_mda() -> None:
    """Gate surfaces the informative feature ``a`` under MDA at
    ``t_stat=2.0`` on a fixture where ``a`` drives ``y``. Smoke test
    that the fold-mean SEM reformulation doesn't break signal detection.

    Historically this test also documented an SFI + ``neg_log_loss`` +
    gate foot-gun (raw-score SFI returns negative means; gate predicate
    never passes). S7 closed that bug by making SFI's default ``mean``
    column baseline-adjusted (``mean_raw − DummyClassifier(prior)
    score``). The companion positive-path pin lives in
    ``test_importance_sfi_baseline.py::
    test_pin04_gate_passes_informative_under_sfi_neg_log_loss``.
    """
    X, y = _fixture(500, 0)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = DecisionTreeClassifier(random_state=0)

    mda = feature_importance_mda(model, X, y, cv=cv, n_repeats=2, random_state=0)
    passing_mda, _ = importance_gate({"mda": mda}, min_features=1, t_stat=2.0)
    assert "a" in passing_mda, f"MDA gate missed informative feature 'a'; passing={passing_mda}"


def test_importance_gate_combines_methods_by_union() -> None:
    """Gate returns the union of features passing under any single
    method's threshold. Pins the OR semantics vs (a hypothetical)
    majority-vote.
    """
    # Construct two DataFrames where different features pass
    df_a = pd.DataFrame({"mean": [1.0, 0.1], "std": [0.1, 0.1]}, index=["a", "b"])
    df_b = pd.DataFrame({"mean": [0.1, 1.0], "std": [0.1, 0.1]}, index=["a", "b"])
    passing, ok = importance_gate({"m1": df_a, "m2": df_b}, min_features=1, t_stat=2.0)
    assert passing == ["a", "b"], f"union semantics broken; got {passing}"
    assert ok
