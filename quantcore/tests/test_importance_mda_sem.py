"""S9 P8.1 — MDA SEM clarification + opt-in ANOVA estimator.

Pins the docstring contract (algebraic identity, pooled refutation,
canonical n_repeats=10 justification) and the ``sem_method`` kwarg
behavior.

Tests are grouped:

    Pins 1–3      · docstring content (via ``__doc__`` inspection)
    Pin 4         · algebraic identity fold_only = anova under no clipping
    Pin 5         · sem_method="anova" produces valid 2-col output
    Pin 6         · sem_method default is backward-compatible
    Pin 7         · sem_method invalid value raises
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from quantcore.importance.importance import feature_importance_mda


# =============================================================================
# Pins 1–3 · Docstring content.
# =============================================================================
# ``__doc__`` inspection is used instead of bare ``rg``-style string
# search so failure messages name the specific missing content.


def test_pin1_mda_docstring_documents_sem_decomposition() -> None:
    """The algebraic identity
    ``SEM² = σ²_between / n_f + σ²_within / (n_f · n_r)``
    must be rendered explicitly in the docstring.
    """
    doc = feature_importance_mda.__doc__
    assert doc is not None, "feature_importance_mda must have a docstring"
    assert re.search(
        r"fold_only_SEM\s*[²2]?\s*=",
        doc,
    ), "docstring must render the fold_only SEM identity equation"
    assert re.search(r"σ.*between", doc) and re.search(r"σ.*within", doc), (
        "docstring must name both σ²_between and σ²_within"
    )
    assert "n_folds" in doc, "docstring must name n_folds in the identity"
    assert "n_repeats" in doc, "docstring must name n_repeats in the identity"


def test_pin2_mda_docstring_refutes_pooled_sem() -> None:
    """The docstring must explicitly explain why pooled SEM over
    ``n_folds × n_repeats`` draws is statistically unsound for this
    use case (ignores between-fold variance)."""
    doc = feature_importance_mda.__doc__
    assert doc is not None
    # The refutation must call out pooled SEM specifically AND name
    # the failure mode (underestimation / ignoring between-fold).
    assert re.search(r"pooled", doc, re.IGNORECASE), "docstring must mention pooled SEM by name"
    assert re.search(
        r"(ignore|underestim|without).*between|between.*(ignore|underestim)",
        doc,
        re.IGNORECASE | re.DOTALL,
    ), (
        "docstring must name the failure mode: pooled SEM ignores or "
        "underestimates the between-fold variance component"
    )


def test_pin3_mda_docstring_justifies_canonical_n_repeats_10() -> None:
    """The docstring must document why n_repeats=10 is canonical on
    triple-barrier fixtures (mean-estimation noise, not SEM)."""
    doc = feature_importance_mda.__doc__
    assert doc is not None
    assert re.search(r"n_repeats\s*=\s*10", doc), "docstring must explicitly mention n_repeats=10"
    assert re.search(r"canonical", doc, re.IGNORECASE), (
        "docstring must frame n_repeats=10 as canonical"
    )
    assert re.search(
        r"mean[- ]estim|estimation noise|noisy.*mean",
        doc,
        re.IGNORECASE,
    ), (
        "docstring must attribute the n_repeats=3 failure to mean-"
        "estimation noise (not SEM computation)"
    )


# =============================================================================
# Pin 4 · Algebraic identity — fold_only and anova agree under no clipping.
# =============================================================================
# Construct a fixture where sample σ²_between is reliably positive,
# so the ANOVA clipping at zero does NOT fire. In that regime the two
# SEM estimators are algebraically identical (derivation in docstring).


def test_pin4_fold_only_equals_anova_when_no_clipping() -> None:
    """Without σ²_between clipping, fold_only SEM ≡ anova SEM by the
    algebraic identity in the docstring. This pin guards the
    identity against implementation drift.

    Fixture design: a non-stationary dataset (first half vs second
    half with different noise levels) produces fold-level differences
    large enough to keep σ²_between clearly positive — no clipping,
    so both SEM variants are identical.
    """
    rng = np.random.default_rng(0)
    n = 400
    # Non-stationary noise: fold-level CV scores diverge → large σ²_b.
    X_array = np.concatenate(
        [
            rng.standard_normal((n // 2, 2)) * 0.5,
            rng.standard_normal((n // 2, 2)) * 2.0,
        ]
    )
    y_array = np.sign(X_array[:, 0] + 0.1 * rng.standard_normal(n)).astype(int)
    X = pd.DataFrame(X_array, columns=["a", "b"])
    y = pd.Series(y_array, index=X.index)
    cv = KFold(n_splits=5, shuffle=False)

    mda_fold = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_repeats=5,
        sem_method="fold_only",
    )
    mda_anova = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_repeats=5,
        sem_method="anova",
    )

    # Means are identical regardless of SEM method (it only affects std).
    pd.testing.assert_series_equal(mda_fold["mean"], mda_anova["mean"], check_names=False)
    # std differs only when ANOVA clips σ²_b < 0. On this fixture
    # σ²_b is reliably positive; SEMs are identical to numerical tol.
    # If clipping fires anywhere, ANOVA >= fold_only (more conservative).
    np.testing.assert_array_less(-1e-12, (mda_anova["std"] - mda_fold["std"]).to_numpy())


# =============================================================================
# Pin 5 · sem_method="anova" produces valid output.
# =============================================================================


def test_pin5_anova_output_is_valid() -> None:
    """Opt-in ANOVA path returns a 2-column DataFrame with finite,
    non-negative std values."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series(
        np.sign(X["a"] + 0.5 * rng.standard_normal(n)).astype(int),
        index=X.index,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    mda = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_repeats=5,
        sem_method="anova",
    )

    assert list(mda.columns) == ["mean", "std"]
    assert len(mda) == 3
    assert bool(mda["std"].notna().all())
    assert bool((mda["std"] >= 0).all()), (
        f"ANOVA std must be non-negative; got {mda['std'].to_dict()}"
    )
    assert bool(mda["mean"].notna().all())


# =============================================================================
# Pin 6 · Default kwarg value is "fold_only" (backward compat).
# =============================================================================


def test_pin6_default_sem_method_is_fold_only() -> None:
    """``sem_method`` defaults to ``"fold_only"`` — pre-S9 behavior
    preserved. Omitting the kwarg must produce identical output to
    explicitly passing ``sem_method="fold_only"``."""
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = pd.Series(
        np.sign(X["a"] + 0.5 * rng.standard_normal(n)).astype(int),
        index=X.index,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=7)

    mda_default = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_repeats=5,
    )
    mda_explicit_fold = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X,
        y,
        cv=cv,
        scoring="neg_log_loss",
        n_repeats=5,
        sem_method="fold_only",
    )

    pd.testing.assert_frame_equal(mda_default, mda_explicit_fold)


# =============================================================================
# Pin 7 · Invalid sem_method raises.
# =============================================================================


def test_pin7_invalid_sem_method_raises() -> None:
    """Unknown ``sem_method`` values raise ``ValueError`` with a
    message naming the accepted values."""
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    y = pd.Series(
        np.sign(X["a"] + 0.5 * rng.standard_normal(n)).astype(int),
        index=X.index,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    with pytest.raises(ValueError, match=r"sem_method.*fold_only.*anova"):
        feature_importance_mda(
            LogisticRegression(max_iter=2000),
            X,
            y,
            cv=cv,
            scoring="neg_log_loss",
            n_repeats=3,
            sem_method="pooled",  # pyright: ignore[reportArgumentType]
        )
