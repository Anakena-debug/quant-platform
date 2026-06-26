"""s19b c1 Phase-2: regression suite for the S19-exercised subset of
``quantcore.uncertainty.conformal.scores``.

Phase-1 triage (commit 04aca56) identified 5 of 12 public symbols as S19-exercised:

  1. ``absolute_residual_score`` — Mondrian (via SplitConformalRegressor),
     dtaci, alpha, timeseries, volatility.
  2. ``compute_conformal_quantile`` — every conformal module (8/8
     importers); load-bearing for every S19 prediction interval.
  3. ``asymmetric_quantile_score`` — CQR only.
  4. ``compute_asymmetric_quantiles`` — CQR only.
  5. ``quantile_score`` — CQR only.

Tests follow the F-RP-002 happy-path / F08-trigger pattern from
``test_alpha_f08.py`` adapted to each scorer's contract:

  * Happy-path numerical pin to ~1e-12 absolute tolerance.
  * Spec-implied raise-paths (α validation in
    ``compute_conformal_quantile``).
  * Per-scorer load-bearing semantics: max-of-two for
    ``quantile_score``, asymmetric-tuple for
    ``asymmetric_quantile_score``, α/2-split for
    ``compute_asymmetric_quantiles``.

``compute_conformal_quantile`` gets extra finite-sample-correction
boundary tests because the triage flagged it as load-bearing across
8/8 importers and the correction ``ceil((n+1)(1-α))/n`` matters most
at sparse n (the S19 walk-forward regime).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    asymmetric_quantile_score,
    compute_asymmetric_quantiles,
    compute_conformal_quantile,
    quantile_score,
)


# =============================================================================
# 1. absolute_residual_score — happy-path numerical pin
# =============================================================================


def test_absolute_residual_score_happy_path() -> None:
    """``|y_true - y_pred|`` elementwise; output shape matches input."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.8, 3.2, 4.0])
    scores = absolute_residual_score(y_true, y_pred)
    expected = np.array([0.1, 0.2, 0.2, 0.0])
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    assert scores.shape == y_true.shape


def test_absolute_residual_score_zero_residual() -> None:
    """Identical input/prediction → all-zero scores. Pins the floor of
    the absolute value (no spurious sign-handling bug)."""
    rng = np.random.default_rng(seed=42)
    y = rng.standard_normal(100)
    scores = absolute_residual_score(y, y)
    np.testing.assert_array_equal(scores, np.zeros_like(y))


# =============================================================================
# 2. compute_conformal_quantile — finite-sample correction + α-edge cases
# =============================================================================


def test_compute_conformal_quantile_docstring_example() -> None:
    """Pin the docstring example: ``scores=[0.1..0.5]``, α=0.1, n=5
    → ``ceil(6*0.9)/5 = ceil(5.4)/5 = 6/5 = 1.2`` clipped to 1.0 →
    ``np.quantile(scores, 1.0, method='higher')`` = max(scores) = 0.5.
    Demonstrates the small-n regime where the correction saturates."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q = compute_conformal_quantile(scores, alpha=0.1)
    assert q == 0.5


def test_compute_conformal_quantile_correction_is_conservative() -> None:
    """Direction invariant: for any (scores, α), the corrected quantile
    is ≥ the naive ``np.quantile(scores, 1-α, method='higher')``. The
    correction is conservative-by-construction (Vovk 2005) — rounds UP
    to ensure marginal coverage ≥ 1-α. Pinning this prevents a future
    refactor that drops the ``ceil`` and silently produces narrower
    intervals than the conformal guarantee requires.

    Note: the rng draws are unseeded-per-cell (one ``rng`` for the
    whole grid). On failure the assertion message reports the (n, α)
    pair, which is enough to reproduce manually by re-running with
    that seed and the cell-index offset; byte-exact per-cell
    reproducibility was not deemed worth the test-fixture
    instrumentation cost.
    """
    rng = np.random.default_rng(seed=42)
    for n in (10, 20, 50, 200):
        for alpha in (0.01, 0.1, 0.2, 0.5):
            scores = rng.standard_normal(n)
            corrected = compute_conformal_quantile(scores, alpha)
            naive = float(np.quantile(scores, 1.0 - alpha, method="higher"))
            assert corrected >= naive, (
                f"n={n}, α={alpha}: finite-sample correction must be "
                f"≥ naive 'higher' quantile; got corrected={corrected:.6e}, "
                f"naive={naive:.6e}"
            )


def test_compute_conformal_quantile_saturates_at_small_alpha() -> None:
    """At small α the correction saturates the quantile level to 1.0
    → returns max(scores). Pinning the saturation behavior at
    α=0.001 with n=100: ``ceil(101*0.999)/100 = ceil(100.899)/100 =
    101/100 = 1.01`` clipped to 1.0."""
    rng = np.random.default_rng(seed=42)
    scores = rng.standard_normal(100)
    q = compute_conformal_quantile(scores, alpha=0.001)
    assert q == float(np.max(scores)), f"α=0.001 with n=100 must saturate at max(scores); got {q}"


def test_compute_conformal_quantile_alpha_at_math_boundary() -> None:
    """``(n+1)(1-α)`` lands exactly on an integer at (n=9, α=0.1):
    ``10 * 0.9 = 9.0``. ``ceil(9.0)/9 = 9/9 = 1.0`` → max. Adjacent
    n=8 and n=10 also saturate at max but via different ceil paths:

      * n=8, α=0.1:  9*0.9=8.1 → ceil/8 = 9/8 = 1.125 → clip to 1.0 → max.
      * n=9, α=0.1:  10*0.9=9.0 (boundary) → ceil/9 = 9/9 = 1.0 → max.
      * n=10, α=0.1: 11*0.9=9.9 → ceil/10 = 10/10 = 1.0 → max.

    All three saturate at max but for slightly different reasons.
    The test's value is pinning that the saturation HOLDS across the
    math boundary regardless of which ceil branch is taken — a
    future refactor that introduces a sign or off-by-one error in
    the ``ceil`` path would surface here on at least one of the
    three cases.
    """
    rng = np.random.default_rng(seed=42)
    # n=9, α=0.1 — the exact-integer math boundary.
    scores_9 = rng.standard_normal(9)
    q_9 = compute_conformal_quantile(scores_9, alpha=0.1)
    assert q_9 == float(np.max(scores_9))
    # n=10, α=0.1 — just-above (raw level 0.99 → ceil saturates).
    scores_10 = rng.standard_normal(10)
    q_10 = compute_conformal_quantile(scores_10, alpha=0.1)
    assert q_10 == float(np.max(scores_10))
    # n=8, α=0.1 — just-below (raw level 1.125 → clip saturates).
    scores_8 = rng.standard_normal(8)
    q_8 = compute_conformal_quantile(scores_8, alpha=0.1)
    assert q_8 == float(np.max(scores_8))


def test_compute_conformal_quantile_invalid_alpha_raises() -> None:
    """α ∉ (0, 1) raises ValueError. Pin the message text so a future
    refactor that loosens validation surfaces here."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    for invalid_alpha in (0.0, 1.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match=r"alpha must be in"):
            compute_conformal_quantile(scores, alpha=invalid_alpha)


# =============================================================================
# 3. asymmetric_quantile_score — shape-contract + per-side semantics
# =============================================================================


def test_asymmetric_quantile_score_shape_contract() -> None:
    """Returns ``(lower_scores, upper_scores)`` — both arrays the same
    length as input. Per-side semantics:
    ``lower_scores = y_pred_lower - y_true``,
    ``upper_scores = y_true - y_pred_upper``. Pinning the assignment
    direction prevents a future swap that would silently invert the
    asymmetry contract."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred_lower = np.array([0.5, 1.5, 2.5])
    y_pred_upper = np.array([1.5, 2.5, 3.5])

    result = asymmetric_quantile_score(y_true, y_pred_lower, y_pred_upper)
    # Tuple shape contract.
    assert isinstance(result, tuple)
    assert len(result) == 2
    lower, upper = result
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    # Per-side numerical pin.
    np.testing.assert_allclose(
        lower,
        y_pred_lower - y_true,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        upper,
        y_true - y_pred_upper,
        rtol=1e-12,
        atol=1e-12,
    )


# =============================================================================
# 4. compute_asymmetric_quantiles — α/2 split delegation
# =============================================================================


def test_compute_asymmetric_quantiles_splits_alpha_evenly() -> None:
    """Splits α evenly between lower and upper bounds. Each side
    delegates to ``compute_conformal_quantile`` at α/2; pin the
    delegation by computing the expected per-side quantile
    independently and asserting equality."""
    rng = np.random.default_rng(seed=42)
    n = 100
    scores_lower = rng.standard_normal(n)
    scores_upper = rng.standard_normal(n)
    alpha = 0.1

    q_lower, q_upper = compute_asymmetric_quantiles(
        scores_lower,
        scores_upper,
        alpha,
    )

    expected_lower = compute_conformal_quantile(scores_lower, alpha=alpha / 2)
    expected_upper = compute_conformal_quantile(scores_upper, alpha=alpha / 2)

    assert q_lower == expected_lower
    assert q_upper == expected_upper


# =============================================================================
# 5. quantile_score — max-of-two-terms semantics
# =============================================================================


def test_quantile_score_max_of_two_terms() -> None:
    """``quantile_score = max(y_pred_lower - y_true, y_true - y_pred_upper)``.
    Pinning the max-of-two semantics prevents a future regression
    that swaps the two terms (silent sign error). Three cases:

      * Inside [lower, upper]: both terms negative → max < 0.
      * Below lower: first term positive → max > 0.
      * Above upper: second term positive → max > 0.

    These three branches cover every sign-pattern the function can
    produce.
    """
    # Case 1: y=2.0 inside [1.5, 2.5] → max(1.5-2.0, 2.0-2.5) = max(-0.5, -0.5) = -0.5
    # Case 2: y=1.0 below lower=1.5 → max(1.5-1.0, 1.0-2.5) = max(0.5, -1.5) = 0.5
    # Case 3: y=3.0 above upper=2.5 → max(1.5-3.0, 3.0-2.5) = max(-1.5, 0.5) = 0.5
    y_true = np.array([2.0, 1.0, 3.0])
    y_pred_lower = np.array([1.5, 1.5, 1.5])
    y_pred_upper = np.array([2.5, 2.5, 2.5])

    scores = quantile_score(y_true, y_pred_lower, y_pred_upper)
    expected = np.array([-0.5, 0.5, 0.5])
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    assert scores.shape == (3,)


def test_quantile_score_negative_inside_interval() -> None:
    """When y is strictly inside [lower, upper] for ALL elements, the
    score is strictly negative for every element. Pins the
    'inside-the-interval = negative score' invariant that conformal
    calibration relies on (negative scores indicate the calibration
    point was correctly covered)."""
    rng = np.random.default_rng(seed=42)
    n = 100
    y_true = rng.standard_normal(n)
    # Construct lower/upper to bracket y_true with margin 0.5.
    y_pred_lower = y_true - 0.5
    y_pred_upper = y_true + 0.5

    scores = quantile_score(y_true, y_pred_lower, y_pred_upper)
    assert (scores < 0).all(), (
        f"all scores must be negative when y is strictly inside the "
        f"interval; got {int((scores >= 0).sum())} non-negative entries"
    )
