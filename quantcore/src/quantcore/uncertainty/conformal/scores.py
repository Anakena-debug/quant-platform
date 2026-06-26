"""
Nonconformity score functions for conformal prediction.

This module provides various score functions that measure how "unusual"
a prediction is compared to calibration data. The choice of score function
significantly impacts the shape and efficiency of prediction intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class ScoreType(Enum):
    """Enumeration of available nonconformity score types."""

    ABSOLUTE_RESIDUAL = auto()
    SIGNED_RESIDUAL = auto()
    NORMALIZED_RESIDUAL = auto()
    QUANTILE_SCORE = auto()
    GAMMA_SCORE = auto()
    CQR_SCORE = auto()


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """
    Container for nonconformity score computation results.

    Attributes:
        scores: Computed nonconformity scores
        score_type: Type of score function used
        metadata: Additional information from score computation
    """

    scores: NDArray[np.floating[Any]]
    score_type: ScoreType
    metadata: dict[str, Any] | None = None


def absolute_residual_score(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Absolute residual nonconformity score.

    The simplest and most commonly used score: |y - ŷ|

    This produces symmetric prediction intervals of constant width.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred: Point predictions, shape (n_samples,)

    Returns:
        Absolute residuals, shape (n_samples,)

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 1.8, 3.2])
        >>> scores = absolute_residual_score(y_true, y_pred)
        >>> np.allclose(scores, [0.1, 0.2, 0.2])
        True
    """
    return np.abs(y_true - y_pred)


def signed_residual_score(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Signed residual nonconformity score.

    Score: y - ŷ (can be positive or negative)

    Useful for detecting asymmetric errors or when direction matters.
    Not typically used for standard conformal prediction intervals.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred: Point predictions, shape (n_samples,)

    Returns:
        Signed residuals, shape (n_samples,)
    """
    return y_true - y_pred


def normalized_residual_score(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    y_pred_std: NDArray[np.floating[Any]] | None = None,
    epsilon: float = 1e-8,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Normalized residual nonconformity score.

    Score: |y - ŷ| / σ̂(x)

    Produces adaptive prediction intervals that widen where the model
    is more uncertain. Requires a model that outputs uncertainty estimates.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred: Point predictions, shape (n_samples,)
        y_pred_std: Predicted standard deviations, shape (n_samples,)
        epsilon: Small constant for numerical stability

    Returns:
        Normalized residuals, shape (n_samples,)

    Raises:
        ValueError: If y_pred_std is not provided
    """
    if y_pred_std is None:
        raise ValueError(
            "normalized_residual_score requires y_pred_std. "
            "Use a model that outputs uncertainty estimates (e.g., GP, ensemble)."
        )

    if np.any(y_pred_std < 0):
        raise ValueError("y_pred_std must be non-negative")

    return np.abs(y_true - y_pred) / (y_pred_std + epsilon)


def quantile_score(
    y_true: NDArray[np.floating[Any]],
    y_pred_lower: NDArray[np.floating[Any]],
    y_pred_upper: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Quantile-based nonconformity score (for CQR).

    Score: max(q̂_lower - y, y - q̂_upper)

    Used in Conformalized Quantile Regression (CQR) to produce
    adaptive prediction intervals based on quantile estimates.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred_lower: Lower quantile predictions, shape (n_samples,)
        y_pred_upper: Upper quantile predictions, shape (n_samples,)

    Returns:
        Quantile nonconformity scores, shape (n_samples,)

    Note:
        Negative scores indicate the true value is inside [lower, upper].
        Positive scores indicate how far outside the interval.
    """
    score_lower = y_pred_lower - y_true  # Positive if y below lower bound
    score_upper = y_true - y_pred_upper  # Positive if y above upper bound
    return np.maximum(score_lower, score_upper)


def asymmetric_quantile_score(
    y_true: NDArray[np.floating[Any]],
    y_pred_lower: NDArray[np.floating[Any]],
    y_pred_upper: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Asymmetric quantile scores for separate lower/upper calibration.

    Returns separate scores for lower and upper bounds, enabling
    asymmetric interval construction.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred_lower: Lower quantile predictions, shape (n_samples,)
        y_pred_upper: Upper quantile predictions, shape (n_samples,)

    Returns:
        Tuple of (lower_scores, upper_scores)
    """
    score_lower = y_pred_lower - y_true
    score_upper = y_true - y_pred_upper
    return score_lower, score_upper


def gamma_score(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    y_pred_std: NDArray[np.floating[Any]] | None = None,
    gamma: float = 0.1,
    epsilon: float = 1e-8,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Gamma-weighted normalized residual score.

    Score: |y - ŷ| / (σ̂(x)^γ + ε)

    Interpolates between absolute residual (γ=0) and fully normalized (γ=1).
    Can improve interval efficiency when uncertainty estimates are noisy.

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred: Point predictions, shape (n_samples,)
        y_pred_std: Predicted standard deviations, shape (n_samples,)
        gamma: Normalization exponent in [0, 1]
        epsilon: Small constant for numerical stability

    Returns:
        Gamma-normalized residuals, shape (n_samples,)
    """
    if y_pred_std is None:
        raise ValueError("gamma_score requires y_pred_std")

    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    return np.abs(y_true - y_pred) / (np.power(y_pred_std, gamma) + epsilon)


def classification_score_aps(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Adaptive Prediction Set (APS) score for classification.

    Score: Σ_{j: π̂_j > π̂_y} π̂_j + u * π̂_y

    where π̂ are the predicted probabilities sorted in decreasing order,
    and u ~ Uniform(0, 1) is used for randomization.

    Args:
        y_true: True class labels, shape (n_samples,)
        y_proba: Predicted class probabilities, shape (n_samples, n_classes)

    Returns:
        APS scores, shape (n_samples,)
    """
    n_samples = len(y_true)
    scores = np.zeros(n_samples)

    # Random tie-breaking
    u = np.random.uniform(0, 1, n_samples)

    for i in range(n_samples):
        probs = y_proba[i]
        true_class = y_true[i]
        true_prob = probs[true_class]

        # Sum probabilities of classes with higher probability than true class
        # Plus randomized fraction of true class probability
        scores[i] = np.sum(probs[probs > true_prob]) + u[i] * true_prob

    return scores


def classification_score_lac(
    y_true: NDArray[np.int_],
    y_proba: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    """
    Least Ambiguous set-valued Classifier (LAC) score.

    Score: 1 - π̂_y

    The simplest classification score, producing prediction sets
    that include all classes with probability above a threshold.

    Args:
        y_true: True class labels, shape (n_samples,)
        y_proba: Predicted class probabilities, shape (n_samples, n_classes)

    Returns:
        LAC scores, shape (n_samples,)
    """
    n_samples = len(y_true)
    scores = np.zeros(n_samples)

    for i in range(n_samples):
        scores[i] = 1 - y_proba[i, y_true[i]]

    return scores


def compute_conformal_quantile(
    scores: NDArray[np.floating[Any]],
    alpha: float,
    method: str = "higher",
) -> float:
    """
    Compute the conformal quantile from calibration scores.

    The quantile includes the finite-sample correction factor
    (1 + 1/n) to ensure valid coverage.

    Args:
        scores: Calibration nonconformity scores, shape (n_calibration,)
        alpha: Target miscoverage rate (e.g., 0.1 for 90% coverage)
        method: Numpy quantile interpolation method

    Returns:
        Conformal quantile threshold

    Example:
        >>> scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> q = compute_conformal_quantile(scores, alpha=0.1)
        >>> q >= np.quantile(scores, 0.9)  # Should be at least this
        True
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(scores)

    # Finite-sample corrected quantile level
    # This ensures P(Y in C(X)) >= 1 - alpha
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n

    # Clip to [0, 1] for edge cases
    quantile_level = np.clip(quantile_level, 0, 1)

    return float(np.quantile(scores, quantile_level, method=method))


def compute_asymmetric_quantiles(
    scores_lower: NDArray[np.floating[Any]],
    scores_upper: NDArray[np.floating[Any]],
    alpha: float,
) -> tuple[float, float]:
    """
    Compute separate quantiles for lower and upper bounds.

    Useful for asymmetric prediction intervals.

    Args:
        scores_lower: Lower bound scores from calibration
        scores_upper: Upper bound scores from calibration
        alpha: Total miscoverage rate

    Returns:
        Tuple of (quantile_lower, quantile_upper)
    """
    # Split alpha evenly between lower and upper
    alpha_half = alpha / 2

    q_lower = compute_conformal_quantile(scores_lower, alpha_half)
    q_upper = compute_conformal_quantile(scores_upper, alpha_half)

    return q_lower, q_upper


# Score function registry for easy lookup
SCORE_FUNCTIONS: dict[ScoreType, Callable[..., NDArray[np.floating[Any]]]] = {
    ScoreType.ABSOLUTE_RESIDUAL: absolute_residual_score,
    ScoreType.SIGNED_RESIDUAL: signed_residual_score,
    ScoreType.NORMALIZED_RESIDUAL: normalized_residual_score,
    ScoreType.QUANTILE_SCORE: quantile_score,
    ScoreType.GAMMA_SCORE: gamma_score,
}


def get_score_function(
    score_type: ScoreType | str,
) -> Callable[..., NDArray[np.floating[Any]]]:
    """
    Get a score function by type.

    Args:
        score_type: Score type enum or string name

    Returns:
        Score function

    Raises:
        ValueError: If score type is unknown
    """
    if isinstance(score_type, str):
        try:
            score_type = ScoreType[score_type.upper()]
        except KeyError:
            valid = [s.name for s in ScoreType]
            raise ValueError(f"Unknown score type: {score_type}. Valid options: {valid}") from None

    if score_type not in SCORE_FUNCTIONS:
        raise ValueError(f"Score function not implemented for {score_type}")

    return SCORE_FUNCTIONS[score_type]
