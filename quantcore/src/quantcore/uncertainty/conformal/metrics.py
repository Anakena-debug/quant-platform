"""
Evaluation metrics for conformal prediction.

This module provides comprehensive metrics for assessing conformal predictors:
- Coverage metrics: marginal, conditional, worst-group
- Efficiency metrics: interval width, prediction set size
- Calibration metrics: coverage deviation, reliability diagrams
- Comparative metrics: for benchmarking different methods
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.uncertainty.conformal.base import PredictionInterval, PredictionSet


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """
    Comprehensive metrics for regression conformal predictors.

    Attributes:
        coverage: Empirical marginal coverage
        mean_width: Average interval width
        median_width: Median interval width
        width_std: Standard deviation of interval widths
        coverage_gap: Difference from target coverage (coverage - (1-alpha))
        cwc: Coverage Width-based Criterion (penalizes wide intervals)
        pinball_loss: Pinball loss for lower and upper bounds
    """

    coverage: float
    mean_width: float
    median_width: float
    width_std: float
    coverage_gap: float
    cwc: float
    pinball_lower: float
    pinball_upper: float
    n_samples: int
    alpha: float

    @property
    def is_valid(self) -> bool:
        """Whether coverage meets target (within statistical tolerance)."""
        # Use binomial CI for coverage
        se = np.sqrt(self.coverage * (1 - self.coverage) / self.n_samples)
        target = 1 - self.alpha
        return self.coverage >= target - 2 * se


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """
    Comprehensive metrics for classification conformal predictors.

    Attributes:
        coverage: Empirical marginal coverage
        mean_size: Average prediction set size
        median_size: Median prediction set size
        empty_rate: Fraction of empty prediction sets
        singleton_rate: Fraction of single-class sets
        coverage_gap: Difference from target coverage
    """

    coverage: float
    mean_size: float
    median_size: float
    size_std: float
    empty_rate: float
    singleton_rate: float
    coverage_gap: float
    n_samples: int
    alpha: float

    @property
    def is_valid(self) -> bool:
        """Whether coverage meets target."""
        se = np.sqrt(self.coverage * (1 - self.coverage) / self.n_samples)
        target = 1 - self.alpha
        return self.coverage >= target - 2 * se


def compute_regression_metrics(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
    alpha: float | None = None,
) -> RegressionMetrics:
    """
    Compute comprehensive metrics for regression intervals.

    Args:
        intervals: Prediction intervals
        y_true: True target values
        alpha: Target miscoverage rate (defaults to intervals.alpha)

    Returns:
        RegressionMetrics instance
    """
    alpha = alpha if alpha is not None else intervals.alpha
    n = len(y_true)

    # Coverage
    covered = intervals.contains(y_true)
    coverage = float(np.mean(covered))

    # Width statistics
    widths = intervals.width
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    width_std = float(np.std(widths))

    # Coverage gap
    target_coverage = 1 - alpha
    coverage_gap = coverage - target_coverage

    # Coverage Width-based Criterion (CWC)
    # Penalizes both under-coverage and wide intervals
    eta = 50  # Penalty strength for under-coverage
    cwc = mean_width * (1 + np.exp(-eta * coverage_gap) if coverage_gap < 0 else 1)

    # Pinball losses for lower and upper quantiles
    alpha_lo = alpha / 2
    alpha_hi = 1 - alpha / 2

    residuals_lower = y_true - intervals.lower
    pinball_lower = float(
        np.mean(
            np.where(
                residuals_lower >= 0, alpha_lo * residuals_lower, (alpha_lo - 1) * residuals_lower
            )
        )
    )

    residuals_upper = y_true - intervals.upper
    pinball_upper = float(
        np.mean(
            np.where(
                residuals_upper >= 0, alpha_hi * residuals_upper, (alpha_hi - 1) * residuals_upper
            )
        )
    )

    return RegressionMetrics(
        coverage=coverage,
        mean_width=mean_width,
        median_width=median_width,
        width_std=width_std,
        coverage_gap=coverage_gap,
        cwc=cwc,
        pinball_lower=pinball_lower,
        pinball_upper=pinball_upper,
        n_samples=n,
        alpha=alpha,
    )


def compute_classification_metrics(
    pred_sets: PredictionSet,
    y_true: NDArray[np.int_],
    alpha: float | None = None,
) -> ClassificationMetrics:
    """
    Compute comprehensive metrics for classification prediction sets.

    Args:
        pred_sets: Prediction sets
        y_true: True class labels
        alpha: Target miscoverage rate (defaults to pred_sets.alpha)

    Returns:
        ClassificationMetrics instance
    """
    alpha = alpha if alpha is not None else pred_sets.alpha
    n = len(y_true)

    # Coverage
    covered = pred_sets.contains(y_true)
    coverage = float(np.mean(covered))

    # Size statistics
    sizes = pred_sets.sizes
    mean_size = float(np.mean(sizes))
    median_size = float(np.median(sizes))
    size_std = float(np.std(sizes))

    # Empty and singleton rates
    empty_rate = float(np.mean(sizes == 0))
    singleton_rate = float(np.mean(sizes == 1))

    # Coverage gap
    coverage_gap = coverage - (1 - alpha)

    return ClassificationMetrics(
        coverage=coverage,
        mean_size=mean_size,
        median_size=median_size,
        size_std=size_std,
        empty_rate=empty_rate,
        singleton_rate=singleton_rate,
        coverage_gap=coverage_gap,
        n_samples=n,
        alpha=alpha,
    )


def conditional_coverage(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_bins: int = 10,
    bin_by: str = "prediction",
) -> dict[str, Any]:
    """
    Compute conditional coverage across different regions.

    Args:
        intervals: Prediction intervals
        y_true: True values
        X: Features (used for binning if bin_by='feature')
        n_bins: Number of bins
        bin_by: 'prediction' (bin by predicted value) or 'feature' (bin by feature)

    Returns:
        Dictionary with conditional coverage statistics
    """
    covered = intervals.contains(y_true)

    if bin_by == "prediction" and intervals.point is not None:
        bin_values = intervals.point
    elif bin_by == "feature":
        # Use first principal component if X is multivariate
        if X.ndim == 1 or X.shape[1] == 1:
            bin_values = X.ravel()
        else:
            # Simple: use first feature
            bin_values = X[:, 0]
    else:
        bin_values = intervals.point if intervals.point is not None else y_true

    # Create bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(bin_values, percentiles)

    bin_coverages = []
    bin_counts = []
    bin_centers = []

    for i in range(n_bins):
        mask = (bin_values >= bin_edges[i]) & (bin_values < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (bin_values >= bin_edges[i]) & (bin_values <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_coverages.append(float(np.mean(covered[mask])))
            bin_counts.append(int(np.sum(mask)))
            bin_centers.append(float(np.mean(bin_values[mask])))

    return {
        "bin_coverages": np.array(bin_coverages),
        "bin_counts": np.array(bin_counts),
        "bin_centers": np.array(bin_centers),
        "bin_edges": bin_edges,
        "worst_coverage": min(bin_coverages) if bin_coverages else 0.0,
        "coverage_std": float(np.std(bin_coverages)) if bin_coverages else 0.0,
    }


def worst_slab_coverage(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    slab_fraction: float = 0.1,
) -> float:
    """
    Compute worst-case coverage over slabs of feature space.

    For each feature, finds the slab (contiguous fraction) with worst coverage.
    This tests conditional coverage more rigorously.

    Args:
        intervals: Prediction intervals
        y_true: True values
        X: Features
        slab_fraction: Size of slab as fraction of data

    Returns:
        Worst slab coverage across all features
    """
    covered = intervals.contains(y_true)
    n = len(y_true)
    slab_size = int(n * slab_fraction)

    if slab_size < 10:
        raise ValueError("slab_size too small for reliable coverage estimation")

    worst_coverage = 1.0

    # Check slabs for each feature
    n_features = X.shape[1] if X.ndim > 1 else 1
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

    for j in range(n_features):
        sorted_idx = np.argsort(X_2d[:, j])

        # Slide window across sorted data
        for start in range(n - slab_size + 1):
            slab_idx = sorted_idx[start : start + slab_size]
            slab_coverage = float(np.mean(covered[slab_idx]))
            worst_coverage = min(worst_coverage, slab_coverage)

    return worst_coverage


def coverage_width_tradeoff(
    intervals_list: list[PredictionInterval],
    y_true: NDArray[np.floating[Any]],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare coverage-width tradeoff across multiple methods.

    Args:
        intervals_list: List of prediction intervals from different methods
        y_true: True values
        labels: Method labels

    Returns:
        Comparison dictionary
    """
    if labels is None:
        labels = [f"Method_{i}" for i in range(len(intervals_list))]

    results = {
        "labels": labels,
        "coverages": [],
        "mean_widths": [],
        "median_widths": [],
        "valid": [],
    }

    for intervals in intervals_list:
        metrics = compute_regression_metrics(intervals, y_true)
        results["coverages"].append(metrics.coverage)
        results["mean_widths"].append(metrics.mean_width)
        results["median_widths"].append(metrics.median_width)
        results["valid"].append(metrics.is_valid)

    return results


def stratified_coverage(
    pred_sets: PredictionSet,
    y_true: NDArray[np.int_],
    n_classes: int | None = None,
) -> dict[str, Any]:
    """
    Compute coverage stratified by class.

    Important for detecting class-conditional coverage gaps.

    Args:
        pred_sets: Prediction sets
        y_true: True class labels
        n_classes: Number of classes (inferred if not provided)

    Returns:
        Per-class coverage statistics
    """
    if n_classes is None:
        n_classes = int(np.max(y_true)) + 1

    covered = pred_sets.contains(y_true)

    class_coverages = []
    class_counts = []

    for c in range(n_classes):
        mask = y_true == c
        if np.sum(mask) > 0:
            class_coverages.append(float(np.mean(covered[mask])))
            class_counts.append(int(np.sum(mask)))
        else:
            class_coverages.append(np.nan)
            class_counts.append(0)

    return {
        "class_coverages": np.array(class_coverages),
        "class_counts": np.array(class_counts),
        "worst_class_coverage": float(np.nanmin(class_coverages)),
        "coverage_std": float(np.nanstd(class_coverages)),
    }


def calibration_error(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
    n_alpha_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration error across different coverage levels.

    A well-calibrated predictor should achieve coverage close to (1-α)
    for any α. This tests calibration across multiple α values.

    Args:
        intervals: Prediction intervals (computed at some fixed α)
        y_true: True values
        n_alpha_bins: Number of α levels to test

    Returns:
        Calibration statistics
    """
    # For each sample, compute at what "α" it would be covered
    # This is based on where y_true falls relative to interval
    if intervals.point is None:
        raise ValueError("Intervals must have point predictions for calibration analysis")

    # Compute normalized residuals
    widths = intervals.width
    residuals = np.abs(y_true - intervals.point)
    normalized = residuals / (widths / 2)  # 1.0 means exactly at boundary

    # For different α levels, check what fraction is covered
    alpha_levels = np.linspace(0.05, 0.95, n_alpha_bins)
    empirical_coverages = []

    for alpha in alpha_levels:
        # At this alpha, interval would be this fraction of original
        threshold = 1 - alpha
        # Fraction of points within this threshold
        empirical = float(np.mean(normalized <= threshold))
        empirical_coverages.append(empirical)

    expected_coverages = 1 - alpha_levels
    calibration_errors = np.array(empirical_coverages) - expected_coverages

    return {
        "alpha_levels": alpha_levels,
        "expected_coverages": expected_coverages,
        "empirical_coverages": np.array(empirical_coverages),
        "calibration_errors": calibration_errors,
        "mean_calibration_error": float(np.mean(np.abs(calibration_errors))),
        "max_calibration_error": float(np.max(np.abs(calibration_errors))),
    }


def winkler_score(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
) -> float:
    """
    Compute Winkler score for interval forecasts.

    A proper scoring rule that penalizes both:
    - Wide intervals (less informative)
    - Intervals that miss the true value

    Lower is better.

    Args:
        intervals: Prediction intervals
        y_true: True values

    Returns:
        Average Winkler score
    """
    alpha = intervals.alpha
    lower = intervals.lower
    upper = intervals.upper
    width = intervals.width

    scores = np.zeros(len(y_true))

    for i in range(len(y_true)):
        if y_true[i] < lower[i]:
            scores[i] = width[i] + (2 / alpha) * (lower[i] - y_true[i])
        elif y_true[i] > upper[i]:
            scores[i] = width[i] + (2 / alpha) * (y_true[i] - upper[i])
        else:
            scores[i] = width[i]

    return float(np.mean(scores))


def interval_score(
    intervals: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
) -> float:
    """
    Compute interval score (equivalent to Winkler score).

    A strictly proper scoring rule for prediction intervals.
    Decomposes into: sharpness + undercoverage penalty.

    Args:
        intervals: Prediction intervals
        y_true: True values

    Returns:
        Average interval score
    """
    # This is equivalent to Winkler score
    return winkler_score(intervals, y_true)


def efficiency_comparison(
    intervals_base: PredictionInterval,
    intervals_new: PredictionInterval,
    y_true: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """
    Compare efficiency of two conformal methods.

    Args:
        intervals_base: Baseline intervals
        intervals_new: New method intervals
        y_true: True values

    Returns:
        Comparison statistics
    """
    metrics_base = compute_regression_metrics(intervals_base, y_true)
    metrics_new = compute_regression_metrics(intervals_new, y_true)

    return {
        "width_reduction": 1 - metrics_new.mean_width / metrics_base.mean_width,
        "coverage_base": metrics_base.coverage,
        "coverage_new": metrics_new.coverage,
        "width_base": metrics_base.mean_width,
        "width_new": metrics_new.mean_width,
        "both_valid": metrics_base.is_valid and metrics_new.is_valid,
        "winkler_base": winkler_score(intervals_base, y_true),
        "winkler_new": winkler_score(intervals_new, y_true),
    }
