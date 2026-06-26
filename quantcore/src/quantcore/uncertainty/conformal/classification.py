"""
Classification conformal prediction methods.

This module implements conformal prediction for classification tasks,
producing prediction sets (subsets of classes) with valid coverage guarantees.

Key methods:
- LAC (Least Ambiguous set-valued Classifier): Simplest, includes all classes above threshold
- APS (Adaptive Prediction Sets): Better conditional coverage, randomized
- RAPS (Regularized APS): Penalizes large prediction sets

References:
    Romano, Sesia, Candès (2020) "Classification with Valid and Adaptive Coverage"
    Angelopoulos et al. (2021) "Uncertainty Sets for Image Classifiers using Conformal Prediction"
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from quantcore.uncertainty.conformal.base import (
    BaseClassificationConformal,
    CalibrationResult,
    PredictionSet,
    PredictionType,
)
from quantcore.uncertainty.conformal.scores import compute_conformal_quantile


_LEGACY_APS_WARN_MSG = "pre-P1.3 rank-1 randomization (F28)"
_LEGACY_RAPS_WARN_MSG = "pre-P1.3 searchsorted-overshoot inclusion (F33)"


def _aps_predict_legacy_rank1_randomization(
    sorted_probs: NDArray[np.floating[Any]],
    quantile: float,
    u: float,
    randomize: bool = True,
) -> int:
    """
    Module-private oracle. Bitwise-reproduces pre-P1.3 APSClassifier.predict
    inclusion logic on pre-sorted descending probabilities.

    Defect: used sorted_probs[0] (top-class probability) in the randomization
    threshold where per-rank sorted_probs[p] is canonical per RSC 2020 Alg 1.
    Biases inclusion test stricter for non-top ranks, producing systematic
    undercoverage. `+1` overshoot at first strict inclusion-test failure
    further distorts the boundary.

    Do NOT use in production. Retained for regression-test pinning only;
    removal anchored to the conformal-integration sprint (S6+).
    """
    warnings.warn(_LEGACY_APS_WARN_MSG, DeprecationWarning, stacklevel=2)
    cumsum = np.cumsum(sorted_probs)
    n_classes = len(sorted_probs)
    if randomize:
        threshold = quantile - u * sorted_probs[0] if n_classes > 0 else quantile
        include_idx = np.searchsorted(cumsum, threshold, side="left")
        include_idx = min(int(include_idx) + 1, n_classes)
    else:
        include_idx = np.searchsorted(cumsum, quantile, side="left") + 1
        include_idx = min(int(include_idx), n_classes)
    return int(include_idx)


def _raps_predict_legacy_overshoot(
    sorted_probs: NDArray[np.floating[Any]],
    penalties: NDArray[np.floating[Any]],
    quantile: float,
    u: float,
) -> int:
    """
    Module-private oracle. Bitwise-reproduces pre-P1.3 RAPSClassifier.predict
    inclusion logic on pre-sorted descending probabilities and pre-computed
    regularization penalties.

    Defect (F33): np.searchsorted(..., side='left') + 1 over-includes by 1
    class at strict-inequality boundaries (where adjusted_cumsum[p] > quantile
    strictly). Unlike F28, the randomization term `u * sorted_probs` is
    per-rank-correct (full array, not sorted_probs[0]); defect (i) from F28
    does NOT apply to RAPS — only defect (ii) overshoot. Net effect:
    over-coverage (prediction sets wastefully large; marginal coverage
    ≥ 1 - α still holds but not tightly).

    Do NOT use in production. Retained for regression-test pinning only;
    removal anchored to the conformal-integration sprint (S6+).
    """
    warnings.warn(_LEGACY_RAPS_WARN_MSG, DeprecationWarning, stacklevel=2)
    cumsum_reg = np.cumsum(sorted_probs) + penalties
    adjusted_cumsum = cumsum_reg - u * sorted_probs
    include_idx = int(np.searchsorted(adjusted_cumsum, quantile, side="left")) + 1
    n_classes = len(sorted_probs)
    include_idx = min(include_idx, n_classes)
    return int(include_idx)


class LACClassifier(BaseClassificationConformal[BaseEstimator]):
    """
    Least Ambiguous set-valued Classifier (LAC).

    The simplest classification conformal predictor. For each test point,
    includes all classes whose predicted probability exceeds a threshold
    calibrated to achieve the target coverage.

    Score: s(x, y) = 1 - f̂(x)_y (one minus probability of true class)

    Prediction set: C(x) = {y : f̂(x)_y ≥ 1 - q̂}

    Pros:
    - Simple and interpretable
    - Deterministic (no randomization)

    Cons:
    - Can have poor conditional coverage
    - May produce unnecessarily large sets for uncertain predictions
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, random_state)
        self._quantile: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.CLASSIFICATION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.int_],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "LACClassifier":
        """
        Fit classifier and calibrate.

        Args:
            X: Features
            y: Class labels (integers 0, 1, ..., K-1)
            calibration_fraction: Fraction for calibration
        """
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, X_cal = X[train_idx], X[cal_idx]
        y_train, y_cal = y[train_idx], y[cal_idx]

        self.model = clone(self.model)
        self.model.fit(X_train, y_train, **fit_params)

        self.calibrate(X_cal, y_cal)
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.int_],
    ) -> CalibrationResult:
        """Calibrate using held-out set."""
        y_proba = self.model.predict_proba(X_cal)

        # LAC score: 1 - probability of true class
        n_samples = len(y_cal)
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            scores[i] = 1 - y_proba[i, y_cal[i]]

        self._quantile = compute_conformal_quantile(scores, self.alpha)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=self._quantile,
            n_calibration=n_samples,
            alpha=self.alpha,
        )
        self._is_fitted = True

        return self._calibration_result

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionSet:
        """
        Generate prediction sets.

        Includes all classes with probability >= 1 - quantile.
        """
        self._check_is_fitted()
        assert self._quantile is not None

        y_proba = self.model.predict_proba(X)
        threshold = 1 - self._quantile

        n_samples = y_proba.shape[0]
        sets: list[set[int]] = []

        for i in range(n_samples):
            pred_set = set(np.where(y_proba[i] >= threshold)[0].tolist())
            # Ensure at least one class (the most probable)
            if len(pred_set) == 0:
                pred_set = {int(np.argmax(y_proba[i]))}
            sets.append(pred_set)

        return PredictionSet(
            sets=sets,
            probabilities=y_proba,
            alpha=self.alpha,
        )


class APSClassifier(BaseClassificationConformal[BaseEstimator]):
    """
    Adaptive Prediction Sets (APS) for classification.

    Uses a randomized score that produces smaller prediction sets
    with better conditional coverage than LAC.

    Score: s(x, y) = Σ_{j: π̂_j > π̂_y} π̂_j + U * π̂_y

    where U ~ Uniform(0, 1) provides randomization for tie-breaking.

    Prediction set: Include classes in decreasing probability order until
    cumulative probability exceeds 1 - quantile.

    Pros:
    - Better conditional coverage
    - Smaller average prediction set size

    Cons:
    - Randomized (different runs may give different sets)
    - More complex to implement
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, random_state)
        self._quantile: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.CLASSIFICATION

    def _compute_aps_score(
        self,
        y_proba: NDArray[np.floating[Any]],
        y_true: NDArray[np.int_],
    ) -> NDArray[np.floating[Any]]:
        """Compute APS nonconformity scores."""
        n_samples = len(y_true)
        scores = np.zeros(n_samples)

        # Random uniform for tie-breaking
        u = self._rng.uniform(0, 1, n_samples)

        for i in range(n_samples):
            probs = y_proba[i]
            true_class = y_true[i]
            true_prob = probs[true_class]

            # Sum probabilities of classes more probable than true class
            # Plus randomized fraction of true class probability
            scores[i] = np.sum(probs[probs > true_prob]) + u[i] * true_prob

        return scores

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.int_],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "APSClassifier":
        """Fit and calibrate."""
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        self.model = clone(self.model)
        self.model.fit(X[train_idx], y[train_idx], **fit_params)

        self.calibrate(X[cal_idx], y[cal_idx])
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.int_],
    ) -> CalibrationResult:
        """Calibrate using held-out set."""
        y_proba = self.model.predict_proba(X_cal)
        scores = self._compute_aps_score(y_proba, y_cal)

        self._quantile = compute_conformal_quantile(scores, self.alpha)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=self._quantile,
            n_calibration=len(y_cal),
            alpha=self.alpha,
        )
        self._is_fitted = True

        return self._calibration_result

    def predict(
        self,
        X: NDArray[np.floating[Any]],
        randomize: bool = True,
    ) -> PredictionSet:
        """
        Generate prediction sets.

        Args:
            X: Test features
            randomize: If True, use randomization for smaller sets.
                      If False, include all classes at the threshold level.
        """
        self._check_is_fitted()
        assert self._quantile is not None

        y_proba = self.model.predict_proba(X)
        n_samples, n_classes = y_proba.shape

        sets: list[set[int]] = []

        for i in range(n_samples):
            probs = y_proba[i]

            # Sort classes by probability (descending)
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]

            # Include classes until cumulative probability exceeds threshold
            cumsum = np.cumsum(sorted_probs)

            if randomize:
                # RSC 2020 Alg 1: include position p iff
                #   cumsum[p] - u * sorted_probs[p] <= quantile
                # Single shared u per test point; per-rank sorted_probs[p] in
                # the randomization term (not sorted_probs[0] — the F28 defect).
                u = float(self._rng.uniform(0, 1))
                scores_per_rank = cumsum - u * sorted_probs
                include_idx = int(np.searchsorted(scores_per_rank, self._quantile, side="right"))
            else:
                # Deterministic: include position p iff cumsum[p] <= quantile.
                include_idx = int(np.searchsorted(cumsum, self._quantile, side="right"))
            include_idx = min(include_idx, n_classes)

            pred_set = set(sorted_idx[:include_idx].tolist())
            sets.append(pred_set)

        return PredictionSet(
            sets=sets,
            probabilities=y_proba,
            alpha=self.alpha,
        )


class RAPSClassifier(BaseClassificationConformal[BaseEstimator]):
    """
    Regularized Adaptive Prediction Sets (RAPS).

    Extends APS with regularization that penalizes large prediction sets,
    leading to smaller sets while maintaining coverage.

    Score: s(x, y) = Σ_{j: π̂_j > π̂_y} π̂_j + U * π̂_y + λ * (o(y) - k_reg)+

    where o(y) is the rank of the true class (by probability) and k_reg
    is a regularization threshold.

    The penalty term encourages including fewer low-probability classes.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        lambda_reg: float = 0.01,
        k_reg: int = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize RAPS classifier.

        Args:
            model: Base classifier with predict_proba
            alpha: Target miscoverage rate
            lambda_reg: Regularization strength (higher = smaller sets)
            k_reg: Rank threshold for regularization penalty
            random_state: Random state
        """
        super().__init__(model, alpha, random_state)
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self._quantile: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.CLASSIFICATION

    def _compute_raps_score(
        self,
        y_proba: NDArray[np.floating[Any]],
        y_true: NDArray[np.int_],
    ) -> NDArray[np.floating[Any]]:
        """Compute RAPS nonconformity scores."""
        n_samples = len(y_true)
        scores = np.zeros(n_samples)

        u = self._rng.uniform(0, 1, n_samples)

        for i in range(n_samples):
            probs = y_proba[i]
            true_class = y_true[i]
            true_prob = probs[true_class]

            # Rank of true class (1 = most probable)
            rank = np.sum(probs > true_prob) + 1

            # Base APS score
            base_score = np.sum(probs[probs > true_prob]) + u[i] * true_prob

            # Regularization penalty
            penalty = self.lambda_reg * max(0, rank - self.k_reg)

            scores[i] = base_score + penalty

        return scores

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.int_],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "RAPSClassifier":
        """Fit and calibrate."""
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        self.model = clone(self.model)
        self.model.fit(X[train_idx], y[train_idx], **fit_params)

        self.calibrate(X[cal_idx], y[cal_idx])
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.int_],
    ) -> CalibrationResult:
        """Calibrate using held-out set."""
        y_proba = self.model.predict_proba(X_cal)
        scores = self._compute_raps_score(y_proba, y_cal)

        self._quantile = compute_conformal_quantile(scores, self.alpha)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=self._quantile,
            n_calibration=len(y_cal),
            alpha=self.alpha,
        )
        self._is_fitted = True

        return self._calibration_result

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionSet:
        """Generate prediction sets with regularization."""
        self._check_is_fitted()
        assert self._quantile is not None

        y_proba = self.model.predict_proba(X)
        n_samples, n_classes = y_proba.shape

        sets: list[set[int]] = []

        for i in range(n_samples):
            probs = y_proba[i]
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]

            # Compute cumulative score with regularization
            cumsum = np.cumsum(sorted_probs)
            penalties = self.lambda_reg * np.maximum(0, np.arange(1, n_classes + 1) - self.k_reg)
            cumsum_reg = cumsum + penalties

            # Random tie-breaking
            u = self._rng.uniform(0, 1)
            adjusted_cumsum = cumsum_reg - u * sorted_probs

            # RSC 2020 / Angelopoulos 2021 RAPS: include position p iff
            #   adjusted_cumsum[p] <= quantile
            # Randomization term `u * sorted_probs` is per-rank-correct
            # (full array, unlike APS's pre-fix sorted_probs[0]); only
            # F33 defect (ii) is fixed here — `side='right'` resolves the
            # inclusion-boundary overshoot that pre-P1.3's
            # `side='left') + 1` introduced.
            include_idx = int(np.searchsorted(adjusted_cumsum, self._quantile, side="right"))
            include_idx = min(include_idx, n_classes)

            pred_set = set(sorted_idx[:include_idx].tolist())
            sets.append(pred_set)

        return PredictionSet(
            sets=sets,
            probabilities=y_proba,
            alpha=self.alpha,
        )


class TopKConformalClassifier(BaseClassificationConformal[BaseEstimator]):
    """
    Top-K Conformal Classifier.

    Always returns exactly K classes, calibrated so that the true class
    is among the top-K with probability at least 1-alpha.

    Useful when you need fixed-size prediction sets (e.g., for UI display).
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        k: int = 3,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, random_state)
        self.k = k
        self._required_k: int | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.CLASSIFICATION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.int_],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "TopKConformalClassifier":
        """Fit and calibrate."""
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        self.model = clone(self.model)
        self.model.fit(X[train_idx], y[train_idx], **fit_params)

        self.calibrate(X[cal_idx], y[cal_idx])
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.int_],
    ) -> CalibrationResult:
        """
        Calibrate to find the minimum K that achieves coverage.

        Determines the smallest K such that top-K accuracy >= 1-alpha.
        """
        y_proba = self.model.predict_proba(X_cal)
        n_samples, n_classes = y_proba.shape

        # Compute rank of true class for each sample
        ranks = np.zeros(n_samples, dtype=np.int_)
        for i in range(n_samples):
            sorted_idx = np.argsort(-y_proba[i])
            ranks[i] = np.where(sorted_idx == y_cal[i])[0][0] + 1

        # Find minimum k that achieves coverage
        for k in range(1, n_classes + 1):
            coverage = np.mean(ranks <= k)
            if coverage >= 1 - self.alpha:
                self._required_k = k
                break
        else:
            self._required_k = n_classes

        # Use rank as score for reporting
        scores = ranks.astype(np.float64)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=float(self._required_k),
            n_calibration=n_samples,
            alpha=self.alpha,
        )
        self._is_fitted = True

        return self._calibration_result

    def predict(
        self,
        X: NDArray[np.floating[Any]],
        k: int | None = None,
    ) -> PredictionSet:
        """
        Generate top-K prediction sets.

        Args:
            X: Test features
            k: Override K (defaults to calibrated value)
        """
        self._check_is_fitted()

        k_to_use = k if k is not None else self._required_k
        assert k_to_use is not None

        y_proba = self.model.predict_proba(X)
        n_samples = y_proba.shape[0]

        sets: list[set[int]] = []
        for i in range(n_samples):
            top_k = np.argsort(-y_proba[i])[:k_to_use]
            sets.append(set(top_k.tolist()))

        return PredictionSet(
            sets=sets,
            probabilities=y_proba,
            alpha=self.alpha,
        )
