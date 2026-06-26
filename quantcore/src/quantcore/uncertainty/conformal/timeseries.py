"""
Time Series Conformal Prediction.

Standard conformal prediction assumes exchangeability (roughly i.i.d. data),
which fails for time series due to temporal dependence and distribution shift.

This module implements methods that extend conformal prediction to time series:
- Adaptive Conformal Inference (ACI): Handles distribution shift
- Rolling window conformal: Fixed window calibration
- Weighted conformal: Exponentially decaying weights

References:
    Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
    Zaffran et al. (2022) "Adaptive Conformal Predictions for Time Series"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from quantcore.uncertainty.conformal.base import (
    PredictionInterval,
)
from quantcore.uncertainty.conformal.diagnostics import effective_sample_size
from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    compute_conformal_quantile,
)


@dataclass
class ACIState:
    """
    State container for Adaptive Conformal Inference.

    Tracks the adaptive alpha parameter and history of errors.
    """

    alpha_t: float  # Current adaptive alpha
    errors: list[bool] = field(default_factory=list)  # Coverage errors
    alphas: list[float] = field(default_factory=list)  # Alpha history
    scores: list[float] = field(default_factory=list)  # Score history
    quantiles: list[float] = field(default_factory=list)  # Quantile history

    @property
    def empirical_coverage(self) -> float:
        """Compute empirical coverage so far."""
        if not self.errors:
            return 1.0
        return 1 - np.mean(self.errors)

    @property
    def n_steps(self) -> int:
        """Number of time steps processed."""
        return len(self.errors)


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for time series.

    ACI maintains valid coverage under distribution shift by dynamically
    adjusting the miscoverage rate based on observed errors. When coverage
    drops below target, it widens intervals; when coverage exceeds target,
    it tightens them.

    The key insight: instead of using fixed α, use adaptive α_t that responds
    to recent coverage:
        α_{t+1} = α_t + γ * (α - err_t)

    where err_t = 1 if y_t was outside the interval, 0 otherwise.

    Attributes:
        target_alpha: Target miscoverage rate
        gamma: Learning rate for alpha adaptation
        window_size: Size of rolling window for score computation

    Example:
        >>> aci = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
        >>> for t in range(T):
        ...     interval = aci.predict(model, X[t:t+1], scores_buffer)
        ...     # Observe y[t]
        ...     aci.update(y[t], interval)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        window_size: int = 100,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        clip_alpha: bool = True,
        alpha_min: float = 0.001,
        alpha_max: float = 0.5,
    ) -> None:
        """
        Initialize ACI.

        Args:
            alpha: Target miscoverage rate
            gamma: Learning rate for alpha adaptation (higher = faster adaptation)
            window_size: Rolling window size for score quantile computation
            score_function: Nonconformity score function
            clip_alpha: Whether to clip alpha to [alpha_min, alpha_max]
            alpha_min: Minimum allowed alpha
            alpha_max: Maximum allowed alpha
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {window_size}")

        self.target_alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.score_function = score_function or absolute_residual_score
        self.clip_alpha = clip_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Initialize state
        self._state = ACIState(alpha_t=alpha)
        self._score_buffer: list[float] = []

    @property
    def state(self) -> ACIState:
        """Current ACI state."""
        return self._state

    @property
    def current_alpha(self) -> float:
        """Current adaptive alpha value."""
        return self._state.alpha_t

    def reset(self) -> None:
        """Reset ACI state."""
        self._state = ACIState(alpha_t=self.target_alpha)
        self._score_buffer = []

    def _compute_quantile(self) -> float:
        """Compute conformal quantile from score buffer."""
        if len(self._score_buffer) < 10:
            # Not enough data, use conservative estimate
            return float("inf")

        # Use most recent window_size scores
        recent_scores = np.array(self._score_buffer[-self.window_size :])
        return compute_conformal_quantile(recent_scores, self._state.alpha_t)

    def _update_alpha(self, error: bool) -> None:
        """
        Update adaptive alpha based on observed error.

        Args:
            error: True if the true value was outside the prediction interval
        """
        # ACI update rule: α_{t+1} = α_t + γ * (α - err_t)
        # If error (err_t=1): alpha increases -> wider intervals
        # If covered (err_t=0): alpha decreases -> tighter intervals
        err_t = 1.0 if error else 0.0
        new_alpha = self._state.alpha_t + self.gamma * (self.target_alpha - err_t)

        if self.clip_alpha:
            new_alpha = np.clip(new_alpha, self.alpha_min, self.alpha_max)

        self._state.alpha_t = new_alpha

    def predict_step(
        self,
        model: BaseEstimator,
        X_t: NDArray[np.floating[Any]],
        return_quantile: bool = False,
    ) -> PredictionInterval | tuple[PredictionInterval, float]:
        """
        Generate prediction interval for a single time step.

        Args:
            model: Fitted prediction model
            X_t: Features for time t, shape (1, n_features) or (n_features,)
            return_quantile: Whether to also return the quantile used

        Returns:
            PredictionInterval (and optionally the quantile)
        """
        X_t = np.atleast_2d(X_t)
        y_pred = model.predict(X_t)

        quantile = self._compute_quantile()
        self._state.quantiles.append(quantile)

        interval = PredictionInterval(
            lower=y_pred - quantile,
            upper=y_pred + quantile,
            point=y_pred,
            alpha=self._state.alpha_t,
        )

        if return_quantile:
            return interval, quantile
        return interval

    def update_step(
        self,
        y_true: float,
        y_pred: float,
        interval: PredictionInterval,
    ) -> bool:
        """
        Update ACI state after observing true value.

        Args:
            y_true: Observed true value
            y_pred: Model prediction
            interval: Prediction interval that was issued

        Returns:
            True if there was a coverage error (y_true outside interval)
        """
        # Compute nonconformity score
        score = float(self.score_function(np.array([y_true]), np.array([y_pred]))[0])
        self._score_buffer.append(score)

        # Check coverage
        error = not interval.contains(np.array([y_true]))[0]

        # Update state
        self._state.errors.append(error)
        self._state.scores.append(score)
        self._state.alphas.append(self._state.alpha_t)

        # Update alpha
        self._update_alpha(error)

        return error

    def run_online(
        self,
        model: BaseEstimator,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        warmup: int = 50,
    ) -> tuple[list[PredictionInterval], ACIState]:
        """
        Run ACI on a full time series in online fashion.

        Args:
            model: Fitted prediction model
            X: Features, shape (T, n_features)
            y: True values, shape (T,)
            warmup: Number of initial steps to use for warmup (no intervals)

        Returns:
            Tuple of (list of intervals, final state)
        """
        T = len(y)

        if warmup >= T:
            raise ValueError(f"warmup ({warmup}) must be less than T ({T})")

        self.reset()
        intervals: list[PredictionInterval] = []

        # Warmup phase: accumulate scores without issuing intervals
        for t in range(warmup):
            y_pred = model.predict(X[t : t + 1])[0]
            score = float(self.score_function(np.array([y[t]]), np.array([y_pred]))[0])
            self._score_buffer.append(score)

        # Online phase: issue intervals and update
        for t in range(warmup, T):
            interval = self.predict_step(model, X[t : t + 1])
            intervals.append(interval)

            y_pred = model.predict(X[t : t + 1])[0]
            self.update_step(y[t], y_pred, interval)

        return intervals, self._state


class RollingConformalRegressor:
    """
    Rolling Window Conformal Prediction for time series.

    Uses a fixed-size rolling window of recent observations for calibration.
    Simpler than ACI but doesn't adapt the miscoverage rate.

    Suitable when the data distribution is relatively stable but you want
    to track gradual changes.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        window_size: int = 100,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        min_samples: int = 30,
    ) -> None:
        """
        Initialize Rolling Conformal Regressor.

        Args:
            model: Base regressor (should be pre-fitted or will be fitted)
            alpha: Target miscoverage rate
            window_size: Size of rolling calibration window
            score_function: Nonconformity score function
            min_samples: Minimum samples needed before issuing intervals
        """
        self.model = model
        self.alpha = alpha
        self.window_size = window_size
        self.score_function = score_function or absolute_residual_score
        self.min_samples = min_samples

        self._score_buffer: list[float] = []
        self._is_fitted = False

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "RollingConformalRegressor":
        """
        Fit the base model and initialize with calibration scores.

        Args:
            X: Training features
            y: Training targets
        """
        self.model = clone(self.model)
        self.model.fit(X, y, **fit_params)

        # Initialize score buffer with training residuals
        y_pred = self.model.predict(X)
        scores = self.score_function(y, y_pred)
        self._score_buffer = list(scores[-self.window_size :])

        self._is_fitted = True
        return self

    def _compute_quantile(self) -> float:
        """Compute conformal quantile from score buffer."""
        if len(self._score_buffer) < self.min_samples:
            return float("inf")

        recent_scores = np.array(self._score_buffer[-self.window_size :])
        return compute_conformal_quantile(recent_scores, self.alpha)

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """
        Generate prediction intervals.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            PredictionInterval
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        y_pred = self.model.predict(X)
        quantile = self._compute_quantile()

        return PredictionInterval(
            lower=y_pred - quantile,
            upper=y_pred + quantile,
            point=y_pred,
            alpha=self.alpha,
        )

    def update(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        refit: bool = False,
        **fit_params: Any,
    ) -> None:
        """
        Update with new observations.

        Args:
            X: New features
            y: New targets
            refit: Whether to refit the model on accumulated data
        """
        y_pred = self.model.predict(X)
        scores = self.score_function(y, y_pred)

        self._score_buffer.extend(scores.tolist())

        # Keep only window_size most recent scores
        if len(self._score_buffer) > self.window_size * 2:
            self._score_buffer = self._score_buffer[-self.window_size :]


class WeightedConformalRegressor:
    """
    Weighted Conformal Prediction with exponentially decaying weights.

    Assigns higher weights to recent observations, making the predictor
    more responsive to recent distribution changes while still using
    historical data.

    The quantile is computed from the weighted empirical distribution
    of nonconformity scores.

    References
    ----------
    Operationally an instance of NexCP from Barber et al. (2023)
    "Conformal Prediction Beyond Exchangeability" (Annals of
    Statistics). Geometric-decay weights ``w_i = decay^{n-i}`` are
    the closed-form choice that yields a coverage gap of

        P(Y_{t+1} ∈ Ĉ_α(X_{t+1})) ≥ 1 - α - ∑ w̃_i · d_TV(P_i, P_{t+1})

    where ``d_TV`` is the total-variation distance between the
    data-generating distribution at time ``i`` and ``t+1``. The
    bound is explicit but ``d_TV`` is not estimable in practice
    (the test-time distribution is unknown), so the bound is
    theoretical comfort, not a runtime monitor.

    Operational handle for "how much weight is concentrated on
    recent observations" is the Kish effective sample size
    ``n_eff = (∑w)² / ∑w²``, exposed as the ``n_eff`` property
    on this class. Per the 2026-04-29 conformal-stack review's
    failure-mode table, ``n_eff < 100`` should trigger an
    investigation: the score quantile becomes degenerate, intervals
    balloon, and the practical decision is to widen ``decay``
    (less aggressive) or grow the calibration window.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        decay: float = 0.99,
        max_samples: int = 500,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
    ) -> None:
        """
        Initialize Weighted Conformal Regressor.

        Args:
            model: Base regressor
            alpha: Target miscoverage rate
            decay: Exponential decay factor (0.99 = 1% decay per step)
            max_samples: Maximum number of scores to keep
            score_function: Nonconformity score function
        """
        self.model = model
        self.alpha = alpha
        self.decay = decay
        self.max_samples = max_samples
        self.score_function = score_function or absolute_residual_score

        self._scores: list[float] = []
        self._weights: list[float] = []
        self._is_fitted = False

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "WeightedConformalRegressor":
        """Fit model and initialize with calibration scores."""
        self.model = clone(self.model)
        self.model.fit(X, y, **fit_params)

        y_pred = self.model.predict(X)
        scores = self.score_function(y, y_pred)

        # Initialize with unit weights (will decay over time)
        n = len(scores)
        weights = self.decay ** np.arange(n - 1, -1, -1)

        self._scores = list(scores)
        self._weights = list(weights)
        self._is_fitted = True

        return self

    def _compute_weighted_quantile(self) -> float:
        """Compute weighted conformal quantile."""
        if len(self._scores) < 10:
            return float("inf")

        scores = np.array(self._scores)
        weights = np.array(self._weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Sort scores and corresponding weights
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Compute weighted quantile
        cumsum = np.cumsum(sorted_weights)
        quantile_level = 1 - self.alpha

        # Find the score where cumulative weight exceeds quantile level
        idx = np.searchsorted(cumsum, quantile_level)
        idx = min(idx, len(sorted_scores) - 1)

        return float(sorted_scores[idx])

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """Generate prediction intervals."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        y_pred = self.model.predict(X)
        quantile = self._compute_weighted_quantile()

        return PredictionInterval(
            lower=y_pred - quantile,
            upper=y_pred + quantile,
            point=y_pred,
            alpha=self.alpha,
        )

    def update(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
    ) -> None:
        """Update with new observations."""
        y_pred = self.model.predict(X)
        scores = self.score_function(y, y_pred)

        # Decay existing weights
        self._weights = [w * self.decay for w in self._weights]

        # Add new scores with unit weight
        self._scores.extend(scores.tolist())
        self._weights.extend([1.0] * len(scores))

        # Trim to max_samples
        if len(self._scores) > self.max_samples:
            self._scores = self._scores[-self.max_samples :]
            self._weights = self._weights[-self.max_samples :]

    @property
    def n_eff(self) -> float:
        """Kish effective sample size of the current weight buffer.

        Dispatches to ``diagnostics.effective_sample_size`` rather
        than reimplementing the formula — the dispatch contract is
        pinned so a future "optimization" that inlines the math
        gets caught. Returns NaN when unfitted (empty weight list).

        See class docstring for operational interpretation
        (``n_eff < 100`` is the practical investigation threshold).
        """
        return effective_sample_size(self._weights)


def expanding_window_backtest(
    model: BaseEstimator,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    alpha: float = 0.1,
    initial_train_size: int = 100,
    step_size: int = 1,
    refit_frequency: int = 20,
    method: Literal["split", "rolling", "aci"] = "rolling",
    **method_kwargs: Any,
) -> dict[str, Any]:
    """
    Backtest conformal prediction with expanding window.

    Simulates real-world deployment where:
    1. Model is trained on historical data
    2. Predictions are made one step ahead
    3. True values are observed and used to update calibration

    Args:
        model: Base regressor
        X: Full feature matrix, shape (T, n_features)
        y: Full target vector, shape (T,)
        alpha: Target miscoverage rate
        initial_train_size: Initial training set size
        step_size: Number of steps between predictions
        refit_frequency: How often to refit the model
        method: Conformal method ("split", "rolling", "aci")
        **method_kwargs: Additional arguments for the conformal method

    Returns:
        Dictionary with backtest results
    """
    T = len(y)

    if initial_train_size >= T:
        raise ValueError("initial_train_size must be less than T")

    results = {
        "intervals": [],
        "y_true": [],
        "y_pred": [],
        "covered": [],
        "interval_width": [],
        "timestamps": [],
    }

    current_train_end = initial_train_size

    # Initial fit
    fitted_model = clone(model)
    fitted_model.fit(X[:current_train_end], y[:current_train_end])

    if method == "rolling":
        cp = RollingConformalRegressor(
            fitted_model,
            alpha=alpha,
            **method_kwargs,
        )
        # Initialize with training data scores
        y_pred_train = fitted_model.predict(X[:current_train_end])
        scores = absolute_residual_score(y[:current_train_end], y_pred_train)
        cp._score_buffer = list(scores)
        cp._is_fitted = True

    elif method == "aci":
        aci = AdaptiveConformalInference(alpha=alpha, **method_kwargs)
        # Warmup with training data
        for t in range(current_train_end):
            y_pred_t = fitted_model.predict(X[t : t + 1])[0]
            score = float(absolute_residual_score(np.array([y[t]]), np.array([y_pred_t]))[0])
            aci._score_buffer.append(score)

    # Online prediction loop
    t = current_train_end
    steps_since_refit = 0

    while t < T:
        # Generate prediction
        X_t = X[t : t + 1]
        y_pred_t = fitted_model.predict(X_t)[0]

        if method == "rolling":
            interval = cp.predict(X_t)
        elif method == "aci":
            interval = aci.predict_step(fitted_model, X_t)

        # Record results
        y_true_t = y[t]
        covered = interval.contains(np.array([y_true_t]))[0]

        results["intervals"].append(interval)
        results["y_true"].append(y_true_t)
        results["y_pred"].append(y_pred_t)
        results["covered"].append(covered)
        results["interval_width"].append(interval.width[0])
        results["timestamps"].append(t)

        # Update conformal predictor
        if method == "rolling":
            score = float(absolute_residual_score(np.array([y_true_t]), np.array([y_pred_t]))[0])
            cp._score_buffer.append(score)
        elif method == "aci":
            aci.update_step(y_true_t, y_pred_t, interval)

        # Check if we need to refit
        steps_since_refit += 1
        if steps_since_refit >= refit_frequency:
            current_train_end = t + 1
            fitted_model = clone(model)
            fitted_model.fit(X[:current_train_end], y[:current_train_end])
            steps_since_refit = 0

            if method == "rolling":
                cp.model = fitted_model

        t += step_size

    # Compute summary statistics
    results["coverage"] = np.mean(results["covered"])
    results["mean_width"] = np.mean(results["interval_width"])
    results["median_width"] = np.median(results["interval_width"])

    return results
