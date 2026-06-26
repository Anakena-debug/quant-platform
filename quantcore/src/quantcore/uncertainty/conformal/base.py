"""
Base classes and protocols for conformal prediction.

This module defines the abstract interfaces that all conformal predictors
must implement, ensuring consistent API across different methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


# Type variables for generic conformal predictors
X = TypeVar("X", bound=ArrayLike)
Y = TypeVar("Y", bound=ArrayLike)
Model = TypeVar("Model", bound="BaseEstimator")


class PredictionType(Enum):
    """Type of prediction output."""

    REGRESSION = auto()
    CLASSIFICATION = auto()
    QUANTILE = auto()


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """
    Immutable container for prediction intervals.

    Attributes:
        lower: Lower bounds of prediction intervals, shape (n_samples,)
        upper: Upper bounds of prediction intervals, shape (n_samples,)
        point: Point predictions (optional), shape (n_samples,)
        alpha: Miscoverage rate used to construct intervals
    """

    lower: NDArray[np.floating[Any]]
    upper: NDArray[np.floating[Any]]
    point: NDArray[np.floating[Any]] | None = None
    alpha: float = 0.1

    def __post_init__(self) -> None:
        """Validate interval consistency."""
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"Lower and upper bounds must have same shape. "
                f"Got {self.lower.shape} and {self.upper.shape}"
            )
        if np.any(self.lower > self.upper):
            raise ValueError("Lower bounds must not exceed upper bounds")
        if self.point is not None and self.point.shape != self.lower.shape:
            raise ValueError(
                f"Point predictions must have same shape as bounds. "
                f"Got {self.point.shape} and {self.lower.shape}"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1). Got {self.alpha}")

    @property
    def width(self) -> NDArray[np.floating[Any]]:
        """Interval widths."""
        return self.upper - self.lower

    @property
    def mean_width(self) -> float:
        """Average interval width."""
        return float(np.mean(self.width))

    @property
    def median_width(self) -> float:
        """Median interval width."""
        return float(np.median(self.width))

    def contains(self, y: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
        """Check if true values fall within intervals."""
        return (self.lower <= y) & (y <= self.upper)

    def coverage(self, y: NDArray[np.floating[Any]]) -> float:
        """Compute empirical coverage rate."""
        return float(np.mean(self.contains(y)))


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """
    Immutable container for classification prediction sets.

    Attributes:
        sets: List of prediction sets, one per sample
        probabilities: Predicted probabilities for all classes, shape (n_samples, n_classes)
        alpha: Miscoverage rate used to construct sets
    """

    sets: list[set[int]]
    probabilities: NDArray[np.floating[Any]]
    alpha: float = 0.1

    def __post_init__(self) -> None:
        """Validate prediction set consistency."""
        if len(self.sets) != self.probabilities.shape[0]:
            raise ValueError(
                f"Number of sets must match number of samples. "
                f"Got {len(self.sets)} sets and {self.probabilities.shape[0]} samples"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1). Got {self.alpha}")

    @property
    def sizes(self) -> NDArray[np.int_]:
        """Sizes of prediction sets."""
        return np.array([len(s) for s in self.sets], dtype=np.int_)

    @property
    def mean_size(self) -> float:
        """Average prediction set size."""
        return float(np.mean(self.sizes))

    def contains(self, y: NDArray[np.int_]) -> NDArray[np.bool_]:
        """Check if true labels are in prediction sets."""
        return np.array([y[i] in self.sets[i] for i in range(len(y))], dtype=np.bool_)

    def coverage(self, y: NDArray[np.int_]) -> float:
        """Compute empirical coverage rate."""
        return float(np.mean(self.contains(y)))


@dataclass
class CalibrationResult:
    """
    Result of calibration procedure.

    Attributes:
        scores: Nonconformity scores on calibration set
        quantile: Computed conformal quantile threshold
        n_calibration: Number of calibration samples used
        alpha: Target miscoverage rate
    """

    scores: NDArray[np.floating[Any]]
    quantile: float
    n_calibration: int
    alpha: float

    @property
    def effective_alpha(self) -> float:
        """Effective miscoverage rate after finite-sample correction."""
        return (1 - self.alpha) * (1 + 1 / self.n_calibration)


@runtime_checkable
class NonconformityScore(Protocol):
    """Protocol for nonconformity score functions."""

    def __call__(
        self,
        y_true: NDArray[np.floating[Any]],
        y_pred: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute nonconformity scores.

        Args:
            y_true: True target values
            y_pred: Predicted values (or prediction parameters)
            **kwargs: Additional prediction outputs (e.g., predicted std)

        Returns:
            Nonconformity scores, same shape as y_true
        """
        ...


class BaseConformalPredictor(ABC, Generic[Model]):
    """
    Abstract base class for all conformal predictors.

    This class defines the interface that all conformal prediction methods
    must implement, regardless of whether they're for regression, classification,
    or other prediction tasks.

    Type Parameters:
        Model: The type of the underlying ML model

    Attributes:
        model: The fitted base model
        alpha: Target miscoverage rate (default 0.1 for 90% coverage)
        is_fitted: Whether the predictor has been calibrated
    """

    def __init__(
        self,
        model: Model,
        alpha: float = 0.1,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize conformal predictor.

        Args:
            model: Base ML model (will be cloned if not fitted)
            alpha: Target miscoverage rate, must be in (0, 1)
            random_state: Random state for reproducibility
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.model = model
        self.alpha = alpha
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._is_fitted = False
        self._calibration_result: CalibrationResult | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the conformal predictor has been calibrated."""
        return self._is_fitted

    @property
    def calibration_result(self) -> CalibrationResult | None:
        """Calibration result if fitted, None otherwise."""
        return self._calibration_result

    def _check_is_fitted(self) -> None:
        """Raise error if predictor is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                "Call fit() or calibrate() before predict()."
            )

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "BaseConformalPredictor[Model]":
        """
        Fit the conformal predictor.

        This includes fitting the base model and computing calibration scores.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            **fit_params: Additional parameters passed to model.fit()

        Returns:
            self
        """
        ...

    @abstractmethod
    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> CalibrationResult:
        """
        Calibrate using a held-out calibration set.

        Assumes the base model is already fitted.

        Args:
            X_cal: Calibration features, shape (n_calibration, n_features)
            y_cal: Calibration targets, shape (n_calibration,)

        Returns:
            CalibrationResult containing scores and quantile
        """
        ...

    @abstractmethod
    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval | PredictionSet:
        """
        Generate prediction intervals or sets.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            PredictionInterval for regression, PredictionSet for classification
        """
        ...

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"alpha={self.alpha}, "
            f"{fitted_str})"
        )


class BaseRegressionConformal(BaseConformalPredictor[Model]):
    """
    Base class for regression conformal predictors.

    Extends BaseConformalPredictor with regression-specific functionality.
    """

    @property
    @abstractmethod
    def prediction_type(self) -> PredictionType:
        """Type of prediction this conformal predictor produces."""
        return PredictionType.REGRESSION

    @abstractmethod
    def predict(self, X: NDArray[np.floating[Any]]) -> PredictionInterval:
        """Generate prediction intervals."""
        ...

    def predict_point(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Generate point predictions using the base model.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Point predictions, shape (n_samples,)
        """
        self._check_is_fitted()
        return self.model.predict(X)  # type: ignore[return-value]


class BaseClassificationConformal(BaseConformalPredictor[Model]):
    """
    Base class for classification conformal predictors.

    Extends BaseConformalPredictor with classification-specific functionality.
    """

    @property
    def prediction_type(self) -> PredictionType:
        """Type of prediction this conformal predictor produces."""
        return PredictionType.CLASSIFICATION

    @abstractmethod
    def predict(self, X: NDArray[np.floating[Any]]) -> PredictionSet:
        """Generate prediction sets."""
        ...

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Generate probability predictions using the base model.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        self._check_is_fitted()
        return self.model.predict_proba(X)  # type: ignore[return-value]


@dataclass
class ConformalConfig:
    """
    Configuration for conformal prediction experiments.

    Attributes:
        alpha: Target miscoverage rate
        calibration_fraction: Fraction of data to use for calibration in split CP
        n_bootstrap: Number of bootstrap samples for Jackknife+
        adaptive: Whether to use adaptive intervals (CQR)
        symmetric: Whether to use symmetric intervals
        random_state: Random state for reproducibility
    """

    alpha: float = 0.1
    calibration_fraction: float = 0.25
    n_bootstrap: int = 100
    adaptive: bool = False
    symmetric: bool = True
    random_state: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError(
                f"calibration_fraction must be in (0, 1), got {self.calibration_fraction}"
            )
        if self.n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be positive, got {self.n_bootstrap}")
