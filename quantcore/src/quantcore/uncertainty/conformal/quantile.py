"""
Conformalized Quantile Regression (CQR) for adaptive prediction intervals.

CQR produces prediction intervals that adapt to local uncertainty:
- Wider intervals in high-variance regions
- Tighter intervals in low-variance regions

This is critical for financial applications where volatility clustering
means constant-width intervals are inappropriate.

Reference:
    Romano, Patterson, Candès (2019) "Conformalized Quantile Regression"
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from quantcore.uncertainty.conformal.base import (
    BaseRegressionConformal,
    CalibrationResult,
    PredictionInterval,
    PredictionType,
)
from quantcore.uncertainty.conformal.scores import (
    asymmetric_quantile_score,
    compute_asymmetric_quantiles,
    compute_conformal_quantile,
    quantile_score,
)


class QuantileRegressorWrapper:
    """
    Wrapper to create quantile regressors from various model types.

    Supports:
    - Native quantile regressors (e.g., QuantileRegressor, GradientBoostingRegressor)
    - Neural networks with quantile loss
    - LightGBM/XGBoost with quantile objective
    """

    def __init__(
        self,
        model: BaseEstimator,
        quantile: float,
        model_type: Literal["sklearn", "lightgbm", "xgboost", "neural"] = "sklearn",
    ) -> None:
        """
        Initialize quantile regressor wrapper.

        Args:
            model: Base model to wrap
            quantile: Target quantile (e.g., 0.05 for 5th percentile)
            model_type: Type of model for proper configuration
        """
        self.model = clone(model)
        self.quantile = quantile
        self.model_type = model_type
        self._configure_model()

    def _configure_model(self) -> None:
        """Configure model for quantile regression."""
        if self.model_type == "sklearn":
            # Works for GradientBoostingRegressor, HistGradientBoostingRegressor
            if hasattr(self.model, "loss"):
                self.model.set_params(loss="quantile", alpha=self.quantile)
            elif hasattr(self.model, "quantile"):
                self.model.set_params(quantile=self.quantile)

        elif self.model_type == "lightgbm":
            self.model.set_params(
                objective="quantile",
                alpha=self.quantile,
            )

        elif self.model_type == "xgboost":
            self.model.set_params(
                objective="reg:quantileerror",
                quantile_alpha=self.quantile,
            )

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "QuantileRegressorWrapper":
        """Fit the quantile regressor."""
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict quantile."""
        return self.model.predict(X)


class CQRRegressor(BaseRegressionConformal[BaseEstimator]):
    """
    Conformalized Quantile Regression.

    Produces adaptive prediction intervals by:
    1. Training quantile regressors for lower and upper quantiles
    2. Calibrating the intervals using conformal prediction

    The resulting intervals:
    - Have exact finite-sample coverage (unlike raw quantile regression)
    - Adapt to heteroscedasticity (unlike standard conformal)
    - Are asymmetric when the conditional distribution is skewed

    Example:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> model = GradientBoostingRegressor()
        >>> cqr = CQRRegressor(model, alpha=0.1)
        >>> cqr.fit(X_train, y_train)
        >>> intervals = cqr.predict(X_test)
        >>> # Intervals will be wider in high-volatility regions
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        quantile_model_type: Literal["sklearn", "lightgbm", "xgboost"] = "sklearn",
        symmetric: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize CQR regressor.

        Args:
            model: Base model that supports quantile regression
            alpha: Target miscoverage rate
            quantile_model_type: Type of model for quantile configuration
            symmetric: If True, use symmetric CQR. If False, use asymmetric.
            random_state: Random state for reproducibility
        """
        super().__init__(model, alpha, random_state)
        self.quantile_model_type = quantile_model_type
        self.symmetric = symmetric

        # Target quantiles
        self._alpha_lo = alpha / 2
        self._alpha_hi = 1 - alpha / 2

        self._model_lo: QuantileRegressorWrapper | None = None
        self._model_hi: QuantileRegressorWrapper | None = None
        self._quantile_correction: float | None = None
        self._quantile_lo: float | None = None
        self._quantile_hi: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.QUANTILE

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "CQRRegressor":
        """
        Fit quantile regressors and calibrate.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
            calibration_fraction: Fraction of data for calibration
            **fit_params: Additional parameters for model.fit()

        Returns:
            self
        """
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        if n_cal < 10:
            raise ValueError(f"Calibration set too small ({n_cal} samples)")

        # Split data
        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, X_cal = X[train_idx], X[cal_idx]
        y_train, y_cal = y[train_idx], y[cal_idx]

        # Fit lower quantile model
        self._model_lo = QuantileRegressorWrapper(
            clone(self.model),
            quantile=self._alpha_lo,
            model_type=self.quantile_model_type,
        )
        self._model_lo.fit(X_train, y_train, **fit_params)

        # Fit upper quantile model
        self._model_hi = QuantileRegressorWrapper(
            clone(self.model),
            quantile=self._alpha_hi,
            model_type=self.quantile_model_type,
        )
        self._model_hi.fit(X_train, y_train, **fit_params)

        # Calibrate
        self.calibrate(X_cal, y_cal)

        return self

    def fit_prefit(
        self,
        model_lo: BaseEstimator,
        model_hi: BaseEstimator,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> "CQRRegressor":
        """
        Calibrate with pre-fitted quantile models.

        Use this when you've already trained the quantile regressors.

        Args:
            model_lo: Fitted lower quantile model
            model_hi: Fitted upper quantile model
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            self
        """
        # Wrap pre-fitted models
        self._model_lo = QuantileRegressorWrapper(
            model_lo, self._alpha_lo, self.quantile_model_type
        )
        self._model_lo.model = model_lo  # Use the already fitted model

        self._model_hi = QuantileRegressorWrapper(
            model_hi, self._alpha_hi, self.quantile_model_type
        )
        self._model_hi.model = model_hi

        self.calibrate(X_cal, y_cal)
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> CalibrationResult:
        """
        Calibrate CQR using calibration set.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            CalibrationResult
        """
        if self._model_lo is None or self._model_hi is None:
            raise RuntimeError("Models must be fitted before calibration")

        # Get quantile predictions on calibration set
        y_pred_lo = self._model_lo.predict(X_cal)
        y_pred_hi = self._model_hi.predict(X_cal)

        if self.symmetric:
            # Symmetric CQR: single correction factor
            scores = quantile_score(y_cal, y_pred_lo, y_pred_hi)
            self._quantile_correction = compute_conformal_quantile(scores, self.alpha)

            self._calibration_result = CalibrationResult(
                scores=scores,
                quantile=self._quantile_correction,
                n_calibration=len(y_cal),
                alpha=self.alpha,
            )
        else:
            # Asymmetric CQR: separate corrections for lower and upper
            scores_lo, scores_hi = asymmetric_quantile_score(y_cal, y_pred_lo, y_pred_hi)
            self._quantile_lo, self._quantile_hi = compute_asymmetric_quantiles(
                scores_lo, scores_hi, self.alpha
            )

            # Store combined scores for reporting
            scores = np.maximum(scores_lo, scores_hi)
            self._calibration_result = CalibrationResult(
                scores=scores,
                quantile=max(self._quantile_lo, self._quantile_hi),
                n_calibration=len(y_cal),
                alpha=self.alpha,
            )

        self._is_fitted = True
        return self._calibration_result

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """
        Generate adaptive prediction intervals.

        Args:
            X: Test features

        Returns:
            PredictionInterval with adaptive widths
        """
        self._check_is_fitted()
        assert self._model_lo is not None
        assert self._model_hi is not None

        # Get quantile predictions
        y_pred_lo = self._model_lo.predict(X)
        y_pred_hi = self._model_hi.predict(X)

        if self.symmetric:
            assert self._quantile_correction is not None
            # Apply symmetric correction
            lower = y_pred_lo - self._quantile_correction
            upper = y_pred_hi + self._quantile_correction
        else:
            assert self._quantile_lo is not None
            assert self._quantile_hi is not None
            # Apply asymmetric corrections
            lower = y_pred_lo - self._quantile_lo
            upper = y_pred_hi + self._quantile_hi

        # Point prediction as midpoint
        point = (lower + upper) / 2

        return PredictionInterval(
            lower=lower,
            upper=upper,
            point=point,
            alpha=self.alpha,
        )

    def predict_raw_quantiles(
        self,
        X: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Get raw quantile predictions without conformal correction.

        Useful for comparing conformalized vs raw quantile regression.

        Returns:
            Tuple of (lower_quantile, upper_quantile) predictions
        """
        self._check_is_fitted()
        assert self._model_lo is not None
        assert self._model_hi is not None

        return self._model_lo.predict(X), self._model_hi.predict(X)


class CQRPlusRegressor(CQRRegressor):
    """
    CQR+ with CV+ style aggregation for tighter intervals.

    Combines CQR's adaptive intervals with CV+'s efficient use of data.
    Uses K-fold cross-validation to compute out-of-fold conformity scores,
    allowing all data to be used for both training and calibration.

    This typically produces tighter intervals than standard CQR while
    maintaining valid coverage.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        n_folds: int = 5,
        quantile_model_type: Literal["sklearn", "lightgbm", "xgboost"] = "sklearn",
        symmetric: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, quantile_model_type, symmetric, random_state)
        self.n_folds = n_folds
        self._fold_models_lo: list[QuantileRegressorWrapper] = []
        self._fold_models_hi: list[QuantileRegressorWrapper] = []

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float = 0.25,  # Ignored, uses all data
        **fit_params: Any,
    ) -> "CQRPlusRegressor":
        """
        Fit using K-fold cross-validation.

        Args:
            X: Features
            y: Targets
            calibration_fraction: Ignored (uses all data via CV)
            **fit_params: Additional parameters for model.fit()

        Returns:
            self
        """
        from sklearn.model_selection import KFold

        n_samples = len(y)
        scores = np.zeros(n_samples)

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=int(self._rng.integers(2**31)) if self._rng else None,
        )

        self._fold_models_lo = []
        self._fold_models_hi = []

        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Fit lower quantile model
            model_lo = QuantileRegressorWrapper(
                clone(self.model),
                quantile=self._alpha_lo,
                model_type=self.quantile_model_type,
            )
            model_lo.fit(X_train, y_train, **fit_params)
            self._fold_models_lo.append(model_lo)

            # Fit upper quantile model
            model_hi = QuantileRegressorWrapper(
                clone(self.model),
                quantile=self._alpha_hi,
                model_type=self.quantile_model_type,
            )
            model_hi.fit(X_train, y_train, **fit_params)
            self._fold_models_hi.append(model_hi)

            # Compute out-of-fold scores
            y_pred_lo = model_lo.predict(X_val)
            y_pred_hi = model_hi.predict(X_val)
            scores[val_idx] = quantile_score(y_val, y_pred_lo, y_pred_hi)

        # Fit full-data models for prediction
        self._model_lo = QuantileRegressorWrapper(
            clone(self.model),
            quantile=self._alpha_lo,
            model_type=self.quantile_model_type,
        )
        self._model_lo.fit(X, y, **fit_params)

        self._model_hi = QuantileRegressorWrapper(
            clone(self.model),
            quantile=self._alpha_hi,
            model_type=self.quantile_model_type,
        )
        self._model_hi.fit(X, y, **fit_params)

        # Compute conformal quantile
        self._quantile_correction = compute_conformal_quantile(scores, self.alpha)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=self._quantile_correction,
            n_calibration=n_samples,
            alpha=self.alpha,
        )
        self._is_fitted = True

        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> CalibrationResult:
        """Not applicable for CQR+ - use fit() instead."""
        raise NotImplementedError("CQR+ uses fit() for integrated CV computation.")
