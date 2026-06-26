"""
Regression conformal prediction methods.

This module implements various conformal prediction methods for regression:
- Split (Inductive) Conformal Prediction
- Cross-Conformal Prediction
- Jackknife+
- CV+

All methods provide finite-sample valid prediction intervals.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from quantcore.uncertainty.conformal.base import (
    BaseRegressionConformal,
    CalibrationResult,
    PredictionInterval,
    PredictionType,
)
from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    compute_conformal_quantile,
)


class SplitConformalRegressor(BaseRegressionConformal[BaseEstimator]):
    """
    Split (Inductive) Conformal Prediction for regression.

    The simplest and most computationally efficient conformal method.
    Splits data into training and calibration sets, fits model on training,
    computes nonconformity scores on calibration, and uses the quantile
    of scores to construct prediction intervals.

    Attributes:
        model: Fitted base regressor
        alpha: Target miscoverage rate
        score_function: Function to compute nonconformity scores

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100)
        >>> cp = SplitConformalRegressor(model, alpha=0.1)
        >>> cp.fit(X_train, y_train)
        >>> intervals = cp.predict(X_test)
        >>> print(f"Coverage: {intervals.coverage(y_test):.2%}")
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize Split Conformal Regressor.

        Args:
            model: Sklearn-compatible regressor with fit() and predict()
            alpha: Target miscoverage rate (default 0.1 for 90% coverage)
            score_function: Nonconformity score function (default: absolute residual)
            random_state: Random state for reproducibility
        """
        super().__init__(model, alpha, random_state)
        self.score_function = score_function or absolute_residual_score
        self._quantile: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.REGRESSION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "SplitConformalRegressor":
        """
        Fit model and calibrate conformal predictor.

        Automatically splits data into training and calibration sets.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
            calibration_fraction: Fraction of data for calibration (default 0.25)
            **fit_params: Additional parameters passed to model.fit()

        Returns:
            self
        """
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        if n_cal < 10:
            raise ValueError(
                f"Calibration set too small ({n_cal} samples). "
                f"Need at least 10 samples for reliable calibration."
            )

        # Shuffle indices
        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, X_cal = X[train_idx], X[cal_idx]
        y_train, y_cal = y[train_idx], y[cal_idx]

        # Fit model on training set
        self.model = clone(self.model)
        self.model.fit(X_train, y_train, **fit_params)

        # Calibrate on calibration set
        self.calibrate(X_cal, y_cal)

        return self

    def fit_prefit(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> "SplitConformalRegressor":
        """
        Calibrate using a pre-fitted model.

        Use this when you've already fitted the model separately.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            self
        """
        self.calibrate(X_cal, y_cal)
        return self

    def calibrate(
        self,
        X_cal: NDArray[np.floating[Any]],
        y_cal: NDArray[np.floating[Any]],
    ) -> CalibrationResult:
        """
        Compute calibration scores and conformal quantile.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            CalibrationResult with scores and quantile
        """
        y_pred_cal = self.model.predict(X_cal)
        scores = self.score_function(y_cal, y_pred_cal)

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
    ) -> PredictionInterval:
        """
        Generate prediction intervals.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            PredictionInterval with lower, upper bounds
        """
        self._check_is_fitted()
        assert self._quantile is not None

        y_pred = self.model.predict(X)

        return PredictionInterval(
            lower=y_pred - self._quantile,
            upper=y_pred + self._quantile,
            point=y_pred,
            alpha=self.alpha,
        )


class CrossConformalRegressor(BaseRegressionConformal[BaseEstimator]):
    """
    Cross-Conformal Prediction for regression.

    Uses K-fold cross-validation to compute out-of-fold predictions,
    allowing all data to be used for both training and calibration.
    Produces slightly tighter intervals than split conformal while
    maintaining valid coverage.

    Note:
        Computational cost is K times split conformal due to K model fits.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        n_folds: int = 5,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """
        Initialize Cross-Conformal Regressor.

        Args:
            model: Sklearn-compatible regressor
            alpha: Target miscoverage rate
            n_folds: Number of cross-validation folds
            score_function: Nonconformity score function
            random_state: Random state for reproducibility
        """
        super().__init__(model, alpha, random_state)
        self.n_folds = n_folds
        self.score_function = score_function or absolute_residual_score
        self._fold_models: list[BaseEstimator] = []
        self._quantile: float | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.REGRESSION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "CrossConformalRegressor":
        """
        Fit using K-fold cross-validation.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
            **fit_params: Additional parameters passed to model.fit()

        Returns:
            self
        """
        n_samples = len(y)
        scores = np.zeros(n_samples)

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=int(self._rng.integers(2**31)) if self._rng else None,
        )

        self._fold_models = []

        for train_idx, val_idx in kf.split(X):
            # Fit model on fold
            fold_model = clone(self.model)
            fold_model.fit(X[train_idx], y[train_idx], **fit_params)
            self._fold_models.append(fold_model)

            # Compute out-of-fold scores
            y_pred_val = fold_model.predict(X[val_idx])
            scores[val_idx] = self.score_function(y[val_idx], y_pred_val)

        # Also fit on full data for prediction
        self.model = clone(self.model)
        self.model.fit(X, y, **fit_params)

        # Compute conformal quantile
        self._quantile = compute_conformal_quantile(scores, self.alpha)

        self._calibration_result = CalibrationResult(
            scores=scores,
            quantile=self._quantile,
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
        """Not applicable for cross-conformal - use fit() instead."""
        raise NotImplementedError(
            "Cross-conformal uses fit() for integrated training and calibration. "
            "Use SplitConformalRegressor if you need separate calibration."
        )

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """Generate prediction intervals using the full-data model."""
        self._check_is_fitted()
        assert self._quantile is not None

        y_pred = self.model.predict(X)

        return PredictionInterval(
            lower=y_pred - self._quantile,
            upper=y_pred + self._quantile,
            point=y_pred,
            alpha=self.alpha,
        )

    def predict_ensemble(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """
        Generate prediction intervals using ensemble of fold models.

        Uses mean of fold model predictions as point estimate.
        May provide more robust predictions but same interval width.
        """
        self._check_is_fitted()
        assert self._quantile is not None

        # Get predictions from all fold models
        preds = np.array([model.predict(X) for model in self._fold_models])
        y_pred = np.mean(preds, axis=0)

        return PredictionInterval(
            lower=y_pred - self._quantile,
            upper=y_pred + self._quantile,
            point=y_pred,
            alpha=self.alpha,
        )


class JackknifePlusRegressor(BaseRegressionConformal[BaseEstimator]):
    """
    Jackknife+ Conformal Prediction for regression.

    Fits n leave-one-out models; for each test point, forms the interval
    from quantiles of {μ̂^{-i}(X) ± R_i}_{i=1}^n where R_i is the LOO
    residual. Produces the tightest intervals among standard conformal
    methods, at the cost of n model fits.

    Coverage guarantee (Barber-Candès-Ramdas-Tibshirani 2021 Thm 1):
    under data exchangeability,

        P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − 2α   (worst-case lower bound)

    The 1 − 2α bound is provable and tight under adversarial data; on
    well-behaved (iid) data, empirical coverage typically tracks 1 − α
    closely. Callers requiring a strict 1 − α guarantee should use
    SplitConformalRegressor (wider intervals, single-split calibration).
    See CVPlusRegressor for a K-fold approximation with the same 1 − 2α
    bound using K model fits instead of n leave-one-out fits (roughly an
    n/K speedup relative to Jackknife+).

    Reference:
        Barber, Candès, Ramdas, Tibshirani (2021)
        "Predictive Inference with the Jackknife+"
        Annals of Statistics 49(1): 486-507. DOI 10.1214/20-AOS1965.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, random_state)
        self.score_function = score_function or absolute_residual_score
        self._loo_residuals: NDArray[np.floating[Any]] | None = None
        self._loo_predictions: NDArray[np.floating[Any]] | None = None
        self._X_train: NDArray[np.floating[Any]] | None = None
        self._y_train: NDArray[np.floating[Any]] | None = None
        self._loo_models: list[BaseEstimator] = []

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.REGRESSION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "JackknifePlusRegressor":
        """
        Fit using leave-one-out cross-validation.

        Warning: This requires n model fits and is expensive for large n.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
        """
        n_samples = len(y)

        if n_samples > 1000:
            import warnings

            warnings.warn(
                f"Jackknife+ requires {n_samples} model fits. "
                "Consider using CrossConformalRegressor for large datasets.",
                UserWarning,
            )

        self._X_train = X.copy()
        self._y_train = y.copy()
        self._loo_residuals = np.zeros(n_samples)
        self._loo_predictions = np.zeros(n_samples)
        self._loo_models = []

        # Leave-one-out fitting
        for i in range(n_samples):
            # Create LOO dataset
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            X_loo = X[mask]
            y_loo = y[mask]

            # Fit model
            loo_model = clone(self.model)
            loo_model.fit(X_loo, y_loo, **fit_params)
            self._loo_models.append(loo_model)

            # Predict on held-out point
            y_pred_i = loo_model.predict(X[i : i + 1])[0]
            self._loo_predictions[i] = y_pred_i
            self._loo_residuals[i] = self.score_function(np.array([y[i]]), np.array([y_pred_i]))[0]

        # Fit full model for base predictions
        self.model = clone(self.model)
        self.model.fit(X, y, **fit_params)

        self._calibration_result = CalibrationResult(
            scores=self._loo_residuals,
            quantile=float(np.quantile(self._loo_residuals, 1 - self.alpha)),
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
        """Not applicable for Jackknife+ - use fit() instead."""
        raise NotImplementedError(
            "Jackknife+ uses fit() for integrated LOO computation. "
            "Use SplitConformalRegressor if you need separate calibration."
        )

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """
        Generate Jackknife+ prediction intervals.

        Uses the special Jackknife+ aggregation that accounts for
        the variability from each LOO model.
        """
        self._check_is_fitted()
        assert self._loo_residuals is not None
        assert self._X_train is not None

        n_test = X.shape[0]
        n_train = len(self._loo_residuals)

        # Get predictions from all LOO models
        loo_preds = np.array([model.predict(X) for model in self._loo_models])

        # Jackknife+ quantile aggregation per Barber et al. 2021 Alg 1.
        # Coverage guarantee documented in class docstring (1 − 2α worst
        # case under exchangeability; empirical ≈ 1 − α on iid data).
        lower = np.zeros(n_test)
        upper = np.zeros(n_test)

        for j in range(n_test):
            # For test point j, construct interval endpoints from each LOO model
            lower_endpoints = loo_preds[:, j] - self._loo_residuals
            upper_endpoints = loo_preds[:, j] + self._loo_residuals

            # Take appropriate quantiles
            k = int(np.ceil((1 - self.alpha) * (n_train + 1)))
            k = min(k, n_train)  # Handle edge case

            lower[j] = np.sort(lower_endpoints)[n_train - k]
            upper[j] = np.sort(upper_endpoints)[k - 1]

        y_pred = self.model.predict(X)

        return PredictionInterval(
            lower=lower,
            upper=upper,
            point=y_pred,
            alpha=self.alpha,
        )


class CVPlusRegressor(BaseRegressionConformal[BaseEstimator]):
    """
    CV+ (Cross-Validation Plus) Conformal Prediction.

    A computationally efficient K-fold approximation to Jackknife+:
    it fits K models instead of n, yielding roughly an n/K speedup
    relative to Jackknife+.

    Coverage guarantee (Barber-Candès-Ramdas-Tibshirani 2021, §4):
    under data exchangeability, CV+ inherits the same worst-case
    1 − 2α lower bound as Jackknife+ (see JackknifePlusRegressor
    docstring for the full statement). K-fold construction trades
    fewer fits for coverage that can degrade as K decreases; empirical
    coverage typically ≈ 1 − α on iid data for K ≥ 5.

    Reference:
        Barber, Candès, Ramdas, Tibshirani (2021)
        "Predictive Inference with the Jackknife+"
        Annals of Statistics 49(1): 486-507. DOI 10.1214/20-AOS1965.
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        n_folds: int = 5,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(model, alpha, random_state)
        self.n_folds = n_folds
        self.score_function = score_function or absolute_residual_score
        self._fold_models: list[BaseEstimator] = []
        self._fold_indices: list[NDArray[np.intp]] = []
        self._residuals: NDArray[np.floating[Any]] | None = None
        self._X_train: NDArray[np.floating[Any]] | None = None

    @property
    def prediction_type(self) -> PredictionType:
        return PredictionType.REGRESSION

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "CVPlusRegressor":
        """
        Fit using K-fold cross-validation with CV+ aggregation.
        """
        n_samples = len(y)
        self._X_train = X.copy()
        self._residuals = np.zeros(n_samples)

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=int(self._rng.integers(2**31)) if self._rng else None,
        )

        self._fold_models = []
        self._fold_indices = []

        for train_idx, val_idx in kf.split(X):
            # Fit model on fold
            fold_model = clone(self.model)
            fold_model.fit(X[train_idx], y[train_idx], **fit_params)
            self._fold_models.append(fold_model)
            self._fold_indices.append(val_idx)

            # Compute out-of-fold residuals
            y_pred_val = fold_model.predict(X[val_idx])
            self._residuals[val_idx] = self.score_function(y[val_idx], y_pred_val)

        # Fit full model
        self.model = clone(self.model)
        self.model.fit(X, y, **fit_params)

        self._calibration_result = CalibrationResult(
            scores=self._residuals,
            quantile=float(np.quantile(self._residuals, 1 - self.alpha)),
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
        """Not applicable for CV+ - use fit() instead."""
        raise NotImplementedError("CV+ uses fit() for integrated CV computation.")

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """
        Generate CV+ prediction intervals.
        """
        self._check_is_fitted()
        assert self._residuals is not None

        n_test = X.shape[0]
        n_train = len(self._residuals)

        # Get predictions from all fold models
        fold_preds = np.array([model.predict(X) for model in self._fold_models])

        # CV+ aggregation
        lower = np.zeros(n_test)
        upper = np.zeros(n_test)

        for j in range(n_test):
            # For each fold model, compute interval endpoints using that fold's residuals
            lower_endpoints = []
            upper_endpoints = []

            for k, (fold_model, val_idx) in enumerate(zip(self._fold_models, self._fold_indices)):
                fold_residuals = self._residuals[val_idx]
                pred = fold_preds[k, j]

                lower_endpoints.extend(pred - fold_residuals)
                upper_endpoints.extend(pred + fold_residuals)

            lower_endpoints = np.array(lower_endpoints)
            upper_endpoints = np.array(upper_endpoints)

            # CV+ quantiles
            k = int(np.ceil((1 - self.alpha) * (n_train + 1)))
            k = min(k, n_train)

            lower[j] = np.sort(lower_endpoints)[n_train - k]
            upper[j] = np.sort(upper_endpoints)[k - 1]

        y_pred = self.model.predict(X)

        return PredictionInterval(
            lower=lower,
            upper=upper,
            point=y_pred,
            alpha=self.alpha,
        )
