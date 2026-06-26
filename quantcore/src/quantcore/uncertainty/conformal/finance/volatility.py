"""
Conformal Prediction for Volatility Forecasting.

This module provides conformal prediction methods specifically designed for
volatility forecasting, a critical task in quantitative finance.

Key Features:
- Integration with GARCH/EGARCH models
- Realized volatility prediction with intervals
- Adaptive methods for volatility regime changes
- Heteroskedasticity-aware conformal prediction

The key insight: volatility itself is heteroskedastic, meaning prediction
intervals for volatility should widen during volatile periods. This module
handles this nested uncertainty properly.

References:
    Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
    Xu & Xie (2023) "Conformal Prediction for Time Series"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    compute_conformal_quantile,
)


@dataclass(frozen=True, slots=True)
class VolatilityForecast:
    """
    Container for volatility forecasts with prediction intervals.

    Attributes:
        point: Point forecast of volatility (e.g., σ or σ²)
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        alpha: Miscoverage rate
        is_variance: Whether forecasts are variance (σ²) or std dev (σ)
    """

    point: NDArray[np.floating[Any]]
    lower: NDArray[np.floating[Any]]
    upper: NDArray[np.floating[Any]]
    alpha: float
    is_variance: bool = False

    def to_std(self) -> "VolatilityForecast":
        """Convert to standard deviation if currently variance."""
        if not self.is_variance:
            return self
        return VolatilityForecast(
            point=np.sqrt(self.point),
            lower=np.sqrt(np.maximum(self.lower, 0)),
            upper=np.sqrt(self.upper),
            alpha=self.alpha,
            is_variance=False,
        )

    def to_variance(self) -> "VolatilityForecast":
        """Convert to variance if currently std dev."""
        if self.is_variance:
            return self
        return VolatilityForecast(
            point=self.point**2,
            lower=self.lower**2,
            upper=self.upper**2,
            alpha=self.alpha,
            is_variance=True,
        )

    @property
    def width(self) -> NDArray[np.floating[Any]]:
        """Interval widths."""
        return self.upper - self.lower

    def contains(self, realized: NDArray[np.floating[Any]]) -> NDArray[np.bool_]:
        """Check if realized volatility falls within intervals."""
        return (self.lower <= realized) & (realized <= self.upper)

    def coverage(self, realized: NDArray[np.floating[Any]]) -> float:
        """Compute empirical coverage."""
        return float(np.mean(self.contains(realized)))


class ConformalVolatility:
    """
    Conformal Prediction for Volatility Forecasting.

    Wraps any volatility forecasting model (GARCH, HAR, neural network)
    and provides distribution-free prediction intervals.

    The key challenge: volatility proxies (squared returns, realized volatility)
    are noisy measurements of true latent volatility. This class handles
    this measurement error appropriately.

    Example:
        >>> from arch import arch_model
        >>> garch = arch_model(returns, vol='Garch', p=1, q=1)
        >>> garch_fit = garch.fit()
        >>> cv = ConformalVolatility(alpha=0.1)
        >>> cv.fit(returns, garch_fit)
        >>> forecast = cv.predict(horizon=5)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        volatility_proxy: Literal["squared", "absolute", "realized"] = "squared",
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
    ) -> None:
        """
        Initialize Conformal Volatility predictor.

        Args:
            alpha: Target miscoverage rate
            volatility_proxy: How to estimate realized volatility
                - 'squared': Use squared returns
                - 'absolute': Use absolute returns
                - 'realized': Use realized volatility (requires high-freq data)
            score_function: Custom nonconformity score
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.volatility_proxy = volatility_proxy
        self.score_function = score_function or absolute_residual_score

        self._model: Any = None
        self._returns: NDArray[np.floating[Any]] | None = None
        self._quantile: float | None = None
        self._calibration_scores: NDArray[np.floating[Any]] | None = None
        self._is_fitted = False

    def _compute_proxy(
        self,
        returns: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute volatility proxy from returns."""
        if self.volatility_proxy == "squared":
            return returns**2
        elif self.volatility_proxy == "absolute":
            return np.abs(returns)
        elif self.volatility_proxy == "realized":
            # For daily data, this is just squared returns
            # For high-freq, would need intraday data
            return returns**2
        else:
            raise ValueError(f"Unknown volatility_proxy: {self.volatility_proxy}")

    def fit(
        self,
        returns: NDArray[np.floating[Any]],
        fitted_model: Any,
        calibration_fraction: float = 0.25,
    ) -> "ConformalVolatility":
        """
        Fit conformal volatility predictor.

        Args:
            returns: Return series used to fit the model
            fitted_model: Already-fitted volatility model (e.g., from arch)
            calibration_fraction: Fraction of data for calibration

        Returns:
            self
        """
        self._returns = np.asarray(returns).flatten()
        self._model = fitted_model
        n = len(self._returns)

        # Get in-sample conditional variance from model
        if hasattr(fitted_model, "conditional_volatility"):
            # arch package
            cond_vol = fitted_model.conditional_volatility
        elif hasattr(fitted_model, "conditional_variance"):
            cond_vol = np.sqrt(fitted_model.conditional_variance)
        else:
            raise ValueError(
                "Model must have conditional_volatility or conditional_variance attribute"
            )

        # Compute realized volatility proxy
        realized = self._compute_proxy(self._returns)

        # Use last portion for calibration
        n_cal = int(n * calibration_fraction)
        cal_start = n - n_cal

        vol_pred_cal = cond_vol[cal_start:]
        realized_cal = realized[cal_start:]

        # Handle proxy type (squared returns vs std dev predictions)
        if self.volatility_proxy == "squared":
            # Compare variance to variance
            vol_pred_cal = vol_pred_cal**2

        # Compute nonconformity scores
        scores = self.score_function(realized_cal, vol_pred_cal)
        self._calibration_scores = scores

        # Compute conformal quantile
        self._quantile = compute_conformal_quantile(scores, self.alpha)
        self._is_fitted = True

        return self

    def predict(
        self,
        horizon: int = 1,
        n_simulations: int = 1000,
    ) -> VolatilityForecast:
        """
        Generate volatility forecasts with prediction intervals.

        Args:
            horizon: Forecast horizon (number of periods)
            n_simulations: Number of simulations for multi-step forecasts

        Returns:
            VolatilityForecast with point and interval forecasts
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")
        if self._quantile is None or self._model is None:
            raise RuntimeError("Calibration not complete")

        # Get forecasts from underlying model
        if hasattr(self._model, "forecast"):
            # arch package
            forecast = self._model.forecast(horizon=horizon)
            if hasattr(forecast, "variance"):
                point_var = forecast.variance.values[-1]  # Last row
                point = np.sqrt(point_var)
            else:
                point = forecast.mean.values[-1]
        else:
            raise ValueError("Model must have forecast() method")

        point = np.atleast_1d(point)

        # Construct prediction intervals
        if self.volatility_proxy == "squared":
            # Intervals are for variance
            lower = point**2 - self._quantile
            upper = point**2 + self._quantile
            lower = np.maximum(lower, 0)  # Variance must be non-negative
            return VolatilityForecast(
                point=point**2,
                lower=lower,
                upper=upper,
                alpha=self.alpha,
                is_variance=True,
            ).to_std()
        else:
            # Intervals are for std dev
            lower = point - self._quantile
            upper = point + self._quantile
            lower = np.maximum(lower, 0)
            return VolatilityForecast(
                point=point,
                lower=lower,
                upper=upper,
                alpha=self.alpha,
                is_variance=False,
            )

    def calibrate_online(
        self,
        new_return: float,
        new_forecast: float,
        decay: float = 0.99,
    ) -> None:
        """
        Online calibration update with new observation.

        Args:
            new_return: Newly observed return
            new_forecast: Volatility forecast that was issued
            decay: Decay factor for old scores (for adaptivity)
        """
        if self._calibration_scores is None:
            raise RuntimeError("Must call fit() before online updates")

        # Compute new score
        realized = self._compute_proxy(np.array([new_return]))[0]
        if self.volatility_proxy == "squared":
            new_forecast = new_forecast**2

        new_score = float(self.score_function(np.array([realized]), np.array([new_forecast]))[0])

        # Decay old scores and add new
        self._calibration_scores = self._calibration_scores * decay
        self._calibration_scores = np.append(self._calibration_scores, new_score)

        # Keep buffer manageable
        if len(self._calibration_scores) > 1000:
            self._calibration_scores = self._calibration_scores[-500:]

        # Recompute quantile
        self._quantile = compute_conformal_quantile(self._calibration_scores, self.alpha)


class GARCHConformal:
    """
    Specialized conformal prediction for GARCH models.

    Handles the specific structure of GARCH(p,q) models:
    - σ²_t = ω + Σ α_i * ε²_{t-i} + Σ β_j * σ²_{t-j}

    Key insight: GARCH forecast errors are heteroskedastic and
    non-Gaussian even if innovations are Gaussian. This class
    properly accounts for this.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        use_standardized_residuals: bool = True,
    ) -> None:
        """
        Initialize GARCH Conformal predictor.

        Args:
            alpha: Target miscoverage rate
            use_standardized_residuals: If True, use standardized residuals
                for more efficient calibration
        """
        self.alpha = alpha
        self.use_standardized_residuals = use_standardized_residuals

        self._garch_model: Any = None
        self._quantile: float | None = None
        self._quantile_upper: float | None = None
        self._quantile_lower: float | None = None
        self._is_fitted = False

    def fit(
        self,
        returns: NDArray[np.floating[Any]],
        p: int = 1,
        q: int = 1,
        vol: str = "Garch",
        dist: str = "normal",
        calibration_fraction: float = 0.25,
    ) -> "GARCHConformal":
        """
        Fit GARCH model and calibrate conformal predictor.

        Args:
            returns: Return series
            p: GARCH p parameter (lags of variance)
            q: GARCH q parameter (lags of squared returns)
            vol: Volatility model ('Garch', 'EGarch', 'GJR-GARCH')
            dist: Innovation distribution ('normal', 't', 'skewt')
            calibration_fraction: Fraction for calibration

        Returns:
            self
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch package required: pip install arch")

        returns = np.asarray(returns).flatten()
        n = len(returns)
        n_cal = int(n * calibration_fraction)

        # Split data
        train_end = n - n_cal
        returns_train = returns[:train_end]

        # Fit GARCH model
        model = arch_model(
            returns_train * 100,  # Scale for numerical stability
            vol=vol,
            p=p,
            q=q,
            dist=dist,
        )
        self._garch_model = model.fit(disp="off")

        # Get conditional volatility for calibration period
        # Re-fit on full data up to each calibration point
        full_model = arch_model(
            returns * 100,
            vol=vol,
            p=p,
            q=q,
            dist=dist,
        )
        full_fit = full_model.fit(disp="off")

        cond_vol = full_fit.conditional_volatility / 100  # Rescale
        realized_vol = np.abs(returns)  # Use absolute returns as proxy

        # Calibration scores
        vol_pred_cal = cond_vol[train_end:]
        realized_cal = realized_vol[train_end:]

        if self.use_standardized_residuals:
            # Standardized residuals: ε_t / σ_t
            # These should be approximately i.i.d. if model is correct
            std_resid = realized_cal / vol_pred_cal
            scores = np.abs(std_resid - 1)  # Score based on deviation from 1
        else:
            scores = np.abs(realized_cal - vol_pred_cal)

        # Compute quantiles
        self._quantile = compute_conformal_quantile(scores, self.alpha)

        # For asymmetric intervals (volatility is bounded below by 0)
        resid = realized_cal - vol_pred_cal
        self._quantile_upper = compute_conformal_quantile(resid, self.alpha / 2)
        self._quantile_lower = compute_conformal_quantile(-resid, self.alpha / 2)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int = 1,
    ) -> VolatilityForecast:
        """
        Generate volatility forecast with prediction intervals.

        Args:
            horizon: Forecast horizon

        Returns:
            VolatilityForecast
        """
        if not self._is_fitted or self._garch_model is None:
            raise RuntimeError("Must fit() before predict()")

        # Get GARCH forecast
        forecast = self._garch_model.forecast(horizon=horizon)
        point_var = forecast.variance.values[-1] / 10000  # Rescale
        point = np.sqrt(point_var)
        point = np.atleast_1d(point)

        if self.use_standardized_residuals:
            # Intervals based on standardized residual distribution
            assert self._quantile is not None
            lower = point * (1 - self._quantile)
            upper = point * (1 + self._quantile)
        else:
            # Direct intervals
            assert self._quantile_lower is not None
            assert self._quantile_upper is not None
            lower = point - self._quantile_lower
            upper = point + self._quantile_upper

        lower = np.maximum(lower, 0)

        return VolatilityForecast(
            point=point,
            lower=lower,
            upper=upper,
            alpha=self.alpha,
            is_variance=False,
        )

    def rolling_forecast(
        self,
        returns: NDArray[np.floating[Any]],
        window_size: int = 500,
        horizon: int = 1,
        refit_frequency: int = 20,
    ) -> dict[str, Any]:
        """
        Rolling window volatility forecasting with conformal intervals.

        Args:
            returns: Full return series
            window_size: Estimation window size
            horizon: Forecast horizon
            refit_frequency: How often to refit GARCH

        Returns:
            Dictionary with forecasts and realized volatility
        """

        returns = np.asarray(returns).flatten()
        n = len(returns)

        if window_size >= n - 10:
            raise ValueError("window_size too large")

        forecasts = []
        realized = []
        coverages = []

        steps_since_refit = refit_frequency  # Force initial fit

        for t in range(window_size, n - horizon):
            window_returns = returns[t - window_size : t]

            # Refit if needed
            if steps_since_refit >= refit_frequency:
                self.fit(
                    window_returns,
                    calibration_fraction=0.2,
                )
                steps_since_refit = 0

            # Forecast
            forecast = self.predict(horizon=horizon)
            forecasts.append(forecast)

            # Realized volatility (using future returns)
            future_returns = returns[t : t + horizon]
            realized_vol = np.sqrt(np.mean(future_returns**2))
            realized.append(realized_vol)

            # Check coverage
            covered = forecast.contains(np.array([realized_vol]))[0]
            coverages.append(covered)

            steps_since_refit += 1

        return {
            "forecasts": forecasts,
            "realized": np.array(realized),
            "coverage": float(np.mean(coverages)),
            "mean_width": float(np.mean([f.width[0] for f in forecasts])),
        }


class HARConformal:
    """
    Conformal Prediction for HAR (Heterogeneous AutoRegressive) model.

    The HAR model captures volatility persistence at multiple frequencies:
    RV_t = β_0 + β_1 * RV_{t-1} + β_2 * RV_{t-5:t-1} + β_3 * RV_{t-22:t-1}

    Commonly used with realized volatility from high-frequency data.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        log_transform: bool = True,
    ) -> None:
        """
        Initialize HAR Conformal predictor.

        Args:
            alpha: Target miscoverage rate
            log_transform: Whether to model log(RV) for better normality
        """
        self.alpha = alpha
        self.log_transform = log_transform

        self._coefficients: NDArray[np.floating[Any]] | None = None
        self._quantile: float | None = None
        self._residual_std: float | None = None
        self._is_fitted = False

    def _compute_har_features(
        self,
        rv: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute HAR features (daily, weekly, monthly averages)."""
        n = len(rv)
        features = np.zeros((n - 22, 4))

        for t in range(22, n):
            features[t - 22, 0] = 1  # Intercept
            features[t - 22, 1] = rv[t - 1]  # Daily
            features[t - 22, 2] = np.mean(rv[t - 5 : t])  # Weekly
            features[t - 22, 3] = np.mean(rv[t - 22 : t])  # Monthly

        return features

    def fit(
        self,
        realized_volatility: NDArray[np.floating[Any]],
        calibration_fraction: float = 0.25,
    ) -> "HARConformal":
        """
        Fit HAR model and calibrate.

        Args:
            realized_volatility: Realized volatility series
            calibration_fraction: Fraction for calibration

        Returns:
            self
        """
        rv = np.asarray(realized_volatility).flatten()

        if self.log_transform:
            rv = np.log(rv + 1e-8)

        n = len(rv)
        n_cal = int((n - 22) * calibration_fraction)

        # Compute features
        X = self._compute_har_features(rv)
        y = rv[22:]

        # Split
        train_end = len(y) - n_cal
        X_train, X_cal = X[:train_end], X[train_end:]
        y_train, y_cal = y[:train_end], y[train_end:]

        # Fit OLS
        self._coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        # Calibration
        y_pred_cal = X_cal @ self._coefficients
        residuals = y_cal - y_pred_cal

        scores = np.abs(residuals)
        self._quantile = compute_conformal_quantile(scores, self.alpha)
        self._residual_std = float(np.std(residuals))

        self._is_fitted = True
        return self

    def predict(
        self,
        recent_rv: NDArray[np.floating[Any]],
    ) -> VolatilityForecast:
        """
        Predict volatility for next period.

        Args:
            recent_rv: Most recent 22 days of realized volatility

        Returns:
            VolatilityForecast
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit() before predict()")

        recent_rv = np.asarray(recent_rv).flatten()
        if len(recent_rv) < 22:
            raise ValueError("Need at least 22 observations")

        rv = recent_rv[-22:]
        if self.log_transform:
            rv = np.log(rv + 1e-8)

        # Compute features for prediction
        features = np.array(
            [
                1,
                rv[-1],
                np.mean(rv[-5:]),
                np.mean(rv),
            ]
        )

        assert self._coefficients is not None
        assert self._quantile is not None

        point = float(features @ self._coefficients)

        if self.log_transform:
            # Transform back
            point_rv = np.exp(point)
            lower_rv = np.exp(point - self._quantile)
            upper_rv = np.exp(point + self._quantile)
        else:
            point_rv = point
            lower_rv = max(0, point - self._quantile)
            upper_rv = point + self._quantile

        return VolatilityForecast(
            point=np.array([point_rv]),
            lower=np.array([lower_rv]),
            upper=np.array([upper_rv]),
            alpha=self.alpha,
            is_variance=False,
        )
