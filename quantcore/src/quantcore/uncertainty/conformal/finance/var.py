"""
Conformal Prediction for Value at Risk (VaR) and Expected Shortfall (ES).

This module provides distribution-free VaR and ES estimates with coverage guarantees.
Unlike parametric approaches (Historical, GARCH, etc.), conformal VaR guarantees
the stated coverage level in finite samples without distributional assumptions.

Key Benefits:
- Distribution-free: No assumption of normality or specific fat-tail distribution
- Finite-sample valid: Coverage guarantees hold for any sample size
- Adaptive: Can incorporate predictive features for conditional VaR
- Backtestable: Coverage can be verified with standard tests

References:
    Kuchibhotla, Kolassa (2020) "Conformal Prediction for Reliable Machine Learning"
    Romano, Patterson, Candès (2019) "Conformalized Quantile Regression"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True, slots=True)
class VaRResult:
    """
    Container for VaR estimates.

    Attributes:
        var: Value at Risk estimate(s)
        alpha: Confidence level (e.g., 0.95 for 95% VaR)
        es: Expected Shortfall estimate(s), if computed
        method: Method used for estimation
    """

    var: NDArray[np.floating[Any]]
    alpha: float
    es: NDArray[np.floating[Any]] | None = None
    method: str = "conformal"

    @property
    def var_scalar(self) -> float:
        """VaR as scalar (for single estimates)."""
        return (
            float(self.var[0])
            if self.var.ndim == 0 or len(self.var) == 1
            else float(self.var.mean())
        )


@dataclass
class VaRBacktestResult:
    """
    Results from VaR backtesting.

    Attributes:
        violations: Boolean array of VaR violations
        violation_rate: Empirical violation rate
        expected_rate: Expected violation rate (1 - alpha)
        kupiec_stat: Kupiec test statistic
        kupiec_pvalue: Kupiec test p-value
        christoffersen_stat: Independence test statistic
        christoffersen_pvalue: Independence test p-value
        dq_stat: Dynamic quantile test statistic
        dq_pvalue: Dynamic quantile test p-value
    """

    violations: NDArray[np.bool_]
    violation_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float | None = None
    christoffersen_pvalue: float | None = None
    dq_stat: float | None = None
    dq_pvalue: float | None = None

    @property
    def passes_kupiec(self) -> bool:
        """Whether VaR passes Kupiec unconditional coverage test at 5%."""
        return self.kupiec_pvalue > 0.05

    @property
    def is_valid(self) -> bool:
        """Whether VaR estimate appears valid (passes basic tests)."""
        return self.passes_kupiec and abs(self.violation_rate - self.expected_rate) < 0.02


class ConformalVaR:
    """
    Conformal Prediction-based Value at Risk.

    Produces VaR estimates with finite-sample coverage guarantees.
    Can be used unconditionally (historical VaR with correction) or
    conditionally (using features like volatility forecasts).

    Example:
        >>> var_model = ConformalVaR(alpha=0.95)
        >>> var_model.fit(returns_train)
        >>> var_estimate = var_model.predict()
        >>> print(f"95% VaR: {var_estimate.var_scalar:.4f}")

    For conditional VaR with features:
        >>> var_model.fit_conditional(X_train, returns_train, model=GradientBoostingRegressor())
        >>> var_estimate = var_model.predict_conditional(X_test)
    """

    def __init__(
        self,
        alpha: float = 0.95,
        window_size: int | None = None,
        method: Literal["historical", "cqr", "adaptive"] = "historical",
    ) -> None:
        """
        Initialize Conformal VaR estimator.

        Args:
            alpha: Confidence level (0.95 = 95% VaR)
            window_size: Rolling window size (None = use all data)
            method: VaR estimation method
                - 'historical': Conformal-corrected historical simulation
                - 'cqr': Conformalized Quantile Regression (requires features)
                - 'adaptive': Adaptive conformal for non-stationary returns
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.window_size = window_size
        self.method = method

        self._returns: NDArray[np.floating[Any]] | None = None
        self._quantile: float | None = None
        self._model: Any = None
        self._is_fitted = False

    def _compute_conformal_quantile_level(self, n: int) -> float:
        """
        Compute conformal-corrected quantile level.

        The correction factor (n+1)/n ensures finite-sample validity.

        For VaR at confidence level alpha (e.g., 0.95):
        - We want P(Loss > VaR) <= 1-alpha (e.g., 0.05)
        - So we need the alpha-quantile of losses (95th percentile)
        """
        return np.ceil((n + 1) * self.alpha) / n

    def fit(
        self,
        returns: NDArray[np.floating[Any]],
    ) -> "ConformalVaR":
        """
        Fit unconditional conformal VaR.

        Args:
            returns: Historical returns (can be negative for losses)

        Returns:
            self
        """
        self._returns = np.asarray(returns).flatten()
        n = len(self._returns)

        if n < 20:
            raise ValueError(f"Need at least 20 observations, got {n}")

        # Use most recent window if specified
        if self.window_size is not None:
            self._returns = self._returns[-self.window_size :]
            n = len(self._returns)

        # Compute conformal-corrected VaR
        # VaR is typically defined for losses (positive = bad)
        # We compute the (1-alpha) quantile of negative returns (losses)
        losses = -self._returns

        quantile_level = self._compute_conformal_quantile_level(n)
        quantile_level = np.clip(quantile_level, 0, 1)

        self._quantile = float(np.quantile(losses, quantile_level))
        self._is_fitted = True

        return self

    def fit_conditional(
        self,
        X: NDArray[np.floating[Any]],
        returns: NDArray[np.floating[Any]],
        model: Any,
        calibration_fraction: float = 0.25,
    ) -> "ConformalVaR":
        """
        Fit conditional conformal VaR using CQR.

        Uses Conformalized Quantile Regression to produce conditional
        VaR estimates that depend on predictive features.

        Args:
            X: Features (e.g., volatility forecast, market conditions)
            returns: Historical returns
            model: Quantile regression model (must support quantile loss)
            calibration_fraction: Fraction of data for calibration

        Returns:
            self
        """
        from quantcore.uncertainty.conformal.quantile import QuantileRegressorWrapper

        returns = np.asarray(returns).flatten()
        losses = -returns
        n = len(losses)

        # Split data
        n_cal = int(n * calibration_fraction)
        indices = np.random.permutation(n)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, X_cal = X[train_idx], X[cal_idx]
        losses_train, losses_cal = losses[train_idx], losses[cal_idx]

        # Fit quantile regressor for (1-alpha) quantile (upper tail of losses)
        self._model = QuantileRegressorWrapper(
            model,
            quantile=1 - self.alpha,
            model_type="sklearn",
        )
        self._model.fit(X_train, losses_train)

        # Calibrate
        losses_pred_cal = self._model.predict(X_cal)
        scores = losses_cal - losses_pred_cal  # Residuals

        quantile_level = self._compute_conformal_quantile_level(len(scores))
        self._quantile = float(np.quantile(scores, quantile_level))

        self._returns = returns
        self._is_fitted = True

        return self

    def predict(self) -> VaRResult:
        """
        Predict unconditional VaR.

        Returns:
            VaRResult with VaR estimate
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        if self._quantile is None:
            raise RuntimeError("Quantile not computed")

        return VaRResult(
            var=np.array([self._quantile]),
            alpha=self.alpha,
            method=f"conformal_{self.method}",
        )

    def predict_conditional(
        self,
        X: NDArray[np.floating[Any]],
    ) -> VaRResult:
        """
        Predict conditional VaR.

        Args:
            X: Features for prediction

        Returns:
            VaRResult with conditional VaR estimates
        """
        if self._model is None:
            raise RuntimeError("Use fit_conditional() before predict_conditional()")

        if self._quantile is None:
            raise RuntimeError("Quantile not computed")

        var_base = self._model.predict(X)
        var = var_base + self._quantile

        return VaRResult(
            var=var,
            alpha=self.alpha,
            method="conformal_cqr",
        )


class ConformalES:
    """
    Conformal Prediction-based Expected Shortfall (CVaR).

    Expected Shortfall is the expected loss given that the loss exceeds VaR.
    This implementation provides distribution-free ES estimates.

    ES = E[Loss | Loss > VaR]

    Note: Unlike VaR, ES is harder to estimate with conformal prediction.
    This implementation uses a plug-in approach with the conformal VaR.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        window_size: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self._var_model = ConformalVaR(alpha=alpha, window_size=window_size)
        self._returns: NDArray[np.floating[Any]] | None = None
        self._es: float | None = None
        self._is_fitted = False

    def fit(
        self,
        returns: NDArray[np.floating[Any]],
    ) -> "ConformalES":
        """
        Fit ES model.

        Args:
            returns: Historical returns

        Returns:
            self
        """
        self._returns = np.asarray(returns).flatten()
        losses = -self._returns

        # First get VaR
        self._var_model.fit(returns)
        var_result = self._var_model.predict()
        var = var_result.var_scalar

        # ES is the mean of losses exceeding VaR
        tail_losses = losses[losses >= var]

        if len(tail_losses) < 5:
            # Not enough tail observations, use conservative estimate
            self._es = float(np.max(losses))
        else:
            # Simple plug-in estimator
            self._es = float(np.mean(tail_losses))

        self._is_fitted = True
        return self

    def predict(self) -> VaRResult:
        """
        Predict ES.

        Returns:
            VaRResult with both VaR and ES
        """
        if not self._is_fitted or self._es is None:
            raise RuntimeError("Model must be fitted before predict()")

        var_result = self._var_model.predict()

        return VaRResult(
            var=var_result.var,
            alpha=self.alpha,
            es=np.array([self._es]),
            method="conformal_es",
        )


def kupiec_test(
    violations: NDArray[np.bool_],
    alpha: float,
) -> tuple[float, float]:
    """
    Kupiec's Proportion of Failures (POF) test.

    Tests whether the violation rate is consistent with the expected rate.
    H0: Violation rate equals expected rate (1 - alpha)

    Args:
        violations: Boolean array of VaR violations
        alpha: VaR confidence level

    Returns:
        Tuple of (test statistic, p-value)
    """
    n = len(violations)
    x = np.sum(violations)  # Number of violations
    p = 1 - alpha  # Expected violation rate

    if x == 0 or x == n:
        # Edge cases: use conservative estimate
        return 0.0, 1.0

    # Likelihood ratio statistic
    p_hat = x / n
    lr = -2 * (x * np.log(p / p_hat) + (n - x) * np.log((1 - p) / (1 - p_hat)))

    # Under H0, LR ~ chi-squared(1)
    pvalue = 1 - stats.chi2.cdf(lr, df=1)

    return float(lr), float(pvalue)


def christoffersen_test(
    violations: NDArray[np.bool_],
) -> tuple[float, float]:
    """
    Christoffersen's independence test.

    Tests whether violations are independent (no clustering).
    H0: Violations are independent

    Args:
        violations: Boolean array of VaR violations

    Returns:
        Tuple of (test statistic, p-value)
    """
    n = len(violations)

    # Count transitions
    n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
    n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
    n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
    n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))

    # Transition probabilities
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0, 1.0

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n - 1)  # Unconditional probability

    if p == 0 or p == 1 or p01 == 0 or p01 == 1 or p11 == 0 or p11 == 1:
        return 0.0, 1.0

    # Likelihood ratio
    ll_unrestricted = (
        n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
    )
    ll_restricted = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)

    lr = -2 * (ll_restricted - ll_unrestricted)
    pvalue = 1 - stats.chi2.cdf(lr, df=1)

    return float(lr), float(pvalue)


def backtest_var(
    var_estimates: NDArray[np.floating[Any]],
    returns: NDArray[np.floating[Any]],
    alpha: float = 0.95,
) -> VaRBacktestResult:
    """
    Comprehensive VaR backtesting.

    Performs multiple tests to assess VaR model validity:
    - Kupiec test (unconditional coverage)
    - Christoffersen test (independence)

    Args:
        var_estimates: VaR estimates (one per return)
        returns: Realized returns
        alpha: VaR confidence level

    Returns:
        VaRBacktestResult with test results
    """
    returns = np.asarray(returns).flatten()
    var_estimates = np.asarray(var_estimates).flatten()

    if len(returns) != len(var_estimates):
        raise ValueError("returns and var_estimates must have same length")

    # Compute violations (loss exceeds VaR)
    losses = -returns
    violations = losses > var_estimates

    violation_rate = float(np.mean(violations))
    expected_rate = 1 - alpha

    # Kupiec test
    kupiec_stat, kupiec_pvalue = kupiec_test(violations, alpha)

    # Christoffersen test
    if len(violations) > 10:
        chris_stat, chris_pvalue = christoffersen_test(violations)
    else:
        chris_stat, chris_pvalue = None, None

    return VaRBacktestResult(
        violations=violations,
        violation_rate=violation_rate,
        expected_rate=expected_rate,
        kupiec_stat=kupiec_stat,
        kupiec_pvalue=kupiec_pvalue,
        christoffersen_stat=chris_stat,
        christoffersen_pvalue=chris_pvalue,
    )


def rolling_var_backtest(
    returns: NDArray[np.floating[Any]],
    alpha: float = 0.95,
    window_size: int = 250,
    method: Literal["conformal", "historical", "parametric"] = "conformal",
) -> dict[str, Any]:
    """
    Rolling window VaR backtesting simulation.

    Simulates real-world VaR deployment:
    1. Estimate VaR using historical window
    2. Observe next-day return
    3. Record violation
    4. Roll window forward

    Args:
        returns: Full return series
        alpha: VaR confidence level
        window_size: Rolling estimation window
        method: Estimation method

    Returns:
        Dictionary with backtest results
    """
    returns = np.asarray(returns).flatten()
    n = len(returns)

    if window_size >= n - 10:
        raise ValueError("window_size too large for available data")

    var_estimates = []
    actual_returns = []
    violations = []

    for t in range(window_size, n):
        # Estimate VaR using past window
        window_returns = returns[t - window_size : t]

        if method == "conformal":
            var_model = ConformalVaR(alpha=alpha)
            var_model.fit(window_returns)
            var = var_model.predict().var_scalar
        elif method == "historical":
            # Simple historical simulation (no conformal correction)
            losses = -window_returns
            var = float(np.quantile(losses, 1 - alpha))
        elif method == "parametric":
            # Gaussian VaR
            mu = np.mean(window_returns)
            sigma = np.std(window_returns)
            var = -mu + sigma * stats.norm.ppf(1 - alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

        var_estimates.append(var)
        actual_returns.append(returns[t])
        violations.append(-returns[t] > var)

    var_estimates = np.array(var_estimates)
    actual_returns = np.array(actual_returns)

    backtest_result = backtest_var(var_estimates, actual_returns, alpha)

    return {
        "var_estimates": var_estimates,
        "returns": actual_returns,
        "backtest": backtest_result,
        "method": method,
        "window_size": window_size,
        "alpha": alpha,
    }
