"""
Conformal Prediction for Alpha Research.

This module provides tools for quantitative alpha research:
- Signal confidence intervals (is this alpha signal real?)
- Uncertainty-aware position sizing (Kelly criterion with intervals)
- Strategy comparison with proper statistical guarantees
- Feature importance with coverage guarantees
- Regime-adaptive alpha models

Key insight: Point predictions of returns are useless without uncertainty.
A 50bps predicted return with ±10bps uncertainty is very different from
50bps ± 200bps. Conformal prediction provides the uncertainty rigorously.

Applications:
- Position sizing: size ∝ signal_strength / interval_width
- Signal filtering: only trade when interval doesn't contain zero
- Model selection: compare strategies by efficiency (coverage/width tradeoff)
- Risk management: worst-case returns from interval bounds
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    compute_conformal_quantile,
)
from quantcore.validation.stats import sharpe_ratio

# S14 imports (mondrian integration). Lazy / module-level import is
# fine because mondrian.py has no sibling-conformal imports beyond
# base.PredictionInterval (S13 P12.3 contract). regression.py is
# the natural source for the default base estimator factory.
from quantcore.uncertainty.conformal.mondrian import MondrianConformal
from quantcore.uncertainty.conformal.regression import SplitConformalRegressor

if TYPE_CHECKING:
    # Avoid eager import of quantile.py for non-cqr code paths;
    # CQRRegressor is lazy-imported inside _fit_cqr at runtime.
    from quantcore.uncertainty.conformal.quantile import CQRRegressor


class _FittablePredictor(Protocol):
    """Structural surface this module needs from sklearn-style predictors.

    sklearn's ``BaseEstimator`` stub doesn't expose ``.fit()`` /
    ``.predict()`` — those live on the mixin subtypes
    (``RegressorMixin``, etc.). This Protocol captures the actual
    surface alpha.py uses, so casts at ``clone()`` boundaries narrow
    the Union ``clone()`` returns (``dict | frozenset | list | set |
    tuple | BaseEstimator``) into a type basedpyright can resolve
    ``.fit()`` / ``.predict()`` on.
    """

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> Any: ...

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]: ...


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """
    Container for alpha signal predictions with uncertainty.

    Attributes:
        expected_return: Point forecast of expected return
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        signal_strength: |expected_return| / interval_width (like t-stat)
        tradeable: Whether signal is statistically significant
        position_weight: Suggested position weight based on uncertainty
    """

    expected_return: NDArray[np.floating[Any]]
    lower: NDArray[np.floating[Any]]
    upper: NDArray[np.floating[Any]]
    alpha: float

    @property
    def interval_width(self) -> NDArray[np.floating[Any]]:
        """Width of prediction interval."""
        return self.upper - self.lower

    @property
    def signal_strength(self) -> NDArray[np.floating[Any]]:
        """
        Signal strength: |E[r]| / (width/2).

        Analogous to t-statistic. Higher = more confident signal.
        """
        half_width = self.interval_width / 2
        return np.abs(self.expected_return) / (half_width + 1e-10)

    @property
    def tradeable(self) -> NDArray[np.bool_]:
        """
        Whether signal is tradeable (interval doesn't contain zero).

        If zero is in the interval, we can't reject the null of no alpha.
        """
        return (self.lower > 0) | (self.upper < 0)

    @property
    def direction(self) -> NDArray[np.int_]:
        """Signal direction: +1 (long), -1 (short), 0 (no trade)."""
        result = np.zeros(len(self.expected_return), dtype=np.int_)
        result[self.lower > 0] = 1  # Confidently positive
        result[self.upper < 0] = -1  # Confidently negative
        return result

    def kelly_weight(
        self,
        risk_aversion: float = 1.0,
        max_leverage: float = 2.0,
    ) -> NDArray[np.floating[Any]]:
        """
        Kelly criterion position weight with uncertainty adjustment.

        Standard Kelly: f = μ/σ² (fraction of capital to bet)
        With interval: we use conservative estimate and scale by confidence

        Args:
            risk_aversion: >1 for fractional Kelly (more conservative)
            max_leverage: Maximum allowed position size

        Returns:
            Position weights in [-max_leverage, max_leverage]
        """
        # Conservative return estimate: use bound closest to zero
        conservative_return = np.where(
            self.expected_return > 0,
            self.lower,  # For longs, use lower bound
            self.upper,  # For shorts, use upper bound
        )

        # Variance estimate from interval width (assuming ~normal)
        # 90% CI ≈ 1.645σ on each side for normal
        z = stats.norm.ppf(1 - self.alpha / 2)
        implied_std = self.interval_width / (2 * z)
        implied_var = implied_std**2 + 1e-10

        # Kelly weight
        raw_weight = conservative_return / (implied_var * risk_aversion)

        # Clip to max leverage
        return np.clip(raw_weight, -max_leverage, max_leverage)


@dataclass
class StrategyMetrics:
    """
    Comprehensive metrics for evaluating alpha strategies with conformal bounds.

    Beyond standard metrics (Sharpe, etc.), includes uncertainty-aware metrics.
    """

    # Standard metrics
    mean_return: float
    std_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Conformal metrics
    coverage: float
    mean_interval_width: float
    signal_strength_mean: float
    tradeable_fraction: float
    directional_accuracy: float

    # Risk-adjusted with uncertainty
    information_ratio: float | None = None
    calmar_ratio: float | None = None

    @property
    def efficiency(self) -> float:
        """Coverage / width tradeoff. Higher = more efficient intervals."""
        return self.coverage / (self.mean_interval_width + 1e-10)


class ConformalAlphaModel:
    """
    Conformal prediction wrapper for alpha models.

    Takes any return prediction model and adds rigorous uncertainty quantification.
    Essential for distinguishing real signals from noise.

    Example:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> model = GradientBoostingRegressor()
        >>> alpha_model = ConformalAlphaModel(model, alpha=0.1)
        >>> alpha_model.fit(X_train, returns_train)
        >>>
        >>> signals = alpha_model.predict(X_test)
        >>> # Only trade when confident
        >>> positions = signals.direction * signals.kelly_weight()
        >>> positions[~signals.tradeable] = 0
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        method: Literal["split", "cv", "cqr", "mondrian"] = "split",
        n_folds: int = 5,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        random_state: int | None = None,
        stratifier: Callable[[NDArray[np.floating[Any]]], NDArray[np.int_]] | None = None,
        mondrian_base_method: Literal["split"] = "split",
    ) -> None:
        """
        Initialize Conformal Alpha Model.

        Args:
            model: Base return prediction model
            alpha: Miscoverage rate (0.1 = 90% intervals)
            method: Conformal method
                - 'split': Split conformal (fast, uses less data)
                - 'cv': Cross-conformal (uses all data, K model fits)
                - 'cqr': Conformalized quantile regression (adaptive widths)
                - 'mondrian': Per-stratum split conformal (S13/S14)
            n_folds: Number of folds for CV method
            score_function: Custom nonconformity score
            random_state: Random seed
            stratifier: REQUIRED when method='mondrian'. Callable
                ``X -> integer labels of shape (n,)`` mapping rows
                to strata. Stratifier signature contract is
                enforced at the MondrianConformal level (S13 P12.3
                Pin 4); this class just forwards it.
            mondrian_base_method: Base conformal estimator to wrap
                each stratum. S14 ships ``"split"`` only;
                future enhancements may add ``"cqr"`` /
                ``"weighted"``. Single-value Literal forces the
                constraint at type-check time.
        """
        # Cast at the boundary: sklearn's BaseEstimator stub doesn't
        # expose .fit()/.predict(); _FittablePredictor narrows to the
        # surface this module actually uses. Runtime object is the
        # caller's BaseEstimator subclass, structurally compatible.
        # The cast is structurally sound (BaseEstimator subclasses have
        # .fit/.predict) but pyright can't prove it through the abstract
        # stub — hence the reportInvalidCast suppression with rationale.
        self.model: _FittablePredictor = cast(_FittablePredictor, model)  # pyright: ignore[reportInvalidCast]
        self.alpha = alpha
        # Preserve Literal narrowing at the field level so
        # `model.method` reads as Literal[...] (not bare str) at
        # downstream call sites such as backtest_alpha_model's
        # ConformalAlphaModel(method=model.method, ...) refit.
        self.method: Literal["split", "cv", "cqr", "mondrian"] = method
        self.n_folds = n_folds
        self.score_function = score_function or absolute_residual_score
        self.random_state = random_state
        self.stratifier = stratifier
        self.mondrian_base_method: Literal["split"] = mondrian_base_method

        self._rng = np.random.default_rng(random_state)
        self._quantile: float | None = None
        self._quantile_lo: float | None = None
        self._quantile_hi: float | None = None
        self._calibration_scores: NDArray[np.floating[Any]] | None = None
        self._mondrian_model: MondrianConformal | None = None
        self._cqr_model: CQRRegressor | None = None
        self._is_fitted = False

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float = 0.25,
        **fit_params: Any,
    ) -> "ConformalAlphaModel":
        """
        Fit the alpha model with conformal calibration.

        Args:
            X: Features (e.g., factor exposures, signals, market data)
            y: Target returns
            calibration_fraction: Fraction for calibration (split method)
            **fit_params: Additional params for model.fit()

        Returns:
            self
        """

        if self.method == "split":
            self._fit_split(X, y, calibration_fraction, **fit_params)

        elif self.method == "cv":
            self._fit_cv(X, y, **fit_params)

        elif self.method == "cqr":
            self._fit_cqr(X, y, calibration_fraction, **fit_params)

        elif self.method == "mondrian":
            self._fit_mondrian(X, y, **fit_params)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True
        return self

    def _fit_split(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float,
        **fit_params: Any,
    ) -> None:
        """Fit using split conformal."""
        n_samples = len(y)
        n_cal = int(n_samples * calibration_fraction)

        indices = self._rng.permutation(n_samples)
        train_idx = indices[n_cal:]
        cal_idx = indices[:n_cal]

        X_train, X_cal = X[train_idx], X[cal_idx]
        y_train, y_cal = y[train_idx], y[cal_idx]

        # Fit model
        self.model = cast(_FittablePredictor, clone(self.model))
        self.model.fit(X_train, y_train, **fit_params)

        # Calibrate
        y_pred_cal = self.model.predict(X_cal)
        self._calibration_scores = self.score_function(y_cal, y_pred_cal)
        self._quantile = compute_conformal_quantile(self._calibration_scores, self.alpha)

    def _fit_cv(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> None:
        """Fit using cross-conformal."""
        n_samples = len(y)
        scores = np.zeros(n_samples)

        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, val_idx in kf.split(X):
            fold_model = cast(_FittablePredictor, clone(self.model))
            fold_model.fit(X[train_idx], y[train_idx], **fit_params)

            y_pred_val = fold_model.predict(X[val_idx])
            scores[val_idx] = self.score_function(y[val_idx], y_pred_val)

        # Fit full model for prediction
        self.model = cast(_FittablePredictor, clone(self.model))
        self.model.fit(X, y, **fit_params)

        self._calibration_scores = scores
        self._quantile = compute_conformal_quantile(scores, self.alpha)

    def _fit_mondrian(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> None:
        """Fit using Mondrian-stratified split conformal.

        Composes ``MondrianConformal`` (S13 P12.3) — does NOT
        reinline the calibration math. Validates that the
        stratifier kwarg was provided; raises with the kwarg name
        if missing. Per-stratum base estimator is selected via
        ``mondrian_base_method`` (S14 ships ``"split"`` only).
        """
        if self.stratifier is None:
            raise ValueError(
                "ConformalAlphaModel(method='mondrian') requires "
                "the `stratifier` kwarg (callable: X -> integer "
                "labels). The stratifier signature contract is "
                "enforced at the MondrianConformal level "
                "(S13 P12.3); this class forwards it without "
                "validation. See MondrianConformal docstring for "
                "the class-side / caller-side contract direction."
            )
        if self.mondrian_base_method == "split":
            base_model = self.model
            base_alpha = self.alpha
            base_random_state = self.random_state

            def _factory() -> SplitConformalRegressor:
                return SplitConformalRegressor(
                    model=cast(BaseEstimator, clone(base_model)),
                    alpha=base_alpha,
                    random_state=base_random_state,
                )
        else:
            raise ValueError(
                f"Unknown mondrian_base_method: "
                f"{self.mondrian_base_method!r}. S14 ships 'split' "
                f"only; future sprints may add 'cqr' / 'weighted'."
            )
        self._mondrian_model = MondrianConformal(
            base_estimator_factory=_factory,
            stratifier=self.stratifier,
            alpha=self.alpha,
            empty_stratum_fallback="global",
        )
        self._mondrian_model.fit(X, y, **fit_params)

    def _fit_cqr(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        calibration_fraction: float,
        **fit_params: Any,
    ) -> None:
        """Fit using Conformalized Quantile Regression."""
        from quantcore.uncertainty.conformal.quantile import CQRRegressor

        # CQR requires quantile regression capability
        # Check if model supports it
        if not hasattr(self.model, "set_params"):
            raise ValueError(
                "CQR requires a model that supports quantile regression "
                "(e.g., GradientBoostingRegressor, LightGBM, XGBoost)"
            )

        cqr = CQRRegressor(
            # Inverse of __init__'s _FittablePredictor cast: hand the
            # underlying object back to a BaseEstimator-typed param.
            # Cast through Any avoids reportInvalidCast (the runtime
            # object is the caller's BaseEstimator subclass).
            model=cast(BaseEstimator, cast(Any, self.model)),
            alpha=self.alpha,
            random_state=self.random_state,
        )
        cqr.fit(X, y, calibration_fraction=calibration_fraction, **fit_params)

        # Store for prediction
        self._cqr_model = cqr
        self._quantile = cqr._quantile_correction

    def predict(
        self,
        X: NDArray[np.floating[Any]],
    ) -> AlphaSignal:
        """
        Generate alpha signals with uncertainty bounds.

        Args:
            X: Features for prediction

        Returns:
            AlphaSignal with expected returns and confidence intervals
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        if self.method == "cqr":
            assert self._cqr_model is not None, (
                "method='cqr' requires _fit_cqr to have run; reached "
                "predict() with _cqr_model unset"
            )
            interval = self._cqr_model.predict(X)
            # CQRRegressor.predict() always sets `point = (lower + upper) / 2`
            # before returning (quantile.py:335) — invariant pin so a future
            # regression that drops the midpoint computation is caught here.
            assert interval.point is not None, (
                "CQRRegressor.predict() must set interval.point — invariant violated"
            )
            return AlphaSignal(
                expected_return=interval.point,
                lower=interval.lower,
                upper=interval.upper,
                alpha=self.alpha,
            )

        if self.method == "mondrian":
            assert self._mondrian_model is not None
            interval, _diagnostic = self._mondrian_model.predict(X)
            # Mondrian intervals are symmetric around the global
            # base predictor's point estimate per stratum, but the
            # MondrianConformal API returns lower/upper without
            # an explicit point. Use the midpoint as the expected
            # return — consistent with how a symmetric split-
            # conformal interval is interpreted.
            expected = (interval.lower + interval.upper) / 2.0
            return AlphaSignal(
                expected_return=expected,
                lower=interval.lower,
                upper=interval.upper,
                alpha=self.alpha,
            )

        # Standard conformal methods (split, cv)
        assert self._quantile is not None
        y_pred = self.model.predict(X)

        return AlphaSignal(
            expected_return=y_pred,
            lower=y_pred - self._quantile,
            upper=y_pred + self._quantile,
            alpha=self.alpha,
        )

    def predict_proba_positive(
        self,
        X: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Estimate probability that true return is positive.

        Useful for classification-style trading decisions.

        Approximation assuming roughly symmetric intervals.
        """
        signal = self.predict(X)

        # P(r > 0) ≈ Φ((E[r] - 0) / σ) where σ from interval width
        z = stats.norm.ppf(1 - self.alpha / 2)
        implied_std = signal.interval_width / (2 * z)

        return stats.norm.cdf(signal.expected_return / (implied_std + 1e-10))


class SignalFilter:
    """
    Filter alpha signals based on conformal prediction confidence.

    Only passes signals that meet uncertainty criteria:
    - Interval doesn't contain zero (statistically significant)
    - Signal strength above threshold
    - Interval width below maximum

    This is the key to avoiding overtrading on noise.
    """

    def __init__(
        self,
        min_signal_strength: float = 1.0,
        max_interval_width: float | None = None,
        require_tradeable: bool = True,
    ) -> None:
        """
        Initialize signal filter.

        Args:
            min_signal_strength: Minimum |E[r]| / half_width ratio
            max_interval_width: Maximum allowed interval width
            require_tradeable: If True, filter out signals where 0 ∈ interval
        """
        self.min_signal_strength = min_signal_strength
        self.max_interval_width = max_interval_width
        self.require_tradeable = require_tradeable

    def filter(
        self,
        signal: AlphaSignal,
    ) -> NDArray[np.bool_]:
        """
        Return boolean mask of signals that pass filter.

        Args:
            signal: AlphaSignal from ConformalAlphaModel

        Returns:
            Boolean array - True where signal passes filter
        """
        mask = np.ones(len(signal.expected_return), dtype=bool)

        # Signal strength filter
        mask &= signal.signal_strength >= self.min_signal_strength

        # Interval width filter
        if self.max_interval_width is not None:
            mask &= signal.interval_width <= self.max_interval_width

        # Tradeable filter (zero not in interval)
        if self.require_tradeable:
            mask &= signal.tradeable

        return mask

    def apply(
        self,
        signal: AlphaSignal,
    ) -> tuple[AlphaSignal, NDArray[np.bool_]]:
        """
        Apply filter and return filtered signal with mask.

        Non-passing signals have their values set to zero.
        """
        mask = self.filter(signal)

        filtered = AlphaSignal(
            expected_return=np.where(mask, signal.expected_return, 0.0),
            lower=np.where(mask, signal.lower, 0.0),
            upper=np.where(mask, signal.upper, 0.0),
            alpha=signal.alpha,
        )

        return filtered, mask


class PortfolioConstructor:
    """
    Construct portfolios from alpha signals with uncertainty.

    Implements several approaches:
    1. Equal weight filtered signals
    2. Signal-strength weighted
    3. Kelly criterion weighted
    4. Risk parity with uncertainty
    """

    def __init__(
        self,
        method: Literal["equal", "signal", "kelly", "risk_parity"] = "kelly",
        risk_aversion: float = 2.0,
        max_position: float = 0.1,
        max_leverage: float = 1.0,
    ) -> None:
        """
        Initialize portfolio constructor.

        Args:
            method: Weighting method
            risk_aversion: Kelly risk aversion (>1 for fractional Kelly)
            max_position: Maximum single position size
            max_leverage: Maximum total leverage
        """
        self.method = method
        self.risk_aversion = risk_aversion
        self.max_position = max_position
        self.max_leverage = max_leverage

    def construct(
        self,
        signal: AlphaSignal,
        filter_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """
        Construct portfolio weights from signals.

        Args:
            signal: AlphaSignal from ConformalAlphaModel
            filter_mask: Optional filter mask (from SignalFilter)

        Returns:
            Portfolio weights (sum of abs = leverage)
        """
        n = len(signal.expected_return)

        if filter_mask is None:
            filter_mask = np.ones(n, dtype=bool)

        # Start with zeros
        weights = np.zeros(n)

        if not np.any(filter_mask):
            return weights

        if self.method == "equal":
            # Equal weight long/short
            directions = signal.direction[filter_mask]
            n_trades = np.sum(np.abs(directions))
            if n_trades > 0:
                weights[filter_mask] = directions / n_trades

        elif self.method == "signal":
            # Weight by expected return magnitude
            returns = signal.expected_return[filter_mask]
            directions = signal.direction[filter_mask]
            abs_returns = np.abs(returns)

            if np.sum(abs_returns) > 0:
                raw_weights = directions * abs_returns
                weights[filter_mask] = raw_weights / np.sum(np.abs(raw_weights))

        elif self.method == "kelly":
            # Kelly criterion with uncertainty adjustment
            kelly = signal.kelly_weight(
                risk_aversion=self.risk_aversion,
                max_leverage=self.max_position,
            )
            weights[filter_mask] = kelly[filter_mask]

            # Normalize to max leverage
            total_leverage = np.sum(np.abs(weights))
            if total_leverage > self.max_leverage:
                weights *= self.max_leverage / total_leverage

        elif self.method == "risk_parity":
            # Inverse volatility weighting (from interval width)
            widths = signal.interval_width[filter_mask]
            directions = signal.direction[filter_mask]

            inv_vol = 1.0 / (widths + 1e-10)
            raw_weights = directions * inv_vol

            if np.sum(np.abs(raw_weights)) > 0:
                weights[filter_mask] = raw_weights / np.sum(np.abs(raw_weights))

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply position limits
        weights = np.clip(weights, -self.max_position, self.max_position)

        # Apply leverage constraint
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > self.max_leverage:
            weights *= self.max_leverage / total_leverage

        return weights


def compute_strategy_metrics(
    signals: AlphaSignal,
    returns_realized: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]] | None = None,
) -> StrategyMetrics:
    """
    Compute comprehensive strategy metrics.

    Args:
        signals: Alpha signals with uncertainty
        returns_realized: Realized returns
        weights: Portfolio weights (if None, uses signal direction)

    Returns:
        StrategyMetrics object
    """

    if weights is None:
        weights = signals.direction.astype(float)
        weights /= np.sum(np.abs(weights)) + 1e-10
    # Narrowing pin: the if-block above guarantees weights is non-None;
    # the augmented `/=` confuses pyright's narrowing through Optional.
    assert weights is not None

    # Portfolio returns
    portfolio_returns = weights * returns_realized

    # Standard metrics
    mean_return = float(np.mean(portfolio_returns))
    std_return = float(np.std(portfolio_returns))
    # F-RP-004a: route through the F08-gated `sharpe_ratio` instead of
    # the pre-fix inline `mean / (std + 1e-10) * sqrt(252)`. Same
    # option-B discipline as F-RP-002 — warn+NaN on F08 trigger
    # rather than raise, so StrategyMetrics keeps its schema validity
    # for downstream callers under degenerate-variance fixtures.
    # Strict callers opt in via `simplefilter("error", UserWarning)`.
    # Local renamed `sharpe_ratio` → `sharpe_value` to avoid shadowing
    # the imported module-level `sharpe_ratio` function from F-RP-002;
    # constructor below threads `sharpe_value` into the dataclass's
    # `sharpe_ratio` field.
    try:
        sharpe_value = float(sharpe_ratio(portfolio_returns, periods_per_year=252))
    except ValueError as exc:
        warnings.warn(
            f"compute_strategy_metrics: F08 degenerate-variance gate "
            f"fired on portfolio_returns; returning NaN Sharpe. "
            f"Reason: {exc}",
            UserWarning,
            stacklevel=2,
        )
        sharpe_value = float("nan")

    # Drawdown
    cumulative = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = float(np.max(drawdown))

    # Conformal metrics
    covered = signals.lower <= returns_realized
    covered &= returns_realized <= signals.upper
    coverage = float(np.mean(covered))

    mean_width = float(np.mean(signals.interval_width))
    signal_strength_mean = float(np.mean(signals.signal_strength))
    tradeable_fraction = float(np.mean(signals.tradeable))

    # Directional accuracy
    predicted_direction = np.sign(signals.expected_return)
    actual_direction = np.sign(returns_realized)
    directional_accuracy = float(np.mean(predicted_direction == actual_direction))

    return StrategyMetrics(
        mean_return=mean_return,
        std_return=std_return,
        sharpe_ratio=sharpe_value,
        max_drawdown=max_drawdown,
        coverage=coverage,
        mean_interval_width=mean_width,
        signal_strength_mean=signal_strength_mean,
        tradeable_fraction=tradeable_fraction,
        directional_accuracy=directional_accuracy,
    )


class FeatureImportanceConformal:
    """
    Feature importance with conformal prediction guarantees.

    Standard feature importance (permutation, SHAP) gives point estimates.
    This adds uncertainty quantification: "Feature X importance is 0.15 ± 0.03"

    Useful for:
    - Distinguishing truly important features from noise
    - Stability analysis of feature rankings
    - Comparing models on feature utilization
    """

    def __init__(
        self,
        model: BaseEstimator,
        alpha: float = 0.1,
        n_permutations: int = 100,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize feature importance calculator.

        Args:
            model: Fitted model
            alpha: Miscoverage rate for importance intervals
            n_permutations: Number of permutations per feature
            random_state: Random seed
        """
        # Same boundary cast as ConformalAlphaModel.__init__ — see that
        # class's docstring for the _FittablePredictor rationale.
        self.model: _FittablePredictor = cast(_FittablePredictor, model)  # pyright: ignore[reportInvalidCast]
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def compute(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        metric: Callable[[NDArray[Any], NDArray[Any]], float] | None = None,
    ) -> dict[str, Any]:
        """
        Compute feature importance with conformal intervals.

        Args:
            X: Features
            y: Targets
            metric: Scoring function (default: negative MSE)

        Returns:
            Dictionary with importance estimates and intervals
        """
        if metric is None:

            def metric(y_true, y_pred):
                return -np.mean((y_true - y_pred) ** 2)

        n_features = X.shape[1]

        # Baseline score
        y_pred_base = self.model.predict(X)
        base_score = metric(y, y_pred_base)

        # Compute importance for each feature
        importances = []
        importance_lower = []
        importance_upper = []

        for j in range(n_features):
            # Permutation scores
            perm_scores = []

            for _ in range(self.n_permutations):
                X_perm = X.copy()
                X_perm[:, j] = self._rng.permutation(X_perm[:, j])

                y_pred_perm = self.model.predict(X_perm)
                perm_score = metric(y, y_pred_perm)
                perm_scores.append(base_score - perm_score)

            perm_scores = np.array(perm_scores)

            # Point estimate
            importance = float(np.mean(perm_scores))

            # Conformal interval using bootstrap scores as calibration
            quantile = compute_conformal_quantile(np.abs(perm_scores - importance), self.alpha)

            importances.append(importance)
            importance_lower.append(importance - quantile)
            importance_upper.append(importance + quantile)

        return {
            "importances": np.array(importances),
            "lower": np.array(importance_lower),
            "upper": np.array(importance_upper),
            "significant": np.array(importance_lower) > 0,  # Lower bound > 0
            "base_score": base_score,
            "alpha": self.alpha,
        }


def backtest_alpha_model(
    model: ConformalAlphaModel,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    initial_train_size: int = 252,
    refit_frequency: int = 21,
    signal_filter: SignalFilter | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
    *,
    t1: NDArray[np.integer[Any]] | None = None,
    embargo: int = 0,
) -> dict[str, Any]:
    """
    Walk-forward backtest of conformal alpha model.

    Simulates realistic deployment:
    1. Train on historical window
    2. Generate signals for next period
    3. Trade based on filtered signals
    4. Observe realized returns
    5. Update calibration

    Args:
        model: ConformalAlphaModel (will be cloned)
        X: Full feature matrix
        y: Full return series
        initial_train_size: Initial training window
        refit_frequency: Days between model refits
        signal_filter: Optional signal filter
        portfolio_constructor: Optional portfolio constructor
        t1: Optional integer barrier-close timestep per event. When the
            label scheme is triple-barrier with horizon ``h``, set
            ``t1[i] = i + h - 1``. Defaults to ``arange(len(y))`` (each
            event closes at its own row), preserving pre-F-RP-001
            behavior at ``embargo=0``.
        embargo: Number of bars after the prediction time ``t`` whose
            barrier-close events must be excluded from training. At each
            refit, training is restricted to events whose
            ``t1 ≤ t - embargo - 1``. For triple-barrier labels with
            vertical horizon ``h``, the recommended setting is
            ``embargo = h + 1`` (h for label horizon + 1 for the
            barrier-close-to-prediction-input gap). The default
            ``embargo=0`` reproduces pre-F-RP-001 semantics. AFML §7.4.

    Returns:
        Backtest results dictionary

    Notes:
        F-RP-001: when ``t1`` is provided AND varies (i.e., triple-
        barrier-style labels are detected) AND ``embargo=0``, a
        ``UserWarning`` fires recommending the caller supply
        ``embargo = h + 1`` to close the AFML §7.4 leakage path.
    """
    if signal_filter is None:
        signal_filter = SignalFilter(min_signal_strength=0.5)

    if portfolio_constructor is None:
        portfolio_constructor = PortfolioConstructor(method="kelly")

    n = len(y)

    # Resolve the effective t1 series: caller-supplied or the trivial
    # "each event closes at its own row" default. The default keeps the
    # postcondition `t1 ≤ t - embargo - 1` reducing to `i ≤ t - 1` at
    # embargo=0, so passing nothing matches pre-F-RP-001 behavior.
    if t1 is None:
        t1_eff: NDArray[np.integer[Any]] = np.arange(n, dtype=np.int64)
    else:
        t1_eff = np.asarray(t1, dtype=np.int64)
        if t1_eff.shape != (n,):
            raise ValueError(f"t1 must have shape ({n},) matching y; got {t1_eff.shape}")
        # Detection: triple-barrier-style labels are indicated by a
        # varying t1. With embargo=0 the leakage path is wide open.
        if embargo == 0 and int(t1_eff.max()) > int(t1_eff.min()):
            warnings.warn(
                "backtest_alpha_model: t1 varies (triple-barrier-style "
                "labels detected) but embargo=0 — training labels can "
                "overlap the prediction-time bar. Set embargo = h + 1 "
                "(label horizon + barrier-close-to-prediction gap) to "
                "close the AFML §7.4 leakage path. See F-RP-001.",
                UserWarning,
                stacklevel=2,
            )

    # Heterogeneous container — keys hold list[AlphaSignal], list[bool],
    # list[float], NDArray[Any], np.floating, and float across the
    # accumulation + post-loop aggregation paths. dict[str, Any] is
    # the honest type; tightening to per-key TypedDict would require
    # an architectural change beyond S17 scope.
    results: dict[str, Any] = {
        "signals": [],
        "weights": [],
        "returns": [],
        "covered": [],
        "trade_mask": [],
    }

    t = initial_train_size
    steps_since_refit = 0
    fitted_model = None

    while t < n:
        # Refit if needed
        if fitted_model is None or steps_since_refit >= refit_frequency:
            # Embargo gate: keep events whose barrier-close has resolved
            # at or before ``t - embargo - 1``. This is the AFML §7.4
            # purging postcondition rendered into a per-refit filter.
            train_idx = np.flatnonzero(t1_eff <= t - embargo - 1)
            fresh_base = cast(BaseEstimator, clone(model.model))
            fitted_model = ConformalAlphaModel(
                model=fresh_base,
                alpha=model.alpha,
                method=model.method,
                n_folds=model.n_folds,
                score_function=model.score_function,
                random_state=model.random_state,
                stratifier=model.stratifier,
                mondrian_base_method=model.mondrian_base_method,
            )
            fitted_model.fit(X[train_idx], y[train_idx])
            steps_since_refit = 0

        # Generate signal for next period
        signal = fitted_model.predict(X[t : t + 1])

        # Filter and construct portfolio
        filtered_signal, mask = signal_filter.apply(signal)
        weights = portfolio_constructor.construct(filtered_signal, mask)

        # Record
        results["signals"].append(signal)
        results["weights"].append(weights[0] if len(weights) > 0 else 0)
        results["returns"].append(y[t])
        results["covered"].append(signal.lower[0] <= y[t] <= signal.upper[0])
        results["trade_mask"].append(mask[0] if len(mask) > 0 else False)

        t += 1
        steps_since_refit += 1

    # Compute aggregate metrics
    weights = np.array(results["weights"])
    returns = np.array(results["returns"])
    portfolio_returns = weights * returns

    results["portfolio_returns"] = portfolio_returns
    results["cumulative_return"] = np.cumsum(portfolio_returns)
    results["coverage"] = np.mean(results["covered"])
    results["trade_rate"] = np.mean(results["trade_mask"])
    # F-RP-002: route through the F08-gated `sharpe_ratio` instead of
    # the pre-fix inline `mean / (std + 1e-10) * sqrt(252)`. The +1e-10
    # path silently softened degenerate-variance PnL into a finite-but-
    # huge Sharpe; the F08 gate in
    # `validation.stats._assert_non_degenerate` exists to surface that.
    # On F08 trigger we emit a `UserWarning` and return NaN — loud-via-
    # warning rather than loud-via-raise so canonical regime-shift
    # fixture pins in `test_empirical_comparison` (where some branches
    # are *expected* to be degenerate) stay green. Strict callers opt
    # in via `warnings.simplefilter("error", UserWarning)`.
    try:
        results["sharpe"] = float(sharpe_ratio(portfolio_returns, periods_per_year=252))
    except ValueError as exc:
        warnings.warn(
            f"backtest_alpha_model: F08 degenerate-variance gate fired on "
            f"portfolio_returns; returning NaN Sharpe. Reason: {exc}",
            UserWarning,
            stacklevel=2,
        )
        results["sharpe"] = float("nan")

    return results
