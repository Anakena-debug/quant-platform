"""
Conformal Prediction Library.

A production-grade implementation of conformal prediction methods for
uncertainty quantification in machine learning, with special focus on
applications in quantitative finance.

Core Features:
- Split, Cross-Conformal, Jackknife+, CV+ for regression
- LAC, APS, RAPS for classification
- Conformalized Quantile Regression (CQR) for adaptive intervals
- Time series methods (ACI, rolling, weighted)
- Financial applications (VaR, volatility forecasting)

Example:
    >>> from quantcore.uncertainty.conformal import SplitConformalRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> model = RandomForestRegressor(n_estimators=100)
    >>> cp = SplitConformalRegressor(model, alpha=0.1)
    >>> cp.fit(X_train, y_train)
    >>> intervals = cp.predict(X_test)
    >>> print(f"Coverage: {intervals.coverage(y_test):.2%}")
    >>> print(f"Mean width: {intervals.mean_width:.4f}")

For financial applications:
    >>> from quantcore.uncertainty.conformal.finance import ConformalVaR, GARCHConformal
    >>> var_model = ConformalVaR(alpha=0.95)
    >>> var_model.fit(returns)
    >>> var = var_model.predict()
"""

__version__ = "0.1.0"
__author__ = "Conformal Prediction Library Contributors"

from quantcore.uncertainty.conformal.base import (
    BaseConformalPredictor,
    BaseRegressionConformal,
    BaseClassificationConformal,
    CalibrationResult,
    ConformalConfig,
    PredictionInterval,
    PredictionSet,
    PredictionType,
)
from quantcore.uncertainty.conformal.scores import (
    ScoreType,
    absolute_residual_score,
    normalized_residual_score,
    quantile_score,
    compute_conformal_quantile,
    get_score_function,
)
from quantcore.uncertainty.conformal.regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifePlusRegressor,
    CVPlusRegressor,
)
from quantcore.uncertainty.conformal.quantile import (
    CQRRegressor,
    CQRPlusRegressor,
    QuantileRegressorWrapper,
)
from quantcore.uncertainty.conformal.classification import (
    LACClassifier,
    APSClassifier,
    RAPSClassifier,
    TopKConformalClassifier,
)
from quantcore.uncertainty.conformal.timeseries import (
    AdaptiveConformalInference,
    RollingConformalRegressor,
    WeightedConformalRegressor,
    expanding_window_backtest,
)
from quantcore.uncertainty.conformal.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    compute_regression_metrics,
    compute_classification_metrics,
    conditional_coverage,
    worst_slab_coverage,
    winkler_score,
    interval_score,
)

__all__ = [
    # Version
    "__version__",
    # Base classes
    "BaseConformalPredictor",
    "BaseRegressionConformal",
    "BaseClassificationConformal",
    "CalibrationResult",
    "ConformalConfig",
    "PredictionInterval",
    "PredictionSet",
    "PredictionType",
    # Scores
    "ScoreType",
    "absolute_residual_score",
    "normalized_residual_score",
    "quantile_score",
    "compute_conformal_quantile",
    "get_score_function",
    # Regression
    "SplitConformalRegressor",
    "CrossConformalRegressor",
    "JackknifePlusRegressor",
    "CVPlusRegressor",
    # Quantile
    "CQRRegressor",
    "CQRPlusRegressor",
    "QuantileRegressorWrapper",
    # Classification
    "LACClassifier",
    "APSClassifier",
    "RAPSClassifier",
    "TopKConformalClassifier",
    # Time series
    "AdaptiveConformalInference",
    "RollingConformalRegressor",
    "WeightedConformalRegressor",
    "expanding_window_backtest",
    # Metrics
    "RegressionMetrics",
    "ClassificationMetrics",
    "compute_regression_metrics",
    "compute_classification_metrics",
    "conditional_coverage",
    "worst_slab_coverage",
    "winkler_score",
    "interval_score",
]
