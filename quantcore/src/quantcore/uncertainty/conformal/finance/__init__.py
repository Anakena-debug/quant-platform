"""
Financial applications of conformal prediction.

This subpackage provides specialized conformal prediction methods for
quantitative finance applications:
- Value at Risk (VaR) with coverage guarantees
- Volatility prediction with adaptive intervals
- Alpha research with signal uncertainty quantification
- Portfolio construction with uncertainty-aware position sizing
"""

from quantcore.uncertainty.conformal.finance.var import (
    ConformalVaR,
    ConformalES,
    backtest_var,
    VaRResult,
    VaRBacktestResult,
)
from quantcore.uncertainty.conformal.finance.volatility import (
    ConformalVolatility,
    GARCHConformal,
)
from quantcore.uncertainty.conformal.finance.alpha import (
    ConformalAlphaModel,
    AlphaSignal,
    SignalFilter,
    PortfolioConstructor,
    StrategyMetrics,
    FeatureImportanceConformal,
    compute_strategy_metrics,
    backtest_alpha_model,
)
from quantcore.uncertainty.conformal.finance.backtest_dtaci import (
    backtest_alpha_model_dtaci,
)
from quantcore.uncertainty.conformal.finance.empirical_comparison import (
    BRANCH_NAMES,
    BranchComparisonResult,
    BranchMetrics,
    SyntheticDiagnostics,
    compare_alpha_branches,
)

__all__ = [
    # VaR
    "ConformalVaR",
    "ConformalES",
    "backtest_var",
    "VaRResult",
    "VaRBacktestResult",
    # Volatility
    "ConformalVolatility",
    "GARCHConformal",
    # Alpha Research
    "ConformalAlphaModel",
    "AlphaSignal",
    "SignalFilter",
    "PortfolioConstructor",
    "StrategyMetrics",
    "FeatureImportanceConformal",
    "compute_strategy_metrics",
    "backtest_alpha_model",
    "backtest_alpha_model_dtaci",
    # S16: empirical comparison harness
    "BRANCH_NAMES",
    "BranchComparisonResult",
    "BranchMetrics",
    "SyntheticDiagnostics",
    "compare_alpha_branches",
]
