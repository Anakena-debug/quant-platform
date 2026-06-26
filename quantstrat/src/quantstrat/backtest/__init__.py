"""Thin wrappers over quantengine's ReplayRunner.

Modules here configure ``HistoricalClock``, ``PaperBroker``, ``Ledger``,
``PortfolioState``, and ``RebalanceConstraints`` with strategy-appropriate parameters
and invoke ``ReplayRunner.run(...)``. No fitting or training — that is done upstream in
``quantcore`` and frozen into the ``Strategy`` object passed in here.
"""

from quantstrat.backtest.evaluate import (
    EvalReport,
    compare_strategies,
    deflated_evaluation,
)
from quantstrat.backtest.runner import BacktestResult, CostRealismError, run_backtest

__all__ = [
    "BacktestResult",
    "CostRealismError",
    "EvalReport",
    "compare_strategies",
    "deflated_evaluation",
    "run_backtest",
]
