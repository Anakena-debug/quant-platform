"""Reporting artefacts — tearsheets, plots, summary tables.

Read-side only: consumes ``PerformanceReport`` objects produced by
``quantstrat.metrics.performance.compute_performance``. Does not compute metrics itself.
"""

from quantstrat.reporting.tearsheet import rolling_sharpe, summary_table, tearsheet

__all__ = ["rolling_sharpe", "summary_table", "tearsheet"]
