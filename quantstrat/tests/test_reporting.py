"""quantstrat — reporting (summary table + tearsheet) tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantstrat.metrics.performance import DrawdownEvent, PerformanceReport, max_drawdown
from quantstrat.reporting import summary_table, tearsheet


def _report() -> PerformanceReport:
    dates = pd.bdate_range("2024-01-01", periods=20)
    nav = pd.Series(1_000_000.0 * np.cumprod(1.0 + np.r_[0.0, np.full(19, 0.001)]), index=dates)
    returns = nav.pct_change().dropna()
    dd = DrawdownEvent(
        magnitude=-0.03, peak_index=dates[2], trough_index=dates[5], recovery_index=dates[8]
    )
    return PerformanceReport(
        ann_return=0.12,
        ann_vol=0.15,
        sortino=1.1,
        max_drawdown=dd,
        calmar=4.0,
        turnover=0.2,
        nav=nav,
        returns=returns,
    )


def test_summary_table_markdown():
    s = summary_table(_report())
    assert "| metric | value |" in s
    assert "Ann. return" in s and "+12.00%" in s
    assert "Max drawdown" in s and "-3.00%" in s


def test_summary_table_text_and_calmar_none():
    rep = _report()
    object.__setattr__(rep, "calmar", None)  # frozen dataclass; simulate no-drawdown case
    s = summary_table(rep, fmt="text")
    assert "Sortino" in s and "Calmar" in s and "n/a" in s


def test_tearsheet_returns_three_panel_figure():
    pytest.importorskip("matplotlib")
    fig = tearsheet(_report())
    try:
        assert len(fig.axes) == 3  # equity, drawdown, rolling Sharpe
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _rich_report() -> PerformanceReport:
    """A report with a real (non-degenerate) return path so the tail block is finite."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=200)
    rets = pd.Series(rng.normal(0.0003, 0.012, len(dates)), index=dates)
    nav = pd.Series(1_000_000.0 * np.cumprod(1.0 + rets.to_numpy()), index=dates)
    returns = nav.pct_change().dropna()
    return PerformanceReport(
        ann_return=0.10,
        ann_vol=0.18,
        sortino=0.9,
        max_drawdown=max_drawdown(returns),
        calmar=1.5,
        turnover=0.3,
        nav=nav,
        returns=returns,
    )


def test_summary_table_surfaces_tail_block_additively():
    rep = _rich_report()
    s = summary_table(rep)
    # existing rows remain (additive — nothing removed or altered)
    for existing in ("Ann. return", "Max drawdown", "Turnover (1-sided)", "Periods"):
        assert existing in s
    # new s68 tail/risk block
    for label in ("VaR (5%)", "CVaR (5%)", "Skew", "Excess kurtosis", "Hit rate", "Best / Worst"):
        assert label in s
    # drawdown geometry row carries the peak → trough → recovery labels
    assert "Max DD dates" in s and "→" in s


def test_summary_table_dd_dates_marks_unrecovered():
    # a report whose worst drawdown never recovers within the window
    dates = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.02, 0.01, -0.05, -0.04, -0.03, -0.02], index=dates)
    nav = pd.Series(1_000_000.0 * np.cumprod(1.0 + returns.to_numpy()), index=dates)
    rep = PerformanceReport(
        ann_return=-0.5,
        ann_vol=0.3,
        sortino=float("nan"),
        max_drawdown=max_drawdown(returns),
        calmar=None,
        turnover=0.1,
        nav=nav,
        returns=returns,
    )
    assert rep.max_drawdown.recovery_index is None  # precondition: unrecovered
    s = summary_table(rep, fmt="text")
    assert "not recovered" in s


def test_summary_table_benchmark_appends_relative_block_additively():
    rep = _rich_report()
    rng = np.random.default_rng(99)
    bench = pd.Series(rng.normal(0.0002, 0.010, len(rep.returns)), index=rep.returns.index)

    plain = summary_table(rep)
    withb = summary_table(rep, benchmark=bench)

    # absent a benchmark, no relative rows are emitted
    assert "Beta" not in plain and "Information ratio" not in plain
    # benchmark -> the full relative block is appended
    for label in (
        "Active return (ann.)",
        "Tracking error (ann.)",
        "Information ratio",
        "Beta",
        "Alpha (ann.)",
        "Correlation",
        "Up / Down capture",
    ):
        assert label in withb
    # additive: every row of the plain table still appears verbatim in the benchmarked one
    for line in plain.splitlines():
        assert line in withb
