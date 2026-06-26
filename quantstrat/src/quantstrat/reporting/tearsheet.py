"""Tearsheet + summary table from a ``PerformanceReport`` (read-side; computes no metrics).

``summary_table`` is dependency-free (markdown/plain text). ``tearsheet`` renders a matplotlib
figure (equity curve, drawdown, rolling Sharpe) and lazy-imports matplotlib so this module loads
without the ``reporting`` extra installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantstrat.metrics.performance import (
    DrawdownEvent,
    PerformanceReport,
    relative_metrics_from_series,
    tail_metrics,
)


def rolling_sharpe(returns: pd.Series, window: int = 63, bars_per_year: int = 252) -> pd.Series:
    """Rolling annualised Sharpe of a simple-return series (``ddof=1``)."""
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    return (mu / sd.replace(0.0, np.nan) * np.sqrt(bars_per_year)).rename("rolling_sharpe")


def _fmt_num(v: float) -> str:
    """Two-decimal float, or ``'n/a'`` for a non-finite value."""
    return "n/a" if not np.isfinite(v) else f"{v:.2f}"


def _dd_dates(dd: DrawdownEvent) -> str:
    """Format a ``DrawdownEvent``'s peak → trough → recovery indices compactly."""

    def lab(idx: object) -> str:
        if idx is None:
            return "—"
        if isinstance(idx, pd.Timestamp):
            return idx.date().isoformat()
        return str(idx)

    rec = "not recovered" if dd.recovery_index is None else lab(dd.recovery_index)
    return f"{lab(dd.peak_index)} → {lab(dd.trough_index)} → {rec}"


def _relative_rows(
    returns: pd.Series, benchmark: pd.Series, *, bars_per_year: int, rf: float
) -> list[tuple[str, str]]:
    """Render the benchmark-relative rows (empty when relative metrics are undefined).

    Alignment + the < 2-pair guard live in
    :func:`quantstrat.metrics.performance.relative_metrics_from_series`.
    """
    rm = relative_metrics_from_series(returns, benchmark, bars_per_year=bars_per_year, rf=rf)
    if rm is None:
        return []
    return [
        ("Active return (ann.)", f"{rm.active_return:+.2%}"),
        ("Tracking error (ann.)", f"{rm.tracking_error:.2%}"),
        ("Information ratio", _fmt_num(rm.information_ratio)),
        ("Beta", _fmt_num(rm.beta)),
        ("Alpha (ann.)", "n/a" if not np.isfinite(rm.alpha) else f"{rm.alpha:+.2%}"),
        ("Correlation", _fmt_num(rm.correlation)),
        ("Up / Down capture", f"{_fmt_num(rm.up_capture)} / {_fmt_num(rm.down_capture)}"),
    ]


def summary_table(
    report: PerformanceReport,
    *,
    fmt: str = "markdown",
    benchmark: pd.Series | None = None,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> str:
    """Compact metrics table from a ``PerformanceReport`` (``fmt`` = 'markdown' | 'text').

    Surfaces the absolute metrics, a tail/distribution block (VaR, CVaR, skew, excess
    kurtosis, hit-rate, best/worst via
    :func:`quantstrat.metrics.performance.tail_metrics`), and the max-drawdown geometry
    (peak → trough → recovery). When ``benchmark`` (a return Series) is provided it is
    aligned to ``report.returns.index``, non-finite pairs are dropped, and a
    benchmark-relative block (active return, tracking error, IR, beta, alpha, correlation,
    up/down capture) is appended via
    :func:`quantstrat.metrics.performance.relative_metrics`. Render-only: all computation
    lives in ``metrics``.
    """
    dd = report.max_drawdown
    pairs: list[tuple[str, str]] = [
        ("Ann. return", f"{report.ann_return:+.2%}"),
        ("Ann. vol", f"{report.ann_vol:.2%}"),
        ("Sortino", f"{report.sortino:.2f}"),
        ("Max drawdown", f"{dd.magnitude:.2%}"),
        ("Calmar", "n/a" if report.calmar is None else f"{report.calmar:.2f}"),
        ("Turnover (1-sided)", f"{report.turnover:.2%}"),
        ("NAV start→end", f"{report.nav.iloc[0]:,.0f} → {report.nav.iloc[-1]:,.0f}"),
        ("Periods", f"{len(report.returns)}"),
    ]

    if len(report.returns) >= 1:
        tm = tail_metrics(report.returns)
        tail_pct = f"{1.0 - tm.level:.0%}"
        pairs += [
            (f"VaR ({tail_pct})", f"{tm.var:+.2%}"),
            (f"CVaR ({tail_pct})", f"{tm.cvar:+.2%}"),
            ("Skew", _fmt_num(tm.skew)),
            ("Excess kurtosis", _fmt_num(tm.excess_kurtosis)),
            ("Hit rate", f"{tm.hit_rate:.2%}"),
            ("Best / Worst", f"{tm.best:+.2%} / {tm.worst:+.2%}"),
        ]

    if benchmark is not None:
        pairs += _relative_rows(report.returns, benchmark, bars_per_year=bars_per_year, rf=rf)

    pairs.append(("Max DD dates", _dd_dates(dd)))

    if fmt == "markdown":
        head = "| metric | value |\n|---|---|\n"
        return head + "\n".join(f"| {k} | {v} |" for k, v in pairs)
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in pairs)


def tearsheet(
    report: PerformanceReport,
    *,
    rolling_window: int = 63,
    bars_per_year: int = 252,
    title: str = "Backtest tearsheet",
):
    """Render equity curve + drawdown + rolling Sharpe → a matplotlib ``Figure``.

    Requires the ``reporting`` extra (matplotlib). Caller owns the figure (save/show/close).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)  # headless-safe; honours an already-chosen backend
    import matplotlib.pyplot as plt

    nav, ret = report.nav, report.returns
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title)

    axes[0].plot(nav.index, nav.to_numpy(), color="tab:blue")
    axes[0].set_ylabel("NAV")
    axes[0].set_title("Equity curve")

    peak = nav.cummax()
    underwater = (nav / peak - 1.0).to_numpy()
    axes[1].fill_between(nav.index, underwater, 0.0, color="tab:red", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_title(f"Underwater (max {report.max_drawdown.magnitude:.1%})")

    rs = rolling_sharpe(ret, rolling_window, bars_per_year)
    axes[2].plot(rs.index, rs.to_numpy(), color="tab:green")
    axes[2].axhline(0.0, color="k", lw=0.6)
    axes[2].set_ylabel("Sharpe")
    axes[2].set_title(f"Rolling Sharpe ({rolling_window}-period)")

    fig.tight_layout()
    return fig


__all__ = ["rolling_sharpe", "summary_table", "tearsheet"]
