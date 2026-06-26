"""Deflated, multiple-testing-aware evaluation of backtest returns.

A realised Sharpe is meaningless without accounting for how many strategies were tried — the
lesson the research arc kept relearning. This module wraps ``quantcore.validation.stats`` to
report the **Probabilistic** and **Deflated** Sharpe (Bailey-López de Prado 2014) on a backtest's
OOS return series, a per-window stability breakdown, and a multi-strategy comparison that
deflates the winner for the number of trials (the canonical cross-trial σ).

Read-side only: operates on a return series (e.g. ``BacktestResult.report.returns``); runs no
backtest itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantcore.validation.stats import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

SIGNIFICANCE = 0.95  # DSR/PSR probability above which we call an edge significant


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Deflated evaluation of one strategy's OOS returns.

    sharpe  : annualised Sharpe of the return series.
    psr     : P[true SR > benchmark] from a single path (probabilistic Sharpe).
    dsr     : P[true SR > benchmark] AFTER deflating for ``n_trials`` (deflated Sharpe).
    n_trials: number of strategy trials the candidate was selected from.
    window_sharpes : per-window annualised Sharpe (regime/time stability).
    is_significant : ``dsr >= SIGNIFICANCE``.
    """

    sharpe: float
    psr: float
    dsr: float
    n_trials: int
    window_sharpes: tuple[float, ...]
    is_significant: bool


def _returns_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=np.float64)
    return x[np.isfinite(x)]


def deflated_evaluation(
    returns: pd.Series | np.ndarray,
    *,
    n_trials: int = 1,
    n_windows: int = 4,
    periods_per_year: int = 252,
    sr_benchmark: float = 0.0,
    sr_std_cross_trial: float | None = None,
) -> EvalReport:
    """Deflated/probabilistic Sharpe + per-window stability for a return series.

    Pass ``BacktestResult.report.returns`` as ``returns``. ``n_trials`` is how many strategy
    variants you tried (>1 deflates the Sharpe). For a single strategy the DSR uses the
    single-path σ̂ fallback (a quantcore ``UserWarning``, suppressed here as intentional).
    """
    x = _returns_array(returns)
    if x.size < 4:
        raise ValueError(f"deflated_evaluation needs >= 4 finite returns; got {x.size}")
    sr = sharpe_ratio(x, periods_per_year=periods_per_year)
    psr, _ = probabilistic_sharpe_ratio(x, sr_benchmark, periods_per_year=periods_per_year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # single-path σ̂ fallback is intentional here
        dsr, _ = deflated_sharpe_ratio(
            x,
            n_trials,
            sr_benchmark,
            periods_per_year=periods_per_year,
            sr_std_cross_trial=sr_std_cross_trial,
        )
    windows = [w for w in np.array_split(x, n_windows) if w.size >= 2]
    window_sharpes = tuple(
        float(sharpe_ratio(w, periods_per_year=periods_per_year)) for w in windows
    )
    return EvalReport(
        sharpe=float(sr),
        psr=float(psr),
        dsr=float(dsr),
        n_trials=int(n_trials),
        window_sharpes=window_sharpes,
        is_significant=bool(dsr >= SIGNIFICANCE),
    )


def compare_strategies(
    returns_by_name: dict[str, pd.Series | np.ndarray],
    *,
    periods_per_year: int = 252,
    sr_benchmark: float = 0.0,
) -> pd.DataFrame:
    """Rank K strategies by Deflated Sharpe, deflating each for the K trials.

    Uses the canonical cross-trial σ (std of the K annualised Sharpes) so the DSR answers
    "is this strategy real, or just the luckiest of K?". Returns a frame sorted by DSR desc.
    """
    series = {name: _returns_array(r) for name, r in returns_by_name.items()}
    n_trials = len(series)
    srs = {
        name: float(sharpe_ratio(x, periods_per_year=periods_per_year))
        for name, x in series.items()
    }
    cross = float(np.std(list(srs.values()), ddof=1)) if n_trials >= 2 else None

    rows = []
    for name, x in series.items():
        psr, _ = probabilistic_sharpe_ratio(x, sr_benchmark, periods_per_year=periods_per_year)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dsr, _ = deflated_sharpe_ratio(
                x,
                n_trials,
                sr_benchmark,
                periods_per_year=periods_per_year,
                sr_std_cross_trial=cross,
            )
        rows.append(
            {
                "strategy": name,
                "sharpe": srs[name],
                "psr": float(psr),
                "dsr": float(dsr),
                "n_trials": n_trials,
                "is_significant": bool(dsr >= SIGNIFICANCE),
            }
        )
    return pd.DataFrame(rows).sort_values("dsr", ascending=False).reset_index(drop=True)


__all__ = ["EvalReport", "SIGNIFICANCE", "compare_strategies", "deflated_evaluation"]
