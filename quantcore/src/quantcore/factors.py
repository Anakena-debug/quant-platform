"""quantcore.factors — canonical cross-sectional factor-panel constructors.

The alpha-factory (:mod:`quantcore.factory`) and the screen (:mod:`quantcore.screening`)
consume factor *panels* shaped ``[dates x assets]`` plus an aligned forward-return panel —
but constructing those was left entirely to the caller (``run_factory``'s docstring:
"candidate construction stays with the caller"). This module promotes the bread-and-butter
cross-sectional factors into the library as small, vectorized constructors over a wide price
panel, so the factory is drivable end-to-end.

Two rules every constructor here obeys:

1. **Point-in-time clean.** A factor at date ``t`` uses only prices at or before ``t`` — no
   lookahead. This is enforced in the tests with :func:`quantcore.leakage.assert_no_lookahead`,
   not just asserted in prose. (The lone exception is :func:`forward_returns`, which *is* the
   forward-looking label — see its note.)
2. **Raw, un-normalized values.** The gates rank each panel cross-sectionally themselves
   (``screen_factors`` via rank-IC, ``run_factory`` via rank-weighted long/short), so factors
   return the raw per-asset characteristic. Each docstring states the expected sign of the
   cross-sectional IC; an anti-predictive orientation simply fails the factory's DSR gate, so
   flip the sign upstream if you want a long signal.

    from quantcore.factors import cross_sectional_momentum, forward_returns
    from quantcore.factory import run_factory

    mom = cross_sectional_momentum(close, lookback=252, skip=21)   # [dates x assets]
    fwd = forward_returns(close, horizon=1)                        # the label
    verdicts = run_factory({"momentum": mom}, fwd)
"""

from __future__ import annotations

from typing import cast

import pandas as pd

__all__ = [
    "cross_sectional_illiquidity",
    "cross_sectional_momentum",
    "cross_sectional_reversal",
    "cross_sectional_volatility",
    "forward_returns",
]


def _check_panel(panel: pd.DataFrame, name: str) -> None:
    """A factor panel must be a date-sorted ``[dates x assets]`` frame for PIT correctness."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a [dates x assets] DataFrame, got {type(panel).__name__}")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(f"{name} index (dates) must be sorted ascending for point-in-time use")


def cross_sectional_momentum(
    close: pd.DataFrame, *, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """Trailing total return over ``[t-lookback, t-skip]`` per asset (Jegadeesh-Titman momentum).

    Higher = stronger past winner. ``skip`` drops the most recent ``skip`` days, whose
    short-term reversal would otherwise contaminate the trend. Point-in-time clean (uses only
    prices at or before ``t-skip``). Expected cross-sectional IC is **positive** — winners keep
    winning.
    """
    _check_panel(close, "close")
    if not 0 <= skip < lookback:
        raise ValueError(f"require 0 <= skip < lookback, got skip={skip}, lookback={lookback}")
    return close.shift(skip) / close.shift(lookback) - 1.0


def cross_sectional_reversal(close: pd.DataFrame, *, lookback: int = 21) -> pd.DataFrame:
    """Short-term reversal: the negative of the trailing ``lookback``-day return per asset.

    Higher = recent loser (expected to bounce). Point-in-time clean. Expected cross-sectional IC
    is **positive** — oversold names outperform over the short horizon; negate for a momentum read.
    """
    _check_panel(close, "close")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    return -(close / close.shift(lookback) - 1.0)


def cross_sectional_volatility(close: pd.DataFrame, *, lookback: int = 63) -> pd.DataFrame:
    """Trailing realized volatility of daily returns over ``lookback`` days, per asset.

    The raw characteristic (higher = more volatile). Point-in-time clean. The low-volatility
    anomaly implies a **negative** expected cross-sectional IC — high-vol names underperform —
    so negate for a long-the-calm signal.
    """
    _check_panel(close, "close")
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2 for a std estimate, got {lookback}")
    # cast: DataFrame.rolling(...).std() is typed as a DataFrame|Series union in pandas-stubs.
    return cast(pd.DataFrame, close.pct_change().rolling(lookback).std())


def cross_sectional_illiquidity(
    close: pd.DataFrame, volume: pd.DataFrame, *, lookback: int = 21
) -> pd.DataFrame:
    """Amihud illiquidity: rolling mean of ``|return| / dollar_volume`` over ``lookback`` days.

    Dollar volume is ``close * volume``; days of non-positive dollar volume become NaN rather
    than dividing by zero. Higher = more illiquid (larger price impact per traded dollar).
    Point-in-time clean. The illiquidity premium implies a **positive** expected cross-sectional
    IC, but it is fragile on a survivorship-biased universe (an artifact in this program's s61
    revalidation) — screen it on a delisting-inclusive panel.
    """
    _check_panel(close, "close")
    _check_panel(volume, "volume")
    abs_return = close.pct_change().abs()
    dollar_volume = close * volume
    dollar_volume = dollar_volume.where(dollar_volume > 0)  # non-positive -> NaN, not div-by-zero
    # cast: DataFrame.rolling(...).mean() is typed as a DataFrame|Series union in pandas-stubs.
    return cast(pd.DataFrame, abs_return.div(dollar_volume).rolling(lookback).mean())


def forward_returns(close: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """The forward ``horizon``-day return per asset, aligned at the decision date ``t``.

    ``forward_returns(close).loc[t]`` equals ``close[t+horizon] / close[t] - 1`` — the return
    earned *after* acting on a factor known at ``t``, and exactly the label the screen / factory
    score factors against. It is **intentionally forward-looking**: never feed it in as a feature.
    (Passed to :func:`quantcore.leakage.assert_no_lookahead` it will, correctly, raise — that is
    the feature/label boundary working as designed.)
    """
    _check_panel(close, "close")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return close.shift(-horizon) / close - 1.0
