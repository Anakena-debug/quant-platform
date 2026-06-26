"""Backtest-level performance metrics.

Public entry point
------------------
``compute_performance(run_frames, price_panel, *, bars_per_year=252, rf=0.0)``
    Takes the dict produced by
    ``quantengine.runtime.state_store.DuckDBStore.load_run(run_id)`` and the price
    panel ``ReplayRunner`` consumed (DatetimeIndex rows, ticker columns, reference
    prices in cells; NaN / non-positive = ticker not in universe). Reconstructs NAV
    internally — callers do not rebuild equity curves themselves — and returns a
    ``PerformanceReport``.

Low-level primitives (``annualized_return``, ``annualized_volatility``,
``sortino_ratio``, ``max_drawdown``, ``calmar_ratio``, ``turnover_from_weights``) take
canonical inputs (returns array / series, weights DataFrame) and are exposed for
ad-hoc use outside the engine flow.

Scope split versus ``quantcore.validation.stats``
-------------------------------------------------
Sharpe ratio, probabilistic Sharpe, deflated Sharpe, probability of backtest
overfitting, and haircut Sharpe are statistical-inference primitives and live in
``quantcore.validation.stats``. Import from there. This module owns only what
``quantcore`` does not: drawdown geometry, downside-only volatility (Sortino),
Calmar, turnover, and the end-to-end engine-output → report orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import pandas as pd

ReturnsLike: TypeAlias = np.ndarray | pd.Series

_MIN_REL_STD = 1e-8
"""Degenerate-dispersion floor, mirroring ``quantcore.validation.stats._MIN_REL_STD``.

Kept as a local constant rather than imported to avoid depending on a private name in
quantcore. Keep the two in sync when either moves.
"""


# ---------------------------------------------------------------------------
# Structured containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrawdownEvent:
    """Largest peak-to-trough equity excursion on a returns series.

    Attributes
    ----------
    magnitude : float
        Non-positive fractional loss, ``equity[trough] / equity[peak] - 1``. 0.0
        means no drawdown was observed on the input series.
    peak_index : pd.Timestamp | int | None
        Index label (for a ``pd.Series`` input) or positional index (for an
        ``np.ndarray`` input) of the high-water mark preceding the trough. ``None``
        iff the input series is empty.
    trough_index : pd.Timestamp | int | None
        Index of the trough. Equal to ``peak_index`` when ``magnitude == 0``.
    recovery_index : pd.Timestamp | int | None
        First index at or after ``trough_index`` where equity returned to or exceeded
        the prior peak. ``None`` if recovery has not occurred within the observed
        series (or if ``magnitude == 0``, where the notion is not meaningful).
    """

    magnitude: float
    peak_index: pd.Timestamp | int | None
    trough_index: pd.Timestamp | int | None
    recovery_index: pd.Timestamp | int | None


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Backtest performance summary.

    Attributes
    ----------
    ann_return : float
        Annualised geometric mean return of the reconstructed NAV.
    ann_vol : float
        Annualised sample standard deviation (``ddof=1``) of period returns.
    sortino : float
        Annualised Sortino ratio. ``NaN`` when the series has no observable
        downside (see :func:`sortino_ratio`).
    max_drawdown : DrawdownEvent
        Structured max-drawdown event — magnitude plus peak, trough, and (when
        observed) recovery indices.
    calmar : float | None
        Annualised return divided by ``|max_drawdown.magnitude|``. ``None`` when no
        drawdown occurred (ratio undefined).
    turnover : float
        Mean one-sided fractional turnover across successive rebalance timestamps,
        computed from the per-timestamp weights derived in NAV reconstruction.
        ``0.0`` if the series has fewer than two non-trivial weight rows.
    nav : pd.Series
        Reconstructed equity curve indexed by the price-panel timestamps.
    returns : pd.Series
        Simple-return series derived from ``nav``; length ``len(nav) - 1``.
    """

    ann_return: float
    ann_vol: float
    sortino: float
    max_drawdown: DrawdownEvent
    calmar: float | None
    turnover: float
    nav: pd.Series
    returns: pd.Series


@dataclass(frozen=True, slots=True)
class TailMetrics:
    """Tail / distribution risk metrics for a simple-return series.

    Attributes
    ----------
    level : float
        Confidence level used for ``var`` / ``cvar`` (e.g. ``0.95``); the tail
        probability is ``1 - level``.
    var : float
        Historical Value-at-Risk — the ``1 - level`` empirical return quantile
        (linear interpolation). **Signed**: a loss is negative, so ``var = -0.02`` at
        ``level=0.95`` reads "5% of periods returned -2% or worse".
    cvar : float
        Conditional VaR / expected shortfall — the mean of returns at or below
        ``var`` (the average tail loss). Signed; ``<= var`` by construction.
    skew : float
        Bias-corrected sample skewness (adjusted Fisher-Pearson G1, matching
        ``scipy.stats.skew(bias=False)`` / ``pandas.Series.skew``). ``NaN`` with fewer
        than 3 observations or ~zero dispersion.
    excess_kurtosis : float
        Bias-corrected sample excess kurtosis (G2, Fisher — normal ⇒ 0, matching
        ``scipy.stats.kurtosis(fisher=True, bias=False)`` / ``pandas.Series.kurt``).
        ``NaN`` with fewer than 4 observations or ~zero dispersion.
    hit_rate : float
        Fraction of strictly-positive returns, in ``[0, 1]``.
    best : float
        Largest single-period return.
    worst : float
        Smallest (most negative) single-period return.
    """

    level: float
    var: float
    cvar: float
    skew: float
    excess_kurtosis: float
    hit_rate: float
    best: float
    worst: float


@dataclass(frozen=True, slots=True)
class RelativeMetrics:
    """Benchmark-relative performance metrics for a return series.

    All ratios are computed on per-period simple returns aligned elementwise with the
    benchmark. Annualised figures use arithmetic scaling (``× bars_per_year`` for means,
    ``× √bars_per_year`` for dispersion), consistent with
    :func:`information_ratio`.

    Attributes
    ----------
    active_return : float
        Annualised mean active return, ``mean(returns - benchmark) × bars_per_year``.
    tracking_error : float
        Annualised standard deviation of the active return, ``std(active, ddof=1) ×
        √bars_per_year``.
    information_ratio : float
        ``active_return / tracking_error`` — the value of :func:`information_ratio` on
        the same inputs (single source of truth). ``NaN`` when tracking error is
        degenerate (active series ≈ constant).
    beta : float
        OLS sensitivity to the benchmark, ``cov(returns, benchmark) / var(benchmark)``
        (``ddof=1``). ``NaN`` when the benchmark has ~zero variance.
    alpha : float
        Annualised Jensen's alpha, ``((mean(returns) - rf) - beta·(mean(benchmark) -
        rf)) × bars_per_year``. ``NaN`` when ``beta`` is undefined.
    correlation : float
        Pearson correlation between ``returns`` and ``benchmark``. ``NaN`` when either
        series has ~zero variance.
    up_capture : float
        ``mean(returns | benchmark > 0) / mean(benchmark | benchmark > 0)`` — the share
        of the benchmark's up-period return captured. ``NaN`` when there are no up
        periods or the up-period benchmark mean is ~zero.
    down_capture : float
        The ``benchmark < 0`` analogue of ``up_capture``. A value below 1 means smaller
        losses than the benchmark in down periods.
    """

    active_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    correlation: float
    up_capture: float
    down_capture: float


@dataclass(frozen=True, slots=True)
class FactorAttribution:
    """Multi-factor OLS attribution of a return series onto K factor series.

    Generalises :class:`RelativeMetrics`' single-benchmark ``beta`` / ``alpha`` to a regression
    ``(returns - rf) = alpha_pb + Σ_i beta_i · factor_i + resid``. Annualised figures use arithmetic
    scaling (``× bars_per_year`` for the intercept, ``× √bars_per_year`` for residual dispersion),
    consistent with :class:`RelativeMetrics`.

    Attributes
    ----------
    alpha : float
        Annualised regression intercept, ``intercept × bars_per_year``. For a single factor at
        ``rf=0`` this equals :class:`RelativeMetrics`.``alpha``. ``NaN`` when the design is
        rank-deficient.
    alpha_tstat : float
        t-statistic of the intercept under homoskedastic OLS standard errors. ``NaN`` when the
        design is rank-deficient or the residuals are ~zero (infinite significance).
    betas : dict[str, float]
        Factor loadings keyed by factor name (key order = the order of ``factors``). For a single
        factor this equals :class:`RelativeMetrics`.``beta``. Per-factor ``NaN`` when rank-deficient.
    beta_tstats : dict[str, float]
        Per-factor t-statistics (same keys as ``betas``).
    r_squared : float
        Ordinary R², ``1 - SS_res / SS_tot``. ``NaN`` when ``returns`` is ~constant.
    residual_volatility : float
        Annualised residual standard deviation, ``√(SS_res / dof) × √bars_per_year`` where
        ``dof = n - (K + 1)`` — the idiosyncratic (un-attributed) risk.
    n_obs : int
        Number of observations entering the regression.
    """

    alpha: float
    alpha_tstat: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    r_squared: float
    residual_volatility: float
    n_obs: int


# ---------------------------------------------------------------------------
# Array primitives — take canonical returns input
# ---------------------------------------------------------------------------


def _as_1d_array(returns: ReturnsLike) -> np.ndarray:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {x.shape}")
    return x


def annualized_return(returns: ReturnsLike, bars_per_year: int = 252) -> float:
    """Annualised geometric mean return on a simple-return series.

    Computes ``(∏(1 + r))^(bars_per_year / T) - 1`` with ``T = len(returns)``.
    Returns ``0.0`` on empty input; returns ``-1.0`` when cumulative product is
    non-positive (full wipeout — geometric annualisation otherwise undefined).
    """
    x = _as_1d_array(returns)
    if x.size == 0:
        return 0.0
    compounded = float(np.prod(1.0 + x))
    if compounded <= 0.0:
        return -1.0
    return compounded ** (bars_per_year / x.size) - 1.0


def annualized_volatility(returns: ReturnsLike, bars_per_year: int = 252) -> float:
    """Annualised sample standard deviation of simple returns (``ddof=1``)."""
    x = _as_1d_array(returns)
    if x.size < 2:
        raise ValueError(f"annualized_volatility: need at least 2 observations; got {x.size}")
    return float(x.std(ddof=1) * np.sqrt(bars_per_year))


def sortino_ratio(
    returns: ReturnsLike,
    rf: float = 0.0,
    bars_per_year: int = 252,
) -> float:
    """Annualised Sortino ratio.

    ``mean(r - rf) * √P / downside_dev`` where
    ``downside_dev = sqrt(mean(min(r - rf, 0) ** 2))`` (AFML §14.7, Sortino-Van der
    Meer 1991). ``rf`` doubles as the minimum-acceptable-return threshold for the
    downside denominator.

    Raises ``ValueError`` when downside deviation is degenerate (no observations
    below ``rf`` on the input, or dispersion within numerical noise) — the ratio is
    undefined in that regime. ``compute_performance`` catches this and records
    ``sortino = NaN`` in the report.
    """
    x = _as_1d_array(returns) - rf
    if x.size < 2:
        raise ValueError(f"sortino_ratio: need at least 2 observations; got {x.size}")
    downside = np.minimum(x, 0.0)
    dd_dev = float(np.sqrt(np.mean(downside * downside)))
    scale = max(float(np.median(np.abs(x))), 1.0)
    if not np.isfinite(dd_dev) or dd_dev < _MIN_REL_STD * scale:
        raise ValueError(
            f"sortino_ratio: downside deviation degenerate "
            f"(dd_dev={dd_dev:.3e} vs scale={scale:.3e})."
        )
    return float(x.mean() / dd_dev * np.sqrt(bars_per_year))


def max_drawdown(returns: ReturnsLike) -> DrawdownEvent:
    """Largest peak-to-trough drawdown on a simple-return series.

    Builds the compounded equity curve, locates the trough (minimum of
    ``equity / running_peak - 1``), the preceding peak, and the first post-trough
    index at or above the peak (``recovery_index``, or ``None`` if the series does
    not recover within its observed range).

    For an ``np.ndarray`` input, indices are positional ``int``. For a ``pd.Series``
    input, indices are the series' index labels (typically ``pd.Timestamp``).
    """
    x = _as_1d_array(returns)
    if x.size == 0:
        return DrawdownEvent(0.0, None, None, None)

    equity = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0

    trough_pos = int(np.argmin(dd))
    magnitude = float(dd[trough_pos])
    peak_pos = int(np.argmax(equity[: trough_pos + 1]))

    recovery_pos: int | None = None
    if magnitude < 0.0:
        peak_value = float(equity[peak_pos])
        post = equity[trough_pos + 1 :]
        recovered = np.where(post >= peak_value)[0]
        if recovered.size > 0:
            recovery_pos = int(recovered[0] + trough_pos + 1)

    def _resolve(pos: int | None) -> pd.Timestamp | int | None:
        if pos is None:
            return None
        if isinstance(returns, pd.Series):
            return returns.index[pos]
        return pos

    return DrawdownEvent(
        magnitude=magnitude,
        peak_index=_resolve(peak_pos),
        trough_index=_resolve(trough_pos),
        recovery_index=_resolve(recovery_pos),
    )


def calmar_ratio(returns: ReturnsLike, bars_per_year: int = 252) -> float | None:
    """Annualised return divided by ``|max_drawdown.magnitude|``.

    Returns ``None`` when no drawdown is observed on the input — ratio undefined,
    but propagated as ``None`` so callers can distinguish "no-loss" from
    "compute failed".
    """
    mdd = max_drawdown(returns).magnitude
    if mdd == 0.0:
        return None
    return annualized_return(returns, bars_per_year) / abs(mdd)


def turnover_from_weights(weights: pd.DataFrame, *, one_sided: bool = True) -> float:
    """Mean turnover across successive rebalance timestamps.

    ``one_sided=True`` (convention) returns ``0.5 · Σ |Δw|`` — the share of
    portfolio value traded per rebalance. ``one_sided=False`` returns the two-sided
    sum. Returns ``0.0`` when fewer than two rows are provided (no delta observable
    rather than raising — useful when the same helper runs on degenerate synthetic
    inputs from ``compute_performance``).
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            f"turnover_from_weights: weights must be a DataFrame; got {type(weights).__name__}"
        )
    if weights.shape[0] < 2:
        return 0.0
    deltas = weights.diff().abs().sum(axis=1).iloc[1:]
    mean_delta = float(deltas.mean())
    return 0.5 * mean_delta if one_sided else mean_delta


# ---------------------------------------------------------------------------
# NAV reconstruction from engine output
# ---------------------------------------------------------------------------


def _reconstruct_nav(
    fills: pd.DataFrame,
    price_panel: pd.DataFrame,
    *,
    initial_cash: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk fills forward over the price panel to build NAV and per-ticker weights.

    Expected ``fills`` schema (per ``quantengine.runtime.state_store`` DDL):

    - ``timestamp`` : ISO-8601 string. Coerced to ``pd.Timestamp`` internally.
    - ``ticker`` : str.
    - ``signed_quantity`` : int; positive = buy, negative = sell.
    - ``price`` : float; per-share fill price.
    - ``commission`` : float; non-negative, optional (treated as 0 when missing).
    - ``seq`` : int; optional tiebreaker when multiple fills share a timestamp.

    Price panel cells that are finite and strictly positive count as valuable;
    ``NaN`` or non-positive cells are treated as "ticker not in universe" per the
    ``quantengine.backtest.replay`` docstring, so held positions in those tickers
    are retained but not marked to market at that step.

    Returns
    -------
    nav : pd.Series
        Equity curve indexed by ``price_panel.index``.
    weights : pd.DataFrame
        Per-timestamp portfolio weights (rows = panel timestamps, columns = panel
        tickers). Cash is implicit: ``1 - weights.sum(axis=1)``.
    """
    ts_index = price_panel.index
    tickers = price_panel.columns
    n_ts = len(ts_index)
    n_tickers = len(tickers)
    ticker_to_col = {t: i for i, t in enumerate(tickers)}

    positions = np.zeros(n_tickers, dtype=np.float64)
    cash = float(initial_cash)
    nav_values = np.empty(n_ts, dtype=np.float64)
    mv_values = np.zeros((n_ts, n_tickers), dtype=np.float64)

    fill_records: list[tuple[pd.Timestamp, str, float, float, float]]
    if fills.empty:
        fill_records = []
    else:
        f = fills.copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        sort_keys = ["timestamp", "seq"] if "seq" in f.columns else ["timestamp"]
        f = f.sort_values(sort_keys, kind="stable").reset_index(drop=True)
        commission = (
            f["commission"].fillna(0.0)
            if "commission" in f.columns
            else pd.Series(0.0, index=f.index)
        )
        fill_records = list(
            zip(
                f["timestamp"].tolist(),
                f["ticker"].tolist(),
                f["signed_quantity"].astype(float).tolist(),
                f["price"].astype(float).tolist(),
                commission.astype(float).tolist(),
                strict=True,
            )
        )

    fill_idx = 0
    n_fills = len(fill_records)

    for i in range(n_ts):
        ts_pd = pd.Timestamp(ts_index[i])
        while fill_idx < n_fills and fill_records[fill_idx][0] <= ts_pd:
            _, ticker, qty, price, commission_i = fill_records[fill_idx]
            col = ticker_to_col.get(ticker)
            if col is not None:
                positions[col] += qty
            cash -= qty * price + commission_i
            fill_idx += 1

        row = price_panel.iloc[i].to_numpy(dtype=np.float64, copy=False)
        valuable = np.where(np.isfinite(row) & (row > 0.0), row, 0.0)
        mv = positions * valuable
        mv_values[i, :] = mv
        nav_values[i] = cash + mv.sum()

    nav = pd.Series(nav_values, index=ts_index, name="nav")
    safe_nav = np.where(nav_values != 0.0, nav_values, np.nan)
    weights = pd.DataFrame(mv_values / safe_nav[:, None], index=ts_index, columns=tickers).fillna(
        0.0
    )
    return nav, weights


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_REQUIRED_FRAMES = ("run", "fills")


def compute_performance(
    run_frames: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    *,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> PerformanceReport:
    """End-to-end performance metrics from engine output + price panel.

    Parameters
    ----------
    run_frames : dict[str, pd.DataFrame]
        Dict returned by
        ``quantengine.runtime.state_store.DuckDBStore.load_run(run_id)``. Must
        contain keys ``'run'`` (one row, carrying ``initial_cash``) and
        ``'fills'`` (schema per ``quantengine.runtime.state_store`` DDL). Other
        frames (``orders``, ``positions``, ``lifecycle``) are accepted but unused
        here.
    price_panel : pd.DataFrame
        The same DataFrame ``ReplayRunner`` consumed — ``DatetimeIndex`` rows,
        ticker columns, reference prices in cells. NaN / non-positive = "ticker
        not in universe at this timestamp".
    bars_per_year : int, default 252
        Annualisation factor. Use 12 for monthly bars, 52 for weekly.
    rf : float, default 0.0
        Per-period risk-free / minimum-acceptable-return, propagated into the
        Sortino denominator.

    Returns
    -------
    PerformanceReport
        Annualised return, annualised vol, Sortino (``NaN`` when downside is
        absent), structured ``DrawdownEvent``, Calmar (``None`` when no drawdown),
        turnover, and the reconstructed ``nav`` / ``returns`` series for downstream
        plotting.
    """
    missing = [k for k in _REQUIRED_FRAMES if k not in run_frames]
    if missing:
        raise KeyError(f"compute_performance: run_frames missing required keys: {missing}")

    run_row = run_frames["run"]
    if run_row.empty:
        raise ValueError("compute_performance: 'run' frame is empty — no run to evaluate.")
    if "initial_cash" not in run_row.columns:
        raise KeyError(
            "compute_performance: 'run' frame missing 'initial_cash' column; "
            "cannot reconstruct NAV."
        )
    ic_val = run_row["initial_cash"].iloc[0]
    if pd.isna(ic_val):
        raise ValueError(
            "compute_performance: 'initial_cash' is NULL in the 'run' frame; "
            "cannot reconstruct NAV."
        )
    initial_cash = float(ic_val)

    nav, weights = _reconstruct_nav(run_frames["fills"], price_panel, initial_cash=initial_cash)
    returns = nav.pct_change().dropna()

    if returns.size < 2:
        ann_ret = 0.0
        ann_vol = 0.0
        sortino = float("nan")
    else:
        ann_ret = annualized_return(returns, bars_per_year)
        ann_vol = annualized_volatility(returns, bars_per_year)
        try:
            sortino = sortino_ratio(returns, rf=rf, bars_per_year=bars_per_year)
        except ValueError:
            sortino = float("nan")

    mdd = max_drawdown(returns)
    calmar = calmar_ratio(returns, bars_per_year)
    turn = turnover_from_weights(weights)

    return PerformanceReport(
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        turnover=turn,
        nav=nav,
        returns=returns,
    )


def information_ratio(
    returns: ReturnsLike, benchmark: ReturnsLike, bars_per_year: int = 252
) -> float:
    """Annualised Information Ratio: mean active return / tracking error.

    ``active = returns - benchmark`` (elementwise; equal length required). Raises ``ValueError``
    on < 2 observations or degenerate (≈ zero) tracking error — IR is undefined there.
    """
    r = _as_1d_array(returns)
    b = _as_1d_array(benchmark)
    if r.shape != b.shape:
        raise ValueError(f"returns and benchmark must align; got {r.shape} vs {b.shape}")
    if r.size < 2:
        raise ValueError(f"information_ratio: need >= 2 observations; got {r.size}")
    active = r - b
    te = float(active.std(ddof=1))
    if not np.isfinite(te) or te <= _MIN_REL_STD:
        raise ValueError("information_ratio: degenerate tracking error (active series ≈ constant)")
    return float(active.mean() / te * np.sqrt(bars_per_year))


def _capture_ratio(r: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean(r | mask) / mean(b | mask); ``NaN`` when the subset is empty or its b-mean ≈ 0."""
    if not mask.any():
        return float("nan")
    denom = float(b[mask].mean())
    if abs(denom) < _MIN_REL_STD:
        return float("nan")
    return float(r[mask].mean()) / denom


def relative_metrics(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> RelativeMetrics:
    """Benchmark-relative metrics: active return, tracking error, IR, beta, alpha, capture.

    ``returns`` and ``benchmark`` must be the same length and already aligned (no index
    join is performed — mirror :func:`information_ratio`). Raises ``ValueError`` on a
    shape mismatch or fewer than 2 observations. Individually-undefined metrics return
    ``NaN`` rather than raising: ``information_ratio`` when tracking error is degenerate;
    ``beta`` / ``alpha`` when the benchmark variance is ~zero; ``correlation`` when either
    series is ~constant; the capture ratios when their period subset is empty.
    """
    r = _as_1d_array(returns)
    b = _as_1d_array(benchmark)
    if r.shape != b.shape:
        raise ValueError(f"returns and benchmark must align; got {r.shape} vs {b.shape}")
    if r.size < 2:
        raise ValueError(f"relative_metrics: need >= 2 observations; got {r.size}")

    active = r - b
    tracking_error = float(active.std(ddof=1)) * np.sqrt(bars_per_year)
    active_return = float(active.mean()) * bars_per_year
    try:
        ir = information_ratio(r, b, bars_per_year)
    except ValueError:
        ir = float("nan")

    var_b = float(b.var(ddof=1))
    scale_b = max(float(np.median(np.abs(b))), 1.0)
    if not np.isfinite(var_b) or var_b < (_MIN_REL_STD * scale_b) ** 2:
        beta = float("nan")
        alpha = float("nan")
    else:
        beta = float(np.cov(r, b, ddof=1)[0, 1]) / var_b
        alpha = ((float(r.mean()) - rf) - beta * (float(b.mean()) - rf)) * bars_per_year

    var_r = float(r.var(ddof=1))
    scale_r = max(float(np.median(np.abs(r))), 1.0)
    if var_r < (_MIN_REL_STD * scale_r) ** 2 or var_b < (_MIN_REL_STD * scale_b) ** 2:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(r, b)[0, 1])

    return RelativeMetrics(
        active_return=active_return,
        tracking_error=tracking_error,
        information_ratio=ir,
        beta=beta,
        alpha=alpha,
        correlation=correlation,
        up_capture=_capture_ratio(r, b, b > 0.0),
        down_capture=_capture_ratio(r, b, b < 0.0),
    )


def relative_metrics_from_series(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> RelativeMetrics | None:
    """Align a benchmark return Series to ``returns`` and compute :class:`RelativeMetrics`.

    Reindexes ``benchmark`` onto ``returns.index``, drops timestamps where either side is
    non-finite, and calls :func:`relative_metrics` on the surviving aligned pairs. Returns
    ``None`` when fewer than 2 aligned observations remain (relative metrics undefined) — the
    Series-level counterpart to the array primitive, shared by ``run_backtest`` and the
    tearsheet so neither re-implements the alignment.
    """
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        raise TypeError("relative_metrics_from_series: returns and benchmark must be Series")
    bench = benchmark.reindex(returns.index)
    mask = (returns.notna() & bench.notna()).to_numpy()
    r = returns.to_numpy()[mask]
    b = bench.to_numpy()[mask]
    if r.size < 2:
        return None
    return relative_metrics(r, b, bars_per_year=bars_per_year, rf=rf)


def factor_attribution(
    returns: ReturnsLike,
    factors: Mapping[str, ReturnsLike],
    *,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> FactorAttribution:
    """OLS attribution of ``returns`` onto the named ``factors``.

    Regresses ``(returns - rf)`` on a design matrix ``[1, f1, …, fK]`` (intercept + one column per
    factor, in the order of ``factors``). Factors are taken AS GIVEN — for asset-pricing use they
    are typically excess / long-short series, so ``rf`` is subtracted from the LHS only; for a single
    benchmark factor at ``rf=0`` the intercept and slope equal :func:`relative_metrics`' ``alpha``
    and ``beta`` (the OLS slope of ``r`` on ``[1, b]`` is ``cov(r, b) / var(b)``).

    Raises ``ValueError`` on an empty factor set, a factor whose length differs from ``returns``, or
    fewer than ``K + 2`` observations (no residual degrees of freedom). A rank-deficient design (a
    constant or collinear factor) yields ``NaN`` alpha / betas / t-stats rather than raising — the
    attribution is undefined there, not erroneous.
    """
    if not factors:
        raise ValueError("factor_attribution: need at least one factor")
    y = _as_1d_array(returns).astype(np.float64)
    names = list(factors)
    cols: list[np.ndarray] = []
    for nm in names:
        f = _as_1d_array(factors[nm]).astype(np.float64)
        if f.shape != y.shape:
            raise ValueError(
                f"factor_attribution: factor {nm!r} must align with returns; got {f.shape} vs {y.shape}"
            )
        cols.append(f)
    k = len(names)
    n = y.size
    if n < k + 2:
        raise ValueError(f"factor_attribution: need >= K+2 = {k + 2} observations; got {n}")

    y_excess = y - rf
    design = np.column_stack([np.ones(n), *cols])  # (n, K+1): intercept + factors
    dof = n - (k + 1)
    nan_map = {nm: float("nan") for nm in names}

    # Rank-deficient design (a constant or collinear factor) → attribution undefined.
    if int(np.linalg.matrix_rank(design)) < k + 1:
        return FactorAttribution(
            alpha=float("nan"),
            alpha_tstat=float("nan"),
            betas=dict(nan_map),
            beta_tstats=dict(nan_map),
            r_squared=float("nan"),
            residual_volatility=float("nan"),
            n_obs=n,
        )

    coef, _, _, _ = np.linalg.lstsq(design, y_excess, rcond=None)
    resid = y_excess - design @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((y_excess - y_excess.mean()) ** 2).sum())
    r_squared = float("nan") if ss_tot <= (_MIN_REL_STD**2) else 1.0 - ss_res / ss_tot

    sigma2 = ss_res / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.clip(sigma2 * np.diag(xtx_inv), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(se > 0.0, coef / se, np.nan)

    return FactorAttribution(
        alpha=float(coef[0]) * bars_per_year,
        alpha_tstat=float(tstats[0]),
        betas={nm: float(coef[i + 1]) for i, nm in enumerate(names)},
        beta_tstats={nm: float(tstats[i + 1]) for i, nm in enumerate(names)},
        r_squared=r_squared,
        residual_volatility=float(np.sqrt(sigma2) * np.sqrt(bars_per_year)),
        n_obs=n,
    )


def factor_attribution_from_frame(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    bars_per_year: int = 252,
    rf: float = 0.0,
) -> FactorAttribution | None:
    """Align a factor DataFrame to ``returns`` and compute :class:`FactorAttribution`.

    Reindexes ``factors`` onto ``returns.index``, drops timestamps where ``returns`` or any factor
    column is non-finite, and calls :func:`factor_attribution` with the surviving columns (order =
    ``factors.columns``). Returns ``None`` when fewer than ``K + 2`` aligned observations remain —
    the DataFrame counterpart to :func:`relative_metrics_from_series`.
    """
    if not isinstance(returns, pd.Series) or not isinstance(factors, pd.DataFrame):
        raise TypeError(
            "factor_attribution_from_frame: returns must be a Series, factors a DataFrame"
        )
    k = factors.shape[1]
    if k == 0:
        raise ValueError("factor_attribution_from_frame: need at least one factor column")
    aligned = factors.reindex(returns.index)
    finite_factors = np.isfinite(aligned.to_numpy(dtype=np.float64)).all(axis=1)
    mask = returns.notna().to_numpy() & finite_factors
    if int(mask.sum()) < k + 2:
        return None
    r = returns.to_numpy()[mask]
    cols = {str(c): aligned[c].to_numpy()[mask] for c in factors.columns}
    return factor_attribution(r, cols, bars_per_year=bars_per_year, rf=rf)


def rolling_volatility(returns: pd.Series, window: int = 63, bars_per_year: int = 252) -> pd.Series:
    """Rolling annualised volatility (``ddof=1``) of a return Series."""
    if not isinstance(returns, pd.Series):
        raise TypeError(
            f"rolling_volatility: returns must be a Series; got {type(returns).__name__}"
        )
    return (returns.rolling(window).std(ddof=1) * np.sqrt(bars_per_year)).rename("rolling_vol")


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """Rolling OLS beta to the benchmark: rolling ``cov(returns, benchmark) / var(benchmark)``.

    Both inputs are aligned return Series (pandas aligns on the index). Returns a Series named
    ``rolling_beta`` with a ``window - 1`` NaN warmup; windows with ~zero benchmark variance are
    ``NaN``.
    """
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        raise TypeError("rolling_beta: returns and benchmark must be Series")
    cov = returns.rolling(window).cov(benchmark)
    std = benchmark.rolling(window).std()  # ddof=1; var = std² (avoids the mis-typed .var())
    return (cov / (std * std).replace(0.0, np.nan)).rename("rolling_beta")


def rolling_information_ratio(
    returns: pd.Series, benchmark: pd.Series, window: int = 63, bars_per_year: int = 252
) -> pd.Series:
    """Rolling annualised information ratio: rolling ``mean(active) / std(active) × √P``.

    ``active = returns - benchmark`` (pandas aligns on the index). Returns a Series named
    ``rolling_information_ratio`` with a ``window - 1`` NaN warmup; windows with ~zero tracking
    error are ``NaN``.
    """
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        raise TypeError("rolling_information_ratio: returns and benchmark must be Series")
    active = returns - benchmark
    mu = active.rolling(window).mean()
    sd = active.rolling(window).std(ddof=1)
    return (mu / sd.replace(0.0, np.nan) * np.sqrt(bars_per_year)).rename(
        "rolling_information_ratio"
    )


def contribution_to_return(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Per-asset contribution to portfolio return: ``Σ_t w_{i,t-1} · r_{i,t}``.

    Weights are lagged one period (the book held into t earns t's return). Returns a Series
    indexed by ticker; the sum approximates the cumulative arithmetic portfolio return.
    """
    if not isinstance(weights, pd.DataFrame) or not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("contribution_to_return: weights and asset_returns must be DataFrames")
    cols = weights.columns.intersection(asset_returns.columns)
    w_lag = weights[cols].shift(1)
    r = asset_returns[cols].reindex(index=weights.index)
    return (w_lag * r).sum(axis=0).rename("contribution")


def tail_metrics(returns: ReturnsLike, *, level: float = 0.95) -> TailMetrics:
    """Tail / distribution risk metrics for a simple-return series.

    Historical (non-parametric) VaR and CVaR at confidence ``level`` plus the
    distribution shape (skew, excess kurtosis) and hit-rate / best / worst. ``var`` is
    the ``1 - level`` empirical return quantile (signed: a loss is negative); ``cvar``
    is the mean of the returns at or below that quantile (expected shortfall). ``skew``
    and ``excess_kurtosis`` are the bias-corrected sample estimators (G1 / G2),
    matching ``scipy.stats`` with ``bias=False`` and pandas' ``.skew()`` / ``.kurt()``.

    Robust to short and degenerate input: the shape statistics return ``NaN`` rather
    than raising when undefined (< 3 / < 4 observations, or dispersion within numerical
    noise). Raises ``ValueError`` only on empty input or ``level`` outside the open
    interval ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"tail_metrics: level must be in (0, 1); got {level}")
    x = _as_1d_array(returns)
    if x.size == 0:
        raise ValueError("tail_metrics: need at least 1 observation; got 0")

    var = float(np.quantile(x, 1.0 - level))
    tail = x[x <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var

    n = x.size
    d = x - x.mean()
    m2 = float(np.mean(d * d))
    scale = max(float(np.median(np.abs(x))), 1.0)
    degenerate = (not np.isfinite(m2)) or m2 < (_MIN_REL_STD * scale) ** 2

    if degenerate or n < 3:
        skew = float("nan")
    else:
        g1 = float(np.mean(d**3)) / m2**1.5
        skew = float(g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0))

    if degenerate or n < 4:
        excess_kurtosis = float("nan")
    else:
        g2 = float(np.mean(d**4)) / (m2 * m2) - 3.0
        excess_kurtosis = float((n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0))

    return TailMetrics(
        level=float(level),
        var=var,
        cvar=cvar,
        skew=skew,
        excess_kurtosis=excess_kurtosis,
        hit_rate=float(np.mean(x > 0.0)),
        best=float(x.max()),
        worst=float(x.min()),
    )


_DRAWDOWN_TABLE_COLUMNS = (
    "magnitude",
    "peak",
    "trough",
    "recovery",
    "depth_periods",
    "recovery_periods",
)


def drawdown_table(returns: ReturnsLike, *, top_n: int = 5) -> pd.DataFrame:
    """Top-``N`` peak-to-trough drawdown episodes, ranked by magnitude (worst first).

    Walks the compounded equity curve, segments it into maximal underwater episodes
    (equity strictly below the running high-water mark), and reports for each: the
    ``peak`` it fell from, the ``trough``, the ``recovery`` point (first return to the
    prior peak, or ``None`` / ``NaN`` if still underwater at series end), and the period
    counts ``depth_periods`` (peak→trough) and ``recovery_periods`` (trough→recovery).

    Index columns hold the series' index labels for a ``pd.Series`` input (typically
    ``pd.Timestamp``) and positional ``int`` for an ``np.ndarray``. The worst row's
    ``magnitude`` and ``trough`` agree with :func:`max_drawdown` (the ``peak`` label can
    differ on a plateau — :func:`max_drawdown` takes the first high, this takes the last
    before the drop). Returns an empty frame (the canonical columns, zero rows) when no
    drawdown is observed.
    """
    if top_n < 1:
        raise ValueError(f"drawdown_table: top_n must be >= 1; got {top_n}")
    x = _as_1d_array(returns)

    def _resolve(pos: int | None) -> pd.Timestamp | int | None:
        if pos is None:
            return None
        if isinstance(returns, pd.Series):
            return cast("pd.Timestamp | int", returns.index[pos])
        return pos

    rows: list[dict[str, object]] = []
    if x.size > 0:
        equity = np.cumprod(1.0 + x)
        running_peak = np.maximum.accumulate(equity)
        underwater = equity < running_peak
        n = x.size
        i = 0
        while i < n:
            if not underwater[i]:
                i += 1
                continue
            start = i
            while i < n and underwater[i]:
                i += 1
            end = i - 1  # inclusive last underwater index
            peak_level = float(running_peak[start])
            # last index before the drop that achieved the high-water mark
            pre = np.nonzero(equity[:start] == peak_level)[0]
            peak_pos = int(pre[-1]) if pre.size > 0 else start - 1
            trough_pos = start + int(np.argmin(equity[start : end + 1]))
            recovery_pos = end + 1 if end + 1 < n else None
            rows.append(
                {
                    "magnitude": float(equity[trough_pos] / peak_level - 1.0),
                    "peak": _resolve(peak_pos),
                    "trough": _resolve(trough_pos),
                    "recovery": _resolve(recovery_pos),
                    "depth_periods": trough_pos - peak_pos,
                    "recovery_periods": (
                        float(recovery_pos - trough_pos)
                        if recovery_pos is not None
                        else float("nan")
                    ),
                }
            )

    table = pd.DataFrame(rows).reindex(columns=list(_DRAWDOWN_TABLE_COLUMNS))
    if table.empty:
        return table
    return table.sort_values("magnitude", kind="stable").head(top_n).reset_index(drop=True)


__all__ = [
    "DrawdownEvent",
    "FactorAttribution",
    "PerformanceReport",
    "RelativeMetrics",
    "TailMetrics",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "compute_performance",
    "contribution_to_return",
    "drawdown_table",
    "factor_attribution",
    "factor_attribution_from_frame",
    "information_ratio",
    "max_drawdown",
    "relative_metrics",
    "relative_metrics_from_series",
    "rolling_beta",
    "rolling_information_ratio",
    "rolling_volatility",
    "sortino_ratio",
    "tail_metrics",
    "turnover_from_weights",
]
