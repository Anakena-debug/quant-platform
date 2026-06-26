"""Cross-sectional alpha → NCO portfolio → Kelly sizing (S20 coordinator).

Composes per-name conformal alpha forecasts into a portfolio rebalanced
periodically via NCO (AFML §16.4) with explicit-k ward clustering and
continuous Kelly sizing (AFML §10) at the portfolio level.

Decoupled from per-name alpha generation: callers supply pre-computed
alpha signals (DataFrame with MultiIndex ``[date, ticker]`` and columns
``expected_return, lower, upper``) plus a daily returns panel; the strategy
handles cross-sectional cov estimation, NCO clustering, and Kelly-
fractionated sizing per rebalance date.

Default cluster-cardinality rule is ``k = ⌈√N_active⌉``, which the cov_n100
sweep showed delivers MV-baseline turnover (or better) while preserving
NCO diversification — the ONC silhouette-t default selects K too aggressively
at high N (≈36 clusters at N=100), inflating inter-cluster turnover.

Public API:
  - CSAlphaNCOConfig         — frozen dataclass of strategy parameters
  - CSAlphaNCOResult         — dataclass holding portfolio PnL + history
  - cs_alpha_nco_backtest()  — walk-forward backtest function (harness path)
  - CSAlphaNCO(Strategy)     — quantengine.strategies.base.Strategy subclass
                               wrapping the same composition for live
                               predict(MarketSnapshot) -> AlphaSignal calls

Read-only on quantcore — uses public APIs only:
  quantcore.covariance.shrinkage.ledoit_wolf_shrinkage
  quantcore.covariance.rmt.denoise_covariance
  quantstrat.portfolio.nco.nco_weights
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

from quantcore.covariance.rmt import denoise_covariance, detone_covariance
from quantcore.covariance.shrinkage import ledoit_wolf_shrinkage
from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal
from quantengine.strategies.base import Strategy
from quantstrat.portfolio.nco import nco_weights


CovEstimator = Literal["sample", "lw", "rmt"]
NClustersRule = Literal["sqrt_n", "fixed", "onc"]
ClusteringMethod = Literal["ward", "single", "complete", "average"]


@dataclass(frozen=True)
class CSAlphaNCOConfig:
    """Strategy configuration. All sizes are in units consistent with the
    panel_returns input — typically simple daily arithmetic returns.

    Defaults reflect the cov_n100 sweep finding: LW + ward + ⌈√N⌉ recovers
    NCO's AFML §16.4 turnover-reduction promise that the ONC default loses
    at high N.
    """

    # ── Covariance estimation ────────────────────────────────────
    cov_estimator: CovEstimator = "lw"
    cov_lookback_days: int = 252  # ~1 trading year
    # Detone (MLAM Ch.2 §2.6): strip the top market mode from the correlation the allocator
    # clusters on. Off by default (back-compat); the 2026-06-09 audit flags the un-detoned market
    # mode as the suspected driver of the live book's sector concentration. Applied in _fit_cov, so
    # both surfaces (backtest + predict) detone identically.
    detone: bool = False
    n_market_factors: int = 1

    # ── Clustering ───────────────────────────────────────────────
    n_clusters_rule: NClustersRule = "sqrt_n"
    n_clusters_fixed: int | None = None  # required iff rule == "fixed"
    clustering_method: ClusteringMethod = "ward"

    # ── Sizing — continuous Kelly under Markowitz/log-utility ────
    kelly_fraction: float = 0.5
    kelly_cap: float = 0.25  # ± cap on portfolio leverage

    # ── Rebalancing ──────────────────────────────────────────────
    rebalance_freq: str = "BMS"  # business-month start

    # ── Execution costs ──────────────────────────────────────────
    # One-way linear cost charged per unit of (two-way L1) turnover, in bps.
    # 0.0 (default) leaves the backtest GROSS — back-compat, and the only sane
    # value for cost-free plumbing tests. For an honest net verdict set this to
    # the measured standing-cost surface (s86 ≈ 6.35bps one-way). `daily_pnl`
    # stays gross; `daily_pnl_net = daily_pnl − cost_bps·1e-4·turnover`.
    cost_bps: float = 0.0

    # ── Signal freshness ─────────────────────────────────────────
    # Drop tickers whose most recent signal is older than this many
    # calendar days from the rebalance date. None = no staleness filter
    # (treat all historical signals as current — only sane for
    # back-of-the-envelope smoke tests on small N).
    max_signal_age_days: int | None = 60

    # ── Validation ───────────────────────────────────────────────
    # Below this many tickers with a fresh-and-valid signal at a
    # rebalance date, the strategy holds cash for that period.
    # Default 2 is a wiring smoke threshold — production callers on
    # large universes should raise this to ≥ ⌈√N⌉ to give the
    # clustering step enough names to form distinct clusters.
    min_active_tickers: int = 2

    def __post_init__(self) -> None:
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if not (0 < self.kelly_cap <= 5):
            raise ValueError(f"kelly_cap must be in (0, 5], got {self.kelly_cap}")
        if self.cov_lookback_days < 30:
            raise ValueError(
                f"cov_lookback_days must be ≥ 30 (sample-cov stability floor); "
                f"got {self.cov_lookback_days}"
            )
        if self.detone and self.n_market_factors < 1:
            raise ValueError(
                f"n_market_factors must be ≥ 1 when detone=True; got {self.n_market_factors}"
            )
        if self.n_clusters_rule == "fixed":
            if self.n_clusters_fixed is None or self.n_clusters_fixed < 1:
                raise ValueError(
                    "n_clusters_fixed must be a positive int when "
                    f"n_clusters_rule='fixed'; got {self.n_clusters_fixed!r}"
                )
        if self.min_active_tickers < 1:
            raise ValueError(f"min_active_tickers must be ≥ 1, got {self.min_active_tickers}")
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps must be ≥ 0, got {self.cost_bps}")
        if self.max_signal_age_days is not None and self.max_signal_age_days < 1:
            raise ValueError(
                f"max_signal_age_days must be ≥ 1 or None, got {self.max_signal_age_days}"
            )


@dataclass
class CSAlphaNCOResult:
    """Backtest result. ``daily_pnl`` is the date-indexed GROSS portfolio return
    series (one entry per panel_returns row); ``daily_pnl_net`` subtracts
    ``daily_costs`` (``config.cost_bps``·turnover, charged on rebalance dates —
    all zero when ``cost_bps`` is 0). ``weights_history`` and ``turnover_history``
    are indexed by rebalance date.
    """

    daily_pnl: pd.Series
    daily_pnl_net: pd.Series
    daily_costs: pd.Series  # date-indexed; nonzero only on rebalance dates
    weights_history: pd.DataFrame  # rebalance_date × ticker
    turnover_history: pd.Series  # rebalance_date → two-way L1 turnover
    rebalance_dates: pd.DatetimeIndex
    n_active_history: pd.Series  # rebalance_date → #tradeable tickers
    portfolio_kelly_history: pd.Series  # rebalance_date → applied leverage
    config: CSAlphaNCOConfig


def _resolve_n_clusters(rule: NClustersRule, n_active: int, fixed: int | None) -> int | None:
    """Return k for ``cluster_assets``/``nco_weights`` given the rule.

    ``onc`` returns None to defer to ONC's silhouette-t search.
    ``sqrt_n`` returns ⌈√N⌉ (capped at n_active for degenerate small N).
    ``fixed`` returns the configured int, capped at n_active.
    """
    if rule == "sqrt_n":
        return min(max(1, int(math.ceil(math.sqrt(n_active)))), n_active)
    if rule == "fixed":
        return min(int(fixed), n_active)  # type: ignore[arg-type]
    if rule == "onc":
        return None
    raise ValueError(f"unknown n_clusters_rule: {rule!r}")


def _fit_cov(
    estimator: CovEstimator,
    returns: np.ndarray,
    *,
    detone: bool = False,
    n_market_factors: int = 1,
) -> np.ndarray:
    if estimator == "sample":
        cov = np.cov(returns, rowvar=False, ddof=1)
    elif estimator == "lw":
        cov, _ = ledoit_wolf_shrinkage(returns)
    elif estimator == "rmt":
        cov = denoise_covariance(returns)
    else:
        raise ValueError(f"unknown cov_estimator: {estimator!r}")
    # Detone last (MLAM denoise→detone sequence). Guarded: detoning needs N > n_market_factors
    # eigenpairs to remove, so a degenerate small-N rebalance keeps the toned cov rather than raising
    # mid-backtest. detone_covariance maps back via the original vols, so variances are preserved.
    if detone and cov.shape[0] > n_market_factors:
        cov = detone_covariance(cov, n_market_factors=n_market_factors)
    return cov


def _portfolio_kelly_leverage(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Continuous-Kelly portfolio leverage: f* = (w'μ) / (w'Σw).

    Returns 0 on degenerate portfolio variance (zero or non-finite). Caller
    is responsible for clipping to a leverage cap.
    """
    mu_p = float(w @ mu)
    var_p = float(w @ cov @ w)
    if var_p <= 0 or not np.isfinite(var_p):
        return 0.0
    return mu_p / var_p


def _latest_signal_per_ticker(
    alpha_signals: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    min_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a per-ticker DataFrame holding each ticker's most recent signal
    with ``min_date ≤ date ≤ asof``.

    Forward-leak prevention: signals dated > asof are dropped, never used.
    Staleness filter: if ``min_date`` is provided, tickers whose most-recent
    signal predates ``min_date`` are dropped (returned set excludes stale
    forecasts).
    """
    if alpha_signals.empty:
        return alpha_signals.iloc[0:0].copy()
    dates_lvl = alpha_signals.index.get_level_values("date")
    avail = alpha_signals[dates_lvl <= asof]
    if min_date is not None:
        avail_dates = avail.index.get_level_values("date")
        avail = avail[avail_dates >= min_date]
    if avail.empty:
        return avail.copy()
    return (
        avail.reset_index()
        .sort_values(["ticker", "date"])
        .drop_duplicates("ticker", keep="last")
        .set_index("ticker")
    )


def cs_alpha_nco_backtest(
    *,
    alpha_signals: pd.DataFrame,
    panel_returns: pd.DataFrame,
    config: CSAlphaNCOConfig,
) -> CSAlphaNCOResult:
    """Walk-forward backtest of the cross-sectional alpha → NCO → Kelly chain.

    Parameters
    ----------
    alpha_signals : DataFrame
        MultiIndex ``[date, ticker]``, columns must include ``expected_return,
        lower, upper``. Each row is one published per-name forecast. At each
        rebalance date the strategy uses the most recent signal per ticker
        with ``date ≤ rebalance_date - 1`` (forward-leak prevention).
    panel_returns : DataFrame
        DatetimeIndex (sorted), ticker columns, daily arithmetic returns.
    config : CSAlphaNCOConfig

    Convention: positions established at rebalance date ``rd`` earn returns
    on the same day onward (returns at ``rd`` are realised intraday).
    """
    # ── Input validation ──────────────────────────────────────────
    if not isinstance(alpha_signals.index, pd.MultiIndex):
        raise TypeError("alpha_signals must have a MultiIndex with names ['date', 'ticker']")
    if list(alpha_signals.index.names) != ["date", "ticker"]:
        raise ValueError(
            f"alpha_signals.index.names must be ['date', 'ticker'], "
            f"got {list(alpha_signals.index.names)}"
        )
    required_cols = {"expected_return", "lower", "upper"}
    missing = required_cols - set(alpha_signals.columns)
    if missing:
        raise ValueError(f"alpha_signals missing required columns: {sorted(missing)}")
    if not isinstance(panel_returns.index, pd.DatetimeIndex):
        raise TypeError("panel_returns must have a DatetimeIndex")

    panel_returns = panel_returns.sort_index()
    tickers_all = list(panel_returns.columns)
    avail_dates = panel_returns.index

    # ── Snap rebalance schedule onto available trading days ───────
    rebal_calendar = pd.date_range(avail_dates.min(), avail_dates.max(), freq=config.rebalance_freq)
    snapped: list[pd.Timestamp] = []
    for rd in rebal_calendar:
        nxt = avail_dates[avail_dates >= rd]
        if len(nxt) > 0:
            snapped.append(nxt[0])
    rebal_dates = pd.DatetimeIndex(sorted(set(snapped)))
    if len(rebal_dates) < 2:
        raise ValueError(
            f"need ≥ 2 rebalance dates; got {len(rebal_dates)} with "
            f"rebalance_freq={config.rebalance_freq!r}"
        )

    # ── Walk-forward state ────────────────────────────────────────
    positions = pd.Series(0.0, index=tickers_all, dtype=float)
    daily_pnl = pd.Series(0.0, index=avail_dates, dtype=float)
    weights_rows: list[pd.Series] = []
    turnover_rows: list[float] = []
    n_active_rows: list[int] = []
    kelly_rows: list[float] = []

    for i, rd in enumerate(rebal_dates):
        loc = avail_dates.get_loc(rd)

        # cov window: [loc - lookback, loc) — strictly before rd to prevent
        # forward leak via the cov estimator
        cov_start = max(0, loc - config.cov_lookback_days)
        cov_window = panel_returns.iloc[cov_start:loc]
        if len(cov_window) < 30:
            # too short for reliable cov; hold prior positions
            weights_rows.append(positions.copy())
            turnover_rows.append(0.0)
            n_active_rows.append(0)
            kelly_rows.append(0.0)
            _accumulate_pnl(daily_pnl, panel_returns, positions, rebal_dates, i)
            continue

        # signals: only those dated strictly before rd (forward-leak prevention),
        # and within the staleness window if configured.
        min_date = (
            rd - pd.Timedelta(days=config.max_signal_age_days)
            if config.max_signal_age_days is not None
            else None
        )
        latest = _latest_signal_per_ticker(
            alpha_signals,
            asof=rd - pd.Timedelta(days=1),
            min_date=min_date,
        )
        valid_cols = cov_window.columns[~cov_window.isna().any()].tolist()
        active = [t for t in latest.index if t in valid_cols]

        if len(active) < config.min_active_tickers:
            new_positions = pd.Series(0.0, index=tickers_all, dtype=float)
            turnover = float((new_positions - positions).abs().sum())
            positions = new_positions
            weights_rows.append(positions.copy())
            turnover_rows.append(turnover)
            n_active_rows.append(len(active))
            kelly_rows.append(0.0)
            _accumulate_pnl(daily_pnl, panel_returns, positions, rebal_dates, i)
            continue

        rets_active = cov_window[active].to_numpy()
        cov = _fit_cov(
            config.cov_estimator,
            rets_active,
            detone=config.detone,
            n_market_factors=config.n_market_factors,
        )
        mu_active = latest.loc[active, "expected_return"].to_numpy(dtype=float)

        k = _resolve_n_clusters(config.n_clusters_rule, len(active), config.n_clusters_fixed)
        if k is None:
            w_active = nco_weights(cov, mu=mu_active)
        else:
            w_active = nco_weights(
                cov,
                mu=mu_active,
                n_clusters=k,
                clustering_method=config.clustering_method,
            )

        f_star = _portfolio_kelly_leverage(w_active, mu_active, cov)
        # Floor leverage at 0 (audit s49): nco_weights returns a sum-to-1,
        # long-biased mu-tilted vector. A negative f* (aggregate edge w'mu < 0)
        # multiplied through would INVERT the whole cross-section into a
        # nonsensical net-short book rather than express a controlled short.
        # On negative aggregate edge the strategy holds cash (leverage 0); any
        # intended shorting must come from signed per-name weights inside
        # nco_weights/mu, not a global negative leverage scalar. Kept
        # byte-identical to the predict() site so the two surfaces cannot drift.
        leverage = float(np.clip(config.kelly_fraction * f_star, 0.0, config.kelly_cap))

        new_positions = pd.Series(0.0, index=tickers_all, dtype=float)
        new_positions.loc[active] = leverage * w_active

        turnover = float((new_positions - positions).abs().sum())
        positions = new_positions
        weights_rows.append(positions.copy())
        turnover_rows.append(turnover)
        n_active_rows.append(len(active))
        kelly_rows.append(leverage)

        _accumulate_pnl(daily_pnl, panel_returns, positions, rebal_dates, i)

    turnover_series = pd.Series(turnover_rows, index=rebal_dates, name="turnover")
    # Linear execution cost: one-way cost_bps charged per unit of (two-way L1)
    # turnover, booked on the rebalance date. daily_pnl stays gross.
    daily_costs = pd.Series(0.0, index=avail_dates, dtype=float)
    if config.cost_bps > 0 and len(rebal_dates) > 0:
        daily_costs.loc[rebal_dates] = config.cost_bps * 1e-4 * turnover_series.to_numpy()
    daily_pnl_net = daily_pnl - daily_costs

    return CSAlphaNCOResult(
        daily_pnl=daily_pnl,
        daily_pnl_net=daily_pnl_net,
        daily_costs=daily_costs,
        weights_history=pd.DataFrame(weights_rows, index=rebal_dates),
        turnover_history=turnover_series,
        rebalance_dates=rebal_dates,
        n_active_history=pd.Series(n_active_rows, index=rebal_dates, name="n_active"),
        portfolio_kelly_history=pd.Series(kelly_rows, index=rebal_dates, name="portfolio_kelly"),
        config=config,
    )


def _accumulate_pnl(
    daily_pnl: pd.Series,
    panel_returns: pd.DataFrame,
    positions: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    i: int,
) -> None:
    """Apply current positions to returns from rebal_dates[i] up to (but not
    including) rebal_dates[i+1]. Mutates ``daily_pnl`` in place."""
    rd = rebal_dates[i]
    avail = panel_returns.index
    loc = avail.get_loc(rd)
    if i + 1 < len(rebal_dates):
        end_loc = avail.get_loc(rebal_dates[i + 1])
    else:
        end_loc = len(avail)
    slc = panel_returns.iloc[loc:end_loc]
    contrib = (slc * positions).sum(axis=1)
    daily_pnl.loc[contrib.index] = contrib.values


# ─────────────────────────────────────────────────────────────────────
# CSAlphaNCO — quantengine.strategies.base.Strategy subclass
# ─────────────────────────────────────────────────────────────────────


class CSAlphaNCO(Strategy):
    """Strategy ABC adapter for the cs_alpha_nco coordinator.

    Wraps the same per-rebalance composition that ``cs_alpha_nco_backtest``
    walks across history, but exposes it as a single ``predict(market) ->
    AlphaSignal`` call against a held panel of pre-computed alpha signals
    and returns history.

    Construction
    ------------
    The strategy is composition-only: it does not fit alpha forecasters
    (that's quantcore). The caller supplies:

        alpha_signals : MultiIndex ``[date, ticker]`` DataFrame with columns
                        ``expected_return, lower, upper`` — typically the
                        output of an ACI-calibrated quantcore pipeline.
        panel_returns : DatetimeIndex × ticker DataFrame of daily returns,
                        used solely for cov-window estimation.
        config        : CSAlphaNCOConfig
        miscoverage   : conformal alpha rate associated with the supplied
                        ``lower``/``upper`` intervals (passed through to the
                        emitted AlphaSignal). Defaults to 0.10 (the
                        quantcore default).

    predict(market)
    ---------------
    Treats ``market.timestamp`` as the rebalance "as-of" date and
    ``market.tickers`` as the candidate universe. The active subset is
    formed by intersecting the universe with tickers that have a fresh
    signal (per ``max_signal_age_days``) and a complete cov window. NCO
    weights + portfolio Kelly leverage are computed exactly as in
    ``cs_alpha_nco_backtest``; the result is packaged as an AlphaSignal
    whose ``kelly_weights`` field carries the leverage-scaled NCO weights
    over ``market.tickers`` (zeros for inactive names).

    The ``update()`` hook is the ABC default no-op: this strategy does not
    own a conformal calibrator (that lives upstream in the quantcore
    pipeline that produced ``alpha_signals``).
    """

    def __init__(
        self,
        *,
        alpha_signals: pd.DataFrame,
        panel_returns: pd.DataFrame,
        config: CSAlphaNCOConfig,
        miscoverage: float = 0.10,
    ) -> None:
        if not isinstance(alpha_signals.index, pd.MultiIndex):
            raise TypeError("alpha_signals must have a MultiIndex with names ['date', 'ticker']")
        if list(alpha_signals.index.names) != ["date", "ticker"]:
            raise ValueError(
                f"alpha_signals.index.names must be ['date', 'ticker'], "
                f"got {list(alpha_signals.index.names)}"
            )
        required_cols = {"expected_return", "lower", "upper"}
        missing = required_cols - set(alpha_signals.columns)
        if missing:
            raise ValueError(f"alpha_signals missing required columns: {sorted(missing)}")
        if not isinstance(panel_returns.index, pd.DatetimeIndex):
            raise TypeError("panel_returns must have a DatetimeIndex")
        if not (0.0 < miscoverage < 1.0):
            raise ValueError(f"miscoverage must be in (0, 1); got {miscoverage}")

        self._alpha_signals: pd.DataFrame = alpha_signals.sort_index()
        self._panel_returns: pd.DataFrame = panel_returns.sort_index()
        self._config: CSAlphaNCOConfig = config
        self._miscoverage: float = float(miscoverage)

    @property
    def config(self) -> CSAlphaNCOConfig:
        return self._config

    def predict(self, market: MarketSnapshot) -> AlphaSignal:
        """Compose alphas → cov → NCO → Kelly into an AlphaSignal.

        Inactive tickers (universe ∖ fresh-signal-and-cov) get zero
        ``kelly_weights`` and a zero-centred wide interval so
        ``AlphaSignal.tradeable`` is False for them.
        """
        cfg = self._config
        tickers = list(market.tickers)
        n = len(tickers)
        # pandas-stubs widens ``Timestamp(str)`` to ``Timestamp | NaTType``
        # even when the string is concrete; cast to the runtime-true type.
        asof: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp(market.timestamp))

        # Cov window: strictly before asof
        avail = self._panel_returns.index
        loc_arr = avail[avail < asof]
        loc = len(loc_arr)
        cov_start = max(0, loc - cfg.cov_lookback_days)
        cov_window = self._panel_returns.iloc[cov_start:loc]

        # Defaults (no-trade outputs) for the universe
        expected_return = np.zeros(n, dtype=np.float64)
        lower = np.full(n, -1e-9, dtype=np.float64)
        upper = np.full(n, 1e-9, dtype=np.float64)
        kelly_weights = np.zeros(n, dtype=np.float64)

        if len(cov_window) < 30:
            return AlphaSignal(
                tickers=tuple(tickers),
                expected_return=expected_return,
                lower=lower,
                upper=upper,
                alpha=self._miscoverage,
                kelly_weights=kelly_weights,
                timestamp=str(market.timestamp),
                metadata={"reason": "cov_window_too_short", "n_active": 0},
            )

        # Latest signal per ticker, strictly before asof. Casts because
        # pandas-stubs widen ``Timestamp - Timedelta`` to
        # ``Timestamp | NaTType`` even when the operand is concrete.
        min_date: pd.Timestamp | None = (
            cast(pd.Timestamp, asof - pd.Timedelta(days=cfg.max_signal_age_days))
            if cfg.max_signal_age_days is not None
            else None
        )
        latest = _latest_signal_per_ticker(
            self._alpha_signals,
            asof=cast(pd.Timestamp, asof - pd.Timedelta(days=1)),
            min_date=min_date,
        )

        # Active subset: in universe AND has fresh signal AND cov column has no NaN
        valid_cov_cols = set(cov_window.columns[~cov_window.isna().any()].tolist())
        active = [t for t in tickers if t in latest.index and t in valid_cov_cols]

        # Populate ticker-aligned mu/lower/upper from the latest signal where present
        # (even tickers we won't trade — the AlphaSignal still carries the forecast).
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        for t in latest.index:
            if t in ticker_to_idx:
                i = ticker_to_idx[t]
                expected_return[i] = float(latest.loc[t, "expected_return"])
                lo = float(latest.loc[t, "lower"])
                hi = float(latest.loc[t, "upper"])
                if lo > hi:
                    lo, hi = hi, lo
                lower[i] = lo
                upper[i] = hi

        if len(active) < cfg.min_active_tickers:
            return AlphaSignal(
                tickers=tuple(tickers),
                expected_return=expected_return,
                lower=lower,
                upper=upper,
                alpha=self._miscoverage,
                kelly_weights=kelly_weights,
                timestamp=str(market.timestamp),
                metadata={"reason": "insufficient_active_tickers", "n_active": len(active)},
            )

        rets_active = cov_window[active].to_numpy()
        cov = _fit_cov(
            cfg.cov_estimator,
            rets_active,
            detone=cfg.detone,
            n_market_factors=cfg.n_market_factors,
        )
        mu_active = latest.loc[active, "expected_return"].to_numpy(dtype=float)

        k = _resolve_n_clusters(cfg.n_clusters_rule, len(active), cfg.n_clusters_fixed)
        if k is None:
            w_active = nco_weights(cov, mu=mu_active)
        else:
            w_active = nco_weights(
                cov,
                mu=mu_active,
                n_clusters=k,
                clustering_method=cfg.clustering_method,
            )

        f_star = _portfolio_kelly_leverage(w_active, mu_active, cov)
        # Floor leverage at 0 — see cs_alpha_nco_backtest for rationale. Kept
        # byte-identical to the backtest site so predict() cannot diverge from
        # the validated backtest (pinned by test_predict_matches_backtest_*).
        leverage = float(np.clip(cfg.kelly_fraction * f_star, 0.0, cfg.kelly_cap))
        for j, t in enumerate(active):
            kelly_weights[ticker_to_idx[t]] = leverage * float(w_active[j])

        return AlphaSignal(
            tickers=tuple(tickers),
            expected_return=expected_return,
            lower=lower,
            upper=upper,
            alpha=self._miscoverage,
            kelly_weights=kelly_weights,
            timestamp=str(market.timestamp),
            metadata={
                "n_active": len(active),
                "portfolio_kelly": leverage,
                "cov_estimator": cfg.cov_estimator,
                "n_clusters": k if k is not None else "onc",
            },
        )
