"""S29 — tradeability semantics + sensitivity sweep harness.

Parameterizes the existing CSAlphaNCO + conformal stack over a
150-cell grid and logs each cell to a local MLflow experiment
``s29-tradeability-sensitivity``.

Sweep axes (B4-amended grid; cell count 5 × 5 × 3 × 2 = 150):

    coverage_level   : 0.50, 0.65, 0.80, 0.90, 0.99
    active_threshold : 1, 3, 5, 10, 20    (maps to CSAlphaNCOConfig.min_active_tickers)
    alpha_spec       : existing_cs_alpha_nco | random_gaussian | momentum_12_1
    universe         : DJ30 | SP500

Two invariants the ADR
makes load-bearing on this harness:

  1. ``n_signal_tradeable`` and ``n_orders`` are computed and logged
     **independently**. The harness does NOT compose them anywhere
     (no ``signal_tradeable ∧ strategy_active`` at run time). The
     post-hoc conjunction is the result-table renderer's job, never
     the harness's.

  2. The harness does NOT import or construct ``AlphaSignal``. It
     operates on raw numpy arrays plus the ``alpha_signals`` DataFrame
     that ``cs_alpha_nco_backtest`` consumes — that DataFrame has
     columns ``expected_return, lower, upper`` and a MultiIndex
     ``[date, ticker]``, which is sufficient for the strategy
     contract.

Per-spec ``random_seed`` semantics (B3 amendment):

    existing_cs_alpha_nco → seeds SplitConformalRegressor.random_state
    random_gaussian       → seeds the per-cell N(0, σ²) draws
    momentum_12_1         → unused (deterministic; logged for tag parity)

The ``random_gaussian`` σ is pinned ex-ante (B2 amendment) to the
cross-sectional std of 1-day forward log returns over the training
window of the central rebalance date on DJ30. Computed once via
``_compute_sigma_random_gaussian()`` and cached; recomputable for
audit.

Degenerate-cell convention (B1 amendment): cells where
``n_orders == 0`` log

    SR_active = NaN     (not 0.0)
    DSR       = NaN     (not 0.0)
    turnover  = 0.0

and carry the tag ``degenerate_no_orders = "true"``. NaN-valued
metrics are NOT logged to MLflow (MLflow rejects non-finite metric
values); they appear in the CSV only.

This module mutates no source code under ``quantcore/src/``,
``quantengine/src/``, ``quantstrat/src/``, or ``quantdata/``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Existing surfaces (read-only; not mutated by S29).
from quantcore.uncertainty.conformal.timeseries_safe import TimeSeriesConformal
from quantcore.validation.stats import deflated_sharpe_ratio, sharpe_ratio_stats
from quantstrat.strategies.cs_alpha_nco import (
    CSAlphaNCOConfig,
    CSAlphaNCOResult,
    cs_alpha_nco_backtest,
)

logger = logging.getLogger("s29.tradeability_sweep")


# ────────────────────────────────────────────────────────────────────
# §1. Sweep identity + grid (B4 amendment)
# ────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME: str = "s29-tradeability-sensitivity"
SPRINT: str = "s29"

COVERAGE_LEVELS: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90, 0.99)
ACTIVE_THRESHOLDS: tuple[int, ...] = (1, 3, 5, 10, 20)
ALPHA_SPECS: tuple[str, ...] = (
    "existing_cs_alpha_nco",
    "random_gaussian",
    "momentum_12_1",
)
UNIVERSES: tuple[str, ...] = ("DJ30", "SP500")
RANDOM_SEED: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class SweepCell:
    """One cell of the (coverage × active × spec × universe) grid."""

    coverage_level: float
    active_threshold: int
    alpha_spec: str
    universe: str
    random_seed: int

    @property
    def conformal_alpha(self) -> float:
        """Miscoverage rate consumed by SplitConformalRegressor."""
        return 1.0 - self.coverage_level

    @property
    def tag_key(self) -> tuple[float, int, str, str, int]:
        """Tuple used for resumption-by-tag deduplication."""
        return (
            self.coverage_level,
            self.active_threshold,
            self.alpha_spec,
            self.universe,
            self.random_seed,
        )


def _enumerate_grid() -> tuple[SweepCell, ...]:
    return tuple(
        SweepCell(
            coverage_level=c,
            active_threshold=a,
            alpha_spec=spec,
            universe=u,
            random_seed=RANDOM_SEED,
        )
        for c in COVERAGE_LEVELS
        for a in ACTIVE_THRESHOLDS
        for spec in ALPHA_SPECS
        for u in UNIVERSES
    )


SWEEP_GRID: tuple[SweepCell, ...] = _enumerate_grid()
assert len(SWEEP_GRID) == 150, f"SWEEP_GRID has {len(SWEEP_GRID)} cells, expected 150"


# ────────────────────────────────────────────────────────────────────
# §2. Pinned chain config (S27 / S23a)
# ────────────────────────────────────────────────────────────────────

START_DATE: str = "2022-01-03"
END_DATE: str = "2024-12-31"
CALIBRATION_FRACTION: float = 0.25
RIDGE_ALPHA: float = 1.0
_MOM_LOOKBACK: int = 5
_VOL_LOOKBACK: int = 20

# 12-1 cross-sectional momentum: 12-month total return minus most-recent month.
_MOM_12M_DAYS: int = 252
_MOM_1M_DAYS: int = 21

PERIODS_PER_YEAR: int = 252

# Central rebalance date for B2 σ computation. Pinned literal so future
# git-blame can see what was used; if the BMS schedule over 2022-2024
# changes (it doesn't), this must be recomputed.
SIGMA_CENTRAL_REBALANCE_DATE: pd.Timestamp = pd.Timestamp("2023-07-03")  # pyright: ignore[reportAssignmentType]


# ────────────────────────────────────────────────────────────────────
# §3. Paths
# ────────────────────────────────────────────────────────────────────

QUANTSTRAT_DIR: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = QUANTSTRAT_DIR.parent
QUANTDATA_DIR: Path = REPO_ROOT / "quantdata"
DUCKDB_PATH: Path = QUANTDATA_DIR / "quant.duckdb"

RESULTS_DIR: Path = REPO_ROOT / "results" / "s29"
MLRUNS_DIR: Path = RESULTS_DIR / "mlruns"
CELL_PNL_DIR: Path = RESULTS_DIR / "cell_pnl"

DJ30_FILE: Path = QUANTDATA_DIR / "dowjones30_tickers.txt"
SP500_FILE: Path = QUANTDATA_DIR / "sp500_tickers.txt"


# ────────────────────────────────────────────────────────────────────
# §4. Panel loaders (DJ30 + SP500)
# ────────────────────────────────────────────────────────────────────


def _load_universe_tickers(universe: str) -> list[str]:
    """Parse the ticker file for the named universe, skipping comments."""
    if universe == "DJ30":
        path = DJ30_FILE
    elif universe == "SP500":
        path = SP500_FILE
    else:
        raise ValueError(f"unknown universe {universe!r} (expected DJ30 or SP500)")
    if not path.exists():
        raise FileNotFoundError(f"ticker file missing: {path}")
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_panel(universe: str) -> pd.DataFrame:
    """Long-form panel ``(ticker, session_date, price)`` for the named universe.

    DJ30 is strict (fails loud on any missing ticker). SP500 is permissive
    (tolerates missing tickers and reports the gap via the logger). In both
    cases, finite-data invariants (no NaN, no inf, positive prices) and
    PIT-ordering invariants (per-ticker strictly monotonic session_date)
    are enforced on the returned panel.
    """
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB catalog missing: {DUCKDB_PATH}")

    parsed = sorted(_load_universe_tickers(universe))

    with contextlib.chdir(QUANTDATA_DIR):
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            placeholders = ",".join(["?"] * len(parsed))
            raw = con.execute(
                f"SELECT ticker, date, close FROM MarketData "
                f"WHERE ticker IN ({placeholders}) "
                f"AND date >= ? AND date <= ? "
                f"ORDER BY ticker, date",
                [*parsed, START_DATE, END_DATE],
            ).df()
        finally:
            con.close()

    if raw.empty:
        raise ValueError(f"empty panel for universe {universe!r}")

    panel = raw.rename(columns={"date": "session_date", "close": "price"})
    panel["ticker"] = panel["ticker"].astype("object")
    panel["session_date"] = pd.to_datetime(panel["session_date"]).astype("datetime64[ns]")
    panel["price"] = panel["price"].astype("float64")
    panel = panel[["ticker", "session_date", "price"]]
    panel = panel.sort_values(by=["ticker", "session_date"], kind="mergesort").reset_index(  # pyright: ignore[reportCallIssue]
        drop=True
    )

    available = sorted(panel["ticker"].unique().tolist())
    missing = sorted(set(parsed) - set(available))

    if universe == "DJ30":
        if missing:
            raise ValueError(f"DJ30 universe gap: missing {missing} (strict mode for DJ30)")
    else:
        if missing:
            logger.info(
                "universe=%s: parsed=%d available=%d missing=%d (first 8: %s)",
                universe,
                len(parsed),
                len(available),
                len(missing),
                missing[:8],
            )

    _validate_panel_invariants(panel)
    return panel


def _validate_panel_invariants(panel: pd.DataFrame) -> None:
    expected_cols = {"ticker", "session_date", "price"}
    if set(panel.columns) != expected_cols:
        raise ValueError(f"panel columns {set(panel.columns)} != expected {expected_cols}")
    price = panel["price"].to_numpy()
    if not np.isfinite(price).all():
        raise ValueError("non-finite prices in panel")
    if (price <= 0).any():
        raise ValueError("non-positive prices in panel")
    dup_mask = panel.duplicated(subset=["ticker", "session_date"], keep=False)
    if dup_mask.any():
        raise ValueError(f"duplicate (ticker, session_date) pairs: {int(dup_mask.sum())} rows")
    for ticker, group in panel.groupby("ticker", sort=False):
        diffs = group["session_date"].diff().dropna()
        if (diffs <= pd.Timedelta(0)).any():
            raise ValueError(f"non-monotonic session_date for ticker {ticker!r}")


def pivot_to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide form: index session_date, columns ticker, values price."""
    wide = panel.pivot(index="session_date", columns="ticker", values="price")
    wide.columns.name = "ticker"
    wide.index.name = "session_date"
    return wide


# ────────────────────────────────────────────────────────────────────
# §5. SIGMA_RANDOM_GAUSSIAN (B2)
# ────────────────────────────────────────────────────────────────────

_SIGMA_CACHE: dict[str, float] = {}


def clear_sigma_cache() -> None:
    """Drop the cached σ. Use in audit/test paths that want a from-scratch
    re-derivation; the production sweep relies on the cached value being
    stable for the duration of a process.
    """
    _SIGMA_CACHE.clear()


def compute_sigma_random_gaussian() -> float:
    """B2 — σ pinned ex-ante to the cross-sectional std of 1-day fwd log
    returns over the training window of the central rebalance date on DJ30.

    Computed once from the raw DJ30 panel; cached. Deterministic given the
    same DuckDB content. NOT derived from any chain-output (predictions);
    NOT recomputed per cell.
    """
    if "sigma" in _SIGMA_CACHE:
        return _SIGMA_CACHE["sigma"]
    panel = load_panel("DJ30")
    wide = pivot_to_wide(panel).sort_index()
    log_close = np.log(wide)
    daily_ret = log_close.diff()
    fwd_log_return = daily_ret.shift(-1)
    train_mask = wide.index < SIGMA_CENTRAL_REBALANCE_DATE
    train_fwd = fwd_log_return.loc[train_mask]
    flat = train_fwd.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    sigma = float(np.std(flat, ddof=0))
    _SIGMA_CACHE["sigma"] = sigma
    return sigma


# ────────────────────────────────────────────────────────────────────
# §6. Features + labels
# ────────────────────────────────────────────────────────────────────


def _existing_features_labels(
    wide_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Existing chain features (mom5, z20, vol20) + 1-day forward log-return label.

    Matches S27 PR2's ``_features_labels`` semantics; reproduced here to
    keep the harness self-contained (the S27 module is in tests/research/
    and is forbidden from modification per S29 §forbidden_actions).
    """
    log_close = np.log(wide_closes)
    daily_ret = log_close.diff()
    mom5 = log_close.diff(_MOM_LOOKBACK)
    rolling_mean_20 = daily_ret.rolling(_VOL_LOOKBACK).mean()
    rolling_std_20 = daily_ret.rolling(_VOL_LOOKBACK).std(ddof=0)
    vol20 = rolling_std_20
    z20 = (daily_ret - rolling_mean_20) / rolling_std_20
    target = daily_ret.shift(-1)
    return mom5, z20, vol20, target


def _momentum_12_1(wide_closes: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum: 12-month log return minus most-recent
    month log return. Shape (dates × tickers).
    """
    log_close = np.log(wide_closes)
    twelve_m = log_close.diff(_MOM_12M_DAYS)
    one_m = log_close.diff(_MOM_1M_DAYS)
    return twelve_m - one_m


def _daily_returns_panel(wide_closes: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from wide closes; the panel cs_alpha_nco_backtest needs."""
    return wide_closes.pct_change().dropna(how="all")


# ────────────────────────────────────────────────────────────────────
# §7. Alpha-spec signal generators (walk-forward at every BMS rebalance)
# ────────────────────────────────────────────────────────────────────


def _bms_rebalance_dates(wide_closes: pd.DataFrame) -> pd.DatetimeIndex:
    """Snap a BMS calendar onto the panel's session-date index."""
    avail = wide_closes.index
    cal = pd.date_range(avail.min(), avail.max(), freq="BMS")
    snapped: list[pd.Timestamp] = []
    for rd in cal:
        nxt = avail[avail >= rd]
        if len(nxt) > 0:
            snapped.append(pd.Timestamp(nxt[0]))  # pyright: ignore[reportArgumentType]
    return pd.DatetimeIndex(sorted(set(snapped)))


def _signals_existing_cs_alpha_nco(
    wide_closes: pd.DataFrame, *, coverage_level: float, random_seed: int
) -> pd.DataFrame:
    """Walk-forward Ridge(α=1.0) + chronological-split TimeSeriesConformal.

    Per S27 PR2: features (mom5, z20, vol20), label = 1-day forward log return,
    cal_size = 0.25 (most-recent tail, NO shuffle). Conformal alpha =
    ``1 - coverage_level``. random_seed is unused here — the chronological
    split is deterministic (audit s49 replaced the shuffled
    SplitConformalRegressor); the param is retained in the signature for run-tag
    parity across the alpha-spec dispatch.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``[date, ticker]``, columns ``expected_return, lower, upper``.
        ``date`` is the rebalance date (signal is dated as of that day; the
        strategy's forward-leak prevention then consumes signals with
        ``date <= rd - 1``).
    """
    conformal_alpha = 1.0 - coverage_level
    mom5, z20, vol20, target = _existing_features_labels(wide_closes)
    rebal_dates = _bms_rebalance_dates(wide_closes)
    tickers: tuple[str, ...] = tuple(wide_closes.columns.tolist())

    rows: list[pd.DataFrame] = []
    for rd in rebal_dates:
        train_mask = wide_closes.index < rd
        m_tr = mom5.loc[train_mask]
        z_tr = z20.loc[train_mask]
        v_tr = vol20.loc[train_mask]
        y_tr = target.loc[train_mask]
        valid = m_tr.notna() & z_tr.notna() & v_tr.notna() & y_tr.notna()
        valid_flat = valid.to_numpy()
        if valid_flat.sum() < 50:
            # Too little training data; emit NaN row.
            rows.append(_emit_nan_signal_row(rd, tickers))
            continue
        X_train = np.column_stack(
            [
                m_tr.to_numpy()[valid_flat],
                z_tr.to_numpy()[valid_flat],
                v_tr.to_numpy()[valid_flat],
            ]
        ).astype(np.float64)
        y_train = y_tr.to_numpy()[valid_flat].astype(np.float64)

        # As-of feature row at the rebalance date.
        if rd not in mom5.index:
            rows.append(_emit_nan_signal_row(rd, tickers))
            continue
        x_rd = np.column_stack(
            [
                mom5.loc[rd].to_numpy(dtype=np.float64),
                z20.loc[rd].to_numpy(dtype=np.float64),
                vol20.loc[rd].to_numpy(dtype=np.float64),
            ]
        )
        # Where features are NaN at rd, the conformal predict will produce NaN.
        # Carry through unchanged; downstream `cs_alpha_nco_backtest._latest_signal_per_ticker`
        # drops the NaN row.
        # Chronological-split conformal (audit s49): the prior
        # SplitConformalRegressor permuted the time-ordered (X, y) before the
        # calibration split (regression.py self._rng.permutation), seeding
        # calibration with near-future neighbours of the test point and biasing
        # interval widths optimistically — a leak into the tradeability gate
        # that consumes these intervals for selection. TimeSeriesConformal takes
        # the most-recent contiguous tail as calibration with NO shuffle, so the
        # finite-sample coverage holds for serially-dependent returns. The split
        # is deterministic, so random_seed no longer affects this signal.
        cp = TimeSeriesConformal(
            Ridge(alpha=RIDGE_ALPHA),
            alpha=conformal_alpha,
            method="split",
        )
        cp.fit(X_train, y_train, cal_size=CALIBRATION_FRACTION)
        # Conformal predict requires finite inputs; mask NaNs to a sentinel.
        x_clean = np.where(np.isfinite(x_rd), x_rd, 0.0)
        point_arr, lower_arr, upper_arr = cp.predict(x_clean)
        point = np.asarray(point_arr, dtype=np.float64)
        lower = np.asarray(lower_arr, dtype=np.float64)
        upper = np.asarray(upper_arr, dtype=np.float64)
        # Restore NaN for tickers with NaN features at rd.
        nan_mask = ~np.isfinite(x_rd).all(axis=1)
        point[nan_mask] = np.nan
        lower[nan_mask] = np.nan
        upper[nan_mask] = np.nan

        df = pd.DataFrame(
            {"expected_return": point, "lower": lower, "upper": upper},
            index=pd.MultiIndex.from_product([[rd], tickers], names=["date", "ticker"]),
        )
        rows.append(df)

    return pd.concat(rows) if rows else _empty_signal_frame()


def _signals_random_gaussian(
    wide_closes: pd.DataFrame, *, coverage_level: float, random_seed: int
) -> pd.DataFrame:
    """Per-ticker N(0, σ²) draws with σ pinned ex-ante (B2). The interval
    is symmetric ±z(α/2)·σ; signal_tradeable computed downstream from
    ``(lower > 0) | (upper < 0)`` per the standard conformal geometry.

    B3 — random_seed seeds the per-cell N(0, σ²) draws via
    ``np.random.default_rng(random_seed)``.
    """
    from scipy.stats import norm

    sigma = compute_sigma_random_gaussian()
    conformal_alpha = 1.0 - coverage_level
    half_width = float(norm.ppf(1.0 - conformal_alpha / 2.0)) * sigma

    rebal_dates = _bms_rebalance_dates(wide_closes)
    tickers: tuple[str, ...] = tuple(wide_closes.columns.tolist())
    rng = np.random.default_rng(random_seed)

    rows: list[pd.DataFrame] = []
    for rd in rebal_dates:
        point = rng.normal(loc=0.0, scale=sigma, size=len(tickers))
        lower = point - half_width
        upper = point + half_width
        df = pd.DataFrame(
            {"expected_return": point, "lower": lower, "upper": upper},
            index=pd.MultiIndex.from_product([[rd], tickers], names=["date", "ticker"]),
        )
        rows.append(df)
    return pd.concat(rows)


def _signals_momentum_12_1(
    wide_closes: pd.DataFrame, *, coverage_level: float, random_seed: int
) -> pd.DataFrame:
    """Cross-sectional 12-1 momentum at every rebalance.

    Interval: symmetric ±z(α/2)·σ_xs where σ_xs is the cross-sectional std
    of the 12-1 signal at the as-of date. The interval shape matches the
    conformal geometry; signal_tradeable falls out of the conformal
    excludes-zero rule unchanged.

    B3 — random_seed is unused for this spec (deterministic given the
    panel); logged for tag parity only.
    """
    from scipy.stats import norm

    del random_seed  # B3: unused for momentum_12_1; logged for tag parity only.
    conformal_alpha = 1.0 - coverage_level
    z_half = float(norm.ppf(1.0 - conformal_alpha / 2.0))

    mom = _momentum_12_1(wide_closes)
    rebal_dates = _bms_rebalance_dates(wide_closes)
    tickers: tuple[str, ...] = tuple(wide_closes.columns.tolist())

    rows: list[pd.DataFrame] = []
    for rd in rebal_dates:
        if rd not in mom.index:
            rows.append(_emit_nan_signal_row(rd, tickers))
            continue
        point = mom.loc[rd].to_numpy(dtype=np.float64)
        if not np.isfinite(point).any():
            rows.append(_emit_nan_signal_row(rd, tickers))
            continue
        sigma_xs = float(np.nanstd(point, ddof=0))
        if not math.isfinite(sigma_xs) or sigma_xs == 0.0:
            rows.append(_emit_nan_signal_row(rd, tickers))
            continue
        half_width = z_half * sigma_xs
        lower = point - half_width
        upper = point + half_width
        df = pd.DataFrame(
            {"expected_return": point, "lower": lower, "upper": upper},
            index=pd.MultiIndex.from_product([[rd], tickers], names=["date", "ticker"]),
        )
        rows.append(df)
    return pd.concat(rows) if rows else _empty_signal_frame()


def _emit_nan_signal_row(rd: pd.Timestamp, tickers: Sequence[str]) -> pd.DataFrame:
    n = len(tickers)
    return pd.DataFrame(
        {
            "expected_return": np.full(n, np.nan),
            "lower": np.full(n, np.nan),
            "upper": np.full(n, np.nan),
        },
        index=pd.MultiIndex.from_product([[rd], tickers], names=["date", "ticker"]),
    )


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expected_return": pd.Series(dtype=np.float64),
            "lower": pd.Series(dtype=np.float64),
            "upper": pd.Series(dtype=np.float64),
        },
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="date"), pd.Index([], dtype=object, name="ticker")],
        ),
    )


def generate_signals(wide_closes: pd.DataFrame, cell: SweepCell) -> pd.DataFrame:
    """Dispatch to the alpha-spec signal generator for the given cell."""
    if cell.alpha_spec == "existing_cs_alpha_nco":
        return _signals_existing_cs_alpha_nco(
            wide_closes,
            coverage_level=cell.coverage_level,
            random_seed=cell.random_seed,
        )
    if cell.alpha_spec == "random_gaussian":
        return _signals_random_gaussian(
            wide_closes,
            coverage_level=cell.coverage_level,
            random_seed=cell.random_seed,
        )
    if cell.alpha_spec == "momentum_12_1":
        return _signals_momentum_12_1(
            wide_closes,
            coverage_level=cell.coverage_level,
            random_seed=cell.random_seed,
        )
    raise ValueError(f"unknown alpha_spec {cell.alpha_spec!r}")


# ────────────────────────────────────────────────────────────────────
# §8. Metric computation
# ────────────────────────────────────────────────────────────────────


def _signal_level_metrics(alpha_signals: pd.DataFrame, n_universe: int) -> dict[str, float]:
    """Signal-level diagnostics at the most-recent rebalance date.

    AC3.metrics signal-level keys. Computed strictly from the raw
    ``(lower, upper, expected_return)`` arrays; NO composition with
    strategy admissibility (N1).
    """
    if alpha_signals.empty:
        return {
            "n_names": float(n_universe),
            "n_active": 0.0,
            "n_signal_tradeable": 0.0,
            "tradeable_fraction": 0.0,
            "median_abs_expected_return": 0.0,
            "median_interval_half_width": 0.0,
            "p90_interval_half_width": 0.0,
            "signal_to_interval_ratio": 0.0,
        }
    last_rd = alpha_signals.index.get_level_values("date").max()
    row = alpha_signals.xs(last_rd, level="date")
    er = row["expected_return"].to_numpy(dtype=np.float64)
    lo = row["lower"].to_numpy(dtype=np.float64)
    hi = row["upper"].to_numpy(dtype=np.float64)
    finite = np.isfinite(er) & np.isfinite(lo) & np.isfinite(hi)
    n_active = int(finite.sum())
    er_f = er[finite]
    lo_f = lo[finite]
    hi_f = hi[finite]
    if n_active == 0:
        return {
            "n_names": float(n_universe),
            "n_active": 0.0,
            "n_signal_tradeable": 0.0,
            "tradeable_fraction": 0.0,
            "median_abs_expected_return": 0.0,
            "median_interval_half_width": 0.0,
            "p90_interval_half_width": 0.0,
            "signal_to_interval_ratio": 0.0,
        }
    # signal_tradeable per the conformal contract; NOT composed with anything.
    signal_tradeable = (lo_f > 0.0) | (hi_f < 0.0)
    n_signal_tradeable = int(signal_tradeable.sum())
    half_width = (hi_f - lo_f) / 2.0
    median_half = float(np.median(half_width))
    p90_half = float(np.quantile(half_width, 0.90))
    median_abs_er = float(np.median(np.abs(er_f)))
    ratio = median_abs_er / median_half if median_half > 0.0 else 0.0
    return {
        "n_names": float(n_universe),
        "n_active": float(n_active),
        "n_signal_tradeable": float(n_signal_tradeable),
        "tradeable_fraction": float(n_signal_tradeable) / float(n_universe),
        "median_abs_expected_return": median_abs_er,
        "median_interval_half_width": median_half,
        "p90_interval_half_width": p90_half,
        "signal_to_interval_ratio": float(ratio),
    }


def _strategy_level_metrics(result: CSAlphaNCOResult, min_active_tickers: int) -> dict[str, float]:
    """Strategy-level diagnostics from CSAlphaNCOResult.

    n_strategy_admissible = rebalance dates where len(active) >= min_active_tickers
    active_fraction       = n_strategy_admissible / n_rebalances
    n_orders              = sum of per-rebalance position-delta counts. NOT
                            from a broker round-trip. N1: this is independent
                            of n_signal_tradeable; the two are NEVER composed
                            in the harness.
    """
    n_active_history = result.n_active_history
    n_rebal = int(len(n_active_history))
    n_admissible = int((n_active_history >= min_active_tickers).sum())
    weights = result.weights_history
    prior = weights.shift(1).fillna(0.0)
    deltas = (weights - prior).abs() > 1e-9
    n_orders = int(deltas.sum().sum())
    return {
        "n_strategy_admissible": float(n_admissible),
        "active_fraction": float(n_admissible) / float(max(n_rebal, 1)),
        "n_orders": float(n_orders),
    }


def _performance_level_metrics(
    result: CSAlphaNCOResult, n_orders: int
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    """Performance-level metrics on daily_pnl.

    Per B1 single NaN convention: when n_orders == 0, SR_active = NaN,
    DSR = NaN (assigned in a later second-pass step), turnover = 0.0.

    Returns
    -------
    metrics : dict
        ``SR_active``, ``turnover`` populated. ``DSR`` deferred to the
        second-pass aggregation step (cross-trial sigma not available
        per-cell).
    daily_pnl_active : np.ndarray
        Active (non-zero-position) portion of daily_pnl, used in the
        DSR second pass.
    return_moments : dict
        ``{"n": ..., "mean": ..., "std": ..., "skew": ..., "kurt": ...}``
        cached for the DSR second pass.
    """
    if n_orders == 0:
        return (
            {"SR_active": float("nan"), "turnover": 0.0},
            np.empty(0, dtype=np.float64),
            {"n": 0.0, "mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0},
        )

    daily_pnl = result.daily_pnl
    # Active days: days where strategy had nonzero position before the day's pnl.
    # CSAlphaNCOResult.daily_pnl records pnl by date; weights are at rebalance dates.
    # Approximation: drop days where daily_pnl == 0 exactly (zero-position byproduct).
    # This is conservative — a real zero return is mis-classified but only on
    # active days, which is a negligible bias for diagnostic purposes.
    pnl_arr = daily_pnl.to_numpy(dtype=np.float64)
    active = pnl_arr[pnl_arr != 0.0]
    if len(active) < 4:
        sr_metrics = {"SR_active": float("nan"), "turnover": float(_mean_turnover(result))}
        return (sr_metrics, active, {"n": 0.0, "mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0})

    stats = sharpe_ratio_stats(active, periods_per_year=PERIODS_PER_YEAR)
    return (
        {"SR_active": float(stats.sr), "turnover": float(_mean_turnover(result))},
        active,
        {
            "n": float(stats.n_obs),
            "mean": float(active.mean()),
            "std": float(active.std(ddof=1)),
            "skew": float(stats.skew),
            "kurt": float(stats.kurt),
        },
    )


def _mean_turnover(result: CSAlphaNCOResult) -> float:
    """Mean turnover over admissible rebalance dates (those with n_active > 0)."""
    n_active = result.n_active_history
    mask = n_active > 0
    if not bool(mask.any()):
        return 0.0
    return float(result.turnover_history[mask].mean())


# ────────────────────────────────────────────────────────────────────
# §9. Cell runner
# ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class CellOutcome:
    """In-memory record of a single cell run, pre-DSR second pass."""

    cell: SweepCell
    metrics: dict[str, float]  # signal + strategy + performance (no DSR yet)
    daily_pnl_active: np.ndarray
    return_moments: dict[str, float]
    wallclock_seconds: float
    degenerate_no_orders: bool


def run_cell(cell: SweepCell, *, panel: pd.DataFrame | None = None) -> CellOutcome:
    """Run one sweep cell end-to-end. Composes existing surfaces only."""
    t0 = time.perf_counter()
    if panel is None:
        panel = load_panel(cell.universe)
    wide = pivot_to_wide(panel).sort_index()
    n_universe = wide.shape[1]
    panel_returns = _daily_returns_panel(wide)

    alpha_signals = generate_signals(wide, cell)

    config = CSAlphaNCOConfig(
        cov_estimator="lw",
        cov_lookback_days=252,
        n_clusters_rule="sqrt_n",
        n_clusters_fixed=None,
        clustering_method="ward",
        kelly_fraction=0.5,
        kelly_cap=0.25,
        rebalance_freq="BMS",
        max_signal_age_days=60,
        min_active_tickers=cell.active_threshold,
    )

    # Drop NaN signal rows before handing to the strategy.
    alpha_signals_clean = alpha_signals.dropna()

    result = cs_alpha_nco_backtest(
        alpha_signals=alpha_signals_clean,
        panel_returns=panel_returns,
        config=config,
    )

    signal_m = _signal_level_metrics(alpha_signals, n_universe)
    strategy_m = _strategy_level_metrics(result, cell.active_threshold)
    n_orders = int(strategy_m["n_orders"])
    perf_m, daily_pnl_active, return_moments = _performance_level_metrics(result, n_orders)

    metrics: dict[str, float] = {}
    metrics.update(signal_m)
    metrics.update(strategy_m)
    metrics.update(perf_m)

    wallclock = time.perf_counter() - t0
    metrics["run_wallclock_seconds"] = float(wallclock)

    degenerate = n_orders == 0
    return CellOutcome(
        cell=cell,
        metrics=metrics,
        daily_pnl_active=daily_pnl_active,
        return_moments=return_moments,
        wallclock_seconds=wallclock,
        degenerate_no_orders=degenerate,
    )


# ────────────────────────────────────────────────────────────────────
# §10. MLflow logging
# ────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def local_tracking(mlruns_dir: Path) -> Iterator[None]:
    """Set MLflow tracking URI to ``file://{mlruns_dir}`` for the block.

    Mirrors the S28 PR1 ``with_local_tracking`` helper but accepts an
    explicit directory (the S28 helper is fixed to a ``tmp_path``
    semantics). Restores the prior URI on exit.
    """
    import mlflow

    mlruns_dir.mkdir(parents=True, exist_ok=True)
    prior_uri = mlflow.get_tracking_uri()
    local_uri = f"file://{mlruns_dir}"
    mlflow.set_tracking_uri(local_uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        yield
    finally:
        mlflow.set_tracking_uri(prior_uri)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _existing_run_keys(mlruns_dir: Path) -> set[tuple[float, int, str, str, int]]:
    """Read the MLflow store and return the set of (coverage, active, spec,
    universe, seed) tuples already logged. Used for resumption.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if not mlruns_dir.exists():
        return set()
    prior_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    client = MlflowClient()
    keys: set[tuple[float, int, str, str, int]] = set()
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return set()
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
        for run in runs:
            p = run.data.params
            try:
                key = (
                    float(p["coverage_level"]),
                    int(p["active_threshold"]),
                    str(p["alpha_spec"]),
                    str(p["universe"]),
                    int(p["random_seed"]),
                )
            except (KeyError, ValueError):
                continue
            keys.add(key)
    finally:
        mlflow.set_tracking_uri(prior_uri)
    return keys


def log_outcome(outcome: CellOutcome, *, git_sha: str) -> str:
    """Log one cell to MLflow under the current tracking URI. Returns run_id.

    NaN-valued metrics are NOT logged to MLflow (MLflow rejects non-finite
    values). They appear in the CSV via the second-pass aggregator.
    """
    import mlflow

    cell = outcome.cell
    params: dict[str, object] = {
        "coverage_level": cell.coverage_level,
        "conformal_alpha": cell.conformal_alpha,
        "active_threshold": cell.active_threshold,
        "alpha_spec": cell.alpha_spec,
        "universe": cell.universe,
        "random_seed": cell.random_seed,
    }
    tags: dict[str, str] = {
        "sprint": SPRINT,
        "experiment_family": "tradeability_sensitivity",
        "git_sha": git_sha,
        "degenerate_no_orders": "true" if outcome.degenerate_no_orders else "false",
    }
    metrics: dict[str, float] = {
        k: v for k, v in outcome.metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)
    }
    run_name = (
        f"cov{cell.coverage_level}_a{cell.active_threshold}_"
        f"{cell.alpha_spec}_{cell.universe}_s{cell.random_seed}"
    )
    run_id: str = ""
    with mlflow.start_run(run_name=run_name) as run:
        assert run is not None
        run_id = str(run.info.run_id)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for k, v in tags.items():
            mlflow.set_tag(k, v)
    return run_id


def _persist_pnl_sidecar(outcome: CellOutcome, run_id: str) -> None:
    """Persist active daily_pnl to ``results/s29/cell_pnl/{run_id}.json``.

    NOT an MLflow artifact (ARCHITECTURE.md invariant 5 — PnL stays out
    of MLflow). The sidecar lives in ``results/`` which is gitignored.
    Consumed by the DSR second pass.
    """
    CELL_PNL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "tag_key": list(outcome.cell.tag_key),
        "daily_pnl_active": outcome.daily_pnl_active.tolist(),
        "return_moments": outcome.return_moments,
        "degenerate_no_orders": outcome.degenerate_no_orders,
    }
    out_path = CELL_PNL_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(payload))


# ────────────────────────────────────────────────────────────────────
# §11. Sweep orchestration
# ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(slots=True)
class SweepReport:
    cells_run: int
    cells_skipped_resume: int
    cells_skipped_infeasible: int
    infeasible_reason: str | None
    wallclock_seconds: float


def run_sweep(
    *,
    mlruns_dir: Path = MLRUNS_DIR,
    cells: Sequence[SweepCell] = SWEEP_GRID,
    cell_budget_seconds: float = 300.0,
    sp500_degradation_threshold_cells: int = 3,
) -> SweepReport:
    """Execute the full sweep. Halts if a cell exceeds ``cell_budget_seconds``.

    SP500 degradation policy (AC4.3): if the first
    ``sp500_degradation_threshold_cells`` SP500 cells each exceed
    ``cell_budget_seconds``, the sweep drops to DJ30-only for remaining
    cells and records the drop diagnostic.
    """
    git_sha = _git_sha()
    t0 = time.perf_counter()

    existing_keys = _existing_run_keys(mlruns_dir)
    logger.info("found %d existing runs; will skip those on resume", len(existing_keys))

    # Pre-load panels once per universe to amortize DuckDB I/O across cells.
    panels: dict[str, pd.DataFrame] = {}

    sp500_over_budget_cells: list[SweepCell] = []
    sp500_dropped = False
    cells_run = 0
    cells_skipped_resume = 0
    cells_skipped_infeasible = 0
    infeasible_reason: str | None = None

    with local_tracking(mlruns_dir):
        for i, cell in enumerate(cells):
            if cell.tag_key in existing_keys:
                cells_skipped_resume += 1
                logger.info("[%d/%d] resume-skip %s", i + 1, len(cells), cell)
                continue
            if sp500_dropped and cell.universe == "SP500":
                cells_skipped_infeasible += 1
                continue

            if cell.universe not in panels:
                try:
                    panels[cell.universe] = load_panel(cell.universe)
                except Exception as exc:  # noqa: BLE001
                    logger.error("panel load failed for %s: %s", cell.universe, exc)
                    if cell.universe == "SP500":
                        sp500_dropped = True
                        infeasible_reason = f"SP500 panel load failed: {exc}"
                        cells_skipped_infeasible += 1
                        continue
                    raise

            try:
                outcome = run_cell(cell, panel=panels[cell.universe])
            except Exception as exc:  # noqa: BLE001
                logger.error("[%d/%d] cell failed %s: %s", i + 1, len(cells), cell, exc)
                if cell.universe == "SP500":
                    sp500_over_budget_cells.append(cell)
                    if len(sp500_over_budget_cells) >= sp500_degradation_threshold_cells:
                        sp500_dropped = True
                        infeasible_reason = (
                            f"SP500 cells failed: {len(sp500_over_budget_cells)} "
                            f"before degradation threshold {sp500_degradation_threshold_cells}"
                        )
                cells_skipped_infeasible += 1
                continue

            if outcome.wallclock_seconds > cell_budget_seconds:
                logger.warning(
                    "[%d/%d] cell exceeded budget: %.1fs > %.1fs — %s",
                    i + 1,
                    len(cells),
                    outcome.wallclock_seconds,
                    cell_budget_seconds,
                    cell,
                )
                if cell.universe == "SP500":
                    sp500_over_budget_cells.append(cell)
                    if len(sp500_over_budget_cells) >= sp500_degradation_threshold_cells:
                        sp500_dropped = True
                        infeasible_reason = (
                            f"SP500 cells consistently >{cell_budget_seconds}s "
                            f"(seen {len(sp500_over_budget_cells)} cells)"
                        )
                        logger.warning("SP500 degradation engaged: %s", infeasible_reason)
                else:
                    # DJ30 over budget — halt and surface.
                    elapsed = time.perf_counter() - t0
                    projected = elapsed / max(cells_run, 1) * len(cells)
                    msg = (
                        f"DJ30 cell exceeded budget: {outcome.wallclock_seconds:.1f}s > "
                        f"{cell_budget_seconds:.1f}s. completed={cells_run}, "
                        f"elapsed={elapsed:.1f}s, projected_total={projected:.1f}s. "
                        f"Halting per AC4.2."
                    )
                    raise RuntimeError(msg)

            run_id = log_outcome(outcome, git_sha=git_sha)
            _persist_pnl_sidecar(outcome, run_id)
            cells_run += 1
            logger.info(
                "[%d/%d] %s — wall=%.2fs SR=%.3f n_orders=%d tradeable=%.3f",
                i + 1,
                len(cells),
                cell,
                outcome.wallclock_seconds,
                outcome.metrics.get("SR_active", float("nan")),
                int(outcome.metrics.get("n_orders", 0)),
                outcome.metrics.get("tradeable_fraction", 0.0),
            )

    return SweepReport(
        cells_run=cells_run,
        cells_skipped_resume=cells_skipped_resume,
        cells_skipped_infeasible=cells_skipped_infeasible,
        infeasible_reason=infeasible_reason,
        wallclock_seconds=time.perf_counter() - t0,
    )


# ────────────────────────────────────────────────────────────────────
# §12. CSV emission + DSR second pass
# ────────────────────────────────────────────────────────────────────


def emit_csv(
    *,
    mlruns_dir: Path = MLRUNS_DIR,
    cell_pnl_dir: Path = CELL_PNL_DIR,
    out_path: Path | None = None,
) -> Path:
    """Aggregate MLflow runs + sidecars into the canonical sweep_runs.csv.

    DSR is computed here in the second pass using
    ``deflated_sharpe_ratio(returns, n_trials=150,
    sr_std_cross_trial=σ̂_across_cells)`` where σ̂_across_cells is the
    std of SR_active across all logged cells.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if out_path is None:
        out_path = RESULTS_DIR / "sweep_runs.csv"

    prior_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise RuntimeError(f"experiment {EXPERIMENT_NAME!r} not found at {mlruns_dir}")
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=2000)
    finally:
        mlflow.set_tracking_uri(prior_uri)

    rows: list[dict[str, Any]] = []
    sr_values: list[float] = []
    for run in runs:
        p = run.data.params
        m = run.data.metrics
        t = run.data.tags
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "coverage_level": float(p.get("coverage_level", "nan")),
            "active_threshold": int(p.get("active_threshold", "0")),
            "alpha_spec": p.get("alpha_spec", ""),
            "universe": p.get("universe", ""),
            "random_seed": int(p.get("random_seed", "0")),
            "conformal_alpha": float(p.get("conformal_alpha", "nan")),
            "degenerate_no_orders": t.get("degenerate_no_orders", ""),
            "git_sha": t.get("git_sha", ""),
        }
        for key in (
            "n_names",
            "n_active",
            "n_signal_tradeable",
            "tradeable_fraction",
            "median_abs_expected_return",
            "median_interval_half_width",
            "p90_interval_half_width",
            "signal_to_interval_ratio",
            "n_strategy_admissible",
            "active_fraction",
            "n_orders",
            "turnover",
            "run_wallclock_seconds",
        ):
            row[key] = float(m[key]) if key in m else float("nan")
        # SR_active: NaN if degenerate (no orders) — not logged to MLflow.
        if t.get("degenerate_no_orders", "false") == "true":
            row["SR_active"] = float("nan")
        else:
            row["SR_active"] = float(m["SR_active"]) if "SR_active" in m else float("nan")
        if math.isfinite(row["SR_active"]):
            sr_values.append(row["SR_active"])
        rows.append(row)

    sr_arr = np.array(sr_values, dtype=np.float64)
    cross_trial_sigma = float(np.std(sr_arr, ddof=1)) if len(sr_arr) >= 2 else float("nan")

    for row in rows:
        if math.isfinite(row["SR_active"]) and math.isfinite(cross_trial_sigma):
            sidecar = cell_pnl_dir / f"{row['run_id']}.json"
            if not sidecar.exists():
                row["DSR"] = float("nan")
                continue
            payload = json.loads(sidecar.read_text())
            active = np.asarray(payload["daily_pnl_active"], dtype=np.float64)
            if len(active) < 4:
                row["DSR"] = float("nan")
                continue
            try:
                p_dsr, _ = deflated_sharpe_ratio(
                    active,
                    n_trials=150,
                    periods_per_year=PERIODS_PER_YEAR,
                    sr_std_cross_trial=cross_trial_sigma,
                )
                row["DSR"] = float(p_dsr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DSR computation failed for %s: %s", row["run_id"], exc)
                row["DSR"] = float("nan")
        else:
            row["DSR"] = float("nan")

    df = pd.DataFrame(rows)
    cols_first = [
        "coverage_level",
        "active_threshold",
        "alpha_spec",
        "universe",
        "random_seed",
    ]
    ordered = cols_first + [c for c in df.columns if c not in cols_first]
    df = df[ordered]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("emitted %d rows -> %s", len(df), out_path)
    return out_path


# ────────────────────────────────────────────────────────────────────
# §13. CLI
# ────────────────────────────────────────────────────────────────────


def _cli() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("S29 sweep starting; %d cells; experiment=%s", len(SWEEP_GRID), EXPERIMENT_NAME)
    report = run_sweep()
    logger.info(
        "sweep done: ran=%d resume_skip=%d infeasible_skip=%d wallclock=%.1fs",
        report.cells_run,
        report.cells_skipped_resume,
        report.cells_skipped_infeasible,
        report.wallclock_seconds,
    )
    if report.infeasible_reason is not None:
        logger.warning("infeasible reason: %s", report.infeasible_reason)
    csv_path = emit_csv()
    logger.info("CSV: %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
