"""quantcore.screening — cross-sectional IC screening with FDR control.

The "screen" gate of alpha discovery: rank candidate factors by their information
coefficient (per-date cross-sectional rank correlation with forward returns), with a
Newey-West HAC-corrected t-stat on the IC series and Benjamini-Hochberg FDR control
across the whole family of factors tested (so screening many factors doesn't manufacture
false positives).

It is *generic* — it operates on already-computed factor-value panels (``[dates x
assets]``) keyed by name (e.g. the names in :mod:`quantcore.catalog`), independent of how
the factors were produced; factor construction belongs to the research pipeline::

    from quantcore.screening import screen_factors, to_json
    results = screen_factors({"mom": mom_panel, "rev": rev_panel}, fwd_returns, fdr=0.10)
    to_json(results)   # ranked, JSON-serializable

This is the light first gate (IC + FDR). Deflated-Sharpe / PBO (:mod:`quantcore.validation`)
are the heavier downstream gates a survivor proceeds to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class FactorScreenResult:
    """One factor's screen verdict. ``rank`` 1 = strongest |mean_ic|."""

    name: str
    n_days: int
    mean_ic: float
    ic_std: float
    ic_ir: float  # mean_ic / ic_std — information ratio of the IC series
    t_stat: float  # mean_ic / SE(mean_ic); Newey-West HAC when hac_lags > 0
    p_value: float
    q_value: float  # Benjamini-Hochberg FDR-adjusted p-value
    significant: bool  # q_value <= fdr
    rank: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ic_rows(fv: np.ndarray, rv: np.ndarray, min_obs: int) -> np.ndarray:
    """Row-wise centered Pearson IC of two aligned ``[dates x assets]`` arrays.

    Inputs are ranked upstream for Spearman. Each row is restricted to pairwise-finite cells and
    centered on that valid set; the result is NaN below ``min_obs`` valid pairs or on zero
    variance. Equivalent to :func:`numpy.corrcoef` per date, vectorized across all dates at once.
    """
    mask = np.isfinite(fv) & np.isfinite(rv)
    n = mask.sum(axis=1)
    safe_n = np.where(n > 0, n, 1)
    fz = np.where(mask, fv, 0.0)
    rz = np.where(mask, rv, 0.0)
    mean_f = fz.sum(axis=1) / safe_n
    mean_r = rz.sum(axis=1) / safe_n
    fc = np.where(mask, fv - mean_f[:, None], 0.0)  # centered on the valid set; invalid -> 0
    rc = np.where(mask, rv - mean_r[:, None], 0.0)
    cov = (fc * rc).sum(axis=1)
    var_f = (fc * fc).sum(axis=1)
    var_r = (rc * rc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = cov / np.sqrt(var_f * var_r)
    ic[(n < min_obs) | (var_f <= 0.0) | (var_r <= 0.0)] = np.nan
    return ic


def cross_sectional_ic(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    min_obs: int = 5,
) -> pd.Series:
    """Per-date cross-sectional IC between factor exposures and forward returns.

    Both inputs are ``[dates x assets]`` and are inner-aligned on shared dates+assets.
    ``method='spearman'`` (rank IC, default) or ``'pearson'``. A date with fewer than
    ``min_obs`` valid asset pairs yields NaN.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    f, r = factor.align(fwd_returns, join="inner")
    if method == "spearman":
        f = f.rank(axis=1)
        r = r.rank(axis=1)

    # Vectorized across all dates (no per-date Python loop). fwd is ranked once per call here;
    # screen_factors reuses a single fwd ranking across a uniform factor family (see below).
    ic = _ic_rows(f.to_numpy(dtype=np.float64), r.to_numpy(dtype=np.float64), min_obs)
    return pd.Series(ic, index=f.index, name="ic")


def _hac_se_of_mean(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett-kernel) HAC standard error of the sample mean of ``x``."""
    n = x.size
    xc = x - x.mean()
    s = float(xc @ xc) / n  # gamma_0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1)
        cov = float(xc[lag:] @ xc[:-lag]) / n
        s += 2.0 * w * cov
    if s <= 0.0:
        return float("nan")
    return float(np.sqrt(s / n))


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (monotone, clipped to [0, 1])."""
    n = pvals.size
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotone non-decreasing
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def screen_factors(
    factors: Mapping[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    hac_lags: int = 0,
    fdr: float = 0.10,
    min_obs: int = 5,
) -> list[FactorScreenResult]:
    """Screen a family of factors by IC, returning FDR-controlled, ranked verdicts.

    ``factors`` maps name -> ``[dates x assets]`` exposure panel; ``fwd_returns`` is the
    ``[dates x assets]`` forward-return panel to predict. ``hac_lags>0`` uses a
    Newey-West HAC SE for the IC t-stat (recommended — daily IC is autocorrelated).
    ``fdr`` is the Benjamini-Hochberg target. Results are sorted by ``|mean_ic|``
    descending (NaN last) and carry a 1-based ``rank``.
    """
    names = list(factors)

    # Reuse a single fwd ranking across the family when every factor shares fwd's exact axes
    # (the common case: all panels pivoted from one frame, e.g. the factory). Saves K-1 redundant
    # rankings of the same returns. Falls back to per-factor alignment when shapes differ (or the
    # method is invalid, so cross_sectional_ic still raises).
    uniform = method in ("spearman", "pearson") and all(
        factors[name].index.equals(fwd_returns.index)
        and factors[name].columns.equals(fwd_returns.columns)
        for name in names
    )
    if uniform:
        shared_rv = (
            fwd_returns.rank(axis=1).to_numpy(np.float64)
            if method == "spearman"
            else fwd_returns.to_numpy(np.float64)
        )

        def _ic_of(name: str) -> pd.Series:
            fac = factors[name]
            fv = (
                fac.rank(axis=1).to_numpy(np.float64)
                if method == "spearman"
                else fac.to_numpy(np.float64)
            )
            return pd.Series(_ic_rows(fv, shared_rv, min_obs), index=fwd_returns.index, name="ic")
    else:

        def _ic_of(name: str) -> pd.Series:
            return cross_sectional_ic(factors[name], fwd_returns, method=method, min_obs=min_obs)

    means: list[float] = []
    stds: list[float] = []
    irs: list[float] = []
    tstats: list[float] = []
    pvals: list[float] = []
    ndays: list[int] = []
    for name in names:
        ic = _ic_of(name).dropna()
        n = int(ic.size)
        ndays.append(n)
        if n < 3:
            means.append(np.nan)
            stds.append(np.nan)
            irs.append(np.nan)
            tstats.append(np.nan)
            pvals.append(1.0)
            continue
        arr = ic.to_numpy(dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        se = (
            _hac_se_of_mean(arr, hac_lags)
            if hac_lags > 0
            else (std / np.sqrt(n) if std > 0 else float("nan"))
        )
        t = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
        p = float(2.0 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else 1.0
        means.append(mean)
        stds.append(std)
        irs.append(mean / std if std > 0 else float("nan"))
        tstats.append(t)
        pvals.append(p)

    qvals = _bh_qvalues(np.asarray(pvals, dtype=np.float64))
    order = sorted(
        range(len(names)),
        key=lambda i: -abs(means[i]) if np.isfinite(means[i]) else float("inf"),
    )
    results: list[FactorScreenResult] = []
    for rank, i in enumerate(order, start=1):
        results.append(
            FactorScreenResult(
                name=names[i],
                n_days=ndays[i],
                mean_ic=round(means[i], 6) if np.isfinite(means[i]) else float("nan"),
                ic_std=round(stds[i], 6) if np.isfinite(stds[i]) else float("nan"),
                ic_ir=round(irs[i], 6) if np.isfinite(irs[i]) else float("nan"),
                t_stat=round(tstats[i], 4) if np.isfinite(tstats[i]) else float("nan"),
                p_value=round(float(pvals[i]), 6),
                q_value=round(float(qvals[i]), 6),
                significant=bool(qvals[i] <= fdr),
                rank=rank,
            )
        )
    return results


def read_panel_frame(path: str | Path) -> pd.DataFrame:
    """Read a long-format panel table for screening — parquet (``.parquet``) or CSV.

    Expected columns: a date column, an asset column, one column per factor, and a
    forward-return column (names are configurable in :func:`screen_long_frame`). Parquet
    requires ``pyarrow``; CSV is dependency-free.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def pivot_panels(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    return_col: str = "forward_return",
    factor_cols: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Pivot a LONG table into ``(name -> [dates x assets] factor panel, forward-return panel)``.

    ``frame`` has one row per (date, asset) with a column per factor plus a forward-return
    column. ``factor_cols`` defaults to every column except date/asset/return. Duplicate
    (date, asset) cells are mean-aggregated (``pivot_table`` default). This is the shared
    long->wide step behind :func:`screen_long_frame` and
    :func:`quantcore.factory.run_factory_frame`.
    """
    missing = {date_col, asset_col, return_col} - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required column(s): {sorted(missing)}")
    if factor_cols is None:
        reserved = {date_col, asset_col, return_col}
        factor_cols = [c for c in frame.columns if c not in reserved]
    if not factor_cols:
        raise ValueError("no factor columns found (only date/asset/return present)")
    fwd = frame.pivot_table(index=date_col, columns=asset_col, values=return_col)
    factors = {
        name: frame.pivot_table(index=date_col, columns=asset_col, values=name)
        for name in factor_cols
    }
    return factors, fwd


def screen_long_frame(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    return_col: str = "forward_return",
    factor_cols: list[str] | None = None,
    method: str = "spearman",
    hac_lags: int = 0,
    fdr: float = 0.10,
    min_obs: int = 5,
) -> list[FactorScreenResult]:
    """Screen a LONG-format table: pivot each factor column to ``[dates x assets]`` and screen.

    Thin wrapper over :func:`pivot_panels` + :func:`screen_factors`; see those for the column
    contract and the statistics produced.
    """
    factors, fwd = pivot_panels(
        frame,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        factor_cols=factor_cols,
    )
    return screen_factors(factors, fwd, method=method, hac_lags=hac_lags, fdr=fdr, min_obs=min_obs)


def to_json(results: list[FactorScreenResult], *, indent: int | None = 2) -> str:
    """Serialize screen results to JSON (a list of result dicts), for tooling / agents."""
    return json.dumps([r.to_dict() for r in results], indent=indent)


__all__ = [
    "FactorScreenResult",
    "cross_sectional_ic",
    "pivot_panels",
    "read_panel_frame",
    "screen_factors",
    "screen_long_frame",
    "to_json",
]
