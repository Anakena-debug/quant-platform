from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class FFDResult:
    series: pd.Series
    d: float
    adf_stat: float
    adf_pval: float
    corr: float
    weights: np.ndarray


@njit(cache=True)
def _ffd_weights(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    w = np.empty(max_size, dtype=np.float64)
    w[0] = 1.0
    k = 1
    while k < max_size:
        w[k] = -w[k - 1] * (d - k + 1.0) / k
        if abs(w[k]) < threshold:
            break
        k += 1
    return w[:k][::-1]


@njit(cache=True)
def _apply_ffd(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n, L = len(values), len(weights)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(L - 1, n):
        out[t] = np.dot(weights, values[t - L + 1 : t + 1])
    return out


def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    return _ffd_weights(d, threshold, max_size)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    w = get_weights_ffd(d, threshold)
    out = _apply_ffd(series.to_numpy(np.float64, copy=False), w)
    return pd.Series(out, index=series.index, name=f"{series.name or 'x'}_ffd")


def get_adf_stat(series: pd.Series) -> tuple[float, float]:
    x = series.dropna()
    if len(x) < 20:
        return np.nan, np.nan
    stat, pval, *_ = adfuller(x, autolag="AIC")
    return float(stat), float(pval)


def find_optimal_d(
    series: pd.Series,
    d_values: np.ndarray | None = None,
    threshold: float = 1e-5,
    pval_threshold: float = 0.05,
) -> FFDResult:
    if d_values is None:
        d_values = np.linspace(0.0, 1.0, 11)
    best = None
    for d in d_values:
        ffd = frac_diff_ffd(series, float(d), threshold)
        stat, pval = get_adf_stat(ffd)
        corr = float(series.corr(ffd))
        if np.isfinite(pval) and pval < pval_threshold:
            best = (float(d), stat, pval, corr, ffd)
            break
    if best is None:
        d = 1.0
        ffd = frac_diff_ffd(series, d, threshold)
        stat, pval = get_adf_stat(ffd)
        corr = float(series.corr(ffd))
        best = (d, stat, pval, corr, ffd)
    d, stat, pval, corr, ffd = best
    return FFDResult(ffd, d, stat, pval, corr, get_weights_ffd(d, threshold))
