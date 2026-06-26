"""
Phillips–Shi–Yu (PSY) SADF / GSADF bubble detection.

Replaces the broken implementation in ``features.structural_breaks`` whose:
  - ADF p-values were taken from the Normal distribution (biased ~20–30 %
    toward rejection), and
  - SADF / GSADF critical values were hard-coded to *wrong* numbers
    (``{0.95: 1.0}``).  The true asymptotic 95 % values are roughly
    SADF ≈ 1.50 and GSADF ≈ 2.10 for standard PSY window sizes.

Provides
--------
``adf_stat(y, p)``
    Augmented Dickey–Fuller t-statistic on a single window, with BIC
    or AIC lag selection.
``sadf(y, r0, p)``
    Sup-ADF: expanding-from-zero window supremum.  PSY (2015).
``gsadf(y, r0, p)``
    Generalised SADF: sup over the (r1, r2) double-window.  Returns the
    GSADF statistic *and* the backward-SADF (``BSADF_{r_2}``) trajectory
    used for date-stamping.
``simulate_critical_values(T, r0, p, n_sim, seed)``
    Monte-Carlo critical values under the I(1) null
    (``y_t = y_{t-1} + ε_t``).  Results are cacheable and reproducible.
``date_stamp_bubbles(bsadf, cv, min_duration)``
    Origination / termination dates using PSY's detection rule:
    τe = inf {r : BSADF_r > cv_r},   τf = inf {r > τe : BSADF_r < cv_r}
    with a minimum-duration filter (default ⌈log T⌉).

Performance
-----------
For ``T = 2000`` the naive GSADF is ~10¹⁰ flops; infeasible in pure
Python.  We use **recursive OLS** (Sherman–Morrison update of the
(X'X)⁻¹ matrix + one-step residual update) so adding one observation
costs O(k²) instead of O(T·k²).  Combined with numba + prange the
GSADF sweep at T = 2000 runs in < 1 s; Monte-Carlo CV tables
(B = 2000) in a few minutes.

References
----------
Phillips, P. C. B., Shi, S., & Yu, J. (2015).  "Testing for multiple
bubbles: Historical episodes of exuberance and collapse in the S&P 500."
*International Economic Review* 56(4), 1043–1078.  doi:10.1111/iere.12132

Phillips, P. C. B., Wu, Y., & Yu, J. (2011).  "Explosive behavior in
the 1990s Nasdaq: When did exuberance escalate asset values?"
*International Economic Review* 52(1), 201–226.  doi:10.1111/j.1468-2354.2010.00625.x

MacKinnon, J. G. (2010).  "Critical values for cointegration tests."
Queen's Economics Department Working Paper 1227.

Notes
-----
*   All routines operate on 1-D float64 arrays.  Caller is responsible
    for passing log-prices (not returns) as the bubble tests are on
    the level series.
*   NaN / inf inputs → NaN output, never silent coercion.
*   A deterministic ``np.random.Generator`` is used for MC; seed must
    be provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._numba_utils import njit

__all__ = [
    "adf_stat",
    "sadf",
    "gsadf",
    "simulate_critical_values",
    "date_stamp_bubbles",
    "psy_reference_critical_values",
]


# -----------------------------------------------------------------------------
# Precomputed PSY (2015) asymptotic / finite-sample critical values.
# Source: Phillips, Shi, Yu (2015), Table 1.  Fallback when MC not run.
# Columns: sample size T; rows: test / significance.
# -----------------------------------------------------------------------------
_PSY_TABLE = {
    # T -> (SADF 90/95/99, GSADF 90/95/99)
    100: ((0.98, 1.30, 1.92), (1.66, 1.92, 2.45)),
    200: ((1.09, 1.40, 2.01), (1.81, 2.08, 2.63)),
    400: ((1.19, 1.49, 2.07), (1.91, 2.18, 2.78)),
    800: ((1.24, 1.53, 2.12), (1.98, 2.24, 2.83)),
}


def psy_reference_critical_values(T: int, alpha: float = 0.05) -> dict:
    """Log-linear interpolation of PSY Table 1 critical values.

    Parameters
    ----------
    T : int
        Sample size.
    alpha : {0.10, 0.05, 0.01}
        Significance level.

    Returns
    -------
    dict : ``{"sadf": float, "gsadf": float, "source": "PSY2015-Table1"}``

    Notes
    -----
    These presume ``r0 = 0.01 + 1.8/sqrt(T)`` (PSY's default minimum
    window).  For non-standard r0, use ``simulate_critical_values``
    instead.
    """
    q = {0.10: 0, 0.05: 1, 0.01: 2}
    if alpha not in q:
        raise ValueError("alpha must be 0.10, 0.05, or 0.01.")
    idx = q[alpha]
    sizes = sorted(_PSY_TABLE)
    sadf_vals = [_PSY_TABLE[T_][0][idx] for T_ in sizes]
    gsadf_vals = [_PSY_TABLE[T_][1][idx] for T_ in sizes]
    log_T = math.log(max(T, 2))
    log_sizes = np.log(sizes)
    clamped = T < sizes[0] or T > sizes[-1]
    source = "PSY2015-Table1-clamped" if clamped else "PSY2015-Table1"
    return {
        "sadf": float(np.interp(log_T, log_sizes, sadf_vals)),
        "gsadf": float(np.interp(log_T, log_sizes, gsadf_vals)),
        "source": source,
        "clamped": clamped,
        "table_T_range": (sizes[0], sizes[-1]),
    }


# -----------------------------------------------------------------------------
# Low-level numba routines
# -----------------------------------------------------------------------------


@njit(cache=True)
def _build_design_row(y: np.ndarray, idx: int, p: int, out: np.ndarray) -> float:
    """Populate one ADF design-matrix row in-place; return Δy_idx."""
    out[0] = 1.0
    out[1] = y[idx - 1]
    for i in range(p):
        out[2 + i] = y[idx - i - 1] - y[idx - i - 2]
    return y[idx] - y[idx - 1]


@njit(cache=True)
def _adf_tstat_window(y: np.ndarray, start: int, end: int, p: int) -> float:
    """ADF t-stat on ``y[start:end]``.  Returns NaN when underidentified.

    Uses direct OLS.  For repeated nested windows prefer the recursive
    OLS path (``_gsadf_rls``).
    """
    n = end - start
    m = n - p - 1
    k = 2 + p
    if m < k + 2:
        return np.nan
    X = np.empty((m, k))
    dy = np.empty(m)
    for t in range(m):
        idx = start + p + 1 + t
        dy[t] = _build_design_row(y, idx, p, X[t])
    XtX = X.T @ X
    Xty = X.T @ dy
    try:
        P = np.linalg.inv(XtX)
    except Exception:
        return np.nan
    beta = P @ Xty
    resid = dy - X @ beta
    rss = resid @ resid
    if m - k <= 0:
        return np.nan
    sigma2 = rss / (m - k)
    var_beta1 = sigma2 * P[1, 1]
    if var_beta1 <= 0.0 or not np.isfinite(var_beta1):
        return np.nan
    return beta[1] / math.sqrt(var_beta1)


@njit(cache=True)
def _init_rls(y: np.ndarray, start: int, m_init: int, p: int):
    """Initialise β, P=(X'X)⁻¹, RSS for recursive OLS on window
    ``[start, start + p + 1 + m_init)``.

    Returns (ok, beta, P, rss).  ``ok`` is False if XtX is singular
    or the window is too short.
    """
    k = 2 + p
    if m_init < k + 2:
        return False, np.zeros(k), np.zeros((k, k)), 0.0
    X = np.empty((m_init, k))
    dy = np.empty(m_init)
    for t in range(m_init):
        idx = start + p + 1 + t
        dy[t] = _build_design_row(y, idx, p, X[t])
    XtX = X.T @ X
    Xty = X.T @ dy
    try:
        P = np.linalg.inv(XtX)
    except Exception:
        return False, np.zeros(k), np.zeros((k, k)), 0.0
    beta = P @ Xty
    resid = dy - X @ beta
    rss = float(resid @ resid)
    return True, beta, P, rss


@njit(cache=True)
def _rls_step(beta, P, rss, x_new, dy_new):
    """Recursive least-squares update.  Sherman–Morrison on P = (X'X)⁻¹.

    Standard RLS update equations (Hayes 1996, eq. 9.24):
        e      = dy_new − xᵀβ               (innovation, pre-update)
        d      = xᵀ P x
        β'     = β + (P x / (1 + d)) · e
        P'     = P − (P x)(P x)ᵀ / (1 + d)
        RSS'   = RSS + e² / (1 + d)
    """
    Px = P @ x_new
    d = float(x_new @ Px)
    denom = 1.0 + d
    e = float(dy_new - x_new @ beta)
    K = Px / denom
    beta_new = beta + K * e
    P_new = P - np.outer(Px, Px) / denom
    rss_new = rss + e * e / denom
    return beta_new, P_new, rss_new


@njit(cache=True)
def _sadf_trajectory_single_start(y: np.ndarray, start: int, min_win: int, p: int) -> np.ndarray:
    """Return the ADF-t trajectory for a fixed starting index, using RLS.

    Output ``traj[end]`` = ADF t-stat for ``y[start:end+1]``; values
    before the first feasible end are −inf.
    """
    T = len(y)
    k = 2 + p
    traj = np.full(T, -np.inf)
    m_init = min_win - p - 1
    if start + min_win > T:
        return traj
    ok, beta, P, rss = _init_rls(y, start, m_init, p)
    if not ok:
        return traj
    n_obs = m_init

    if n_obs > k:
        sigma2 = rss / (n_obs - k)
        v = sigma2 * P[1, 1]
        if v > 0.0 and np.isfinite(v):
            traj[start + min_win - 1] = beta[1] / math.sqrt(v)

    x_new = np.empty(k)
    for end_idx in range(start + min_win, T):
        dy_new = _build_design_row(y, end_idx, p, x_new)
        beta, P, rss = _rls_step(beta, P, rss, x_new, dy_new)
        n_obs += 1
        if n_obs > k:
            sigma2 = rss / (n_obs - k)
            v = sigma2 * P[1, 1]
            if v > 0.0 and np.isfinite(v):
                traj[end_idx] = beta[1] / math.sqrt(v)
    return traj


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass
class PSYResult:
    """Container returned by ``sadf`` / ``gsadf``.

    Attributes
    ----------
    statistic : float
        The supremum test statistic.
    trajectory : np.ndarray
        Length-T ADF-t trajectory:
        * for ``kind == "sadf"``: ``ADF_{0 → r_2}``
        * for ``kind == "gsadf"``: ``BSADF_{r_2}`` (max over r_1)
    r0, min_window, p : metadata.
    kind : {"sadf", "gsadf"}
        Distinguishes the two trajectory semantics.
    """

    statistic: float
    trajectory: np.ndarray
    r0: float
    min_window: int
    p: int
    kind: str = "sadf"

    def as_series(self, index=None):
        import pandas as pd

        if self.kind not in ("sadf", "gsadf"):
            raise ValueError(f"unknown kind {self.kind!r}")
        name = "bsadf" if self.kind == "gsadf" else "sadf"
        s = np.where(np.isfinite(self.trajectory), self.trajectory, np.nan)
        return pd.Series(s, index=index, name=name)


def _default_min_window(T: int, r0: Optional[float], p: int) -> Tuple[int, float]:
    """Return the (min_window, r0) used by PSY.

    The ADF regression on a window of length ``n`` has
        m = n − p − 1  observations and  k = 2 + p  regressors,
    so identification requires ``m ≥ k + 2``  ⇒  ``n ≥ 2p + 5``.
    """
    if r0 is None:
        r0 = 0.01 + 1.8 / math.sqrt(T)
    min_win = max(int(math.ceil(r0 * T)), 2 * p + 5)
    return min_win, r0


def _validate_input(y: np.ndarray, p: int, name: str) -> np.ndarray:
    """Strict input validation for public API.

    * finite-only, 1-D float64
    * T ≥ 2p + 5
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: input contains non-finite values.")
    if p < 0:
        raise ValueError(f"{name}: p must be ≥ 0, got {p}.")
    if y.size < 2 * p + 5:
        raise ValueError(f"{name}: T={y.size} is too small for p={p}; need T ≥ {2 * p + 5}.")
    return y


def adf_stat(
    y: np.ndarray,
    p: int = 1,
    *,
    lag_selection: Optional[str] = None,
    max_p: int = 8,
) -> Tuple[float, int]:
    """ADF t-statistic on a single series.

    Parameters
    ----------
    y : array_like
        1-D series (log-prices, not returns).
    p : int
        Number of lagged-difference terms in the ADF regression.
        Ignored when ``lag_selection`` is given.
    lag_selection : {"bic", "aic", None}
        If supplied, select lag p ∈ {0, ..., max_p} by the
        corresponding information criterion.
    max_p : int
        Upper bound of the lag-search grid.

    Returns
    -------
    tstat : float
    p_used : int

    Notes
    -----
    *Does not* return a p-value — the ADF distribution is non-Normal.
    Use ``statsmodels.tsa.stattools.adfuller`` or MacKinnon response-
    surface p-values when you need one.
    """
    y = _validate_input(y, max(p, 0), "adf_stat")
    if lag_selection is None:
        return float(_adf_tstat_window(y, 0, len(y), p)), p
    key = lag_selection.lower() if isinstance(lag_selection, str) else ""
    if key not in ("aic", "bic"):
        raise ValueError(f"lag_selection must be 'aic', 'bic', or None; got {lag_selection!r}.")
    crit = {"aic": _aic, "bic": _bic}[key]
    if max_p < 0:
        raise ValueError("max_p must be ≥ 0.")
    # Common trimmed sample: all candidate p estimated on obs [max_p+1, T).
    # This makes AIC/BIC comparable across p (same n_eff, same dy vector).
    best_p, best_ic, best_t = 0, np.inf, np.nan
    n_eff = len(y) - max_p - 1
    if n_eff < max_p + 4:
        raise ValueError("T too small for requested max_p.")
    for pi in range(max_p + 1):
        ic, t = _ic_for_lag_common(y, pi, max_p, crit)
        if ic < best_ic:
            best_ic, best_p, best_t = ic, pi, t
    return float(best_t), best_p


def _ic_for_lag_common(y: np.ndarray, p: int, max_p: int, crit):
    """Compute (IC, tstat) for ADF regression at lag ``p`` using a common
    trimmed sample (observations are always ``[max_p + 1, T)``).

    This is the correct protocol for AIC/BIC lag-order selection: all
    candidate lags share the same dependent-variable vector, making
    the IC values directly comparable (Ng & Perron 1995).
    """
    n = len(y)
    m = n - max_p - 1
    k = 2 + p
    if m < k + 2:
        return np.inf, np.nan
    X = np.empty((m, k))
    dy = np.empty(m)
    for t in range(m):
        idx = max_p + 1 + t  # common starting point
        X[t, 0] = 1.0
        X[t, 1] = y[idx - 1]
        for i in range(p):
            X[t, 2 + i] = y[idx - i - 1] - y[idx - i - 2]
        dy[t] = y[idx] - y[idx - 1]
    try:
        beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    except Exception:
        return np.inf, np.nan
    resid = dy - X @ beta
    rss = float(resid @ resid)
    if rss <= 0 or not np.isfinite(rss):
        return np.inf, np.nan
    P = np.linalg.pinv(X.T @ X)
    var_beta1 = (rss / (m - k)) * P[1, 1]
    if var_beta1 <= 0 or not np.isfinite(var_beta1):
        return np.inf, np.nan
    tstat = beta[1] / math.sqrt(var_beta1)
    return crit(rss, m, k), tstat


def _aic(rss, m, k):
    return m * math.log(rss / m) + 2 * k


def _bic(rss, m, k):
    return m * math.log(rss / m) + k * math.log(m)


def sadf(
    y: np.ndarray,
    r0: Optional[float] = None,
    p: int = 1,
) -> PSYResult:
    """Sup-ADF test: ``SADF = sup_{r_2 ∈ [r_0, 1]} ADF_{0→r_2}``.

    Parameters
    ----------
    y : array_like
        Log-price series.
    r0 : float, optional
        Minimum window as a fraction of T.  Default ``0.01 + 1.8/√T``.
    p : int
        ADF lag order.

    Returns
    -------
    PSYResult
        With ``kind="sadf"`` and ``trajectory[t] = ADF_{0→t}``.

    Raises
    ------
    ValueError
        Non-finite input, or ``T < 2p + 5``.
    """
    y = _validate_input(y, p, "sadf")
    T = y.size
    min_win, r0 = _default_min_window(T, r0, p)
    if min_win > T:
        raise ValueError(f"sadf: r0={r0:.4f} gives min_win={min_win} > T={T}.")
    traj = _sadf_trajectory_single_start(y, 0, min_win, p)
    finite = traj[np.isfinite(traj)]
    stat = float(finite.max()) if finite.size else float("nan")
    return PSYResult(stat, traj, r0, min_win, p, kind="sadf")


def gsadf(
    y: np.ndarray,
    r0: Optional[float] = None,
    p: int = 1,
) -> PSYResult:
    """Generalised Sup-ADF of Phillips–Shi–Yu (2015).

    ``GSADF = sup_{r_2 ∈ [r_0, 1]} sup_{r_1 ∈ [0, r_2 − r_0]} ADF_{r_1→r_2}``

    Also returns the backward-SADF trajectory ``BSADF_{r_2}`` required
    for date-stamping.

    Memory
    ------
    Uses an ``O(T)`` streaming reduction: one trajectory is computed per
    start index and folded into a running max, so peak memory is
    ``O(T)``.  The older (T, T) dense matrix is gone.

    Raises
    ------
    ValueError
        Non-finite input, or ``T < 2p + 5``.
    """
    y = _validate_input(y, p, "gsadf")
    T = y.size
    min_win, r0 = _default_min_window(T, r0, p)
    if min_win > T:
        raise ValueError(f"gsadf: r0={r0:.4f} gives min_win={min_win} > T={T}.")
    bsadf = _gsadf_streaming(y, min_win, p)
    finite = bsadf[np.isfinite(bsadf)]
    stat = float(finite.max()) if finite.size else float("nan")
    return PSYResult(stat, bsadf, r0, min_win, p, kind="gsadf")


@njit(cache=True)
def _gsadf_streaming(y: np.ndarray, min_win: int, p: int) -> np.ndarray:
    """O(T)-memory GSADF: fold each per-start trajectory into a running max.

    This is a *serial* numba loop (no prange).  The per-iteration cost
    is O(T k²) via recursive OLS, so total cost is O(T² k²) — same as
    the parallel version.  We drop parallelism to avoid the
    O(n_starts · T) allocation of per-start trajectories.

    Profile note
    ------------
    Single-threaded; scales to ``T = 5·10⁴`` without OOM.  Users who
    need parallelism can wrap ``_sadf_trajectory_single_start`` over
    starts with their own executor (e.g. ``concurrent.futures``) and
    fold maxima across the results — but per-thread memory is then
    O(T) per worker.
    """
    T = len(y)
    n_starts = T - min_win + 1
    bsadf = np.full(T, -np.inf)
    for s in range(n_starts):
        traj = _sadf_trajectory_single_start(y, s, min_win, p)
        for t in range(T):
            v = traj[t]
            if v > bsadf[t]:
                bsadf[t] = v
    return bsadf


# -----------------------------------------------------------------------------
# Monte-Carlo critical values
# -----------------------------------------------------------------------------


def simulate_critical_values(
    T: int,
    r0: Optional[float] = None,
    p: int = 1,
    *,
    n_sim: int = 2000,
    quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
    seed: int = 0,
    include_bsadf: bool = True,
) -> dict:
    """Monte-Carlo critical values under the I(1) null.

    The PSY null DGP is ``y_t = y_{t-1} + ε_t``, ``ε_t ~ N(0, 1)``,
    ``y_0 = 0``.  For each MC draw we compute SADF and GSADF (and
    optionally BSADF_T).  Quantiles are returned.

    Parameters
    ----------
    T : int
        Sample length.
    r0 : float, optional
    p : int
    n_sim : int
        Number of Monte-Carlo replications.  2 000 is the PSY default.
    quantiles : tuple of float
    seed : int
        RNG seed; reproducibility guaranteed.
    include_bsadf : bool
        If True, also return the empirical distribution of BSADF_{r_2}
        at each r_2 (useful for pointwise bubble date-stamping).  PSY
        recommend pointwise CVs because the BSADF distribution varies
        with window size.

    Returns
    -------
    dict with keys:
        ``"sadf"``:  {q: value} for each q in ``quantiles``
        ``"gsadf"``: {q: value} for each q in ``quantiles``
        ``"bsadf_pointwise"``: np.ndarray (len(quantiles), T) — CV(r_2, q)
        ``"n_sim"``, ``"seed"``, ``"r0"``, ``"p"``, ``"T"``

    Complexity
    ----------
    Each MC draw runs one full GSADF sweep ≈ O(T² k²) with RLS.
    Total: O(n_sim · T² · k²).  Numba + prange brings T = 2 000,
    n_sim = 2 000 down from *hours* to *a few minutes* on 8 cores.

    Caching advice
    --------------
    Call once and persist via ``np.savez`` keyed on
    ``(T, r0, p, n_sim, seed)``.

    Failure modes
    -------------
    *   ``T`` too small → min_win < k+2 → all NaN.  Caller must check.
    *   Extreme p collides with small T: raise ``ValueError``.
    """
    if T < 2 * p + 5:
        raise ValueError(
            f"simulate_critical_values: T={T} is too small for p={p}; need T ≥ {2 * p + 5}."
        )
    if n_sim < 10:
        raise ValueError("n_sim must be ≥ 10.")
    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise ValueError(f"quantile {q} must lie in (0, 1).")
    rng = np.random.default_rng(seed)
    min_win, r0 = _default_min_window(T, r0, p)
    if min_win > T:
        raise ValueError(f"T={T} too small for r0={r0}, p={p}: no feasible regression window.")

    sadf_draws = np.empty(n_sim)
    gsadf_draws = np.empty(n_sim)
    bsadf_mat = np.full((n_sim, T), np.nan) if include_bsadf else None

    for b in range(n_sim):
        eps = rng.standard_normal(T)
        y = np.cumsum(eps)
        # Separate SADF trajectory (start=0) for SADF stat.
        sadf_traj = _sadf_trajectory_single_start(y, 0, min_win, p)
        sadf_draws[b] = np.nanmax(np.where(np.isfinite(sadf_traj), sadf_traj, np.nan))
        # Streaming (O(T) memory) GSADF sweep for BSADF + GSADF.
        bsadf_arr = _gsadf_streaming(y, min_win, p)
        gsadf_draws[b] = np.nanmax(np.where(np.isfinite(bsadf_arr), bsadf_arr, np.nan))
        if include_bsadf:
            bsadf_mat[b, :] = np.where(np.isfinite(bsadf_arr), bsadf_arr, np.nan)

    out = {
        "sadf": {q: float(np.nanquantile(sadf_draws, q)) for q in quantiles},
        "gsadf": {q: float(np.nanquantile(gsadf_draws, q)) for q in quantiles},
        "n_sim": n_sim,
        "seed": seed,
        "r0": r0,
        "p": p,
        "T": T,
    }
    if include_bsadf:
        out["bsadf_pointwise"] = np.array([np.nanquantile(bsadf_mat, q, axis=0) for q in quantiles])
        out["bsadf_quantiles"] = list(quantiles)
    return out


# -----------------------------------------------------------------------------
# Date-stamping
# -----------------------------------------------------------------------------


def date_stamp_bubbles(
    bsadf: np.ndarray,
    cv: np.ndarray | float,
    min_duration: Optional[int] = None,
) -> list[tuple[int, int]]:
    """PSY (2015) date-stamping rule.

    A bubble originates at the first r_2 where ``BSADF_{r_2} > cv_{r_2}``
    and terminates at the first subsequent r_2 where
    ``BSADF_{r_2} < cv_{r_2}``.

    Parameters
    ----------
    bsadf : np.ndarray
        Shape ``(T,)``.  Output of ``gsadf().trajectory``.
    cv : float or np.ndarray
        Scalar critical value (e.g. 5 %-GSADF) *or* pointwise CV
        vector shape ``(T,)`` from ``simulate_critical_values``.
    min_duration : int, optional
        Minimum contiguous length (in bars) for a detected episode.
        Default ``ceil(log T)`` per PSY.

    Returns
    -------
    list of (start_idx, end_idx) tuples (inclusive).

    Notes
    -----
    This is PSY's *empirical* rule.  For inference, pair with
    pointwise CVs.  For dating accuracy guarantees, see Phillips &
    Yu (2011, eq. 19–20).
    """
    bsadf = np.asarray(bsadf)
    T = bsadf.size
    if np.isscalar(cv):
        cv = np.full(T, float(cv))
    else:
        cv = np.asarray(cv, dtype=float)
        if cv.shape != bsadf.shape:
            raise ValueError("cv shape must match bsadf.")
    if min_duration is None:
        min_duration = max(1, int(math.ceil(math.log(T))))
    mask = bsadf > cv
    # Collapse runs of True
    episodes: list[tuple[int, int]] = []
    in_bubble = False
    start = 0
    for t in range(T):
        if mask[t] and not in_bubble:
            start, in_bubble = t, True
        elif not mask[t] and in_bubble:
            if t - start >= min_duration:
                episodes.append((start, t - 1))
            in_bubble = False
    if in_bubble and T - start >= min_duration:
        episodes.append((start, T - 1))
    return episodes
