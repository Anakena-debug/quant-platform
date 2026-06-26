"""Block bootstrap + Patton-Politis-White 2009 block-length selector.

Provides the dependence-adjusted resampling primitive used downstream by
P2.3 DSR cross-trial σ̂ testing (AR(1) invariant 4 in
`tests/test_dsr_cross_trial_sigma.py`).

Primitives
----------
- `block_bootstrap`        — moving-block (Kuensch 1989) / circular-block
                             (Politis-Romano 1992) resampling.
- `politis_white_block_length` — data-driven optimal block length per
                                  Patton-Politis-White 2009 JCE
                                  correction to Politis-White 2004.
- `_block_bootstrap_core`  — numba @njit(cache=True) hot loop.

Canon references
----------------
- Kuensch 1989: "The Jackknife and the Bootstrap for General Stationary
  Observations", Ann. Stat. 17(3), pp 1217-1241.
- Politis & Romano 1992: "A Circular Block-Resampling Procedure for
  Stationary Data", in Exploring the Limits of Bootstrap.
- Politis & White 2004: "Automatic Block-Length Selection for the
  Dependent Bootstrap", Econometric Reviews 23(1), pp 53-70,
  DOI 10.1081/ETC-120028836.
- Patton, Politis & White 2009: "Correction to 'Automatic Block-Length
  Selection for the Dependent Bootstrap' by D. Politis and H. White",
  Econometric Reviews 28(4), pp 372-375, DOI 10.1080/07474930802459016.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        # Shim matches P0.3 / P1.1 / P2.1 precedent: decorated function
        # remains correct at ~10-50x slowdown under numba-absent venvs.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


# =====================================================================
# Public API
# =====================================================================


def block_bootstrap(
    x: np.ndarray | pd.Series,
    *,
    block_size: int,
    n_replicates: int,
    rng: np.random.Generator,
    circular: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Moving-block or circular-block bootstrap resampling.

    Preserves short-range dependence by resampling contiguous blocks of
    length ``block_size`` rather than individual observations.

    Parameters
    ----------
    x : np.ndarray | pd.Series, shape (n,)
        1-D series to resample.
    block_size : int
        Length of each bootstrap block. Must satisfy ``1 <= block_size
        <= n``. Values > ``n // 2`` emit a ``UserWarning`` (too few
        possible block starts, statistical properties degrade).
    n_replicates : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Seeded RNG. All randomness flows from here.
    circular : bool, default False
        If True, block starts wrap around the end of ``x`` (Politis-
        Romano 1992). If False, block starts are restricted to
        ``[0, n - block_size]`` (Kuensch 1989 moving block).

    Returns
    -------
    np.ndarray, shape (n_replicates, n)
        if ``x`` is ``np.ndarray``.
    pd.DataFrame, shape (n_replicates, n)
        if ``x`` is ``pd.Series``. Columns mirror ``x.index``; index is
        ``RangeIndex(n_replicates)``.

    Raises
    ------
    ValueError
        If ``block_size < 1`` or ``block_size > n`` or
        ``n_replicates < 1``.

    Notes
    -----
    - Length preserved: each replicate has exactly ``n`` samples.
      Moving block truncates the trailing partial block; circular
      block wraps.
    - Determinism: the same ``rng`` produces bitwise-identical output
      across runs (both the numba @njit and the Python fallback).
    - Circular block has uniform inclusion probability per sample;
      moving block under-samples the last ``block_size - 1`` samples
      by ~``block_size / n``.
    """
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    n = x_arr.size

    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) exceeds series length ({n})")
    if block_size > n // 2:
        warnings.warn(
            f"block_size={block_size} > n/2={n // 2}; few possible block "
            f"starts, statistical properties degrade",
            UserWarning,
            stacklevel=2,
        )
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")

    n_blocks = (n + block_size - 1) // block_size  # ceil(n / block_size)

    # Draw block starts via rng (outside numba to keep seed contract clean).
    if circular:
        block_starts = rng.integers(0, n, size=(n_replicates, n_blocks))
    else:
        high = n - block_size + 1
        if high <= 0:
            # n == block_size: only one possible start (0).
            block_starts = np.zeros((n_replicates, n_blocks), dtype=np.int64)
        else:
            block_starts = rng.integers(0, high, size=(n_replicates, n_blocks))
    block_starts = np.ascontiguousarray(block_starts, dtype=np.int64)

    out = _block_bootstrap_core(
        x_arr,
        int(block_size),
        int(n_replicates),
        int(n_blocks),
        block_starts,
        bool(circular),
    )

    if isinstance(x, pd.Series):
        return pd.DataFrame(
            out,
            index=pd.RangeIndex(n_replicates),
            columns=x.index,
        )
    return out


def politis_white_block_length(
    x: np.ndarray | pd.Series,
    *,
    max_lag: int | None = None,
) -> int:
    """Optimal circular-block bootstrap length (Patton-Politis-White 2009).

    Implements Patton-Politis-White 2009 Eq. corrected from Politis-White
    2004 Theorem 3.1:

    .. math::
       b_{opt}^{CB} = \\left\\lceil \\left(
         \\frac{2 \\hat{G}^2}{\\hat{D}_{CB}}
       \\right)^{1/3} \\cdot n^{1/3} \\right\\rceil

    where ``hat_G`` and ``hat_D_CB`` are flat-top-kernel-weighted sums
    of sample auto-covariances; the bandwidth ``M`` is selected
    automatically per Politis-White 2004 §2.2 (first lag beyond which
    all of the next ``K_n`` autocorrelations stay below the threshold
    ``c_n = 2 sqrt(log10(n) / n)``).

    Parameters
    ----------
    x : np.ndarray | pd.Series, shape (n,)
        Stationary (or differenced-to-stationarity) input. No
        stationarity check is performed; caller's responsibility.
    max_lag : int | None
        Autocorrelation truncation lag. Default
        ``min(ceil(log10(n) * n), n - 1)`` per Politis-White 2004 §4.

    Returns
    -------
    int
        Optimal block length clipped to ``[1, n // 2]``.

    Raises
    ------
    ValueError
        On degenerate-variance input (``var(x) == 0``).

    Notes
    -----
    - Monotone non-decreasing in ``|rho(1)|`` for AR(1) processes
      (invariant 3 in the P2.2 test suite).
    - At ``n < 50`` the estimator is noisy (Patton-Politis-White 2009).
      We return the value regardless; caller sanity-checks.
    - Near-unit-root inputs produce block lengths that can exceed
      ``n // 2``; the clip emits a ``UserWarning`` flagging possible
      non-stationarity.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    n = x_arr.size

    if n < 4:
        raise ValueError(f"politis_white_block_length needs n >= 4, got {n}")

    x_centered = x_arr - x_arr.mean()
    var_x = float((x_centered * x_centered).mean())  # biased, R_hat(0)
    if var_x <= 0.0 or not np.isfinite(var_x):
        raise ValueError(
            f"politis_white_block_length: degenerate-variance input "
            f"(var={var_x:.3e}); cannot estimate dependence structure"
        )

    if max_lag is None:
        max_lag = min(
            int(np.ceil(np.log10(max(n, 10)) * n)),
            n - 1,
        )
    max_lag = max(1, min(max_lag, n - 1))

    # Biased sample autocorrelations rho[k] for k in [0, max_lag].
    rho = np.empty(max_lag + 1, dtype=np.float64)
    rho[0] = 1.0
    for k in range(1, max_lag + 1):
        rho[k] = float((x_centered[k:] * x_centered[: n - k]).mean() / var_x)

    # Automatic bandwidth M per Politis-White 2004 §2.2:
    # find first k0 in [1, ...] such that |rho[k0+1..k0+K_n]| all < c_n.
    K_n = max(5, int(np.sqrt(np.log10(max(n, 10)))))
    c_n = 2.0 * np.sqrt(np.log10(max(n, 10)) / n)
    m_hat: int | None = None
    for k0 in range(1, max_lag - K_n):
        window = np.abs(rho[k0 + 1 : k0 + 1 + K_n])
        if bool(np.all(window < c_n)):
            m_hat = k0
            break
    if m_hat is None:
        m_hat = max(1, max_lag // 2)
    M = min(2 * m_hat, max_lag)

    # Flat-top kernel lambda(u).
    def _lam(u: float) -> float:
        au = abs(u)
        if au <= 0.5:
            return 1.0
        if au <= 1.0:
            return 2.0 * (1.0 - au)
        return 0.0

    # G_hat and D_CB_hat. Using R(-k) = R(k): double the positive-k sum.
    # R_hat(k) = var_x * rho[k] for k in [0, M].
    g_hat = 0.0
    d_cb_hat = var_x * var_x  # k=0 term (lambda(0)=1, |0|*... = 0 in G)
    for k in range(1, M + 1):
        lam = _lam(k / M)
        rk = var_x * float(rho[k])
        g_hat += lam * k * rk
        d_cb_hat += 2.0 * lam * (rk * rk)
    g_hat *= 2.0
    d_cb_hat *= 4.0 / 3.0

    if d_cb_hat <= 0.0 or not np.isfinite(d_cb_hat) or not np.isfinite(g_hat):
        # Defensive: fallback to a mild block length.
        b = max(1, int(np.round(n ** (1.0 / 3.0))))
    else:
        b = int(np.ceil((2.0 * g_hat * g_hat / d_cb_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)))

    # Clip to [1, n // 2]; warn on clip at upper bound (near-unit-root).
    if b > n // 2:
        warnings.warn(
            f"politis_white_block_length: selected block={b} clipped to "
            f"n//2={n // 2}; input may be near-unit-root / non-stationary",
            UserWarning,
            stacklevel=2,
        )
        b = n // 2
    b = max(1, b)
    return int(b)


# =====================================================================
# Numba private core
# =====================================================================


@njit(cache=True)
def _block_bootstrap_core(
    x: np.ndarray,
    block_size: int,
    n_replicates: int,
    n_blocks: int,
    block_starts: np.ndarray,
    circular: bool,
) -> np.ndarray:
    """Assemble block-bootstrap replicates from pre-drawn block-start indices.

    JIT-compiled hot loop. `block_starts` is pre-drawn outside so the
    caller's `np.random.Generator` drives every randomness decision.
    """
    n = x.shape[0]
    out = np.empty((n_replicates, n), dtype=np.float64)
    for r in range(n_replicates):
        idx = 0
        for b in range(n_blocks):
            start = block_starts[r, b]
            for k in range(block_size):
                if idx >= n:
                    break
                if circular:
                    pos = (start + k) % n
                else:
                    pos = start + k
                out[r, idx] = x[pos]
                idx += 1
            if idx >= n:
                break
    return out
