"""Marchenko-Pastur denoised covariance estimator (MLDP §2.6).

Constant-residual variant: eigenvalues at or below the MP edge λ+ are
replaced with their arithmetic mean (preserving total trace); eigenvalues
above λ+ are kept. The covariance is reconstructed via the Σ → (D, corr)
split, with σ² fit by KDE to the bulk MP density (with a documented
fallback for pathological inputs).

Convention: ``q = N / T`` (``n_features / n_samples``) — reciprocal of
MLDP's ``q = T / N``. Edge and density formulas adjusted accordingly:
λ± = σ²(1 ± √q)² with q = N/T (equivalent to σ²(1 ± √(1/q_MLDP))² with
q_MLDP = T/N).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde

_SIGMA2_LOWER = 1e-4
_SIGMA2_UPPER = 1.0


def _lam_plus(q: float, sigma2: float) -> float:
    return float(sigma2 * (1.0 + np.sqrt(q)) ** 2)


def _lam_minus(q: float, sigma2: float) -> float:
    return float(sigma2 * (1.0 - np.sqrt(q)) ** 2)


def marchenko_pastur_threshold(
    n_samples: int,
    n_features: int,
    sigma2: float = 1.0,
) -> float:
    """Upper Marchenko-Pastur edge λ+ = σ²(1 + √q)², q = n_features / n_samples.

    For an n_samples × n_features matrix of iid noise with variance σ², the
    sample correlation eigenvalues are bounded above (in the bulk) by λ+.
    Eigenvalues above λ+ are interpreted as signal; at or below as noise.
    """
    q = n_features / n_samples
    return _lam_plus(q, sigma2)


def _mp_density(
    lam: NDArray[np.float64],
    q: float,
    sigma2: float,
) -> NDArray[np.float64]:
    """Closed-form MP density f_MP(λ; σ², q) on (λ_-, λ_+); zero outside."""
    lp = _lam_plus(q, sigma2)
    lm = _lam_minus(q, sigma2)
    density = np.zeros_like(lam)
    inside = (lam > lm) & (lam < lp)
    if np.any(inside):
        x = lam[inside]
        density[inside] = np.sqrt(np.maximum((lp - x) * (x - lm), 0.0)) / (
            2.0 * np.pi * sigma2 * q * x
        )
    return density


def fit_mp_sigma2(
    eigenvalues: NDArray[np.float64],
    q: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Estimate noise σ² by KDE-vs-MP-density SSE minimisation.

    Input contract: ``eigenvalues`` MUST be eigenvalues of a CORRELATION
    matrix (trace = N). The (1e-4, 1.0) σ² bound is only well-posed in
    that domain. Caller is responsible for the Σ → (D, corr) split (i.e.,
    ``denoise_covariance`` standardises Σ → corr internally before
    calling).

    Algorithm: SSE between ``scipy.stats.gaussian_kde`` of empirical
    eigenvalues (Silverman/Scott bandwidth — equivalent in 1-D) and the
    closed-form MP density f_MP(λ; σ², q) on a uniform grid of 1000
    points over [λ_-(σ²), λ_+(σ²)]; scalar minimisation via
    ``scipy.optimize.minimize_scalar`` with bounds=(1e-4, 1.0) and
    method='bounded'.

    Fallback: on KDE bandwidth-selection failure, optimiser non-
    convergence, NaN gradient, or empirical bulk too narrow, return
    median(eigenvalues[eigenvalues < λ+(σ²=1)]), clipped to (1e-4, 1.0)
    — a single pass, no λ+ refinement (s83 F21: the doc previously
    promised a one-shot refinement the code never implemented). Both
    success and fallback paths return values in [1e-4, 1.0].
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    def _fallback() -> float:
        initial_lam_plus = _lam_plus(q, sigma2=1.0)
        bulk = eigenvalues[eigenvalues < initial_lam_plus]
        sigma2_est = float(np.median(bulk)) if bulk.size > 0 else 1.0
        return float(np.clip(sigma2_est, _SIGMA2_LOWER, _SIGMA2_UPPER))

    try:
        kde = gaussian_kde(eigenvalues)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return _fallback()

    def _sse(sigma2: float) -> float:
        lp = _lam_plus(q, sigma2)
        lm = _lam_minus(q, sigma2)
        if not np.isfinite(lp) or lp <= lm:
            return float(np.inf)
        # Floor at 1e-8: MP density's 1/x term blows up as lm→0 (q→1).
        grid = np.linspace(max(lm, 1e-8), lp, 1000)
        kde_vals = kde(grid)
        mp_vals = _mp_density(grid, q, sigma2)
        return float(np.sum((kde_vals - mp_vals) ** 2))

    try:
        result = minimize_scalar(
            _sse,
            bounds=(_SIGMA2_LOWER, _SIGMA2_UPPER),
            method="bounded",
            options={"maxiter": max_iter, "xatol": tol},
        )
    except (ValueError, RuntimeError):
        return _fallback()
    if not result.success or not np.isfinite(result.fun):
        return _fallback()
    return float(np.clip(result.x, _SIGMA2_LOWER, _SIGMA2_UPPER))


def denoise_eigenvalues_constant_residual(
    eigenvalues: NDArray[np.float64],
    q: float,
    *,
    sigma2: float | None = None,
) -> NDArray[np.float64]:
    """Constant-residual eigenvalue clipping (MLDP §2.6).

    Eigenvalues at or below λ+(σ²) are replaced with their arithmetic
    mean (preserving total trace); eigenvalues above λ+ are kept. If
    ``sigma2`` is None, fit it via ``fit_mp_sigma2``.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if sigma2 is None:
        sigma2 = fit_mp_sigma2(eigenvalues, q)
    lp = _lam_plus(q, sigma2)
    denoised = eigenvalues.copy()
    below = eigenvalues <= lp
    if np.any(below):
        denoised[below] = float(np.mean(eigenvalues[below]))
    return denoised


@dataclass(frozen=True)
class RMTDenoiseResult:
    """Intermediate + final artifacts of MLDP §2.6 covariance denoising.

    Exposes everything ``LeakageFreeRMTDenoiser`` persists so the denoise pipeline has a single
    implementation: ``cov_denoised``/``corr_denoised``, the standardization ``std``, the corr
    ``eigvals``/``eigvecs`` and their denoised ``eigvals_denoised``, the fitted ``sigma2``, the
    aspect ratio ``q`` and the MP edge ``lambda_plus``.
    """

    cov_denoised: NDArray[np.float64]
    corr_denoised: NDArray[np.float64]
    std: NDArray[np.float64]
    eigvals: NDArray[np.float64]
    eigvecs: NDArray[np.float64]
    eigvals_denoised: NDArray[np.float64]
    sigma2: float
    q: float
    lambda_plus: float


def denoise_covariance_full(
    returns: NDArray[np.float64],
    *,
    ddof: int = 1,
    sigma2: float | Literal["auto"] = "auto",
) -> RMTDenoiseResult:
    """MLDP §2.6 covariance denoising — the SINGLE source of truth, returning all intermediates.

    ``denoise_covariance`` returns just ``cov_denoised``; ``LeakageFreeRMTDenoiser`` persists the
    intermediates — both delegate here. ``sigma2='auto'`` fits via :func:`fit_mp_sigma2`; a
    positive float is used directly. Steps: Σ̂ → (D, corr) → eigh → σ² → constant-residual clip →
    reconstruct.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(
            f"`returns` must be 2-D (rows=samples, cols=features); got ndim={returns.ndim}"
        )
    n_samples, n_features = returns.shape
    if n_samples < 2:
        raise ValueError(
            f"`returns` must have at least 2 samples (rows); got n_samples={n_samples}"
        )

    cov_sample = np.cov(returns, rowvar=False, ddof=ddof)
    std = np.sqrt(np.diag(cov_sample))
    std_safe = np.where(std > 0, std, 1.0)
    inv_std = 1.0 / std_safe
    corr = cov_sample * inv_std[:, None] * inv_std[None, :]

    eigvals, eigvecs = np.linalg.eigh(corr)
    q = n_features / n_samples
    sigma2_hat = fit_mp_sigma2(eigvals, q) if isinstance(sigma2, str) else float(sigma2)
    eigvals_denoised = denoise_eigenvalues_constant_residual(eigvals, q, sigma2=sigma2_hat)

    corr_denoised = (eigvecs * eigvals_denoised) @ eigvecs.T
    # s83 F14: constant-residual clipping preserves the trace but NOT the
    # per-asset diagonal (diag_i = 1 + Σ_noise V_ij²(λ̄ - λ_j) ≠ 1).
    # MLDP's cov2corr renormalization restores the unit diagonal before
    # recomposing, so per-asset variances are preserved exactly (executed
    # repro pre-fix: diag ∈ [0.44, 1.21], up to 55.9% variance distortion
    # fed into NCO/min-var). detone_covariance already did this; the two
    # paths are now consistent. Trades exact trace-N preservation at the
    # corr level for an exact unit diagonal — the MLAM reference behavior.
    d = np.sqrt(np.clip(np.diag(corr_denoised), 1e-12, None))
    corr_denoised = corr_denoised * (1.0 / d)[:, None] * (1.0 / d)[None, :]
    np.fill_diagonal(corr_denoised, 1.0)
    cov_denoised = corr_denoised * std[:, None] * std[None, :]
    return RMTDenoiseResult(
        cov_denoised=cov_denoised,
        corr_denoised=corr_denoised,
        std=std,
        eigvals=eigvals,
        eigvecs=eigvecs,
        eigvals_denoised=eigvals_denoised,
        sigma2=float(sigma2_hat),
        q=q,
        lambda_plus=_lam_plus(q, sigma2_hat),
    )


def denoise_covariance(
    returns: NDArray[np.float64],
    *,
    ddof: int = 1,
) -> NDArray[np.float64]:
    """End-to-end MLDP §2.6 covariance denoising on a returns matrix.

    Steps (Σ → (D, corr) split):
      1. Sample covariance Σ̂ from ``returns`` (rows=samples, cols=features).
      2. Standardize: D = diag(sqrt(diag(Σ̂))); corr = D⁻¹·Σ̂·D⁻¹.
      3. Eigendecompose corr → (eigvals, eigvecs).
      4. Fit σ² via ``fit_mp_sigma2`` on corr eigenvalues (corr trace = N).
      5. Denoise eigvals (constant-residual clipping at λ+(σ²)).
      6. Reconstruct corr_denoised = V·diag(λ_denoised)·V^T.
      7. Recompose Σ̂_denoised = D·corr_denoised·D.
    """
    return denoise_covariance_full(returns, ddof=ddof).cov_denoised


def detone_covariance(
    cov: NDArray[np.float64],
    n_market_factors: int = 1,
) -> NDArray[np.float64]:
    """Detone a covariance matrix — strip the market mode (MLAM Ch.2 §2.6).

    Removes the top ``n_market_factors`` eigenpairs (the largest eigenvalues — the systematic /
    market component) from the **correlation** matrix, then maps back to covariance via the
    *original* volatilities. Detoning is done in correlation space, NOT on ``cov`` directly:
    detoning the covariance would conflate the market mode with the volatility structure.

    Sequence:
      1. D = diag(sqrt(diag(cov))); corr = D⁻¹·cov·D⁻¹.
      2. eigh(corr); zero the top ``n_market_factors`` eigenvalues (the market mode).
      3. Reconstruct corr from the surviving eigenpairs.
      4. Renormalize to unit diagonal (rescale so diag = 1) — a congruence transform, so it
         preserves the rank deficiency.
      5. Recompose Σ_detoned = D·corr_detoned·D using the ORIGINAL vols.

    The detoned correlation is **rank-deficient by ``n_market_factors``** (singular by
    construction — its top ``n_market_factors`` eigenvalues are zero). Variances are preserved
    (``diag(Σ_detoned) == diag(cov)``); only off-diagonal co-movement is stripped. Downstream
    optimizers (min-variance, NCO) MUST use a pseudo-inverse or a shrinkage floor — a naive matrix
    inverse will raise / produce exploding weights.

    Parameters
    ----------
    cov : (N, N)
        Symmetric covariance with a strictly-positive finite diagonal. Typically the *denoised*
        Σ from :func:`denoise_covariance` (denoise → detone is the MLAM sequence), but any valid
        covariance works.
    n_market_factors : int, default 1
        Number of leading eigenpairs to remove. Must satisfy ``1 <= n_market_factors < N``.
        ``1`` removes the single market mode; the general MLAM form allows more.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"`cov` must be a square 2-D matrix; got shape {cov.shape}")
    n = cov.shape[0]
    if not 1 <= n_market_factors < n:
        raise ValueError(f"`n_market_factors` must satisfy 1 <= k < N={n}; got {n_market_factors}")
    var = np.diag(cov)
    if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
        raise ValueError("`cov` must have a strictly-positive finite diagonal (variances).")

    std = np.sqrt(var)
    inv_std = 1.0 / std
    corr = cov * inv_std[:, None] * inv_std[None, :]

    # eigh returns ascending eigenvalues; the market mode is the LARGEST → drop the top k (last k).
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals_detoned = eigvals.copy()
    eigvals_detoned[n - n_market_factors :] = 0.0
    corr_detoned = (eigvecs * eigvals_detoned) @ eigvecs.T

    # Renormalize to unit diagonal (congruence transform → rank deficiency preserved).
    d = np.sqrt(np.clip(np.diag(corr_detoned), 1e-12, None))
    corr_detoned = corr_detoned * (1.0 / d)[:, None] * (1.0 / d)[None, :]
    np.fill_diagonal(corr_detoned, 1.0)

    # Map back to covariance via the ORIGINAL vols (variances preserved on the diagonal).
    return corr_detoned * std[:, None] * std[None, :]
