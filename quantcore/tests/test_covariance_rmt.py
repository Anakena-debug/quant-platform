"""Tests for quantcore.covariance.rmt — RMT denoising (MLDP §2.6).

Pinned values loaded from tests/spikes/s19_pr1_recorded.json (output of
the pre-emission measurement script tests/spikes/s19_pr1_measurements.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantcore.covariance import (
    denoise_covariance,
    denoise_eigenvalues_constant_residual,
    detone_covariance,
    fit_mp_sigma2,
    marchenko_pastur_threshold,
)

_RECORDED = json.loads((Path(__file__).parent / "spikes" / "s19_pr1_recorded.json").read_text())


@pytest.mark.parametrize(
    "n_samples,n_features,sigma2",
    [
        (100, 10, 0.5),  # q = 0.1
        (100, 10, 1.0),
        (100, 10, 2.0),
        (100, 50, 0.5),  # q = 0.5
        (100, 50, 1.0),
        (100, 50, 2.0),
        (100, 100, 0.5),  # q = 1.0
        (100, 100, 1.0),
        (100, 100, 2.0),
        (100, 200, 0.5),  # q = 2.0
        (100, 200, 1.0),
        (100, 200, 2.0),
    ],
)
def test_marchenko_pastur_threshold_matches_analytic(
    n_samples: int, n_features: int, sigma2: float
) -> None:
    """λ+ = σ²(1 + √q)² with q = N/T (this codebase's convention)."""
    q = n_features / n_samples
    expected = sigma2 * (1.0 + np.sqrt(q)) ** 2
    measured = marchenko_pastur_threshold(n_samples, n_features, sigma2)
    assert measured == pytest.approx(expected, abs=1e-12)


def test_denoise_eigenvalues_constant_residual_preserves_trace() -> None:
    """Constant-residual replacement preserves total eigenvalue sum."""
    rng = np.random.default_rng(20260503)
    eigenvalues = np.sort(rng.uniform(0.1, 5.0, size=20))
    denoised = denoise_eigenvalues_constant_residual(eigenvalues, q=0.5, sigma2=1.0)
    assert np.sum(denoised) == pytest.approx(np.sum(eigenvalues), abs=1e-10)


def test_denoise_covariance_reduces_frobenius_error_vs_sample() -> None:
    """RMT denoising reduces Frobenius error vs Σ_true compared to sample.

    At q=0.05 with total spike spectrum [5, 3, 2], the Frobenius metric
    is bulk-dominated; this test defends "denoising helps overall", not
    "spike-recovery quality" (see
    ``test_denoise_covariance_preserves_above_edge_eigenvalues`` for the
    latter). Pin at 0.879093 = measured (0.7326) + 20% margin defends
    ratio < 1 with margin against seed-to-seed variation
    (2σ ≈ 0.07 from a 5-seed A/B variance check).
    """
    spec = _RECORDED["spec_b"]
    rng_sigma = np.random.default_rng(spec["seed_sigma"])
    rng_returns = np.random.default_rng(spec["seed_returns"])
    n_samples = spec["n_samples"]
    n_features = spec["n_features"]
    spike_added = spec["spike_eigenvalues_added"]
    K = len(spike_added)
    A = rng_sigma.standard_normal((n_features, K))
    L, _ = np.linalg.qr(A)
    Sigma_true = L @ np.diag(spike_added) @ L.T + np.eye(n_features)
    chol = np.linalg.cholesky(Sigma_true)
    Z = rng_returns.standard_normal((n_samples, n_features))
    returns = Z @ chol.T
    cov_sample = np.cov(returns, rowvar=False)
    cov_denoised = denoise_covariance(returns)
    err_sample = np.linalg.norm(cov_sample - Sigma_true, "fro")
    err_denoised = np.linalg.norm(cov_denoised - Sigma_true, "fro")
    ratio = err_denoised / err_sample
    pin_threshold = spec["ratio"] * 1.20
    assert ratio < pin_threshold
    assert ratio < 1.0


def test_denoise_covariance_preserves_above_edge_eigenvalues() -> None:
    """Top-K denoised eigenvalues recover the known total spike spectrum.

    Directly tests spike-preservation (which the Frobenius ratio doesn't
    isolate at q=0.05). Asserts top-3 eigenvalues of ``cov_denoised`` are
    within 30% relative error of the known total spike spectrum [5, 3, 2].
    """
    spec = _RECORDED["spec_b"]
    rng_sigma = np.random.default_rng(spec["seed_sigma"])
    rng_returns = np.random.default_rng(spec["seed_returns"])
    n_samples = spec["n_samples"]
    n_features = spec["n_features"]
    spike_added = spec["spike_eigenvalues_added"]
    spike_total = spec["spike_eigenvalues_total"]
    K = len(spike_added)
    A = rng_sigma.standard_normal((n_features, K))
    L, _ = np.linalg.qr(A)
    Sigma_true = L @ np.diag(spike_added) @ L.T + np.eye(n_features)
    chol = np.linalg.cholesky(Sigma_true)
    Z = rng_returns.standard_normal((n_samples, n_features))
    returns = Z @ chol.T
    cov_denoised = denoise_covariance(returns)
    eigvals_denoised = np.sort(np.linalg.eigvalsh(cov_denoised))[::-1][:K]
    expected = np.array(sorted(spike_total, reverse=True))
    rel_err = np.abs(eigvals_denoised - expected) / expected
    assert np.all(rel_err < 0.30), (
        f"rel_err={rel_err.tolist()}, "
        f"eigvals_denoised={eigvals_denoised.tolist()}, "
        f"expected={expected.tolist()}"
    )


def test_fit_mp_sigma2_recovers_known_noise_variance() -> None:
    """fit_mp_sigma2 recovers σ²=1 from a pure-noise Wishart spectrum."""
    spec = _RECORDED["spec_a"]
    rng = np.random.default_rng(spec["seed"])
    n_samples = spec["n_samples"]
    n_features = spec["n_features"]
    sigma2_true = spec["sigma2_true"]
    Z = rng.standard_normal((n_samples, n_features)) * np.sqrt(sigma2_true)
    cov_sample = np.cov(Z, rowvar=False)
    std = np.sqrt(np.diag(cov_sample))
    corr = cov_sample / np.outer(std, std)
    eigvals = np.linalg.eigvalsh(corr)
    q = n_features / n_samples
    sigma2_hat = fit_mp_sigma2(eigvals, q)
    abs_err = abs(sigma2_hat - sigma2_true)
    # Pin: measured (5.5e-7) + 50% margin ≈ 8.2e-7, rounded up to 1e-6
    # for slack against minor scipy/numpy upgrades.
    assert abs_err < 1e-6


@pytest.mark.parametrize(
    "name,eigvals",
    [
        ("all_equal", np.ones(20)),
        ("only_spikes_no_bulk", np.array([5.0, 3.0, 2.0])),
        ("single_eigenvalue", np.array([1.0])),
    ],
)
def test_fit_mp_sigma2_falls_back_on_pathological_input(name: str, eigvals: np.ndarray) -> None:
    """Fallback path returns finite σ² ∈ [1e-4, 1.0] without exception."""
    sigma2_hat = fit_mp_sigma2(eigvals, q=0.05)
    assert np.isfinite(sigma2_hat)
    # Upper bound 1.0 + 1e-10 defends against scipy numerical drift on the
    # clip ceiling; lower bound 1e-4 is a hard clip floor from _SIGMA2_LOWER.
    assert 1e-4 <= sigma2_hat <= 1.0 + 1e-10


@pytest.mark.parametrize(
    "name,returns",
    [
        ("1d_input", np.zeros(50)),
        ("n_samples_1", np.zeros((1, 5))),
        ("n_samples_0", np.zeros((0, 5))),
    ],
)
def test_denoise_covariance_shape_guards(name: str, returns: np.ndarray) -> None:
    """Public-API shape guards on ``denoise_covariance`` raise ValueError."""
    with pytest.raises(ValueError):
        denoise_covariance(returns)


# ---------------------------------------------------------------------------
# Detoning (MLAM Ch.2 §2.6) — s76 / advisory A1
# ---------------------------------------------------------------------------
def _market_cov(std: np.ndarray, rho: float) -> np.ndarray:
    """Covariance from an equicorrelation matrix (one dominant market mode)."""
    n = std.size
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(std, std)


def test_detone_unit_diagonal_rank_and_market_mode_removed() -> None:
    std = np.array([0.10, 0.20, 0.30, 0.15, 0.25])
    cov = _market_cov(std, rho=0.85)
    det = detone_covariance(cov, n_market_factors=1)
    # Map back to correlation to inspect the structural invariants.
    d = np.sqrt(np.diag(det))
    corr_det = det / np.outer(d, d)
    assert np.allclose(np.diag(corr_det), 1.0)  # unit diagonal after renormalization
    assert np.allclose(det, det.T)  # symmetric
    # Rank-deficient by exactly n_market_factors (singular by construction).
    assert np.linalg.matrix_rank(det) == std.size - 1
    # The largest eigenvalue (the market mode) is gone: top eigenvalue of the detoned corr
    # is far below the toned market eigenvalue (~1 + (N-1)·rho).
    toned_top = np.linalg.eigvalsh(cov / np.outer(std, std))[-1]
    detoned_top = np.linalg.eigvalsh(corr_det)[-1]
    assert detoned_top < 0.5 * toned_top


def test_detone_preserves_variances_and_strips_comovement() -> None:
    std = np.array([0.10, 0.20, 0.30, 0.15])
    cov = _market_cov(std, rho=0.80)
    det = detone_covariance(cov)
    # Variances (the diagonal) are preserved exactly; only off-diagonal co-movement changes.
    assert np.allclose(np.diag(det), np.diag(cov))
    off = ~np.eye(cov.shape[0], dtype=bool)
    corr_toned = cov / np.outer(std, std)
    corr_det = det / np.outer(np.sqrt(np.diag(det)), np.sqrt(np.diag(det)))
    # The common market mode was the source of the 0.80 cross-correlation → it collapses.
    assert np.mean(np.abs(corr_det[off])) < 0.5 * np.mean(np.abs(corr_toned[off]))


def test_detone_is_singular_requires_pseudo_inverse() -> None:
    """The detoned Σ is rank-deficient: astronomically ill-conditioned (a naive inverse would
    explode), but a pseudo-inverse min-variance solve stays finite — the downstream contract."""
    std = np.array([0.10, 0.20, 0.30, 0.15, 0.25, 0.12])
    cov = _market_cov(std, rho=0.7)
    det = detone_covariance(cov, n_market_factors=1)
    assert np.linalg.matrix_rank(det) == std.size - 1  # singular by construction
    assert np.linalg.cond(det) > 1e10  # naive inverse would blow up
    ones = np.ones(std.size)
    w_raw = np.linalg.pinv(det) @ ones  # pseudo-inverse min-variance
    w = w_raw / w_raw.sum()
    assert np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0)


def test_detone_n_market_factors_param() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 8))
    x[:, :4] += 3.0 * rng.standard_normal((400, 1))  # inject 2 common modes
    x[:, 4:] += 2.0 * rng.standard_normal((400, 1))
    cov = np.cov(x, rowvar=False)
    assert np.linalg.matrix_rank(detone_covariance(cov, n_market_factors=1)) == 7
    assert np.linalg.matrix_rank(detone_covariance(cov, n_market_factors=2)) == 6


@pytest.mark.parametrize(
    "cov,k",
    [
        (np.eye(3)[:2], 1),  # non-square
        (np.eye(3), 0),  # k < 1
        (np.eye(3), 3),  # k >= N
    ],
)
def test_detone_validation_raises(cov: np.ndarray, k: int) -> None:
    with pytest.raises(ValueError):
        detone_covariance(cov, n_market_factors=k)


def test_detone_rejects_nonpositive_diagonal() -> None:
    bad = np.array([[1.0, 0.1, 0.0], [0.1, 0.0, 0.1], [0.0, 0.1, 1.0]])  # zero variance row
    with pytest.raises(ValueError):
        detone_covariance(bad)


# ---------------------------------------------------------------------------
# s83 F14 — unit-diagonal renormalization before recomposition
# ---------------------------------------------------------------------------


def _factor_returns(seed: int = 7, T: int = 250, N: int = 50, K: int = 3) -> np.ndarray:
    """Factor-structured returns: K signal eigenvalues survive the MP clip.

    Load-bearing fixture choice: on PURE-NOISE input every eigenvalue is
    clipped to the common mean and the recomposed corr collapses to the
    identity, whose diagonal is 1 trivially — the pre-fix defect is
    invisible there (the s83 audit's first repro made exactly that
    mistake). Signal + noise mixtures are where the missing cov2corr
    renormalization distorted the diagonal (executed pre-fix repro on
    this fixture: diag ∈ [0.44, 1.21], max variance distortion 55.9%).
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=(N, K))
    factors = rng.normal(size=(T, K))
    eps = rng.normal(size=(T, N)) * 0.8
    return factors @ beta.T + eps


def test_denoised_corr_has_unit_diagonal_under_factor_structure() -> None:
    from quantcore.covariance.rmt import denoise_covariance_full

    res = denoise_covariance_full(_factor_returns())
    assert (res.eigvals > res.lambda_plus).sum() >= 2, "fixture must keep signal eigvals"
    np.testing.assert_allclose(np.diag(res.corr_denoised), 1.0, rtol=0.0, atol=1e-12)


def test_denoised_cov_preserves_sample_variances() -> None:
    """diag(Σ_denoised) == diag(Σ_sample): denoising must reshape
    co-movement, never per-asset variance (MLAM §2; same invariant
    detone_covariance already enforced — the two paths now agree)."""
    from quantcore.covariance.rmt import denoise_covariance_full

    returns = _factor_returns(seed=11)
    res = denoise_covariance_full(returns)
    sample_var = np.var(returns, axis=0, ddof=1)
    np.testing.assert_allclose(np.diag(res.cov_denoised), sample_var, rtol=1e-10)


def test_denoised_cov_stays_symmetric_psd() -> None:
    from quantcore.covariance.rmt import denoise_covariance_full

    res = denoise_covariance_full(_factor_returns(seed=3))
    np.testing.assert_allclose(res.cov_denoised, res.cov_denoised.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(res.corr_denoised)
    assert eigvals.min() > -1e-10, f"renormalization broke PSD: min eig {eigvals.min()}"
