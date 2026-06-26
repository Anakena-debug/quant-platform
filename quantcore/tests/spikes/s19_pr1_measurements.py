"""S19 PR1 pre-emission measurements (one-shot, not pytest-discovered).

Measures three values used to pin Step-3 tests:
  A. Wishart σ² recovery — pin |σ²_hat - 1| with measured + 50% margin
  B. RMT Frobenius ratio — pin ratio < measured + 20% margin
  C. LW δ̂ oracle on three (N, T) regimes — pin byte-exact via abs=1e-6

Run:
  uv run --directory quantcore python tests/spikes/s19_pr1_measurements.py

Output: prints to stdout AND writes ``s19_pr1_recorded.json`` next to
this file. Step-3 tests load that JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantcore.covariance import (
    denoise_covariance,
    fit_mp_sigma2,
    ledoit_wolf_shrinkage,
)


def _measure_spec_a() -> dict:
    """Spec A — Wishart σ² recovery (N=50, T=1000, σ²=1, seed=20260503)."""
    seed = 20260503
    rng = np.random.default_rng(seed)
    n_samples, n_features = 1000, 50
    sigma2_true = 1.0
    Z = rng.standard_normal((n_samples, n_features)) * np.sqrt(sigma2_true)
    cov_sample = np.cov(Z, rowvar=False)
    std = np.sqrt(np.diag(cov_sample))
    corr = cov_sample / np.outer(std, std)
    eigvals = np.linalg.eigvalsh(corr)
    q = n_features / n_samples
    sigma2_hat = fit_mp_sigma2(eigvals, q)
    abs_err = abs(sigma2_hat - sigma2_true)
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "q": q,
        "sigma2_true": sigma2_true,
        "sigma2_hat": float(sigma2_hat),
        "abs_err": float(abs_err),
        "seed": seed,
    }


def _measure_spec_b() -> dict:
    """Spec B — Frobenius ratio (N=50, T=1000, total spike spectrum=[5,3,2]).

    Σ_true = L · Λ · L^T + I where Λ = diag([4, 2, 1]). The +I floor
    lifts both the bulk noise eigenvalues (0 → 1) AND the signal
    eigenvalues ([4, 2, 1] → [5, 3, 2]). The effective spike spectrum
    above the noise floor is therefore [5, 3, 2] as intended.

    Two independent RNGs (rng_sigma for Σ_true sampling, rng_returns
    for the returns draw) — keeps each measurement self-contained.
    """
    seed = 20260503
    rng_sigma = np.random.default_rng(seed)
    rng_returns = np.random.default_rng(seed + 1)
    n_samples, n_features = 1000, 50
    spike_eigenvalues_added = [4.0, 2.0, 1.0]  # values inside diag(Λ)
    spike_eigenvalues_total = [5.0, 3.0, 2.0]  # post +I lift
    K = len(spike_eigenvalues_added)
    A = rng_sigma.standard_normal((n_features, K))
    L, _ = np.linalg.qr(A)  # orthonormal columns
    Lambda = np.diag(spike_eigenvalues_added)
    Sigma_true = L @ Lambda @ L.T + np.eye(n_features)
    chol = np.linalg.cholesky(Sigma_true)
    Z = rng_returns.standard_normal((n_samples, n_features))
    returns = Z @ chol.T
    cov_sample = np.cov(returns, rowvar=False)
    cov_denoised = denoise_covariance(returns)
    err_sample = float(np.linalg.norm(cov_sample - Sigma_true, ord="fro"))
    err_denoised = float(np.linalg.norm(cov_denoised - Sigma_true, ord="fro"))
    ratio = err_denoised / err_sample
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "spike_eigenvalues_added": spike_eigenvalues_added,
        "spike_eigenvalues_total": spike_eigenvalues_total,
        "err_sample_fro": err_sample,
        "err_denoised_fro": err_denoised,
        "ratio": float(ratio),
        "seed_sigma": seed,
        "seed_returns": seed + 1,
    }


def _measure_spec_c() -> dict:
    """Spec C — LW δ̂ oracle on three (N, T) regimes."""
    regimes = [
        ("well_conditioned", 10, 1000, 20260503),
        ("marginal", 50, 100, 20260504),
        ("ill_conditioned", 100, 80, 20260505),
    ]
    out: dict[str, dict] = {}
    for name, n_features, n_samples, seed in regimes:
        # Single-RNG pattern intentional — test mirrors measurement script
        # to defend against a future "helpful" refactor to dual RNGs that
        # breaks the oracle pin.
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n_features, n_features))
        Sigma_true = A @ A.T / n_features + 0.1 * np.eye(n_features)
        chol = np.linalg.cholesky(Sigma_true)
        Z = rng.standard_normal((n_samples, n_features))
        returns = Z @ chol.T
        cov_lw, delta_hat = ledoit_wolf_shrinkage(returns)
        out[name] = {
            "n_features": n_features,
            "n_samples": n_samples,
            "q": n_features / n_samples,
            "seed": seed,
            "delta_hat": float(delta_hat),
            "cov_lw_trace": float(np.trace(cov_lw)),
        }
    return out


def main() -> None:
    spec_a = _measure_spec_a()
    spec_b = _measure_spec_b()
    spec_c = _measure_spec_c()

    bar = "=" * 70
    print(bar)
    print(
        f"Spec A — Wishart σ² recovery (N={spec_a['n_features']}, "
        f"T={spec_a['n_samples']}, q={spec_a['q']:.4f})"
    )
    print(f"  σ²_hat        = {spec_a['sigma2_hat']:.10f}")
    print(f"  |σ²_hat - 1|  = {spec_a['abs_err']:.10f}")
    print(
        f"  → suggested test pin: assert |σ²_hat - 1| < "
        f"{1.5 * spec_a['abs_err']:.6f}  (measured + 50% margin)"
    )
    print()
    print(
        f"Spec B — Frobenius ratio (N={spec_b['n_features']}, "
        f"T={spec_b['n_samples']}, total spikes={spec_b['spike_eigenvalues_total']})"
    )
    print(f"  ‖sample - true‖_F   = {spec_b['err_sample_fro']:.10f}")
    print(f"  ‖denoised - true‖_F = {spec_b['err_denoised_fro']:.10f}")
    print(f"  ratio (denoised/sample) = {spec_b['ratio']:.10f}")
    print(
        f"  → suggested test pin: assert ratio < "
        f"{1.2 * spec_b['ratio']:.6f}  (measured + 20% margin)"
    )
    print()
    print("Spec C — LW δ̂ oracle (3 regimes, byte-exact pins):")
    for name, vals in spec_c.items():
        print(
            f"  {name:18s}: N={vals['n_features']:3d}, "
            f"T={vals['n_samples']:4d}, q={vals['q']:.4f}, "
            f"seed={vals['seed']}, δ̂={vals['delta_hat']:.10f}"
        )
    print("  → test pins: pytest.approx(δ̂, abs=1e-6)")
    print(bar)

    output_path = Path(__file__).parent / "s19_pr1_recorded.json"
    output_path.write_text(
        json.dumps(
            {"spec_a": spec_a, "spec_b": spec_b, "spec_c": spec_c},
            indent=2,
        )
    )
    print(f"\nRecorded values written to: {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
