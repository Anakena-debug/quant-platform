"""Tests for quantcore.covariance.shrinkage — Ledoit-Wolf wrapper.

Pinned values loaded from tests/spikes/s19_pr1_recorded.json (output of
the pre-emission measurement script tests/spikes/s19_pr1_measurements.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantcore.covariance import ledoit_wolf_shrinkage

_RECORDED = json.loads((Path(__file__).parent / "spikes" / "s19_pr1_recorded.json").read_text())


@pytest.mark.parametrize(
    "regime",
    ["well_conditioned", "marginal", "ill_conditioned"],
)
def test_ledoit_wolf_matches_sklearn_oracle(regime: str) -> None:
    """δ̂ matches recorded sklearn 1.8.x oracle to abs=1e-6.

    Three (N, T) regimes recorded:
      well_conditioned (q=0.01): δ̂ ≈ 0.014  (sample cov already good)
      marginal         (q=0.50): δ̂ ≈ 0.359  (moderate shrinkage)
      ill_conditioned  (q=1.25): δ̂ ≈ 0.583  (heavy shrinkage)
    Recorded values may drift on sklearn upgrades — this test is the
    canary.
    """
    cfg = _RECORDED["spec_c"][regime]
    # Single-RNG pattern intentional — test mirrors measurement script to
    # defend against a future "helpful" refactor to dual RNGs that breaks
    # the oracle pin.
    rng = np.random.default_rng(cfg["seed"])
    n_features = cfg["n_features"]
    n_samples = cfg["n_samples"]
    A = rng.standard_normal((n_features, n_features))
    Sigma_true = A @ A.T / n_features + 0.1 * np.eye(n_features)
    chol = np.linalg.cholesky(Sigma_true)
    Z = rng.standard_normal((n_samples, n_features))
    returns = Z @ chol.T
    cov_lw, delta_hat = ledoit_wolf_shrinkage(returns)
    assert delta_hat == pytest.approx(cfg["delta_hat"], abs=1e-6)
    assert np.trace(cov_lw) == pytest.approx(cfg["cov_lw_trace"], abs=1e-4)


@pytest.mark.parametrize(
    "seed",
    [20260503, 20260504, 20260505, 20260506, 20260507],
)
def test_ledoit_wolf_shrinkage_intensity_in_unit_interval(seed: int) -> None:
    """δ̂ ∈ [0, 1] on randomized fixtures."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((200, 50))
    _, delta_hat = ledoit_wolf_shrinkage(returns)
    assert 0.0 <= delta_hat <= 1.0


def test_ledoit_wolf_high_dim_shrinks_heavily() -> None:
    """At q ≫ 1 (small-N regime), δ̂ lands well above 0.5.

    At q=20, sklearn's LW δ̂ lands ~0.6 — heavy shrinkage but well below
    1. Saturation to 1 would require a more extreme regime (q≫20) or
    non-standard-normal data with degenerate covariance structure. The
    qualitative claim is "high-q triggers heavy shrinkage," not
    "δ̂ → 1."
    """
    rng = np.random.default_rng(20260503)
    n_features, n_samples = 100, 5
    returns = rng.standard_normal((n_samples, n_features))
    _, delta_hat = ledoit_wolf_shrinkage(returns)
    assert delta_hat > 0.5


def test_ledoit_wolf_shrinkage_matches_class_form() -> None:
    """S23b PR3 byte-equal pin: function-form rewrite matches class-form.

    The wrapper was previously implemented via sklearn's class-form
    `LedoitWolf(assume_centered=False).fit(X)` — which eagerly computes
    `self.precision_` via `pinvh(cov)` → `eigh` as a side-effect of
    `EmpiricalCovariance._set_covariance`, consuming ~56% of total
    cs_alpha_nco_backtest wallclock per the PR1 cProfile baseline (see
    `quantstrat/tests/spikes/s23b_pr1_recorded.json` at SHA fa97dfc).
    cs_alpha_nco never reads `precision_`, so PR3 narrows the wrapper to
    sklearn's function-form `ledoit_wolf` which returns `(cov, shrinkage)`
    directly without the precision-matrix construction.

    sklearn's function-form and class-form share the same closed-form
    Ledoit-Wolf 2004 numerics by construction (the class form just
    additionally constructs the precision matrix). This test pins
    byte-equality (atol=0, rtol=0) on a recorded fixture; if a future
    sklearn upgrade silently changes one path's numerics relative to the
    other, the pin catches it.
    """
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(20260509)
    returns = rng.standard_normal((252, 100))  # T=252, N=100

    # Reference: class-form output (the previous implementation)
    lw_ref = LedoitWolf(assume_centered=False).fit(returns)
    cov_ref = lw_ref.covariance_
    shrinkage_ref = float(lw_ref.shrinkage_)

    # New: function-form-based wrapper
    cov_new, shrinkage_new = ledoit_wolf_shrinkage(returns)

    np.testing.assert_array_equal(cov_new, cov_ref)
    assert shrinkage_new == shrinkage_ref
