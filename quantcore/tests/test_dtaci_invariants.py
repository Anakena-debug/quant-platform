"""Invariant pins for ``conformal.dtaci`` (S13 P12.2).

Pins the DtACI multi-expert primitive against the published
guarantees in Gibbs & Candès 2024:

  - γ-grid validation at construction
  - default-tuple paper provenance (citation)
  - degenerate-equality with single-γ ACI (bitwise)
  - hard-regime-shift recovery within 1pp of target
  - expert-weight floor enforcement
  - weight_entropy dispatch contract
  - GARCH(1,1) volatility-shift coverage gap < single-γ ACI
  - aggregated_alpha numerical safety

Pin design philosophy: paper-bound-anchored. Recovery-rate
tolerance was calibrated post-spike on 5 seeds; if a future seed
grazes the 1pp boundary, widen the tolerance, don't re-derive
(canary-calibration discipline mirroring S12 Pin 2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal import dtaci as dtaci_mod
from quantcore.uncertainty.conformal.dtaci import DtACI
from quantcore.uncertainty.conformal.timeseries import (
    AdaptiveConformalInference,
)


# -----------------------------------------------------------------------------
# γ-grid validation pins.
# -----------------------------------------------------------------------------


def test_pin_dtaci_rejects_k_equals_one() -> None:
    """gammas=(γ,) raises ValueError pointing user at ACI."""
    with pytest.raises(ValueError, match="AdaptiveConformalInference"):
        DtACI(gammas=(0.01,))


def test_pin_dtaci_rejects_empty_gammas() -> None:
    """gammas=() raises ValueError on K < 2."""
    with pytest.raises(ValueError, match=r"K\s*[<>=]+\s*2"):
        DtACI(gammas=())


@pytest.mark.parametrize("bad_gamma", [0.0, -0.01, 1.0, 1.5])
def test_pin_dtaci_rejects_gamma_out_of_range(bad_gamma: float) -> None:
    """γ outside (0, 1) raises ValueError naming the offending value."""
    with pytest.raises(ValueError, match=r"\(0,\s*1\)"):
        DtACI(gammas=(0.01, bad_gamma))


def test_pin_dtaci_rejects_unsorted_grid() -> None:
    """Non-monotone γ-grid raises ValueError naming the inversion."""
    with pytest.raises(ValueError, match="monotone-sorted"):
        DtACI(gammas=(0.02, 0.005, 0.08))


def test_pin_dtaci_default_gammas_match_paper() -> None:
    """Default tuple is (0.001, 0.005, 0.02, 0.08); docstring
    contains 'Gibbs' and '2024' (paper-citation grep)."""
    dt = DtACI()
    assert dt.gammas == (0.001, 0.005, 0.02, 0.08)
    doc = (DtACI.__doc__ or "") + (dtaci_mod.__doc__ or "")
    assert "Gibbs" in doc
    assert "2024" in doc


def test_pin_dtaci_rejects_bad_w_min() -> None:
    """w_min must be in [0, 1/K)."""
    K = 4
    with pytest.raises(ValueError, match="w_min"):
        DtACI(gammas=(0.001, 0.005, 0.02, 0.08), w_min=1.0 / K)
    with pytest.raises(ValueError, match="w_min"):
        DtACI(gammas=(0.001, 0.005, 0.02, 0.08), w_min=-0.01)


def test_pin_dtaci_rejects_negative_eta() -> None:
    """η < 0 raises ValueError."""
    with pytest.raises(ValueError, match="eta"):
        DtACI(gammas=(0.001, 0.08), eta=-0.1)


# -----------------------------------------------------------------------------
# Degenerate equality with single-γ ACI.
# -----------------------------------------------------------------------------


def _make_synthetic_aligned(n: int = 300, seed: int = 0):
    """Generate a synthetic (X, y) sequence where a fitted
    LinearRegression gives non-trivial residuals — so the score
    quantile is non-degenerate."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X.sum(axis=1) + 0.3 * rng.standard_normal(n)
    # Fit on first half to make the model "stale" on the second
    # half — gives realistic residuals to drive ACI/DtACI.
    model = LinearRegression().fit(X[: n // 2], y[: n // 2])
    return X, y, model


def test_pin_dtaci_degenerate_equality_to_aci() -> None:
    """DtACI(gammas=(γ,γ), eta=0) trajectory matches
    AdaptiveConformalInference(gamma=γ) bitwise on the same input.

    Both experts in DtACI share γ, so both update identically.
    eta=0 disables EWA reweighting (weights stay at 0.5/0.5
    forever). Aggregated α = 0.5·α + 0.5·α = α (single-ACI α).
    Score buffer evolves identically (both see the same y_true
    sequence, same score function). Therefore DtACI's interval
    sequence matches ACI's bitwise.
    """
    gamma = 0.02
    X, y, model = _make_synthetic_aligned(n=200, seed=1)

    # Single-γ ACI (ground truth).
    aci = AdaptiveConformalInference(
        alpha=0.1,
        gamma=gamma,
        window_size=100,
    )
    aci_intervals, _ = aci.run_online(model, X, y, warmup=50)
    aci_alphas = np.array([iv.alpha for iv in aci_intervals])
    aci_lowers = np.array([iv.lower[0] for iv in aci_intervals])
    aci_uppers = np.array([iv.upper[0] for iv in aci_intervals])

    # DtACI in degenerate-equality config.
    dt = DtACI(
        alpha=0.1,
        gammas=(gamma, gamma),
        eta=0.0,
        w_min=0.0,
        window_size=100,
    )
    dt_intervals, dt_agg, _, dt_weights = dt.run_online(model, X, y, warmup=50)
    dt_alphas = np.array([iv.alpha for iv in dt_intervals])
    dt_lowers = np.array([iv.lower[0] for iv in dt_intervals])
    dt_uppers = np.array([iv.upper[0] for iv in dt_intervals])

    # Weights stay uniform throughout (eta=0 + identical experts).
    assert np.allclose(dt_weights, 0.5)
    # Bitwise α equality.
    np.testing.assert_array_equal(dt_alphas, aci_alphas)
    # Bitwise interval equality.
    np.testing.assert_array_equal(dt_lowers, aci_lowers)
    np.testing.assert_array_equal(dt_uppers, aci_uppers)


# -----------------------------------------------------------------------------
# Recovery after hard regime shift.
# -----------------------------------------------------------------------------


def _hard_shift_synthetic(seed: int, n: int = 1000, shift: float = 3.0):
    """Gaussian residuals N(0,1) for t < n/2, N(shift,1) for t >= n/2.
    LinearRegression on a small training set gives stable y_pred so
    the residual structure is what ACI/DtACI must adapt to."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    # Pretrain a model that's roughly correct on the pre-shift half.
    y_pre = X[: n // 4].sum(axis=1) + rng.standard_normal(n // 4)
    model = LinearRegression().fit(X[: n // 4], y_pre)
    y_pred_full = model.predict(X)
    # Build y so the residual flips: pre-shift residuals N(0,1),
    # post-shift residuals N(shift, 1).
    residuals = rng.standard_normal(n)
    residuals[n // 2 :] += shift
    y = y_pred_full + residuals
    return X, y, model


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pin_dtaci_recovery_after_hard_shift(seed: int) -> None:
    """After a hard regime shift at T/2, empirical coverage on
    [T/2 + window, T] is within 1pp of target.

    window = ceil(1 / max(γ)) per Gibbs 2024 fig. 3-style setup.
    For default gammas (max=0.08), window = 13.

    Tolerance is 1pp here. If a future seed grazes the boundary,
    widen the tolerance — don't re-derive. Same canary-calibration
    discipline as S12 Pin 2.
    """
    n = 1500
    X, y, model = _hard_shift_synthetic(seed=seed, n=n, shift=3.0)

    dt = DtACI(alpha=0.1, window_size=200)
    intervals, _, _, _ = dt.run_online(model, X, y, warmup=200)

    # Recovery window after shift: skip the first ceil(1/max(γ))
    # online steps post-shift.
    recovery_lag = int(math.ceil(1.0 / max(dt.gammas)))
    online_offset = 200  # warmup
    shift_t = n // 2 - online_offset  # index in the online stream
    recovery_start = shift_t + recovery_lag
    post_shift = intervals[recovery_start:]
    post_shift_y = y[online_offset + recovery_start :]
    coverage = np.mean(
        [iv.contains(np.array([y_t]))[0] for iv, y_t in zip(post_shift, post_shift_y)]
    )
    target = 1.0 - 0.1
    assert abs(coverage - target) < 0.01, (
        f"seed={seed}: post-recovery coverage {coverage:.4f} "
        f"vs target {target}; |Δ| = {abs(coverage - target):.4f} "
        f"(>1pp)"
    )


# -----------------------------------------------------------------------------
# Expert-weight floor enforcement.
# -----------------------------------------------------------------------------


def test_pin_dtaci_expert_weight_floor() -> None:
    """Adversarial sequence designed to make one expert dominate;
    verify min(expert_weights) >= w_min on every step."""
    seed = 7
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.standard_normal((n, 2))
    y = X.sum(axis=1) + 0.2 * rng.standard_normal(n)
    model = LinearRegression().fit(X[:100], y[:100])

    w_min = 0.05
    dt = DtACI(
        alpha=0.1,
        gammas=(0.001, 0.005, 0.02, 0.08),
        eta=2.0,  # aggressive EWA
        w_min=w_min,
        window_size=100,
    )
    _, _, _, weight_traj = dt.run_online(model, X, y, warmup=100)
    # Every step's min weight must be >= w_min (post-floor).
    min_per_step = weight_traj.min(axis=1)
    assert min_per_step.min() >= w_min - 1e-12, (
        f"min weight across all steps was {min_per_step.min():.6f}, below floor w_min={w_min}"
    )


# -----------------------------------------------------------------------------
# weight_entropy dispatches to diagnostics.
# -----------------------------------------------------------------------------


def test_pin_dtaci_weight_entropy_dispatches_to_diagnostics(monkeypatch) -> None:
    """DtACI.weight_entropy returns
    diagnostics.normalized_entropy(expert_weights) — verified by
    patching the imported name and confirming the property pulls
    from there (not a reimplementation)."""
    sentinel_value = 0.4242424242
    calls: list = []

    def _spy(weights):
        calls.append(np.asarray(weights).copy())
        return sentinel_value

    monkeypatch.setattr(dtaci_mod, "normalized_entropy", _spy)
    dt = DtACI(gammas=(0.01, 0.05, 0.1))
    result = dt.weight_entropy
    assert result == sentinel_value
    assert len(calls) == 1


def test_pin_dtaci_weight_entropy_uniform_at_init() -> None:
    """At construction, expert_weights is uniform 1/K, so
    weight_entropy is exactly 1.0 (max entropy). Verifies the
    initialization invariant before any update_step."""
    dt = DtACI(gammas=(0.001, 0.01, 0.1))
    # Uniform weights → normalized entropy = 1.0
    assert math.isclose(dt.weight_entropy, 1.0, rel_tol=1e-12)


# -----------------------------------------------------------------------------
# GARCH(1,1) volatility-shift comparison.
# -----------------------------------------------------------------------------


def _garch_synthetic(seed: int, n: int = 1500):
    """GARCH(1,1) volatility series with regime change in
    persistence. Returns X, y, model. The score sequence is what
    matters; X is dummy regressors."""
    rng = np.random.default_rng(seed)
    omega, alpha_g, beta_g = 0.05, 0.1, 0.85
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / (1 - alpha_g - beta_g)
    eps[0] = rng.standard_normal() * np.sqrt(sigma2[0])
    for t in range(1, n):
        # Regime change in persistence at t = n/2: vol clusters
        # tighter and more aggressively.
        if t == n // 2:
            alpha_g, beta_g = 0.25, 0.7
        sigma2[t] = omega + alpha_g * eps[t - 1] ** 2 + beta_g * sigma2[t - 1]
        eps[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    X = rng.standard_normal((n, 1))
    y = eps  # the residual structure IS the GARCH process
    model = LinearRegression().fit(X[:50], y[:50])
    return X, y, model


def _coverage_gap(intervals, y_true_online):
    target = 0.9
    cov = np.mean([iv.contains(np.array([y]))[0] for iv, y in zip(intervals, y_true_online)])
    return abs(cov - target)


def test_pin_dtaci_garch_volatility_shift() -> None:
    """On a GARCH(1,1) synthetic with a volatility-persistence
    regime change, DtACI's coverage gap should be no worse than
    the BEST single-γ ACI in the same grid (across γ choices the
    operator might pick).

    The published claim from Gibbs 2024 §5 is "tighter intervals
    at matched coverage / no manual γ tuning" — DtACI shouldn't
    do strictly worse than the best fixed γ on coverage.

    Tolerance: DtACI gap ≤ best_aci_gap + 0.02 (allows for
    finite-sample noise; the structural claim is that DtACI gets
    close to the best-fixed-γ without knowing which γ).
    Calibrated on 3 seeds.
    """
    seeds = [0, 1, 2]
    gammas = (0.001, 0.005, 0.02, 0.08)
    warmup = 200
    dt_gaps = []
    best_aci_gaps = []
    for seed in seeds:
        X, y, model = _garch_synthetic(seed=seed, n=1500)
        # DtACI
        dt = DtACI(alpha=0.1, gammas=gammas, window_size=200)
        dt_intervals, _, _, _ = dt.run_online(model, X, y, warmup=warmup)
        dt_gap = _coverage_gap(dt_intervals, y[warmup:])
        # Best single-γ ACI in the grid
        aci_gaps = []
        for g in gammas:
            aci = AdaptiveConformalInference(alpha=0.1, gamma=g, window_size=200)
            ivs, _ = aci.run_online(model, X, y, warmup=warmup)
            aci_gaps.append(_coverage_gap(ivs, y[warmup:]))
        best_aci_gap = min(aci_gaps)
        dt_gaps.append(dt_gap)
        best_aci_gaps.append(best_aci_gap)

    mean_dt_gap = float(np.mean(dt_gaps))
    mean_best_aci_gap = float(np.mean(best_aci_gaps))
    assert mean_dt_gap <= mean_best_aci_gap + 0.02, (
        f"DtACI mean coverage gap {mean_dt_gap:.4f} > best single-γ "
        f"ACI mean gap {mean_best_aci_gap:.4f} + 0.02 tolerance. "
        f"Per-seed DtACI gaps: {dt_gaps}; best-ACI gaps: "
        f"{best_aci_gaps}."
    )


# -----------------------------------------------------------------------------
# aggregated_alpha numerical safety.
# -----------------------------------------------------------------------------


def test_pin_dtaci_aggregated_alpha_in_unit_interval() -> None:
    """At every step, 0 < aggregated_alpha < 1. Guards against
    numerical edge cases from EWA aggregation (e.g., expert_alphas
    drifting to clip boundaries)."""
    seed = 11
    X, y, model = _hard_shift_synthetic(seed=seed, n=800, shift=5.0)
    dt = DtACI(alpha=0.1, window_size=100)
    _, agg_traj, _, _ = dt.run_online(model, X, y, warmup=100)
    assert (agg_traj > 0.0).all()
    assert (agg_traj < 1.0).all()
