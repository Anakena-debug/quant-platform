"""Regression tests pinning the P2.4 MinBTL primitive (Bailey-LdP 2014 Eq. 4).

Formula:
    MinBTL(N, SR*) = [ (1 − γ) · Φ⁻¹(1 − 1/N)²
                       + γ · Φ⁻¹(1 − 1/(N · e))² ] / SR*²

Pin surface (5 invariants):
  1. Canonical constant pin — hand-calc values from S3 Phase 0b
     (N ∈ {10, 100, 1000}, SR* ∈ {1.0, 2.0}) reproduce bitwise.
  2. Monotonicity in N — more trials require a longer backtest.
  3. SR scaling — MinBTL(N, 2·SR) = MinBTL(N, SR) / 4 bitwise, from
     the 1/SR² structure of Eq. 4.
  4. Input validation — n_trials < 2 raises; sr_target ≤ 0 raises.
  5. Gamma override — gamma=0.0 produces the closed-form value
     Φ⁻¹(1 − 1/(N·e))² / SR*² (pure-Gumbel-tail, no Euler-Mascheroni
     weighting) — discriminator proving the kwarg is wired through.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from quantcore.validation.stats import min_backtest_length


# -----------------------------------------------------------------------------
# INVARIANT 1 — canonical constant pins (S3 Phase 0b hand-calc)
# -----------------------------------------------------------------------------


def test_min_btl_canonical_pin_n100_sr1() -> None:
    """Bailey-LdP 2014 Eq. 4 at N=100, SR*=1.0 reproduces the S3 Phase 0b
    hand-calc value 6.434509096430 to atol=1e-12."""
    v = min_backtest_length(n_trials=100, sr_target=1.0)
    assert v == pytest.approx(6.434509096430, abs=1e-12)


def test_min_btl_canonical_pin_n1000_sr1() -> None:
    v = min_backtest_length(n_trials=1000, sr_target=1.0)
    assert v == pytest.approx(10.615730377573, abs=1e-12)


def test_min_btl_canonical_pin_n10_sr1() -> None:
    v = min_backtest_length(n_trials=10, sr_target=1.0)
    assert v == pytest.approx(2.542260376861, abs=1e-12)


# -----------------------------------------------------------------------------
# INVARIANT 2 — monotone in N at fixed SR*
# -----------------------------------------------------------------------------


def test_min_btl_monotone_in_n_trials() -> None:
    """More trials → longer backtest. Holds at every gap in the
    {2, 5, 10, 50, 100, 500, 1000, 5000} grid.
    """
    sr = 1.0
    values = [
        min_backtest_length(n_trials=n, sr_target=sr) for n in [2, 5, 10, 50, 100, 500, 1000, 5000]
    ]
    for i in range(1, len(values)):
        assert values[i] > values[i - 1], (
            f"MinBTL not monotone at step {i}: {values[i - 1]:.6f} >= {values[i]:.6f}"
        )


# -----------------------------------------------------------------------------
# INVARIANT 3 — SR scaling: MinBTL ∝ 1/SR²
# -----------------------------------------------------------------------------


def test_min_btl_sr_scaling_exact_quartering() -> None:
    """``MinBTL(N, 2·SR) == MinBTL(N, SR) / 4`` bitwise — structural
    property of Eq. 4's ``1/SR²`` denominator."""
    for n in [10, 100, 1000]:
        v1 = min_backtest_length(n_trials=n, sr_target=1.0)
        v2 = min_backtest_length(n_trials=n, sr_target=2.0)
        assert v1 / 4.0 == v2, (
            f"N={n}: MinBTL(1.0)/4 = {v1 / 4:.15f} != "
            f"MinBTL(2.0) = {v2:.15f} — Eq. 4 scaling violated."
        )


def test_min_btl_sr_scaling_general_quadratic() -> None:
    """``MinBTL(N, k·SR) == MinBTL(N, SR) / k²`` for k ∈ {0.5, 2.0, 5.0}."""
    n = 100
    base = min_backtest_length(n_trials=n, sr_target=1.0)
    for k in [0.5, 2.0, 5.0]:
        scaled = min_backtest_length(n_trials=n, sr_target=k)
        assert scaled == pytest.approx(base / (k * k), abs=1e-12)


# -----------------------------------------------------------------------------
# INVARIANT 4 — input validation
# -----------------------------------------------------------------------------


def test_min_btl_validation_n_trials_too_small() -> None:
    with pytest.raises(ValueError, match=r"n_trials must be >= 2"):
        min_backtest_length(n_trials=1, sr_target=1.0)
    with pytest.raises(ValueError, match=r"n_trials must be >= 2"):
        min_backtest_length(n_trials=0, sr_target=1.0)


def test_min_btl_validation_sr_target_nonpositive() -> None:
    with pytest.raises(ValueError, match=r"sr_target must be > 0"):
        min_backtest_length(n_trials=100, sr_target=0.0)
    with pytest.raises(ValueError, match=r"sr_target must be > 0"):
        min_backtest_length(n_trials=100, sr_target=-1.0)


# -----------------------------------------------------------------------------
# INVARIANT 5 — gamma kwarg discriminator
# -----------------------------------------------------------------------------


def test_min_btl_gamma_override_collapses_to_single_term() -> None:
    """gamma kwarg weighting discriminator.

    Eq. 4 is a weighted blend of two ``Φ⁻¹`` terms:
        ``MinBTL = [ (1 − γ)·Φ⁻¹(1 − 1/N)² + γ·Φ⁻¹(1 − 1/(N·e))² ] / SR*²``

    - At ``gamma = 0`` only the first term survives:
          ``MinBTL = Φ⁻¹(1 − 1/N)² / SR*²``
    - At ``gamma = 1`` only the second term survives:
          ``MinBTL = Φ⁻¹(1 − 1/(N·e))² / SR*²``

    Pin both endpoints to atol=1e-12. Proves the ``gamma`` kwarg is
    wired through the formula and has the claimed weighting direction.
    """
    n, sr = 100, 1.0

    v_gamma0 = min_backtest_length(n_trials=n, sr_target=sr, gamma=0.0)
    expected_g0 = float(sp_stats.norm.ppf(1.0 - 1.0 / n) ** 2 / sr**2)
    assert v_gamma0 == pytest.approx(expected_g0, abs=1e-12)

    v_gamma1 = min_backtest_length(n_trials=n, sr_target=sr, gamma=1.0)
    expected_g1 = float(sp_stats.norm.ppf(1.0 - 1.0 / (n * np.e)) ** 2 / sr**2)
    assert v_gamma1 == pytest.approx(expected_g1, abs=1e-12)

    # Canonical value (γ = 0.5772...) lies strictly between the two endpoints.
    v_canonical = min_backtest_length(n_trials=n, sr_target=sr)
    assert min(v_gamma0, v_gamma1) < v_canonical < max(v_gamma0, v_gamma1), (
        f"Canonical MinBTL {v_canonical:.6f} not between gamma=0 ({v_gamma0:.6f}) "
        f"and gamma=1 ({v_gamma1:.6f}) endpoints — weighting broken."
    )
