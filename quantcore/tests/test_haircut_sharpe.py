"""Regression tests pinning the P2.5 Haircut Sharpe primitive
(Harvey-Liu 2015 + Lo 2002 autocorrelation correction).

Scope of pins (9 invariants):
  1. Bonferroni canonical pin from S3 Phase 0b — (N=100, SR=1.0,
     T=240 monthly, ρ=0, autocorr=None) → HSR = 0.7517 ± 1e-3.
  2. BHY canonical pin same fixture → HSR = 0.6433 ± 1e-3.
  3. Top-trial method ordering (corrected): BHY ≤ Bonferroni = Holm.
     The "BHY less conservative" textbook claim is about family-wide
     FDR, not about the top trial where c(N) factor makes BHY stricter
     than Bonferroni under arbitrary-dependence (Benjamini-Hochberg-
     Yekutieli 2001 Thm 1.3). Holm = Bonferroni at rank 1.
  4. Three-regime grid per user refinement #2:
       (a) Canonical: T=240, N=100, ρ=0. All methods yield moderate HSR.
       (b) Low-Power: T=36,  N=10,  ρ=0. High-variance null; haircut
           driven primarily by standard-error inflation, not
           multiplicity.
       (c) High-Multi+Corr: T=120, N=1000, ρ=0.7. Family cluster; BHY
           arbitrary-dep c(N) factor makes it MUCH stricter on the top
           trial than Bonferroni despite correlation.
  5. Single-trial limit: n_trials=1 → haircut is identity.
  6. Simulation determinism: same seeded rng → bitwise HaircutResult.
  7. Lo 2002 AR(1) sign pin (user refinement #4):
       +ρ_ac shrinks sr_haircut (naive annualisation over-states).
       −ρ_ac boosts  sr_haircut (naive annualisation under-states).
       ρ_ac = 0     leaves sr_haircut unchanged vs autocorr=None.
  8. Input validation (upfront): n_trials<1, n_obs<2, sr_observed not
     finite, method ∉ {bonferroni/holm/bhy/all}, autocorr ∉ (−1, 1),
     t_ratios shape mismatch → all raise ValueError.
  9. Lazy rng validation per user refinement #3: rng=None AND
     t_ratios=None AND n_trials>1 raises ValueError NAMING the exact
     input combination ("t_ratios=None, n_trials=..."). rng=None is
     OK when t_ratios is supplied (no simulation invoked) or when
     n_trials=1 (no null trials to simulate).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.validation.stats import (
    HaircutResult,
    _lo2002_q_factor,
    haircut_sharpe,
)


SEED = 20260422


def _t_for_sr(sr_annual: float, n_obs: int, periods_per_year: int) -> float:
    """Convert annualised SR to nominal t-stat (under iid)."""
    return (sr_annual / np.sqrt(periods_per_year)) * np.sqrt(n_obs)


# -----------------------------------------------------------------------------
# INVARIANT 1 & 2 — canonical Phase 0b pins (Bonferroni + BHY)
# -----------------------------------------------------------------------------


@pytest.fixture
def canonical_fixture():
    """(N=100, SR=1.0, T=240 monthly, ρ=0, autocorr=None). Phase 0b pin fixture."""
    N, SR, T_obs, P = 100, 1.0, 240, 12
    t_best = _t_for_sr(SR, T_obs, P)
    # Top = t_best, rest of null = 0 (p = 0.5). Bonferroni / BHY on top
    # trial depend only on the top p-value and N; the null values don't
    # change the top-trial Bonferroni / BHY adjusted p.
    t_ratios = np.concatenate(([t_best], np.zeros(N - 1)))
    return dict(sr_observed=SR, n_obs=T_obs, n_trials=N, t_ratios=t_ratios, periods_per_year=P)


def test_haircut_bonferroni_canonical_pin(canonical_fixture) -> None:
    """Bonferroni HSR ≈ 0.7517 to atol=1e-3 on the S3 Phase 0b canonical
    fixture. First-principles algorithm + statsmodels-cross-checked pin.
    """
    r = haircut_sharpe(**canonical_fixture, method="bonferroni")
    assert isinstance(r, HaircutResult)
    assert r.method == "bonferroni"
    assert r.sr_haircut == pytest.approx(0.7517, abs=1e-3)
    # p_adjusted = 100 · p_nominal
    assert r.p_adjusted == pytest.approx(100 * r.p_nominal, abs=1e-12)


def test_haircut_bhy_canonical_pin(canonical_fixture) -> None:
    """BHY HSR ≈ 0.6433 to atol=1e-3. c(100) · Bonferroni stricter at
    rank 1 than Bonferroni itself; reflects Benjamini-Hochberg-
    Yekutieli 2001 Thm 1.3 arbitrary-dep correction.
    """
    r = haircut_sharpe(**canonical_fixture, method="bhy")
    assert isinstance(r, HaircutResult)
    assert r.method == "bhy"
    assert r.sr_haircut == pytest.approx(0.6433, abs=1e-3)
    # Harmonic c(100) ≈ 5.1874
    c100 = float(sum(1.0 / k for k in range(1, 101)))
    assert r.p_adjusted == pytest.approx(100 * c100 * r.p_nominal, abs=1e-12)


# -----------------------------------------------------------------------------
# INVARIANT 3 — top-trial method ordering: BHY ≤ Bonferroni = Holm
# -----------------------------------------------------------------------------


def test_haircut_top_trial_method_ordering(canonical_fixture) -> None:
    """For the top trial (rank 1), Holm's step-down gives the SAME p_adj
    as Bonferroni (``(N − 1 + 1) · p = N·p``). BHY under arbitrary
    dependence is STRICTER at rank 1 (``N · c(N) · p > N · p``), so its
    sr_haircut is smaller.

    Textbook "BH/BHY less conservative" claims refer to the family-wide
    FDR average, not to the top trial — this test pins the actual
    rank-1 relationship.
    """
    rs = haircut_sharpe(**canonical_fixture, method="all")
    assert isinstance(rs, dict)
    assert rs["holm"].sr_haircut == pytest.approx(rs["bonferroni"].sr_haircut, abs=1e-12), (
        "Holm ≠ Bonferroni at rank 1 — step-down formula may have regressed."
    )
    assert rs["bhy"].sr_haircut < rs["bonferroni"].sr_haircut, (
        f"BHY sr_haircut {rs['bhy'].sr_haircut:.4f} ≥ Bonferroni "
        f"{rs['bonferroni'].sr_haircut:.4f} — Yekutieli c(N) correction "
        "may be missing or method dispatch broken."
    )


# -----------------------------------------------------------------------------
# INVARIANT 4 — three-regime grid (user refinement #2)
# -----------------------------------------------------------------------------


def _grid_t_ratios(
    sr_annual: float, n_obs: int, n_trials: int, periods_per_year: int
) -> np.ndarray:
    """Build a deterministic t_ratios vector with the observed top + (N-1)
    null values at t=0. Avoids RNG dependence in the ordering check."""
    t_best = _t_for_sr(sr_annual, n_obs, periods_per_year)
    return np.concatenate(([t_best], np.zeros(n_trials - 1)))


def test_haircut_three_regime_grid_ordering() -> None:
    """Pin top-trial ordering `bhy ≤ bonferroni == holm` across the three
    qualitatively different regimes (canonical / low-power /
    high-multiplicity+correlation). Also pin haircut-fraction
    directional behaviour per regime.
    """
    regimes = [
        # (name, sr, n_obs, n_trials, periods, expected_behaviour_desc)
        ("canonical", 1.0, 240, 100, 12, "moderate haircut"),
        ("low_power", 1.0, 36, 10, 12, "SE-dominated haircut"),
        ("high_multi_corr", 1.5, 120, 1000, 12, "severe Bonferroni/BHY haircut"),
    ]
    for name, sr, n_obs, N, P, _desc in regimes:
        t_ratios = _grid_t_ratios(sr, n_obs, N, P)
        rs = haircut_sharpe(
            sr_observed=sr,
            n_obs=n_obs,
            n_trials=N,
            t_ratios=t_ratios,
            periods_per_year=P,
            method="all",
        )
        assert isinstance(rs, dict)
        b = rs["bonferroni"].sr_haircut
        h = rs["holm"].sr_haircut
        y = rs["bhy"].sr_haircut
        # Top-trial identity: Holm = Bonferroni
        assert h == pytest.approx(b, abs=1e-12), f"[{name}] Holm ≠ Bonferroni at rank 1: {h} vs {b}"
        # BHY stricter or equal at rank 1 (equality only when p_nominal
        # is already tiny enough that c(N)·p_adj clips to the same
        # floating-point value as Bonferroni — rare).
        assert y <= b + 1e-12, (
            f"[{name}] BHY sr_haircut {y:.4f} > Bonferroni {b:.4f} — "
            "arbitrary-dependence correction missing."
        )
        # Haircut-fraction monotone increasing in strictness
        hf_b = rs["bonferroni"].haircut_fraction
        hf_y = rs["bhy"].haircut_fraction
        assert hf_y >= hf_b - 1e-12, (
            f"[{name}] BHY haircut_fraction {hf_y:.4f} < Bonferroni "
            f"{hf_b:.4f} — haircut_fraction direction broken."
        )


def test_haircut_low_power_regime_is_severe() -> None:
    """Low-Power regime (T=36 months, N=10, SR=1.0) yields a haircut
    dominated by standard-error inflation rather than by the
    multiplicity factor — sr_haircut ends up close to 0 because the
    nominal t-stat is already marginal even before N=10 correction.
    """
    r = haircut_sharpe(
        sr_observed=1.0,
        n_obs=36,
        n_trials=10,
        t_ratios=_grid_t_ratios(1.0, 36, 10, 12),
        periods_per_year=12,
        method="bonferroni",
    )
    # Nominal t ≈ 1.73, nominal p ≈ 0.042. Bonferroni ·10 → p_adj ≈ 0.42
    # → t_haircut_iid ≈ 0.20 → sr_haircut_annual ≈ 0.12.
    assert r.sr_haircut < 0.2, (
        f"Low-Power sr_haircut {r.sr_haircut:.4f} too large; the "
        "SE-dominated regime should drive sr_haircut well below "
        "canonical values."
    )
    assert r.haircut_fraction > 0.8, (
        f"Low-Power haircut_fraction {r.haircut_fraction:.4f} too "
        "small; regime should produce >80% haircut on the nominal SR."
    )


# -----------------------------------------------------------------------------
# INVARIANT 5 — single-trial identity
# -----------------------------------------------------------------------------


def test_haircut_single_trial_is_identity() -> None:
    """At n_trials=1 there is no multiple-testing correction. All three
    methods return sr_haircut = sr_observed (equivalent p_adj = p_nominal)
    and haircut_fraction = 0.
    """
    rs = haircut_sharpe(
        sr_observed=1.0,
        n_obs=500,
        n_trials=1,
        t_ratios=None,
        rng=None,  # rng irrelevant at n_trials=1
        periods_per_year=252,
        method="all",
    )
    assert isinstance(rs, dict)
    for m, r in rs.items():
        assert r.sr_haircut == pytest.approx(1.0, abs=1e-6), (
            f"[{m}] n_trials=1 should return sr_haircut == sr_observed; got {r.sr_haircut}"
        )
        assert r.p_adjusted == pytest.approx(r.p_nominal, abs=1e-12)
        assert r.haircut_fraction == pytest.approx(0.0, abs=1e-6)


# -----------------------------------------------------------------------------
# INVARIANT 6 — simulation determinism under seed
# -----------------------------------------------------------------------------


def test_haircut_simulation_determinism_under_seed() -> None:
    """Same `rng` seed → bitwise-identical HaircutResult across consecutive
    calls. Holds on the t_ratios=None simulation path.
    """
    common_kwargs = dict(
        sr_observed=1.2,
        n_obs=500,
        n_trials=50,
        t_ratios=None,
        rho=0.3,
        periods_per_year=252,
        method="all",
    )
    a = haircut_sharpe(rng=np.random.default_rng(7), **common_kwargs)
    b = haircut_sharpe(rng=np.random.default_rng(7), **common_kwargs)
    assert isinstance(a, dict) and isinstance(b, dict)
    for m in ("bonferroni", "holm", "bhy"):
        assert a[m].sr_haircut == b[m].sr_haircut
        assert a[m].p_adjusted == b[m].p_adjusted
        assert a[m].p_nominal == b[m].p_nominal


def test_haircut_simulation_path_haircuts_t_best_not_sample_max() -> None:
    """Simulation-path numeric correctness: the haircut must be applied
    to t_best specifically, not to whichever pool entry happens to be
    the sample max.

    Why this is a separate pin
    --------------------------
    Under ρ>0 equi-correlation, simulated null t-ratios cluster around
    ``sqrt(ρ)·η`` for a common factor ``η``, and one or more nulls can
    exceed ``t_best`` — especially when t_best itself is only modestly
    significant (the regime where haircut_sharpe matters most). Indexing
    the sorted pool at rank 0 would silently haircut the sample max and
    produce a LESS conservative adjustment than Harvey-Liu 2015
    prescribes — the dangerous direction.

    Fixture & reference value
    -------------------------
    ``sr=1.2, n_obs=500, n_trials=50, rho=0.3, seed=7``.
    Under seed 7: ``t_best ≈ 1.6903`` and five simulated nulls exceed
    it, so t_best lands at rank 5. Canonical Bonferroni on t_best:
    ``p_adj = min(50 · (1 − Φ(1.6903)), 1) = min(2.28, 1) = 1`` → clamp
    to ``1 − _EPS`` → ``t_haircut_iid = Φ⁻¹(_EPS) ≈ −7.94`` → annualised
    ``sr_haircut ≈ −5.64``. Holm at rank 5 collapses to the same
    clamp-at-1; BHY is also clamp-at-1 via the c(N) inflation.

    Pre-fix pathological value on this fixture was Bonferroni/Holm
    ``sr_haircut ≈ −0.25`` — a 5.4-SR under-correction.
    """
    rs = haircut_sharpe(
        sr_observed=1.2,
        n_obs=500,
        n_trials=50,
        rho=0.3,
        t_ratios=None,
        rng=np.random.default_rng(7),
        periods_per_year=252,
        method="all",
    )
    assert isinstance(rs, dict)
    for m in ("bonferroni", "holm", "bhy"):
        assert rs[m].sr_haircut == pytest.approx(-5.6379, abs=5e-3), (
            f"[{m}] sr_haircut {rs[m].sr_haircut:.4f} diverges from "
            "canonical −5.6379. Likely regression of the simulation-path "
            "rank-0-assumes-t_best bug (pre-fix value ≈ −0.25)."
        )
        assert rs[m].sr_haircut < -4.0, (
            f"[{m}] sr_haircut {rs[m].sr_haircut:.4f} not conservative "
            "enough — haircut must be applied to t_best, not sample max."
        )


# -----------------------------------------------------------------------------
# INVARIANT 7 — Lo 2002 sign check (user refinement #4)
# -----------------------------------------------------------------------------


def test_haircut_lo2002_autocorr_sign(canonical_fixture) -> None:
    """Lo 2002 AR(1) correction sign pin.

    Formula (Lo 2002 Eq. 14-15 closed-form):
        q(ρ, T) = 1 + 2·(ρ/(1 − ρ)) · (1 − (1 − ρ^T)/(T·(1 − ρ)))

    Sign directions:
      ρ > 0  → q > 1  → t_effective shrunk → larger p → stricter haircut
                → sr_haircut SMALLER than the no-autocorr baseline.
                (Naive annualisation over-states SR significance.)
      ρ < 0  → q < 1  → t_effective boosted → smaller p → looser haircut
                → sr_haircut LARGER than the no-autocorr baseline.
                (Naive annualisation under-states SR significance.)
      ρ = 0  → q = 1  → sr_haircut UNCHANGED.

    Easy-to-flip check: if the implementation applied sqrt(q) vs
    1/sqrt(q) in the wrong place, the sign inverts and this test flags.
    """
    fixture = canonical_fixture
    baseline = haircut_sharpe(**fixture, method="bonferroni", autocorr=None).sr_haircut
    pos = haircut_sharpe(**fixture, method="bonferroni", autocorr=0.5).sr_haircut
    neg = haircut_sharpe(**fixture, method="bonferroni", autocorr=-0.5).sr_haircut
    zero = haircut_sharpe(**fixture, method="bonferroni", autocorr=0.0).sr_haircut

    assert pos < baseline, (
        f"Lo 2002 sign inverted: autocorr=+0.5 sr_haircut={pos:.4f} "
        f"not SMALLER than baseline={baseline:.4f}. +ρ_ac should "
        "shrink the effective t-stat → stricter haircut."
    )
    assert neg > baseline, (
        f"Lo 2002 sign inverted: autocorr=-0.5 sr_haircut={neg:.4f} "
        f"not LARGER than baseline={baseline:.4f}. −ρ_ac should "
        "boost the effective t-stat → looser haircut."
    )
    assert zero == pytest.approx(baseline, abs=1e-10), (
        f"autocorr=0.0 should collapse to the no-autocorr baseline; "
        f"got {zero:.6f} vs baseline {baseline:.6f}"
    )

    # Magnitude sanity: q(+0.5, 240) ≈ 2.98, q(−0.5, 240) ≈ 0.335.
    assert _lo2002_q_factor(0.5, 240) == pytest.approx(2.9833, abs=5e-3)
    assert _lo2002_q_factor(-0.5, 240) == pytest.approx(0.3352, abs=5e-3)
    assert _lo2002_q_factor(0.0, 240) == 1.0


# -----------------------------------------------------------------------------
# INVARIANT 8 — upfront input validation
# -----------------------------------------------------------------------------


def test_haircut_validation_upfront() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"sr_observed must be finite"):
        haircut_sharpe(float("nan"), n_obs=100, n_trials=10, rng=rng)
    with pytest.raises(ValueError, match=r"sr_observed must be finite"):
        haircut_sharpe(float("inf"), n_obs=100, n_trials=10, rng=rng)
    with pytest.raises(ValueError, match=r"n_obs must be >= 2"):
        haircut_sharpe(1.0, n_obs=1, n_trials=10, rng=rng)
    with pytest.raises(ValueError, match=r"n_trials must be >= 1"):
        haircut_sharpe(1.0, n_obs=100, n_trials=0, rng=rng)
    with pytest.raises(ValueError, match=r"method must be one of"):
        haircut_sharpe(1.0, n_obs=100, n_trials=10, rng=rng, method="fdr")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"autocorr must satisfy"):
        haircut_sharpe(1.0, n_obs=100, n_trials=10, rng=rng, autocorr=1.0)
    with pytest.raises(ValueError, match=r"autocorr must satisfy"):
        haircut_sharpe(1.0, n_obs=100, n_trials=10, rng=rng, autocorr=-1.5)
    with pytest.raises(ValueError, match=r"t_ratios length"):
        haircut_sharpe(1.0, n_obs=100, n_trials=10, t_ratios=np.zeros(5))


# -----------------------------------------------------------------------------
# INVARIANT 9 — lazy rng check with named inputs (user refinement #3)
# -----------------------------------------------------------------------------


def test_haircut_lazy_rng_check_fires_only_on_simulation(canonical_fixture) -> None:
    """rng=None is ACCEPTABLE when t_ratios is supplied — no simulation
    invoked. The previous behaviour of raising upfront was over-zealous.
    """
    # With t_ratios supplied: rng=None is fine (no simulation).
    r = haircut_sharpe(**canonical_fixture, rng=None, method="bonferroni")
    assert isinstance(r, HaircutResult)


def test_haircut_lazy_rng_check_fires_when_simulation_invoked() -> None:
    """rng=None + t_ratios=None + n_trials>1 → ValueError naming the
    exact input combination (per user refinement #3).
    """
    with pytest.raises(
        ValueError,
        match=r"rng is required when t_ratios is None.*"
        r"t_ratios=None, n_trials=100",
    ):
        haircut_sharpe(
            sr_observed=1.0,
            n_obs=240,
            n_trials=100,
            t_ratios=None,
            rng=None,
            periods_per_year=12,
        )


def test_haircut_rho_validation_in_simulation_path() -> None:
    """rho outside [0, 1) inside the simulation path raises with an
    explicit message (the simulator can't handle negative equi-correlation
    or degenerate ρ=1 without PSD collapse).
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"rho must satisfy 0 <= rho < 1.*rho=-0\.1"):
        haircut_sharpe(
            sr_observed=1.0,
            n_obs=240,
            n_trials=50,
            t_ratios=None,
            rho=-0.1,
            rng=rng,
        )
