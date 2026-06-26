"""Regression tests pinning the P2.3 DSR cross-trial σ̂ kwarg fix (F05 / A-2).

The pre-P2.3 ``deflated_sharpe_ratio`` substituted the single-path PSR
σ̂(SR) of the evaluation return series for both the expected-max term
and the z-score denominator. Bailey-LdP 2014 Eq. 3-4 prescribes the
cross-trial ``V^{1/2}[{SR_n}]`` — the std of SR estimates **across** the
``n_trials`` alternatives. Substitution produces ~5% systematic
over-optimism on Gaussian-null fixtures.

The P2.3 fix adds ``sr_std_cross_trial: float | None = None`` kwarg:
  - when provided: used for both ``E_max`` and the z-denominator;
  - when None:     single-path fallback + ``UserWarning``.

Pin surface (4 invariants):
  1. Canonical pin — audit reference: N=100 trials × T=252 days,
     Gaussian null, seed 0 → DSR p = 0.5927.
  2. Legacy pin — same fixture via
     ``_deflated_sharpe_ratio_legacy_single_path`` → DSR p = 0.6416.
  3. Fallback UserWarning emitted exactly once per call when
     ``sr_std_cross_trial`` is omitted.
  4. Block-bootstrap dependence linkage — AR(1) ρ=0.5 returns: block
     bootstrap σ̂ exceeds IID σ̂; passing block σ̂ into DSR gives
     ``p(block) ≤ p(iid)`` in the overfitting regime (z > 0,
     DSR near 1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quantcore.validation.stats import (
    _deflated_sharpe_ratio_legacy_single_path,
    deflated_sharpe_ratio,
)
from quantcore.weights.block_bootstrap import block_bootstrap


SEED = 0  # matches audit §4.F05 repro seed for bitwise canonical pin


# -----------------------------------------------------------------------------
# INVARIANT 1 — canonical DSR pin (audit §4.F05)
# -----------------------------------------------------------------------------


def test_dsr_canonical_gaussian_null_pin() -> None:
    """N=100 × T=252 Gaussian null, seed 0, best-of-100 SR:
    when ``sr_std_cross_trial`` is the true cross-trial std, DSR p
    must match the audit canonical target 0.5927 to 2 decimal places.
    """
    rng = np.random.default_rng(SEED)
    N_TRIALS, T, PERIODS = 100, 252, 252
    R = rng.standard_normal(size=(N_TRIALS, T))
    srs = (R.mean(axis=1) / R.std(axis=1, ddof=1)) * np.sqrt(PERIODS)
    v_sqrt_cross = float(srs.std(ddof=1))
    best_idx = int(np.argmax(srs))
    r_best = R[best_idx]

    p, emax = deflated_sharpe_ratio(
        r_best,
        n_trials=N_TRIALS,
        sr_std_cross_trial=v_sqrt_cross,
    )

    # Audit pin: 0.5927.
    assert abs(p - 0.5927) < 0.005, (
        f"Canonical DSR p = {p:.4f} diverges from audit pin 0.5927. "
        "F05 fix may have regressed the kwarg handling."
    )
    # E_max sanity — should equal v_sqrt * (CDF weighted) ≈ 2.7023 (audit).
    assert abs(emax - 2.7023) < 0.01


# -----------------------------------------------------------------------------
# INVARIANT 2 — legacy oracle pin (discriminator)
# -----------------------------------------------------------------------------


def test_dsr_legacy_single_path_pin() -> None:
    """Same fixture as Invariant 1 via the legacy oracle reproduces the
    pre-P2.3 inflated DSR p ≈ 0.6416 (audit §4.F05). Discriminator for
    the fix: any caller that still uses single-path σ̂ lands at 0.64.
    """
    rng = np.random.default_rng(SEED)
    N_TRIALS, T, PERIODS = 100, 252, 252
    R = rng.standard_normal(size=(N_TRIALS, T))
    srs = (R.mean(axis=1) / R.std(axis=1, ddof=1)) * np.sqrt(PERIODS)
    best_idx = int(np.argmax(srs))
    r_best = R[best_idx]

    with pytest.warns(DeprecationWarning, match=r"F05"):
        p, _emax = _deflated_sharpe_ratio_legacy_single_path(
            r_best,
            n_trials=N_TRIALS,
        )

    assert abs(p - 0.6416) < 0.005, (
        f"Legacy single-path DSR p = {p:.4f} diverges from audit pin 0.6416. "
        "Oracle reproduction has drifted."
    )


# -----------------------------------------------------------------------------
# INVARIANT 3 — fallback UserWarning when sr_std_cross_trial omitted
# -----------------------------------------------------------------------------


def test_dsr_fallback_emits_user_warning() -> None:
    """Omitting ``sr_std_cross_trial`` triggers a single UserWarning that
    references F05 / Bailey-LdP 2014 Eq. 3-4. The function still
    returns a numerical DSR — the warning is advisory, not a raise.
    """
    rng = np.random.default_rng(SEED)
    returns = rng.standard_normal(252) * 0.01 + 0.001

    with pytest.warns(UserWarning, match=r"sr_std_cross_trial|Eq\. 3-4") as record:
        p, _emax = deflated_sharpe_ratio(returns, n_trials=100)

    # Filter to UserWarnings only (scipy may emit DeprecationWarnings).
    user_warnings = [w for w in record if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, (
        f"Expected exactly 1 UserWarning, got {len(user_warnings)}. F05 fallback pattern broken."
    )
    assert 0.0 <= p <= 1.0


def test_dsr_kwarg_suppresses_warning() -> None:
    """Providing ``sr_std_cross_trial`` does NOT emit the fallback
    UserWarning (only the F05-specific one; other library warnings
    are allowed).
    """
    rng = np.random.default_rng(SEED)
    returns = rng.standard_normal(252) * 0.01 + 0.001

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deflated_sharpe_ratio(
            returns,
            n_trials=100,
            sr_std_cross_trial=1.0,
        )

    # No F05 fallback warning when the kwarg is supplied.
    f05_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and ("sr_std_cross_trial" in str(w.message) or "F05" in str(w.message))
    ]
    assert not f05_warnings, (
        f"Unexpected F05 fallback warning when kwarg provided: "
        f"{[str(w.message) for w in f05_warnings]}"
    )


# -----------------------------------------------------------------------------
# INVARIANT 4 — block-bootstrap dependence linkage (A-4 × A-2)
# -----------------------------------------------------------------------------


def test_dsr_block_bootstrap_dependence_correction() -> None:
    """AR(1) ρ=0.5: block-bootstrap σ̂(SR) exceeds IID σ̂(SR); feeding
    block σ̂ into DSR yields a more conservative p in the overfitting
    regime.

    Coupling advisory A-4 → A-2: block bootstrap (P2.2) is the
    dependence-preserving alternative to IID resampling, and DSR
    (P2.3) accepts the resulting σ̂ through the new kwarg.

    Fixture — "best of many trials" construction to ensure the
    overfitting regime (observed SR > emax, so z > 0, DSR > 0.5):
      - Generate N_TRIALS=100 independent AR(1) ρ=0.5 return paths,
        daily-scale volatility ≈ 1% per day, T=2000 each.
      - Pick the highest-SR path as the candidate "winning strategy".
        By construction the candidate has observed SR well above mean
        (sample max over 100 iid-ish paths).
      - IID bootstrap that path (block_size=1) → σ̂_iid.
      - Circular block bootstrap (block_size=32) → σ̂_block.
      - Compute DSR under both.

    Assertions:
      a. σ̂_block > σ̂_iid (dependence correction; re-pins P2.2 Inv. 2).
      b. p_block ≤ p_iid (conservative direction in overfitting regime).
      c. |p_block - p_iid| > 1e-3 (non-trivial magnitude).
    """
    rng = np.random.default_rng(42)
    N_PATHS, T, rho = 100, 2000, 0.5
    vol = 0.01  # ~1% daily return std (realistic scale)
    eps = rng.standard_normal(size=(N_PATHS, T))
    paths = np.empty((N_PATHS, T))
    paths[:, 0] = eps[:, 0] * vol
    for t in range(1, T):
        paths[:, t] = rho * paths[:, t - 1] + eps[:, t] * vol
    # Best-SR path = the "winning" overfit candidate.
    srs = (paths.mean(axis=1) / paths.std(axis=1, ddof=1)) * np.sqrt(252.0)
    best_idx = int(np.argmax(srs))
    returns = paths[best_idx]

    def _resample_sr_std(block_size: int, seed: int) -> float:
        rep = block_bootstrap(
            returns,
            block_size=block_size,
            n_replicates=300,
            rng=np.random.default_rng(seed),
            circular=True,
        )
        arr = np.asarray(rep)
        mu = arr.mean(axis=1)
        sd = arr.std(axis=1, ddof=1)
        sr = (mu / sd) * np.sqrt(252.0)
        return float(sr.std(ddof=1))

    sigma_iid = _resample_sr_std(block_size=1, seed=SEED + 300)
    sigma_block = _resample_sr_std(block_size=32, seed=SEED + 301)

    # (a) dependence preservation.
    assert sigma_block > sigma_iid, (
        f"AR(1) ρ={rho}: block σ̂(SR)={sigma_block:.4f} not greater than "
        f"IID σ̂(SR)={sigma_iid:.4f}. P2.2 block-bootstrap dependence "
        "pin regressed, or fixture too noisy."
    )

    p_iid, _ = deflated_sharpe_ratio(
        returns,
        n_trials=N_PATHS,
        sr_std_cross_trial=sigma_iid,
    )
    p_block, _ = deflated_sharpe_ratio(
        returns,
        n_trials=N_PATHS,
        sr_std_cross_trial=sigma_block,
    )

    # Sanity: the "best path" construction places us in the overfitting
    # regime (observed SR > emax, so z > 0, DSR > 0.5).
    assert p_iid > 0.5, (
        f"Fixture not in overfitting regime (p_iid={p_iid:.4f} <= 0.5). "
        "The best-of-100 construction should yield observed SR > emax; "
        "if it does not, the AR(1) paths may be too noisy for this "
        "seed or scale."
    )
    # (b) conservative direction.
    assert p_block <= p_iid, (
        f"Block-bootstrap DSR p={p_block:.4f} exceeds IID DSR "
        f"p={p_iid:.4f}. Expected block ≤ IID in overfitting regime "
        "(larger σ̂ shrinks z; if z > 0 that pulls p down toward 0.5)."
    )
    # (c) non-trivial correction magnitude.
    assert abs(p_block - p_iid) > 1e-3, (
        f"Block vs IID DSR correction is trivially small "
        f"(|Δp|={abs(p_block - p_iid):.3e}). Either fixture too short "
        "for block effect to materialise, or σ̂ gap too small."
    )
