"""P1.2 regression tests for `validation/stats.py` degenerate-input handling.

Pins:

  - All four public functions (`sharpe_ratio`, `sharpe_ratio_stats`,
    `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`) raise
    `ValueError` on near-constant return series instead of silently
    returning `0.0` / `NaN` / `(1.0, 22.83)` "certain skill" values.
  - `sharpe_ratio` raises on `len(x) < 2` (companion fix; closes the
    last silent-failure surface in the module).
  - Four `_legacy_unchecked` private oracles bitwise-reproduce the
    pre-P1.2 behaviour for forensic comparison; emit `DeprecationWarning`.
  - `_assert_non_degenerate` enforces `sd >= 1e-8 * scale(x)` with
    `scale(x) = max(median(|x|), 1.0)`.

Provenance of pinned float values:
deterministic re-execution in `quantcore/.venv`; numpy 2.4.4,
scipy 1.17.1, python 3.11.14; three identical re-runs of
`stats.py@b753e3c`. All values pinned at `atol=0` exact equality.

Discriminators: 18 of 22 tests fail on `main@b753e3c` for intended
reasons. 4 of 22 (Invariant 2) are non-regression baselines that pass
on both main and fix; included to fail-loud against future
over-aggressive `min_rel_std` tightening.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from quantcore.validation.stats import (
    SharpeStats,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_ratio_stats,
)


# =====================================================================
# Fixtures (P1.2 §Hand-executed)
#
# Note: each `_fixture_*` consumes a fresh `default_rng(0)` per call.
# Do not cache at module level — repeated calls must yield the same
# sequence to keep oracle pins reproducible.
# =====================================================================


def _fixture_a() -> np.ndarray:
    """Branch A — literal constant; sd=0 exactly. No RNG required."""
    return np.ones(252)


def _fixture_b() -> np.ndarray:
    """Branch B — noise-dominated near-constant; the "certain skill" fixture.

    Pinned RNG: `np.random.default_rng(0)`. sd ≈ 1.015e-10. Pre-fix
    `sharpe_ratio(x) ≈ 1.56e11`, `PSR=1.0`, `DSR=1.0` — the dangerous
    branch (looks like a publishable positive result).
    """
    return 1.0 + np.random.default_rng(0).normal(0, 1e-10, 252)


def _fixture_c() -> np.ndarray:
    """Realistic returns — non-regression baseline.

    Pinned RNG: `np.random.default_rng(0)`. N(1e-4, 1e-2). sd ≈ 1.015e-2,
    well above the `1e-8 * scale=1` gate. Passes pre-fix and post-fix.
    """
    return np.random.default_rng(0).normal(1e-4, 1e-2, 252)


def _fixture_d() -> np.ndarray:
    """n=1 — companion-fix discriminator for `sharpe_ratio`'s `len < 2` raise."""
    return np.array([0.5])


# =====================================================================
# Pinned values from §0 deterministic probe.
#
# Provenance: captured via `repr()` from
# the executed run (numpy 2.4.4, scipy 1.17.1, python 3.11.14;
# three identical re-runs). Every value below is `atol=0` exact.
#
# Copy-paste discipline: these literals are pasted verbatim from the
# probe stdout. Do not retype. A single-digit typo would produce a
# regression-test failure that looks like a real bug — a 20-minute
# debug cycle for a zero-cost discipline violation. To regenerate
# (e.g., after a scipy upgrade per §Limits), re-run the probe and
# paste new values; do not partially update.
# =====================================================================

EXPECTED_B_SHARPE = 156357575955.66602
EXPECTED_B_SR_STD = 6848475714.618785
EXPECTED_B_SKEW = -0.020052393333825656
EXPECTED_B_KURT = -0.07387842820029267
EXPECTED_B_PSR_P = 1.0
EXPECTED_B_PSR_Z = 22.83100393010148
EXPECTED_B_DSR_P = 1.0
EXPECTED_B_DSR_EMAX = 10783598227.04103


# =====================================================================
# Public-function table for parametrised Invariants 1 and 2.
# `functools.partial` over `lambda` gives a clean repr in `pytest -v`.
# =====================================================================

PUBLIC_FNS = [
    pytest.param(sharpe_ratio, id="sharpe_ratio"),
    pytest.param(sharpe_ratio_stats, id="sharpe_ratio_stats"),
    pytest.param(probabilistic_sharpe_ratio, id="probabilistic_sharpe_ratio"),
    pytest.param(
        functools.partial(deflated_sharpe_ratio, n_trials=10),
        id="deflated_sharpe_ratio_n10",
    ),
]


# =====================================================================
# Invariant 1 — Degenerate fixtures raise `ValueError`. (8 tests)
#
# Discriminator. Pre-fix: 8/8 fail (no raise; `pytest.raises` reports
# DID NOT RAISE). Post-fix: 8/8 pass.
# =====================================================================


@pytest.mark.parametrize(
    "fixture_factory",
    [
        pytest.param(_fixture_a, id="fixture_A_ones"),
        pytest.param(_fixture_b, id="fixture_B_noise_1e_10"),
    ],
)
@pytest.mark.parametrize("fn", PUBLIC_FNS)
def test_inv1_degenerate_fixture_raises(fixture_factory, fn) -> None:
    """Each (degenerate fixture, public fn) pair raises ValueError."""
    x = fixture_factory()
    with warnings.catch_warnings():
        # scipy may emit RuntimeWarning from skew/kurtosis on near-constant
        # input; that is not the failure mode under test.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.raises(ValueError, match="degenerate variance"):
            fn(x)


# =====================================================================
# Invariant 2 — Realistic fixture passes. (4 tests, NON-REGRESSION)
#
# These pass pre-fix and post-fix. Their purpose is forward-stability:
# if a future PR tightens `_MIN_REL_STD` and accidentally rejects
# realistic returns, these tests fail-loud at that PR. Same pattern as
# P1.1's IV4'/IV5' introspection invariants.
# =====================================================================


@pytest.mark.parametrize("fn", PUBLIC_FNS)
def test_inv2_realistic_fixture_passes(fn) -> None:
    """Realistic returns produce finite values from each public fn."""
    x = _fixture_c()
    result = fn(x)
    if isinstance(result, tuple):
        floats = result
    elif isinstance(result, SharpeStats):
        floats = (result.sr, result.sr_std, result.skew, result.kurt)
    else:
        floats = (result,)
    for v in floats:
        assert np.isfinite(v), f"non-finite return from realistic fixture: {v!r}"


# =====================================================================
# Invariant 3 — `sharpe_ratio` raises on len(x) < 2. (1 test)
#
# Companion fix. Discriminator: pre-fix returns 0.0 silently (matches
# the same silent-failure pattern P1.2 is fixing elsewhere in the
# module); post-fix raises ValueError, aligning with
# `sharpe_ratio_stats`'s pre-existing `n < 4` raise.
# =====================================================================


def test_inv3_sharpe_ratio_raises_on_len_lt_2() -> None:
    """`sharpe_ratio(np.array([0.5]))` raises with the precise message.

    Tightened regex: false-pass (a refactor that drops the message)
    is worse than refactor-break in a regression test.
    """
    with pytest.raises(ValueError, match=r"at least 2 observations for sample std"):
        sharpe_ratio(_fixture_d())


# =====================================================================
# Invariant 4 — Legacy oracles bitwise-pin pre-P1.2 behaviour. (2 tests)
#
# Discriminator: oracles don't exist on main → ImportError. Post-fix:
# oracles return values matching §0 probe at atol=0 exact equality.
#
# NaN handled via np.isnan (cannot use == comparison on NaN).
#
# Tuple equality is decomposed into per-component `float()`-cast
# assertions — robust to future refactors that might return np.float64
# or 0-d ndarrays, and gives per-component diff in pytest's traceback.
# =====================================================================


def test_inv4_legacy_oracles_branch_a_bitwise() -> None:
    """All four `_legacy_unchecked` oracles on Fixture A — pin atol=0."""
    from quantcore.validation.stats import (
        _deflated_sharpe_ratio_legacy_unchecked,
        _probabilistic_sharpe_ratio_legacy_unchecked,
        _sharpe_ratio_legacy_unchecked,
        _sharpe_ratio_stats_legacy_unchecked,
    )

    x = _fixture_a()
    with warnings.catch_warnings():
        # Suppress DeprecationWarning emitted by oracles + scipy
        # RuntimeWarning on Fixture A's degenerate skew/kurt.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        assert float(_sharpe_ratio_legacy_unchecked(x)) == 0.0

        s = _sharpe_ratio_stats_legacy_unchecked(x)
        assert float(s.sr) == 0.0
        assert np.isnan(s.sr_std)
        assert np.isnan(s.skew)
        assert np.isnan(s.kurt)
        assert s.n_obs == 252

        psr_p, psr_z = _probabilistic_sharpe_ratio_legacy_unchecked(x)
        assert np.isnan(psr_p)
        assert np.isnan(psr_z)

        dsr_p, dsr_emax = _deflated_sharpe_ratio_legacy_unchecked(x, n_trials=10)
        assert float(dsr_p) == 0.0
        assert np.isnan(dsr_emax)


def test_inv4_legacy_oracles_branch_b_bitwise() -> None:
    """All four `_legacy_unchecked` oracles on Fixture B — pin atol=0."""
    from quantcore.validation.stats import (
        _deflated_sharpe_ratio_legacy_unchecked,
        _probabilistic_sharpe_ratio_legacy_unchecked,
        _sharpe_ratio_legacy_unchecked,
        _sharpe_ratio_stats_legacy_unchecked,
    )

    x = _fixture_b()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        assert float(_sharpe_ratio_legacy_unchecked(x)) == EXPECTED_B_SHARPE

        s = _sharpe_ratio_stats_legacy_unchecked(x)
        assert float(s.sr) == EXPECTED_B_SHARPE
        assert float(s.sr_std) == EXPECTED_B_SR_STD
        assert float(s.skew) == EXPECTED_B_SKEW
        assert float(s.kurt) == EXPECTED_B_KURT
        assert s.n_obs == 252

        psr_p, psr_z = _probabilistic_sharpe_ratio_legacy_unchecked(x)
        assert float(psr_p) == EXPECTED_B_PSR_P
        assert float(psr_z) == EXPECTED_B_PSR_Z

        dsr_p, dsr_emax = _deflated_sharpe_ratio_legacy_unchecked(x, n_trials=10)
        assert float(dsr_p) == EXPECTED_B_DSR_P
        assert float(dsr_emax) == EXPECTED_B_DSR_EMAX


# =====================================================================
# Invariant 5 — Legacy oracles emit `DeprecationWarning`. (4 tests)
#
# Discriminator: oracles don't exist on main → AttributeError on getattr.
# Post-fix: each call emits exactly one DeprecationWarning matching
# the canonical text "pre-P1.2 unchecked-degenerate-input".
#
# Note: `pytest.warns` semantics is "at least 1 matching warning",
# not "exactly 1". The §Algorithm docstring claims exactly-one, so we
# enforce it explicitly via `len(matching) == 1` after capture. The
# `match=` filter is dropped from `pytest.warns` to avoid double-
# filtering (pytest filters once for the ≥1 assertion; we filter once
# for the exact-count assertion — keep the two stages distinct).
# =====================================================================


@pytest.mark.parametrize(
    "oracle_name",
    [
        "_sharpe_ratio_legacy_unchecked",
        "_sharpe_ratio_stats_legacy_unchecked",
        "_probabilistic_sharpe_ratio_legacy_unchecked",
        "_deflated_sharpe_ratio_legacy_unchecked",
    ],
)
def test_inv5_legacy_oracle_emits_deprecation_warning(oracle_name: str) -> None:
    """Each oracle emits exactly one matching DeprecationWarning per call."""
    from quantcore.validation import stats

    oracle = getattr(stats, oracle_name)
    x = _fixture_a()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with pytest.warns(DeprecationWarning) as rec:
            if oracle_name == "_deflated_sharpe_ratio_legacy_unchecked":
                oracle(x, n_trials=10)
            else:
                oracle(x)

    matching = [
        w
        for w in rec
        if issubclass(w.category, DeprecationWarning)
        and "pre-P1.2 unchecked-degenerate-input" in str(w.message)
    ]
    assert len(matching) == 1, (
        f"expected exactly 1 matching DeprecationWarning, got {len(matching)}: "
        f"{[str(w.message) for w in matching]}"
    )


# =====================================================================
# Invariant 6 — `_assert_non_degenerate` threshold edge + scale. (3 tests)
#
# Discriminator: helper doesn't exist on main → ImportError.
# Post-fix: gate behaves per §Method threshold rationale.
#
# Defensive σ-pin: each RNG-seeded fixture asserts the empirical
# `sd(x, ddof=1)` is within 10% of the target σ. This catches a future
# numpy default_rng(0) semantics change before the resulting empirical
# σ silently crosses the gate threshold and inverts the test verdict.
# Per Bessel's correction, sd(s/σ) ≈ 1/√(2(n-1)) ≈ 0.045 for n=252,
# so 10% is ~2.2σ — wide enough to absorb realistic float drift,
# narrow enough to flag an algorithmic shift.
# =====================================================================


def test_inv6_threshold_edge_just_above_passes() -> None:
    """sd ~ 5e-8 passes the 1e-8 * scale=1 gate (~5x margin)."""
    from quantcore.validation.stats import _assert_non_degenerate

    target_sigma = 5e-8
    x = 1.0 + np.random.default_rng(0).normal(0, target_sigma, 252)
    emp_sigma = float(x.std(ddof=1))
    assert abs(emp_sigma - target_sigma) < 0.1 * target_sigma, (
        f"RNG drift: expected σ≈{target_sigma:.2e}, got {emp_sigma:.2e}"
    )
    _assert_non_degenerate(x)  # must not raise


def test_inv6_threshold_edge_just_below_raises() -> None:
    """sd ~ 1e-9 fails the 1e-8 * scale=1 gate (10x below)."""
    from quantcore.validation.stats import _assert_non_degenerate

    target_sigma = 1e-9
    x = 1.0 + np.random.default_rng(0).normal(0, target_sigma, 252)
    emp_sigma = float(x.std(ddof=1))
    assert abs(emp_sigma - target_sigma) < 0.1 * target_sigma, (
        f"RNG drift: expected σ≈{target_sigma:.2e}, got {emp_sigma:.2e}"
    )
    with pytest.raises(ValueError, match="degenerate variance"):
        _assert_non_degenerate(x)


def test_inv6_scale_aware_high_offset_raises() -> None:
    """Scale-aware: at offset=1e6 with sd_target=1e-3, gate fires because
    sd=1e-3 < 1e-8 * scale=1e6 = 1e-2.

    An absolute-mask implementation (`sd <= 1e-12`) would pass this
    silently (1e-3 > 1e-12); the relative gate catches it because the
    median(|x|) scale lifts the threshold to 1e-2. Locks down the scale
    parameter's load-bearing role.

    Gate margin is 10x (sd=1e-3 vs threshold=1e-2); empirical-σ pin
    keeps it stable under RNG drift.
    """
    from quantcore.validation.stats import _assert_non_degenerate

    target_sigma = 1e-3
    x = 1e6 + np.random.default_rng(0).normal(0, target_sigma, 252)
    emp_sigma = float(x.std(ddof=1))
    assert abs(emp_sigma - target_sigma) < 0.1 * target_sigma, (
        f"RNG drift: expected σ≈{target_sigma:.2e}, got {emp_sigma:.2e}"
    )
    with pytest.raises(ValueError, match="degenerate variance"):
        _assert_non_degenerate(x)
