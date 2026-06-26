"""Unit tests for the S20 high-q covariance fixture (PR2).

Six tests required by the §B PR2 acceptance grep — four parametrised
over the canonical q grid {0.05, 0.2, 0.5, 0.9}, two single-instance:

  1. ``test_high_q_panel_q_realized_matches_q_target``  — parametrised
     5%-tolerance bracket.
  2. ``test_high_q_panel_eigenvalues_above_mp_edge``    — parametrised
     placement vs (1+√q)² σ²; bulk within S19-style 0.7× / 1.3× slack.
  3. ``test_high_q_panel_factor_structure_recoverable`` — parametrised
     30%-rel-err vs ``population_spike_eigs`` (sorted descending).
  4. ``test_high_q_panel_seed_determinism``             — single-q;
     mirrors S19 PR4 test 6.
  5. ``test_high_q_panel_cumsum_matches_loop_form``     — algebra-only
     pin (rtol=0, atol=1e-12); mirrors S19 PR4 test 7.
  6. ``test_high_q_panel_leak_rate_nonzero_at_embargo_0`` — parametrised
     ``mean > 0`` (NOT the F-RP-005 5%-gate; S20 isn't anchoring
     F-RP-005).

F-RP-007 + F-RP-008 file-level enforcement: this file is in the 5-file
negative-grep surface; ``\\b(mu|mu_hat|expected_returns)\\s*=`` and
conformal-axis identifiers are excluded by the sprint-plan acceptance
gate.

Pre-emission verification (Step 2 of S20 PR2): all q ∈ {0.05, 0.2, 0.5,
0.9} produce ``fold_train_naive = 2259 ≥ 100`` and strictly positive
``factor_variances``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The fixture lives in tests/spikes/ (not auto-collected by pytest because
# the filename does not match test_*).  Mirrors the S19 PR4 import pattern
# at tests/test_s19_leak_injection_fixture.py:40-49.
_FIXTURE_PATH = Path(__file__).parent / "spikes" / "s20_high_q_cov_fixture.py"
_spec = importlib.util.spec_from_file_location("s20_high_q_cov_fixture", _FIXTURE_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_FIXTURE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules["s20_high_q_cov_fixture"] = _module
_spec.loader.exec_module(_module)

build_high_q_cov_panel = _module.build_high_q_cov_panel
_solve_n_t_for_q = _module._solve_n_t_for_q
_spike_lambdas_multiplicative_near_mp = _module._spike_lambdas_multiplicative_near_mp


# Canonical S20 q grid.  Pinned in source so a future refactor that
# silently shrinks the grid (e.g., to skip q=0.9 for runtime) fires the
# acceptance grep ``rg -q 'def test_high_q_panel_q_realized_matches_q_target'``
# AND fails this module's parametrisation pin.
Q_GRID = (0.05, 0.2, 0.5, 0.9)
SEED_CANON = 20260502


# ---------------------------------------------------------------------------
# 1. q_realized tracks q_target within 5% tolerance (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_target", Q_GRID)
def test_high_q_panel_q_realized_matches_q_target(q_target: float) -> None:
    """``|q_realized - q_target| / q_target < 0.05`` across the q grid.

    The fixed-T-vary-N strategy solves ``n_assets = round(q_target *
    fold_train_naive)`` with ``fold_train_naive = 2259`` at canonical
    (T=2520, h=11, K=10).  Integer rounding induces a sub-1% gap at
    every grid point — measured rel_err ≤ 4.4e-4 (Step 2 verification);
    the 5% bracket leaves three orders of magnitude of headroom for
    future grid-knob drift.
    """
    panel = build_high_q_cov_panel(q_target=q_target, seed=SEED_CANON)
    md = panel["metadata"]
    rel_err = abs(md["q_realized"] - md["q_target"]) / md["q_target"]
    assert rel_err < 0.05, (
        f"q_realized={md['q_realized']!r} vs q_target={md['q_target']!r} "
        f"(rel_err={rel_err:.6f}) exceeds 5% tolerance.  "
        f"n_assets={md['n_assets']}, fold_train_naive={md['fold_train_naive']}."
    )


# ---------------------------------------------------------------------------
# 2. Top-K sample eigenvalues land above MP edge; bulk within slack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_target", Q_GRID)
def test_high_q_panel_eigenvalues_above_mp_edge(q_target: float) -> None:
    """Top-K sample eigs > (1+√q)² σ²; bulk within 0.7× / 1.3× MP-edge slack.

    The multiplicative-near-MP parameterisation places **population**
    spikes at ``m_j × mp_upper_edge`` with ``m_j = (1.15, 1.35, 1.75)``;
    sample top-K eigenvalues land above the bulk edge with a small BBP
    upward bias.  At canonical seed (Step 2 verification) the smallest
    sample top-K eig at every q exceeds the edge.

    Bulk slack mirrors S19 PR4 test 3: bulk eigs in
    ``[0.7 · mp_lower · σ², 1.3 · mp_upper · σ²]``.  The slack absorbs
    finite-T fluctuations past the population MP edges; tightening it
    here would require reducing seed variance via averaging, which is
    out of scope for a fixture-sanity test.
    """
    panel = build_high_q_cov_panel(q_target=q_target, seed=SEED_CANON)
    md = panel["metadata"]
    returns = panel["returns"]
    sigma2 = md["sigma2"]
    q_realized = md["q_realized"]
    K = md["k_factors"]
    mp_upper = md["mp_upper_edge"]
    mp_lower = (1.0 - np.sqrt(q_realized)) ** 2 * sigma2

    sample_cov = np.cov(returns, rowvar=False, ddof=1)
    sample_eigs = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]

    top_k = sample_eigs[:K]
    bulk = sample_eigs[K:]

    # Top-K must clear the bulk edge.  m_j > 1 ⇒ population spike >
    # mp_upper; sample spike inherits with BBP correction.
    assert (top_k > mp_upper).all(), (
        f"top-K sample eigs {top_k.tolist()} not all above mp_upper={mp_upper:.4f} "
        f"at q={q_target} (q_realized={q_realized:.4f})."
    )

    # Bulk stays within the S19-style 0.7× / 1.3× slack.  At q=0.9
    # mp_lower ≈ 0.0026 → slack lower = 0.00184; bulk_min ≈ 0.011 ✓
    # (per Step 2 measurement).  At q=0.05 bulk_max ≈ 1.467 < 1.3×1.497
    # = 1.946 ✓.
    assert bulk.min() >= 0.7 * mp_lower, (
        f"bulk_min={bulk.min():.6f} below 0.7 × mp_lower = {0.7 * mp_lower:.6f} at q={q_target}."
    )
    assert bulk.max() <= 1.3 * mp_upper, (
        f"bulk_max={bulk.max():.6f} above 1.3 × mp_upper = {1.3 * mp_upper:.6f} at q={q_target}."
    )


# ---------------------------------------------------------------------------
# 3. Factor structure is recoverable to 30% rel err
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_target", Q_GRID)
def test_high_q_panel_factor_structure_recoverable(q_target: float) -> None:
    """Top-K sample eigs match ``population_spike_eigs`` within 30% rel err.

    Both vectors sorted descending; element-wise rel err.  Step 2
    measurement: max rel err is ≈ 4.6% at q=0.05, growing to ≈ 24.8% at
    q=0.9 (BBP upward bias scales with q).  The 30% bracket holds at
    every grid point with measurement headroom.

    A tighter bracket would couple this test to seed-level finite-T
    fluctuations; a looser bracket would invite silent factor-structure
    regressions (e.g., a refactor that swaps factor_variances for
    population_spike_eigs in Σ_true construction).  30% is the S19 PR4
    convention.
    """
    panel = build_high_q_cov_panel(q_target=q_target, seed=SEED_CANON)
    md = panel["metadata"]
    K = md["k_factors"]
    pop_desc = np.sort(np.asarray(md["population_spike_eigs"], dtype=np.float64))[::-1]

    sample_cov = np.cov(panel["returns"], rowvar=False, ddof=1)
    sample_eigs = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]
    sample_top = sample_eigs[:K]

    rel_err = np.abs(sample_top - pop_desc) / pop_desc
    assert (rel_err < 0.30).all(), (
        f"top-K sample eigs {sample_top.tolist()} not within 30% rel err of "
        f"population_spike_eigs (sorted desc) {pop_desc.tolist()} at q={q_target}; "
        f"rel_err={rel_err.tolist()}."
    )


# ---------------------------------------------------------------------------
# 4. Seed determinism (single-q)
# ---------------------------------------------------------------------------


def test_high_q_panel_seed_determinism() -> None:
    """Two builds at canonical seed produce byte-identical returns + true_cov.

    Single-q (``q_target=0.5``) — the algebraic invariant is
    seed-determined, not q-determined; one panel suffices.  Defends
    against silent RNG-stream drift (numpy upgrades, factor-form
    re-orderings) the same way as S19 PR4 test 6.

    Different-seed branch confirms the RNG plumbing is wired (same-seed
    determinism alone could be satisfied by a constant-output bug that
    ignores seed).
    """
    p1 = build_high_q_cov_panel(q_target=0.5, seed=SEED_CANON)
    p2 = build_high_q_cov_panel(q_target=0.5, seed=SEED_CANON)
    np.testing.assert_array_equal(p1["returns"], p2["returns"])
    np.testing.assert_array_equal(p1["true_cov"], p2["true_cov"])

    p3 = build_high_q_cov_panel(q_target=0.5, seed=SEED_CANON + 1)
    assert not np.array_equal(p1["returns"], p3["returns"]), (
        "Different seeds produced identical returns — RNG plumbing is broken."
    )


# ---------------------------------------------------------------------------
# 5. Cumsum vectorisation matches the explicit loop (rtol=0, atol=1e-12)
# ---------------------------------------------------------------------------


def test_high_q_panel_cumsum_matches_loop_form() -> None:
    """Cumsum prefix-sum trick agrees with the explicit loop form bit-tight.

    Pins the algebraic equivalence of the EW h-bar return computation::

        ew_returns = returns.mean(axis=1)
        csum       = concatenate(([0], cumsum(ew_returns)))
        cum_h_ret  = csum[h:n_events + h] - csum[:n_events]

    against the explicit ``[ew[i:i+h].sum() for i in range(n_events)]``
    form.  rtol=0 / atol=1e-12 is bit-tight; any drift is a real
    algebra bug, not floating-point rounding.  Single-q (``q_target=0.5``)
    — the identity is seed-and-shape-deterministic.

    Step 2 measurement: max abs diff ≈ 4.6e-16 (well inside the 1e-12
    atol).
    """
    panel = build_high_q_cov_panel(q_target=0.5, seed=SEED_CANON)
    returns = panel["returns"]
    md = panel["metadata"]
    h = md["horizon"]
    n_events = md["n_events"]

    ew = returns.mean(axis=1)
    csum = np.concatenate(([0.0], np.cumsum(ew)))
    vec = csum[h : n_events + h] - csum[:n_events]
    loop = np.array([ew[i : i + h].sum() for i in range(n_events)], dtype=np.float64)

    np.testing.assert_allclose(vec, loop, rtol=0.0, atol=1e-12)

    # Round-trip through the panel's labels: the fixture takes sign() of
    # the cum_h_ret vector, so labels must equal sign(loop) exactly.
    expected_labels = pd.Series(
        np.sign(loop), index=panel["events"].index, name="label", dtype=np.float64
    )
    pd.testing.assert_series_equal(panel["labels"], expected_labels)


# ---------------------------------------------------------------------------
# 6. Structural overlap rate is strictly positive at every q
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_target", Q_GRID)
def test_high_q_panel_leak_rate_nonzero_at_embargo_0(q_target: float) -> None:
    """``structural_overlap_rate.mean > 0`` at every q grid point.

    NOT the F-RP-005 5%-gate (closed by S19 PR4 on its own canonical
    fixture).  The S20 invariant is just that overlapping labels exist
    so PR1's embargo arithmetic regression and PR3's purge mechanism
    aren't vacuous.

    The structural rate is ``(T, h, K)``-determined (vertical-only
    triple-barrier closed form ``r̄ = 2(h-1)(K-1)/(T-h+1) ≈ 7.17%`` at
    canonical knobs); it is q_target-independent.  Parametrising over
    the q grid is a correctness pin against any future fixture knob
    that would couple it to ``q_target`` (e.g., a sizing strategy that
    varies T).
    """
    panel = build_high_q_cov_panel(q_target=q_target, seed=SEED_CANON)
    rate = panel["structural_overlap_rate"]
    assert rate["mean"] > 0.0, (
        f"structural_overlap_rate.mean={rate['mean']!r} not > 0 at q={q_target}; "
        f"per_fold={rate['per_fold']!r}.  Overlapping labels are required "
        "so PR1's embargo regression and PR3's purge are not vacuous."
    )
