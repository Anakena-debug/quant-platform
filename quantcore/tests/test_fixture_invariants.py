"""Cross-fixture invariant pins (S12 P11.2).

Four pins that EVERY registered TML composition fixture must
satisfy at canonical SNR. Pins are anchored on
CONSTRUCTION-known properties of each DGP, not on V1
measurements (see S12 sprint plan, "Pin design philosophy"
section).

  Pin 1 — gate recall on construction-informative set
  Pin 2 — gate precision (CANARY) — no constructed-noise
          features in selected
  Pin 3 — MDA rank stability via direct dense-rank equality
          across n_repeats=10 vs 20 (skips fixtures whose
          runner does not expose ``mda_n_repeats``)
  Pin 4 — AFML canonical: precision_lift mean > 0 AND
          positive on ≥ 4/5 seeds

Module-scoped ``runs_cache`` fixture builds all per-(fixture,
seed, config) pipelines once. Configs:

  - ``canonical``     — every registered fixture
  - ``canonical_n20`` — only fixtures whose runner accepts
                        ``mda_n_repeats``; used by Pin 3.

The cv_mean / abstain_oos / low-SNR observables are NOT pinned
cross-fixture (they are DGP-specific by S12's Shape B-restricted
findings) — see ``test_fixture_recordings.py`` for the
non-asserting record.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.fixtures.fixture_calibration import (
    FIXTURE_REGISTRY,
    INFORMATIVE_FEATURES_BY_CONSTRUCTION,
    REGISTRY_SEEDS,
    is_noise_feature,
)


# -----------------------------------------------------------------------------
# Module-scoped runs cache: build pipelines once, reuse across pins.
# -----------------------------------------------------------------------------


def _runner_supports_n_repeats(runner) -> bool:
    """Predicate: does this runner expose ``mda_n_repeats`` kwarg?

    Used to gate Pin 3 (V1's runner does not currently expose it;
    extending V1's run_pipeline is out of S12 scope per the
    plan's Out of scope section).
    """
    return "mda_n_repeats" in inspect.signature(runner).parameters


@pytest.fixture(scope="module")
def runs_cache() -> dict[tuple[str, int, str], dict]:
    """Build per-(fixture, seed, config) pipeline runs once.

    Cache keys: ``(fixture_name, seed, config)`` where config is
    ``"canonical"`` for every registered fixture and
    ``"canonical_n20"`` only for fixtures whose runner accepts
    ``mda_n_repeats``.

    Build cost: ~5–8 minutes wall time on a dev machine with two
    fixtures and 5 seeds. Module scope means this cost is paid
    once per pytest invocation regardless of how many pins read
    from the cache.
    """
    cache: dict[tuple[str, int, str], dict] = {}
    for fixture_name, meta in FIXTURE_REGISTRY.items():
        runner = meta.runner
        for seed in REGISTRY_SEEDS:
            cache[(fixture_name, seed, "canonical")] = runner(seed=seed)
            if _runner_supports_n_repeats(runner):
                cache[(fixture_name, seed, "canonical_n20")] = runner(seed=seed, mda_n_repeats=20)
    return cache


# -----------------------------------------------------------------------------
# Pin 1 — gate recall on construction-informative set at canonical SNR.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURE_REGISTRY))
def test_pin_gate_recall_canonical(runs_cache, fixture_name):
    """Pin 1 — every registered fixture's canonical-SNR gate must
    select the constructed-informative feature set on at least 4
    of 5 seeds.

    Subset check via ``frozenset.issubset``: requires
    ``INFORMATIVE_FEATURES_BY_CONSTRUCTION ⊆ set(selected)``. For
    the current registry both fixtures have only ``{"x_info"}``
    as constructed-informative; future multi-factor fixtures
    register a multi-element frozenset and the subset semantics
    naturally extend.

    Calibration evidence (S12 multi-seed, 2026-04-25):
      V1 (AR(1), phi=0.9): 5/5 seeds with x_info ⊆ selected.
      V2-B5 (HMM, p=0.9):  5/5 seeds with x_info ⊆ selected.
    Margin: 1 seed of headroom on the 4/5 floor.
    """
    required = INFORMATIVE_FEATURES_BY_CONSTRUCTION
    n_with_required = sum(
        1
        for seed in REGISTRY_SEEDS
        if required.issubset(set(runs_cache[(fixture_name, seed, "canonical")]["selected"]))
    )
    assert n_with_required >= 4, (
        f"{fixture_name}: gate selected the required informative "
        f"set {set(required)} on {n_with_required}/5 canonical "
        f"seeds; minimum 4/5 required."
    )


# -----------------------------------------------------------------------------
# Pin 2 — CANARY: gate precision (no constructed-noise features at canonical).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURE_REGISTRY))
def test_pin_gate_precision_canonical(runs_cache, fixture_name):
    """Pin 2 — CANARY PIN. Gate-precision contract calibrated at
    V2-B5's empirical floor (3/5 noise-clean seeds) with ZERO
    margin.

    This pin fires DELIBERATELY when ANY of:

      1. The REGISTRY_SEEDS list changes — different seeds may
         produce different noise-feature admission rates on V2-B5.
      2. The V2-B5 DGP definition changes — innovation
         distribution, regime persistence, feature construction.
      3. The gate primitives change — ``importance_gate``, MDA,
         SFI behavior, or t_stat threshold.

    A failure here is NOT a "re-calibrate the floor" signal. It
    is an "investigate what changed" signal. The registry's
    contract is that all fixtures must clear ≥ 3/5 noise-clean
    seeds at canonical SNR; below that floor is a regression of
    the gate's structural noise-rejection capability.

    If a legitimately-different fixture surfaces (e.g., much
    higher dimensionality where 5/5 is unrealistic), the response
    is to add a per-fixture override or a separate registry tier
    — NOT to relax this pin's tolerance silently. Open a
    dedicated sprint for that. The 3/5 floor is a deliberate
    canary, not a soft target.

    Calibration evidence (S12 multi-seed, 2026-04-25):
      V1 (AR(1), phi=0.9): 5/5 noise-clean — 2 seeds of margin.
      V2-B5 (HMM, p=0.9):  3/5 noise-clean — 0 margin (canary).
    """
    n_clean = sum(
        1
        for seed in REGISTRY_SEEDS
        if not any(
            is_noise_feature(f) for f in runs_cache[(fixture_name, seed, "canonical")]["selected"]
        )
    )
    assert n_clean >= 3, (
        f"{fixture_name}: noise features admitted on "
        f"{5 - n_clean}/5 canonical seeds; minimum 3/5 noise-clean "
        f"required (canary calibrated at V2-B5 empirical floor; "
        f"see test docstring before relaxing tolerance)."
    )


# -----------------------------------------------------------------------------
# Pin 3 — MDA rank stability: direct dense-rank equality, n_repeats 10 vs 20.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURE_REGISTRY))
def test_pin_mda_rank_stability(runs_cache, fixture_name):
    """Pin 3 — MDA rank stability for construction-informative
    features under ``n_repeats=10`` vs ``n_repeats=20`` at
    canonical SNR.

    Construction-anchored test: requires that every feature in
    ``INFORMATIVE_FEATURES_BY_CONSTRUCTION`` has identical
    integer dense-rank in the MDA-mean vector at n_repeats=10
    and n_repeats=20, on every seed in ``REGISTRY_SEEDS`` (5/5
    strict). Uses ``pd.Series.rank(method="dense")`` for
    integer-valued ranks even with ties; no float tolerance
    needed.

    Why construction-anchored, not full-vector: the S12 spike
    claimed full-vector dense-rank stability across 5 seeds on
    V2-B5. Empirical re-verification during the P11.2 cache
    build (2026-04-25) showed 3/5 full-vector identical, with
    2/5 seeds (seed=42, seed=123) flipping ranks among
    noise-equivalent features whose mean MDA drops fall within
    ~0.005 of zero. That ordering is RNG fluctuation, not a
    pipeline-stability property. The informative feature(s)
    hold rank-top on 5/5 seeds at both n=10 and n=20: that is
    the structural claim that survives.

    Skipped for fixtures whose runner does not accept
    ``mda_n_repeats`` kwarg (V1 currently — its ``run_pipeline``
    signature does not expose the parameter). A future sprint
    can extend Pin 3 to V1 by adding the kwarg to V1's runner;
    out of S12 scope (V1 fixture file is explicitly NOT modified
    by S12 — see "Out of scope" in the sprint plan).

    Calibration evidence (S12 multi-seed, 2026-04-25; corrected
    post-empirical re-verification):
      V2-B5: 5/5 seeds with x_info dense-rank identical at
             n=10 and n=20 (rank=5/5 = top of MDA in all cases).
             Full-vector dense-rank equality holds on 3/5 seeds
             only; the informative-rank property is what's
             pinned cross-fixture.
      V1:    skipped (kwarg not exposed).
    """
    runner = FIXTURE_REGISTRY[fixture_name].runner
    if not _runner_supports_n_repeats(runner):
        pytest.skip(
            f"{fixture_name}: runner does not accept "
            f"mda_n_repeats kwarg; rank stability untestable on "
            f"this fixture without runner modification (out of "
            f"scope for S12)."
        )
    required = INFORMATIVE_FEATURES_BY_CONSTRUCTION
    for seed in REGISTRY_SEEDS:
        mda_n10_mean = runs_cache[(fixture_name, seed, "canonical")]["mda"]["mean"]
        mda_n20_mean = runs_cache[(fixture_name, seed, "canonical_n20")]["mda"]["mean"]
        common = mda_n10_mean.index.intersection(mda_n20_mean.index)
        ranks_n10 = mda_n10_mean.loc[common].rank(method="dense")
        ranks_n20 = mda_n20_mean.loc[common].rank(method="dense")
        for feature in required:
            assert feature in common, (
                f"{fixture_name} seed={seed}: required informative "
                f"feature {feature!r} missing from MDA mean index "
                f"(common with n=20 run); fixture's MDA call did "
                f"not score this feature."
            )
            r10 = int(ranks_n10[feature])
            r20 = int(ranks_n20[feature])
            assert r10 == r20, (
                f"{fixture_name} seed={seed}: dense-rank of "
                f"construction-informative feature {feature!r} "
                f"differs between n_repeats=10 (rank={r10}) and "
                f"n_repeats=20 (rank={r20}). Pipeline-relevant "
                f"signal stability regressed — informative-"
                f"feature rank should be n_repeats-invariant in "
                f"this range."
            )


# -----------------------------------------------------------------------------
# Pin 4 — AFML canonical: precision_lift > 0 AND sign-stable across seeds.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(FIXTURE_REGISTRY))
def test_pin_afml_canonical_precision_lift(runs_cache, fixture_name):
    """Pin 4 — AFML meta-labeling thesis at canonical SNR.
    ``precision_lift`` (MetaLabeler-on-takes accuracy minus
    primary unconditional accuracy) must be:

      (i)  positive on average across seeds, AND
      (ii) positive on at least 4 of 5 individual seeds.

    Both conditions must hold. A fixture that produces +0.001
    mean lift by happening to produce small-positive lift on
    5/5 seeds passes; one that produces +0.05 mean lift but
    with 2/5 negative seeds fails (mean dragged up by one big
    outlier; sign not stable enough for a structural pin).

    Calibrated for canonical SNR ONLY. The S12 multi-seed spike
    established that ``precision_lift`` sign-flips at low-SNR on
    V2-B5 (mean −0.013), and that low-SNR behavior is NOT
    cross-fixture invariant.

    Calibration evidence (S12 multi-seed, 2026-04-25):
      V1 (AR(1), phi=0.9): mean +0.012 ± 0.006; 5/5 seeds positive.
      V2-B5 (HMM, p=0.9):  mean +0.010 ± 0.003; 5/5 seeds positive.
    Margin: 1 seed of headroom on the 4/5 sign-stability rule.
    """
    lifts = np.array(
        [runs_cache[(fixture_name, seed, "canonical")]["precision_lift"] for seed in REGISTRY_SEEDS]
    )
    assert lifts.mean() > 0, (
        f"{fixture_name}: precision_lift mean across seeds = "
        f"{lifts.mean():+.4f}; AFML thesis requires positive "
        f"mean at canonical SNR. Per-seed: {lifts.tolist()}."
    )
    n_positive = int((lifts > 0).sum())
    assert n_positive >= 4, (
        f"{fixture_name}: precision_lift positive on only "
        f"{n_positive}/5 seeds; minimum 4/5 required for sign "
        f"stability. Per-seed: {lifts.tolist()}."
    )
