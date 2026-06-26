"""S8 P7.1 — Canonical AFML Ch. 3 TML composition pin.

20 pins on the end-to-end pipeline executed by
``tests.fixtures.tml_composition_spike.run_pipeline``. Pins are
grouped:

    Pin 0           · determinism (byte-exact)
    Pin 1  – 7      · plumbing / structural invariants
    Pin 8  – 10     · MDI (ranking + simplex-sum + design-regression)
    Pin 11 – 14     · gate (MDA / SFI / canonical selection / rejection)
    Pin 15 – 17     · end-to-end CV
    Pin 18 – 19     · value-add (abstain rate + precision lift)
    Pin 20          · IS-leakage warning suppression (regression)
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from quantcore.cv.purged_kfold import PurgedKFold
from tests.fixtures.tml_composition_spike import (
    EMBARGO_PCT,
    FIXTURE_SEEDS,
    LOW_SNR_DRIFT_COEF,
    MDA_N_REPEATS,
    N_SPLITS,
    build_fixture,
    run_pipeline,
)


# =============================================================================
# Shared fixture: one run_pipeline call per seed, cached at module scope.
# run_pipeline is expensive (~30–45 s / seed); caching keeps the suite
# under ~4 minutes wall time while letting every pin read arbitrary
# intermediate artefacts.
# =============================================================================


@pytest.fixture(scope="module")
def pipeline_by_seed() -> dict[int, dict[str, object]]:
    return {s: run_pipeline(s) for s in FIXTURE_SEEDS}


# =============================================================================
# Pin 0 · Determinism — byte-exact equality on repeated run_pipeline.
# =============================================================================
# This MUST be the first pin verified. If it fails, every subsequent
# empirical margin in this file is meaningless — triage any CI failure
# against this pin before investigating downstream breakage.


def test_determinism_byte_exact() -> None:
    """run_pipeline(42) produces byte-identical numerical artefacts on
    two consecutive calls. No tolerance.
    """
    out1 = run_pipeline(42)
    out2 = run_pipeline(42)
    # Data-producing steps
    pd.testing.assert_series_equal(out1["close"], out2["close"])
    pd.testing.assert_frame_equal(out1["features"], out2["features"])
    pd.testing.assert_frame_equal(out1["events"], out2["events"])
    pd.testing.assert_frame_equal(out1["labels"], out2["labels"])
    pd.testing.assert_series_equal(out1["weights"], out2["weights"])
    # Importance
    pd.testing.assert_frame_equal(out1["mdi"], out2["mdi"])
    pd.testing.assert_frame_equal(out1["mda"], out2["mda"])
    pd.testing.assert_frame_equal(out1["sfi"], out2["sfi"])
    # Gate + CV
    assert out1["selected"] == out2["selected"]
    assert out1["gate_passed"] == out2["gate_passed"]
    np.testing.assert_array_equal(out1["cv_scores_selected"], out2["cv_scores_selected"])
    np.testing.assert_array_equal(out1["cv_scores_all"], out2["cv_scores_all"])
    np.testing.assert_array_equal(out1["oos_meta_pred"], out2["oos_meta_pred"])
    np.testing.assert_array_equal(out1["oos_primary_pred"], out2["oos_primary_pred"])
    assert out1["abstain_oos"] == out2["abstain_oos"]
    assert out1["precision_lift"] == out2["precision_lift"]
    assert out1["mdi_sum"] == out2["mdi_sum"]


# =============================================================================
# Pin 1 · End-to-end runs without exception.
# =============================================================================


def test_pipeline_runs_end_to_end(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        assert out["seed"] == seed
        assert isinstance(out["events"], pd.DataFrame)
        assert isinstance(out["labels"], pd.DataFrame)
        assert isinstance(out["weights"], pd.Series)
        assert isinstance(out["mdi"], pd.DataFrame)
        assert isinstance(out["mda"], pd.DataFrame)
        assert isinstance(out["sfi"], pd.DataFrame)
        assert isinstance(out["cv_scores_selected"], np.ndarray)
        assert isinstance(out["cv_scores_all"], np.ndarray)


# =============================================================================
# Pin 2 · Event count bounded. Spike: [386, 422] — pin [350, 450].
# =============================================================================


def test_pin2_event_count_bounded(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        assert 350 <= out["n_events"] <= 450, (
            f"seed {seed}: n_events={out['n_events']} outside [350, 450]"
        )


# =============================================================================
# Pin 3 · Bin balance. Spike min class share 0.356; pin ≥ 0.30.
# =============================================================================


def test_pin3_bin_balance(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        n_active = out["n_active"]
        min_class = min(out["bin_minus1"], out["bin_plus1"])
        share = min_class / n_active if n_active > 0 else 0.0
        assert share >= 0.30, (
            f"seed {seed}: min class share {share:.3f} < 0.30 "
            f"(-1={out['bin_minus1']} +1={out['bin_plus1']} active={n_active})"
        )
        assert out["bin_zero"] <= 15, f"seed {seed}: bin_zero={out['bin_zero']} exceeds 15"


# =============================================================================
# Pin 4 · Sample-weight invariants.
# =============================================================================


def test_pin4_weight_invariants(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        w = out["weights"]
        labels = out["labels"]
        n = len(labels)
        assert not w.isna().any(), f"seed {seed}: weights contain NaN"
        assert (w > 0).all(), f"seed {seed}: non-positive weight(s) (min={w.min()})"
        assert w.max() < 10.0, f"seed {seed}: weight max {w.max()} ≥ 10"
        # BootstrapConfig default normalize_weights_to_n=True implies
        # sum(weights) == n_labels (the "n" in normalize_weights_to_n).
        assert math.isclose(w.sum(), n, rel_tol=1e-6), (
            f"seed {seed}: sum(weights)={w.sum()} != n_labels={n}"
        )
        assert list(w.index) == list(labels.index), f"seed {seed}: weights.index != labels.index"


# =============================================================================
# Pin 5 · t1 consistency (plumbing).
# =============================================================================


def test_pin5_t1_consistency(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        labels = out["labels"]
        events = out["events"]
        assert list(labels.index) == list(events.index), (
            f"seed {seed}: labels.index != events.index"
        )
        # Datetime-compatibility: pandas DatetimeIndex or compatible.
        assert pd.api.types.is_datetime64_any_dtype(labels["t1"]), (
            f"seed {seed}: labels['t1'] is not datetime-typed"
        )
        # t1 >= event start time element-wise.
        assert (labels["t1"].to_numpy() >= labels.index.to_numpy()).all(), (
            f"seed {seed}: some t1 < event start"
        )


# =============================================================================
# Pin 6 · bin==0 drop UserWarning fires with correct count.
# =============================================================================


def test_pin6_drop_zero_warning(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        bin_zero = out["bin_zero"]
        if bin_zero == 0:
            # Defensive: if a seed happens to have zero zero-labels,
            # no warning is expected. (All spike seeds have 4–7.)
            assert not any("dropping" in str(w.message) for w in out["pin6_warnings"])
            continue
        # Find the drop-zero warning and verify it names the right count.
        pattern = rf"dropping {bin_zero}\s+zero-labeled"
        drop_matches = [w for w in out["pin6_warnings"] if re.search(pattern, str(w.message))]
        assert len(drop_matches) >= 1, (
            f"seed {seed}: no UserWarning matched {pattern!r}; "
            f"got: {[str(w.message) for w in out['pin6_warnings']]}"
        )


# =============================================================================
# Pin 7 · PurgedKFold first-fold purge + embargo.
# =============================================================================


def test_pin7_purge_and_embargo(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        labels = out["labels"]
        X_a = out["X_a"]
        t1_a = out["t1_a"]
        # Use the ACTIVE subset's labels (matching the actual CV path).
        labels_a = labels.loc[X_a.index]
        n = len(labels_a)
        # PurgedKFold uses int(round(embargo_pct * n)) internally
        # (cv/purged_kfold.py:219) — match that, not ceil.
        # Deviation-log note: plan §Pin 7 specified ⌈0.01·n⌉; actual
        # implementation rounds, so pin follows implementation.
        embargo_n = int(round(EMBARGO_PCT * n))
        cv = PurgedKFold(n_splits=N_SPLITS, t1=t1_a, embargo_pct=EMBARGO_PCT)
        splits = list(cv.split(X_a))
        assert len(splits) == N_SPLITS
        train_idx, test_idx = splits[0]
        assert len(test_idx) > 0, f"seed {seed}: empty test fold"
        assert len(train_idx) > 0, f"seed {seed}: empty train fold"

        test_start = labels_a.index[test_idx[0]]
        test_end_t1 = labels_a.iloc[test_idx]["t1"].max()
        test_end_pos = int(test_idx[-1])
        # Purge: no train sample's label span overlaps test span.
        for i in train_idx:
            i_int = int(i)
            t0_i = labels_a.index[i_int]
            t1_i = labels_a.iloc[i_int]["t1"]
            purged = (t1_i < test_start) or (t0_i > test_end_t1)
            assert purged, (
                f"seed {seed}: train row at pos {i_int} "
                f"(t0={t0_i}, t1={t1_i}) overlaps test span "
                f"[{test_start}, {test_end_t1}]"
            )
            # Embargo: if train is AFTER test, it must be strictly past
            # the embargo buffer.
            if i_int > test_end_pos:
                assert i_int > test_end_pos + embargo_n, (
                    f"seed {seed}: train pos {i_int} inside embargo "
                    f"(test ends pos {test_end_pos}, embargo "
                    f"extends to {test_end_pos + embargo_n})"
                )


# =============================================================================
# Pin 8 · MDI ranking — x_info is top.
# =============================================================================


def test_pin8_mdi_ranks_x_info_first(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        top = out["mdi"]["mean"].idxmax()
        assert top == "x_info", f"seed {seed}: MDI top feature is {top!r}, expected 'x_info'"


# =============================================================================
# Pin 9 · MDI simplex sum invariant.
# Spike: deviation ≡ 0 (floating-point) across all seeds; rtol 1e-3
# has ~1000× slack but guards against future normalization regressions.
# =============================================================================


def test_pin9_mdi_simplex_sum(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        dev = out["mdi_sum_deviation"]
        assert dev < 1e-3, f"seed {seed}: |sum(mdi.mean) - 1| = {dev:.2e} ≥ 1e-3"


# =============================================================================
# Pin 10 · MDI keeps noise at t=2.0 (DESIGN-REGRESSION pin).
# =============================================================================
# This pin fires under TWO conditions:
#   (a) MDI behaviour regressed silently (original purpose), OR
#   (b) Successful MDI debiasing has landed (MDI-AIR / MDI-sample-
#       split follow-up tickets). Case (b) is the DESIRED outcome of
#       future work — when it happens, §Design decision #1 in the
#       S8 plan must be revisited.
# Do NOT weaken this pin without re-running the simplex argument on
# the new MDI implementation. Spike margin is ~18σ (t ∈ [20, 27]).


def test_pin10_mdi_noise_passes_tstat_design_regression(
    pipeline_by_seed,
) -> None:
    for seed, out in pipeline_by_seed.items():
        mdi = out["mdi"]
        for noise in ("x_noise_1", "x_noise_2"):
            mean = float(mdi.loc[noise, "mean"])
            std = float(mdi.loc[noise, "std"])
            assert std > 0, f"seed {seed}: MDI[{noise}].std is 0"
            assert mean / std > 2.0, (
                f"seed {seed}: MDI[{noise}] t={mean / std:.2f} ≤ 2.0 — "
                "MDI's compositional-simplex failure mode no longer "
                "holds. Re-read §Design decision #1 in the S8 plan "
                "before weakening this pin."
            )


# =============================================================================
# Pin 11 · MDA x_info passes gate at t > 2.
# =============================================================================


def test_pin11_mda_xinfo_passes(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        mda = out["mda"]
        mean = float(mda.loc["x_info", "mean"])
        std = float(mda.loc["x_info", "std"])
        assert std > 0, f"seed {seed}: MDA[x_info].std is 0"
        assert mean / std > 2.0, f"seed {seed}: MDA[x_info] t={mean / std:.2f} ≤ 2.0"


# =============================================================================
# Pin 12 · SFI x_info passes gate at t > 2.
# =============================================================================


def test_pin12_sfi_xinfo_passes(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        sfi = out["sfi"]
        mean = float(sfi.loc["x_info", "mean"])
        std = float(sfi.loc["x_info", "std"])
        assert std > 0, f"seed {seed}: SFI[x_info].std is 0"
        assert mean / std > 2.0, f"seed {seed}: SFI[x_info] t={mean / std:.2f} ≤ 2.0"


# =============================================================================
# Pin 13 · Canonical gate {MDA, SFI} selects informative features.
# =============================================================================


def test_pin13_gate_selects_informative(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        selected = set(out["selected"])
        assert "x_info" in selected, f"seed {seed}: x_info not in gate-selected {selected}"
        assert "x_correlated" in selected, (
            f"seed {seed}: x_correlated not in gate-selected {selected}"
        )
        assert out["gate_passed"], (
            f"seed {seed}: gate_passed=False (min_features=1 but selected={selected})"
        )


# =============================================================================
# Pin 14 · Canonical gate rejects noise features in 5/5 seeds.
# =============================================================================


def test_pin14_gate_rejects_noise(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        selected = set(out["selected"])
        assert "x_noise_1" not in selected, (
            f"seed {seed}: x_noise_1 false-positive in gate {selected}"
        )
        assert "x_noise_2" not in selected, (
            f"seed {seed}: x_noise_2 false-positive in gate {selected}"
        )


# =============================================================================
# Pin 15 · MetaLabeler cv_score_purged mean accuracy ≥ 0.70.
# =============================================================================


def test_pin15_cv_mean_floor(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        cv_mean = out["cv_mean_selected"]
        assert cv_mean >= 0.70, f"seed {seed}: cv_mean_selected={cv_mean:.3f} < 0.70"


# =============================================================================
# Pin 16 · Composition preserves signal (one-sided).
# =============================================================================


def test_pin16_pruning_preserves_signal(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        cv_sel = out["cv_mean_selected"]
        cv_all = out["cv_mean_all"]
        assert cv_sel >= cv_all - 0.05, (
            f"seed {seed}: cv_mean_selected={cv_sel:.3f} < "
            f"cv_mean_all={cv_all:.3f} - 0.05 (pruning destroyed "
            f"> 5pp of signal)"
        )


# =============================================================================
# Pin 17 · Seed-to-seed stability.
# =============================================================================


def test_pin17_seed_stability(pipeline_by_seed) -> None:
    cv_means = np.array([pipeline_by_seed[s]["cv_mean_selected"] for s in FIXTURE_SEEDS])
    cross_seed_std = float(cv_means.std(ddof=1))
    assert cross_seed_std < 0.10, (
        f"cross-seed std of cv_mean_selected = {cross_seed_std:.3f} >= 0.10 (means = {cv_means})"
    )


# =============================================================================
# Pin 18 · OOS abstain rate ∈ [0.03, 0.50] per seed.
# =============================================================================


def test_pin18_abstain_rate_bounds(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        abstain = out["abstain_oos"]
        assert 0.03 <= abstain <= 0.50, (
            f"seed {seed}: abstain_oos={abstain:.3f} outside [0.03, 0.50]"
        )


# =============================================================================
# Pin 19 · AFML thesis — aggregate precision lift > 0 + per-seed
#          floor ≥ -0.01.
# =============================================================================
# Calibrated from spike v4 OOS: per-seed lifts [+0.001, +0.014],
# mean +0.0087, cross-seed σ ≈ 0.005. Aggregate has effect-size 1.7
# (heuristic headroom) and t-stat 3.9 on 4 df against null (lift==0).
# See S8 plan §Pin 19 for triage protocol on first-CI flap.


def test_pin19_precision_lift_afml_thesis(pipeline_by_seed) -> None:
    lifts = np.array([pipeline_by_seed[s]["precision_lift"] for s in FIXTURE_SEEDS])
    # Aggregate: AFML thesis at ensemble level.
    assert lifts.mean() > 0.0, (
        f"aggregate precision_lift mean = {lifts.mean():+.4f} ≤ 0; "
        f"AFML meta-labeling thesis not supported (per-seed lifts: "
        f"{lifts})"
    )
    # Per-seed floor: no seed exhibits destructive interference > 1pp.
    assert lifts.min() >= -0.01, (
        f"per-seed precision_lift min = {lifts.min():+.4f} < -0.01; "
        f"one or more seeds show destructive meta interference "
        f"(per-seed lifts: {lifts})"
    )


# =============================================================================
# Pin 20 · OOS path suppresses IS-leakage warning.
# =============================================================================
# Regex broadened per v2 review nit #4 — pin survives removal of the
# sprint-number "S6" reference from production warnings.


_IS_LEAK_PATTERN = re.compile(r"in[- ]sample|IS[- ]leakage")


def test_pin20_oos_suppresses_is_leakage_warning(pipeline_by_seed) -> None:
    for seed, out in pipeline_by_seed.items():
        offenders = [w for w in out["cv_warnings"] if _IS_LEAK_PATTERN.search(str(w.message))]
        assert not offenders, (
            f"seed {seed}: cv_score_purged emitted "
            f"{len(offenders)} IS-leakage warning(s) under "
            f"meta_features_oos=True: "
            f"{[str(w.message) for w in offenders]}"
        )


# =============================================================================
# Supplementary smoke: build_fixture exposes the documented columns.
# Not a numbered pin, but guards the fixture contract.
# =============================================================================


def test_build_fixture_contract() -> None:
    close, features = build_fixture(42)
    assert isinstance(close, pd.Series)
    assert close.name == "close"
    assert list(features.columns) == ["x_info", "x_noise_1", "x_noise_2", "x_correlated"]
    assert len(close) == len(features) == 1000


def test_mda_n_repeats_constant_is_canonical() -> None:
    # Guard against accidental rollback to the default-3 value.
    # §Design decision #2: canonical is 10 on this fixture class.
    assert MDA_N_REPEATS == 10


# =============================================================================
# S11 P10.3 — Low-SNR graceful-degradation pin (Pin 14).
# =============================================================================
# Spike-determined exact-outcome pin. Spiked 2026-04-25 under
# OMP_NUM_THREADS=1 (CI-config match), seed=42,
# drift_coef=LOW_SNR_DRIFT_COEF (drift/noise=0.5 vs canonical 1.5).
#
# Outcome: A — pipeline completes gracefully with degraded
# performance. Verified deterministic across 3 consecutive runs.
# Tolerance rel=1e-8 abs=1e-10 (BLAS-drift safe; reviewer P1.2
# precedent from S10).


_LOW_SNR_EXPECTED_SELECTED = ["x_correlated", "x_info"]
# Re-pinned 2026-05-17 (S31 — log-return CUSUM migration). Pre-S31 value
# under `close.pct_change()` semantics was 0.5411764705882354; the
# log-return migration shifted `cusum_filter` event timing by a handful
# of bars per seed, propagating to a new low-SNR CV mean.
# The "selected" pin still holds —
# gate behavior is unchanged; only the numerical CV mean drifted.
#
# S38 (F1, 2026-05-30): the MetaLabeler OOS meta-label leak fix re-pinned
# this from 0.5176470588235295 (44/85) to 0.49411764705882355 (42/85). The
# pipeline runs MetaLabeler(meta_features_oos=True); pre-S38 the meta-label
# z was derived from the IN-SAMPLE primary even on the OOS path, inflating
# the low-SNR meta CV accuracy. De-leaking it degrades the weak-signal meta
# ~2 CV samples closer to true chance — exactly the "graceful degradation"
# this test asserts. The "selected" pin is unaffected (the importance-gate
# output did not change), confirming this is the MetaLabeler change, not a
# gate/selection regression.
_LOW_SNR_EXPECTED_CV_MEAN_SELECTED = 0.49411764705882355


def test_low_snr_graceful_degradation() -> None:
    """At drift/noise=0.5 (vs canonical 1.5), the pipeline:
    - completes without raising (graceful degradation),
    - still selects the informative-signal features (x_info,
      x_correlated) — gate doesn't collapse to empty even on weak
      signal,
    - drops cv_mean_selected to ~0.49 (vs canonical >0.75) — meta-
      labeler accuracy near random because primary can barely
      distinguish weak drift from noise (post-S38 F1 leak fix; the
      pre-fix in-sample-z leak previously inflated this to ~0.52),
    - non-trivial bin_zero count (~18 vs canonical ~5) because
      vertical-barrier hits dominate when drift is weak.

    Pin captures the exact post-pipeline values observed at spike
    time. Future regressions register as conscious changes.

    Failure-mode triage:
    - If selected mismatches: gate behavior changed (likely from
      P10.1/P10.2 schema changes); investigate gate inputs.
    - If cv_mean_selected drifts > rel=1e-8: BLAS environment
      changed OR a primitive's numerical output shifted. Try
      OMP_NUM_THREADS=1 first (matches CI config the tolerance
      was sized for); see S11 plan §Watch-items §"Follow-up:
      CI matrix" for the multi-threading drift escalation path.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_pipeline(42, drift_coef=LOW_SNR_DRIFT_COEF)

    # Selection: gate identifies the same informative features as
    # the strong-signal case (x_info + x_correlated). Locks the
    # exact contents AND order — sorted output contract from P10.2
    # Pin 13.
    assert out["selected"] == _LOW_SNR_EXPECTED_SELECTED, (
        f"low-SNR gate selection drifted; expected "
        f"{_LOW_SNR_EXPECTED_SELECTED}, got {out['selected']}"
    )

    # CV mean: graceful degradation to ~chance. Tight tolerance
    # rel=1e-8 abs=1e-10 (BLAS-drift safe per reviewer P1.2).
    assert out["cv_mean_selected"] == pytest.approx(
        _LOW_SNR_EXPECTED_CV_MEAN_SELECTED, rel=1e-8, abs=1e-10
    ), (
        f"low-SNR cv_mean_selected drifted beyond rel=1e-8; "
        f"expected {_LOW_SNR_EXPECTED_CV_MEAN_SELECTED}, "
        f"got {out['cv_mean_selected']!r}"
    )
