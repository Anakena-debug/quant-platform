"""Asserting invariant pins + non-asserting recordings emission for the
S16 empirical-comparison harness (P15.3).

Asserting pins (architectural invariants only — not exact metric numbers):

  * Pin 1 — ``compare_alpha_branches`` returns ``BranchComparisonResult``
    with ``metrics`` keyed by ``(refit_frequency, branch)``; 5 branches
    × 2 cadences = 10 cells.
  * Pin 2 — every (cadence, branch) cell on synthetic A produces
    non-empty per-step arrays of length ``n - initial_train_size``.
  * Pin 3 — same on synthetic B.
  * Pin 4 — dispatch fans to TWO callable targets per cadence
    (``backtest_alpha_model`` × 4, ``backtest_alpha_model_dtaci`` × 1);
    with two cadences the totals are 8 / 2. THIS IS THE ROUTE C
    RECONSIDERATION TRIGGER PIN — dispatch shape stays at two
    callables regardless of cadence count.
  * Pin 5 — caller-passed factories produce fresh objects per
    (cadence, branch) cell; nothing the caller owns is mutated.
  * Pin 6 — ``BranchMetrics`` aggregate fields well-formed
    (coverage∈[0,1], mean_width≥0, sharpe finite) for every cell.
  * Pin 7 — ``synthetic_diagnostics`` exposes finite design /
    realized SNR + realized oracle Sharpe; synthetic B's overall
    Sharpe is strictly less than synthetic A's (sanity check that
    σ × 3 actually produced lower realized SNR).
  * Pin 8 — ``n_refits`` matches
    ``floor((n - initial_train_size - 1) / refit_frequency)`` for
    every cell — moves with the cadence.
  * Pin 9 — DtACI branch on synthetic A at ``refit_frequency=21``
    is byte-equivalent to ``backtest_alpha_model_dtaci`` invoked
    directly with matching parameters (the S15 baseline). Regression
    pin: ensures wrapping DtACI in the harness does not perturb its
    walk-forward output.

Recordings emission (4 non-asserting tests, one per (synthetic,
cadence) cell). Mirrors S13/S14/S15 ``latest.md`` overwrite pattern;
asserts only file existence + non-zero size.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.dtaci import DtACI
from quantcore.uncertainty.conformal.finance import (
    BRANCH_NAMES,
    BranchComparisonResult,
    BranchMetrics,
    SyntheticDiagnostics,
    backtest_alpha_model_dtaci,
    compare_alpha_branches,
)
from quantcore.uncertainty.conformal.finance import empirical_comparison as ec_module


# -----------------------------------------------------------------------------
# Synthetic generators — module-internal helpers, NOT in the harness module.
# Synthetic A is byte-equivalent to test_dtaci_alpha_integration.py:65-70.
# Synthetic B is the σ × 3 low-SNR variant locked in P15.1.
# -----------------------------------------------------------------------------


def _canonical_regime_shift_synthetic(
    seed: int = 11, n: int = 600
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Canonical-SNR regime-shift synthetic. Matches the S15
    test_dtaci_alpha_integration helper exactly: noise σ schedule
    [1.0, 0.3] across the n//2 split."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    noise_scale = np.where(np.arange(n) < n // 2, 1.0, 0.3)
    y = X.sum(axis=1) + noise_scale * rng.standard_normal(n)
    return X, y


def _low_snr_regime_shift_synthetic(
    seed: int = 11, n: int = 600
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """σ × 3 low-SNR variant. Same generator, same seed, same shape;
    only the noise schedule changes from [1.0, 0.3] to [3.0, 0.9]
    (canonical schedule × 3). Operationally low-SNR on this DGP per
    P15.1's decision doc — NOT V2-B5-equivalent."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    noise_scale = np.where(np.arange(n) < n // 2, 3.0, 0.9)
    y = X.sum(axis=1) + noise_scale * rng.standard_normal(n)
    return X, y


def _oracle_predictor(X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Oracle for the regime-shift synthetic family: y = X.sum + noise."""
    return X.sum(axis=1)


def _mondrian_stratifier(X: NDArray[np.floating[Any]]) -> NDArray[np.int_]:
    """Two-stratum stratifier on sign of X[:, 0]. Matches the
    byte-exact pin convention (test_alpha_branches_byte_exact.py)."""
    return (X[:, 0] > 0).astype(np.int_)


def _base_factory() -> LinearRegression:
    return LinearRegression()


def _cqr_factory() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=0)


# Small-synthetic kwargs used for fast invariant pins.
_SMALL_KW: dict[str, Any] = {
    "alpha": 0.1,
    "initial_train_size": 100,
    "refit_frequencies": (20, 50),
    "dtaci_warmup": 30,
}

# Full-synthetic kwargs — match S15 baseline for the byte-equivalence pin
# and the recordings.
REC_SEED: int = 11
REC_N_BARS: int = 600
REC_N_TRAIN: int = 252
REC_WARMUP: int = 50


# =============================================================================
# Pin 1 — metrics keyed by (cadence, branch); 5 branches × 2 cadences = 10
# =============================================================================


def test_pin_compare_returns_metrics_per_cadence_per_branch() -> None:
    X, y = _canonical_regime_shift_synthetic(seed=11, n=200)
    result = compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=100,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )
    assert isinstance(result, BranchComparisonResult)
    assert result.refit_frequencies == (20, 50)
    assert set(result.metrics.keys()) == {20, 50}
    for cadence in (20, 50):
        per_branch = result.metrics[cadence]
        assert set(per_branch.keys()) == set(BRANCH_NAMES), (
            f"cadence={cadence} branches mismatch: "
            f"{sorted(per_branch.keys())} vs {sorted(BRANCH_NAMES)}"
        )
        for branch, m in per_branch.items():
            assert isinstance(m, BranchMetrics)
            assert m.branch == branch
            assert m.refit_frequency == cadence


# =============================================================================
# Pins 2 & 3 — end-to-end shape pins on synthetic A and synthetic B.
# Per-step arrays must have length n - initial_train_size for every cell.
# =============================================================================


def _assert_shapes(result: BranchComparisonResult, n: int, initial_train: int) -> None:
    expected_len = n - initial_train
    for cadence in result.refit_frequencies:
        for branch in BRANCH_NAMES:
            preds = result.raw_predictions[cadence][branch]
            lower = result.raw_intervals_lower[cadence][branch]
            upper = result.raw_intervals_upper[cadence][branch]
            weights = result.raw_weights[cadence][branch]
            covered = result.raw_covered[cadence][branch]
            for arr_name, arr in (
                ("predictions", preds),
                ("lower", lower),
                ("upper", upper),
                ("weights", weights),
                ("covered", covered),
            ):
                assert arr.shape == (expected_len,), (
                    f"cadence={cadence} branch={branch} {arr_name} shape "
                    f"{arr.shape} != ({expected_len},)"
                )
            m = result.metrics[cadence][branch]
            assert m.n_steps == expected_len


def test_pin_each_branch_runs_end_to_end_canonical_synthetic() -> None:
    n = 200
    X, y = _canonical_regime_shift_synthetic(seed=11, n=n)
    result = compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )
    _assert_shapes(result, n, _SMALL_KW["initial_train_size"])


def test_pin_each_branch_runs_end_to_end_low_snr_synthetic() -> None:
    n = 200
    X, y = _low_snr_regime_shift_synthetic(seed=11, n=n)
    result = compare_alpha_branches(
        X,
        y,
        synthetic_name="low_snr_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )
    _assert_shapes(result, n, _SMALL_KW["initial_train_size"])


# =============================================================================
# Pin 4 — Route C reconsideration trigger pin: dispatch fans to TWO helpers.
# =============================================================================


def test_pin_dispatch_uses_two_helpers_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two callable targets per cadence regardless of cadence count.
    With 5 branches × 2 cadences, ``backtest_alpha_model`` is called
    8 times (4 batch branches × 2 cadences) and
    ``backtest_alpha_model_dtaci`` is called 2 times (dtaci × 2)."""
    counts = {"batch": 0, "dtaci": 0}
    orig_batch = ec_module.backtest_alpha_model
    orig_dtaci = ec_module.backtest_alpha_model_dtaci

    def spy_batch(*args: Any, **kwargs: Any) -> Any:
        counts["batch"] += 1
        return orig_batch(*args, **kwargs)

    def spy_dtaci(*args: Any, **kwargs: Any) -> Any:
        counts["dtaci"] += 1
        return orig_dtaci(*args, **kwargs)

    monkeypatch.setattr(ec_module, "backtest_alpha_model", spy_batch)
    monkeypatch.setattr(ec_module, "backtest_alpha_model_dtaci", spy_dtaci)

    n = 200
    X, y = _canonical_regime_shift_synthetic(seed=11, n=n)
    compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )

    assert counts["batch"] == 8, (
        f"expected 8 batch helper calls (4 branches × 2 cadences); got {counts['batch']}"
    )
    assert counts["dtaci"] == 2, (
        f"expected 2 dtaci helper calls (1 branch × 2 cadences); got {counts['dtaci']}"
    )


# =============================================================================
# Pin 5 — factories return fresh objects per cell; caller objects untouched.
# =============================================================================


def test_pin_caller_models_never_mutated() -> None:
    """Each (cadence, branch) cell calls the appropriate factory once,
    receiving a fresh ``BaseEstimator``. The caller never sees a
    pre-constructed ``ConformalAlphaModel`` / ``DtACI`` instance, so
    nothing the caller owns is mutated. We verify by counting factory
    calls and asserting object-identity uniqueness."""
    base_calls: list[LinearRegression] = []
    cqr_calls: list[GradientBoostingRegressor] = []

    def counting_base() -> LinearRegression:
        m = LinearRegression()
        base_calls.append(m)
        return m

    def counting_cqr() -> GradientBoostingRegressor:
        m = GradientBoostingRegressor(random_state=0)
        cqr_calls.append(m)
        return m

    n = 200
    X, y = _canonical_regime_shift_synthetic(seed=11, n=n)
    compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=counting_base,
        cqr_model_factory=counting_cqr,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )

    # 4 non-cqr branches × 2 cadences
    assert len(base_calls) == 8
    # cqr × 2 cadences
    assert len(cqr_calls) == 2
    # All instances are unique (each call returned a fresh object)
    assert len({id(m) for m in base_calls}) == 8
    assert len({id(m) for m in cqr_calls}) == 2


# =============================================================================
# Pin 6 — aggregate metrics are well-formed across every cell.
# =============================================================================


def test_pin_aggregate_metrics_well_formed() -> None:
    n = 200
    X, y = _canonical_regime_shift_synthetic(seed=11, n=n)
    result = compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )
    for cadence in result.refit_frequencies:
        for branch in BRANCH_NAMES:
            m = result.metrics[cadence][branch]
            assert 0.0 <= m.coverage_overall <= 1.0
            assert 0.0 <= m.coverage_rolling_mean_window50 <= 1.0
            assert m.mean_width_overall >= 0.0
            assert m.mean_width_rolling_mean_window50 >= 0.0
            # F-RP-004b: under canonical regime-shift synthetics some
            # branches are *expected* to be degenerate (no tradeable
            # signal). Post-fix `compare_alpha_branches` returns NaN
            # for those branches' Sharpe (loud-via-warning at site 1).
            # The strict pre-fix `math.isfinite(...)` assertion fails
            # NaN; tolerate NaN here AND assert finite-or-NaN. Pinning
            # only finite would break under canonical_small + low_snr
            # fixtures; pinning only NaN would mask real numerical
            # bugs that produce inf. The OR captures both valid
            # post-F-RP-004b states.
            assert math.isfinite(m.realized_sharpe_252) or math.isnan(m.realized_sharpe_252), (
                f"sharpe for {branch}@refit={cadence} is neither finite "
                f"nor NaN: {m.realized_sharpe_252}"
            )
            assert math.isfinite(m.realized_sharpe_252_rolling_window50) or math.isnan(
                m.realized_sharpe_252_rolling_window50
            )
            assert 0.0 <= m.tradeable_fraction <= 1.0
            assert 0.0 <= m.nonzero_signal_fraction <= 1.0
            assert m.mean_signal_strength >= 0.0


# =============================================================================
# Pin 7 — synthetic diagnostics finite + sanity on σ × 3.
# =============================================================================


def test_pin_synthetic_diagnostics_well_formed() -> None:
    """Both synthetics expose finite SNR + Sharpe diagnostics; synthetic B's
    realized oracle Sharpe is strictly less than synthetic A's, confirming
    σ × 3 actually produced lower realized SNR. Diagnostics are
    cadence-independent (same value if computed under either cadence)."""
    n = 600
    X_a, y_a = _canonical_regime_shift_synthetic(seed=REC_SEED, n=n)
    X_b, y_b = _low_snr_regime_shift_synthetic(seed=REC_SEED, n=n)

    # Run with a single cadence to keep this fast — we're only checking
    # synthetic-level diagnostics, not branch metrics.
    res_a = compare_alpha_branches(
        X_a,
        y_a,
        synthetic_name="canonical",
        seed=REC_SEED,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        design_snr_per_regime=(3.0, 33.3),
        alpha=0.1,
        initial_train_size=REC_N_TRAIN,
        refit_frequencies=(21,),
        dtaci_warmup=REC_WARMUP,
        branches=("split",),  # only need diagnostics, not all branches
    )
    res_b = compare_alpha_branches(
        X_b,
        y_b,
        synthetic_name="low_snr",
        seed=REC_SEED,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        design_snr_per_regime=(3.0 / 9.0, 33.3 / 9.0),  # σ×3 → SNR ÷ 9
        alpha=0.1,
        initial_train_size=REC_N_TRAIN,
        refit_frequencies=(21,),
        dtaci_warmup=REC_WARMUP,
        branches=("split",),
    )

    for sd in (res_a.synthetic_diagnostics, res_b.synthetic_diagnostics):
        assert isinstance(sd, SyntheticDiagnostics)
        for v in (
            sd.design_snr_overall,
            sd.design_snr_regime_A,
            sd.design_snr_regime_B,
            sd.realized_snr_overall,
            sd.realized_snr_regime_A,
            sd.realized_snr_regime_B,
            sd.realized_oracle_sharpe_252_overall,
            sd.realized_oracle_sharpe_252_regime_A,
            sd.realized_oracle_sharpe_252_regime_B,
        ):
            assert v is not None and math.isfinite(v), f"non-finite diagnostic: {v}"

    # Sanity: synthetic B is operationally lower-SNR than synthetic A.
    assert (
        res_b.synthetic_diagnostics.realized_oracle_sharpe_252_overall
        < res_a.synthetic_diagnostics.realized_oracle_sharpe_252_overall
    ), (
        "synthetic B realized oracle Sharpe should be < synthetic A's "
        "(σ × 3 should reduce SNR); got "
        f"A={res_a.synthetic_diagnostics.realized_oracle_sharpe_252_overall} "
        f"B={res_b.synthetic_diagnostics.realized_oracle_sharpe_252_overall}"
    )


# =============================================================================
# Pin 8 — n_refits matches the cadence formula (moves with the cadence).
# =============================================================================


def test_pin_n_refits_matches_cadence() -> None:
    """For each (cadence, branch) cell:
    ``n_refits == floor((n - initial_train_size - 1) / refit_frequency)``.
    This is the unified semantic across batch (in-loop after initial fit)
    and dtaci (``len(refit_points)``)."""
    n = 200
    X, y = _canonical_regime_shift_synthetic(seed=11, n=n)
    result = compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical_small",
        seed=11,
        base_model_factory=_base_factory,
        cqr_model_factory=_cqr_factory,
        mondrian_stratifier=_mondrian_stratifier,
        regime_shift_index=n // 2,
        oracle_predictor=_oracle_predictor,
        **_SMALL_KW,
    )
    n_steps = n - _SMALL_KW["initial_train_size"]
    for cadence in result.refit_frequencies:
        expected = (n_steps - 1) // cadence
        for branch in BRANCH_NAMES:
            m = result.metrics[cadence][branch]
            assert m.n_refits == expected, (
                f"branch={branch} cadence={cadence} n_refits={m.n_refits} "
                f"!= expected {expected} (n_steps={n_steps})"
            )


# =============================================================================
# Pin 9 — DtACI byte-equivalence vs S15 baseline at refit=21 on synthetic A.
# =============================================================================


def test_pin_dtaci_branch_byte_equivalent_to_s15_baseline_at_refit21() -> None:
    """The harness's dtaci branch wraps ``backtest_alpha_model_dtaci``.
    Calling the helper directly with the S15 baseline parameters and
    comparing observable output against the harness's metrics for the
    same cell pins that wrapping does NOT perturb the walk-forward
    output (no accidental re-warmup, re-init, or kwarg mismatch)."""
    X, y = _canonical_regime_shift_synthetic(seed=REC_SEED, n=REC_N_BARS)

    # Direct S15-style helper call.
    direct_dtaci = DtACI(alpha=0.1)
    direct = backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=direct_dtaci,
        X=X,
        y=y,
        initial_train_size=REC_N_TRAIN,
        refit_frequency=21,
        warmup=REC_WARMUP,
    )
    direct_coverage = float(np.mean(direct["covered"]))
    direct_widths = np.array([float(s.upper[0] - s.lower[0]) for s in direct["signals"]])
    direct_mean_width = float(np.mean(direct_widths))

    shift_idx_pred = (REC_N_BARS // 2) - REC_N_TRAIN
    direct_covered = np.asarray(direct["covered"], dtype=np.bool_)
    direct_cov_a = float(np.mean(direct_covered[:shift_idx_pred]))
    direct_cov_b = float(np.mean(direct_covered[shift_idx_pred:]))
    direct_w_a = float(np.mean(direct_widths[:shift_idx_pred]))
    direct_w_b = float(np.mean(direct_widths[shift_idx_pred:]))

    # Harness-wrapped call — single cadence, single branch.
    res = compare_alpha_branches(
        X,
        y,
        synthetic_name="canonical",
        seed=REC_SEED,
        base_model_factory=_base_factory,
        alpha=0.1,
        initial_train_size=REC_N_TRAIN,
        refit_frequencies=(21,),
        dtaci_warmup=REC_WARMUP,
        branches=("dtaci",),
        regime_shift_index=REC_N_BARS // 2,
        oracle_predictor=_oracle_predictor,
    )
    m = res.metrics[21]["dtaci"]

    def _eq(a: float, b: float) -> bool:
        return round(a, 6) == round(b, 6)

    assert _eq(m.coverage_overall, direct_coverage), (
        f"coverage_overall: harness {m.coverage_overall} vs direct {direct_coverage}"
    )
    assert _eq(m.mean_width_overall, direct_mean_width), (
        f"mean_width_overall: harness {m.mean_width_overall} vs direct {direct_mean_width}"
    )
    assert m.coverage_regime_A is not None
    assert m.coverage_regime_B is not None
    assert m.mean_width_regime_A is not None
    assert m.mean_width_regime_B is not None
    assert _eq(m.coverage_regime_A, direct_cov_a)
    assert _eq(m.coverage_regime_B, direct_cov_b)
    assert _eq(m.mean_width_regime_A, direct_w_a)
    assert _eq(m.mean_width_regime_B, direct_w_b)


# =============================================================================
# Recordings emission (P15.3) — 4 non-asserting tests, one per
# (synthetic, cadence) cell. Mirrors S13/S14/S15 latest.md overwrite
# pattern; asserts only file existence + non-zero size.
# =============================================================================


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _module_hash() -> str:
    p = (
        Path(__file__).parent.parent
        / "src"
        / "quantcore"
        / "uncertainty"
        / "conformal"
        / "finance"
        / "empirical_comparison.py"
    )
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _fmt_float(v: float | None) -> str:
    if v is None:
        return "—"
    if math.isnan(v):
        return "NaN"
    return f"{v:+.4f}"


def _fmt_bool_or_none(v: bool | None) -> str:
    if v is None:
        return "—"
    return "TRUE" if v else "false"


def _directional_correctness(
    predictions: NDArray[np.floating[Any]],
    returns: NDArray[np.floating[Any]],
    trade_mask: NDArray[np.bool_],
) -> tuple[float, float, float]:
    """Return (overall, on-takes, on-takes minus overall) — the
    precision_lift-equivalent metric."""
    if predictions.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    overall = float(np.mean(np.sign(predictions) == np.sign(returns)))
    if trade_mask.any():
        takes = float(np.mean(np.sign(predictions[trade_mask]) == np.sign(returns[trade_mask])))
    else:
        takes = float("nan")
    if math.isnan(takes):
        delta = float("nan")
    else:
        delta = takes - overall
    return overall, takes, delta


@pytest.fixture(scope="module")
def recordings_artifact_dir() -> Path:
    """Directory where recordings are written. Uses the canonical
    fixtures/empirical_comparison_recordings/ path (gitignored)."""
    d = Path(__file__).parent / "fixtures" / "empirical_comparison_recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def comparison_caches() -> dict[tuple[str, int], dict[str, Any]]:
    """Build the 4 (synthetic, cadence) caches once per test module.

    Each cache holds the BranchComparisonResult plus the ``y`` slice
    over the walk-forward range, so the recording renderer can
    compute directional-correctness without re-walking the synthetic.
    """
    caches: dict[tuple[str, int], dict[str, Any]] = {}
    for synthetic_name, generator, design_snr in (
        ("canonical_regime_shift", _canonical_regime_shift_synthetic, (3.0, 33.3)),
        ("low_snr_regime_shift", _low_snr_regime_shift_synthetic, (3.0 / 9.0, 33.3 / 9.0)),
    ):
        X, y = generator(seed=REC_SEED, n=REC_N_BARS)
        for cadence in (21, 63):
            t0 = time.perf_counter()
            result = compare_alpha_branches(
                X,
                y,
                synthetic_name=synthetic_name,
                seed=REC_SEED,
                base_model_factory=_base_factory,
                cqr_model_factory=_cqr_factory,
                mondrian_stratifier=_mondrian_stratifier,
                regime_shift_index=REC_N_BARS // 2,
                oracle_predictor=_oracle_predictor,
                design_snr_per_regime=design_snr,
                alpha=0.1,
                initial_train_size=REC_N_TRAIN,
                refit_frequencies=(cadence,),
                dtaci_warmup=REC_WARMUP,
            )
            runtime = time.perf_counter() - t0
            caches[(synthetic_name, cadence)] = {
                "result": result,
                "y_walk": y[REC_N_TRAIN:],
                "runtime_seconds": runtime,
            }
    return caches


def _write_recording(
    cache_entry: dict[str, Any],
    cadence: int,
    artifact_dir: Path,
    *,
    synthetic_name: str,
) -> Path:
    """Render the markdown for one (synthetic, cadence) cell and write it.

    Replaces the placeholder _build_returns_for_metric call by passing the
    cached y_walk through the rendering closure.
    """
    result: BranchComparisonResult = cache_entry["result"]
    y_walk: NDArray[np.floating[Any]] = cache_entry["y_walk"]
    n_steps_expected = result.n_bars - result.initial_train_size
    n_refits_expected = (n_steps_expected - 1) // cadence

    L: list[str] = []
    sd = result.synthetic_diagnostics
    L.append(f"# S16 P15.3 empirical-comparison recordings — `{synthetic_name}`, refit={cadence}d")
    L.append("")
    L.append(f"- Generated (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    L.append(f"- Git SHA (HEAD): `{_git_sha()}`")
    L.append(f"- empirical_comparison.py sha256[:16]: `{_module_hash()}`")
    L.append(
        f"- Synthetic: {synthetic_name}, seed={result.seed}, "
        f"N={result.n_bars}, N_train={result.initial_train_size}, "
        f"refit_frequency={cadence}, target α={result.alpha}, "
        f"regime_shift_index={result.regime_shift_index}"
    )
    L.append(f"- Helper runtime: {cache_entry['runtime_seconds']:.2f} s")
    L.append("")
    L.append(
        "**Non-asserting**. Asserting invariant pins live in the same "
        "file (test_pin_*); this artifact captures per-(cadence, branch) "
        "metrics so a future regression that doesn't break the asserting "
        "pins still surfaces to a human reader."
    )
    L.append("")

    L.append("## Synthetic-level diagnostics (cadence-independent)")
    L.append("")
    L.append("| metric | overall | regime A | regime B |")
    L.append("|---|---|---|---|")
    L.append(
        f"| design SNR | {_fmt_float(sd.design_snr_overall)} | "
        f"{_fmt_float(sd.design_snr_regime_A)} | "
        f"{_fmt_float(sd.design_snr_regime_B)} |"
    )
    L.append(
        f"| realized SNR | {_fmt_float(sd.realized_snr_overall)} | "
        f"{_fmt_float(sd.realized_snr_regime_A)} | "
        f"{_fmt_float(sd.realized_snr_regime_B)} |"
    )
    L.append(
        f"| realized oracle Sharpe (ann.) | "
        f"{_fmt_float(sd.realized_oracle_sharpe_252_overall)} | "
        f"{_fmt_float(sd.realized_oracle_sharpe_252_regime_A)} | "
        f"{_fmt_float(sd.realized_oracle_sharpe_252_regime_B)} |"
    )
    L.append("")

    L.append("## Per-branch summary")
    L.append("")
    L.append(
        "| branch | n_steps | n_refits | coverage | mean width | tradeable | "
        "mean signal strength | realized Sharpe (ann.) | terminal cum return |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for branch in BRANCH_NAMES:
        m = result.metrics[cadence][branch]
        L.append(
            f"| {branch} | {m.n_steps} | {m.n_refits} | "
            f"{_fmt_float(m.coverage_overall)} | "
            f"{_fmt_float(m.mean_width_overall)} | "
            f"{_fmt_float(m.tradeable_fraction)} | "
            f"{_fmt_float(m.mean_signal_strength)} | "
            f"{_fmt_float(m.realized_sharpe_252)} | "
            f"{_fmt_float(m.cumulative_return_terminal)} |"
        )
    L.append("")

    L.append("## Per-branch per-regime coverage and width")
    L.append("")
    L.append("| branch | cov A | cov B | width A | width B |")
    L.append("|---|---|---|---|---|")
    for branch in BRANCH_NAMES:
        m = result.metrics[cadence][branch]
        L.append(
            f"| {branch} | {_fmt_float(m.coverage_regime_A)} | "
            f"{_fmt_float(m.coverage_regime_B)} | "
            f"{_fmt_float(m.mean_width_regime_A)} | "
            f"{_fmt_float(m.mean_width_regime_B)} |"
        )
    L.append("")

    L.append("## Per-branch failure-mode flags")
    L.append("")
    L.append(
        "| branch | dtaci expert collapse | coverage undershoot severe | "
        "coverage overshoot severe |"
    )
    L.append("|---|---|---|---|")
    for branch in BRANCH_NAMES:
        m = result.metrics[cadence][branch]
        L.append(
            f"| {branch} | {_fmt_bool_or_none(m.expert_collapse_dtaci)} | "
            f"{_fmt_bool_or_none(m.coverage_undershoot_severe)} | "
            f"{_fmt_bool_or_none(m.coverage_overshoot_severe)} |"
        )
    L.append("")

    L.append("## Per-branch directional correctness (precision_lift-equivalent)")
    L.append("")
    L.append(
        "Directional correctness is `mean(sign(pred) == sign(return))`. "
        "On-takes restricts to bars where the SignalFilter + "
        "PortfolioConstructor produced a non-zero weight. The delta "
        "(on-takes minus overall) is the alpha-pipeline analogue of the "
        "AFML meta-labeling `precision_lift` metric. "
        "**Recording-only — not asserted in S16; deferral evidence for "
        "the S17 hypothesis spike per the anomaly trigger response.**"
    )
    L.append("")
    L.append("| branch | overall | on-takes | delta (on-takes − overall) |")
    L.append("|---|---|---|---|")
    for branch in BRANCH_NAMES:
        preds = result.raw_predictions[cadence][branch]
        trades = result.raw_weights[cadence][branch] != 0.0
        overall, takes, delta = _directional_correctness(preds, y_walk, trades)
        L.append(
            f"| {branch} | {_fmt_float(overall)} | {_fmt_float(takes)} | {_fmt_float(delta)} |"
        )
    L.append("")

    L.append("## Refit-cadence cross-check")
    L.append("")
    L.append(
        f"Expected `n_refits` per cell: "
        f"`floor((n - initial_train_size - 1) / refit_frequency)` = "
        f"{n_refits_expected}."
    )
    L.append("")

    L.append("## Anomaly trigger evaluation")
    L.append("")
    L.append(
        "The precision_lift sign-flip anomaly trigger condition (a) "
        "fires under structural reading on the low-SNR cells. The "
        "deferred response is recording-only; the S17 plan-commit "
        "evaluates whether to pull the hypothesis spike forward "
        "based on the directional-correctness deltas above (3+ of "
        "5 branches with sign-flip-equivalent → recommend pulling)."
    )
    L.append("")

    artifact_path = artifact_dir / f"{synthetic_name}_refit{cadence}_latest.md"
    artifact_path.write_text("\n".join(L))
    return artifact_path


@pytest.mark.parametrize(
    "synthetic_name,cadence",
    [
        ("canonical_regime_shift", 21),
        ("canonical_regime_shift", 63),
        ("low_snr_regime_shift", 21),
        ("low_snr_regime_shift", 63),
    ],
)
def test_record_branch_comparison(
    synthetic_name: str,
    cadence: int,
    comparison_caches: dict[tuple[str, int], dict[str, Any]],
    recordings_artifact_dir: Path,
) -> None:
    """Write the empirical-comparison recording for one (synthetic, cadence)
    cell. Asserts only file existence + non-zero size; content is
    intentionally not pinned (asserting pins live above)."""
    cache_entry = comparison_caches[(synthetic_name, cadence)]
    artifact_path = _write_recording(
        cache_entry,
        cadence,
        recordings_artifact_dir,
        synthetic_name=synthetic_name,
    )
    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0


# =============================================================================
# F-RP-004b — three-site silent-Sharpe sibling fix in compare_alpha_branches
#
# Pin three behaviors on a deliberately-degenerate fixture, mirroring the
# F-RP-002 happy-path / F08-trigger pattern at backtest_alpha_model scope.
# Plan revision in commit 0f5ff3c documents the three-site warning-frequency
# asymmetry: sites 1 and 3 warn-on-trigger; site 2 propagates NaN silently
# under site-1 aggregate-warning subsumption.
# =============================================================================


def test_compare_alpha_branches_nan_on_degenerate() -> None:
    """F-RP-004b regression: pin three behaviors on a deliberately-
    degenerate fixture (``y = np.zeros(n)``, no signal):

      (a) site 1 (line 325 pre-fix) emits ≥1 ``UserWarning`` matching
          the F08 prefix ``compute_branch_metrics:.*degenerate-variance``.
      (b) site 2 (line 335 pre-fix) propagates NaN SILENTLY — no
          ``UserWarning`` mentioning a rolling-window phrase appears
          in the captured corpus.
      (c) at least one (cadence × branch) cell's
          ``realized_sharpe_252_rolling_window50`` is NaN.

    Clause (b) is load-bearing per s19b plan-revision commit 0f5ff3c:
    a future regression adding per-iteration warning to the rolling
    loop would silently pass (a) and (c) but fail (b). Pin the
    *absence* of rolling-prefixed warnings via
    ``warnings.catch_warnings(record=True)`` + post-hoc message
    inspection rather than ``pytest.warns`` (which is presence-only).

    Fixture rationale: ``y = np.zeros(n)`` forces every branch's
    ``portfolio_returns`` to identical zeros, tripping F08 on all
    10 (cadence × branch) cells deterministically. ``canonical_small``
    only trips F08 on Mondrian (per the s19a F-RP-002 audit); the
    determinism here makes the assertion fixture-independent across
    numpy/scipy/sklearn version drift.

    Pre-emission verification:
      * ``sharpe_ratio(np.zeros(200), periods_per_year=252)`` raises
        ValueError with `degenerate variance: sd=0.000e+00` — confirmed
        the F08 gate trips on the load-bearing primitive before the
        full ``compare_alpha_branches`` cascade.
      * ``rg "rolling" empirical_comparison.py`` shows "rolling" appears
        only in field names and the ``_rolling_mean`` helper, never in
        a warning message — so the negative filter for clause (b) is
        falsifiable.
    """
    n = 200
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((n, 3))
    y = np.zeros(n, dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compare_alpha_branches(
            X,
            y,
            synthetic_name="degenerate_zero_y",
            seed=42,
            base_model_factory=_base_factory,
            cqr_model_factory=_cqr_factory,
            mondrian_stratifier=_mondrian_stratifier,
            regime_shift_index=n // 2,
            oracle_predictor=_oracle_predictor,
            **_SMALL_KW,
        )

    # (a) Site 1 must emit ≥1 UserWarning matching the F08 prefix.
    site1_warnings = [
        w
        for w in caught
        if "compute_branch_metrics" in str(w.message) and "degenerate-variance" in str(w.message)
    ]
    assert len(site1_warnings) >= 1, (
        f"F-RP-004b site 1 must emit UserWarning on F08 trigger; "
        f"captured {len(caught)} total warnings, none matching "
        f"compute_branch_metrics + degenerate-variance"
    )

    # (b) Site 2 must propagate NaN SILENTLY. No warning should
    # mention "rolling" in any form. Verified pre-emission via
    # rg "rolling" empirical_comparison.py: "rolling" appears only
    # in field names and the _rolling_mean helper, never in a
    # warning message — so the negative filter is meaningful.
    site2_warnings = [w for w in caught if "rolling" in str(w.message).lower()]
    assert len(site2_warnings) == 0, (
        f"F-RP-004b site 2 must propagate NaN silently; got "
        f"{len(site2_warnings)} rolling-prefixed warnings: "
        f"{[str(w.message) for w in site2_warnings]}"
    )

    # (c) At least one cell's rolling-window Sharpe is NaN. With
    # y=zeros every cell should be NaN; ≥1 is sufficient for the
    # contract.
    nan_count = sum(
        1
        for cadence in result.refit_frequencies
        for branch in BRANCH_NAMES
        if math.isnan(result.metrics[cadence][branch].realized_sharpe_252_rolling_window50)
    )
    assert nan_count >= 1, (
        f"F-RP-004b: ≥1 (cadence × branch) cell must produce NaN "
        f"realized_sharpe_252_rolling_window50 on degenerate fixture; "
        f"got {nan_count}/{len(result.refit_frequencies) * len(BRANCH_NAMES)}"
    )
