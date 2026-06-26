"""Asserting integration pins for backtest_alpha_model_dtaci (S15
P14.2) plus the non-asserting alpha-trajectory recordings emission
(P14.3 part 2).

Asserting pins (P14.2):

  * Pin 1 — end-to-end runs and produces non-empty per-step outputs
    of the expected length on the regime-shift synthetic.
  * Pin 2 — returned diagnostics dict has all 13 plan-pinned keys.
  * Pin 3 — composition: predict_step / update_step are each
    invoked once per walk-forward step on the caller's DtACI
    (verified on the LAST ``n - initial_train_size`` calls so the
    pin is portable across init-shape choices).
  * Pin 4 — missing-warmup invariant: at the first walk-forward
    predict_step, ``dtaci.n_scores >= warmup``.
  * Pin 5 — state-continuity: DtACI object identity stays stable
    across the loop and the score buffer is monotonic
    non-decreasing (no reset within walk-forward).
  * Pin 6 — model-fit-count: base model is fit
    ``1 + floor((n - initial_train_size) / refit_frequency)``
    times, with progressively-larger training windows.
  * Pin 7 — clone invariant: the caller's ``base_model`` instance
    is never mutated (no ``coef_`` attribute appears post-call).
  * Pin 8 — signature-validation: bad ``initial_train_size`` /
    ``warmup`` / ``refit_frequency`` raise ``ValueError`` naming
    the offending kwarg.
  * Pin 9 — adaptation: the ``aggregated_alpha`` trajectory is
    nonconstant on the regime-shift synthetic (rounded to 1e-6).

Recordings emission (P14.3 part 2):

  * ``test_record_dtaci_alpha_trajectory`` writes
    ``fixtures/dtaci_alpha_recordings/latest.md`` with
    regime-stratified coverage, mean interval width,
    aggregated_alpha trajectory summary, expert-weight entropy
    trajectory summary, and refit points. Mirrors S13 P12.2
    ``test_record_dtaci_recovery_rates`` and S14
    ``test_record_alpha_branch_summary`` shape (single-file
    overwrite, gitignored, asserts only file existence).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.dtaci import DtACI
from quantcore.uncertainty.conformal.finance import (
    backtest_alpha_model_dtaci,
)


# -----------------------------------------------------------------------------
# Regime-shift synthetic — high-vol then low-vol with a hard transition.
# Integration counterpart to S13 P12.2's hard-shift recovery test.
# -----------------------------------------------------------------------------


def _regime_shift_synthetic(seed: int = 11, n: int = 400):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    noise_scale = np.where(np.arange(n) < n // 2, 1.0, 0.3)
    y = X.sum(axis=1) + noise_scale * rng.standard_normal(n)
    return X, y


# -----------------------------------------------------------------------------
# Pin 1 — end-to-end on regime-shift synthetic.
# -----------------------------------------------------------------------------


def test_pin_end_to_end_regime_shift_synthetic() -> None:
    X, y = _regime_shift_synthetic()
    n = len(y)
    initial_train_size = 252

    dtaci = DtACI(alpha=0.1)
    results = backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=21,
        warmup=50,
    )

    expected_len = n - initial_train_size
    per_step_keys = [
        "signals",
        "weights",
        "returns",
        "predictions",
        "intervals",
        "covered",
        "trade_mask",
        "aggregated_alpha",
        "expert_alphas",
        "expert_weights",
        "weight_entropy",
        "interval_width",
    ]
    for key in per_step_keys:
        assert len(results[key]) == expected_len, (
            f"results[{key!r}] has length {len(results[key])}, expected {expected_len}"
        )
    assert expected_len > 0


# -----------------------------------------------------------------------------
# Pin 2 — returned diagnostics dict has all 13 plan-pinned keys.
# -----------------------------------------------------------------------------


def test_pin_returned_diagnostics_keys() -> None:
    X, y = _regime_shift_synthetic(n=300)
    dtaci = DtACI(alpha=0.1)
    results = backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=200,
        refit_frequency=21,
        warmup=50,
    )
    expected_keys = {
        "signals",
        "weights",
        "returns",
        "predictions",
        "intervals",
        "covered",
        "trade_mask",
        "aggregated_alpha",
        "expert_alphas",
        "expert_weights",
        "weight_entropy",
        "interval_width",
        "refit_points",
    }
    assert set(results.keys()) == expected_keys


# -----------------------------------------------------------------------------
# Pin 3 — composition: predict_step / update_step per walk-forward step.
# -----------------------------------------------------------------------------


def test_pin_composition_predict_update_per_walk_forward_step(monkeypatch) -> None:
    """Within the walk-forward loop, predict_step + update_step are
    each invoked once per step on the caller's DtACI instance.

    Portable form: assert that the LAST (n - initial_train_size)
    predict_step calls have X_t == X[t:t+1] and the LAST
    (n - initial_train_size) update_step calls have y_true == y[t]
    for t in [initial_train_size, n). Init-phase calls (whatever
    shape they take) are deliberately excluded.
    """
    X, y = _regime_shift_synthetic(n=320)
    n = len(y)
    initial_train_size = 250

    predict_X_args: list[np.ndarray] = []
    update_y_args: list[float] = []
    orig_predict_step = DtACI.predict_step
    orig_update_step = DtACI.update_step

    def spied_predict(self, model, X_t):
        predict_X_args.append(np.asarray(X_t).copy())
        return orig_predict_step(self, model, X_t)

    def spied_update(self, y_true, y_pred):
        update_y_args.append(float(y_true))
        return orig_update_step(self, y_true, y_pred)

    monkeypatch.setattr(DtACI, "predict_step", spied_predict)
    monkeypatch.setattr(DtACI, "update_step", spied_update)

    dtaci = DtACI(alpha=0.1)
    backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=21,
        warmup=50,
    )

    walk_forward_n = n - initial_train_size
    assert len(predict_X_args) >= walk_forward_n
    assert len(update_y_args) >= walk_forward_n

    for offset in range(walk_forward_n):
        t = initial_train_size + offset
        actual_X = predict_X_args[-walk_forward_n + offset].ravel()
        expected_X = X[t : t + 1].ravel()
        np.testing.assert_array_equal(actual_X, expected_X)

    for offset in range(walk_forward_n):
        t = initial_train_size + offset
        actual_y = update_y_args[-walk_forward_n + offset]
        assert actual_y == float(y[t])


# -----------------------------------------------------------------------------
# Pin 4 — missing-warmup invariant.
# -----------------------------------------------------------------------------


def test_pin_warmup_score_buffer_populated_before_first_walk_forward_predict(
    monkeypatch,
) -> None:
    """At the first walk-forward predict_step, dtaci.n_scores >= warmup.

    The first walk-forward call is identified by slicing the LAST
    (n - initial_train_size) predict_step invocations; its captured
    n_scores must be >= warmup.
    """
    X, y = _regime_shift_synthetic(n=320)
    n = len(y)
    initial_train_size = 250
    warmup = 50

    n_scores_at_predict: list[int] = []
    orig = DtACI.predict_step

    def spied(self, model, X_t):
        n_scores_at_predict.append(self.n_scores)
        return orig(self, model, X_t)

    monkeypatch.setattr(DtACI, "predict_step", spied)

    dtaci = DtACI(alpha=0.1)
    backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=21,
        warmup=warmup,
    )

    walk_forward_n = n - initial_train_size
    assert len(n_scores_at_predict) >= walk_forward_n
    first_wf_n_scores = n_scores_at_predict[-walk_forward_n]
    assert first_wf_n_scores >= warmup, (
        f"At first walk-forward predict_step, n_scores={first_wf_n_scores} < warmup={warmup}"
    )


# -----------------------------------------------------------------------------
# Pin 5 — state-continuity across refit boundaries.
# -----------------------------------------------------------------------------


def test_pin_dtaci_state_continuity_across_refits(monkeypatch) -> None:
    """DtACI is never reset/reconstructed within walk-forward.
    Object identity stays stable; the score buffer length is
    monotonic non-decreasing across the walk-forward range
    (including across refit boundaries)."""
    X, y = _regime_shift_synthetic(n=400)
    initial_train_size = 252

    dtaci = DtACI(alpha=0.1)
    initial_dtaci_id = id(dtaci)

    n_scores_trace: list[int] = []
    orig = DtACI.update_step

    def spied(self, y_true, y_pred):
        out = orig(self, y_true, y_pred)
        n_scores_trace.append(self.n_scores)
        return out

    monkeypatch.setattr(DtACI, "update_step", spied)

    backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=21,
        warmup=50,
    )

    assert id(dtaci) == initial_dtaci_id

    arr = np.asarray(n_scores_trace, dtype=int)
    diffs = np.diff(arr)
    assert np.all(diffs >= 0), (
        f"n_scores trace is not monotonic non-decreasing; min diff = {diffs.min()}"
    )


# -----------------------------------------------------------------------------
# Pin 6 — refit-boundary fit count.
# -----------------------------------------------------------------------------


def test_pin_model_fit_count_matches_expected_boundaries() -> None:
    """Base model is fit 1 + floor((n - initial_train_size) /
    refit_frequency) times, with progressively-larger training
    windows. Verified via a counting subclass of LinearRegression
    (sklearn.clone preserves the type, so refits also count)."""
    X, y = _regime_shift_synthetic(n=400)
    initial_train_size = 252
    refit_frequency = 21

    fit_calls: list[int] = []

    class CountingLR(LinearRegression):
        def fit(self, X_in, y_in, **kwargs):
            fit_calls.append(int(X_in.shape[0]))
            return super().fit(X_in, y_in, **kwargs)

    dtaci = DtACI(alpha=0.1)
    backtest_alpha_model_dtaci(
        base_model=CountingLR(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=initial_train_size,
        refit_frequency=refit_frequency,
        warmup=50,
    )

    n = len(y)
    expected_fits = 1 + (n - initial_train_size) // refit_frequency
    assert len(fit_calls) == expected_fits

    assert fit_calls[0] == initial_train_size
    for i, X_size in enumerate(fit_calls[1:], start=1):
        expected_t = initial_train_size + i * refit_frequency
        assert X_size == expected_t


# -----------------------------------------------------------------------------
# Pin 7 — clone invariant: caller's base_model is never mutated.
# -----------------------------------------------------------------------------


def test_pin_clone_invariant_caller_base_model_never_mutated() -> None:
    """Helper clones base_model at every fit boundary; the caller's
    instance never has fit() invoked on it (no .coef_ post-call)."""
    X, y = _regime_shift_synthetic(n=320)

    base = LinearRegression()
    assert not hasattr(base, "coef_")

    dtaci = DtACI(alpha=0.1)
    backtest_alpha_model_dtaci(
        base_model=base,
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=250,
        refit_frequency=21,
        warmup=50,
    )

    assert not hasattr(base, "coef_"), "caller's base_model was mutated: .coef_ appeared post-call"


# -----------------------------------------------------------------------------
# Pin 8 — signature validation.
# -----------------------------------------------------------------------------


def test_pin_initial_train_size_at_least_len_y_raises() -> None:
    X, y = _regime_shift_synthetic(n=100)
    dtaci = DtACI(alpha=0.1)

    with pytest.raises(ValueError, match="initial_train_size"):
        backtest_alpha_model_dtaci(
            base_model=LinearRegression(),
            dtaci=dtaci,
            X=X,
            y=y,
            initial_train_size=100,
            refit_frequency=21,
            warmup=50,
        )


def test_pin_warmup_at_least_initial_train_size_raises() -> None:
    X, y = _regime_shift_synthetic(n=300)
    dtaci = DtACI(alpha=0.1)

    with pytest.raises(ValueError, match="warmup"):
        backtest_alpha_model_dtaci(
            base_model=LinearRegression(),
            dtaci=dtaci,
            X=X,
            y=y,
            initial_train_size=50,
            refit_frequency=21,
            warmup=50,
        )


def test_pin_refit_frequency_below_one_raises() -> None:
    X, y = _regime_shift_synthetic(n=300)
    dtaci = DtACI(alpha=0.1)

    with pytest.raises(ValueError, match="refit_frequency"):
        backtest_alpha_model_dtaci(
            base_model=LinearRegression(),
            dtaci=dtaci,
            X=X,
            y=y,
            initial_train_size=200,
            refit_frequency=0,
            warmup=50,
        )


# -----------------------------------------------------------------------------
# Pin 9 — aggregated_alpha trajectory is nonconstant on regime-shift synthetic.
# -----------------------------------------------------------------------------


def test_pin_aggregated_alpha_trajectory_nonconstant() -> None:
    """DtACI online α-state evolves over the regime-shift synthetic.
    Asserts state changes (rounded to 1e-6) — proves the helper
    consumes DtACI's online machinery rather than freezing it.
    Does NOT pin a specific trajectory shape or coverage outcome."""
    X, y = _regime_shift_synthetic(n=400)
    dtaci = DtACI(alpha=0.1)

    results = backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=252,
        refit_frequency=21,
        warmup=50,
    )

    alpha_trace = np.asarray(results["aggregated_alpha"], dtype=np.float64)
    rounded = np.round(alpha_trace, 6)
    unique_values = np.unique(rounded)
    assert len(unique_values) > 1, (
        f"aggregated_alpha trajectory is constant at "
        f"{unique_values[0]} — DtACI state did not evolve"
    )


# =============================================================================
# Recordings emission (P14.3 part 2). Non-asserting — writes a
# Markdown trace of DtACI online state across the regime-shift
# walk-forward, then asserts only file existence + non-zero size.
# Mirrors S13 P12.2 + S14 alpha_recordings shape.
# =============================================================================

REC_SEED: int = 11
REC_N_BARS: int = 600
REC_N_TRAIN: int = 252
REC_REFIT_FREQ: int = 21
REC_WARMUP: int = 50


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


def _backtest_dtaci_module_hash() -> str:
    p = (
        Path(__file__).parent.parent
        / "src"
        / "quantcore"
        / "uncertainty"
        / "conformal"
        / "finance"
        / "backtest_dtaci.py"
    )
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _fmt_float(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    return f"{v:+.4f}"


@pytest.fixture(scope="module")
def dtaci_recordings_cache() -> dict[str, object]:
    """Run the helper end-to-end on a regime-shift synthetic and
    capture per-step DtACI state for recording. Hard regime shift
    at REC_N_BARS // 2 forces visible α-adaptation across the
    walk-forward window."""
    X, y = _regime_shift_synthetic(seed=REC_SEED, n=REC_N_BARS)
    shift_index = REC_N_BARS // 2

    dtaci = DtACI(alpha=0.1)
    t0 = time.perf_counter()
    results = backtest_alpha_model_dtaci(
        base_model=LinearRegression(),
        dtaci=dtaci,
        X=X,
        y=y,
        initial_train_size=REC_N_TRAIN,
        refit_frequency=REC_REFIT_FREQ,
        warmup=REC_WARMUP,
    )
    runtime_s = time.perf_counter() - t0

    n = REC_N_BARS
    walk_forward_n = n - REC_N_TRAIN
    t_indices = np.arange(REC_N_TRAIN, n)

    covered = np.asarray(results["covered"], dtype=bool)
    interval_widths = np.asarray(results["interval_width"], dtype=np.float64)
    aggregated_alpha = np.asarray(results["aggregated_alpha"], dtype=np.float64)
    weight_entropy = np.asarray(results["weight_entropy"], dtype=np.float64)
    trade_mask = np.asarray(results["trade_mask"], dtype=bool)

    # Regime split on test indices: regime A (high-vol) on
    # t < shift_index, regime B (low-vol) on t >= shift_index.
    regime_A_mask = t_indices < shift_index
    regime_B_mask = ~regime_A_mask

    def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(arr[mask].mean())

    return {
        "runtime_seconds": runtime_s,
        "walk_forward_n": int(walk_forward_n),
        "regime_A_n": int(regime_A_mask.sum()),
        "regime_B_n": int(regime_B_mask.sum()),
        "shift_index": int(shift_index),
        "coverage_overall": float(covered.mean()),
        "coverage_regime_A": _safe_mean(covered.astype(np.float64), regime_A_mask),
        "coverage_regime_B": _safe_mean(covered.astype(np.float64), regime_B_mask),
        "mean_width_overall": float(interval_widths.mean()),
        "mean_width_regime_A": _safe_mean(interval_widths, regime_A_mask),
        "mean_width_regime_B": _safe_mean(interval_widths, regime_B_mask),
        "tradeable_fraction": float(trade_mask.mean()),
        "agg_alpha_first": float(aggregated_alpha[0]),
        "agg_alpha_terminal": float(aggregated_alpha[-1]),
        "agg_alpha_min": float(aggregated_alpha.min()),
        "agg_alpha_max": float(aggregated_alpha.max()),
        "agg_alpha_std": float(aggregated_alpha.std()),
        "weight_entropy_first": float(weight_entropy[0]),
        "weight_entropy_terminal": float(weight_entropy[-1]),
        "weight_entropy_min": float(weight_entropy.min()),
        "weight_entropy_max": float(weight_entropy.max()),
        "weight_entropy_mean": float(weight_entropy.mean()),
        "refit_points": list(results["refit_points"]),
        "n_refits": len(results["refit_points"]),
    }


def _render_md(cache: dict[str, object]) -> str:
    L: list[str] = []
    L.append("# S15 P14.3 DtACI alpha-trajectory recordings — `latest.md`")
    L.append("")
    L.append(f"- Generated (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    L.append(f"- Git SHA (HEAD): `{_git_sha()}`")
    L.append(f"- backtest_dtaci.py sha256[:16]: `{_backtest_dtaci_module_hash()}`")
    L.append(
        f"- Setup: regime-shift synthetic, seed={REC_SEED}, "
        f"N={REC_N_BARS}, N_train={REC_N_TRAIN}, "
        f"refit_frequency={REC_REFIT_FREQ}, warmup={REC_WARMUP}, "
        f"target α=0.1. Hard regime shift at index "
        f"{cache['shift_index']}: regime A (high-vol, σ=1.0) → "
        f"regime B (low-vol, σ=0.3)."
    )
    L.append(
        f"- Walk-forward steps: {cache['walk_forward_n']} "
        f"(regime A: {cache['regime_A_n']}, "
        f"regime B: {cache['regime_B_n']})"
    )
    L.append(f"- Helper runtime: {cache['runtime_seconds']:.2f} s")
    L.append("")
    L.append(
        "**Non-asserting**. Asserting integration pins live in the "
        "same file (test_pin_*); this artifact captures qualitative "
        "DtACI online-state evolution across the regime-shift "
        "walk-forward so a future regression that doesn't break "
        "the asserting pins still surfaces to a human reader."
    )
    L.append("")

    L.append("## Regime-stratified coverage and width")
    L.append("")
    L.append("| metric | overall | regime A (high-vol) | regime B (low-vol) |")
    L.append("|---|---|---|---|")
    L.append(
        f"| coverage | {_fmt_float(cache['coverage_overall'])} | "
        f"{_fmt_float(cache['coverage_regime_A'])} | "
        f"{_fmt_float(cache['coverage_regime_B'])} |"
    )
    L.append(
        f"| mean width | {_fmt_float(cache['mean_width_overall'])} | "
        f"{_fmt_float(cache['mean_width_regime_A'])} | "
        f"{_fmt_float(cache['mean_width_regime_B'])} |"
    )
    L.append("")
    L.append(f"Tradeable fraction (post-SignalFilter): {_fmt_float(cache['tradeable_fraction'])}.")
    L.append("")

    L.append("## Aggregated α trajectory")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| α at first walk-forward step | {_fmt_float(cache['agg_alpha_first'])} |")
    L.append(f"| α at terminal step | {_fmt_float(cache['agg_alpha_terminal'])} |")
    L.append(f"| α min | {_fmt_float(cache['agg_alpha_min'])} |")
    L.append(f"| α max | {_fmt_float(cache['agg_alpha_max'])} |")
    L.append(f"| α std | {_fmt_float(cache['agg_alpha_std'])} |")
    L.append("")

    L.append("## Expert-weight entropy trajectory")
    L.append("")
    L.append(
        "Normalized Shannon entropy in [0, 1]; 1.0 = uniform "
        "(no expert dominance), < 0.2 signals expert collapse "
        "per the 2026-04-29 conformal-stack failure-mode table."
    )
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(
        f"| entropy at first walk-forward step | {_fmt_float(cache['weight_entropy_first'])} |"
    )
    L.append(f"| entropy at terminal step | {_fmt_float(cache['weight_entropy_terminal'])} |")
    L.append(f"| entropy min | {_fmt_float(cache['weight_entropy_min'])} |")
    L.append(f"| entropy max | {_fmt_float(cache['weight_entropy_max'])} |")
    L.append(f"| entropy mean | {_fmt_float(cache['weight_entropy_mean'])} |")
    L.append("")

    L.append("## Refit points")
    L.append("")
    L.append(
        f"{cache['n_refits']} refits during walk-forward "
        f"(expected: floor({cache['walk_forward_n']} / "
        f"{REC_REFIT_FREQ}) = "
        f"{cache['walk_forward_n'] // REC_REFIT_FREQ})."
    )
    L.append("")
    rp = cache["refit_points"]
    if rp:
        L.append(f"Indices: {rp}")
    L.append("")

    return "\n".join(L)


def test_record_dtaci_alpha_trajectory(
    dtaci_recordings_cache: dict[str, object],
) -> None:
    """Write the DtACI alpha-trajectory recordings artifact.
    Asserts only file existence + non-zero size; content is
    intentionally not pinned (asserting pins live in the same
    file)."""
    artifact_dir = Path(__file__).parent / "fixtures" / "dtaci_alpha_recordings"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "latest.md"
    md = _render_md(dtaci_recordings_cache)
    artifact_path.write_text(md)
    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0
