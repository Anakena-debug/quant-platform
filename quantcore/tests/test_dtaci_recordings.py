"""DtACI recovery-rate recordings (S13 P12.2 non-asserting).

Mirrors S12 P11.3's pattern: a non-asserting test that writes a
single-file Markdown artifact with the empirical recovery-rate
trajectory of DtACI on the hard-regime-shift synthetic. The
asserting pin (``test_pin_dtaci_recovery_after_hard_shift`` in
``test_dtaci_invariants.py``) checks that coverage clears the 1pp
tolerance after the recovery window; this recording captures the
full trajectory so a future regression (e.g., a numpy update
shifting trajectories by 0.5pp) makes the cause obvious.

Cross-fixture invariant pins live in
``test_dtaci_invariants.py``; this file is the recording side of
the contract.

Output: ``quantcore/tests/fixtures/dtaci_recordings/latest.md``.
The directory is git-tracked (with ``.gitignore``); the artifact
itself is overwritten each run and not version-controlled.
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


SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_BARS: int = 1500
SHIFT_MAGNITUDE: float = 3.0
WARMUP: int = 200
ALPHA: float = 0.1
WINDOW_SIZE: int = 200


def _hard_shift_synthetic(seed: int, n: int, shift: float):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y_pre = X[: n // 4].sum(axis=1) + rng.standard_normal(n // 4)
    model = LinearRegression().fit(X[: n // 4], y_pre)
    y_pred_full = model.predict(X)
    residuals = rng.standard_normal(n)
    residuals[n // 2 :] += shift
    y = y_pred_full + residuals
    return X, y, model


@pytest.fixture(scope="module")
def recordings_cache() -> dict[int, dict]:
    """Build per-seed DtACI run on the hard-regime-shift synthetic.

    Captures the full coverage trajectory and milestone-aligned
    diagnostics (weight_entropy, aggregated_alpha) at:
      - end-of-warmup (first online step)
      - shift point T/2 (in online-stream coords)
      - shift + 1·recovery_lag (recovery_lag = ceil(1/max(γ)))
      - shift + 2·recovery_lag
      - end-of-trace
    """
    cache: dict[int, dict] = {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        X, y, model = _hard_shift_synthetic(seed=seed, n=N_BARS, shift=SHIFT_MAGNITUDE)
        dt_obj = DtACI(alpha=ALPHA, window_size=WINDOW_SIZE)
        intervals, agg_traj, expert_alpha_traj, weight_traj = dt_obj.run_online(
            model, X, y, warmup=WARMUP
        )
        runtime_s = time.perf_counter() - t0

        # Hit / miss array (online-stream indexing).
        y_online = y[WARMUP:]
        hits = np.array(
            [iv.contains(np.array([y_t]))[0] for iv, y_t in zip(intervals, y_online)],
            dtype=np.float64,
        )
        recovery_lag = int(math.ceil(1.0 / max(dt_obj.gammas)))
        shift_idx = N_BARS // 2 - WARMUP  # online-stream index

        # Milestones (online-stream indices).
        milestones = {
            "end_of_warmup": 0,
            "pre_shift": shift_idx - 1,
            "shift_t": shift_idx,
            "shift+lag": min(shift_idx + recovery_lag, len(intervals) - 1),
            "shift+2lag": min(shift_idx + 2 * recovery_lag, len(intervals) - 1),
            "end_of_trace": len(intervals) - 1,
        }

        # Coverage windowed: rolling 50-step coverage at each
        # milestone (centered if possible, else trailing).
        def _windowed_coverage(t: int, half: int = 25) -> float:
            lo = max(0, t - half)
            hi = min(len(hits), t + half + 1)
            return float(hits[lo:hi].mean())

        # Coverage on intervals after the recovery window (the
        # asserting pin's measurement).
        recovery_start = shift_idx + recovery_lag
        recovery_coverage = (
            float(hits[recovery_start:].mean()) if recovery_start < len(hits) else float("nan")
        )

        # Time to recover within 1pp: first index t after shift such
        # that windowed coverage on [t, t+50] is within 1pp of
        # target. Returns NaN if never recovers within trace.
        target = 1.0 - ALPHA
        time_to_recover = float("nan")
        for t in range(shift_idx, len(hits) - 50):
            window_cov = float(hits[t : t + 50].mean())
            if abs(window_cov - target) < 0.01:
                time_to_recover = float(t - shift_idx)
                break

        cache[seed] = {
            "milestones": milestones,
            "windowed_coverage": {k: _windowed_coverage(v) for k, v in milestones.items()},
            "agg_alpha": {k: float(agg_traj[v]) for k, v in milestones.items()},
            "weight_entropy_at_end": float(
                _normalized_entropy_inline(weight_traj[milestones["end_of_trace"]])
            ),
            "min_weight_at_end": float(weight_traj[milestones["end_of_trace"]].min()),
            "recovery_coverage": recovery_coverage,
            "recovery_lag": recovery_lag,
            "time_to_recover_steps": time_to_recover,
            "runtime_seconds": runtime_s,
            "n_intervals": len(intervals),
        }
    return cache


def _normalized_entropy_inline(weights: np.ndarray) -> float:
    """Inline copy of diagnostics.normalized_entropy. Avoiding the
    import here is intentional: the recordings test is non-
    asserting and shouldn't couple to the diagnostics module's
    contract beyond what dtaci.py already exposes via
    ``DtACI.weight_entropy``. (We use the inline form because we
    only need entropy at a snapshot row from the trajectory, not
    the live state of the DtACI instance.)"""
    K = weights.size
    if K < 2:
        return float("nan")
    s = float(weights.sum())
    if s == 0.0:
        return float("nan")
    p = weights / s
    mask = p > 0.0
    p_log_p = np.zeros_like(p)
    p_log_p[mask] = p[mask] * np.log(p[mask])
    return -float(np.sum(p_log_p)) / float(np.log(K))


# -----------------------------------------------------------------------------
# Markdown rendering helpers (mirroring P11.3 layout).
# -----------------------------------------------------------------------------


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
        / "dtaci.py"
    )
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:+.4f}"
    return str(v)


def _render_recordings_md(cache: dict[int, dict]) -> str:
    lines: list[str] = []
    lines.append("# S13 DtACI recovery-rate recordings — `latest.md`")
    lines.append("")
    lines.append(f"- Generated (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"- Git SHA (HEAD): `{_git_sha()}`")
    lines.append(f"- dtaci.py sha256[:16]: `{_module_hash()}`")
    lines.append(
        f"- Setup: hard-regime-shift synthetic, N={N_BARS}, shift "
        f"magnitude={SHIFT_MAGNITUDE}, warmup={WARMUP}, "
        f"DtACI window_size={WINDOW_SIZE}, target α={ALPHA}, "
        f"default γ-grid (0.001, 0.005, 0.02, 0.08)."
    )
    lines.append("")
    lines.append(
        "**Non-asserting**. The asserting pin "
        "(`test_pin_dtaci_recovery_after_hard_shift` in "
        "`test_dtaci_invariants.py`) verifies post-recovery "
        "coverage is within 1pp of target. This artifact captures "
        "the full trajectory so a future regression "
        "(e.g., numpy update shifting trajectories by 0.5pp) is "
        "triageable from the recording."
    )
    lines.append("")

    # Section 1: per-seed milestone snapshot.
    lines.append("## Per-seed milestone trajectory")
    lines.append("")
    lines.append(
        "Windowed coverage (rolling ±25 steps around each milestone "
        "in online-stream coords). `recovery_coverage` = mean hit "
        "rate over `[shift+lag, end_of_trace]` (the asserting pin's "
        "measurement). `time_to_recover_steps` = first online step "
        "post-shift where 50-step rolling coverage hits target±1pp; "
        "NaN if never recovers within trace."
    )
    lines.append("")
    cols = [
        "seed",
        "cov@end_of_warmup",
        "cov@pre_shift",
        "cov@shift_t",
        "cov@shift+lag",
        "cov@shift+2lag",
        "cov@end_of_trace",
        "recovery_coverage",
        "time_to_recover_steps",
        "agg_α@end_of_trace",
        "weight_entropy@end",
        "min_weight@end",
        "runtime_seconds",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for seed in SEEDS:
        row = cache[seed]
        cells = [
            str(seed),
            _fmt(row["windowed_coverage"]["end_of_warmup"]),
            _fmt(row["windowed_coverage"]["pre_shift"]),
            _fmt(row["windowed_coverage"]["shift_t"]),
            _fmt(row["windowed_coverage"]["shift+lag"]),
            _fmt(row["windowed_coverage"]["shift+2lag"]),
            _fmt(row["windowed_coverage"]["end_of_trace"]),
            _fmt(row["recovery_coverage"]),
            _fmt(row["time_to_recover_steps"]),
            _fmt(row["agg_alpha"]["end_of_trace"]),
            _fmt(row["weight_entropy_at_end"]),
            _fmt(row["min_weight_at_end"]),
            f"{row['runtime_seconds']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Section 2: summary stats across seeds.
    lines.append("## Summary across seeds")
    lines.append("")
    rec_cov = np.array([cache[s]["recovery_coverage"] for s in SEEDS])
    ttr = np.array([cache[s]["time_to_recover_steps"] for s in SEEDS], dtype=np.float64)
    we = np.array([cache[s]["weight_entropy_at_end"] for s in SEEDS])
    target = 1.0 - ALPHA
    abs_gap = np.abs(rec_cov - target)
    summary_lines = [
        f"- recovery_coverage:  mean={rec_cov.mean():+.4f}, "
        f"std={rec_cov.std():.4f}, |Δ from target|: max={abs_gap.max():.4f}",
        f"- time_to_recover_steps:  mean={np.nanmean(ttr):.1f}, "
        f"median={np.nanmedian(ttr):.1f}, max={np.nanmax(ttr):.1f}",
        f"- weight_entropy@end:  mean={we.mean():+.4f}, std={we.std():.4f}, min={we.min():+.4f}",
        f"- target coverage = {target}, asserting pin tolerance = 0.01 "
        f"(1pp); recovery_lag = ceil(1/max(γ)) = "
        f"{cache[SEEDS[0]]['recovery_lag']} steps.",
    ]
    lines.extend(summary_lines)
    lines.append("")
    return "\n".join(lines)


def test_record_dtaci_recovery_trajectories(
    recordings_cache: dict[int, dict],
) -> None:
    """Write the recordings artifact. Asserts only that the file
    exists with non-zero size; the content is intentionally not
    pinned (the asserting pin lives in
    test_dtaci_invariants.py)."""
    artifact_dir = Path(__file__).parent / "fixtures" / "dtaci_recordings"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "latest.md"
    md = _render_recordings_md(recordings_cache)
    artifact_path.write_text(md)
    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0
