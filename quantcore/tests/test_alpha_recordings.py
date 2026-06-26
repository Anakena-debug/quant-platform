"""Non-asserting alpha-branches recordings artifact (S14 P13.1).

Mirrors S12 P11.3 and S13 P12.2 patterns: a single-file Markdown
artifact with summary statistics for each ConformalAlphaModel
branch (split / cv / cqr / mondrian) on a fixed-seed two-regime
synthetic. Single-file overwrite, gitignored.

The asserting byte-exact pin
(``test_alpha_branches_byte_exact.py``) verifies the existing
branches haven't drifted. This artifact captures branch-level
summary stats — coverage, mean width, signal strength, tradeable
fraction — so a future qualitative regression (e.g., a numpy
update that shifts coverage by 1pp without breaking byte-exact)
shows up clearly when read by a human.

For mondrian specifically, we record per-stratum coverage so the
S13 P12.3 per-stratum guarantee (≥ 1-α-0.03) is visible at the
integration level too.

Output: ``quantcore/tests/fixtures/alpha_recordings/latest.md``.
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.finance.alpha import ConformalAlphaModel


SEED: int = 11
N_BARS: int = 600
N_TRAIN: int = 450


def _two_regime_synthetic(seed: int = SEED, n: int = N_BARS):
    """Random regime assignment so any contiguous slice (e.g.,
    train/test split) contains both strata. Using a half-half
    block split would put the test set entirely in one regime."""
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 2, size=n).astype(np.int_)
    X_base = rng.standard_normal((n, 3))
    X = np.column_stack([X_base, regime.astype(np.float64)])
    noise_scale = np.where(regime == 0, 1.0, 0.3)
    y = X_base.sum(axis=1) + noise_scale * rng.standard_normal(n)
    return X, y


def _stratifier(X_in: np.ndarray) -> np.ndarray:
    return X_in[:, -1].astype(np.int_)


@pytest.fixture(scope="module")
def recordings_cache() -> dict[str, dict]:
    """Build per-branch summary stats on the same synthetic."""
    X, y = _two_regime_synthetic()
    X_tr, X_te = X[:N_TRAIN], X[N_TRAIN:]
    y_tr, y_te = y[:N_TRAIN], y[N_TRAIN:]
    test_strata = _stratifier(X_te)

    cache: dict[str, dict] = {}

    def _summarize(name: str, sig, runtime_s: float) -> dict:
        covered = (sig.lower <= y_te) & (y_te <= sig.upper)
        per_stratum_cov: dict[int, float] = {}
        for s in np.unique(test_strata):
            mask = test_strata == s
            per_stratum_cov[int(s)] = float(covered[mask].mean())
        return {
            "name": name,
            "n_test": int(sig.lower.shape[0]),
            "coverage": float(covered.mean()),
            "per_stratum_coverage": per_stratum_cov,
            "mean_width": float((sig.upper - sig.lower).mean()),
            "median_width": float(np.median(sig.upper - sig.lower)),
            "tradeable_fraction": float(sig.tradeable.mean()),
            "mean_signal_strength": float(sig.signal_strength.mean()),
            "runtime_seconds": runtime_s,
        }

    # split
    t0 = time.perf_counter()
    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="split",
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    cache["split"] = _summarize("split", m.predict(X_te), time.perf_counter() - t0)

    # cv
    t0 = time.perf_counter()
    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="cv",
        n_folds=5,
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    cache["cv"] = _summarize("cv", m.predict(X_te), time.perf_counter() - t0)

    # cqr
    t0 = time.perf_counter()
    gbr = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED)
    m = ConformalAlphaModel(
        model=gbr,
        alpha=0.1,
        method="cqr",
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    cache["cqr"] = _summarize("cqr", m.predict(X_te), time.perf_counter() - t0)

    # mondrian
    t0 = time.perf_counter()
    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        stratifier=_stratifier,
        random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    cache["mondrian"] = _summarize("mondrian", m.predict(X_te), time.perf_counter() - t0)

    return cache


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


def _alpha_module_hash() -> str:
    p = (
        Path(__file__).parent.parent
        / "src"
        / "quantcore"
        / "uncertainty"
        / "conformal"
        / "finance"
        / "alpha.py"
    )
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:+.4f}"
    return str(v)


def _render_md(cache: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# S14 alpha-branches recordings — `latest.md`")
    lines.append("")
    lines.append(f"- Generated (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"- Git SHA (HEAD): `{_git_sha()}`")
    lines.append(f"- alpha.py sha256[:16]: `{_alpha_module_hash()}`")
    lines.append(
        f"- Setup: 2-regime synthetic, seed={SEED}, N={N_BARS}, "
        f"N_train={N_TRAIN}, target α=0.1. Strata: regime 0 "
        f"(high-vol, σ=1.0) and regime 1 (low-vol, σ=0.3)."
    )
    lines.append("")
    lines.append(
        "**Non-asserting**. The asserting pin "
        "(`test_alpha_branches_byte_exact.py`) verifies bitwise-"
        "exact AlphaSignal arrays for split/cv/cqr branches "
        "against pre-S14 references. This artifact captures "
        "qualitative branch-level summary stats (coverage, mean "
        "width, signal strength, per-stratum coverage) so a "
        "future regression that doesn't break byte-exact still "
        "shows up to a human reader."
    )
    lines.append("")

    cols = [
        "branch",
        "n_test",
        "coverage",
        "stratum0_cov",
        "stratum1_cov",
        "mean_width",
        "median_width",
        "tradeable_fraction",
        "mean_signal_strength",
        "runtime_seconds",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for branch in ("split", "cv", "cqr", "mondrian"):
        row = cache[branch]
        per = row["per_stratum_coverage"]
        cells = [
            branch,
            str(row["n_test"]),
            _fmt(row["coverage"]),
            _fmt(per.get(0, float("nan"))),
            _fmt(per.get(1, float("nan"))),
            _fmt(row["mean_width"]),
            _fmt(row["median_width"]),
            _fmt(row["tradeable_fraction"]),
            _fmt(row["mean_signal_strength"]),
            f"{row['runtime_seconds']:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Per-stratum coverage on mondrian should clear the S13 "
        "P12.3 threshold of 1-α-0.03 = 0.87 on each stratum; the "
        "split/cv/cqr branches are NOT calibrated per-stratum, so "
        "their stratum0/stratum1 coverages may diverge from the "
        "global rate (this is the comparison the S14 plan defers "
        "to S16+ as empirical-comparison work)."
    )
    return "\n".join(lines)


def test_record_alpha_branch_summary(recordings_cache: dict[str, dict]) -> None:
    """Write the alpha-branches recordings artifact. Asserts only
    file existence + non-zero size; content is intentionally not
    pinned (asserting pin lives in
    test_alpha_branches_byte_exact.py).
    """
    artifact_dir = Path(__file__).parent / "fixtures" / "alpha_recordings"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "latest.md"
    md = _render_md(recordings_cache)
    artifact_path.write_text(md)
    assert artifact_path.exists()
    assert artifact_path.stat().st_size > 0
