"""FB1 spike runner: lead-lag empirical-lift comparison.

Compares two ``PathSignatureTransformer`` configs on identical FB1
inputs:

* A — ``depth=2, augmentations=("addtime",)``      (no lead-lag)
* B — ``depth=2, augmentations=("addtime", "lead-lag")``

For each, computes (i) MDA importance-gate pass rate at ``t_stat=2.0``
and (ii) split-conformal interval coverage on a chronological 60/20/20
train/calibration/holdout split. Coverage uses a manual chronological
split rather than ``ConformalAlphaModel(method="split")`` because the
latter shuffles indices in a way that violates temporal ordering on
time-series data (the defect xfailed in
``tests/test_conformal_temporal_order_regressions.py``); the FB1
comparison must be methodologically defensible, so the calibration
slice is post-train, post-1bar-overlap, pre-holdout.

Kill-switch (S18 plan §10):

* Lift criterion: ``(pass_rate_b - pass_rate_a) / pass_rate_a >= 0.05``
* Coverage criterion: ``cov_a - cov_b <= 0.01`` (lead-lag must not
  degrade coverage by more than 1pp absolute)

Trigger fires (drop lead-lag from production default) if EITHER
criterion is violated.

Run:

    uv run --directory quantcore python tests/spikes/run_fb1.py

Emits a JSON result block to stdout. Decision doc is hand-authored
from the metrics; this runner produces them, the human writes the
narrative.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

# Make `tests` package importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quantcore.importance.importance import (  # noqa: E402
    feature_importance_mda,
    importance_gate,
)
from quantcore.preprocessing.path_signature import (  # noqa: E402
    PathSignatureTransformer,
)
from tests.spikes.fb1_leadlag_fixture import (  # noqa: E402
    SEED,
    WINDOW_SIZE,
    build_fb1_inputs,
)


def _chronological_conformal_coverage(
    features: pd.DataFrame,
    targets: pd.Series,
    base_estimator,
    *,
    alpha: float = 0.1,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Split-conformal interval coverage on a chronological holdout.

    Bypasses ``ConformalAlphaModel(method="split")`` to avoid its
    shuffle-on-time-series defect. Implements the canonical split-
    conformal recipe:

      1. Fit the base model on ``X[:n_train]``.
      2. Compute absolute-residual scores on the calibration slice
         ``X[n_train : n_train + n_cal]``.
      3. Set ``q = ceil((n_cal + 1)(1 - alpha))``-th order statistic.
      4. Predict ``y_hold = model.predict(X_hold)``; interval is
         ``[y_hold - q, y_hold + q]``; coverage is the empirical
         fraction of holdout targets in interval.

    Returns a dict with ``coverage`` plus diagnostics (interval width,
    point-prediction MSE) for completeness.
    """
    n = len(features)
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    if n_train + n_cal >= n - 1:
        raise ValueError(f"train_frac + cal_frac too large for n={n}")

    X = features.to_numpy(dtype=np.float64)
    y = targets.to_numpy(dtype=np.float64)

    X_train = X[:n_train]
    X_cal = X[n_train : n_train + n_cal]
    X_hold = X[n_train + n_cal :]
    y_train = y[:n_train]
    y_cal = y[n_train : n_train + n_cal]
    y_hold = y[n_train + n_cal :]

    rng = np.random.default_rng(seed)
    model = clone(base_estimator)
    if hasattr(model, "set_params"):
        model.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
    model.fit(X_train, y_train)

    scores_cal = np.abs(y_cal - model.predict(X_cal))
    n_cal_actual = len(scores_cal)
    q_rank = int(np.ceil((n_cal_actual + 1) * (1.0 - alpha)))
    q_rank = min(max(q_rank, 1), n_cal_actual)
    q = float(np.sort(scores_cal)[q_rank - 1])

    y_pred_hold = model.predict(X_hold)
    lower = y_pred_hold - q
    upper = y_pred_hold + q
    covered = (lower <= y_hold) & (y_hold <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "interval_half_width": q,
        "n_train": n_train,
        "n_cal": n_cal_actual,
        "n_hold": int(len(y_hold)),
        "mse_holdout": float(np.mean((y_hold - y_pred_hold) ** 2)),
    }


def _build_features(bars: pd.DataFrame, augmentations: tuple[str, ...]) -> pd.DataFrame:
    """Fit + transform PathSignatureTransformer at depth=2, drop final no-target row."""
    t = PathSignatureTransformer(
        depth=2,
        augmentations=augmentations,  # type: ignore[arg-type]
        rescaling="post",
        window_size=WINDOW_SIZE,
    )
    t.fit(bars)
    return t.transform(bars).iloc[:-1]  # drop last event (no forward target)


def main() -> dict:
    print(f"FB1 spike: seed={SEED}, window_size={WINDOW_SIZE}")
    bars, targets, _t1, cv = build_fb1_inputs()
    print(f"  bars: {bars.shape}, n_events: {len(targets)}")

    # ----- Build features ------------------------------------------------
    print("\n[features] depth=2, augmentations='addtime' only ...")
    features_a = _build_features(bars, augmentations=("addtime",))
    print(f"  features_a: {features_a.shape}  (D_a = {features_a.shape[1]})")

    print("[features] depth=2, augmentations='addtime' + 'lead-lag' ...")
    features_b = _build_features(bars, augmentations=("addtime", "lead-lag"))
    print(f"  features_b: {features_b.shape}  (D_b = {features_b.shape[1]})")

    assert (features_a.index == targets.index).all(), "feature/target index drift A"
    assert (features_b.index == targets.index).all(), "feature/target index drift B"

    # ----- MDA importance + gate -----------------------------------------
    rf_kwargs = {"n_estimators": 20, "max_depth": 5, "n_jobs": -1, "random_state": SEED}
    rf = RandomForestRegressor(**rf_kwargs)

    print("\n[MDA] running on A ...")
    mda_a = feature_importance_mda(
        rf,
        features_a,
        targets,
        cv,
        scoring="neg_mean_squared_error",
        n_repeats=2,
        random_state=SEED,
    )
    print("[MDA] running on B ...")
    mda_b = feature_importance_mda(
        rf,
        features_b,
        targets,
        cv,
        scoring="neg_mean_squared_error",
        n_repeats=2,
        random_state=SEED,
    )

    gate_a, _ = importance_gate({"mda": mda_a}, min_features=1, t_stat=2.0)
    gate_b, _ = importance_gate({"mda": mda_b}, min_features=1, t_stat=2.0)
    pass_rate_a = len(gate_a) / features_a.shape[1]
    pass_rate_b = len(gate_b) / features_b.shape[1]
    print(f"  pass_rate_a = {pass_rate_a:.6f} ({len(gate_a)}/{features_a.shape[1]})")
    print(f"  pass_rate_b = {pass_rate_b:.6f} ({len(gate_b)}/{features_b.shape[1]})")

    # ----- Conformal holdout coverage ------------------------------------
    print("\n[conformal] chronological 60/20/20 ...")
    cov_a_diag = _chronological_conformal_coverage(features_a, targets, rf, seed=SEED)
    cov_b_diag = _chronological_conformal_coverage(features_b, targets, rf, seed=SEED)
    cov_a = cov_a_diag["coverage"]
    cov_b = cov_b_diag["coverage"]
    print(f"  cov_a = {cov_a:.4f}  (half-width {cov_a_diag['interval_half_width']:.6f})")
    print(f"  cov_b = {cov_b:.4f}  (half-width {cov_b_diag['interval_half_width']:.6f})")

    # ----- Kill-switch ---------------------------------------------------
    rel_lift = (pass_rate_b - pass_rate_a) / max(pass_rate_a, 1e-12)
    cov_diff = cov_a - cov_b
    lift_fail = rel_lift < 0.05
    cov_fail = cov_diff > 0.01
    trigger_fired = bool(lift_fail or cov_fail)

    outcome = "drop_lead_lag" if trigger_fired else "pass"
    production_default = (
        ("basepoint", "addtime") if trigger_fired else ("basepoint", "addtime", "lead-lag")
    )

    print("\n=== KILL-SWITCH ===")
    print(f"  rel_lift = {rel_lift:+.4f}  threshold >= 0.05 -> fail={lift_fail}")
    print(f"  cov_diff = {cov_diff:+.4f}  threshold <= 0.01 -> fail={cov_fail}")
    print(f"  trigger_fired = {trigger_fired}")
    print(f"  outcome = {outcome}")

    return {
        "fb1_outcome": outcome,
        "fixture": "build_fb1_inputs(n=2000, window=64, seed=20260501)",
        "seed": SEED,
        "model": {"name": "RandomForestRegressor", **rf_kwargs},
        "mda": {"scoring": "neg_mean_squared_error", "n_repeats": 2, "n_splits": cv.n_splits},
        "metrics": {
            "pass_rate_a": float(pass_rate_a),
            "pass_rate_b": float(pass_rate_b),
            "cov_a": float(cov_a),
            "cov_b": float(cov_b),
            "rel_lift": float(rel_lift),
            "cov_diff": float(cov_diff),
            "n_features_a": int(features_a.shape[1]),
            "n_features_b": int(features_b.shape[1]),
            "n_passing_a": len(gate_a),
            "n_passing_b": len(gate_b),
        },
        "conformal_diagnostics": {"a": cov_a_diag, "b": cov_b_diag},
        "kill_switch": {
            "lift_threshold": 0.05,
            "lift_fail": bool(lift_fail),
            "cov_threshold": 0.01,
            "cov_fail": bool(cov_fail),
            "trigger_fired": trigger_fired,
        },
        "production_default_augmentations": list(production_default),
    }


if __name__ == "__main__":
    result = main()
    print("\n=== FB1 OUTCOME (json) ===")
    print(json.dumps(result, indent=2))
