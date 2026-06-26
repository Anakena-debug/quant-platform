"""TML composition sprint (S8) fixture + pipeline runner.

Deterministic fixture builder and full-pipeline executor for the
AFML Chapter 3 triple-barrier meta-labeling pipeline pin
(``test_tml_composition.py``).

Pipeline
--------

    get_daily_vol + cusum_filter + get_events
        → apply_triple_barrier
        → get_sample_weights
        → feature_importance_{mdi, mda(n_repeats=10), sfi}
        → importance_gate({mda, sfi}, t_stat=2.0)
        → MetaLabeler(meta_features_oos=True,
                      oos_cv=KFold(5, shuffle=True, random_state=seed))
        → cv_score_purged(PurgedKFold(n_splits=5, embargo_pct=0.01))

Two design decisions from §Design decision #1 / #2 are baked in:
  * canonical gate is {MDA, SFI} — MDI is out of t-stat gating per
    the simplex-normalization argument (still computed for ranking)
  * MDA uses ``n_repeats=10`` (canonical ``MDA_N_REPEATS``); default
    3 produces a 1/5 seed false-positive that ``n_repeats=10`` resolves
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_predict

from quantcore.cv.purged_kfold import PurgedKFold, cv_score_purged
from quantcore.importance.importance import (
    feature_importance_mda,
    feature_importance_mdi,
    feature_importance_sfi,
    importance_gate,
)
from quantcore.labels.labelling import (
    TripleBarrierConfig,
    apply_triple_barrier,
    cusum_filter,
    get_daily_vol,
    get_events,
)
from quantcore.labels.meta import MetaLabeler
from quantcore.weights.bootstrap import BootstrapConfig, get_sample_weights

# -----------------------------------------------------------------------------
# Canonical sprint constants.
# -----------------------------------------------------------------------------
# Changing any of these invalidates the empirical pin margins
# calibrated in the S8 plan (spike v4, 2026-04-24). Any modification
# requires re-running the spike and updating the plan's observed-
# ranges table.

FIXTURE_SEEDS: tuple[int, ...] = (42, 7, 123, 2026, 4321)
FIXTURE_N: int = 1000
MDA_N_REPEATS: int = 10
CUSUM_THRESHOLD: float = 0.01
VOL_SPAN: int = 30
AR_PHI: float = 0.9
DRIFT_COEF: float = 0.006
VOL_COEF: float = 0.004
VERTICAL_BARS: int = 5
PT_SL: tuple[float, float] = (1.0, 1.0)
EMBARGO_PCT: float = 0.01
N_SPLITS: int = 5

# S11 P10.3: low-SNR weak-signal coefficient (drift/noise = 0.5 vs
# canonical 1.5). Used by test_low_snr_graceful_degradation in
# test_tml_composition.py to pin the pipeline's behavior near
# breakdown.
LOW_SNR_DRIFT_COEF: float = 0.002

_RATIONALE_PIPELINE = "S8 composition pin: AR(1) drift with LogReg primary on daily close."
_RATIONALE_BASELINE = "S8 composition pin: all-features baseline for pruning comparison."
_RATIONALE_OOS_PROBE = "S8 composition pin: OOS meta value-add probe via cross_val_predict."
_RATIONALE_PIN6 = "S8 composition pin: drop-zero warning capture on full y."


def build_fixture(
    seed: int,
    n: int = FIXTURE_N,
    *,
    drift_coef: float = DRIFT_COEF,
    vol_coef: float = VOL_COEF,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the deterministic close + feature panel on ``n`` daily bars.

    ``x_info`` is AR(1) with ``phi=AR_PHI``, normalized so that
    stationary ``Var(x_info) = 1`` (via innovation coefficient
    ``sqrt(1 - phi**2)``). Its persistence is what makes it *informative*
    about the 5-bar-ahead triple-barrier outcome.

    ``x_correlated`` has ``rho(x_info, x_correlated) ~= 0.83`` on this
    parametrization. It is NOT a true-redundancy probe — it carries
    marginal predictive power of its own through the correlation.
    True-redundancy fixture filed as ticket #3 (see S8 plan).

    ``close`` follows ``log-return = drift_coef * sign(x_info) +
    vol_coef * N(0, 1)``. ``drift_coef`` and ``vol_coef`` parameterize
    the SNR — defaults match S8's canonical strong-signal regime
    (drift/noise = 1.5 per bar). S11 P10.3's
    ``test_low_snr_graceful_degradation`` overrides drift_coef to
    LOW_SNR_DRIFT_COEF (drift/noise = 0.5).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    x_info = np.empty(n)
    x_info[0] = rng.standard_normal()
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x_info[t] = AR_PHI * x_info[t - 1] + np.sqrt(1.0 - AR_PHI * AR_PHI) * eps[t]

    x_noise_1 = rng.standard_normal(n)
    x_noise_2 = rng.standard_normal(n)
    x_correlated = 0.6 * x_info + 0.4 * rng.standard_normal(n)

    drift = drift_coef * np.sign(x_info)
    innov = vol_coef * rng.standard_normal(n)
    log_ret = drift + innov
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="close")

    features = pd.DataFrame(
        {
            "x_info": x_info,
            "x_noise_1": x_noise_1,
            "x_noise_2": x_noise_2,
            "x_correlated": x_correlated,
        },
        index=idx,
    )
    return close, features


def _scorer_accuracy(est: Any, X_test: Any, y_test: Any) -> float:
    return float(accuracy_score(y_test, est.predict(X_test)))


def _metalabeler(seed: int, rationale: str) -> MetaLabeler:
    """Construct the canonical MetaLabeler for this sprint."""
    return MetaLabeler(
        primary_model=LogisticRegression(max_iter=2000),
        meta_model=RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1),
        economic_rationale=rationale,
        meta_features_oos=True,
        oos_cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
    )


def run_pipeline(
    seed: int,
    *,
    drift_coef: float = DRIFT_COEF,
    vol_coef: float = VOL_COEF,
) -> dict[str, Any]:
    """Execute the full AFML Ch. 3 pipeline on a seeded fixture.

    Returns a dict of numerical artefacts consumed by
    ``test_tml_composition.py``'s 20 pins. Never prints. Captures
    warnings from two distinct fit sites separately (``pin6_warnings``
    and ``cv_warnings``).
    """
    close, features = build_fixture(seed, drift_coef=drift_coef, vol_coef=vol_coef)

    # --- labels ---
    target = get_daily_vol(close, span=VOL_SPAN).dropna()
    # cusum_filter returns a DatetimeIndex; .intersection() widens
    # the stub type — narrow back explicitly.
    t_events_raw = cusum_filter(close, threshold=CUSUM_THRESHOLD)
    t_events = pd.DatetimeIndex(t_events_raw.intersection(target.index))
    cfg = TripleBarrierConfig(vertical_bars=VERTICAL_BARS, pt_sl=PT_SL, min_ret=0.0)
    events = get_events(close, t_events, target, cfg)
    labels = apply_triple_barrier(close, events)

    # --- weights (AFML §4.10) ---
    t1_labels = cast(pd.Series, labels["t1"])
    weights = get_sample_weights(
        close,
        t1_labels,
        config=BootstrapConfig(normalize_weights_to_n=True),
    )

    # --- feature matrix aligned on label index ---
    X = cast(pd.DataFrame, features.loc[labels.index].copy())
    y = cast(pd.Series, labels["bin"].copy())

    # --- active subset (non-zero labels) ---
    active = y != 0
    X_a = cast(pd.DataFrame, X[active])
    y_a = cast(pd.Series, y[active])
    w_a = cast(pd.Series, weights[active])
    t1_a = cast(pd.Series, labels.loc[X_a.index, "t1"])
    y_a_int = y_a.astype(int)
    y_a_sign = np.sign(y_a.to_numpy()).astype(int)

    # --- MDI (for ranking + regression pin, NOT for gating) ---
    rf_mdi = RandomForestClassifier(
        n_estimators=200,
        # Stubs type max_features narrowly as str; runtime accepts int.
        max_features=1,  # pyright: ignore[reportArgumentType]
        random_state=seed,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_mdi.fit(X_a.to_numpy(), y_a_sign)
    mdi = feature_importance_mdi(
        rf_mdi,
        feature_names=list(X_a.columns),
        X=X_a.to_numpy(),
        y=y_a_sign,
        method="oob_corrected",
    )

    # --- MDA (canonical n_repeats=10) ---
    cv_imp = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    mda = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X_a,
        pd.Series(y_a_sign, index=X_a.index),
        cv=cv_imp,
        scoring="neg_log_loss",
        n_repeats=MDA_N_REPEATS,
    )

    # --- SFI (baseline="prior" per S7) ---
    sfi = feature_importance_sfi(
        LogisticRegression(max_iter=2000),
        X_a,
        pd.Series(y_a_sign, index=X_a.index),
        cv=cv_imp,
        scoring="neg_log_loss",
        baseline="prior",
    )

    # --- gate: canonical {MDA, SFI} (MDI excluded per §DD #1) ---
    selected, gate_passed = importance_gate({"mda": mda, "sfi": sfi}, min_features=1, t_stat=2.0)
    selected_with_mdi, gate_passed_with_mdi = importance_gate(
        {"mdi": mdi, "mda": mda, "sfi": sfi},
        min_features=1,
        t_stat=2.0,
        allow_mdi=True,  # S11: explicit opt-in for the design-regression pin (Pin 10).
    )

    X_sel = X_a[list(selected)] if selected else X_a

    # --- Pin 6 support: full-y MetaLabeler.fit for drop-zero warning ---
    pin6_ml = _metalabeler(seed, _RATIONALE_PIN6)
    with warnings.catch_warnings(record=True) as pin6_caught:
        warnings.simplefilter("always")
        pin6_ml.fit(
            X.to_numpy(),
            y.to_numpy().astype(int),
            sample_weight=weights.to_numpy(),
        )
        pin6_warnings = list(pin6_caught)

    # --- cv_score_purged on selected features (Pin 15/17/20) ---
    ml_selected = _metalabeler(seed, _RATIONALE_PIPELINE)
    with warnings.catch_warnings(record=True) as cv_caught:
        warnings.simplefilter("always")
        cv_scores_selected = cv_score_purged(
            estimator=ml_selected,
            X=X_sel,
            y=y_a_int,
            sample_weight=w_a,
            t1=t1_a,
            embargo_pct=EMBARGO_PCT,
            n_splits=N_SPLITS,
            scoring=_scorer_accuracy,
        )
        cv_warnings = list(cv_caught)

    # --- cv_score_purged on all features (Pin 16 comparison) ---
    ml_all = _metalabeler(seed, _RATIONALE_BASELINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores_all = cv_score_purged(
            estimator=ml_all,
            X=X_a,
            y=y_a_int,
            sample_weight=w_a,
            t1=t1_a,
            embargo_pct=EMBARGO_PCT,
            n_splits=N_SPLITS,
            scoring=_scorer_accuracy,
        )

    # --- OOS value-add probe (Pin 18 + 19) ---
    # Nested CV: outer PurgedKFold for OOS predictions, inner KFold
    # (inside MetaLabeler) for OOS meta features. Inner primary fits
    # fire benign warnings that are unrelated to outer-loop leakage;
    # suppress them here. Pin 20 uses cv_warnings from the non-nested
    # cv_score_purged path above.
    purged_cv_outer = PurgedKFold(n_splits=N_SPLITS, t1=t1_a, embargo_pct=EMBARGO_PCT)
    ml_oos_eval = _metalabeler(seed, _RATIONALE_OOS_PROBE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oos_meta_pred = np.asarray(
            cross_val_predict(
                ml_oos_eval,
                X_sel.to_numpy(),
                y_a_int.to_numpy(),
                cv=purged_cv_outer,
            )
        )
        oos_primary_pred = np.asarray(
            cross_val_predict(
                LogisticRegression(max_iter=2000),
                X_sel.to_numpy(),
                y_a_sign,
                cv=purged_cv_outer,
            )
        )

    abstain_oos = float((oos_meta_pred == 0).mean())
    primary_oos_overall_acc = float((np.sign(oos_primary_pred) == y_a_sign).mean())
    takes_mask = oos_meta_pred != 0
    if takes_mask.sum() > 0:
        primary_oos_acc_on_takes = float(
            (np.sign(oos_primary_pred[takes_mask]) == y_a_sign[takes_mask]).mean()
        )
    else:
        primary_oos_acc_on_takes = float("nan")
    precision_lift = primary_oos_acc_on_takes - primary_oos_overall_acc

    # --- simplex sum invariant (Pin 9) ---
    mdi_mean_col = cast(pd.Series, mdi["mean"])
    mdi_sum = float(mdi_mean_col.sum())
    mdi_sum_deviation = abs(mdi_sum - 1.0)

    # --- bin counts (Pin 3 + Pin 6 count-match) ---
    bin_minus1 = int((y == -1).sum())
    bin_zero = int((y == 0).sum())
    bin_plus1 = int((y == 1).sum())
    n_active = int(active.sum())

    return {
        "seed": seed,
        "close": close,
        "features": features,
        "events": events,
        "labels": labels,
        "weights": weights,
        "X": X,
        "y": y,
        "X_a": X_a,
        "y_a": y_a,
        "w_a": w_a,
        "t1_a": t1_a,
        "mdi": mdi,
        "mda": mda,
        "sfi": sfi,
        "mdi_sum": mdi_sum,
        "mdi_sum_deviation": mdi_sum_deviation,
        "selected": list(selected),
        "selected_with_mdi": list(selected_with_mdi),
        "gate_passed": gate_passed,
        "gate_passed_with_mdi": gate_passed_with_mdi,
        "cv_scores_selected": cv_scores_selected,
        "cv_scores_all": cv_scores_all,
        "cv_mean_selected": float(cv_scores_selected.mean()),
        "cv_mean_all": float(cv_scores_all.mean()),
        "pin6_warnings": pin6_warnings,
        "cv_warnings": cv_warnings,
        "oos_meta_pred": oos_meta_pred,
        "oos_primary_pred": oos_primary_pred,
        "abstain_oos": abstain_oos,
        "primary_oos_overall_acc": primary_oos_overall_acc,
        "primary_oos_acc_on_takes": primary_oos_acc_on_takes,
        "precision_lift": precision_lift,
        "bin_minus1": bin_minus1,
        "bin_zero": bin_zero,
        "bin_plus1": bin_plus1,
        "n_active": n_active,
        "n_events": len(events),
        "n_labels": len(labels),
    }
