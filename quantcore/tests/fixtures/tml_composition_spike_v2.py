"""TML composition fixture v2-B5 — HMM-modulated drift stress test (S12 spike).

V2-B5 fixture mirrors the structural footprint of the S8 fixture
(``tml_composition_spike.py``) but swaps the data-generating process
to test whether canonical numbers are STRUCTURAL or specific to the
S8 AR(1) joint distribution.

Round history (this file is the third iteration of V2):

  Round 1 — pure iid x_info, h=7, c=0.006: gate empty-selects
            because horizon-SNR floors at 1/sqrt(h-1) ≈ 0.41 vs
            V1's empirical 1.11.
  Round 2 — pure iid x_info, h=7, c=0.030 (recalibration attempt):
            same empty-selection. SNR ceiling is c-independent
            for sign-modulated iid drift; bumping c can't escape it.
  Round 3 — THIS FILE: HMM-driven drift breaks the iid SNR ceiling
            because z_t is autocorrelated by construction; iid noise
            survives in the FEATURE channel xi_t. Calibrated by
            full-formula SNR matching at h=5 (matches V1's horizon).

DGP delta from V1 (S8 AR(1) fixture)
------------------------------------

V1 (Gaussian AR(1)):
    x_info[t]   = phi · x_info[t-1] + sqrt(1 - phi²) · N(0, 1)   (phi=0.9)
    drift[t]    = c_v1 · sign(x_info[t])
    eta[t]      = N(0, 1)
    log_ret[t]  = drift[t] + sigma · eta[t]

V2-B5 (HMM-driven drift, iid Student-t feature noise):
    z[t]        = 2-state Markov chain ∈ {-1, +1}, stay-prob p=0.90
    xi[t]       = standardized Student-t(df=4)                   (heavy tails)
    x_info[t]   = z[t] + xi[t]                                    (bimodal feature)
    drift[t]    = c_b5 · z[t]                                     (drift driven by latent regime)
    eta[t]      = N(0, 1)                                         (Gaussian)
    log_ret[t]  = drift[t] + sigma · eta[t]

The persistence (drift coherence over the horizon) lives in the
LATENT state z_t, which has E[regime length] = 1/(1-p) = 10 bars at
p=0.90 — matching V1's effective AR(1) persistence at phi=0.9. The
OBSERVABLE feature x_info inherits z's autocorrelation but its
innovation noise xi is iid heavy-tailed. This decouples the
"horizon-coherent drift" property (preserved) from the
"Gaussian-AR feature" property (changed).

Calibration to V1
-----------------

Full SNR formula (corrected from round-1/2 algebra error):

    SNR_horizon = c · E[S | d_0=+1] / sqrt(c² · Var[S | d_0=+1] + h · sigma²)
    where  S = sum_{s=0}^{h-1} d_{t+s}    (d = sign-process for V1, z for B5)

V1 reference (phi=0.9, h=5, c=0.006, sigma=0.004):
    E[S | x_0>0]  = sum_{s=0}^{4} (2/pi) · arcsin(phi^s) ≈ 3.290  (closed form)
    Var[S | x_0>0] ≈ 6.49  (Monte Carlo, N=200k)
    SNR_v1 ≈ 1.11

V2-B5 calibration (p=0.90, h=5):
    E[S | z_0=+1]  = (1 - (2p-1)^h) / (1 - (2p-1)) ≈ 3.36  (analytic)
    Var[S | z_0=+1] ≈ 6.82  (analytic via E[z_s z_u] = (2p-1)^|s-u|)
    Solve c such that SNR_b5 = 1.11 → c_b5 ≈ 0.00584

Fixture parameters
------------------
- horizon h = 5 (matches V1; single-variable test of DGP swap)
- N = 1500
- features = 5 (1 info + 1 correlated + 3 noise)
- p_regime = 0.90 (E[regime length] = 10 ≈ V1 effective AR(1) persistence)
- c_drift = 0.00584 (calibrated for SNR_b5 ≈ V1's 1.11)
- LOW_SNR_c = c_drift / 3 = 0.00195 (preserves V1's 1/3 ratio)
- vol_coef = 0.004 (unchanged from V1)
- z_0 ~ Uniform({-1, +1}) — stationary distribution of symmetric chain;
  matches V1's stationary AR(1) initialization (no burn-in needed)
- xi noise: Student-t(df=4) standardized to unit variance
- eta noise: Gaussian (return-noise distribution held constant; xi is
  the only heavy-tail surface for clean attribution)
- CUSUM threshold: TBD via pilot at production N=1500; sweep
  [0.005, 0.025] in 16 steps; pick threshold yielding n_events
  closest to 400 within range [380, 420] (matches V1's absolute count
  for downstream sample-size-comparable cv statistics).

ML stack constraint (carried from V2-orig spec)
-----------------------------------------------
S8's MetaLabeler.primary_model is LogisticRegression(max_iter=2000);
RandomForest is the meta_model. V2-B5 mirrors S8's exact ML stack
(LogReg primary, RF meta, RF MDI-forest, LogReg for MDA/SFI) to
hold the estimator surface constant and isolate DGP-attributable
changes.

Spec authorization: 2026-04-25 (round 3 / B5 + (0.90, 5) calibration).
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
# V2-B5 sprint constants.
# -----------------------------------------------------------------------------

FIXTURE_SEEDS_V2: tuple[int, ...] = (42, 7, 123, 2026, 4321)
FIXTURE_N_V2: int = 1500
MDA_N_REPEATS_V2: int = 10  # canonical to test
VOL_SPAN_V2: int = 30  # same as V1
DRIFT_COEF_V2: float = 0.00584  # SNR-matched to V1 ≈ 1.11
VOL_COEF_V2: float = 0.004  # same as V1
VERTICAL_BARS_V2: int = 5  # match V1 (single-var DGP test)
PT_SL_V2: tuple[float, float] = (1.0, 1.0)
EMBARGO_PCT_V2: float = 0.01  # same as V1
N_SPLITS_V2: int = 5  # same as V1
P_REGIME_V2: float = 0.90  # 2-state HMM stay-probability
T_DOF_V2: int = 4  # Student-t df for xi feature noise

# CUSUM threshold: PILOT-DETERMINED at 2026-04-25 against seed=42,
# N=1500, sweep [0.005, 0.025]/16 steps. Picked threshold whose
# n_events is closest to 400 (within [380, 420]). At threshold=0.01567
# the pilot yielded n_events=395 (|395-400|=5; in window).
# V1 literal threshold (0.01) yielded n_events=582 on the same fixture
# — preserved as CUSUM_THRESHOLD_V2_DEFAULT for the sidecar diagnostic
# (count-vs-rate trade-off documented in the V2-B5 spec review).
CUSUM_THRESHOLD_V2_DEFAULT: float = 0.01  # V1 literal (sidecar diagnostic)
CUSUM_THRESHOLD_V2: float = 0.01567  # pilot-set production value

# Low-SNR drift coefficient: preserves V1's 1/3 ratio. SNR_b5 ≈ 0.37
# under this c (sub-V1 by 1/3, matching V1's low-SNR design).
LOW_SNR_DRIFT_COEF_V2: float = DRIFT_COEF_V2 / 3.0

# Rationale strings.
_RATIONALE_PIPELINE_V2 = "S12 V2-B5: HMM-modulated drift; LogReg primary on daily close."
_RATIONALE_BASELINE_V2 = "S12 V2-B5: all-features baseline for pruning comparison."
_RATIONALE_OOS_PROBE_V2 = "S12 V2-B5: OOS meta value-add probe via cross_val_predict."
_RATIONALE_PIN6_V2 = "S12 V2-B5: drop-zero warning capture on full y."


def _simulate_regime(rng: np.random.Generator, n: int, p_stay: float) -> np.ndarray:
    """Simulate a 2-state symmetric Markov chain z[0..n-1] in {-1, +1}.

    Initial state z[0] ~ Uniform({-1, +1}) (stationary distribution
    of the symmetric chain). Stay-prob p_stay; switch-prob 1-p_stay.
    """
    z = np.empty(n, dtype=np.int64)
    z[0] = 1 if rng.random() < 0.5 else -1
    flips = rng.random(n - 1) > p_stay  # True ⇒ switch
    for t in range(1, n):
        z[t] = -z[t - 1] if flips[t - 1] else z[t - 1]
    return z


def build_fixture_v2(
    seed: int,
    n: int = FIXTURE_N_V2,
    *,
    drift_coef: float = DRIFT_COEF_V2,
    vol_coef: float = VOL_COEF_V2,
    p_regime: float = P_REGIME_V2,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build the V2-B5 deterministic close + feature panel.

    DGP (HMM-driven drift, iid Student-t feature noise):
        z[t]        ~ Markov chain ∈ {-1, +1}, stay-prob p_regime
        xi[t]       ~ iid Student-t(df=4) standardized to unit variance
        x_info[t]   = z[t] + xi[t]
        drift[t]    = drift_coef · z[t]
        eta[t]      ~ iid N(0, 1)
        log_ret[t]  = drift[t] + vol_coef · eta[t]

    Features (5 total):
      x_info       : z[t] + xi[t]                            (bimodal informative)
      x_correlated : 0.6 · x_info[t] + 0.4 · N(0, 1)         (correlated informative)
      x_noise_1/2/3: iid N(0, 1)                              (pure noise)

    Returns (close, features) with shape contract identical to V1.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")

    # Latent regime sequence (drives drift; persistence p_regime).
    z = _simulate_regime(rng, n, p_regime)

    # Student-t(df=4) standardized to unit variance.
    # Var(t_df) = df/(df-2) for df>2; df=4 → var=2; rescale by sqrt(2).
    xi_raw = rng.standard_t(T_DOF_V2, size=n)
    xi = xi_raw / np.sqrt(T_DOF_V2 / (T_DOF_V2 - 2))

    # Informative feature: bimodal z + heavy-tailed iid xi.
    x_info = z.astype(np.float64) + xi

    # iid noise features.
    x_noise_1 = rng.standard_normal(n)
    x_noise_2 = rng.standard_normal(n)
    x_noise_3 = rng.standard_normal(n)

    # Correlated informative feature.
    x_correlated = 0.6 * x_info + 0.4 * rng.standard_normal(n)

    # Return path: drift carried by z; Gaussian innovation eta.
    eta = rng.standard_normal(n)
    log_ret = drift_coef * z + vol_coef * eta
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="close")

    features = pd.DataFrame(
        {
            "x_info": x_info,
            "x_noise_1": x_noise_1,
            "x_noise_2": x_noise_2,
            "x_noise_3": x_noise_3,
            "x_correlated": x_correlated,
        },
        index=idx,
    )
    return close, features


def _scorer_accuracy_v2(est: Any, X_test: Any, y_test: Any) -> float:
    return float(accuracy_score(y_test, est.predict(X_test)))


def _metalabeler_v2(seed: int, rationale: str) -> MetaLabeler:
    """Construct the canonical MetaLabeler — identical to V1's stack."""
    return MetaLabeler(
        primary_model=LogisticRegression(max_iter=2000),
        meta_model=RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1),
        economic_rationale=rationale,
        meta_features_oos=True,
        oos_cv=KFold(n_splits=N_SPLITS_V2, shuffle=True, random_state=seed),
    )


def run_pipeline_v2(
    seed: int,
    *,
    drift_coef: float = DRIFT_COEF_V2,
    vol_coef: float = VOL_COEF_V2,
    p_regime: float = P_REGIME_V2,
    mda_n_repeats: int = MDA_N_REPEATS_V2,
    cusum_threshold: float = CUSUM_THRESHOLD_V2,
) -> dict[str, Any]:
    """Execute the full AFML Ch. 3 pipeline on the V2-B5 fixture.

    Same end-to-end pipeline as V1's run_pipeline, parameterized for
    spike-time exploration:
      - drift_coef / vol_coef: SNR control.
      - p_regime: regime persistence (locked to V2-B5 spec at 0.90).
      - mda_n_repeats: MDA permutation count (canonical 10).
      - cusum_threshold: CUSUM event threshold (set by pilot).

    Returns the same dict shape as V1's run_pipeline plus
    ``cusum_threshold_used`` and ``mda_n_repeats_used``.
    """
    close, features = build_fixture_v2(
        seed,
        drift_coef=drift_coef,
        vol_coef=vol_coef,
        p_regime=p_regime,
    )

    # --- labels ---
    target = get_daily_vol(close, span=VOL_SPAN_V2).dropna()
    t_events_raw = cusum_filter(close, threshold=cusum_threshold)
    t_events = pd.DatetimeIndex(t_events_raw.intersection(target.index))
    cfg = TripleBarrierConfig(vertical_bars=VERTICAL_BARS_V2, pt_sl=PT_SL_V2, min_ret=0.0)
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

    # --- MDI (ranking; NOT for gating per S8 §DD#1 + S11 P10.1) ---
    rf_mdi = RandomForestClassifier(
        n_estimators=200,
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

    # --- MDA (parameterized n_repeats for spike stability check) ---
    cv_imp = KFold(n_splits=N_SPLITS_V2, shuffle=True, random_state=seed)
    mda = feature_importance_mda(
        LogisticRegression(max_iter=2000),
        X_a,
        pd.Series(y_a_sign, index=X_a.index),
        cv=cv_imp,
        scoring="neg_log_loss",
        n_repeats=mda_n_repeats,
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

    # --- gate: canonical {MDA, SFI} (MDI excluded per S8 §DD#1) ---
    selected, gate_passed = importance_gate({"mda": mda, "sfi": sfi}, min_features=1, t_stat=2.0)
    selected_with_mdi, gate_passed_with_mdi = importance_gate(
        {"mdi": mdi, "mda": mda, "sfi": sfi},
        min_features=1,
        t_stat=2.0,
        allow_mdi=True,
    )

    X_sel = X_a[list(selected)] if selected else X_a

    # --- Pin-6-analogue: full-y MetaLabeler.fit for drop-zero warning ---
    pin6_ml = _metalabeler_v2(seed, _RATIONALE_PIN6_V2)
    with warnings.catch_warnings(record=True) as pin6_caught:
        warnings.simplefilter("always")
        pin6_ml.fit(
            X.to_numpy(),
            y.to_numpy().astype(int),
            sample_weight=weights.to_numpy(),
        )
        pin6_warnings = list(pin6_caught)

    # --- cv_score_purged on selected features ---
    ml_selected = _metalabeler_v2(seed, _RATIONALE_PIPELINE_V2)
    with warnings.catch_warnings(record=True) as cv_caught:
        warnings.simplefilter("always")
        cv_scores_selected = cv_score_purged(
            estimator=ml_selected,
            X=X_sel,
            y=y_a_int,
            sample_weight=w_a,
            t1=t1_a,
            embargo_pct=EMBARGO_PCT_V2,
            n_splits=N_SPLITS_V2,
            scoring=_scorer_accuracy_v2,
        )
        cv_warnings = list(cv_caught)

    # --- cv_score_purged on all features ---
    ml_all = _metalabeler_v2(seed, _RATIONALE_BASELINE_V2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores_all = cv_score_purged(
            estimator=ml_all,
            X=X_a,
            y=y_a_int,
            sample_weight=w_a,
            t1=t1_a,
            embargo_pct=EMBARGO_PCT_V2,
            n_splits=N_SPLITS_V2,
            scoring=_scorer_accuracy_v2,
        )

    # --- OOS value-add probe ---
    purged_cv_outer = PurgedKFold(n_splits=N_SPLITS_V2, t1=t1_a, embargo_pct=EMBARGO_PCT_V2)
    ml_oos_eval = _metalabeler_v2(seed, _RATIONALE_OOS_PROBE_V2)
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

    mdi_mean_col = cast(pd.Series, mdi["mean"])
    mdi_sum = float(mdi_mean_col.sum())
    mdi_sum_deviation = abs(mdi_sum - 1.0)

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
        "mda_n_repeats_used": mda_n_repeats,
        "cusum_threshold_used": cusum_threshold,
    }
