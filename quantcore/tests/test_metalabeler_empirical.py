"""Empirical tests for MetaLabeler (S5 / P4.2, AFML §3.5).

Covers sample-weight pass-through (pins 15-16) and the three
calibrated precision/volume/monotonicity pins (17-19) from the
sprint plan. Thresholds were set via the pre-commit P4.0
calibration spike on a heteroskedastic-noise fixture (5 seeds × 2
fixtures). Pin 18 (volume ≤ 0.70) has the tightest margin (~1.4σ
from max); see plan §"CI-flap watch".

Composition pattern pinned in the docstring:
    MetaLabeler + PurgedKFold + feature_importance_mdi(oob_corrected)
    + sample weights from quantcore.weights.get_sample_weights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from quantcore.labels import MetaLabeler


# -----------------------------------------------------------------------------
# Sample-weight pins (LogisticRegression meta — tree models ignore small-
# weight deltas, so the pin is only well-posed on a linear classifier).
# -----------------------------------------------------------------------------


def _weight_fixture(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    signal = X[:, 0] + 0.6 * X[:, 1]
    y = np.sign(signal + 0.8 * rng.standard_normal(n)).astype(int)
    return X, y


def _make_logreg_metalabeler():
    # Sample-weight invariance pins test the IS path (deterministic on
    # the full training set, not the OOS cross_val_predict path).
    # Explicit opt-in to IS post-S8 default flip.
    return MetaLabeler(
        primary_model=LogisticRegression(penalty=None, max_iter=2000),
        meta_model=LogisticRegression(penalty=None, max_iter=2000),
        economic_rationale="sample-weight pin fixture",
        meta_features_oos=False,
    )


def test_pin15_uniform_weight_scale_invariance() -> None:
    """LogReg(penalty=None): fit with w=1 vs w=2·1 yields bit-identical
    ``meta_.coef_`` — uniform rescaling is absorbed by the un-regularized
    MLE objective. Pins the pass-through path: the scale-invariance
    property holds only if weights actually reach the meta's fit.
    """
    X, y = _weight_fixture(300, 0)
    n = X.shape[0]
    w1 = np.ones(n, dtype=np.float64)
    w2 = 2.0 * np.ones(n, dtype=np.float64)

    ml_w1 = _make_logreg_metalabeler()
    ml_w2 = _make_logreg_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_w1.fit(X, y, sample_weight=w1)
        ml_w2.fit(X, y, sample_weight=w2)

    np.testing.assert_allclose(
        ml_w1.meta_.coef_,
        ml_w2.meta_.coef_,
        atol=1e-8,
        err_msg="Uniform weight rescaling should leave meta.coef_ "
        "unchanged on un-regularized LogReg — mismatch suggests "
        "sample_weight did not reach the meta fit.",
    )


def test_pin16_imbalanced_weight_changes_coef() -> None:
    """LogReg(penalty=None): fit with uniform w vs imbalanced w
    (heavy on the positive class) yields materially different
    ``meta_.coef_``. Pins that weight IS being consumed by the meta
    (not silently discarded).
    """
    X, y = _weight_fixture(300, 0)
    n = X.shape[0]
    w_uniform = np.ones(n, dtype=np.float64)
    w_imbal = np.ones(n, dtype=np.float64)
    active_pos = y == 1
    w_imbal[active_pos] = 5.0

    ml_u = _make_logreg_metalabeler()
    ml_i = _make_logreg_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_u.fit(X, y, sample_weight=w_uniform)
        ml_i.fit(X, y, sample_weight=w_imbal)

    coef_diff = np.linalg.norm(ml_u.meta_.coef_ - ml_i.meta_.coef_)
    assert coef_diff > 1e-3, (
        f"Imbalanced sample_weight should change meta.coef_ (L2 norm of "
        f"diff was {coef_diff:.2e}); weight is being silently dropped."
    )


def test_pin16b_sample_weight_mask_aligned_with_drop_zero() -> None:
    """When drop_zero=True drops k rows, sample_weight must be masked
    by the same boolean. Pin by observing that passing
    ``sample_weight`` of length n (original) works without shape errors.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = np.sign(X[:, 0] + 0.6 * X[:, 1] + 0.5 * rng.standard_normal(300)).astype(float)
    # Inject zeros.
    zero_mask = rng.random(300) < 0.15
    y[zero_mask] = 0.0
    w = rng.uniform(0.5, 2.0, size=300)

    ml = _make_logreg_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Must not raise "sample_weight length mismatch" — the
        # zero-mask must be applied consistently to X, y, and weight.
        ml.fit(X, y, sample_weight=w)
    assert ml.meta_.n_features_in_ == X.shape[1] + 1


# -----------------------------------------------------------------------------
# Empirical pins — heteroskedastic fixture, GBM meta, 5 seeds.
# Thresholds from pre-commit P4.0 calibration spike (2026-04-23).
# -----------------------------------------------------------------------------

SEEDS = [0, 1, 2, 42, 20260423]


def _heteroskedastic_fixture(n: int, seed: int):
    """Pre-commit spike fixture — feature 3 drives noise amplitude
    (NOT direction), so meta has signal to filter on beyond what
    linear primary captures."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    signal = X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 2]
    noise_amp = 0.5 + 1.5 * np.abs(X[:, 3])
    noise = noise_amp * rng.standard_normal(n)
    y = np.sign(signal + noise).astype(int)
    return X, y


def _run_empirical(seed: int):
    """Fit MetaLabeler on the calibration fixture and return test-set
    metrics needed by pins 17, 18, 19.
    """
    X, y = _heteroskedastic_fixture(1500, seed)
    X_tr, X_te = X[:1000], X[1000:]
    y_tr, y_te = y[:1000], y[1000:]

    # Empirical calibration pins 17/18/19 compute precision-lift on
    # IS meta features — explicit opt-in post-S8 default flip.
    ml = MetaLabeler(
        primary_model=LogisticRegression(
            penalty=None,
            max_iter=2000,
            random_state=seed,
        ),
        meta_model=GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=seed,
        ),
        economic_rationale="heteroskedastic-noise empirical fixture",
        meta_features_oos=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X_tr, y_tr)

    # Test-set precision of primary alone (takes all non-zero rows).
    primary_pred_te = ml.primary_.predict(X_te)
    z_te = (np.sign(primary_pred_te) == np.sign(y_te)).astype(int)
    primary_prec = float(z_te.mean())

    # Meta proba on test.
    meta_proba = ml.predict_proba(X_te)[:, 1]

    precs = {}
    vols = {}
    for tau in [0.3, 0.5, 0.6, 0.7, 0.8]:
        take = meta_proba > tau
        n_take = int(take.sum())
        vols[tau] = n_take / len(y_te)
        precs[tau] = float(z_te[take].mean()) if n_take else float("nan")

    return {
        "primary_prec": primary_prec,
        "precs": precs,
        "vols": vols,
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_pin17_precision_lift_at_tau08(seed: int) -> None:
    """Pin 17: at τ=0.8, meta-filtered precision exceeds primary
    precision by ≥ 0.04. Spike 5-seed stats: mean +9.9pp, min +7.4pp,
    σ 1.6pp. Pin at ≥4pp has ~2σ margin from the worst seed.
    """
    m = _run_empirical(seed)
    lift = m["precs"][0.8] - m["primary_prec"]
    assert lift >= 0.04, (
        f"seed={seed}: prec@0.8={m['precs'][0.8]:.4f}, primary="
        f"{m['primary_prec']:.4f}, lift={lift:.4f} < 0.04. AFML §3.5 "
        "precision-lift claim violated on the calibration fixture."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_pin18_volume_filter_at_tau08(seed: int) -> None:
    """Pin 18: at τ=0.8, meta takes ≤ 70% of test rows (filter is
    restrictive). Spike: max 0.608, σ 0.064. 1.4σ margin — the
    tightest pin. If it flaps on sklearn/numpy drift, relax to 0.75
    per plan §"CI-flap watch"; do NOT weaken pins 17 or 19.
    """
    m = _run_empirical(seed)
    vol = m["vols"][0.8]
    assert vol <= 0.70, (
        f"seed={seed}: vol@0.8={vol:.4f} > 0.70. Meta filter is not "
        "restrictive enough — if this is a systematic drift, consult "
        "plan §CI-flap watch before relaxing."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_pin19_monotone_precision_in_tau(seed: int) -> None:
    """Pin 19: ``prec@0.3 ≤ prec@0.5 ≤ prec@0.7 ≤ prec@0.8`` (weakly
    monotone; strict on 5/5 seeds in the spike). Pins the structural
    "filter → higher precision" claim central to AFML §3.5 meta-
    labeling.
    """
    m = _run_empirical(seed)
    prec_seq = [m["precs"][tau] for tau in [0.3, 0.5, 0.6, 0.7, 0.8]]
    for i in range(len(prec_seq) - 1):
        assert prec_seq[i + 1] >= prec_seq[i] - 1e-12, (
            f"seed={seed}: precision not monotone in τ; sequence "
            f"{prec_seq} — pin 19 violated at τ={[0.3, 0.5, 0.6, 0.7, 0.8][i]} "
            f"→ τ={[0.3, 0.5, 0.6, 0.7, 0.8][i + 1]}."
        )
    # Strict separation between extremes (from spike).
    assert prec_seq[-1] > prec_seq[0] + 0.02, (
        f"seed={seed}: prec@0.8 - prec@0.3 = "
        f"{prec_seq[-1] - prec_seq[0]:.4f} < 0.02; filter has no material "
        "precision effect."
    )
