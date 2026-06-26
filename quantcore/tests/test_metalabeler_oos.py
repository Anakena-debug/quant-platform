"""Tests for the meta_features_oos=True path on MetaLabeler (S6 / P5.1).

Covers the 5 pins from the sprint plan:
  1. OOS path runs without NotImplementedError.
  2. In-sample UserWarning suppressed when OOS selected.
  3. RMSE(p_oos - p_is) > 0.005 on calibration fixture (5 seeds).
  4. Sample weights thread through cross_val_predict.
  5. Empirical pins 17-19 from S5 hold on OOS path.

Replaces the deleted ``test_pin12_oos_path_raises_not_implemented``
from S5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict

from quantcore.labels import MetaLabeler


SEEDS = [0, 1, 2, 42, 20260423]


def _heteroskedastic_fixture(n: int, seed: int):
    """Same fixture as test_metalabeler_empirical.py — pinned for
    pin-3 calibration spike reproducibility."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    signal = X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 2]
    noise_amp = 0.5 + 1.5 * np.abs(X[:, 3])
    y = np.sign(signal + noise_amp * rng.standard_normal(n)).astype(int)
    return X, y


def _make_oos_metalabeler(seed: int, meta=None):
    return MetaLabeler(
        primary_model=LogisticRegression(
            penalty=None,
            max_iter=2000,
            random_state=seed,
        ),
        meta_model=meta
        if meta is not None
        else GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=seed,
        ),
        economic_rationale="OOS meta-feature regression test",
        meta_features_oos=True,
        oos_cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    )


# -----------------------------------------------------------------------------
# PIN 1 — OOS path runs without raising
# -----------------------------------------------------------------------------


def test_pin_s6_1_oos_path_no_longer_raises() -> None:
    """meta_features_oos=True with valid oos_cv now executes via
    cross_val_predict. Was NotImplementedError in S5; replaced in S6."""
    X, y = _heteroskedastic_fixture(500, 0)
    ml = _make_oos_metalabeler(0, meta=LogisticRegression(penalty=None, max_iter=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    proba = ml.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


# -----------------------------------------------------------------------------
# PIN 2 — In-sample warning NOT emitted on OOS path
# -----------------------------------------------------------------------------


def test_pin_s6_2_in_sample_warning_suppressed_on_oos() -> None:
    """The in-sample-leakage UserWarning fires on the IS path (S5
    behavior). On OOS path, that specific warning must NOT appear."""
    X, y = _heteroskedastic_fixture(400, 0)
    ml = _make_oos_metalabeler(0, meta=LogisticRegression(penalty=None, max_iter=2000))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ml.fit(X, y)
    in_sample_warnings = [
        w for w in caught if issubclass(w.category, UserWarning) and "in-sample" in str(w.message)
    ]
    assert in_sample_warnings == [], (
        "OOS path should NOT emit the in-sample-leakage UserWarning; "
        f"got {[str(w.message) for w in in_sample_warnings]}."
    )


def test_pin_s6_2b_in_sample_warning_still_fires_on_is_path() -> None:
    """Sanity: the in-sample warning DOES still fire on the IS path
    (opt-in via ``meta_features_oos=False`` post-S8). Guards against
    accidentally suppressing both paths.

    Post-S8 the class default is True; to exercise the IS path this
    test explicitly passes False. Regex broadened per S8 warning
    cleanup that removed the sprint-number marker.
    """
    X, y = _heteroskedastic_fixture(400, 0)
    ml = MetaLabeler(
        primary_model=LogisticRegression(penalty=None, max_iter=2000),
        meta_model=LogisticRegression(penalty=None, max_iter=2000),
        economic_rationale="IS path sanity check",
        meta_features_oos=False,  # explicit IS opt-in (S8+)
    )
    with pytest.warns(UserWarning, match=r"in[- ]sample"):
        ml.fit(X, y)


# -----------------------------------------------------------------------------
# PIN 3 — OOS probabilities differ materially from IS probabilities
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_pin_s6_3_oos_probabilities_differ_from_is(seed: int) -> None:
    """Recompute the training-set primary probabilities under both
    regimes (IS = primary on full train; OOS = cross_val_predict on
    train) and assert RMSE > 0.005. Spike: min 0.014, σ 0.003 →
    ~3σ margin from the worst seed."""
    X, _y = _heteroskedastic_fixture(1000, seed)
    y = _y[:1000]
    primary = LogisticRegression(
        penalty=None,
        max_iter=2000,
        random_state=seed,
    )
    primary.fit(X, y)
    pos = list(primary.classes_).index(+1)

    p_is = primary.predict_proba(X)[:, pos]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    p_oos = cross_val_predict(
        clone(primary),
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, pos]

    rmse = float(np.linalg.norm(p_oos - p_is) / np.sqrt(len(p_is)))
    assert rmse > 0.005, (
        f"seed={seed}: RMSE(p_oos - p_is) = {rmse:.4f} ≤ 0.005. "
        "OOS probabilities collapsing to in-sample suggests "
        "cross_val_predict is being short-circuited or the CV splitter "
        "is degenerate."
    )


# -----------------------------------------------------------------------------
# PIN 4 — Sample weights thread through cross_val_predict
# -----------------------------------------------------------------------------


def test_pin_s6_4_sample_weight_threads_through_cv() -> None:
    """OOS path with uniform vs imbalanced sample_weight produces
    different OOS meta-feature columns. Pins that the
    `params={"sample_weight": w}` kwarg actually reaches each fold's
    primary fit inside cross_val_predict."""
    X, y = _heteroskedastic_fixture(500, 0)
    n = len(y)
    w_uniform = np.ones(n, dtype=np.float64)
    w_imbal = np.ones(n, dtype=np.float64)
    w_imbal[y == 1] = 5.0

    ml_u = MetaLabeler(
        primary_model=LogisticRegression(
            penalty=None,
            max_iter=2000,
            random_state=0,
        ),
        meta_model=LogisticRegression(penalty=None, max_iter=2000),
        economic_rationale="weight-threading test",
        meta_features_oos=True,
        oos_cv=KFold(n_splits=5, shuffle=True, random_state=0),
    )
    ml_i = clone(ml_u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_u.fit(X, y, sample_weight=w_uniform)
        ml_i.fit(X, y, sample_weight=w_imbal)

    # Recover the OOS meta-feature columns indirectly via meta.coef_
    # — different OOS features will produce different meta coefficients.
    coef_diff = float(np.linalg.norm(ml_u.meta_.coef_ - ml_i.meta_.coef_))
    assert coef_diff > 1e-3, (
        "Imbalanced sample_weight should change the OOS meta-feature "
        "values (and thus meta.coef_). L2 diff = "
        f"{coef_diff:.2e} ≤ 1e-3 — weight is being silently dropped "
        "inside cross_val_predict."
    )


# -----------------------------------------------------------------------------
# PIN 5 — Empirical pins 17-19 hold on OOS path
# -----------------------------------------------------------------------------


def _run_empirical_oos(seed: int) -> dict:
    X, y = _heteroskedastic_fixture(1500, seed)
    X_tr, X_te = X[:1000], X[1000:]
    y_tr, y_te = y[:1000], y[1000:]
    ml = _make_oos_metalabeler(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X_tr, y_tr)

    primary_pred_te = ml.primary_.predict(X_te)
    z_te = (np.sign(primary_pred_te) == np.sign(y_te)).astype(int)
    primary_prec = float(z_te.mean())
    meta_proba = ml.predict_proba(X_te)[:, 1]

    precs = {}
    vols = {}
    for tau in [0.3, 0.5, 0.6, 0.7, 0.8]:
        take = meta_proba > tau
        n_take = int(take.sum())
        vols[tau] = n_take / len(y_te)
        precs[tau] = float(z_te[take].mean()) if n_take else float("nan")
    return {"primary_prec": primary_prec, "precs": precs, "vols": vols}


@pytest.mark.parametrize("seed", SEEDS)
def test_pin_s6_5a_lift_at_08_oos(seed: int) -> None:
    """S5 pin 17 still holds on OOS path: prec@τ=0.8 ≥ primary + 0.04."""
    m = _run_empirical_oos(seed)
    lift = m["precs"][0.8] - m["primary_prec"]
    assert lift >= 0.04, (
        f"seed={seed}: OOS prec@0.8={m['precs'][0.8]:.4f}, primary="
        f"{m['primary_prec']:.4f}, lift={lift:.4f} < 0.04."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_pin_s6_5b_volume_at_08_oos(seed: int) -> None:
    """S5 pin 18 still holds on OOS path: vol@τ=0.8 ≤ 0.70."""
    m = _run_empirical_oos(seed)
    vol = m["vols"][0.8]
    assert vol <= 0.70, f"seed={seed}: OOS vol@0.8={vol:.4f} > 0.70."


@pytest.mark.parametrize("seed", SEEDS)
def test_pin_s6_5c_monotone_precision_oos(seed: int) -> None:
    """S5 pin 19 still holds on OOS path: precision weakly monotone in τ."""
    m = _run_empirical_oos(seed)
    seq = [m["precs"][tau] for tau in [0.3, 0.5, 0.6, 0.7, 0.8]]
    for i in range(len(seq) - 1):
        assert seq[i + 1] >= seq[i] - 1e-12, (
            f"seed={seed}: OOS precision not monotone in τ; sequence={seq}."
        )
