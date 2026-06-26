"""MetaLabeler OOS meta-label leak regression (closes F1, S38).

Pre-S38, ``MetaLabeler.fit`` on the ``meta_features_oos=True`` path computed
the OOS primary PROBABILITY feature via ``cross_val_predict`` (correct) but
computed the meta-LABEL ``z`` from the IN-SAMPLE primary
(``self.primary_.predict(X_a)``, primary fit on the full active set). The
in-sample primary is overconfident on its own training rows, so the meta-
label base rate ``mean(z)`` was inflated and the meta-classifier learned an
inflated P(primary correct) — a real label leak (AFML §3.5 / §7.4.1).

S38 derives ``z`` from the SAME OOS folds (argmax of the OOS class
probabilities). These tests pin that the OOS-path meta-label base rate is now
materially BELOW the in-sample base rate (the leak signature), and that the
in-sample path is unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from quantcore.labels import MetaLabeler

SEEDS = [0, 1, 2, 42, 20260423]

# A HIGH-VARIANCE primary (RandomForest) makes the in-sample-z leak large and
# the regression sharp: the forest near-memorizes its training rows, so the
# in-sample base rate ≈ 0.95+ while the honest OOS base rate ≈ true accuracy
# (~0.7). For a low-variance primary (e.g. unpenalized LogisticRegression) the
# same leak still exists but is only ~0.01 — which is exactly why threading the
# OOS derivation matters most for tree/ensemble primaries.


def _heteroskedastic_fixture(n: int, seed: int):
    """Same fixture family as test_metalabeler_oos.py (y in {-1, +1})."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    signal = X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 2]
    noise_amp = 0.5 + 1.5 * np.abs(X[:, 3])
    y = np.sign(signal + noise_amp * rng.standard_normal(n)).astype(int)
    return X, y


def _oos_metalabeler(seed: int) -> MetaLabeler:
    return MetaLabeler(
        primary_model=RandomForestClassifier(n_estimators=60, random_state=seed),
        meta_model=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        economic_rationale="OOS z-leak regression",
        meta_features_oos=True,
        oos_cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    )


def _in_sample_base_rate(ml: MetaLabeler, X, y) -> float:
    """The base rate the LEAKY (in-sample) path would produce: primary fit on
    the full active set, predicted in-sample, compared to realized sign."""
    active = y != 0
    Xa, ya = X[active], y[active]
    is_pred = ml.primary_.predict(Xa)
    return float((np.sign(is_pred) == np.sign(ya)).mean())


@pytest.mark.parametrize("seed", SEEDS)
def test_oos_meta_label_base_rate_below_in_sample(seed: int) -> None:
    """The OOS meta-label base rate (mean z used to fit meta) must sit
    materially below the in-sample base rate. Pre-S38 they were EQUAL
    (z came from the in-sample primary on both paths)."""
    X, y = _heteroskedastic_fixture(1200, seed)
    ml = _oos_metalabeler(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)

    oos_rate = ml._meta_label_base_rate_
    is_rate = _in_sample_base_rate(ml, X, y)

    assert oos_rate < is_rate - 0.10, (
        f"seed={seed}: OOS meta-label base rate {oos_rate:.4f} is not "
        f"materially below the in-sample base rate {is_rate:.4f} — the "
        "in-sample-z leak (F1) has regressed (the OOS path is using the "
        "in-sample primary prediction for z)."
    )
    # Sanity: the OOS base rate is still an informative signal (> coin flip).
    assert oos_rate > 0.5


@pytest.mark.parametrize("seed", SEEDS)
def test_in_sample_path_base_rate_matches_in_sample_predict(seed: int) -> None:
    """On the IS path (meta_features_oos=False) the meta-label base rate
    EQUALS the in-sample computation — the IS path is unchanged by S38.

    Uses a low-variance LogisticRegression primary so the in-sample z keeps
    both classes (a memorizing forest would make z all-ones and the meta
    un-fittable — itself a symptom of why the leak is harmful)."""
    X, y = _heteroskedastic_fixture(1200, seed)
    ml = MetaLabeler(
        primary_model=LogisticRegression(penalty=None, max_iter=2000, random_state=seed),
        meta_model=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        economic_rationale="IS path base-rate check",
        meta_features_oos=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)

    is_rate = _in_sample_base_rate(ml, X, y)
    assert ml._meta_label_base_rate_ == pytest.approx(is_rate, abs=1e-12)
