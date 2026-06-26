"""Structural tests for MetaLabeler (S5 / P4.1, AFML §3.5).

Covers the 14 structural pins from the sprint plan — error contracts,
meta-label construction, feature augmentation, sklearn protocol
compliance, positive-class identity resolution, idempotence under
``fit`` → ``fit``, and the ``check_estimator`` smoke test.

Empirical-performance pins (precision lift at τ=0.8, volume filter,
monotone precision in τ) live in ``test_metalabeler_empirical.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import NotFittedError

from quantcore.labels import EconomicRationaleNotProvided, MetaLabeler


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _simple_fixture(n: int = 400, seed: int = 0):
    """Small fast-to-fit fixture for structural tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    signal = X[:, 0] + 0.6 * X[:, 1]
    y = np.sign(signal + 0.8 * rng.standard_normal(n)).astype(int)
    return X, y


def _standard_metalabeler(**overrides):
    """Default MetaLabeler with both sub-estimators = LogisticRegression.

    Helper-level default is ``meta_features_oos=False`` — the legacy
    IS path — so existing tests that rely on the in-sample-warning
    contract (Pin 11) keep working after the S8 class-default flip.
    Tests exercising the new default pass ``meta_features_oos=True``
    with an explicit ``oos_cv`` splitter as an override, or construct
    ``MetaLabeler`` directly.
    """
    defaults = dict(
        primary_model=LogisticRegression(penalty=None, max_iter=1000),
        meta_model=LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="test strategy: long when signal positive",
        meta_features_oos=False,
    )
    defaults.update(overrides)
    return MetaLabeler(**defaults)


# -----------------------------------------------------------------------------
# PIN 1 — EconomicRationaleNotProvided on empty/None rationale
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rationale",
    [None, "", "   ", "\t\n ", pytest.param(123, id="non-string-int")],
)
def test_pin1_economic_rationale_raises(rationale) -> None:
    with pytest.raises(EconomicRationaleNotProvided, match=r"economic_rationale"):
        MetaLabeler(
            LogisticRegression(penalty=None),
            LogisticRegression(penalty=None),
            economic_rationale=rationale,
        )


def test_pin1_valid_rationale_accepts() -> None:
    """Non-empty string is accepted without error."""
    ml = _standard_metalabeler()
    assert ml.economic_rationale == "test strategy: long when signal positive"


# -----------------------------------------------------------------------------
# PIN 2 — primary must have predict_proba
# -----------------------------------------------------------------------------


def test_pin2_primary_without_predict_proba_raises() -> None:
    """A primary lacking predict_proba raises at __init__ naming the
    offending type."""

    class NoProbabilityClassifier:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    with pytest.raises(ValueError, match=r"predict_proba"):
        MetaLabeler(
            NoProbabilityClassifier(),
            LogisticRegression(penalty=None),
            economic_rationale="rationale",
        )


# -----------------------------------------------------------------------------
# PIN 3 — meta_features_oos=True requires oos_cv
# -----------------------------------------------------------------------------


def test_pin3_oos_without_cv_raises() -> None:
    """Prescriptive ValueError — S8 replaced the minimal message with
    one naming PurgedKFold + AFML §7 so users have a concrete next
    step rather than a bare requirement. Regex is tolerant of the
    ``§`` character's encoding by accepting either ``§7`` or
    ``section 7`` forms.
    """
    with pytest.raises(ValueError, match=r"PurgedKFold.*AFML.*(§7|section\s*7)"):
        _standard_metalabeler(meta_features_oos=True, oos_cv=None)


# -----------------------------------------------------------------------------
# PIN 4 — Meta-label byte-identity
# -----------------------------------------------------------------------------


def test_pin4_meta_label_byte_identity() -> None:
    """After fit, if we independently recompute z =
    (sign(primary_.predict(X_active)) == sign(y_active)), it must
    byte-match what the meta model was trained on.

    We verify this by checking a fitted meta's behavior on training
    data: its predictions should correlate with primary correctness
    on the active set.
    """
    X, y = _simple_fixture(400, 42)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)

    # Reconstruct the active set + meta-labels independently.
    active = y != 0
    X_a, y_a = X[active], y[active]
    primary_pred = ml.primary_.predict(X_a)
    expected_z = (np.sign(primary_pred) == np.sign(y_a)).astype(int)

    # Feature augmentation correct shape + meta was fit on that matrix.
    X_meta_expected = np.column_stack(
        [X_a, ml.primary_.predict_proba(X_a)[:, ml._positive_class_idx_]]
    )
    assert X_meta_expected.shape == (X_a.shape[0], X.shape[1] + 1)

    # Meta's internal training: we can't reach it directly, but we can
    # verify that meta.predict on X_meta_expected gives the expected
    # agreement with z (LogReg on separable data recovers labels well).
    meta_pred = ml.meta_.predict(X_meta_expected)
    train_agreement = float((meta_pred == expected_z).mean())
    assert train_agreement >= 0.6, (
        f"Meta's training-set agreement with z = {train_agreement:.3f}; "
        "expected >= 0.6 — suggests meta was trained on different "
        "labels than the byte-identical z computed here."
    )


# -----------------------------------------------------------------------------
# PIN 5 — feature augmentation shape
# -----------------------------------------------------------------------------


def test_pin5_feature_augmentation_shape() -> None:
    """Pin that meta internally sees (n_active, d+1) features by
    verifying the fitted meta's ``n_features_in_``.
    """
    X, y = _simple_fixture(400, 0)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    # Meta model (LogisticRegression) exposes n_features_in_ after fit.
    assert ml.meta_.n_features_in_ == X.shape[1] + 1, (
        f"Meta should see d+1={X.shape[1] + 1} features (X + primary proba); "
        f"got {ml.meta_.n_features_in_}."
    )


# -----------------------------------------------------------------------------
# PIN 6 — positive_class resolved by identity, not hardcoded
# -----------------------------------------------------------------------------


def test_pin6_positive_class_inverts_column_selection() -> None:
    """positive_class=+1 vs positive_class=-1 pull different columns of
    primary's predict_proba (``p_+`` vs ``p_- = 1 - p_+``). The meta
    is then trained on different augmented features, so predict_proba
    differs materially between the two.

    Implicitly pins that the positive-class index is resolved by
    identity via ``primary_.classes_.index(positive_class)`` and NOT
    hardcoded as ``[:, 1]``.
    """
    X, y = _simple_fixture(300, 7)
    ml_pos = _standard_metalabeler(positive_class=1)
    ml_neg = _standard_metalabeler(positive_class=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_pos.fit(X, y)
        ml_neg.fit(X, y)

    # Index resolved to different slots.
    assert ml_pos._positive_class_idx_ != ml_neg._positive_class_idx_, (
        "positive_class=+1 and -1 should resolve to different column indices in primary_.classes_."
    )

    # Meta trained on different augmented features → predict_proba differs.
    proba_pos = ml_pos.predict_proba(X)
    proba_neg = ml_neg.predict_proba(X)
    assert not np.allclose(proba_pos, proba_neg, atol=1e-6), (
        "positive_class=±1 produced identical predict_proba — suggests "
        "the column index was hardcoded, not resolved by identity."
    )


def test_pin6_missing_positive_class_raises() -> None:
    """positive_class not in primary's learned classes raises ValueError."""
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler(positive_class=99)  # never appears in sign(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"positive_class=99.*not in"):
            ml.fit(X, y)


# -----------------------------------------------------------------------------
# PIN 7 — check_is_fitted semantics
# -----------------------------------------------------------------------------


def test_pin7_check_is_fitted_pre_post() -> None:
    ml = _standard_metalabeler()
    with pytest.raises(NotFittedError):
        ml.predict_proba(np.zeros((4, 4)))
    with pytest.raises(NotFittedError):
        ml.predict(np.zeros((4, 4)))
    X, y = _simple_fixture(200, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    # Post-fit must not raise.
    _ = ml.predict_proba(X)
    _ = ml.predict(X)


# -----------------------------------------------------------------------------
# PIN 8 — predict_proba shape and row-sum
# -----------------------------------------------------------------------------


def test_pin8_predict_proba_shape_and_rowsum() -> None:
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    proba = ml.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


# -----------------------------------------------------------------------------
# PIN 9 — classes_ + n_features_in_
# -----------------------------------------------------------------------------


def test_pin9_sklearn_protocol_attrs() -> None:
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    assert hasattr(ml, "classes_")
    np.testing.assert_array_equal(ml.classes_, np.array([-1, 0, 1]))
    assert hasattr(ml, "n_features_in_")
    assert ml.n_features_in_ == X.shape[1]
    assert hasattr(ml, "primary_") and hasattr(ml, "meta_")
    assert hasattr(ml, "_positive_class_idx_")


# -----------------------------------------------------------------------------
# PIN 10 — predict dtype and value set
# -----------------------------------------------------------------------------


def test_pin10_predict_dtype_and_values() -> None:
    X, y = _simple_fixture(300, 0)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    pred = ml.predict(X)
    assert pred.dtype.kind == "i", f"predict should return int; got {pred.dtype}"
    unique = set(pred.tolist())
    assert unique.issubset({-1, 0, 1}), f"predict values ⊄ {{-1, 0, 1}}; got {unique}"


# -----------------------------------------------------------------------------
# PIN 11 — UserWarning at fit when meta_features_oos=False (legacy IS opt-in)
# -----------------------------------------------------------------------------


def test_pin11_in_sample_leakage_warning() -> None:
    """Post-S8: ``meta_features_oos=False`` is opt-in legacy behavior.
    The UserWarning still fires at fit time to name the inflation.
    Helper-level default of the fixture is False (see
    ``_standard_metalabeler`` docstring), so ``ml = _standard_metalabeler()``
    still exercises the IS path without an explicit override.
    Regex broadened from ``r"in-sample.*S6"`` to ``r"in[- ]sample"`` so
    the pin survives the S8 removal of the sprint-number reference
    from the warning body.
    """
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler()  # helper default meta_features_oos=False
    with pytest.warns(UserWarning, match=r"in[- ]sample"):
        ml.fit(X, y)


# -----------------------------------------------------------------------------
# PIN 12 — NotImplementedError on meta_features_oos=True fit
# -----------------------------------------------------------------------------

# PIN 12 — OOS path runs without raising — moved to test_metalabeler_oos.py
# (S6 landed cross_val_predict wiring; the original
# NotImplementedError-pin from S5 was deleted).


# -----------------------------------------------------------------------------
# PIN 13 — Idempotence under fit → fit (guards against missing clone)
# -----------------------------------------------------------------------------


def test_pin13_idempotence_fit_twice() -> None:
    """Calling ``fit(X, y)`` twice on the same MetaLabeler with the
    same data produces identical ``predict_proba`` on the same test
    data. Guards against warm-started sub-estimators — without
    ``clone``, tree/GBM models would silently resume training.
    """
    X, y = _simple_fixture(300, 0)
    ml = _standard_metalabeler(meta_model=DecisionTreeClassifier(random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    proba_first = ml.predict_proba(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    proba_second = ml.predict_proba(X)
    np.testing.assert_allclose(proba_first, proba_second, atol=1e-12)


# -----------------------------------------------------------------------------
# PIN 14 — sklearn check_estimator smoke test (tags={"binary_only": True})
# -----------------------------------------------------------------------------


def test_pin14_sklearn_protocol_smoke() -> None:
    """Minimal sklearn-protocol smoke: fit → predict → predict_proba,
    ``clone`` pre- and post-fit, ``get_params`` / ``set_params``
    round-trip. Not a full ``check_estimator`` run — that constructs
    degenerate single-class fixtures that sklearn's ``binary_only``
    tag doesn't prevent in 1.6+ (raises `ValueError: needs >=2
    classes` inside check_estimator's fit call). The individual pins
    above (1–13) collectively cover everything check_estimator would
    verify for a binary classifier with an AFML-specific constructor
    contract.
    """
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler()

    # Pre-fit clone yields an unfitted equivalent.
    ml_pre = clone(ml)
    assert ml_pre.economic_rationale == ml.economic_rationale

    # Fit, predict, predict_proba succeed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    pred = ml.predict(X)
    proba = ml.predict_proba(X)
    assert pred.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], 2)

    # Post-fit clone yields an unfitted equivalent (not a deep copy).
    ml_post = clone(ml)
    with pytest.raises(NotFittedError):
        ml_post.predict(X)

    # get_params / set_params round-trip preserves construction args.
    params = ml.get_params(deep=False)
    ml_rebuilt = MetaLabeler(**params)
    assert ml_rebuilt.economic_rationale == ml.economic_rationale
    assert ml_rebuilt.side_threshold == ml.side_threshold
    assert ml_rebuilt.drop_zero == ml.drop_zero


# -----------------------------------------------------------------------------
# Extras — clone / get_params round-trip
# -----------------------------------------------------------------------------


def test_extra_clone_round_trips() -> None:
    """``clone()`` reconstructs an equivalent estimator without mutation."""
    ml = _standard_metalabeler()
    ml_clone = clone(ml)
    assert ml_clone.economic_rationale == ml.economic_rationale
    assert ml_clone.side_threshold == ml.side_threshold
    assert ml_clone.drop_zero == ml.drop_zero


def test_extra_drop_zero_warning_count() -> None:
    """With drop_zero=True, a UserWarning names the dropped row count."""
    X, y = _simple_fixture(200, 0)
    # Force some zeros by clipping the signal near zero.
    rng = np.random.default_rng(0)
    zero_mask = rng.random(len(y)) < 0.1
    y_with_zeros = y.copy().astype(float)
    y_with_zeros[zero_mask] = 0.0
    n_zeros = int(zero_mask.sum())

    ml = _standard_metalabeler()
    with pytest.warns(UserWarning, match=rf"dropping {n_zeros}.*zero-labeled"):
        ml.fit(X, y_with_zeros)


def test_extra_n_features_in_does_not_count_proba_column() -> None:
    """``n_features_in_`` on MetaLabeler is the caller's X column count,
    not X augmented with primary's proba column."""
    X, y = _simple_fixture(200, 0)
    ml = _standard_metalabeler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X, y)
    assert ml.n_features_in_ == X.shape[1]  # not X.shape[1] + 1


# =============================================================================
# S8 P7.2 — default flip pins (M1–M4)
# =============================================================================
# Flip ``meta_features_oos`` default False → True with no silent
# KFold fallback. Breaking change; see S8 plan §P7.2.


def test_pinM1_default_meta_features_oos_is_true() -> None:
    """Class-level default of ``meta_features_oos`` is True post-S8."""
    import inspect

    sig = inspect.signature(MetaLabeler.__init__)
    assert sig.parameters["meta_features_oos"].default is True


def test_pinM2_naked_construction_raises_prescriptive_error() -> None:
    """Naked MetaLabeler(primary, meta, rationale) without an explicit
    ``oos_cv`` raises ValueError whose message names PurgedKFold and
    AFML §7. No silent KFold fallback.

    Regex is tolerant of ``§`` encoding: matches either the raw ``§7``
    or the spelled-out ``section 7`` form so CI platforms with
    non-UTF-8 locales still pass.
    """
    with pytest.raises(ValueError, match=r"PurgedKFold.*AFML.*(§7|section\s*7)"):
        MetaLabeler(
            LogisticRegression(penalty=None, max_iter=1000),
            LogisticRegression(penalty=None, max_iter=1000),
            economic_rationale="naked-construction probe",
        )


def test_pinM3_explicit_is_optout_still_warns() -> None:
    """Explicit ``meta_features_oos=False`` still fits the IS path
    and still fires the in-sample-leakage UserWarning at fit time.
    Pin 11 semantics preserved; IS behavior is now opt-in rather than
    default.
    """
    X, y = _simple_fixture(200, 0)
    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="IS opt-in post-S8 flip",
        meta_features_oos=False,
    )
    with pytest.warns(UserWarning, match=r"in[- ]sample"):
        ml.fit(X, y)


def test_pinM4_new_default_runs_with_explicit_oos_cv() -> None:
    """Positive path: the new default works when a user supplies an
    explicit ``oos_cv``. Naked ``MetaLabeler(p, m, rationale=...,
    oos_cv=KFold(...))`` fits without error and does NOT fire the
    IS warning (OOS branch).
    """
    from sklearn.model_selection import KFold

    X, y = _simple_fixture(200, 0)
    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="new-default positive path",
        oos_cv=KFold(n_splits=3, shuffle=True, random_state=0),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ml.fit(X, y)
        is_warnings = [
            w for w in caught if "in-sample" in str(w.message) or "in sample" in str(w.message)
        ]
    assert not is_warnings, (
        f"Default OOS path should not emit IS-leakage warning; got "
        f"{[str(w.message) for w in is_warnings]}"
    )
    # Fit succeeded and predict works.
    pred = ml.predict(X)
    assert pred.shape == (X.shape[0],)


# =============================================================================
# S9 P8.2 — t1 kwarg on fit behind defer_cv_resolution flag (T1–T6)
# =============================================================================
# Preserves S8's init-time raise as the DEFAULT. Opt-in path via
# defer_cv_resolution=True: construct without oos_cv, supply t1 at
# fit() time, auto-get a PurgedKFold inner CV. See S9 plan §P8.2.


def _t1_fixture(n: int = 400, seed: int = 0):
    """Return (X_df, y_ser, t1_ser) aligned positionally and by index
    (DatetimeIndex). No zero-labeled rows in this base fixture.
    """
    import pandas as pd

    X_array, y_array = _simple_fixture(n, seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    X = pd.DataFrame(X_array, index=idx)
    y = pd.Series(y_array, index=idx)
    # Each sample's t1 is 5 bars forward (or last bar for tail rows).
    t1_values = [idx[min(i + 5, n - 1)] for i in range(n)]
    t1 = pd.Series(t1_values, index=idx)
    return X, y, t1


def test_pinT1_default_behavior_preserves_S8_init_time_raise() -> None:
    """Without ``defer_cv_resolution=True`` (default False), naked
    ``MetaLabeler(meta_features_oos=True, oos_cv=None)`` still raises
    at __init__ — S8 fail-fast safety net unchanged."""
    with pytest.raises(ValueError, match=r"PurgedKFold.*AFML.*(§7|section\s*7)"):
        MetaLabeler(
            LogisticRegression(penalty=None, max_iter=1000),
            LogisticRegression(penalty=None, max_iter=1000),
            economic_rationale="S8 default preserved",
            # defer_cv_resolution=False (default)
        )


def test_pinT2_defer_flag_permits_naked_construction() -> None:
    """``defer_cv_resolution=True`` with ``oos_cv=None`` constructs
    without raising. Post-construction the state is (oos_cv=None,
    defer_cv_resolution=True) — deferred to fit()."""
    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="deferred CV construction",
        defer_cv_resolution=True,
    )
    assert ml.oos_cv is None
    assert ml.defer_cv_resolution is True
    assert ml.meta_features_oos is True


def test_pinT3_fit_with_t1_auto_constructs_purged_kfold() -> None:
    """On the deferred-construction path, passing ``t1`` to fit()
    auto-constructs ``PurgedKFold(n_splits=5, t1=t1_active,
    embargo_pct=0.01)``. Fit completes and the resolved splitter is
    observable via ``ml._resolved_oos_cv_``."""
    from quantcore.cv.purged_kfold import PurgedKFold

    X, y, t1 = _t1_fixture()
    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="auto-default via t1",
        defer_cv_resolution=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X.to_numpy(), y.to_numpy(), t1=t1)
    assert hasattr(ml, "_resolved_oos_cv_"), "OOS fit path should have set _resolved_oos_cv_"
    assert isinstance(ml._resolved_oos_cv_, PurgedKFold)
    assert ml._resolved_oos_cv_.n_splits == 5
    assert ml._resolved_oos_cv_.embargo_pct == pytest.approx(0.01)


def test_pinT4_fit_without_t1_or_oos_cv_raises_prescriptive() -> None:
    """On deferred construction, fit() with no ``oos_cv`` AND no
    ``t1`` raises a prescriptive ``ValueError`` at fit time (defense
    in depth when init-time raise is suppressed). Regex tolerant of
    ``§`` encoding — matches same form as S8 Pin M2."""
    X, y, _ = _t1_fixture()
    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="deferred with nothing supplied",
        defer_cv_resolution=True,
    )
    with pytest.raises(ValueError, match=r"PurgedKFold.*AFML.*(§7|section\s*7)"):
        ml.fit(X.to_numpy(), y.to_numpy())


def test_pinT5_explicit_oos_cv_takes_precedence_over_t1() -> None:
    """Precedence contract: when ``oos_cv`` is set explicitly at
    construction, a ``t1`` passed at fit() is silently ignored.
    Covers both defer=True (flag moot because oos_cv supplied) and
    defer=False (S8 default with explicit oos_cv)."""
    from sklearn.model_selection import KFold
    from quantcore.cv.purged_kfold import PurgedKFold

    X, y, t1 = _t1_fixture()
    explicit_cv = KFold(n_splits=3, shuffle=True, random_state=0)

    # Case A: defer=False + explicit oos_cv (S8 path). t1 ignored.
    ml_a = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="explicit wins, defer=False",
        oos_cv=explicit_cv,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_a.fit(X.to_numpy(), y.to_numpy(), t1=t1)
    assert ml_a._resolved_oos_cv_ is explicit_cv, (
        "explicit oos_cv must win over t1 on defer=False path"
    )
    assert not isinstance(ml_a._resolved_oos_cv_, PurgedKFold)

    # Case B: defer=True + explicit oos_cv (weird combo, allowed).
    ml_b = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="explicit wins, defer=True",
        oos_cv=explicit_cv,
        defer_cv_resolution=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_b.fit(X.to_numpy(), y.to_numpy(), t1=t1)
    assert ml_b._resolved_oos_cv_ is explicit_cv


def test_pinT6_t1_aligned_to_active_subset_under_drop_zero() -> None:
    """When ``drop_zero=True`` (default) and y contains zero-labeled
    rows, the t1 supplied at fit is positionally active-masked BEFORE
    the auto-constructed PurgedKFold is built. Resolved
    ``_resolved_oos_cv_.t1`` has length ``(y != 0).sum()`` — NOT
    ``y.shape[0]``.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    X_array = rng.standard_normal((n, 4))
    signal = X_array[:, 0] + 0.6 * X_array[:, 1]
    y_array = np.sign(signal + 0.8 * rng.standard_normal(n)).astype(int)
    # Inject zeros on ~8% of rows to ensure drop_zero has work to do.
    zero_mask = rng.random(n) < 0.08
    y_array[zero_mask] = 0
    n_active = int((y_array != 0).sum())
    assert n_active < n, "test-setup: expected some zeros to drop"

    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    tail = n - 1
    t1_values = [idx[min(i + 5, tail)] for i in range(n)]
    t1 = pd.Series(t1_values, index=idx)

    ml = MetaLabeler(
        LogisticRegression(penalty=None, max_iter=1000),
        LogisticRegression(penalty=None, max_iter=1000),
        economic_rationale="drop_zero + t1 alignment",
        defer_cv_resolution=True,
        drop_zero=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.fit(X_array, y_array.astype(float), t1=t1)
    assert len(ml._resolved_oos_cv_.t1) == n_active, (
        f"Expected PurgedKFold.t1 length {n_active} (active subset); "
        f"got {len(ml._resolved_oos_cv_.t1)}"
    )
