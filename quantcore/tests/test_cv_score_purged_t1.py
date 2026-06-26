"""S10 P9.1 — ``cv_score_purged`` propagates ``t1`` to ``estimator.fit``.

10 tests on the additive propagation path.

  Pin 1    · t1 propagates when fit signature accepts explicit kwarg
  Pin 1b   · sample_weight × t1 simultaneous propagation (same slice)
  Pin 2    · t1 silently skipped when fit signature lacks it
  Pin 3    · **kwargs-accepting fit signature receives t1
  Pin 4    · backward compat — closed-form estimator, rtol=1e-12
  Pin 5    · MetaLabeler(defer_cv_resolution=True) end-to-end, observer
  Pin 6    · S8 composition pin — MANUAL, re-run via test_tml_composition
  Pin 7    · Pipeline + t1 raises (version-robust, no message coupling)

Plus three robustness tests added in the second code review:

  Probe   · inspect.signature ValueError → accepts_t1=False fallback
  Align   · positional-alignment — scrambled index still slices by pos
  Guard   · non-sliceable t1 raises TypeError at entry
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantcore.cv import purged_kfold as _pk_module
from quantcore.cv.purged_kfold import PurgedKFold, cv_score_purged
from quantcore.labels.meta import MetaLabeler


# =============================================================================
# Test fixture helpers.
# =============================================================================


def _make_fixture(n: int = 200, seed: int = 42):
    """Small synthetic fixture. Returns (X_df, y_ser, t1_ser, sw_ser)
    with matching positional / datetime index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    X_arr = rng.standard_normal((n, 3))
    y_arr = np.sign(X_arr[:, 0] + 0.3 * rng.standard_normal(n)).astype(int)
    X = pd.DataFrame(X_arr, index=idx, columns=["a", "b", "c"])
    y = pd.Series(y_arr, index=idx)
    tail = n - 1
    t1 = pd.Series([idx[min(i + 5, tail)] for i in range(n)], index=idx)
    sw = pd.Series(rng.uniform(0.5, 1.5, size=n), index=idx)
    return X, y, t1, sw


# Fake estimator classes for probing fit-kwargs dispatch.


class _FakeWithT1:
    """Fake estimator whose fit signature accepts an explicit ``t1`` kwarg."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def fit(self, X, y, sample_weight=None, t1=None):
        self.calls.append(
            {
                "X_shape": X.shape,
                "sample_weight": None if sample_weight is None else sample_weight.copy(),
                "t1": None if t1 is None else t1.copy(),
            }
        )
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return 0.0


class _FakeWithoutT1:
    """Fake estimator whose fit signature has neither t1 nor **kwargs."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def fit(self, X, y, sample_weight=None):
        self.calls.append({"X_shape": X.shape})
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return 0.0


class _FakeWithKwargs:
    """Fake estimator whose fit uses **kwargs (VAR_KEYWORD)."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def fit(self, X, y, **fit_params):
        self.calls.append({"fit_params": {k: v for k, v in fit_params.items()}})
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def score(self, X, y):
        return 0.0


# =============================================================================
# Pin 1 · t1 propagates when fit signature accepts it (explicit kwarg).
# =============================================================================


def test_pin1_t1_propagates_to_explicit_kwarg() -> None:
    X, y, t1, _ = _make_fixture()
    fake = _FakeWithT1()

    scores = cv_score_purged(fake, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    assert scores.shape == (5,)
    assert len(fake.calls) == 5, f"expected 5 fit calls (n_splits); got {len(fake.calls)}"

    # Reconstruct the expected per-fold slices from the same PurgedKFold
    # config cv_score_purged builds internally.
    outer = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)
    splits = list(outer.split(X))
    for call, (train_idx, _test_idx) in zip(fake.calls, splits):
        recorded_t1 = call["t1"]
        expected_t1 = t1.iloc[train_idx]
        assert recorded_t1 is not None, "t1 should have been propagated"
        pd.testing.assert_series_equal(recorded_t1, expected_t1, check_names=False)

    # Cross-fold slices MUST differ (catches a fit_kwargs-dict reuse bug).
    # Tightened per review S4: direct index comparison on all folds vs
    # fold 0, without a disjunction that could short-circuit on length.
    fold0_idx = fake.calls[0]["t1"].index
    differs_somewhere = any(not fold0_idx.equals(c["t1"].index) for c in fake.calls[1:])
    assert differs_somewhere, (
        "cross-fold t1 slices share the same index — likely a fit_kwargs-dict reuse bug"
    )


# =============================================================================
# Pin 1b · sample_weight × t1 simultaneous propagation.
# =============================================================================


def test_pin1b_sample_weight_and_t1_both_propagate_same_slice() -> None:
    X, y, t1, sw = _make_fixture()
    fake = _FakeWithT1()

    cv_score_purged(fake, X, y, sample_weight=sw, t1=t1, embargo_pct=0.05, n_splits=5)

    outer = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)
    splits = list(outer.split(X))
    for call, (train_idx, _test_idx) in zip(fake.calls, splits):
        assert call["sample_weight"] is not None
        assert call["t1"] is not None
        pd.testing.assert_series_equal(call["sample_weight"], sw.iloc[train_idx], check_names=False)
        pd.testing.assert_series_equal(call["t1"], t1.iloc[train_idx], check_names=False)
        # Index consistency between the two sliced kwargs.
        assert call["sample_weight"].index.equals(call["t1"].index)


# =============================================================================
# Pin 2 · t1 silently NOT propagated when fit signature lacks it.
# =============================================================================


def test_pin2_t1_not_propagated_when_fit_lacks_it() -> None:
    X, y, t1, _ = _make_fixture()
    fake = _FakeWithoutT1()

    # No raise — the accepts_t1 probe short-circuits. If cv_score_purged
    # were to pass t1 to this estimator's fit, it would raise TypeError
    # (unexpected kwarg). Successful completion IS the pin.
    scores = cv_score_purged(fake, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    assert scores.shape == (5,)
    assert len(fake.calls) == 5


# =============================================================================
# Pin 3 · **kwargs-accepting fit signature receives t1.
# =============================================================================


def test_pin3_kwargs_accepting_fit_receives_t1() -> None:
    X, y, t1, _ = _make_fixture()
    fake = _FakeWithKwargs()

    cv_score_purged(fake, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    outer = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)
    splits = list(outer.split(X))
    for call, (train_idx, _test_idx) in zip(fake.calls, splits):
        assert "t1" in call["fit_params"], (
            "**kwargs-accepting fit must receive t1 via VAR_KEYWORD branch"
        )
        pd.testing.assert_series_equal(
            call["fit_params"]["t1"], t1.iloc[train_idx], check_names=False
        )


# =============================================================================
# Pin 4 · Backward compat — closed-form Ridge, rtol=1e-12.
# =============================================================================
# Ridge(solver='cholesky') is closed-form (direct linear-system solve,
# no iterative BLAS accumulation). BLAS-stable across Accelerate /
# OpenBLAS / MKL at rtol=1e-12 empirically (reviewer P1.2 / B1).
# LogReg+LBFGS was rejected because iterative solvers drift in the
# 1e-6 to 1e-8 range across BLAS vendors — would flake on CI.
#
# Pin 4 asserts STRUCTURAL invariance: the presence of the accepts_t1
# probe + its None/absent-kwarg short-circuit path doesn't perturb
# numerical output on estimators whose fit signature lacks t1.


# Expected R² scores generated 2026-04-24 on macOS Accelerate with
# seed=42, n=200, 3 features, Ridge(solver='cholesky', alpha=1.0),
# y.astype(float) ∈ {-1, +1}, embargo_pct=0.05, n_splits=5. Closed-
# form solve verified byte-exact across repeated local runs. Plain-
# KFold reference (no purge, no embargo): mean R² 0.5838 vs
# cv_score_purged's 0.5853 — embargo perturbs train sets by a handful
# of samples, values track each other to ≤ 0.01 per fold. Values are
# real (strong linear signal on feature `a`; Ridge captures ~58% of
# variance), NOT the product of tiny-variance inflation or a slicing
# bug (sanity-checked against plain sklearn KFold pre-commit).
# Committed at full 16-digit precision so rtol=1e-12 is meaningful.
_PIN4_RIDGE_EXPECTED = np.array(
    [
        0.5452808141100086,
        0.6496892378538383,
        0.5608122379983618,
        0.5932354430593642,
        0.5775606222132356,
    ]
)


def test_pin4_backward_compat_no_t1_routing_closed_form() -> None:
    X, y, t1, _ = _make_fixture()
    # Ridge.fit signature: (X, y, sample_weight=None) — no t1, no **kwargs.
    # accepts_t1 probe returns False; t1 is not routed to fit.
    ridge = Ridge(solver="cholesky", alpha=1.0)

    scores = cv_score_purged(ridge, X, y.astype(float), t1=t1, embargo_pct=0.05, n_splits=5)

    np.testing.assert_allclose(scores, _PIN4_RIDGE_EXPECTED, rtol=1e-12, atol=1e-14)


# =============================================================================
# Pin 5 · MetaLabeler integration — end-to-end auto-resolution.
# =============================================================================


class _ObservingMetaLabeler(MetaLabeler):
    """MetaLabeler subclass that records each fit's resolved inner CV.

    Used by Pin 5 to validate that EVERY fold auto-resolves to a
    PurgedKFold with the correctly-sliced t1.

    The ClassVar list is managed via the ``observed_resolutions``
    pytest fixture (clears before + after yield). Do NOT access this
    ClassVar directly from tests.
    """

    _observed_resolutions: ClassVar[list[object]] = []

    def fit(self, X, y=None, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        super().fit(X, y, **kwargs)
        type(self)._observed_resolutions.append(self._resolved_oos_cv_)
        return self


@pytest.fixture
def observed_resolutions():
    """Fixture-scope the observer list's lifetime. Clears before yield
    and after yield — double-safety against cross-test state leakage."""
    _ObservingMetaLabeler._observed_resolutions.clear()
    yield _ObservingMetaLabeler._observed_resolutions
    _ObservingMetaLabeler._observed_resolutions.clear()


def test_pin5_end_to_end_autowire(observed_resolutions) -> None:
    X, y, t1, _ = _make_fixture()

    # Compute expected per-fold train sizes (non-zero y only — matches
    # drop_zero=True) from the SAME outer PurgedKFold split that
    # cv_score_purged builds internally. Reuses S8-pinned split logic.
    outer = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)
    per_fold_train_active = [
        int((y.iloc[train_idx] != 0).sum()) for train_idx, _te in outer.split(X)
    ]

    ml = _ObservingMetaLabeler(
        primary_model=LogisticRegression(max_iter=1000),
        meta_model=LogisticRegression(max_iter=1000),
        economic_rationale="S10 Pin 5 — auto-wire end-to-end",
        defer_cv_resolution=True,
        # Pin drop_zero=True explicitly so the default-flip risk is
        # contained (per review S1). Expected per_fold_train_active
        # computation assumes drop_zero=True.
        drop_zero=True,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cv_score_purged(ml, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert len(observed_resolutions) == 5, (
        f"expected 5 per-fold resolutions (n_splits); got {len(observed_resolutions)}"
    )

    for fold_cv, expected_len in zip(observed_resolutions, per_fold_train_active):
        assert isinstance(fold_cv, PurgedKFold), (
            f"auto-resolved CV must be PurgedKFold; got {type(fold_cv).__name__}"
        )
        assert len(fold_cv.t1) == expected_len, (
            f"auto-resolved PurgedKFold.t1 length {len(fold_cv.t1)} "
            f"!= expected (train ∩ active) size {expected_len}"
        )

    # Contract assertion: each fit must produce a FRESH resolution.
    # If ``_resolve_oos_cv`` ever caches across fits, all 5 observed
    # instances become the same Python object → only 1 distinct id,
    # and folds 1-4 would inherit fold-0's t1 slice. The per-fold
    # length check alone can miss this when PurgedKFold fold sizes
    # happen to be similar (~n·(k-1)/k each). Explicit id set is the
    # direct test for the caching-leak failure mode.
    distinct_ids = {id(cv) for cv in observed_resolutions}
    assert len(distinct_ids) == 5, (
        f"expected 5 distinct resolved PurgedKFold instances (one "
        f"per fold); got {len(distinct_ids)} distinct among "
        f"{len(observed_resolutions)} resolutions. MetaLabeler may "
        f"be caching _resolved_oos_cv_ across fit calls — inner CV "
        f"would inherit fold-0's t1 slice on folds 1+."
    )


# =============================================================================
# Pin 6 · S8 composition pin re-runs green post-S10.
# =============================================================================
# MANUAL: see stop_gate.acceptance; re-run tests/test_tml_composition.py.
# No code in this file.


# =============================================================================
# Pin 7 · sklearn.Pipeline + t1 — version-robust raise check.
# =============================================================================


def test_pin7_pipeline_plus_t1_raises_not_silent() -> None:
    X, y, t1, _ = _make_fixture()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # Pipeline.fit has **params signature; our probe returns
    # accepts_t1=True. sklearn then raises because t1 isn't step-
    # namespaced. We assert exception class only, NOT message text —
    # sklearn has reworded its metadata-routing error across versions.
    with pytest.raises((ValueError, TypeError)):
        cv_score_purged(pipe, X, y, t1=t1, embargo_pct=0.05, n_splits=3)


# =============================================================================
# Robustness — inspect.signature ValueError → accepts_t1=False fallback.
# =============================================================================
# Reviewer M1: the ``except (TypeError, ValueError)`` branch in
# cv_score_purged has no direct test coverage in the happy-path pins.
# Simulate the numba/C-extension failure mode via monkeypatch so we
# don't need numba as a test dependency, and don't depend on specific
# built-in callables that raise ValueError (which varies by Python
# version).


def test_probe_exception_path_skips_t1(monkeypatch) -> None:
    """When ``inspect.signature`` raises ValueError (numba-jitted or
    C-extension-backed fit), the probe falls through to
    ``accepts_t1=False`` and t1 is not propagated. Production code's
    DO-NOT-DELETE comment flags this as safety-critical.
    """

    def raising_signature(_obj):
        raise ValueError("synthetic: no signature for this callable (simulated numba)")

    # Patch the inspect.signature reference used inside purged_kfold.py.
    monkeypatch.setattr(_pk_module.inspect, "signature", raising_signature)

    X, y, t1, _ = _make_fixture()
    fake = _FakeWithT1()  # would normally accept t1

    # No raise — probe hits ValueError, falls through to False.
    scores = cv_score_purged(fake, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    assert scores.shape == (5,)
    assert len(fake.calls) == 5
    # t1 was NOT propagated despite fake.fit accepting it, because the
    # probe failed. This is the conservative fallback.
    for call in fake.calls:
        assert call["t1"] is None, (
            f"t1 should NOT have been propagated when probe raised; "
            f"got {call['t1']!r} on fold {len(fake.calls)}"
        )


def test_probe_exception_path_typeerror_also_skips(monkeypatch) -> None:
    """TypeError from ``inspect.signature`` (non-callable .fit) also
    takes the fallback path."""

    def raising_signature(_obj):
        raise TypeError("synthetic: not a callable")

    monkeypatch.setattr(_pk_module.inspect, "signature", raising_signature)

    X, y, t1, _ = _make_fixture()
    fake = _FakeWithT1()

    cv_score_purged(fake, X, y, t1=t1, embargo_pct=0.05, n_splits=5)

    for call in fake.calls:
        assert call["t1"] is None


# =============================================================================
# Robustness — positional alignment, scrambled-but-aligned index.
# =============================================================================
# Reviewer M2: docstring pins positional alignment as a hard
# precondition. Verify cv_score_purged slices ``t1`` positionally
# (by train_idx), not label-wise (by index), so callers with
# non-datetime or non-standard indices still get correct slices
# provided positional order is preserved.


def test_positional_alignment_x_index_differs_from_t1_index() -> None:
    """cv_score_purged slices t1 positionally via ``.iloc[train_idx]``,
    NOT via ``.loc[X.index[train_idx]]``. This matters when X.index
    and t1.index have different label types: the label-lookup variant
    would raise KeyError, while positional slicing works regardless.

    Regression probe: give X a RangeIndex while t1 keeps its
    DatetimeIndex. PurgedKFold's constraints on t1 (monotonic index,
    values >= index) still hold because only X's index changed. If
    cv_score_purged ever accidentally switches to label-based
    slicing, this test catches the change immediately (integers not
    found in a datetime index → KeyError on the first fold).

    Rejected alternative: changing t1's index type too (e.g., to
    RangeIndex) hits PurgedKFold's separate ``values >= index``
    ordering check, which requires both sides comparable.
    """
    X, y, t1, _ = _make_fixture()

    # X and y: strip to RangeIndex. t1 unchanged (keeps DatetimeIndex).
    X_r = X.reset_index(drop=True)
    y_r = y.reset_index(drop=True)

    fake = _FakeWithT1()
    cv_score_purged(fake, X_r, y_r, t1=t1, embargo_pct=0.05, n_splits=5)

    outer = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.05)
    for call, (train_idx, _te) in zip(fake.calls, outer.split(X_r)):
        # Compare VALUES (datetimes), not label-based lookup.
        np.testing.assert_array_equal(call["t1"].values, t1.iloc[train_idx].values)


# =============================================================================
# Robustness — non-sliceable t1 raises TypeError at entry.
# =============================================================================
# Reviewer S5: the entry-guard ``hasattr(t1, "__getitem__")`` check in
# production code has no test coverage. Make sure callers who pass a
# non-sliceable t1 get a clear error.


def test_entry_guard_non_sliceable_t1_raises() -> None:
    X, y, _, _ = _make_fixture()

    with pytest.raises(TypeError, match="sliceable"):
        cv_score_purged(
            Ridge(),
            X,
            y,
            t1=42,  # pyright: ignore[reportArgumentType]
            embargo_pct=0.05,
            n_splits=3,
        )
