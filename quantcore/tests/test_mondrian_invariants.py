"""Invariant pins for ``conformal.mondrian`` (S13 P12.3).

Six pins covering the MondrianConformal contract:
  1. Per-stratum coverage on a 2-regime synthetic
  2. Empty-stratum-by-construction (3-regime; stratum 3 unseen at
     fit time): fallback='global' returns global-pool interval AND
     diagnostic.used_fallback=True for those rows
  3. Empty-stratum fallback='raise' raises with named stratum
  4. Stratifier-call structural spy (class-side guarantee)
  5. Single-stratum-constant reduces to base estimator bitwise
  6. Class docstring states contract direction (caller-side vs
     class-side guarantee)

Pin 2 is specifically constructed to exercise the empty-stratum
code path — without it, the fallback-to-global behavior would be
untested code that nobody discovers until S14 hits a real fixture
(2026-04-29 review watch-item).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.mondrian import MondrianConformal
from quantcore.uncertainty.conformal.regression import SplitConformalRegressor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _split_factory(random_state: int = 42):
    """Factory for fresh SplitConformalRegressor instances. Each
    stratum gets its own; ``random_state`` is held constant so
    each factory call produces a deterministic train/cal split.
    Different per-stratum data ensures the splits index different
    rows even with the same seed."""

    def _make() -> SplitConformalRegressor:
        return SplitConformalRegressor(
            model=LinearRegression(), alpha=0.1, random_state=random_state
        )

    return _make


def _two_regime_data(seed: int = 0, n: int = 2000):
    """Two-regime synthetic. Regime A (high-vol) and regime B
    (low-vol). Stratifier is the ground-truth regime label.

    Returns (X, y, regime_labels, stratifier_callable)."""
    rng = np.random.default_rng(seed)
    # Half-and-half regime split, deterministic.
    regime = np.zeros(n, dtype=np.int_)
    regime[n // 2 :] = 1
    X_base = rng.standard_normal((n, 3))
    # Encode the regime in an extra column so the stratifier has
    # something to read directly from X.
    X = np.column_stack([X_base, regime.astype(np.float64)])
    # Returns: regime A → high-vol; regime B → low-vol.
    noise_scale = np.where(regime == 0, 1.0, 0.2)
    y = X_base.sum(axis=1) + noise_scale * rng.standard_normal(n)

    def stratifier(X_in: np.ndarray) -> np.ndarray:
        # Read regime label from the last column (which we encoded).
        # NOTE: this is a TEST-ONLY stratifier; in a real pipeline
        # the regime label would come from a leakage-free side
        # model.
        return X_in[:, -1].astype(np.int_)

    return X, y, regime, stratifier


def _three_regime_with_unseen_at_fit(seed: int = 0, n: int = 2000):
    """Three-regime synthetic; calibration only sees regimes 0/1.
    Test set includes regime 2 (unseen at fit time) on purpose.

    Returns (X_train, y_train, X_test, y_test, stratifier)."""
    rng = np.random.default_rng(seed)
    n_train, n_test = n, n // 4

    # Train: regimes 0 and 1 only, balanced.
    train_regime = np.zeros(n_train, dtype=np.int_)
    train_regime[n_train // 2 :] = 1
    X_train_base = rng.standard_normal((n_train, 3))
    X_train = np.column_stack([X_train_base, train_regime.astype(np.float64)])
    train_noise = np.where(train_regime == 0, 1.0, 0.5)
    y_train = X_train_base.sum(axis=1) + train_noise * rng.standard_normal(n_train)

    # Test: mix of regimes 0, 1, AND 2. Regime 2 unseen at fit.
    test_regime = rng.integers(0, 3, size=n_test).astype(np.int_)
    X_test_base = rng.standard_normal((n_test, 3))
    X_test = np.column_stack([X_test_base, test_regime.astype(np.float64)])
    test_noise = np.where(test_regime == 0, 1.0, np.where(test_regime == 1, 0.5, 0.3))
    y_test = X_test_base.sum(axis=1) + test_noise * rng.standard_normal(n_test)

    def stratifier(X_in: np.ndarray) -> np.ndarray:
        return X_in[:, -1].astype(np.int_)

    return X_train, y_train, X_test, y_test, stratifier


# -----------------------------------------------------------------------------
# Pin 1 — per-stratum coverage on 2-regime synthetic.
# -----------------------------------------------------------------------------


def test_pin_mondrian_per_stratum_coverage_two_regime() -> None:
    """Two-regime synthetic; per-stratum empirical coverage on a
    held-out test set must clear the published 1-α-η threshold
    on EACH stratum (η = 0.03).

    Without Mondrian, a globally-fit conformal regressor would
    over-cover the low-vol regime and under-cover the high-vol
    regime; per-stratum calibration restores conditional coverage
    in both.
    """
    X, y, _regime, stratifier = _two_regime_data(seed=0, n=4000)
    # Train/test split: hold out 25% per regime.
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(y))
    cut = int(0.75 * len(y))
    train_idx, test_idx = perm[:cut], perm[cut:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    mc = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=stratifier,
        alpha=0.1,
    )
    mc.fit(X_tr, y_tr)
    intervals, diagnostic = mc.predict(X_te)

    # Per-stratum coverage check.
    test_labels = diagnostic["stratum_labels"]
    target = 1.0 - 0.1
    threshold = target - 0.03
    for s in np.unique(test_labels):
        mask = test_labels == s
        # Should not have any fallback rows in this two-regime
        # synthetic (both regimes are seen at fit).
        assert not diagnostic["used_fallback"][mask].any(), (
            f"unexpected fallback on stratum {s} in two-regime test"
        )
        cov = float(intervals.contains(y_te)[mask].mean())
        assert cov >= threshold, (
            f"stratum {s}: coverage {cov:.4f} < threshold {threshold:.4f} (target {target}, η=0.03)"
        )


# -----------------------------------------------------------------------------
# Pin 2 — empty stratum exercised by construction (fallback='global').
# -----------------------------------------------------------------------------


def test_pin_mondrian_empty_stratum_fallback_global_exercised() -> None:
    """Three-regime synthetic, regime 2 unseen at fit time. With
    empty_stratum_fallback='global':

      (i)   fitted strata = {0, 1} only.
      (ii)  predict on test set with regime-2 rows succeeds.
      (iii) regime-2 rows have used_fallback=True.
      (iv)  regime-0/1 rows have used_fallback=False.
      (v)   the fallback-to-global path is EXERCISED (i.e., not
            untested code).

    This pin specifically targets the empty-stratum code path
    that would otherwise stay untested until S14 hits a real
    fixture (per 2026-04-29 review watch-item).
    """
    X_tr, y_tr, X_te, y_te, stratifier = _three_regime_with_unseen_at_fit(seed=11, n=2000)
    mc = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=stratifier,
        alpha=0.1,
        empty_stratum_fallback="global",
    )
    mc.fit(X_tr, y_tr)

    # Fit-time strata = {0, 1} only.
    assert set(mc.per_stratum_n.keys()) == {0, 1}

    intervals, diagnostic = mc.predict(X_te)
    test_labels = diagnostic["stratum_labels"]
    used_fallback = diagnostic["used_fallback"]

    # Regime 2 should be present in test set (otherwise the pin
    # is vacuous); at least one regime-2 row must exist.
    n_regime_2 = (test_labels == 2).sum()
    assert n_regime_2 > 0, (
        "test set has no regime-2 rows; empty-stratum code path "
        "would not be exercised — pin is vacuous"
    )

    # All regime-2 rows: used_fallback=True.
    assert used_fallback[test_labels == 2].all(), (
        "regime-2 rows did not use the global-pool fallback"
    )
    # All regime-0/1 rows: used_fallback=False.
    assert not used_fallback[test_labels != 2].any(), (
        "regime-0/1 rows incorrectly flagged as fallback"
    )
    # Intervals are well-formed (lower <= upper for all rows).
    assert (intervals.lower <= intervals.upper).all()


# -----------------------------------------------------------------------------
# Pin 3 — empty-stratum fallback='raise'.
# -----------------------------------------------------------------------------


def test_pin_mondrian_empty_stratum_fallback_raise() -> None:
    """Same three-regime setup, fallback='raise': predict raises
    ValueError naming the unseen stratum label."""
    X_tr, y_tr, X_te, y_te, stratifier = _three_regime_with_unseen_at_fit(seed=11, n=2000)
    mc = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=stratifier,
        alpha=0.1,
        empty_stratum_fallback="raise",
    )
    mc.fit(X_tr, y_tr)
    with pytest.raises(ValueError, match=r"Stratum 2 unseen"):
        mc.predict(X_te)


# -----------------------------------------------------------------------------
# Pin 4 — stratifier-call signature (class-side guarantee).
# -----------------------------------------------------------------------------


def test_pin_mondrian_stratifier_called_with_X_only() -> None:
    """Structural leakage guard: the spy-wrapped stratifier
    records every (args, kwargs) call from MondrianConformal's
    fit and predict. Verify:

      - Every call has exactly 1 positional argument.
      - Every call has zero kwargs.
      - The single positional argument has the same shape as
        the expected X (rows × features).

    The test does NOT verify the stratifier's INTERNAL behavior
    is leakage-free — that's a caller-side guarantee per the
    docstring contract direction. This pin bounds the class-side
    half of the contract.
    """
    X, y, _regime, _strat = _two_regime_data(seed=2, n=200)
    expected_n_features = X.shape[1]

    calls: list[tuple[tuple, dict]] = []

    def spy_stratifier(X_in: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        # Capture the FULL signature: extra positional args show up
        # in `args`, kwargs in `kwargs`. If MondrianConformal ever
        # passed y or t1, they'd land in `args` or `kwargs`.
        calls.append((args, dict(kwargs)))
        return X_in[:, -1].astype(np.int_)

    mc = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=spy_stratifier,
        alpha=0.1,
    )
    mc.fit(X[:150], y[:150])
    mc.predict(X[150:])

    # At least 2 calls (one in fit, one in predict).
    assert len(calls) >= 2, (
        f"stratifier called {len(calls)} times; expected at least two (fit + predict)"
    )
    # Every call had exactly 1 positional arg (the spy received it
    # as the named X_in; extras would have shown up in `args`).
    for i, (extra_args, extra_kwargs) in enumerate(calls):
        assert extra_args == (), (
            f"stratifier call #{i} received extra positional "
            f"args: {extra_args}; class-side guarantee violated"
        )
        assert extra_kwargs == {}, (
            f"stratifier call #{i} received kwargs: {extra_kwargs}; class-side guarantee violated"
        )

    # Additionally verify the SHAPE of what was passed: only X
    # arrays were passed, no y vectors. Sniff the actual X arg
    # via a separate spy that captures the X argument.
    captured_xs: list[np.ndarray] = []

    def shape_capture(X_in: np.ndarray) -> np.ndarray:
        captured_xs.append(np.asarray(X_in))
        return X_in[:, -1].astype(np.int_)

    mc2 = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=shape_capture,
        alpha=0.1,
    )
    mc2.fit(X[:150], y[:150])
    mc2.predict(X[150:])
    for i, captured in enumerate(captured_xs):
        assert captured.ndim == 2, (
            f"stratifier call #{i} received non-2D array of shape {captured.shape}"
        )
        assert captured.shape[1] == expected_n_features, (
            f"stratifier call #{i} received array with "
            f"{captured.shape[1]} columns; expected "
            f"{expected_n_features} (feature count of X). "
            f"A different column count would indicate y or t1 "
            f"was concatenated."
        )


# -----------------------------------------------------------------------------
# Pin 5 — single-stratum-constant reduces to base estimator bitwise.
# -----------------------------------------------------------------------------


def test_pin_mondrian_single_stratum_reduces_to_base_estimator() -> None:
    """Stratifier returning a constant label for all X reduces
    Mondrian's intervals to the base estimator's intervals
    bitwise-identically. Degenerate-case sanity check.

    Constructs two SplitConformalRegressor instances with identical
    seed-determined behavior: the base instance fits on full
    (X_train, y_train) directly; the Mondrian instance with a
    constant stratifier fits on the same data. Mondrian dispatches
    to its single-stratum estimator, which is the base estimator
    fit on the SAME (X, y). Intervals must match bitwise.
    """
    X, y, _regime, _strat = _two_regime_data(seed=3, n=600)
    cut = int(0.75 * len(y))
    X_tr, y_tr = X[:cut], y[:cut]
    X_te = X[cut:]

    # Direct base — fixed random_state so the train/cal split is
    # deterministic and matches the Mondrian-internal estimator.
    base = SplitConformalRegressor(model=LinearRegression(), alpha=0.1, random_state=42)
    base.fit(X_tr, y_tr)
    base_iv = base.predict(X_te)

    # Mondrian with constant stratifier — same random_state so the
    # internal estimator's split matches the direct-base split.
    def constant_stratifier(X_in: np.ndarray) -> np.ndarray:
        return np.zeros(X_in.shape[0], dtype=np.int_)

    mc = MondrianConformal(
        base_estimator_factory=lambda: SplitConformalRegressor(
            model=LinearRegression(), alpha=0.1, random_state=42
        ),
        stratifier=constant_stratifier,
        alpha=0.1,
    )
    mc.fit(X_tr, y_tr)
    mc_iv, _diag = mc.predict(X_te)

    # Bitwise equality on bounds.
    np.testing.assert_array_equal(mc_iv.lower, base_iv.lower)
    np.testing.assert_array_equal(mc_iv.upper, base_iv.upper)


# -----------------------------------------------------------------------------
# Pin 6 — docstring states contract direction.
# -----------------------------------------------------------------------------


def test_pin_mondrian_docstring_states_contract_direction() -> None:
    """Grep guard: the MondrianConformal class docstring contains
    both 'Class-side guarantee' and 'Caller-side guarantee'. The
    contract direction IS the spec — losing it from the docstring
    loses the spec."""
    doc = MondrianConformal.__doc__ or ""
    assert "Class-side guarantee" in doc, (
        "MondrianConformal docstring missing 'Class-side guarantee' "
        "header; contract direction not stated"
    )
    assert "Caller-side guarantee" in doc, (
        "MondrianConformal docstring missing 'Caller-side guarantee' "
        "header; contract direction not stated"
    )


# -----------------------------------------------------------------------------
# Bonus structural checks (not in plan acceptance, but cheap and useful).
# -----------------------------------------------------------------------------


def test_mondrian_validates_alpha_at_init() -> None:
    """alpha out of (0, 1) raises ValueError at construction."""
    with pytest.raises(ValueError, match=r"alpha"):
        MondrianConformal(
            base_estimator_factory=_split_factory(),
            stratifier=lambda X: np.zeros(X.shape[0], dtype=np.int_),
            alpha=0.0,
        )


def test_mondrian_validates_fallback_at_init() -> None:
    """Invalid empty_stratum_fallback raises ValueError at
    construction."""
    with pytest.raises(ValueError, match=r"empty_stratum_fallback"):
        MondrianConformal(
            base_estimator_factory=_split_factory(),
            stratifier=lambda X: np.zeros(X.shape[0], dtype=np.int_),
            alpha=0.1,
            empty_stratum_fallback="silent",  # type: ignore[arg-type]
        )


def test_mondrian_unfitted_predict_raises() -> None:
    """predict() before fit() raises RuntimeError."""
    mc = MondrianConformal(
        base_estimator_factory=_split_factory(),
        stratifier=lambda X: np.zeros(X.shape[0], dtype=np.int_),
        alpha=0.1,
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        mc.predict(np.zeros((5, 3)))
