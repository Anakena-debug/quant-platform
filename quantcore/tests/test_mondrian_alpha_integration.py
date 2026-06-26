"""Integration pins for ConformalAlphaModel(method='mondrian') (S14 P13.1).

Five pins covering the new branch:

  1. Fit produces valid AlphaSignal with non-empty arrays on a
     2-regime synthetic.
  2. predict() dispatches through MondrianConformal (composition,
     not inlined math) — pinned via spy on the imported name.
  3. Missing stratifier raises ValueError naming the kwarg.
  4. Invalid mondrian_base_method raises ValueError.
  5. backtest_alpha_model with method='mondrian' runs end-to-end
     and produces non-empty results.

Out of scope (per S14 plan):
  - Empirical comparison vs split/cv/cqr branches (S16+).
  - mondrian-on-cqr / mondrian-on-weighted base methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from quantcore.uncertainty.conformal.finance import alpha as alpha_mod
from quantcore.uncertainty.conformal.finance.alpha import (
    AlphaSignal,
    ConformalAlphaModel,
    PortfolioConstructor,
    SignalFilter,
    backtest_alpha_model,
)


# -----------------------------------------------------------------------------
# Two-regime synthetic + canonical stratifier.
# -----------------------------------------------------------------------------


def _two_regime_synthetic(seed: int = 7, n: int = 400):
    """Two-regime synthetic with regime label encoded in the last
    feature column. Regime A (high-vol) and B (low-vol)."""
    rng = np.random.default_rng(seed)
    regime = np.zeros(n, dtype=np.int_)
    regime[n // 2 :] = 1
    X_base = rng.standard_normal((n, 3))
    X = np.column_stack([X_base, regime.astype(np.float64)])
    noise_scale = np.where(regime == 0, 1.0, 0.3)
    y = X_base.sum(axis=1) + noise_scale * rng.standard_normal(n)
    return X, y


def _stratifier(X_in: np.ndarray) -> np.ndarray:
    """Read regime label from the last column."""
    return X_in[:, -1].astype(np.int_)


# -----------------------------------------------------------------------------
# Pin 1 — fit produces valid AlphaSignal.
# -----------------------------------------------------------------------------


def test_pin_mondrian_fit_produces_valid_alpha_signal() -> None:
    """method='mondrian' fit produces an AlphaSignal whose
    lower/upper arrays are non-empty, well-formed (lower ≤ upper),
    and consistent with the test set size."""
    X, y = _two_regime_synthetic(seed=7, n=400)
    X_tr, X_te = X[:300], X[300:]
    y_tr = y[:300]

    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        stratifier=_stratifier,
        random_state=42,
    )
    m.fit(X_tr, y_tr)
    sig = m.predict(X_te)

    assert isinstance(sig, AlphaSignal)
    assert sig.lower.shape == (X_te.shape[0],)
    assert sig.upper.shape == (X_te.shape[0],)
    assert sig.expected_return.shape == (X_te.shape[0],)
    assert (sig.lower <= sig.upper).all(), "lower > upper somewhere"
    assert sig.alpha == 0.1


# -----------------------------------------------------------------------------
# Pin 2 — predict() dispatches through MondrianConformal.
# -----------------------------------------------------------------------------


def test_pin_mondrian_predict_dispatches_through_mondrian_conformal(
    monkeypatch,
) -> None:
    """predict() on a mondrian-method model calls
    MondrianConformal.predict — verified by patching the imported
    name in alpha.py and confirming the resulting AlphaSignal
    arrays come from the patched return value, not from inline
    quantile math.
    """
    sentinel_lower = np.array([-9.99, -9.99, -9.99, -9.99, -9.99])
    sentinel_upper = np.array([+9.99, +9.99, +9.99, +9.99, +9.99])
    sentinel_diag = {"used_fallback": np.zeros(5, dtype=bool)}

    class _SpyMondrian:
        called: list[tuple] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, **fit_params):
            type(self).called.append(("fit", X.shape, y.shape))
            return self

        def predict(self, X):
            type(self).called.append(("predict", X.shape))
            from quantcore.uncertainty.conformal.base import (
                PredictionInterval,
            )

            interval = PredictionInterval(
                lower=sentinel_lower,
                upper=sentinel_upper,
                alpha=0.1,
            )
            return interval, sentinel_diag

    monkeypatch.setattr(alpha_mod, "MondrianConformal", _SpyMondrian)

    X, y = _two_regime_synthetic(seed=2, n=300)
    X_tr, X_te = X[:200], X[200:205]  # 5 test rows to match sentinel
    y_tr = y[:200]

    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        stratifier=_stratifier,
        random_state=42,
    )
    m.fit(X_tr, y_tr)
    sig = m.predict(X_te)

    # AlphaSignal arrays must come from the spy's predict return.
    np.testing.assert_array_equal(sig.lower, sentinel_lower)
    np.testing.assert_array_equal(sig.upper, sentinel_upper)
    # expected_return is midpoint of (sentinel_lower, sentinel_upper) = 0
    np.testing.assert_array_equal(sig.expected_return, np.zeros(5))
    # Spy was called for both fit and predict.
    fit_calls = [c for c in _SpyMondrian.called if c[0] == "fit"]
    pred_calls = [c for c in _SpyMondrian.called if c[0] == "predict"]
    assert len(fit_calls) >= 1, "MondrianConformal.fit not invoked"
    assert len(pred_calls) >= 1, "MondrianConformal.predict not invoked"


# -----------------------------------------------------------------------------
# Pin 3 — missing stratifier raises with kwarg name.
# -----------------------------------------------------------------------------


def test_pin_mondrian_missing_stratifier_raises() -> None:
    """method='mondrian' without stratifier raises ValueError
    naming the kwarg."""
    X, y = _two_regime_synthetic(seed=3, n=200)
    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        # stratifier intentionally omitted
        random_state=42,
    )
    with pytest.raises(ValueError, match=r"stratifier"):
        m.fit(X[:150], y[:150])


# -----------------------------------------------------------------------------
# Pin 4 — invalid mondrian_base_method raises.
# -----------------------------------------------------------------------------


def test_pin_mondrian_invalid_base_method_raises() -> None:
    """mondrian_base_method outside the supported set raises
    ValueError naming the offending value."""
    X, y = _two_regime_synthetic(seed=4, n=200)
    m = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        stratifier=_stratifier,
        mondrian_base_method="cqr",  # type: ignore[arg-type]  # not in S14
        random_state=42,
    )
    with pytest.raises(ValueError, match=r"mondrian_base_method"):
        m.fit(X[:150], y[:150])


# -----------------------------------------------------------------------------
# Pin 5 — backtest_alpha_model with method='mondrian' runs end-to-end.
# -----------------------------------------------------------------------------


def test_pin_mondrian_backtest_runs_end_to_end() -> None:
    """backtest_alpha_model walks forward with method='mondrian',
    refits cleanly at the refit boundary, and produces a non-empty
    results dict with the expected keys.

    Default refit_frequency=21 with initial_train_size=252 means
    multiple refits within a 400-row series; each refit must
    re-construct MondrianConformal via the forwarded stratifier.
    The test would fail (silently or noisily) if backtest's refit
    path didn't forward the stratifier kwarg.
    """
    X, y = _two_regime_synthetic(seed=5, n=400)

    template = ConformalAlphaModel(
        model=LinearRegression(),
        alpha=0.1,
        method="mondrian",
        stratifier=_stratifier,
        random_state=42,
    )
    results = backtest_alpha_model(
        model=template,
        X=X,
        y=y,
        initial_train_size=252,
        refit_frequency=21,
        signal_filter=SignalFilter(min_signal_strength=0.0),
        portfolio_constructor=PortfolioConstructor(method="equal"),
    )

    expected_keys = {
        "signals",
        "weights",
        "returns",
        "covered",
        "trade_mask",
        "portfolio_returns",
        "cumulative_return",
        "coverage",
        "trade_rate",
        "sharpe",
    }
    assert expected_keys.issubset(set(results.keys())), (
        f"missing keys: {expected_keys - set(results.keys())}"
    )
    # Walk-forward produced at least 100 step results.
    assert len(results["signals"]) > 100
    assert len(results["weights"]) > 100
    # Coverage is a real number in [0, 1] (the backtest didn't
    # silently drop all signals).
    assert 0.0 <= results["coverage"] <= 1.0
