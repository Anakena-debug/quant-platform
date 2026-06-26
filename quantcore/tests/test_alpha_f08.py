"""F-RP-002 regression: backtest_alpha_model delegates Sharpe to the
F08-gated quantcore.validation.stats.sharpe_ratio, instead of computing
inline with `+1e-10` softening that masks degenerate-variance PnL."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from quantcore.uncertainty.conformal.finance import alpha as alpha_mod
from quantcore.uncertainty.conformal.finance.alpha import (
    AlphaSignal,
    ConformalAlphaModel,
    backtest_alpha_model,
    compute_strategy_metrics,
)


class _TrivialModel(BaseEstimator):
    """Minimal sklearn-compatible inner model. `BaseEstimator` is required:
    `backtest_alpha_model` calls `sklearn.base.clone(model.model)` at every
    refit (alpha.py:957), and clone needs `get_params`/`set_params` —
    inherited from `BaseEstimator`. Without the inheritance, the test
    fails with `TypeError: Cannot clone object … does not implement a
    'get_params' method` on the first refit (test-infrastructure
    failure that masquerades as a code-under-test failure).
    """

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> "_TrivialModel":
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return np.zeros(len(X))

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return np.full((len(X), 2), 0.5)


def test_sharpe_delegates_to_validation_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """`backtest_alpha_model` must call validation.stats.sharpe_ratio, not
    inline `np.mean / (np.std + 1e-10)`. We spy on the imported symbol at
    the use site and assert it was invoked."""
    calls: list[tuple[NDArray[np.floating[Any]], tuple[Any, ...], dict[str, Any]]] = []

    def spy(returns: NDArray[np.floating[Any]], *args: Any, **kwargs: Any) -> float:
        calls.append((np.asarray(returns).copy(), args, kwargs))
        return 1.23

    # alpha.py is expected to do `from quantcore.validation.stats import sharpe_ratio`,
    # so the symbol lives at alpha_mod.sharpe_ratio.
    assert hasattr(alpha_mod, "sharpe_ratio"), (
        "F-RP-002: `sharpe_ratio` is not imported into alpha.py; the inline "
        "computation likely still stands."
    )
    monkeypatch.setattr(alpha_mod, "sharpe_ratio", spy)

    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    t1 = np.arange(n) + 1

    inner = _TrivialModel()
    model = ConformalAlphaModel(
        model=inner,
        alpha=0.1,
        method="split",
        random_state=0,
    )
    out = backtest_alpha_model(
        model=model,
        X=X,
        y=y,
        t1=t1,
        embargo=1,
        initial_train_size=100,
        refit_frequency=25,
    )

    assert calls, "F-RP-002: validation.stats.sharpe_ratio was never called"
    assert out["sharpe"] == pytest.approx(1.23), (
        "F-RP-002: returned Sharpe is not the delegated value, suggesting "
        "the inline computation still runs alongside the import."
    )


def test_sharpe_warns_and_returns_nan_on_f08_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the delegated `sharpe_ratio` raises ValueError (the F08
    degenerate-variance gate), `backtest_alpha_model` catches it, emits
    `UserWarning` mentioning F08/degenerate, and returns NaN for
    `results["sharpe"]`. Surfaces the F08 trigger loudly to pytest's
    warning recorder, CI logs, and the S20 harness without raising —
    so canonical regime-shift fixtures with expected-degenerate
    branches keep returning shape-valid metrics dicts."""

    def raising_spy(returns: NDArray[np.floating[Any]], *args: Any, **kwargs: Any) -> float:
        raise ValueError("zero variance: degenerate returns")

    monkeypatch.setattr(alpha_mod, "sharpe_ratio", raising_spy)

    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    t1 = np.arange(n) + 1

    inner = _TrivialModel()
    model = ConformalAlphaModel(
        model=inner,
        alpha=0.1,
        method="split",
        random_state=0,
    )

    with pytest.warns(UserWarning, match=r"F08|degenerate"):
        out = backtest_alpha_model(
            model=model,
            X=X,
            y=y,
            t1=t1,
            embargo=1,
            initial_train_size=100,
            refit_frequency=25,
        )
    assert np.isnan(out["sharpe"])


# =========================================================================
# F-RP-004a — compute_strategy_metrics sibling at alpha.py:759
#
# Same option-B pattern (warn+NaN on F08 trigger) applied to the
# StrategyMetrics-emitting helper. Spy-based delegation pin + warning-
# path pin, mirroring the F-RP-002 tests above at backtest_alpha_model
# scope.
# =========================================================================


def _make_signal(rng: np.random.Generator, n: int = 100) -> AlphaSignal:
    """Synthetic AlphaSignal fixture for compute_strategy_metrics tests."""
    expected_return = rng.normal(size=n)
    lower = expected_return - 0.5
    upper = expected_return + 0.5
    return AlphaSignal(
        expected_return=expected_return,
        lower=lower,
        upper=upper,
        alpha=0.1,
    )


def test_compute_strategy_metrics_delegates_to_validation_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`compute_strategy_metrics` must call validation.stats.sharpe_ratio,
    not inline `mean_return / (std_return + 1e-10) * sqrt(252)`. Spy on
    the imported symbol at the use site and assert (a) the spy was
    invoked AND (b) the returned StrategyMetrics.sharpe_ratio equals
    the delegated value, ruling out an inline computation surviving
    alongside the import."""
    calls: list[tuple[NDArray[np.floating[Any]], tuple[Any, ...], dict[str, Any]]] = []

    def spy(returns: NDArray[np.floating[Any]], *args: Any, **kwargs: Any) -> float:
        calls.append((np.asarray(returns).copy(), args, kwargs))
        return 4.56

    monkeypatch.setattr(alpha_mod, "sharpe_ratio", spy)

    rng = np.random.default_rng(0)
    signal = _make_signal(rng)
    returns_realized = rng.normal(size=len(signal.expected_return))

    metrics = compute_strategy_metrics(signal, returns_realized)

    assert calls, "F-RP-004a: validation.stats.sharpe_ratio was never called"
    assert metrics.sharpe_ratio == pytest.approx(4.56), (
        "F-RP-004a: returned StrategyMetrics.sharpe_ratio is not the "
        "delegated value, suggesting the inline computation still runs "
        "alongside the import."
    )


def test_compute_strategy_metrics_warns_and_returns_nan_on_f08_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the delegated `sharpe_ratio` raises ValueError (the F08
    degenerate-variance gate), `compute_strategy_metrics` catches it,
    emits `UserWarning` mentioning F08/degenerate, and returns NaN in
    `StrategyMetrics.sharpe_ratio`. Schema-validity of StrategyMetrics
    is preserved (other fields populate normally) so downstream
    callers can detect the degeneracy via `np.isnan(...)`."""

    def raising_spy(returns: NDArray[np.floating[Any]], *args: Any, **kwargs: Any) -> float:
        raise ValueError("zero variance: degenerate returns")

    monkeypatch.setattr(alpha_mod, "sharpe_ratio", raising_spy)

    rng = np.random.default_rng(0)
    signal = _make_signal(rng)
    returns_realized = rng.normal(size=len(signal.expected_return))

    with pytest.warns(UserWarning, match=r"F08|degenerate"):
        metrics = compute_strategy_metrics(signal, returns_realized)

    assert np.isnan(metrics.sharpe_ratio)
    # Schema-validity: other fields must still populate.
    assert np.isfinite(metrics.mean_return)
    assert np.isfinite(metrics.std_return)
    assert np.isfinite(metrics.coverage)
