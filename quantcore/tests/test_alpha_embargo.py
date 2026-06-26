"""F-RP-001 regression tests — `backtest_alpha_model` embargo + t1 plumbing.

Pre-S-finding-RP-001, ``backtest_alpha_model`` fits each refit on
``X[:t], y[:t]`` with no notion of label-window overlap. With triple-
barrier labels at horizon ``h ≥ 2``, training events ``i ∈ [t-h+1, t-1]``
have ``t1[i] ≥ t`` — their barrier resolution falls AFTER the prediction
time, allowing label-noise from ``[t..t+h-2]`` to inform the model.
This is the AFML §7.4 leakage scenario.

The fix adds an ``embargo: int = 0`` parameter and an optional ``t1``
series. At each refit at time ``t`` with embargo ``k``, training is
restricted to events whose ``t1 ≤ t - k - 1``. Default ``embargo=0``
preserves current behavior. When triple-barrier-style labels are
detected (``t1`` provided AND varies) AND ``embargo=0``, a UserWarning
fires recommending ``embargo=h+1``.

Three discriminators:

  * ``test_embargo_reduces_r2_on_overlapping_labels`` — synthetic with
    overlapping h=10 labels; R² without embargo exceeds R² with
    embargo=h+1 by ≥ 0.05 absolute (literal threshold from spec).
  * ``test_embargo_postcondition_t1_constraint`` — at every refit, the
    set of training events has ``max(t1) ≤ t - embargo - 1``. Replaces
    the dropped PurgedKFold-equivalence comparison; directly tests the
    embargo logic.
  * ``test_warns_on_varying_t1_and_zero_embargo`` (+ companions) —
    UserWarning fires only on triple-barrier-style input with
    embargo=0; silent for ``t1=None`` and ``t1`` constant.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from quantcore.uncertainty.conformal.finance import (
    ConformalAlphaModel,
    backtest_alpha_model,
)


def _r2_from_results(results: dict[str, Any]) -> float:
    """Out-of-sample R² over the walk-forward predictions."""
    preds = np.array([float(s.expected_return[0]) for s in results["signals"]])
    actuals = np.asarray(results["returns"], dtype=np.float64)
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def test_embargo_reduces_r2_on_overlapping_labels() -> None:
    """Overlapping h-bar labels with no embargo inflate out-of-sample R².

    Construction: a single monotone-in-time feature (so nearest in
    feature space ≡ nearest in time) and label
    ``y[i] = Σ noise[i..i+h-1]``. Consecutive labels share ``(h-1)/h``
    of their noise composition. A k=3 kNN regressor without embargo
    averages ``{y[t-1], y[t-2], y[t-3]}`` to predict ``y[t]``; those
    training labels share noise terms with ``y[t]`` (the prediction-
    time bar's forward window), giving a memorization shortcut. With
    embargo=h+1, the nearest available training events are at
    ``t - 21, t - 22, t - 23``, whose label-windows do not overlap
    ``y[t]`` at all — the shortcut is closed.

    Algebraic prediction (var-of-noise = 1):
      * Without embargo: corr(y_hat, y) ≈ 8 / √(82/9 · 10) ≈ 0.838,
        so R² ≈ 0.69.
      * With embargo=h+1: cov(y_hat, y) = 0, so R² ≈ -0.91 (predictor
        worse than the mean baseline).

    The 0.05 threshold is the spec literal; the realised delta on this
    fixture is much larger, leaving room for stochastic split-fraction
    drift without flipping verdict.
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    h = 10

    # Monotone single feature: nearest in feature ≡ nearest in time.
    # Multi-cycle periodic features collapse distant-in-time pairs
    # into near-neighbors and dilute the leakage signal.
    X: NDArray[np.floating[Any]] = np.linspace(0.0, 1.0, n).reshape(-1, 1)

    # Overlapping h-bar label. noise array of length n+h supplies the
    # forward window; var(y[i]) = h, cov(y[i], y[j]) = max(0, h-|i-j|).
    noise = rng.standard_normal(n + h)
    y: NDArray[np.floating[Any]] = np.array(
        [noise[i : i + h].sum() for i in range(n)],
        dtype=np.float64,
    )
    t1: NDArray[np.integer[Any]] = np.arange(n, dtype=np.int64) + (h - 1)

    base = KNeighborsRegressor(n_neighbors=3)

    model_leak = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)
    res_leak = backtest_alpha_model(
        model_leak,
        X,
        y,
        initial_train_size=100,
        refit_frequency=21,
        t1=t1,
        embargo=0,
    )

    model_safe = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)
    res_safe = backtest_alpha_model(
        model_safe,
        X,
        y,
        initial_train_size=100,
        refit_frequency=21,
        t1=t1,
        embargo=h + 1,
    )

    r2_leak = _r2_from_results(res_leak)
    r2_safe = _r2_from_results(res_safe)
    delta = r2_leak - r2_safe

    assert delta >= 0.05, (
        f"Expected R² leakage delta ≥ 0.05; got Δ={delta:.4f} "
        f"(no embargo R²={r2_leak:.4f}, embargo={h + 1} R²={r2_safe:.4f})"
    )


def test_embargo_postcondition_t1_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    """At every refit at time t with embargo=k, every event in the training
    set satisfies ``t1 ≤ t - k - 1``.

    Direct test of the embargo logic via spying on the inner
    ``ConformalAlphaModel.fit`` to capture the y array passed to each
    refit. We set y[i]=i so y values double as event indices, giving
    direct readback of the training-event set.

    This assertion replaces the dropped PurgedKFold-equivalence
    comparison from the original spec — it tests the fix's postcondition
    rather than equivalence to a sibling implementation.
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    h = 10
    embargo = h + 1
    initial_train_size = 100
    refit_frequency = 50

    X = rng.standard_normal((n, 3))
    # y[i] = i lets us recover indices from captured y arrays.
    y: NDArray[np.floating[Any]] = np.arange(n, dtype=np.float64)
    t1: NDArray[np.integer[Any]] = np.arange(n, dtype=np.int64) + (h - 1)

    captured: list[NDArray[np.floating[Any]]] = []
    orig_fit = ConformalAlphaModel.fit

    def spy_fit(
        self: ConformalAlphaModel,
        X_in: NDArray[np.floating[Any]],
        y_in: NDArray[np.floating[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> ConformalAlphaModel:
        captured.append(np.asarray(y_in, dtype=np.float64).copy())
        return orig_fit(self, X_in, y_in, *args, **kwargs)

    monkeypatch.setattr(ConformalAlphaModel, "fit", spy_fit)

    base = LinearRegression()
    model = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)
    backtest_alpha_model(
        model,
        X,
        y,
        initial_train_size=initial_train_size,
        refit_frequency=refit_frequency,
        t1=t1,
        embargo=embargo,
    )

    # The walk-forward refits at t = initial_train_size, then every
    # refit_frequency bars. Reconstruct the schedule deterministically
    # and pair with captured fits.
    refit_times: list[int] = []
    t = initial_train_size
    steps_since_refit = 0
    fitted = False
    while t < n:
        if not fitted or steps_since_refit >= refit_frequency:
            refit_times.append(t)
            steps_since_refit = 0
            fitted = True
        t += 1
        steps_since_refit += 1

    assert len(captured) == len(refit_times), (
        f"refit count mismatch: captured={len(captured)} vs predicted={len(refit_times)}"
    )

    for refit_idx, (t_at_refit, y_train) in enumerate(zip(refit_times, captured)):
        # Each captured y array is the training-event values at this
        # refit; y[i]=i, so values ARE indices.
        if y_train.size == 0:
            # Empty training set is acceptable degenerate case at very
            # small t with large embargo; pinning the indexing rather
            # than re-deriving emptiness.
            assert t_at_refit - h - embargo < 0
            continue
        max_idx_in_train = int(y_train.max())
        max_t1_in_train = int(t1[max_idx_in_train])
        assert max_t1_in_train <= t_at_refit - embargo - 1, (
            f"Refit #{refit_idx} at t={t_at_refit}, embargo={embargo}: "
            f"training set contains event with t1={max_t1_in_train} > "
            f"t-embargo-1={t_at_refit - embargo - 1}"
        )


def test_warns_on_varying_t1_and_zero_embargo() -> None:
    """t1 provided AND varies AND embargo=0 → UserWarning recommending h+1.

    The fixture is pure-noise X/y; LinearRegression on it produces no
    tradeable signals, so the post-loop ``sharpe_ratio`` call trips the
    F-RP-002 F08 gate. Under option B that gate emits its own
    UserWarning rather than raising — both warnings are recorded.
    ``pytest.warns(..., match=...)`` matches at least one warning whose
    message contains the F-RP-001 phrase ``embargo = h + 1``; the F08
    warning text does not match that pattern, so the multi-warning
    context does not confuse the assertion.
    """
    rng = np.random.default_rng(seed=42)
    n = 200
    h = 10

    X = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    t1: NDArray[np.integer[Any]] = np.arange(n, dtype=np.int64) + (h - 1)

    base = LinearRegression()
    model = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)

    with pytest.warns(UserWarning, match=r"embargo.*h\s*\+\s*1"):
        backtest_alpha_model(
            model,
            X,
            y,
            initial_train_size=100,
            refit_frequency=21,
            t1=t1,
            embargo=0,
        )


def test_no_warning_when_t1_none() -> None:
    """t1=None (no triple-barrier signal) + embargo=0 → silent for the
    F-RP-001 detection warning. Pure-noise fixture trips F-RP-002's F08
    warning — orthogonal — so we filter only the F-RP-001 message to
    `error`, leaving F08 warnings to pass through.

    Regex pinned to the actual F-RP-001 prefix at `alpha.py:975` so a
    future warning that happens to contain "embargo" / "h" / "1" /
    "+" in some other order is not promoted to error.
    """
    rng = np.random.default_rng(seed=42)
    n = 200

    X = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)

    base = LinearRegression()
    model = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"backtest_alpha_model: t1 varies.*embargo=0",
            category=UserWarning,
        )
        backtest_alpha_model(
            model,
            X,
            y,
            initial_train_size=100,
            refit_frequency=21,
            embargo=0,
        )


def test_no_warning_when_t1_constant() -> None:
    """t1 provided but constant (non-TB labels) + embargo=0 → silent for
    the F-RP-001 detection warning. A constant t1 series fails the
    "varies" half of the detection predicate, so the warning must not
    fire — even though t1 is explicitly supplied. Pure-noise fixture
    trips F-RP-002's F08 warning (orthogonal); filter scopes the
    `error` action to the F-RP-001 message verbatim.
    """
    rng = np.random.default_rng(seed=42)
    n = 200

    X = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    t1_constant: NDArray[np.integer[Any]] = np.zeros(n, dtype=np.int64)

    base = LinearRegression()
    model = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"backtest_alpha_model: t1 varies.*embargo=0",
            category=UserWarning,
        )
        backtest_alpha_model(
            model,
            X,
            y,
            initial_train_size=100,
            refit_frequency=21,
            t1=t1_constant,
            embargo=0,
        )
