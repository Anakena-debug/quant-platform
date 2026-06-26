"""DtACI-aware walk-forward backtest helper for the alpha pipeline.

Sibling to ``backtest_alpha_model`` in ``alpha.py``. Consumes
``DtACI`` directly (online conformal, per-step adaptation) without
wrapping it in a class and without grafting an online method
literal onto ``ConformalAlphaModel`` — see the S15 sprint plan
and the route decision committed at 1a58db5.

Architectural invariants pinned by the S15 acceptance gate:

  * ``ConformalAlphaModel`` is not modified — DtACI's online
    consumption surface lives in this file, not as a fifth
    method literal on the batch class.
  * The caller's ``base_model`` instance is never mutated; every
    fit boundary clones it.
  * The caller's ``DtACI`` instance IS mutated in place; state
    (score buffer, expert α, expert weights) is preserved across
    ``refit_frequency`` boundaries.
  * Score buffer length grows monotonically over the full
    walk-forward window; ``DtACI.window_size`` caps only the
    quantile-computation slice, not the stored buffer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from quantcore.uncertainty.conformal.dtaci import DtACI
from quantcore.uncertainty.conformal.finance.alpha import (
    AlphaSignal,
    PortfolioConstructor,
    SignalFilter,
)


def backtest_alpha_model_dtaci(
    base_model: BaseEstimator,
    dtaci: DtACI,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    *,
    initial_train_size: int = 252,
    refit_frequency: int = 21,
    warmup: int = 50,
    signal_filter: SignalFilter | None = None,
    portfolio_constructor: PortfolioConstructor | None = None,
) -> dict[str, Any]:
    """Walk-forward backtest with DtACI online α-adaptation.

    The DtACI score buffer carries calibration history across
    refit boundaries; only the base model is re-fit. ``dtaci`` is
    mutated in place — caller passes a fresh ``DtACI(...)``.
    ``base_model`` is never mutated; every fit boundary clones it.

    Lifecycle:

      1. Fit ``clone(base_model)`` on ``X[:initial_train_size]``.
      2. Initialize ``dtaci`` over the training window via
         ``dtaci.run_online(...)``: score-buffer warmup over the
         first ``warmup`` rows, then online phase
         (``predict_step`` + ``update_step``) over the remainder.
         By the start of walk-forward, the score buffer holds
         ``initial_train_size`` entries.
      3. For each ``t`` in ``[initial_train_size, n)``: refit if
         the steps-since-refit counter has reached
         ``refit_frequency``; issue a prediction interval via
         ``dtaci.predict_step``; build an ``AlphaSignal``;
         filter + construct portfolio weight; record per-step
         diagnostics; observe ``y[t]`` and call
         ``dtaci.update_step``.

    Parameters
    ----------
    base_model : BaseEstimator
        Sklearn-compatible regressor. Cloned at every fit
        boundary; the caller's instance is never mutated.
    dtaci : DtACI
        Online conformal primitive. Mutated in place. Reset
        during init (via ``run_online``), then state evolves
        across the full walk-forward window without further
        resets.
    X : array, shape (n, n_features)
        Feature matrix.
    y : array, shape (n,)
        1-step-ahead target series.
    initial_train_size : int, default 252
        Length of the initial training window. Must be in
        ``[1, len(y))``.
    refit_frequency : int, default 21
        Steps between base-model refits during walk-forward.
        Must be ``>= 1``.
    warmup : int, default 50
        Number of leading rows in the initial training window
        used purely for score-buffer warmup (no expert-state
        update). Must satisfy ``0 <= warmup < initial_train_size``.
    signal_filter : SignalFilter, optional
        Defaults to ``SignalFilter(min_signal_strength=0.5)``,
        matching ``backtest_alpha_model`` for behavior parity.
    portfolio_constructor : PortfolioConstructor, optional
        Defaults to ``PortfolioConstructor(method="kelly")``,
        matching ``backtest_alpha_model``.

    Returns
    -------
    results : dict
        Per-step diagnostics. All per-step lists/arrays have
        length ``n - initial_train_size``; ``refit_points`` is a
        list of ``t`` values at which a refit happened (length
        ``floor((n - initial_train_size) / refit_frequency)``).

        Keys:
          ``signals``           — list[AlphaSignal]
          ``weights``           — list[float]
          ``returns``           — list[float] (realized ``y[t]``)
          ``predictions``       — list[float] (point ``y_pred``)
          ``intervals``         — list[PredictionInterval]
          ``covered``           — list[bool]
          ``trade_mask``        — list[bool]
          ``aggregated_alpha``  — list[float] (DtACI public state
                                  captured BEFORE update_step)
          ``expert_alphas``     — list[ndarray, shape (K,)]
          ``expert_weights``    — list[ndarray, shape (K,)]
          ``weight_entropy``    — list[float]
          ``interval_width``    — list[float]
          ``refit_points``      — list[int]

    Raises
    ------
    ValueError
        If ``initial_train_size >= len(y)`` or
        ``warmup >= initial_train_size`` or ``refit_frequency < 1``.
    """
    n = len(y)
    if initial_train_size < 1 or initial_train_size >= n:
        raise ValueError(
            f"initial_train_size ({initial_train_size}) must be in [1, len(y)) = [1, {n})"
        )
    if warmup < 0 or warmup >= initial_train_size:
        raise ValueError(
            f"warmup ({warmup}) must be in [0, initial_train_size) = [0, {initial_train_size})"
        )
    if refit_frequency < 1:
        raise ValueError(f"refit_frequency ({refit_frequency}) must be >= 1")

    if signal_filter is None:
        signal_filter = SignalFilter(min_signal_strength=0.5)
    if portfolio_constructor is None:
        portfolio_constructor = PortfolioConstructor(method="kelly")

    fitted = clone(base_model).fit(X[:initial_train_size], y[:initial_train_size])

    dtaci.run_online(
        fitted,
        X[:initial_train_size],
        y[:initial_train_size],
        warmup=warmup,
    )

    results: dict[str, Any] = {
        "signals": [],
        "weights": [],
        "returns": [],
        "predictions": [],
        "intervals": [],
        "covered": [],
        "trade_mask": [],
        "aggregated_alpha": [],
        "expert_alphas": [],
        "expert_weights": [],
        "weight_entropy": [],
        "interval_width": [],
        "refit_points": [],
    }

    steps_since_refit = 0

    for t in range(initial_train_size, n):
        if steps_since_refit >= refit_frequency:
            fitted = clone(base_model).fit(X[:t], y[:t])
            steps_since_refit = 0
            results["refit_points"].append(int(t))

        interval = dtaci.predict_step(fitted, X[t : t + 1])
        y_pred = float(fitted.predict(X[t : t + 1])[0])

        signal = AlphaSignal(
            expected_return=np.asarray(interval.point, dtype=np.float64),
            lower=np.asarray(interval.lower, dtype=np.float64),
            upper=np.asarray(interval.upper, dtype=np.float64),
            alpha=interval.alpha,
        )

        filtered_signal, mask = signal_filter.apply(signal)
        weights = portfolio_constructor.construct(filtered_signal, mask)
        weight = float(weights[0]) if len(weights) else 0.0

        results["signals"].append(signal)
        results["weights"].append(weight)
        results["returns"].append(float(y[t]))
        results["predictions"].append(y_pred)
        results["intervals"].append(interval)
        results["covered"].append(bool(signal.lower[0] <= y[t] <= signal.upper[0]))
        results["trade_mask"].append(bool(mask[0]) if len(mask) else False)
        results["aggregated_alpha"].append(dtaci.aggregated_alpha)
        results["expert_alphas"].append(dtaci.expert_alphas)
        results["expert_weights"].append(dtaci.expert_weights)
        results["weight_entropy"].append(dtaci.weight_entropy)
        results["interval_width"].append(float(signal.upper[0] - signal.lower[0]))

        dtaci.update_step(float(y[t]), y_pred)
        steps_since_refit += 1

    return results
