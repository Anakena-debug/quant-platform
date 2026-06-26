"""Refit-path n_folds / score_function forwarding pins (S16 P15.4).

Pre-S16, ``backtest_alpha_model`` reconstructs ``ConformalAlphaModel``
at every refit boundary but silently drops ``n_folds`` and
``score_function`` from the input model — the refit path uses the
class defaults (``n_folds=5``, ``score_function=absolute_residual_score``)
regardless of what the caller's input model was configured with.
This is open_question 1 from the S15 retro
and was named in the S14 retro before that.

S16 P15.4 forwards both kwargs through the refit reconstruction at
``alpha.py:899-908``. These two tests demonstrate the bug pre-fix
(both FAIL) and the correctness post-fix (both PASS):

  * ``test_pin_refit_preserves_n_folds_cv_branch`` — monkeypatches
    ``KFold.__init__`` to capture ``n_splits`` per call; asserts
    every observed value equals the caller-provided ``n_folds``.
  * ``test_pin_refit_preserves_score_function_split_branch`` —
    counts invocations of a custom ``score_function`` callable;
    asserts at least one in-loop refit calibration invokes it
    (i.e., the count exceeds the pre-fix "initial fit only" count
    of 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from quantcore.uncertainty.conformal.finance import (
    ConformalAlphaModel,
    backtest_alpha_model,
)


def _toy_synthetic(
    seed: int = 7, n: int = 200
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X.sum(axis=1) + 0.3 * rng.standard_normal(n)
    return X, y


def test_pin_refit_preserves_n_folds_cv_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``ConformalAlphaModel(method='cv', n_folds=7)`` must use 7
    folds at every refit, not silently revert to the class default 5.

    Pre-fix, the refit path reconstructs ``ConformalAlphaModel`` without
    forwarding ``n_folds``, so subsequent refits use ``n_folds=5``.
    Post-fix, the value is preserved across the walk-forward."""
    X, y = _toy_synthetic()

    observed_n_splits: list[int] = []
    orig_init = KFold.__init__

    def spied_init(self: KFold, n_splits: int = 5, *args: Any, **kwargs: Any) -> None:
        observed_n_splits.append(int(n_splits))
        orig_init(self, n_splits, *args, **kwargs)

    monkeypatch.setattr(KFold, "__init__", spied_init)

    model = ConformalAlphaModel(
        model=LinearRegression(),
        method="cv",
        n_folds=7,
    )
    backtest_alpha_model(
        model,
        X,
        y,
        initial_train_size=100,
        refit_frequency=21,
    )

    assert len(observed_n_splits) >= 2, (
        f"expected at least 2 KFold constructions (initial fit + ≥1 refit); got {observed_n_splits}"
    )
    assert all(s == 7 for s in observed_n_splits), (
        f"refit dropped n_folds — observed n_splits values: {observed_n_splits}"
    )


def test_pin_refit_preserves_score_function_split_branch() -> None:
    """A ``ConformalAlphaModel(method='split', score_function=custom)``
    must invoke ``custom`` at every refit calibration, not just the
    initial fit.

    Pre-fix, the refit path reconstructs ``ConformalAlphaModel`` without
    forwarding ``score_function``, which resolves to the default
    ``absolute_residual_score`` at line 227 of ``alpha.py``. Counter
    stays at 1 (initial fit only).
    Post-fix, every refit's calibration invokes the custom function."""
    X, y = _toy_synthetic()

    counter = {"calls": 0}

    def custom_score(
        y_true: NDArray[np.floating[Any]],
        y_pred: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        counter["calls"] += 1
        return np.abs(y_true - y_pred)

    model = ConformalAlphaModel(
        model=LinearRegression(),
        method="split",
        score_function=custom_score,
    )
    backtest_alpha_model(
        model,
        X,
        y,
        initial_train_size=100,
        refit_frequency=21,
    )

    assert counter["calls"] >= 2, (
        f"custom_score was invoked {counter['calls']} times — refit "
        "reset to default absolute_residual_score (the initial fit's "
        "calibration is the only one using the custom function)"
    )
