"""P0.5 — xfail pins for six conformal fitters with shuffle-on-time-series defect.

Each test asserts that the fitter's internal train/calibration (or
train/validation-per-fold) split preserves temporal ordering on a
time-ordered input.  All six currently FAIL because the fitters use
``np.random.permutation`` or ``KFold(shuffle=True)``, which interleave
indices from different time periods.

``strict=True`` means a future refactor that accidentally satisfies the
ordering triggers XPASS → CI failure → conscious decision to flip the
flag.  This is the mechanism that prevents silent "fixes" without
integration into the AFML-correct conformal path.

Real fix deferred to the conformal-integration sprint.

Test approach
-------------
Each fitter is given ``X = [[0], [1], ..., [n-1]]`` so that ``X[idx]``
values *are* the original positional indices.  A ``_Recorder`` estimator
captures the training positions seen during ``fit()``.  After the
conformal fit, we recover which positions went to train vs calibration
(or train vs validation per fold) and assert temporal ordering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from quantcore.uncertainty.conformal.regression import (
    CrossConformalRegressor,
    CVPlusRegressor,
    SplitConformalRegressor,
)
from quantcore.uncertainty.conformal.quantile import CQRPlusRegressor, CQRRegressor
from quantcore.uncertainty.conformal.finance.var import ConformalVaR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_N = 100  # dataset size — large enough for all fitters' minimum checks
_X = np.arange(_N, dtype=np.float64).reshape(-1, 1)
_Y = np.sin(np.linspace(0, 4 * np.pi, _N))  # arbitrary continuous targets
_RETURNS = np.random.RandomState(0).normal(0.001, 0.02, _N)  # for VaR


class _Recorder(BaseEstimator):
    """Minimal sklearn-compatible estimator that records which X rows it saw."""

    def fit(self, X: Any, y: Any, **kw: Any) -> "_Recorder":
        self.X_train_ = np.asarray(X).ravel().copy()
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params: Any) -> "_Recorder":
        return self


def _assert_temporal_order_split(train_positions: np.ndarray, n: int, label: str) -> None:
    """Assert max(train) < min(cal) — the temporal-order invariant."""
    all_pos = set(range(n))
    cal_positions = sorted(all_pos - set(train_positions.astype(int)))
    assert len(cal_positions) > 0, f"{label}: empty calibration set"
    assert max(train_positions) < min(cal_positions), (
        f"{label}: temporal order violated — "
        f"max(train)={max(train_positions):.0f}, "
        f"min(cal)={min(cal_positions)}"
    )


def _assert_temporal_order_kfold(
    fold_models: list, n: int, label: str, *, via_wrapper: bool = False
) -> None:
    """Assert temporal ordering in every KFold fold."""
    for i, fm in enumerate(fold_models):
        model = fm.model if via_wrapper else fm
        train_pos = model.X_train_
        all_pos = set(range(n))
        val_pos = sorted(all_pos - set(train_pos.astype(int)))
        assert len(val_pos) > 0, f"{label} fold {i}: empty validation set"
        assert max(train_pos) < min(val_pos), (
            f"{label} fold {i}: temporal order violated — "
            f"max(train)={max(train_pos):.0f}, "
            f"min(val)={min(val_pos)}"
        )


# ---------------------------------------------------------------------------
# Tests — six xfail pins
# ---------------------------------------------------------------------------
_XFAIL_REASON = (
    "Shuffle-on-time-series defect: fit() uses sklearn KFold "
    "(shuffle=True) or np.random.permutation which violates "
    "temporal ordering. Fix deferred to "
    "the conformal-integration sprint; this xfail pins the defect "
    "so an accidental fix forces a conscious decision via XPASS-CI-fail."
)


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_split_conformal_temporal_order_preserved():
    """SplitConformalRegressor.fit (regression.py:112) — rng.permutation(n)."""
    cp = SplitConformalRegressor(_Recorder(), alpha=0.1, random_state=42)
    cp.fit(_X, _Y, calibration_fraction=0.5)
    _assert_temporal_order_split(cp.model.X_train_, _N, "SplitConformal")


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_cross_conformal_temporal_order_preserved():
    """CrossConformalRegressor.fit (regression.py:265-269) — KFold(shuffle=True)."""
    cp = CrossConformalRegressor(_Recorder(), alpha=0.1, n_folds=5, random_state=42)
    cp.fit(_X, _Y)
    _assert_temporal_order_kfold(cp._fold_models, _N, "CrossConformal")


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_cvplus_temporal_order_preserved():
    """CVPlusRegressor.fit (regression.py:557-561) — KFold(shuffle=True)."""
    cp = CVPlusRegressor(_Recorder(), alpha=0.1, n_folds=5, random_state=42)
    cp.fit(_X, _Y)
    _assert_temporal_order_kfold(cp._fold_models, _N, "CVPlus")


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_cqr_temporal_order_preserved():
    """CQRRegressor.fit (quantile.py:186) — rng.permutation(n)."""
    cp = CQRRegressor(_Recorder(), alpha=0.1, random_state=42)
    cp.fit(_X, _Y, calibration_fraction=0.5)
    # CQR wraps model in QuantileRegressorWrapper; training data is in
    # self._model_lo.model (the cloned _Recorder).
    _assert_temporal_order_split(cp._model_lo.model.X_train_, _N, "CQR")


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_cqrplus_temporal_order_preserved():
    """CQRPlusRegressor.fit (quantile.py:415-419) — KFold(shuffle=True)."""
    cp = CQRPlusRegressor(_Recorder(), alpha=0.1, n_folds=5, random_state=42)
    cp.fit(_X, _Y)
    _assert_temporal_order_kfold(cp._fold_models_lo, _N, "CQRPlus", via_wrapper=True)


@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)
def test_conformal_var_conditional_temporal_order_preserved():
    """ConformalVaR.fit_conditional (finance/var.py:217) — np.random.permutation."""
    cv = ConformalVaR(alpha=0.95)
    cv.fit_conditional(_X, _RETURNS, model=_Recorder(), calibration_fraction=0.3)
    _assert_temporal_order_split(cv._model.model.X_train_, _N, "ConformalVaR")
