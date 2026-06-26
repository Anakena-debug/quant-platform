"""
Time-series-safe conformal prediction
=====================================

Drop-in replacement for the shuffled split-conformal code in
``regression.py`` / ``quantile.py`` / ``classification.py``.  The
standard split-conformal guarantee
(Vovk et al. 2005, Lei et al. 2018) requires **exchangeability** of
calibration and test data, which fails for financial time series
because of serial dependence and non-stationarity.

This module provides four *ordered-data* alternatives.  None of them
recovers the IID split-conformal finite-sample bound in the presence
of serial dependence; each trades assumptions for different failure
modes:

    +--------------+-------------------------------+----------------------+
    | method       | Assumption                    | Coverage property    |
    +==============+===============================+======================+
    | ``"split"``  | Temporal split (no shuffle).  | Marginal 1−α only if |
    |              | Calibration residuals approx. | residuals are        |
    |              | exchangeable.                 | (approximately)      |
    |              |                               | exchangeable.  NOT   |
    |              |                               | guaranteed under     |
    |              |                               | drift.               |
    +--------------+-------------------------------+----------------------+
    | ``"rolling"``| Local stationarity over last  | Empirical; widely    |
    |              | W bars.                       | used but no formal   |
    |              |                               | finite-sample bound. |
    +--------------+-------------------------------+----------------------+
    | ``"block"``  | β-mixing residuals; block     | Asymptotic 1−α       |
    |              | b = ⌈T^{1/3}⌉.                | under β-mixing       |
    |              |                               | (Chernozhukov et al. |
    |              |                               | 2018).  Finite-      |
    |              |                               | sample behaviour     |
    |              |                               | depends on mixing    |
    |              |                               | rate.                |
    +--------------+-------------------------------+----------------------+
    | ``"aci"``    | Any data; gradient method on  | *Long-run* empirical |
    |              | α_t (Gibbs & Candès 2021).    | coverage → 1−α as    |
    |              |                               | T→∞ (Thm. 1).  No    |
    |              |                               | finite-window        |
    |              |                               | guarantee; interval  |
    |              |                               | can degenerate to    |
    |              |                               | ∅ or ℝ under shocks. |
    +--------------+-------------------------------+----------------------+

Score functions supported:

    ``"abs_residual"``           s(x, y) = |y − f(x)|
    ``"signed_residual"``        Asymmetric:  runs two calibrations
                                 at level α/2 on ``y − f(x)`` giving a
                                 possibly-asymmetric interval (AFML-
                                 style upper / lower PI).
    ``"studentized_abs_residual"``
                                 s(x, y) = |y − f(x)| / σ̂(x),
                                 where σ̂(x) is returned by
                                 ``estimator.predict_scale(X)``.
                                 Recommended for heteroskedastic
                                 financial returns (Lei et al. 2018,
                                 §4.2 — "locally weighted" conformal).
    ``"cqr"``                    s(x, y) = max(q_lo − y, y − q_hi)
                                 (needs ``.predict_quantiles``).
                                 Quantile non-crossing is enforced.
    callable                     User-supplied
                                 ``s(y_true, preds_dict) → ndarray``.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005).  *Algorithmic Learning
in a Random World*.  Springer.  doi:10.1007/b106715

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
(2018).  "Distribution-free predictive inference for regression."
*JASA* 113(523), 1094–1111.  doi:10.1080/01621459.2017.1307116

Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2018).  "Exact and robust
conformal inference methods for predictive machine learning with
dependent data."  *COLT 2018 / arXiv:1802.06300.*

Gibbs, I., & Candès, E. (2021).  "Adaptive conformal inference under
distribution shift."  *NeurIPS 2021*, arXiv:2106.00170.

Romano, Y., Patterson, E., & Candès, E. (2019).  "Conformalized
quantile regression."  *NeurIPS 2019*, arXiv:1905.03222.

Barber, R. F., Candès, E., Ramdas, A., Tibshirani, R. J. (2023).
"Conformal prediction beyond exchangeability."  *Ann. Statist.*
51(2), 816–845.  doi:10.1214/23-AOS2276
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "TimeSeriesConformal",
    "ACIRegressor",
    "BlockConformal",
    "quantile_hi",
    "finite_sample_quantile",
]

# -----------------------------------------------------------------------------
# Quantile helpers
# -----------------------------------------------------------------------------


def finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal (ceil) quantile.

    Returns the ``⌈(n+1)(1-α)⌉ / n`` order statistic — the canonical
    *finite-sample* upper bound in split conformal (Lei et al. 2018,
    Theorem 2.2).  When n is small this differs meaningfully from
    ``np.quantile``.

    Parameters
    ----------
    scores : np.ndarray
        Non-conformity scores on calibration set (any order).
    alpha : float
        Miscoverage level.  Returned interval covers 1 − α.

    Returns
    -------
    float
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return np.inf
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    # np.quantile uses linear interpolation; we want the *discrete*
    # upper-ceil so use "higher" method for the exact guarantee.
    return float(np.quantile(s, q_level, method="higher"))


quantile_hi = finite_sample_quantile  # alias used in older code


def _default_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """|y - ŷ|, default absolute-residual score."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------


@dataclass
class TimeSeriesConformal:
    """Time-series-safe conformal regressor / quantile-regressor wrapper.

    Parameters
    ----------
    estimator : object
        Must implement ``fit(X, y)`` and ``predict(X)``.  For CQR, must
        additionally implement ``predict_quantiles(X) → (q_lo, q_hi)``.
    alpha : float, default 0.1
        Target miscoverage.  Produces a 1 − α PI.
    method : {"split", "rolling", "block", "aci"}
    window : int, default 500
        Rolling-window length for ``method="rolling"``.  Ignored
        otherwise.
    block_size : int, optional
        Block size for ``method="block"``.  Default ``⌈T_cal^{1/3}⌉``.
    aci_gamma : float, default 0.005
        Step size for ACI updates.  0.005–0.05 typical; too large →
        noisy α; too small → slow adaptation.
    score : str or callable, default ``"abs_residual"``
        Non-conformity score.  Options: ``"abs_residual"``,
        ``"signed_residual"``, ``"cqr"``, or
        ``fn(y_true, preds_dict) → ndarray``.

    Attributes
    ----------
    scores_ : np.ndarray
        Non-conformity scores on the calibration set (chronological).
    q_hat_ : float
        Current calibrated quantile (for split/block).
    alpha_t_ : float
        Current adaptive alpha (for ACI).
    n_cal_ : int

    Notes
    -----
    *   For stateful online usage (live trading), call
        ``update(x_new, y_new)`` after each observation to append the
        new non-conformity score.  For ACI, this also updates
        ``alpha_t_``.
    *   No data is shuffled at any point.
    """

    estimator: object
    alpha: float = 0.1
    method: Literal["split", "rolling", "block", "aci"] = "rolling"
    window: int = 500
    block_size: Optional[int] = None
    aci_gamma: float = 0.005
    aci_window: Optional[int] = None  # rolling score buffer for ACI
    score: Union[str, Callable] = "abs_residual"

    # populated in fit / update
    scores_: np.ndarray = field(default_factory=lambda: np.empty(0))
    # For asymmetric "signed_residual" we keep separate upper/lower
    # calibrated quantiles.  For "abs_residual"/"cqr"/callable, q_lo_
    # is unused and q_hi_ equals q_hat_.
    q_hat_: float = field(default=np.inf)
    q_hi_score_: float = field(default=np.inf)  # signed: upper α/2
    q_lo_score_: float = field(default=-np.inf)  # signed: lower α/2
    alpha_t_: float = field(default=0.1)
    n_cal_: int = 0

    def __post_init__(self) -> None:
        # -- parameter validation -------------------------------------
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1); got {self.alpha}.")
        if self.method not in ("split", "rolling", "block", "aci"):
            raise ValueError(f"method must be one of split/rolling/block/aci; got '{self.method}'.")
        if self.window is not None and self.window < 2:
            raise ValueError("window must be ≥ 2.")
        if self.block_size is not None and self.block_size < 2:
            raise ValueError("block_size must be ≥ 2.")
        if not (0.0 < self.aci_gamma < 1.0):
            raise ValueError(f"aci_gamma must be in (0,1); got {self.aci_gamma}.")
        if self.aci_window is not None and self.aci_window < 2:
            raise ValueError("aci_window must be ≥ 2 or None.")
        if isinstance(self.score, str) and self.score not in (
            "abs_residual",
            "signed_residual",
            "studentized_abs_residual",
            "cqr",
        ):
            raise ValueError(f"unknown score '{self.score}'.")
        # sync alpha_t_ with alpha (dataclass default is a literal 0.1)
        self.alpha_t_ = float(self.alpha)

    # ---------------------------------------------------------------
    # Score helpers
    # ---------------------------------------------------------------
    def _score_from_preds(self, y_true: np.ndarray, preds: dict) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        if callable(self.score):
            return np.asarray(self.score(y_true, preds), dtype=float)
        if self.score == "abs_residual":
            return np.abs(y_true - preds["point"])
        if self.score == "signed_residual":
            # Keep the *signed* residual.  Asymmetric calibration is
            # performed downstream in ``_refresh_q_hat`` which takes
            # the upper 1−α/2 quantile and the lower α/2 quantile
            # separately — this restores the conformal guarantee
            # that plain mirrored-|r| symmetry destroys.
            return y_true - preds["point"]
        if self.score == "studentized_abs_residual":
            scale = preds.get("scale", None)
            if scale is None:
                raise ValueError("studentized_abs_residual requires estimator.predict_scale(X).")
            eps = 1e-8
            return np.abs(y_true - preds["point"]) / np.maximum(scale, eps)
        if self.score == "cqr":
            q_lo, q_hi = preds["q_lo"], preds["q_hi"]
            return np.maximum(q_lo - y_true, y_true - q_hi)
        raise ValueError(f"unknown score '{self.score}'.")

    def _predict_all(self, X) -> dict:
        preds = {}
        if self.score == "cqr":
            q_lo, q_hi = self.estimator.predict_quantiles(X)  # type: ignore[attr-defined]
            q_lo = np.asarray(q_lo, dtype=float)
            q_hi = np.asarray(q_hi, dtype=float)
            # Enforce non-crossing (Chernozhukov et al. 2010; mandatory
            # for CQR to give a well-defined interval).
            q_lo_c = np.minimum(q_lo, q_hi)
            q_hi_c = np.maximum(q_lo, q_hi)
            preds["q_lo"] = q_lo_c
            preds["q_hi"] = q_hi_c
            preds["point"] = 0.5 * (q_lo_c + q_hi_c)
        else:
            preds["point"] = np.asarray(
                self.estimator.predict(X),
                dtype=float,  # type: ignore[attr-defined]
            )
            if self.score == "studentized_abs_residual":
                if not hasattr(self.estimator, "predict_scale"):
                    raise AttributeError(
                        "studentized_abs_residual requires the "
                        "estimator to expose .predict_scale(X)."
                    )
                preds["scale"] = np.asarray(
                    self.estimator.predict_scale(X),
                    dtype=float,  # type: ignore[attr-defined]
                )
        return preds

    # ---------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        *,
        cal_size: Optional[Union[int, float]] = None,
        refit_estimator: bool = True,
    ) -> "TimeSeriesConformal":
        """Temporal fit: first portion = train, second = calibration.

        Parameters
        ----------
        X : array-like / DataFrame
        y : array-like / Series
        cal_size : int or float in (0,1), optional
            Size of the calibration fold.  Default 0.2·len(X).
            **Must be the most recent portion** — NO SHUFFLING.
        refit_estimator : bool, default True
            If False, assume the estimator is already fitted on
            an earlier, disjoint set; in that case ``X, y`` are
            treated entirely as calibration data.
        """
        n = len(X)
        if cal_size is None:
            cal_n = max(50, int(0.2 * n))
        elif isinstance(cal_size, float):
            cal_n = max(2, int(cal_size * n))
        else:
            cal_n = int(cal_size)
        if cal_n >= n and refit_estimator:
            raise ValueError("cal_size must leave room for training.")

        if refit_estimator:
            X_tr = X[:-cal_n] if not hasattr(X, "iloc") else X.iloc[:-cal_n]
            y_tr = y[:-cal_n] if not hasattr(y, "iloc") else y.iloc[:-cal_n]
            self.estimator.fit(X_tr, y_tr)  # type: ignore[attr-defined]
            X_cal = X[-cal_n:] if not hasattr(X, "iloc") else X.iloc[-cal_n:]
            y_cal = y[-cal_n:] if not hasattr(y, "iloc") else y.iloc[-cal_n:]
        else:
            X_cal, y_cal = X, y

        preds = self._predict_all(X_cal)
        scores = self._score_from_preds(np.asarray(y_cal, dtype=float), preds)
        self.scores_ = np.asarray(scores, dtype=float)
        self.n_cal_ = self.scores_.size
        self.alpha_t_ = float(self.alpha)
        self._refresh_q_hat()
        return self

    # ---------------------------------------------------------------
    def _slice_for_method(self) -> np.ndarray:
        """Return the score buffer slice used by the active method."""
        s = self.scores_
        if self.method == "rolling":
            w = min(self.window, s.size)
            return s[-w:]
        if self.method == "aci" and self.aci_window is not None:
            w = min(self.aci_window, s.size)
            return s[-w:]
        return s

    def _refresh_q_hat(self) -> None:
        """Recompute calibrated quantile(s) per the active method.

        Branches
        --------
        *   ``signed_residual``: two-sided asymmetric.  The upper
            bound uses the ``1 − α/2`` finite-sample quantile of
            ``y − ŷ`` (positive tail), the lower uses the ``α/2``
            quantile (negative tail).  This yields a *proper* 1−α
            conformal interval (Romano et al. 2019, §2.2) and is
            NOT symmetric in general.
        *   ``block``: block-max reduction, then finite-sample quantile.
        *   Otherwise: single-tail finite-sample quantile on the score
            buffer slice determined by the method.
        """
        s_slice = self._slice_for_method()

        if self.score == "signed_residual":
            # Asymmetric calibration: two separate tails at α/2.
            half_alpha = self.alpha_t_ / 2.0 if self.method == "aci" else self.alpha / 2.0
            # Upper: quantile of signed residuals at 1 − α/2 (rhs tail).
            # Equivalent to applying the ceil-quantile to (y-ŷ).
            self.q_hi_score_ = finite_sample_quantile(s_slice, half_alpha)
            # Lower: for the left tail, use the finite-sample quantile
            # of the *negated* scores — which corresponds to
            # choosing the α/2 *lower* ceil-quantile.
            self.q_lo_score_ = -finite_sample_quantile(-s_slice, half_alpha)
            # q_hat_ kept as a scalar "width" for legacy callers
            # (max of |upper|, |lower|) — but predict() uses the
            # asymmetric fields.
            self.q_hat_ = float(max(abs(self.q_hi_score_), abs(self.q_lo_score_)))
            return

        if self.method == "block":
            b = self.block_size or max(2, int(np.ceil(s_slice.size ** (1 / 3))))
            n = s_slice.size
            n_blocks = n // b
            if n_blocks < 4:
                self.q_hat_ = finite_sample_quantile(s_slice, self.alpha)
            else:
                block_max = np.array([s_slice[i * b : (i + 1) * b].max() for i in range(n_blocks)])
                self.q_hat_ = finite_sample_quantile(block_max, self.alpha)
            self.q_hi_score_ = self.q_hat_
            self.q_lo_score_ = -self.q_hat_
            return

        # split / rolling / aci (non-signed scores)
        alpha_eff = self.alpha_t_ if self.method == "aci" else self.alpha
        self.q_hat_ = finite_sample_quantile(s_slice, alpha_eff)
        self.q_hi_score_ = self.q_hat_
        self.q_lo_score_ = -self.q_hat_

    # ---------------------------------------------------------------
    # predict
    # ---------------------------------------------------------------
    def predict(
        self,
        X,
        return_interval: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Point + prediction interval.

        Returns
        -------
        (y_hat, lo, hi) if return_interval, else y_hat.
        """
        preds = self._predict_all(X)
        y_hat = preds["point"]
        if not return_interval:
            return y_hat
        q = self.q_hat_
        if self.score == "cqr":
            lo = preds["q_lo"] - q
            hi = preds["q_hi"] + q
        elif self.score == "signed_residual":
            # Asymmetric: y_hat + q_lo_score_ (negative) and
            # y_hat + q_hi_score_ (positive).
            lo = y_hat + self.q_lo_score_
            hi = y_hat + self.q_hi_score_
        elif self.score == "studentized_abs_residual":
            # Width scales locally with predicted σ̂(x).
            scale = preds.get("scale")
            lo = y_hat - q * scale
            hi = y_hat + q * scale
        else:  # abs_residual or callable
            lo = y_hat - q
            hi = y_hat + q
        return y_hat, lo, hi

    # ---------------------------------------------------------------
    # update (online)
    # ---------------------------------------------------------------
    def update(
        self,
        x_new,
        y_new,
        *,
        refit_after: Optional[int] = None,
        refit_X=None,
        refit_y=None,
    ) -> None:
        """Append one observation to the non-conformity score buffer.

        For ``method="aci"`` this additionally updates ``alpha_t_``:

            err_t   = 1  if  y_new ∉ [lo, hi]  else 0
            α_{t+1} = α_t + γ · (α  −  err_t)

        (Gibbs & Candès 2021, Algorithm 1.)

        Parameters
        ----------
        x_new : 1-row feature vector / row-DataFrame.
        y_new : scalar true value.
        refit_after, refit_X, refit_y :
            Optional: refit the underlying estimator every N updates,
            using a provided window of past (X, y).  If omitted the
            estimator is *never* refit here — caller handles refresh
            cadence.
        """
        # ---- Input shaping ---------------------------------------
        # Accept: scalar row, 1-D ndarray, pandas Series (single row),
        # pandas DataFrame (1 row), or 2-D ndarray (1 row).
        if isinstance(x_new, pd.DataFrame):
            X2 = x_new.iloc[[0]] if len(x_new) >= 1 else x_new
        elif isinstance(x_new, pd.Series):
            X2 = x_new.to_frame().T
        else:
            arr = np.asarray(x_new)
            X2 = arr.reshape(1, -1) if arr.ndim == 1 else arr

        y_new = float(y_new)
        preds = self._predict_all(X2)
        score_new = float(self._score_from_preds(np.array([y_new]), preds)[0])

        if self.method == "aci":
            # Use *current* calibrated quantile(s) (pre-update) for
            # the coverage test.
            q = self.q_hat_
            if self.score == "cqr":
                lo = float(preds["q_lo"][0]) - q
                hi = float(preds["q_hi"][0]) + q
            elif self.score == "signed_residual":
                lo = float(preds["point"][0]) + self.q_lo_score_
                hi = float(preds["point"][0]) + self.q_hi_score_
            elif self.score == "studentized_abs_residual":
                scale_0 = float(np.atleast_1d(preds["scale"])[0])
                lo = float(preds["point"][0]) - q * scale_0
                hi = float(preds["point"][0]) + q * scale_0
            else:
                lo = float(preds["point"][0]) - q
                hi = float(preds["point"][0]) + q
            err = int(y_new < lo or y_new > hi)
            self.alpha_t_ = float(
                np.clip(
                    self.alpha_t_ + self.aci_gamma * (self.alpha - err),
                    1e-4,
                    1 - 1e-4,
                )
            )

        self.scores_ = np.append(self.scores_, score_new)
        self.n_cal_ += 1

        # Optional periodic refit
        if refit_after is not None and refit_X is not None:
            if self.n_cal_ % refit_after == 0:
                self.estimator.fit(refit_X, refit_y)  # type: ignore[attr-defined]

        self._refresh_q_hat()


# -----------------------------------------------------------------------------
# Convenience aliases — legacy naming consistent with the repository
# -----------------------------------------------------------------------------


class ACIRegressor(TimeSeriesConformal):
    """Shortcut for Gibbs & Candès (2021) ACI with abs_residual score.

    Parameters
    ----------
    estimator : object
    alpha : float
    gamma : float
        ACI step size γ.
    aci_window : int, optional
        If supplied, the score buffer used to compute the calibrated
        quantile is restricted to the most recent ``aci_window`` scores.
        Essential under regime drift — the unbounded buffer otherwise
        freezes around stale residuals and forces γ to fight a
        slowly-moving distribution.  50-500 is a reasonable range
        for hourly/daily returns.
    """

    def __init__(
        self,
        estimator,
        alpha: float = 0.1,
        gamma: float = 0.005,
        aci_window: Optional[int] = None,
    ):
        super().__init__(
            estimator=estimator,
            alpha=alpha,
            method="aci",
            aci_gamma=gamma,
            aci_window=aci_window,
            score="abs_residual",
        )


class BlockConformal(TimeSeriesConformal):
    """Shortcut for Chernozhukov et al. (2018) block conformal."""

    def __init__(
        self,
        estimator,
        alpha: float = 0.1,
        block_size: Optional[int] = None,
        score: Union[str, Callable] = "abs_residual",
    ):
        super().__init__(
            estimator=estimator,
            alpha=alpha,
            method="block",
            block_size=block_size,
            score=score,
        )
