"""Dynamic-tuning Adaptive Conformal Inference (DtACI).

Multi-expert ACI with EWA aggregation over per-expert pinball loss.

References
----------
Gibbs & Candès (2024) "Conformal Inference for Online Prediction
with Arbitrary Distribution Shifts" — DtACI. The default
``gammas=(0.001, 0.005, 0.02, 0.08)`` mirrors the empirical setup
in §5; the grid spans ~1.5 orders of magnitude so EWA aggregation
has meaningful expert diversity at any regime cadence the operator
might encounter at deployment.

Implementation note
-------------------
DtACI is implemented as a self-contained primitive that maintains
its own score buffer and applies the ACI update rule per-expert,
rather than literally composing K instances of
``AdaptiveConformalInference``. Reasons:

  - The aggregated α requires querying the score quantile at
    ``∑_k w_k α_k`` against a single calibration pool. Composing
    K ACI instances would either need K duplicated score buffers
    (wasteful) or reaching into ``_compute_quantile`` /
    ``_score_buffer`` (private-state coupling).

  - The S13 stop-gate forbids edits to ``timeseries.py``'s ACI
    body, which precludes adding a public ``quantile_at(α)``
    accessor that would make composition clean.

  - The bitwise-equality pin (``DtACI(gammas=(γ,γ), eta=0)`` matches
    ``AdaptiveConformalInference(gamma=γ)``) verifies the rule is
    implemented identically to ACI; that's the semantic guarantee
    the "wrapper around ACI" framing was buying.

Logged in S13 deviation log alongside this docstring.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from quantcore.uncertainty.conformal.base import PredictionInterval
from quantcore.uncertainty.conformal.diagnostics import normalized_entropy
from quantcore.uncertainty.conformal.scores import (
    absolute_residual_score,
    compute_conformal_quantile,
)


class DtACI:
    """Dynamic-tuning Adaptive Conformal Inference.

    Runs ``K`` experts in parallel, each tracking its own α via
    the ACI update rule with its own step size γ_k. The aggregated
    α is a weighted average over experts; expert weights update by
    exponentially weighted average (EWA) on per-step pinball loss.
    A floor ``w_min`` prevents any single expert's weight from
    collapsing to numerical zero.

    The aggregated α at step t is::

        α_t = ∑_k w_t^(k) · α_t^(k)

    Each expert k runs the ACI update::

        α_{t+1}^(k) = α_t^(k) + γ_k · (α_target - err_t^(k))

    where ``err_t^(k)`` is 1 if y_t was outside expert k's interval
    (formed at expert-k's α), 0 otherwise. Weights update via::

        w_{t+1}^(k) ∝ w_t^(k) · exp(-η · ℓ_t^(k))

    with the asymmetric pinball loss::

        ℓ_t^(k) = α · (y_t - upper_t^(k))_+ + (1-α) · (lower_t^(k) - y_t)_+

    γ-grid validation is enforced at construction (NOT first
    update): K ≥ 2 (single-γ → use AdaptiveConformalInference
    directly), all γ_k in (0, 1), monotone-sorted ascending. Bad
    grids raise ValueError naming the offending value.

    Default ``gammas=(0.001, 0.005, 0.02, 0.08)`` is from
    Gibbs & Candès 2024 §5.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate.
    gammas : tuple of float
        Step sizes for each expert. Must satisfy K = len(gammas)
        ≥ 2, all entries in (0, 1), monotone-sorted ascending.
    eta : float
        EWA learning rate. ``eta=0`` disables expert reweighting
        (weights remain at 1/K throughout); used by the
        bitwise-equality pin against single-γ ACI.
    w_min : float
        Floor for individual expert weights. Applied AFTER
        renormalization. Must satisfy K · w_min ≤ 1; the
        renormalization step otherwise cannot succeed.
    window_size : int
        Rolling window size for score-quantile computation.
    score_function : callable, optional
        Nonconformity score function. Defaults to
        ``absolute_residual_score``.
    clip_alpha : bool
        Clip per-expert α to [alpha_min, alpha_max].
    alpha_min : float
        Lower clip on per-expert α.
    alpha_max : float
        Upper clip on per-expert α.

    Attributes
    ----------
    expert_alphas : np.ndarray, shape (K,)
        Current per-expert α values.
    expert_weights : np.ndarray, shape (K,)
        Current per-expert weights (sum to 1, all ≥ w_min).
    aggregated_alpha : float
        Weight-aggregated α. Used as the target miscoverage rate
        for the prediction interval issued at each step.
    weight_entropy : float
        Normalized Shannon entropy of expert_weights, in [0, 1].
        Per the 2026-04-29 conformal-stack review failure-mode
        table, ``weight_entropy < 0.2`` signals expert collapse.
        Dispatches to ``diagnostics.normalized_entropy``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gammas: tuple[float, ...] = (0.001, 0.005, 0.02, 0.08),
        eta: float = 0.5,
        w_min: float = 0.01,
        window_size: int = 100,
        score_function: Callable[..., NDArray[np.floating[Any]]] | None = None,
        clip_alpha: bool = True,
        alpha_min: float = 0.001,
        alpha_max: float = 0.5,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        gammas = tuple(float(g) for g in gammas)
        K = len(gammas)
        if K < 2:
            raise ValueError(
                f"DtACI requires K >= 2 experts (got K={K} in "
                f"gammas={gammas}). For a single γ, use "
                f"AdaptiveConformalInference directly — DtACI's "
                f"value-add is multi-expert EWA aggregation."
            )
        for g in gammas:
            if not 0.0 < g < 1.0:
                raise ValueError(f"all gammas must be in (0, 1); got {g} in gammas={gammas}")
        if list(gammas) != sorted(gammas):
            raise ValueError(
                f"gammas must be monotone-sorted ascending; got "
                f"{gammas} (sort to {tuple(sorted(gammas))})"
            )

        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        if not 0.0 <= w_min < 1.0 / K:
            # K * w_min must be < 1 for the floor + renormalize step
            # to produce a well-defined distribution. We require strict
            # < because K * w_min == 1 means the floor pins all weights
            # uniformly with no information from EWA.
            raise ValueError(
                f"w_min must be in [0, 1/K) = [0, {1.0 / K}); got w_min={w_min} with K={K}"
            )
        if window_size < 10:
            raise ValueError(f"window_size must be >= 10, got {window_size}")

        self.target_alpha = alpha
        self.gammas = gammas
        self._gammas_arr = np.asarray(gammas, dtype=np.float64)
        self.eta = eta
        self.w_min = w_min
        self.window_size = window_size
        self.score_function = score_function or absolute_residual_score
        self.clip_alpha = clip_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Per-expert α state, all initialized at target.
        self._expert_alphas = np.full(K, alpha, dtype=np.float64)
        # Uniform initial weights.
        self._expert_weights = np.full(K, 1.0 / K, dtype=np.float64)
        # Shared score buffer.
        self._score_buffer: list[float] = []

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def expert_alphas(self) -> NDArray[np.floating[Any]]:
        """Current per-expert α values; shape (K,)."""
        return self._expert_alphas.copy()

    @property
    def expert_weights(self) -> NDArray[np.floating[Any]]:
        """Current per-expert weights; shape (K,), sums to 1."""
        return self._expert_weights.copy()

    @property
    def aggregated_alpha(self) -> float:
        """Weight-aggregated α: ``∑_k w_k · α_k``.

        Clipped to (1e-6, 1 - 1e-6) for numerical safety in the
        downstream quantile computation. The clip range is wider
        than the per-expert clip so it never bites under normal
        operation.
        """
        agg = float(np.dot(self._expert_weights, self._expert_alphas))
        return float(np.clip(agg, 1e-6, 1.0 - 1e-6))

    @property
    def weight_entropy(self) -> float:
        """Normalized expert-weight entropy via
        ``diagnostics.normalized_entropy``. < 0.2 signals collapse.
        Dispatch contract pinned in test_dtaci_invariants.py."""
        return normalized_entropy(self._expert_weights)

    @property
    def n_scores(self) -> int:
        """Length of the score buffer (calibration pool size)."""
        return len(self._score_buffer)

    def reset(self) -> None:
        """Reset to initial state (uniform weights, α at target,
        empty score buffer)."""
        K = len(self.gammas)
        self._expert_alphas = np.full(K, self.target_alpha, dtype=np.float64)
        self._expert_weights = np.full(K, 1.0 / K, dtype=np.float64)
        self._score_buffer = []

    # ------------------------------------------------------------------
    # Internal: quantile at a given α from the shared score buffer.
    # ------------------------------------------------------------------

    def _quantile_at(self, alpha: float) -> float:
        """Score quantile at the given α, mirroring ACI's
        ``_compute_quantile`` exactly so the bitwise-equality pin
        holds.

        Returns inf when the score buffer has fewer than 10 entries
        (matches ACI's warmup behavior).
        """
        if len(self._score_buffer) < 10:
            return float("inf")
        recent = np.array(self._score_buffer[-self.window_size :])
        return compute_conformal_quantile(recent, alpha)

    # ------------------------------------------------------------------
    # Online step API: predict_step then update_step.
    # ------------------------------------------------------------------

    def predict_step(
        self,
        model: BaseEstimator,
        X_t: NDArray[np.floating[Any]],
    ) -> PredictionInterval:
        """Issue a prediction interval at the aggregated α.

        Returns a single ``PredictionInterval`` (not per-expert).
        Per-expert intervals are computed internally and used by
        ``update_step`` for the EWA pinball loss; they are not
        exposed in the API surface to keep the contract narrow.
        """
        X_t = np.atleast_2d(X_t)
        y_pred = model.predict(X_t)

        agg_alpha = self.aggregated_alpha
        q_agg = self._quantile_at(agg_alpha)

        return PredictionInterval(
            lower=y_pred - q_agg,
            upper=y_pred + q_agg,
            point=y_pred,
            alpha=agg_alpha,
        )

    def update_step(
        self,
        y_true: float,
        y_pred: float,
    ) -> None:
        """Update expert α values and weights after observing y_true.

        Each expert's err is computed against ITS OWN interval at
        ITS OWN α — not the aggregated interval. This is what gives
        DtACI its diversity: experts disagree on whether y was
        covered, and the EWA reweights toward experts whose
        intervals were closer to right.

        Parameters
        ----------
        y_true : float
            Observed true value at this step.
        y_pred : float
            Model point prediction at this step (used to form
            per-expert intervals via ``y_pred ± q_k``).
        """
        # Per-expert α-update via ACI rule. Use score buffer at the
        # current state (BEFORE adding y_true's score; this matches
        # ACI's update_step ordering).
        K = len(self.gammas)
        per_expert_q = np.array([self._quantile_at(self._expert_alphas[k]) for k in range(K)])
        per_expert_lower = y_pred - per_expert_q
        per_expert_upper = y_pred + per_expert_q
        # err_k = 1 if y_true outside [lower_k, upper_k] else 0
        per_expert_err = ((y_true < per_expert_lower) | (y_true > per_expert_upper)).astype(
            np.float64
        )
        # ACI update: α_{t+1} = α_t + γ * (α_target - err)
        new_alphas = self._expert_alphas + self._gammas_arr * (self.target_alpha - per_expert_err)
        if self.clip_alpha:
            new_alphas = np.clip(new_alphas, self.alpha_min, self.alpha_max)
        self._expert_alphas = new_alphas

        # EWA reweighting via per-expert pinball loss.
        # ℓ_k = α · (y - u_k)_+ + (1-α) · (l_k - y)_+
        over_upper = np.maximum(0.0, y_true - per_expert_upper)
        under_lower = np.maximum(0.0, per_expert_lower - y_true)
        losses = self.target_alpha * over_upper + (1.0 - self.target_alpha) * under_lower
        # Replace inf-quantile losses (warmup) with 0 — no expert
        # has issued a real interval yet, so EWA shouldn't penalize
        # them differentially.
        losses = np.where(np.isfinite(losses), losses, 0.0)
        # exp(-η · ℓ) renormalize, applied multiplicatively to
        # current weights.
        new_weights = self._expert_weights * np.exp(-self.eta * losses)
        s = new_weights.sum()
        if s <= 0.0 or not np.isfinite(s):
            # Numerical degeneracy: revert to uniform.
            new_weights = np.full(K, 1.0 / K, dtype=np.float64)
        else:
            new_weights = new_weights / s
        # Apply floor and renormalize. Note: K · w_min < 1 was
        # enforced at construction so the renormalize step is well-
        # defined.
        new_weights = np.maximum(new_weights, self.w_min)
        new_weights = new_weights / new_weights.sum()
        self._expert_weights = new_weights

        # Add new score to shared buffer.
        score = float(self.score_function(np.array([y_true]), np.array([y_pred]))[0])
        self._score_buffer.append(score)

    # ------------------------------------------------------------------
    # Convenience: full online run with warmup phase.
    # ------------------------------------------------------------------

    def run_online(
        self,
        model: BaseEstimator,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        warmup: int = 50,
    ) -> tuple[
        list[PredictionInterval],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
    ]:
        """Run DtACI online over a full sequence; mirrors ACI's
        ``run_online`` shape.

        Returns
        -------
        intervals : list of PredictionInterval
            One interval per online step (post-warmup).
        agg_alpha_traj : np.ndarray, shape (T - warmup,)
            Aggregated α at each online step.
        expert_alpha_traj : np.ndarray, shape (T - warmup, K)
            Per-expert α at each online step.
        weight_traj : np.ndarray, shape (T - warmup, K)
            Per-expert weights at each online step.
        """
        T = len(y)
        if warmup >= T:
            raise ValueError(f"warmup ({warmup}) must be less than T ({T})")

        self.reset()
        intervals: list[PredictionInterval] = []
        K = len(self.gammas)
        agg_alpha_traj = np.zeros(T - warmup, dtype=np.float64)
        expert_alpha_traj = np.zeros((T - warmup, K), dtype=np.float64)
        weight_traj = np.zeros((T - warmup, K), dtype=np.float64)

        # Warmup: accumulate scores without issuing intervals or
        # updating expert state. Mirrors ACI's run_online warmup.
        for t in range(warmup):
            y_pred_t = float(model.predict(X[t : t + 1])[0])
            score = float(self.score_function(np.array([y[t]]), np.array([y_pred_t]))[0])
            self._score_buffer.append(score)

        # Online phase.
        for t in range(warmup, T):
            interval = self.predict_step(model, X[t : t + 1])
            intervals.append(interval)
            idx = t - warmup
            agg_alpha_traj[idx] = interval.alpha
            expert_alpha_traj[idx] = self._expert_alphas
            weight_traj[idx] = self._expert_weights
            y_pred_t = float(model.predict(X[t : t + 1])[0])
            self.update_step(float(y[t]), y_pred_t)

        return intervals, agg_alpha_traj, expert_alpha_traj, weight_traj
