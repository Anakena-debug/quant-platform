"""Mondrian conformal prediction: per-stratum calibration.

Wraps an existing conformal regressor (any with the standard
``fit(X, y) -> self`` and ``predict(X) -> PredictionInterval``
contract — Split, CQR, CQR+, Weighted, etc.) and dispatches
calibration per-stratum. Restores within-stratum conditional
coverage when the stratifier captures the conditioning structure.

References
----------
Vovk et al. (2005), Boström & Johansson (2020) for Mondrian
conformal prediction. Practical per-stratum tolerance ``η = 0.03``
is the published threshold used as Pin 1's contract.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from quantcore.uncertainty.conformal.base import PredictionInterval


class MondrianConformal:
    """Mondrian conformal prediction primitive.

    Wraps an existing conformal regressor (any with the standard
    fit/predict contract) and dispatches per-stratum calibration.

    Restores conditional coverage::

        P(Y ∈ Ĉ(X) | R) ≥ 1 - α - η

    when the stratifier ``R`` captures the conditioning structure.
    Empty-stratum fallback to a globally-fit base estimator
    surfaces a ``used_fallback`` flag in the prediction diagnostic
    so the caller knows when the per-stratum guarantee does NOT
    apply.

    Contract direction (this docstring is the spec)
    -----------------------------------------------
    **Class-side guarantee.** ``MondrianConformal`` never passes
    future information (``y``, ``t1``, future ``X``) to the
    stratifier callable. The stratifier receives ``X`` only, at
    the decision-time slice the caller passes in. Verified by
    structural-spy pin in
    ``test_mondrian_invariants.py::test_pin_mondrian_stratifier_called_with_X_only``.

    **Caller-side guarantee.** The stratifier itself uses only
    ``F_t``-measurable inputs. A stratifier that internally
    consults future state (e.g., a regime label derived from the
    full sample's realized vol) passes the call-signature contract
    trivially but re-injects the leakage the conformal layer is
    meant to bound. Violations of the caller-side guarantee are
    SILENT — the class cannot detect them and will not try to.

    Both halves are required for Mondrian's coverage guarantee.
    Caller-side is documented here, not enforced.

    Parameters
    ----------
    base_estimator_factory : callable
        Zero-arg factory returning a fresh base conformal
        regressor instance. Each stratum receives its own instance,
        fit on that stratum's calibration subset.
    stratifier : callable
        ``X → labels`` mapping. Must accept a 2D ``X`` array and
        return integer-valued labels of shape ``(n,)``. Only
        called with ``X``; class-side guarantee is structural.
    alpha : float
        Target miscoverage rate.
    empty_stratum_fallback : {"global", "raise"}
        Behavior when ``predict`` is called on a row whose stratum
        was unseen during ``fit``:

          - ``"global"``: dispatch to a globally-fit base estimator
            (fit on all training data without stratification),
            and flag the row in ``diagnostic["used_fallback"]``.
            The per-stratum guarantee does NOT hold for fallback
            rows; the global interval is the best available.
          - ``"raise"``: raise ``ValueError`` naming the unseen
            stratum label. Use when the calling code cannot
            tolerate silent fallback.

    Attributes
    ----------
    is_fitted : bool
    per_stratum_n : dict[int, int]
        Calibration sample count per stratum at fit time.

    Notes
    -----
    The diagnostic dict returned by ``predict`` contains per-
    stratum ``n_eff`` when the base estimator exposes it (e.g.,
    ``WeightedConformalRegressor`` after S13 P12.1). For non-
    weighted base estimators, the entry is omitted rather than
    set to NaN — keeps the diagnostic surface honest.
    """

    def __init__(
        self,
        base_estimator_factory: Callable[[], Any],
        stratifier: Callable[[NDArray[np.floating[Any]]], NDArray[np.int_]],
        alpha: float = 0.1,
        empty_stratum_fallback: Literal["global", "raise"] = "global",
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if empty_stratum_fallback not in ("global", "raise"):
            raise ValueError(
                f"empty_stratum_fallback must be 'global' or "
                f"'raise', got {empty_stratum_fallback!r}"
            )

        self.base_estimator_factory = base_estimator_factory
        self.stratifier = stratifier
        self.alpha = alpha
        self.empty_stratum_fallback = empty_stratum_fallback

        self._per_stratum_estimators: dict[int, Any] = {}
        self._global_estimator: Any | None = None
        self._per_stratum_n: dict[int, int] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def per_stratum_n(self) -> dict[int, int]:
        return dict(self._per_stratum_n)

    def fit(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        **fit_params: Any,
    ) -> "MondrianConformal":
        """Fit one base estimator per stratum + a global fallback.

        Class-side guarantee: ``self.stratifier`` is invoked with
        ``X`` as the sole positional argument; ``y`` is never
        forwarded.
        """
        # Class-side leakage guard: we call stratifier with X only.
        labels = np.asarray(self.stratifier(X))
        if labels.shape[0] != X.shape[0]:
            raise ValueError(
                f"stratifier returned {labels.shape[0]} labels for "
                f"{X.shape[0]} input rows; shape mismatch"
            )

        unique_labels = np.unique(labels)
        self._per_stratum_estimators = {}
        self._per_stratum_n = {}
        for s in unique_labels:
            s_int = int(s)
            mask = labels == s
            est = self.base_estimator_factory()
            est.fit(X[mask], y[mask], **fit_params)
            self._per_stratum_estimators[s_int] = est
            self._per_stratum_n[s_int] = int(mask.sum())

        # Global fallback: a base estimator fit on all training
        # data without stratification. Used for predict() rows
        # whose stratum was unseen during fit.
        self._global_estimator = self.base_estimator_factory()
        self._global_estimator.fit(X, y, **fit_params)

        self._is_fitted = True
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> tuple[PredictionInterval, dict[str, Any]]:
        """Return per-row prediction intervals + diagnostic.

        Returns
        -------
        intervals : PredictionInterval
            Lower / upper bounds, one per row of X.
        diagnostic : dict
            - ``stratum_labels``: shape (n,), label per row
            - ``used_fallback``: shape (n,) bool, True where the
              row's stratum was unseen at fit time and the global
              estimator was used
            - ``per_stratum_n``: dict[int, int], calibration count
              per fit-time stratum
            - ``per_stratum_n_eff``: dict[int, float], OPTIONAL
              (only present when the base estimator exposes
              ``n_eff``)
        """
        if not self._is_fitted:
            raise RuntimeError("MondrianConformal is not fitted")

        labels = np.asarray(self.stratifier(X))
        if labels.shape[0] != X.shape[0]:
            raise ValueError(
                f"stratifier returned {labels.shape[0]} labels for "
                f"{X.shape[0]} input rows; shape mismatch"
            )

        n = labels.shape[0]
        lowers = np.empty(n, dtype=np.float64)
        uppers = np.empty(n, dtype=np.float64)
        used_fallback = np.zeros(n, dtype=bool)

        # Group by stratum to make per-stratum predict calls in
        # batch (cheaper than per-row dispatch).
        unique_test = np.unique(labels)
        for s in unique_test:
            s_int = int(s)
            mask = labels == s
            X_s = X[mask]
            if s_int in self._per_stratum_estimators:
                interval = self._per_stratum_estimators[s_int].predict(X_s)
            else:
                if self.empty_stratum_fallback == "raise":
                    seen = sorted(self._per_stratum_estimators.keys())
                    raise ValueError(
                        f"Stratum {s_int} unseen during fit "
                        f"(fit-time strata: {seen}); "
                        f"empty_stratum_fallback='raise'."
                    )
                # fallback = "global"
                assert self._global_estimator is not None
                interval = self._global_estimator.predict(X_s)
                used_fallback[mask] = True
            lowers[mask] = np.asarray(interval.lower).reshape(-1)
            uppers[mask] = np.asarray(interval.upper).reshape(-1)

        # Per-stratum n_eff if the base estimator exposes it.
        per_stratum_n_eff: dict[int, float] = {}
        for s_int, est in self._per_stratum_estimators.items():
            n_eff_val = getattr(est, "n_eff", None)
            if n_eff_val is not None:
                # Properties resolve to a value; methods/None we
                # skip silently. Wrap in float() to normalize.
                try:
                    per_stratum_n_eff[s_int] = float(n_eff_val)
                except (TypeError, ValueError):
                    pass

        diagnostic: dict[str, Any] = {
            "stratum_labels": labels,
            "used_fallback": used_fallback,
            "per_stratum_n": dict(self._per_stratum_n),
        }
        if per_stratum_n_eff:
            diagnostic["per_stratum_n_eff"] = per_stratum_n_eff

        intervals = PredictionInterval(
            lower=lowers,
            upper=uppers,
            alpha=self.alpha,
        )
        return intervals, diagnostic
