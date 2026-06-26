"""Ledoit-Wolf shrinkage covariance estimator.

Wraps ``sklearn.covariance.ledoit_wolf`` (the function-form, sklearn≥1.8.0)
and exposes the shrinkage intensity δ̂ explicitly. The function-form
returns ``(shrunk_cov, shrinkage)`` directly without the
``EmpiricalCovariance._set_covariance`` side-effect that the class form
triggers — ``self.precision_`` populated via ``pinvh(cov)`` → ``eigh``
to satisfy the parent class's API contract, which cs_alpha_nco never
reads. Per the PR1 cProfile baseline (S23b, SHA fa97dfc), the
precision-matrix construction consumed ~56% of total
cs_alpha_nco_backtest wallclock; bypassing it via the function-form
restores that wallclock. sklearn's function-form and class-form share
the same closed-form Ledoit-Wolf 2004 numerics by construction —
``test_ledoit_wolf_shrinkage_matches_class_form`` pins byte-equality.

Test fixtures pin against sklearn 1.8.x δ̂ output; recorded values may
drift on sklearn upgrades — the byte-exact LW oracle test catches this.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import ledoit_wolf as _sklearn_ledoit_wolf


def ledoit_wolf_shrinkage(
    returns: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Ledoit-Wolf shrunk covariance with explicit shrinkage intensity.

    Returns
    -------
    (Σ̂_shrunk, δ̂)
        Σ̂_shrunk is the shrunk covariance matrix; δ̂ ∈ [0, 1] is the
        shrinkage intensity.

    Wraps ``sklearn.covariance.ledoit_wolf`` — closed-form Ledoit-Wolf
    (2004) estimator with target F = (trace(Σ̂_sample) / N) · I. Uses
    the function-form rather than the ``LedoitWolf`` class so the
    precision-matrix side-effect of ``EmpiricalCovariance._set_covariance``
    (which we never read) is skipped.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(
            f"`returns` must be 2-D (rows=samples, cols=features); got ndim={returns.ndim}"
        )
    if returns.shape[0] < 2:
        raise ValueError(
            f"`returns` must have at least 2 samples (rows); got n_samples={returns.shape[0]}"
        )
    cov, shrinkage = _sklearn_ledoit_wolf(returns)
    return cov, float(shrinkage)
