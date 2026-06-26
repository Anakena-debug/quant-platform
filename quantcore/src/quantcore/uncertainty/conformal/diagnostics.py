"""Pure numerical diagnostic utilities for conformal primitives.

Two free-standing functions consumed by sibling conformal modules:

  - ``effective_sample_size`` — Kish's n_eff for any weight vector.
    Used by ``WeightedConformalRegressor.n_eff`` (and naturally
    composes with ``MondrianConformal``'s per-stratum diagnostics
    when the per-stratum base estimator is weighted).

  - ``normalized_entropy`` — Shannon entropy normalized to [0, 1].
    Used by ``DtACI.weight_entropy`` for expert-weight collapse
    detection per the 2026-04-29 conformal-stack review's failure-
    mode table (``H(w)/log K < 0.2`` signals collapse).

Import contract (one-way): this module imports ONLY from numpy and
the standard library. It does NOT import from any sibling conformal
module (``timeseries``, ``dtaci``, ``mondrian``, ``quantile``,
``regression``, ``finance/*``, etc.). Sibling modules import from
``diagnostics``; ``diagnostics`` imports from none of them. This
prevents circular-import setup latent in the design and keeps the
utilities provably pure (no class state, no policy decisions).

Pinned by ``test_diagnostics_invariants.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def effective_sample_size(weights: NDArray[np.floating] | list[float]) -> float:
    """Kish's effective sample size: ``n_eff = (∑w)² / ∑w²``.

    Returns the number of equally-weighted observations carrying the
    same statistical precision as the given weighted observations.
    For uniform weights, ``n_eff = n``; for a one-hot weight vector,
    ``n_eff = 1``; for all-zero weights, ``n_eff = NaN`` (no
    information).

    Parameters
    ----------
    weights : array-like of float
        Non-negative weights. Negative weights are accepted (no
        runtime check) but produce statistically meaningless output;
        callers are expected to pass non-negative weights from
        geometric decay, EWA aggregation, etc.

    Returns
    -------
    float
        ``n_eff = (∑w)² / ∑w²``, or ``NaN`` if input is empty or all
        zero (``∑w² = 0``).

    Notes
    -----
    Closed-form for geometric decay ``w_i = ρ^{n-i}``, i = 1..n:

        ``n_eff = (1 - ρ^n)(1 + ρ) / [(1 - ρ)(1 + ρ^n)]``

    Equivalent to the unfactored form
    ``(1-ρ^n)² · (1-ρ²) / [(1-ρ)² · (1-ρ^{2n})]`` after canceling
    ``(1-ρ^n)`` and ``(1-ρ)`` once each. The factored form is
    numerically more stable as ρ → 1: it has ``(1-ρ)`` linearly in
    the denominator, not quadratically, so catastrophic cancellation
    is bounded.

    Pinned at rel=1e-10 in
    ``test_diagnostics_invariants.py::test_pin_n_eff_geometric_decay_closed_form``.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return float("nan")
    s = float(w.sum())
    s2 = float((w * w).sum())
    if s2 == 0.0:
        return float("nan")
    return s * s / s2


def normalized_entropy(weights: NDArray[np.floating] | list[float]) -> float:
    """Shannon entropy of a weight vector, normalized to ``[0, 1]``.

    Computed as ``H(w) / log(K)`` where ``K`` is the number of
    weights and ``H(w) = -∑ p_i log p_i`` with ``p_i = w_i / ∑w``.

    Uniform weights produce ``1.0`` exactly (maximum entropy). A
    one-hot weight vector produces ``0.0`` exactly (zero entropy)
    via the ``0 · log(0) = 0`` convention. Mixed weight vectors with
    some zero entries produce a finite value in ``(0, 1)`` without
    NaN propagation.

    Parameters
    ----------
    weights : array-like of float
        Non-negative weights, length ``K >= 2`` for a meaningful
        return value.

    Returns
    -------
    float
        ``H(w) / log(K)`` in ``[0, 1]``. Returns ``NaN`` if:
          - ``K < 2`` (normalization is undefined; ``log(1) = 0``)
          - all weights are zero (no probability distribution)

    Notes
    -----
    Pinned by closed-form agreement on ``K=2``, ``w=(p, 1-p)``:

        ``H/log(2) = -(p log p + (1-p) log(1-p)) / log 2``

    Used by ``DtACI.weight_entropy`` to surface expert-weight
    collapse: per the 2026-04-29 conformal-stack review, the
    practical failure threshold is ``H(w)/log K < 0.2``.
    """
    w = np.asarray(weights, dtype=np.float64)
    K = w.size
    if K < 2:
        return float("nan")
    s = float(w.sum())
    if s == 0.0:
        return float("nan")
    p = w / s
    # 0 · log(0) = 0 convention via masked indexing: np.log is
    # called ONLY on the positive-mass entries, so no RuntimeWarning
    # is emitted for zero or negative entries. The np.where form
    # would still evaluate np.log(p) eagerly on the masked-away
    # branch and produce "divide by zero" / "invalid value"
    # RuntimeWarnings even though the result is discarded. Pinned in
    # test_pin_normalized_entropy_no_runtime_warning.
    mask = p > 0.0
    p_log_p = np.zeros_like(p)
    p_log_p[mask] = p[mask] * np.log(p[mask])
    H = -float(np.sum(p_log_p))
    return H / float(np.log(K))
