"""Hierarchical Risk Parity (HRP) portfolio construction (MLAM Ch.7 §7.4).

HRP weights from a covariance matrix Σ̂ via:
  1. Tree clustering on correlation distance (single linkage — the HRP default).
  2. Quasi-diagonalization: reorder assets by the dendrogram leaf order so similar
     assets sit adjacent (Σ̂ becomes ~block-diagonal).
  3. Recursive bisection: split the ordered list in two, allocate between the halves
     by inverse cluster-variance, recurse.

Unlike NCO (AFML §16.4 — cluster, then GMV/MV within and between clusters via Σ̂⁻¹),
HRP needs **no matrix inversion**: it is the robust, inversion-free baseline NCO is
benchmarked against, and it composes with a detoned (singular) Σ̂ since it touches only
the variances (diagonal) and the correlation (clustering). Returns long-only weights
summing to 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _quasi_diag_order(cov: NDArray[np.float64]) -> list[int]:
    """Dendrogram leaf order (single-linkage on correlation distance d=√((1−ρ)/2))."""
    std = np.sqrt(np.diag(cov))
    std_safe = np.where(std > 0, std, 1.0)
    corr = cov * (1.0 / std_safe)[:, None] * (1.0 / std_safe)[None, :]
    d = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    d = (d + d.T) / 2.0  # enforce symmetry against float drift (matches nco.py)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="single")
    return [int(i) for i in leaves_list(z)]


def _cluster_var(cov: NDArray[np.float64], items: list[int]) -> float:
    """Inverse-variance-weighted variance of a sub-cluster (MLAM Snippet 7.4).

    Zero-variance names are dropped from the inverse-variance weighting (1/∞ → 0); a
    fully-degenerate sub-cluster falls back to equal weight, so the result is finite.
    """
    sub = cov[np.ix_(items, items)]
    diag = np.diag(sub)
    inv = 1.0 / np.where(diag > 0.0, diag, np.inf)
    total = float(inv.sum())
    ivp = inv / total if total > 0.0 else np.ones(len(items)) / len(items)
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hierarchical Risk Parity weights from a covariance matrix Σ̂ (MLAM §7.4).

    Long-only, sums to 1, no matrix inversion. ``cov`` must be square; for N == 1 returns
    ``[1.0]``. The single-linkage quasi-diagonalization + recursive inverse-variance
    bisection is the canonical HRP; it serves as the inversion-free baseline for
    :func:`quantstrat.portfolio.nco.nco_weights`.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"`cov` must be a square 2-D matrix; got shape {cov.shape}")
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    w = np.ones(n)
    clusters: list[list[int]] = [_quasi_diag_order(cov)]
    while clusters:
        next_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)
            denom = var_left + var_right
            # Allocate inversely to cluster variance (the lower-variance half gets more).
            alpha = 0.5 if denom <= 0.0 else 1.0 - var_left / denom
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters

    total = float(w.sum())
    return w / total if total > 0.0 else np.ones(n) / n


__all__ = ["hrp_weights"]
