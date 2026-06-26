"""Nested Clustered Optimization (NCO) for portfolio construction (AFML §16.4).

NCO weights from a fitted covariance matrix Σ̂ via:
  1. Convert Σ̂ → corr → distance d_ij = √((1 − ρ_ij)/2).
  2. Cluster: ONC algorithm (MLDP §4.4) for n_clusters=None, else
     explicit-k hierarchical clustering with the requested linkage.
  3. Within each cluster: GMV (mu=None) or MV (mu provided) weights,
     restricted to the cluster's columns of Σ̂.
  4. Reduce to cluster-level Σ̂_clusters and run inter-cluster GMV/MV.
  5. Compose: w_i = w_within(i, cluster(i)) · w_between(cluster(i)).

Default (``mu=None``, ``n_clusters=None``) is GMV with ONC clustering
— the benchmark harness's primary path. The convention: n_clusters
provided ⇒ explicit-k clustering with ``clustering_method`` linkage;
n_clusters=None ⇒ ONC determines k via silhouette t-stat search +
recursive re-clustering of below-mean-t clusters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

_LinkageMethod = Literal["ward", "single", "complete", "average"]


def _cluster_with_linkage(
    corr: NDArray[np.float64],
    n_clusters: int,
    method: _LinkageMethod = "ward",
) -> NDArray[np.intp]:
    """Hierarchical clustering on correlation distance with chosen linkage.

    Distance matrix d_ij = √((1 − ρ_ij)/2). Symmetry enforced via
    ``d = (d + d.T) / 2`` to defend against floating-point asymmetry
    in ``corr`` (real-world inputs like sample-cov estimators can be
    slightly asymmetric); without this, ``squareform(checks=False)``
    silently accepts asymmetric input and produces garbage.

    ward linkage on non-Euclidean distance is approximate (Lance-
    Williams update treats distances as Euclidean) but is the
    conventional choice in MLDP/HRP literature.

    Returns 0-indexed integer labels (length N).
    """
    d = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    d = (d + d.T) / 2.0  # enforce symmetry against floating-point drift
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method=method)
    return fcluster(Z, t=n_clusters, criterion="maxclust") - 1


def _silhouette_cluster_tstat(
    corr: NDArray[np.float64],
    labels: NDArray[np.intp],
) -> tuple[float, NDArray[np.float64]]:
    """Mean and per-cluster silhouette t-stat (López de Prado §4.4).

    For each cluster k: t_k = mean(s_k) / std(s_k) where s_k are the
    silhouette scores of points in cluster k.

    Edge cases handled (return t_k = 0.0):
      - Single-asset cluster (silhouette undefined; std = 0)
      - Constant silhouette within a cluster (std < 1e-12; rare but
        possible on degenerate inputs)
      - Degenerate clustering with fewer than 2 unique labels OR
        n_unique == n_samples (``scipy.cluster.hierarchy.fcluster``
        can return < n_clusters under tied distances on perfectly-
        symmetric inputs; ``sklearn.metrics.silhouette_samples``
        requires ``2 ≤ n_unique < n_samples``) → returns
        ``(0.0, zeros)``.

    Returns (mean_t_stat, per_cluster_t_stats).
    """
    unique_labels = np.unique(labels)
    n_unique = len(unique_labels)
    n_samples = len(labels)
    if n_unique < 2 or n_unique >= n_samples:
        return 0.0, np.zeros(max(n_unique, 1))
    d = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    s = silhouette_samples(d, labels, metric="precomputed")
    t_per_cluster = np.zeros(len(unique_labels))
    for i, k in enumerate(unique_labels):
        s_k = s[labels == k]
        if len(s_k) < 2 or s_k.std() < 1e-12:
            t_per_cluster[i] = 0.0
        else:
            t_per_cluster[i] = float(s_k.mean() / s_k.std())
    return float(t_per_cluster.mean()), t_per_cluster


def _onc_top_level(
    corr: NDArray[np.float64],
    method: _LinkageMethod = "ward",
) -> tuple[NDArray[np.intp], float]:
    """ONC top-level k-search via silhouette t-stat over k ∈ [2, ⌊N/2⌋].

    ``k_max = N // 2`` follows the MLDP §4.4 heuristic; for small N
    (5, 7) this restricts the search to k=2 only — for N=4 the range
    is k ∈ [2, 2] (single value, but still well-defined).

    Returns (best_labels, best_mean_tstat). For N < 4 (k_max < 2),
    returns all-zeros + -inf to signal "no meaningful clustering".
    """
    n = corr.shape[0]
    k_max = max(2, n // 2)
    if n < 4:
        return np.zeros(n, dtype=np.intp), float("-inf")
    best_mean_t = float("-inf")
    best_labels: NDArray[np.intp] | None = None
    for k in range(2, k_max + 1):
        labels = _cluster_with_linkage(corr, n_clusters=k, method=method)
        mean_t, _ = _silhouette_cluster_tstat(corr, labels)
        if mean_t > best_mean_t:
            best_mean_t = mean_t
            best_labels = labels
    if best_labels is None:
        return np.zeros(n, dtype=np.intp), float("-inf")
    return best_labels, float(best_mean_t)


def _onc_cluster(
    corr: NDArray[np.float64],
    *,
    method: _LinkageMethod = "ward",
    _parent_t_stat: float = float("-inf"),
) -> NDArray[np.intp]:
    """ONC recursive clustering (MLDP §4.4).

    Returns LABELS (NOT k*) — the recursive partition is the load-
    bearing output. Discarding the partition (returning only k*)
    would silently downgrade ONC to silhouette-argmax; the recursive-
    split branch would never run, defeating the test that
    discriminates the two implementations.

    Termination per recursion (either condition halts further
    subdivision of the current cluster):
      (i)  Cluster size < 2 (cannot subdivide).
      (ii) New clustering's mean t-stat ≤ ``_parent_t_stat``
           (subdivision doesn't improve over parent).
    """
    n = corr.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.intp)

    labels, mean_t = _onc_top_level(corr, method=method)
    if mean_t <= _parent_t_stat:
        # Termination (ii): clustering doesn't improve; collapse to single cluster.
        return np.zeros(n, dtype=np.intp)

    # Recursive re-clustering on below-mean-t clusters
    _, per_cluster_t = _silhouette_cluster_tstat(corr, labels)
    cluster_ids = np.unique(labels)
    next_label = int(labels.max()) + 1
    labels = labels.copy()  # avoid mutating the result of _onc_top_level

    for cid, t_k in zip(cluster_ids, per_cluster_t):
        if t_k >= mean_t:
            continue  # cluster is above-mean: don't re-cluster
        member_indices = np.where(labels == cid)[0]
        if len(member_indices) < 2:
            continue  # termination (i): too small to recurse
        sub_corr = corr[np.ix_(member_indices, member_indices)]
        sub_labels = _onc_cluster(sub_corr, method=method, _parent_t_stat=t_k)
        if sub_labels.max() == 0:
            continue  # recursion collapsed to single cluster (no subdivision applied)
        # Sub-cluster label allocation: first sub-cluster keeps parent's
        # cid (so above-mean clusters retain identity even after
        # neighboring below-mean clusters are subdivided); others get
        # fresh labels from next_label. The final relabel-to-consecutive
        # pass below normalizes to 0-indexed contiguous labels.
        for i, sub_cid in enumerate(np.unique(sub_labels)):
            absolute_indices = member_indices[sub_labels == sub_cid]
            if i == 0:
                labels[absolute_indices] = cid
            else:
                labels[absolute_indices] = next_label
                next_label += 1

    # Relabel to consecutive 0-based labels
    unique = np.unique(labels)
    relabel = {old: new for new, old in enumerate(unique)}
    return np.array([relabel[lbl] for lbl in labels], dtype=np.intp)


def _inv_sigma_dot(
    sigma: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Σ⁻¹·target with ``np.linalg.pinv`` fallback on singular Σ.

    Tries ``np.linalg.solve`` first (well-conditioned path, byte-equal to
    the analytic inverse); falls back to ``np.linalg.pinv`` on
    ``LinAlgError`` (singular or near-singular Σ — uses Moore-Penrose
    pseudoinverse).
    """
    try:
        return np.linalg.solve(sigma, target)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(sigma) @ target


def cluster_assets(
    corr: NDArray[np.float64],
    *,
    method: Literal["onc", "ward", "single", "complete", "average"] = "onc",
    n_clusters: int | None = None,
) -> NDArray[np.intp]:
    """Public dispatcher returning the cluster-label vector.

    Contract:
      method="onc" (default) — ONC algorithm; ``n_clusters`` is IGNORED
      (ONC selects k internally via silhouette t-stat search).

      method ∈ {"ward", "single", "complete", "average"} — explicit-k
      hierarchical clustering with the chosen linkage. ``n_clusters``
      is REQUIRED; raises ``ValueError`` if not provided.

    Public for benchmark introspection (e.g., the cov_benchmark harness
    in S19 PR5 reports cluster cardinality alongside portfolio metrics).
    """
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"`corr` must be a square 2-D matrix; got shape {corr.shape}")
    if method == "onc":
        return _onc_cluster(corr)
    if n_clusters is None:
        raise ValueError(
            f"n_clusters must be provided when method={method!r}; "
            f"only method='onc' selects k internally."
        )
    return _cluster_with_linkage(corr, n_clusters=n_clusters, method=method)


def nco_weights(
    cov: NDArray[np.float64],
    mu: NDArray[np.float64] | None = None,
    *,
    n_clusters: int | None = None,
    clustering_method: _LinkageMethod = "ward",
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """NCO weights from a fitted covariance matrix Σ̂ (AFML §16.4).

    Steps:
      1. Convert Σ̂ → corr → distance d_ij = √((1 − ρ_ij)/2).
      2. Cluster: ``n_clusters`` is None ⇒ ONC (silhouette t-stat
         search + recursive re-clustering, MLDP §4.4); else explicit-k
         hierarchical clustering with ``clustering_method`` linkage.
      3. Within each cluster: GMV (``mu=None``) or MV (``mu`` provided)
         weights, restricted to the cluster's columns of Σ̂.
      4. Reduce to a cluster-level Σ̂_clusters (cluster-aggregated
         covariance via within-cluster weights) and run inter-cluster
         GMV/MV.
      5. Compose: w_i = w_within(i, cluster(i)) · w_between(cluster(i)).

    Default (``mu=None``, ``n_clusters=None``) is GMV with ONC
    clustering — the benchmark harness's primary path.

    ``random_state`` is currently IGNORED: hierarchical clustering is
    deterministic given input. Reserved for future stochastic variants.

    Returns a 1-D weights vector of length N (sums to 1).

    Degeneracy fallback: if the intra- or inter-cluster GMV sum
    (``np.sum(Σ⁻¹·𝟏)``) is near zero (rare; occurs when ``Σ⁻¹·𝟏`` is
    nearly orthogonal to ``𝟏``), the affected cluster falls back
    silently to equal-weight allocation. This preserves the contract
    that ``nco_weights`` always returns valid weights summing to 1.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"`cov` must be a square 2-D matrix; got shape {cov.shape}")
    n = cov.shape[0]

    # Σ → corr (standardize via diagonal scaling)
    std = np.sqrt(np.diag(cov))
    std_safe = np.where(std > 0, std, 1.0)
    inv_std = 1.0 / std_safe
    corr = cov * inv_std[:, None] * inv_std[None, :]

    # Cluster
    if n_clusters is None:
        labels = _onc_cluster(corr, method=clustering_method)
    else:
        labels = _cluster_with_linkage(corr, n_clusters=n_clusters, method=clustering_method)

    unique_labels = np.unique(labels)
    n_clusters_actual = len(unique_labels)
    cluster_member_lists = [np.where(labels == cid)[0] for cid in unique_labels]

    # Intra-cluster weights (GMV or MV)
    intra_weights = np.zeros(n)
    for members in cluster_member_lists:
        sub_cov = cov[np.ix_(members, members)]
        if mu is not None:
            sub_target = np.asarray(mu, dtype=np.float64)[members]
        else:
            sub_target = np.ones(len(members))
        sub_w = _inv_sigma_dot(sub_cov, sub_target)
        s = float(np.sum(sub_w))
        if abs(s) < 1e-12:
            sub_w = np.ones(len(members)) / len(members)
        else:
            sub_w = sub_w / s
        intra_weights[members] = sub_w

    # Inter-cluster Σ̂ via within-cluster weight aggregation
    cov_clusters = np.zeros((n_clusters_actual, n_clusters_actual))
    for i, members_i in enumerate(cluster_member_lists):
        w_i = intra_weights[members_i]
        for j, members_j in enumerate(cluster_member_lists):
            w_j = intra_weights[members_j]
            cov_clusters[i, j] = w_i @ cov[np.ix_(members_i, members_j)] @ w_j

    # Inter-cluster weights (GMV or MV)
    if mu is not None:
        mu_arr = np.asarray(mu, dtype=np.float64)
        inter_target = np.array(
            [float(intra_weights[members] @ mu_arr[members]) for members in cluster_member_lists]
        )
    else:
        inter_target = np.ones(n_clusters_actual)
    inter_w = _inv_sigma_dot(cov_clusters, inter_target)
    s_inter = float(np.sum(inter_w))
    if abs(s_inter) < 1e-12:
        inter_w = np.ones(n_clusters_actual) / n_clusters_actual
    else:
        inter_w = inter_w / s_inter

    # Compose final weights: w_i = w_within(i, cluster(i)) · w_between(cluster(i))
    final_weights = np.zeros(n)
    for i, members in enumerate(cluster_member_lists):
        final_weights[members] = intra_weights[members] * inter_w[i]
    return final_weights
