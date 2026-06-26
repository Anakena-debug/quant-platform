"""Tests for quantstrat.portfolio.nco — Nested Clustered Optimization (AFML §16.4).

Seven tests:
  1. test_nco_weights_sum_to_one_no_short — basic contract
  2. test_nco_collapses_to_gmv_at_one_cluster — analytic GMV pin at k=1
  3. test_nco_clusters_are_deterministic_under_seed — determinism
  4. test_nco_handles_singular_cov_via_pinv — pinv fallback on rank-deficient Σ
  5. test_nco_onc_recovers_three_block_structure — clean 3-block sanity
  6. test_nco_onc_recursion_fires_on_below_mean_cluster — mock-based discriminator
  7. test_nco_onc_consistent_on_wishart_sample — Wishart regression at fixed seed

Test 6 is the primary recursion-branch discriminator.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from quantstrat.portfolio.nco import (
    _onc_cluster,
    cluster_assets,
    nco_weights,
)


def _build_block_corr(
    n_blocks: int,
    block_size: int,
    intra: float,
    inter: float,
) -> np.ndarray:
    """Build a uniform block-correlation matrix (helper for tests 1, 5)."""
    n = n_blocks * block_size
    corr = np.full((n, n), inter)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            elif i // block_size == j // block_size:
                corr[i, j] = intra
    return corr


def _build_heterogeneous_intra_corr(
    rho_a: float = 0.45,
    rho_b: float = 0.20,
) -> np.ndarray:
    """Build the heterogeneous-intra base for the Wishart regression test."""
    intras = [0.75, 0.65, 0.55, 0.45]
    cross = 0.05
    n_assets, block_size = 16, 4
    corr = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                corr[i, j] = 1.0
                continue
            bi, bj = i // block_size, j // block_size
            si, sj = bi // 2, bj // 2
            if bi == bj:
                corr[i, j] = intras[bi]
            elif si == sj:
                corr[i, j] = rho_a if si == 0 else rho_b
            else:
                corr[i, j] = cross
    return corr


def test_nco_weights_sum_to_one_no_short() -> None:
    """NCO weights sum to 1 with no short positions on a well-conditioned fixture.

    Fixture: 6-asset 2-block correlation with positive correlations
    (intra=0.7, inter=0.3). Σ = corr (unit variance per asset).
    """
    cov = _build_block_corr(n_blocks=2, block_size=3, intra=0.7, inter=0.3)
    weights = nco_weights(cov)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (weights >= 0).all(), (
        f"Expected no short positions; got {weights} with negatives at "
        f"{np.where(weights < 0)[0].tolist()}"
    )


def test_nco_collapses_to_gmv_at_one_cluster() -> None:
    """At n_clusters=1, NCO collapses to global GMV.

    Pin against analytic GMV weights w* = Σ̂⁻¹𝟏 / (𝟏ᵀΣ̂⁻¹𝟏) at 1e-10 on a
    NON-diagonal Σ̂ (diagonal Σ̂ would let inverse-variance and GMV
    coincide and obscure the test).
    """
    n = 4
    cov = np.array(
        [
            [1.0, 0.7, 0.3, 0.4],
            [0.7, 1.2, 0.5, 0.2],
            [0.3, 0.5, 0.8, 0.6],
            [0.4, 0.2, 0.6, 1.5],
        ]
    )
    inv_cov_ones = np.linalg.solve(cov, np.ones(n))
    expected = inv_cov_ones / inv_cov_ones.sum()
    weights = nco_weights(cov, n_clusters=1)
    np.testing.assert_allclose(weights, expected, atol=1e-10)


def test_nco_clusters_are_deterministic_under_seed() -> None:
    """Same Σ̂ → same cluster labels on two independent calls (byte-exact)."""
    rng = np.random.default_rng(20260503)
    n = 20
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + 0.1 * np.eye(n)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    labels1 = cluster_assets(corr)
    labels2 = cluster_assets(corr)
    np.testing.assert_array_equal(labels1, labels2)


def test_nco_handles_singular_cov_via_pinv() -> None:
    """Rank-deficient Σ̂ doesn't raise; pinv fallback produces valid weights.

    Fixture: 6-asset cov with last column = first column (rank-deficient).
    ``_inv_sigma_dot`` catches the resulting LinAlgError on
    ``np.linalg.solve`` and falls back to ``np.linalg.pinv`` per the
    documented contract.
    """
    rng = np.random.default_rng(42)
    n = 6
    A = rng.standard_normal((n, n))
    A[:, -1] = A[:, 0]  # rank-deficient
    cov = A @ A.T  # PSD, rank-deficient
    weights = nco_weights(cov, n_clusters=1)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.isfinite(weights).all()


def test_nco_onc_recovers_three_block_structure() -> None:
    """ONC recovers k=3 on a 12-asset 3-block fixture (Wishart-sampled).

    Sanity check on the silhouette-argmax branch — does NOT exercise
    the recursive-split branch (use ``test_nco_onc_recursion_fires_on_
    below_mean_cluster`` for that). Asserts each 4-asset block maps to
    exactly one cluster label.

    Σ_true: clean 3-block (intra=0.7, inter=0.05). Wishart sample with
    T=300, seed=43 — natural sample noise breaks the perfect within-
    cluster silhouette uniformity that defeats t-stat-based ONC on
    analytic block fixtures. Distinct seed from test 3
    (``test_nco_clusters_are_deterministic_under_seed``) — avoids
    implicit coupling.
    """
    n, block_size = 12, 4
    Sigma_true = _build_block_corr(n_blocks=3, block_size=block_size, intra=0.7, inter=0.05)
    chol = np.linalg.cholesky(Sigma_true)
    rng = np.random.default_rng(43)
    Z = rng.standard_normal((300, n))
    sample_corr = np.corrcoef(Z @ chol.T, rowvar=False)
    labels = _onc_cluster(sample_corr)
    assert len(np.unique(labels)) == 3
    for block_start in range(0, n, block_size):
        block_labels = labels[block_start : block_start + block_size]
        assert len(set(block_labels.tolist())) == 1, (
            f"Block {block_start}..{block_start + block_size - 1} should map "
            f"to one cluster; got labels {block_labels.tolist()}"
        )


def test_nco_onc_recursion_fires_on_below_mean_cluster() -> None:
    """Mock-based unit test: recursion fires on below-mean cluster only.

    Discriminates the recursion code path from silhouette-argmax-only
    via direct control-flow assertions, independent of fixture algebra.

    Setup: 12 assets, top-level produces 2 clusters of 6 each. Per-
    cluster t-stats stubbed as [0.8, 0.2] with mean=0.5 — cluster 1
    is below mean. Recursion on cluster 1 produces 2 sub-clusters of
    3 each; sub mean_t=0.4 > parent t=0.2 ⇒ applied. Sub per-cluster
    t-stats both at mean (0.4, 0.4) ⇒ no further recursion.

    Assertions:
      - ``_onc_top_level`` called exactly 2 times (top + 1 recursion).
        If silhouette-argmax-only is silently substituted, call_count
        would be 1 (no recursion), failing this test loudly.
      - ``_silhouette_cluster_tstat`` called exactly 2 times.
      - The 2nd ``_onc_top_level`` call's first arg is a 6×6 sub-corr
        (cluster 1's members), NOT cluster 0 (which would also be
        6×6 here, but distinguishable by call count: 2, not 3).
      - Final result has 3 unique cluster labels matching
        ``[0]*6 + [1]*3 + [2]*3`` deterministically (cluster 0
        unchanged; cluster 1 split into 2 sub-clusters per the F01-
        style label-allocation pattern: first sub keeps parent cid,
        others get fresh labels).
    """
    corr = np.eye(12)  # placeholder; mocks bypass corr-dependent computation
    top_labels = np.array([0] * 6 + [1] * 6, dtype=np.intp)
    sub_labels = np.array([0] * 3 + [1] * 3, dtype=np.intp)
    onc_top_returns = [
        (top_labels, 0.5),  # top-level: mean_t=0.5
        (sub_labels, 0.4),  # recursion on cluster 1: sub mean_t=0.4 > parent t=0.2 ⇒ applied
    ]
    sil_t_returns = [
        (0.5, np.array([0.8, 0.2])),  # top: cluster 0 above mean, cluster 1 below
        (0.4, np.array([0.4, 0.4])),  # sub: both at mean ⇒ no further recursion
    ]
    with (
        patch("quantstrat.portfolio.nco._onc_top_level", side_effect=onc_top_returns) as mock_top,
        patch(
            "quantstrat.portfolio.nco._silhouette_cluster_tstat", side_effect=sil_t_returns
        ) as mock_sil,
    ):
        result = _onc_cluster(corr)

    assert mock_top.call_count == 2, (
        f"Expected 2 _onc_top_level calls (top + 1 recursion on below-mean "
        f"cluster), got {mock_top.call_count}. If 1: recursion didn't fire "
        f"(silhouette-argmax-only regression). If 3: recursion fired on the "
        f"above-mean cluster too (gate broken)."
    )
    assert mock_sil.call_count == 2

    sub_corr_arg = mock_top.call_args_list[1][0][0]
    assert sub_corr_arg.shape == (6, 6), (
        f"2nd _onc_top_level call should receive cluster 1's 6×6 sub-corr; "
        f"got shape {sub_corr_arg.shape}"
    )

    expected = np.array([0] * 6 + [1] * 3 + [2] * 3, dtype=np.intp)
    np.testing.assert_array_equal(result, expected)


def test_nco_onc_consistent_on_wishart_sample() -> None:
    """Regression defense: pin labels byte-exact on a Wishart-sampled fixture.

    Σ_true: heterogeneous-intra base (intra_a=(0.75, 0.65),
    intra_b=(0.55, 0.45), cross=0.05) at ρ_A=0.45, ρ_B=0.20. Sample
    T=200 returns from MVN(0, Σ_true) at seed=42; use sample
    correlation as input. Pin the resulting labels array byte-exact —
    catches scipy/sklearn drift AND label-allocation drift in our
    recursive code.

    This fixture does NOT discriminate ONC from silhouette-argmax-only
    (top-level k*=4 directly); discrimination is provided by
    ``test_nco_onc_recursion_fires_on_below_mean_cluster``.
    """
    Sigma_true = _build_heterogeneous_intra_corr(rho_a=0.45, rho_b=0.20)
    chol = np.linalg.cholesky(Sigma_true)
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((200, 16))
    returns = Z @ chol.T
    sample_corr = np.corrcoef(returns, rowvar=False)
    labels = _onc_cluster(sample_corr)
    expected = np.array([0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.intp)
    np.testing.assert_array_equal(labels, expected)
