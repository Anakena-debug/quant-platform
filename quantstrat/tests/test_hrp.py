"""quantstrat — Hierarchical Risk Parity weights (s77 / audit advisory A5).

HRP (MLAM Ch.7 §7.4) is the inversion-free baseline for ``nco_weights``: tree linkage →
quasi-diagonalization → recursive inverse-variance bisection. Long-only, sums to 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantstrat.portfolio.hrp import hrp_weights
from quantstrat.portfolio.nco import nco_weights


def _block_cov(
    n_blocks: int, block_size: int, intra: float, inter: float, var: np.ndarray
) -> np.ndarray:
    n = n_blocks * block_size
    corr = np.full((n, n), inter)
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
            elif i // block_size == j // block_size:
                corr[i, j] = intra
    std = np.sqrt(var)
    return corr * np.outer(std, std)


def test_hrp_two_asset_is_inverse_variance_exactly():
    # For two uncorrelated assets HRP's single bisection IS inverse-variance: alpha = 1 - vL/(vL+vR).
    w = hrp_weights(np.diag([0.01, 0.04]))
    np.testing.assert_allclose(w, [0.8, 0.2])


def test_hrp_sum_to_one_and_long_only():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.standard_normal((300, 8)), rowvar=False)
    w = hrp_weights(cov)
    assert w.shape == (8,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.0)


def test_hrp_favors_low_variance():
    # Uncorrelated names with monotone variances: the lowest-variance name gets the most weight,
    # the highest the least (HRP allocates inversely to risk).
    var = np.array([0.01, 0.04, 0.09, 0.16])
    w = hrp_weights(np.diag(var))
    assert np.argmax(w) == int(np.argmin(var))
    assert np.argmin(w) == int(np.argmax(var))
    assert np.isclose(w.sum(), 1.0)


def test_hrp_vs_nco_both_valid_and_differ():
    # Same heterogeneous-vol block cov → HRP and NCO are both valid books and (being different
    # allocators) do not coincide. This is the comparison-baseline contract.
    var = np.array([0.01, 0.012, 0.05, 0.06, 0.20, 0.22])
    cov = _block_cov(n_blocks=3, block_size=2, intra=0.6, inter=0.1, var=var)
    w_hrp = hrp_weights(cov)
    w_nco = nco_weights(cov)
    for w in (w_hrp, w_nco):
        assert np.isclose(w.sum(), 1.0) and np.all(np.isfinite(w))
    assert np.any(np.abs(w_hrp - w_nco) > 1e-3)


def test_hrp_single_asset():
    np.testing.assert_array_equal(hrp_weights(np.array([[0.04]])), [1.0])


def test_hrp_rejects_non_square():
    with pytest.raises(ValueError):
        hrp_weights(np.zeros((3, 4)))


def test_hrp_handles_degenerate_zero_variance_name():
    # A zero-variance name must not produce NaN/inf (the _cluster_var 1/∞ guard).
    cov = np.diag([0.01, 0.0, 0.04])
    w = hrp_weights(cov)
    assert np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0)
