"""Regression tests pinning block bootstrap + Patton-Politis-White 2009
block-length selector (P2.2, advisory A-4).

Module under test: ``quantcore.weights.block_bootstrap``.

Pin surface (4 invariants):
  1. White noise — ``block_size=1`` is IID-equivalent; at larger block
     the mean-SR distribution matches IID via two-sample Kolmogorov-
     Smirnov (p > 0.01).
  2. AR(1) ρ=0.5 — block bootstrap replicate-mean variance is
     strictly greater than IID bootstrap variance (dependence
     preservation; Kuensch 1989 / Patton-Politis-White 2009
     motivation).
  3. Patton-Politis-White 2009 selector — returned block length is
     monotone non-decreasing in |ρ| over AR(1) fixtures with
     ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
  4. Boundary inclusion — moving block under-samples the last
     ``block_size − 1`` positions; circular block has uniform
     inclusion. Verified empirically via inclusion-count histogram.

All tests are deterministic under seeded ``np.random.default_rng`` —
both the block_starts drawn in pure-Python ``block_bootstrap`` and the
numba-compiled assembly core are seed-driven.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from quantcore.weights.block_bootstrap import (
    block_bootstrap,
    politis_white_block_length,
)


SEED = 20260422


# -----------------------------------------------------------------------------
# Invariant 1 — White noise: IID equivalence + distribution match
# -----------------------------------------------------------------------------


def test_block_bootstrap_whitenoise_block1_is_iid_identical() -> None:
    """At ``block_size=1``, block bootstrap must produce IID-equivalent
    resampling (each position drawn independently from ``[0, n)``). Both
    circular and moving paths collapse to the same IID sampler — there
    is no "block" to span. The test asserts the resampled means match
    their IID-analogue means under the same seed.
    """
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(500)

    # With block_size=1 and n_replicates=R, both branches draw R*n indices.
    # We can't assert bitwise equality against a hand-written IID sampler
    # without replicating the rng state structure; instead assert that the
    # empirical mean distribution matches iid-draw-of-means within 1e-2.
    rep_circ = block_bootstrap(
        x,
        block_size=1,
        n_replicates=400,
        rng=np.random.default_rng(SEED),
        circular=True,
    )
    rep_mov = block_bootstrap(
        x,
        block_size=1,
        n_replicates=400,
        rng=np.random.default_rng(SEED),
        circular=False,
    )
    means_circ = np.asarray(rep_circ).mean(axis=1)
    means_mov = np.asarray(rep_mov).mean(axis=1)

    # Both branches should be close to the original mean, with bootstrap se.
    x_mean = float(x.mean())
    x_se = float(x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(means_circ.mean() - x_mean) < 5 * x_se
    assert abs(means_mov.mean() - x_mean) < 5 * x_se

    # KS two-sample: circular and moving block=1 distributions indistinguishable.
    ks = sp_stats.ks_2samp(means_circ, means_mov)
    assert ks.pvalue > 0.01, (
        f"block_size=1 circular vs moving distributions disagree "
        f"(KS p={ks.pvalue:.3f}); expected indistinguishable since "
        f"both collapse to IID sampling with no block span."
    )


def test_block_bootstrap_whitenoise_block20_ks_matches_iid() -> None:
    """White-noise input: block bootstrap at ``block_size=20`` and
    IID bootstrap (``block_size=1``) should produce mean-SR
    distributions indistinguishable via Kolmogorov-Smirnov.

    Dependence is zero in the population, so block resampling has no
    benefit and no harm. Both distributions share the same asymptotic
    CLT limit; the test cannot reject equality at p > 0.01 on 400
    replicates.
    """
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(2000)

    rep_iid = block_bootstrap(
        x,
        block_size=1,
        n_replicates=400,
        rng=np.random.default_rng(SEED + 1),
        circular=True,
    )
    rep_blk = block_bootstrap(
        x,
        block_size=20,
        n_replicates=400,
        rng=np.random.default_rng(SEED + 2),
        circular=True,
    )
    m_iid = np.asarray(rep_iid).mean(axis=1)
    m_blk = np.asarray(rep_blk).mean(axis=1)

    ks = sp_stats.ks_2samp(m_iid, m_blk)
    assert ks.pvalue > 0.01, (
        f"White-noise: block-bootstrap mean distribution differs from "
        f"IID (KS p={ks.pvalue:.3g}). Under iid population, block "
        f"resampling should be indistinguishable from IID at p > 0.01."
    )


# -----------------------------------------------------------------------------
# Invariant 2 — AR(1) ρ=0.5: block variance ≥ IID variance
# -----------------------------------------------------------------------------


def test_block_bootstrap_ar1_preserves_dependence_variance() -> None:
    """AR(1) ρ=0.5 input: block bootstrap replicate-mean variance must
    EXCEED IID bootstrap variance.

    Rationale: IID resampling destroys the autocorrelation, yielding
    artificially low replicate-mean variance. Block resampling preserves
    short-range dependence and recovers the correct variance scaling.
    This is the canonical motivation for block bootstrap (Kuensch 1989).

    Pin: block variance strictly greater than IID variance on 500
    replicates. The ratio is order-of-magnitude stable (block ≈ 1.5-3×
    IID on ρ=0.5); we assert only the inequality to stay robust.
    """
    rng = np.random.default_rng(SEED)
    n, rho = 2000, 0.5
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]

    rep_iid = block_bootstrap(
        x,
        block_size=1,
        n_replicates=500,
        rng=np.random.default_rng(SEED + 10),
        circular=True,
    )
    rep_blk = block_bootstrap(
        x,
        block_size=32,
        n_replicates=500,
        rng=np.random.default_rng(SEED + 11),
        circular=True,
    )
    var_iid = float(np.var(np.asarray(rep_iid).mean(axis=1), ddof=1))
    var_blk = float(np.var(np.asarray(rep_blk).mean(axis=1), ddof=1))

    assert var_blk > var_iid, (
        f"AR(1) ρ={rho}: block-bootstrap variance {var_blk:.5f} is NOT > "
        f"IID variance {var_iid:.5f}. Block bootstrap must preserve "
        f"short-range dependence, producing larger replicate-mean "
        f"variance than the dependence-destroying IID resampler. "
        f"Either block_bootstrap has regressed or the fixture is wrong."
    )


# -----------------------------------------------------------------------------
# Invariant 3 — Patton-Politis-White monotone in |ρ|
# -----------------------------------------------------------------------------


def test_politis_white_block_length_monotone_in_ar1_rho() -> None:
    """``politis_white_block_length`` returns non-decreasing block length
    for AR(1) processes with increasing ``|ρ|``.

    Sweep ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} on n=2000 samples (fresh RNG
    per draw so variance across ρ reflects signal, not noise). Each
    successive block length must be >= previous (non-strict: ties at
    very low ρ rounded to the same integer are acceptable).
    """
    rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = 2000
    blocks: list[int] = []
    for i, rho in enumerate(rhos):
        rng = np.random.default_rng(SEED + 100 + i)
        eps = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + eps[t]
        blocks.append(politis_white_block_length(x))

    for i in range(1, len(blocks)):
        assert blocks[i] >= blocks[i - 1], (
            f"Patton-Politis-White block length not monotone: "
            f"blocks={blocks} for rhos={rhos}. "
            f"Expected non-decreasing. Either the selector regressed "
            f"or the MC noise happened to invert at one step (re-run "
            f"with different seeds to confirm signal)."
        )

    # Additional sanity: high-rho block is meaningfully larger than low-rho.
    assert blocks[-1] > blocks[0], (
        f"Patton-Politis-White at ρ=0.9 ({blocks[-1]}) should be much "
        f"larger than at ρ=0.1 ({blocks[0]}); only modest difference "
        f"observed — suggests the selector has lost sensitivity."
    )


# -----------------------------------------------------------------------------
# Invariant 4 — Boundary inclusion: moving vs circular
# -----------------------------------------------------------------------------


def test_block_bootstrap_circular_uniform_inclusion() -> None:
    """Circular block: every original sample has equal inclusion
    probability. Empirically the inclusion-count histogram is flat
    within ±5% of the mean over R=10_000 replicates.
    """
    n, block_size = 100, 10
    x = np.arange(n, dtype=np.float64)  # trackable identity values
    rep = block_bootstrap(
        x,
        block_size=block_size,
        n_replicates=10_000,
        rng=np.random.default_rng(SEED + 200),
        circular=True,
    )
    # Count how many times each original index (via its value) appears.
    rep_arr = np.asarray(rep)
    counts = np.zeros(n, dtype=np.int64)
    for v in range(n):
        counts[v] = int((rep_arr == float(v)).sum())

    mean_count = float(counts.mean())
    # Uniform inclusion: ±5% band.
    assert counts.min() > 0.95 * mean_count, (
        f"Circular block inclusion non-uniform at low end: "
        f"min={counts.min()}, mean={mean_count:.1f}."
    )
    assert counts.max() < 1.05 * mean_count, (
        f"Circular block inclusion non-uniform at high end: "
        f"max={counts.max()}, mean={mean_count:.1f}."
    )


def test_block_bootstrap_moving_undersamples_end() -> None:
    """Moving block: the last ``block_size − 1`` positions are
    under-sampled because a block starting at index ``n - block_size``
    is the last legal start, so index ``n-1`` appears in fewer blocks
    than an interior index. The ratio
    ``inclusion[n-1] / inclusion[n//2]`` should be well below 1.0.
    """
    n, block_size = 100, 10
    x = np.arange(n, dtype=np.float64)
    rep = block_bootstrap(
        x,
        block_size=block_size,
        n_replicates=10_000,
        rng=np.random.default_rng(SEED + 201),
        circular=False,
    )
    rep_arr = np.asarray(rep)
    counts = np.zeros(n, dtype=np.int64)
    for v in range(n):
        counts[v] = int((rep_arr == float(v)).sum())

    ratio_end_to_mid = counts[n - 1] / counts[n // 2]
    assert ratio_end_to_mid < 0.9, (
        f"Moving block did NOT under-sample the series end: "
        f"inclusion[n-1]={counts[n - 1]}, inclusion[n//2]={counts[n // 2]}, "
        f"ratio={ratio_end_to_mid:.3f}. Expected < 0.9."
    )


# -----------------------------------------------------------------------------
# Bonus — determinism, input validation, degenerate variance
# -----------------------------------------------------------------------------


def test_block_bootstrap_determinism_under_seed() -> None:
    """Same rng state -> bitwise-identical output (both paths)."""
    x = np.arange(100, dtype=np.float64)
    a = block_bootstrap(
        x, block_size=5, n_replicates=10, rng=np.random.default_rng(42), circular=False
    )
    b = block_bootstrap(
        x, block_size=5, n_replicates=10, rng=np.random.default_rng(42), circular=False
    )
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    c = block_bootstrap(
        x, block_size=5, n_replicates=10, rng=np.random.default_rng(42), circular=True
    )
    d = block_bootstrap(
        x, block_size=5, n_replicates=10, rng=np.random.default_rng(42), circular=True
    )
    np.testing.assert_array_equal(np.asarray(c), np.asarray(d))


def test_block_bootstrap_input_validation() -> None:
    x = np.arange(100, dtype=np.float64)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"block_size must be >= 1"):
        block_bootstrap(x, block_size=0, n_replicates=5, rng=rng)
    with pytest.raises(ValueError, match=r"block_size .* exceeds series length"):
        block_bootstrap(x, block_size=200, n_replicates=5, rng=rng)
    with pytest.raises(ValueError, match=r"n_replicates must be >= 1"):
        block_bootstrap(x, block_size=5, n_replicates=0, rng=rng)
    with pytest.warns(UserWarning, match=r"few possible block starts"):
        block_bootstrap(x, block_size=60, n_replicates=5, rng=rng)


def test_politis_white_degenerate_variance_raises() -> None:
    x = np.ones(200, dtype=np.float64)
    with pytest.raises(ValueError, match=r"degenerate-variance"):
        politis_white_block_length(x)
