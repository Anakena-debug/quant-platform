"""F-RP-c3 (s19b): FFD regression suite for `quantcore.features.features`.

Closes the s19a known-unknowns finding c3: `features/features.py`
(83 LOC implementing AFML §5.4 fractional differentiation) had no
dedicated test file. Every S19 feature pipeline transits FFD via
`find_optimal_d` and `frac_diff_ffd`; a silent bug in the weight
recurrence, the deterministic d* search, or the warmup-truncation
contract would corrupt every feature S19 builds.

Four tests, per the s19b sprint plan:

  * `test_ffd_weights_closed_form` — `get_weights_ffd(d, threshold)`
    matches the AFML §5.4 weight recurrence
    ``w_k = -w_{k-1} * (d - k + 1) / k`` for d ∈ {0.1, 0.3, 0.5, 0.7,
    0.9} to ~1e-12.
  * `test_ffd_round_trip_stationarity` — on a recorded non-stationary
    GBM-cumsum fixture, `find_optimal_d` returns d* < 1.0 AND
    `adfuller(frac_diff_ffd(series, d*)).pvalue < 0.05`.
  * `test_find_optimal_d_deterministic` — same input → same d* /
    adf_stat / adf_pval / corr on two independent calls; pin d*
    byte-exact on the recorded fixture so a future numpy/scipy/
    statsmodels upgrade that perturbs adfuller's tie-breaking
    surfaces here.
  * `test_frac_diff_ffd_preserves_length_and_index` — output `len`
    matches input; index identical; name is ``{input.name}_ffd``;
    warmup region (first L-1 rows where L = len(weights)) is
    all-NaN; post-warmup region is all-finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcore.features.features import (
    FFDResult,
    find_optimal_d,
    frac_diff_ffd,
    get_weights_ffd,
)


# =============================================================================
# Test 1 — closed-form weight recurrence (AFML §5.4)
# =============================================================================


def test_ffd_weights_closed_form() -> None:
    """`get_weights_ffd(d, threshold=1e-5)` matches the AFML §5.4
    weight recurrence ``w_k = -w_{k-1} * (d - k + 1) / k`` for
    d ∈ {0.1, 0.3, 0.5, 0.7, 0.9} to absolute tolerance 1e-12.

    Implementation note: `get_weights_ffd` returns weights in
    REVERSED order (apply-from-newest-bar convention used by
    `_apply_ffd`). We forward-reorder for the recurrence check;
    the return-order convention itself is pinned implicitly by
    `test_frac_diff_ffd_preserves_length_and_index` since reversing
    the order would shift the warmup region.
    """
    threshold = 1e-5
    for d in (0.1, 0.3, 0.5, 0.7, 0.9):
        w_reversed = get_weights_ffd(d, threshold=threshold)
        # Forward-reorder for AFML recurrence: w[0] = 1, w[k] = -w[k-1] * (d - k + 1) / k.
        w = w_reversed[::-1]
        assert w[0] == 1.0, f"d={d}: w[0] must be exactly 1.0, got {w[0]!r}"
        for k in range(1, len(w)):
            expected = -w[k - 1] * (d - k + 1) / k
            assert np.isclose(w[k], expected, rtol=1e-12, atol=1e-12), (
                f"d={d}, k={k}: AFML §5.4 recurrence violation; "
                f"expected {expected:.6e}, got {w[k]:.6e}, "
                f"|diff|={abs(w[k] - expected):.6e}"
            )


# =============================================================================
# Test 2 — round-trip stationarity on a non-stationary fixture
# =============================================================================


def _gbm_cumsum_fixture(seed: int = 42, n: int = 500) -> pd.Series:
    """Deterministic non-stationary GBM-cumsum fixture. Used by tests
    2 and 3. Seed pinned per sprint constraint (`np.random.default_rng(42)`).
    """
    rng = np.random.default_rng(seed=seed)
    log_returns = rng.normal(loc=0.001, scale=0.01, size=n)
    return pd.Series(np.cumsum(log_returns), name="gbm_cumsum")


def test_ffd_round_trip_stationarity() -> None:
    """End-to-end contract: non-stationary input → stationary FFD
    output at the chosen d. `find_optimal_d` must achieve
    stationarity at d < 1.0 (i.e., a fractional order — not a full
    integer-order difference) and adfuller must confirm at p < 0.05.

    Bytes-exact d* pinning lives in `test_find_optimal_d_deterministic`;
    here we test the contract, not the exact value.
    """
    series = _gbm_cumsum_fixture()
    result = find_optimal_d(series)

    assert isinstance(result, FFDResult), (
        f"find_optimal_d must return FFDResult; got {type(result).__name__}"
    )
    assert result.d < 1.0, (
        f"find_optimal_d should achieve stationarity at d < 1.0 "
        f"(fractional, not integer); got d={result.d}"
    )
    # adfuller p-value must clear the function's pval_threshold default (0.05).
    assert result.adf_pval < 0.05, (
        f"adfuller p-value at returned d* must be < 0.05; got "
        f"adf_pval={result.adf_pval} at d={result.d}"
    )


# =============================================================================
# Test 3 — determinism + byte-exact d* pin
# =============================================================================


def test_find_optimal_d_deterministic() -> None:
    """Same series input → same d*, adf_stat, adf_pval, corr on two
    independent calls. Pin d* byte-exact on the recorded fixture so
    an adfuller / numpy linspace tie-breaking change surfaces here.

    The byte-exact d* value is ``np.linspace(0.0, 1.0, 11)[7]``
    (= 0.7000000000000001 in float64) on this fixture. The float
    representation comes from numpy's linspace endpoint handling,
    NOT from a hand-typed literal — anchoring on the linspace
    construction makes the pin survive a numpy upgrade that
    rebuilds the linspace, and fail loudly if the linspace
    semantics shift.
    """
    series = _gbm_cumsum_fixture()

    r1 = find_optimal_d(series)
    r2 = find_optimal_d(series)

    # Determinism.
    assert r1.d == r2.d
    assert r1.adf_stat == r2.adf_stat
    assert r1.adf_pval == r2.adf_pval
    assert r1.corr == r2.corr

    # Byte-exact d* pin: ``np.linspace(0, 1, 11)[7]``.
    expected_d = float(np.linspace(0.0, 1.0, 11)[7])
    assert r1.d == expected_d, (
        f"recorded fixture's d* must be {expected_d!r} "
        f"(np.linspace(0, 1, 11)[7]); got {r1.d!r}. If this changed "
        f"legitimately (e.g., adfuller upgrade rejected d=0.7 and "
        f"the search advanced to d=0.8), update this pin and "
        f"document the bump in the commit message."
    )


# =============================================================================
# Test 4 — length / index / warmup contract
# =============================================================================


def test_frac_diff_ffd_preserves_length_and_index() -> None:
    """`frac_diff_ffd` output preserves `len` + index + name of input.
    The first L-1 rows (where L = len of weights at this `(d, threshold)`)
    are NaN by construction; rows from index L-1 onward are finite.
    Same-length output is the load-bearing contract — callers
    downstream of S19's feature pipeline rely on the FFD output
    aligning bar-by-bar with the input price series.

    Fixture sized so the post-warmup region is non-empty: at
    d=0.7 with threshold=1e-5, weights have length 372, leaving
    ~628 valid rows at n=1000. Smaller d (e.g., 0.4) or smaller
    n would yield an all-NaN output and the post-warmup assertion
    would trivialize.
    """
    rng = np.random.default_rng(seed=42)
    n = 1000  # > weights length at (d=0.7, threshold=1e-5) so post-warmup is non-empty
    series = pd.Series(
        rng.standard_normal(n).cumsum(),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="cumsum_test",
    )
    threshold = 1e-5
    d = 0.7

    weights = get_weights_ffd(d, threshold=threshold)
    L = len(weights)
    # Fixture-design invariant: keep both regions non-empty.
    assert 0 < L < n, (
        f"test fixture invariant: weights length {L} must be in "
        f"(0, {n}) so warmup AND post-warmup regions are non-empty"
    )

    out = frac_diff_ffd(series, d, threshold=threshold)

    # Length contract.
    assert len(out) == n, f"expected len {n}, got {len(out)}"
    # Index contract: identical to input (preserves freq, label-by-label).
    pd.testing.assert_index_equal(out.index, series.index)
    # Name contract: `{input.name}_ffd`.
    assert out.name == f"{series.name}_ffd", f"name must be `{series.name}_ffd`, got {out.name!r}"
    # Warmup contract: first L-1 rows are NaN.
    warmup = out.iloc[: L - 1]
    assert warmup.isna().all(), (
        f"warmup region (first {L - 1} rows) must be all-NaN; "
        f"got {int(warmup.notna().sum())} non-NaN entries"
    )
    # Post-warmup contract: from index L-1 onward are finite (no NaN).
    post = out.iloc[L - 1 :]
    assert post.notna().all(), (
        f"post-warmup region (rows {L - 1}..{n - 1}) must be all-finite; "
        f"got {int(post.isna().sum())} NaN entries"
    )
