"""Regression tests closing F01 (LeakageFreePCA within-fold fit-state).

Before the P2 fix, ``LeakageFreePCA.transform`` recomputed ``col_means``
from the arriving X — so at test time NaN rows were filled with
test-set means rather than the training means computed in ``fit``. On a
regime-shifted fixture this produced ~10x divergence in the first
principal component.

The fix persists ``self.col_means_`` at fit-time and reuses it at
transform-time. These two tests pin the fix:

  TEST 1 (ensemble-level). After the class is fit honestly on training
  data, the full downstream pipeline (PCA -> Ridge ->
  SplitConformalRegressor) hits the target 1-alpha coverage on a
  disjoint test set, within a 3 sigma normal-approx band. This guards
  against any future change that silently breaks the honest fit/transform
  contract in a way that corrupts conformal scores.

  TEST 2 (direct unit test). F01 is specifically the col_means-for-
  NaN-fill defect: ``fit`` previously computed col_means but did not
  persist them; ``transform`` recomputed them from the arriving X. The
  unit test below constructs a train / test pair with a known mean
  shift AND a NaN at a known position, then asserts:
    (a) ``self.col_means_`` is persisted at fit-time and matches the
        training means,
    (b) the persisted col_means differ materially from the test-set
        col_means (the fixture has an intentional 10x mean shift),
    (c) ``transform(X_test)`` fills the NaN with the persisted
        training mean, not the test-set mean — proved by bitwise
        equality to the fix-path reconstruction and bitwise inequality
        to the pre-fix reconstruction.

  The ensemble contract test originally specified in the S1 dispatch
  (honest > leaky interval width) was replaced at sprint time with the
  direct unit test above, after empirical investigation showed that on
  a pure-Gaussian no-NaN synthesis the col_means code path is a no-op
  (``np.where(never_triggers, ...)`` is identity), so the ensemble
  test could not discriminate the F01 defect from PCA basis leakage,
  which is a separate concern outside this class's contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from quantcore.covariance import denoise_covariance, ledoit_wolf_shrinkage
from quantcore.covariance.transformer import (
    LeakageFreeLedoitWolfShrinkage,
    LeakageFreeRMTDenoiser,
)
from quantcore.preprocessing.transformers import LeakageFreePCA
from quantcore.uncertainty.conformal import SplitConformalRegressor


# -----------------------------------------------------------------------------
# Fixture for TEST 1 — low-rank linear signal in high-dim features
# -----------------------------------------------------------------------------

N = 2000
D = 50
LEAK_RANK = 5
NOISE = 0.1
N_TRAIN = 800
N_CAL = 600
N_TEST = 600
SEED = 20260422

assert N_TRAIN + N_CAL + N_TEST == N


def _synthesize(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) with an intrinsic rank-LEAK_RANK linear signal."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(N, LEAK_RANK))
    W = rng.standard_normal(size=(D, LEAK_RANK))
    eps = rng.standard_normal(size=(N, D))
    X = Z @ W.T + NOISE * eps
    beta_z = rng.standard_normal(size=LEAK_RANK)
    y = Z @ beta_z + NOISE * rng.standard_normal(size=N)
    return X, y


def _split(
    X: np.ndarray, y: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_train, X_cal, X_test = X[:N_TRAIN], X[N_TRAIN : N_TRAIN + N_CAL], X[N_TRAIN + N_CAL :]
    y_train, y_cal, y_test = y[:N_TRAIN], y[N_TRAIN : N_TRAIN + N_CAL], y[N_TRAIN + N_CAL :]
    return X_train, X_cal, X_test, y_train, y_cal, y_test


# -----------------------------------------------------------------------------
# TEST 1 — ensemble-level coverage
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_leakage_free_pca_split_conformal_coverage(alpha: float) -> None:
    """Honestly-fit PCA + Ridge + Split Conformal hits 1 - alpha to 3 sigma."""
    X, y = _synthesize()
    X_train, X_cal, X_test, y_train, y_cal, y_test = _split(X, y)

    pca = LeakageFreePCA(n_components=LEAK_RANK).fit(X_train)
    X_train_pc = np.asarray(pca.transform(X_train))
    X_cal_pc = np.asarray(pca.transform(X_cal))
    X_test_pc = np.asarray(pca.transform(X_test))

    ridge = Ridge(alpha=1.0).fit(X_train_pc, y_train)
    cp = SplitConformalRegressor(ridge, alpha=alpha)
    cp.fit_prefit(X_cal_pc, y_cal)

    interval = cp.predict(X_test_pc)
    covered = (y_test >= interval.lower) & (y_test <= interval.upper)
    coverage = float(covered.mean())

    target = 1.0 - alpha
    sigma = float(np.sqrt(alpha * (1.0 - alpha) / N_TEST))
    tol = 3.0 * sigma

    assert target - tol <= coverage <= target + tol, (
        f"Split Conformal coverage {coverage:.4f} outside 3 sigma band "
        f"[{target - tol:.4f}, {target + tol:.4f}] for alpha={alpha}. "
        f"If LeakageFreePCA.transform is recomputing col_means from X at "
        f"transform time (F01 regression), calibration and test projections "
        f"drift relative to the model's training-time PCA basis."
    )


# -----------------------------------------------------------------------------
# TEST 2 — direct F01 unit test
# -----------------------------------------------------------------------------


def test_leakage_free_pca_transform_reuses_train_col_means() -> None:
    """F01 contract: transform() fills NaN with self.col_means_ from fit.

    Fixture:
      - X_train: mean ≈ [1,2,3,4,5] (100 rows, 5 features).
      - X_test:  mean ≈ [10,20,30,40,50] (20 rows, 5 features) — a 10x
        regime shift relative to training.
      - X_test[0, 0] set to NaN to exercise the fill code path.

    The pre-P2 code at transformers.py:351 did:
      col_means = np.nanmean(X_arr, axis=0)   # recomputes from test X
      X_filled = np.where(np.isnan(X_arr), col_means, X_arr)
    -> X_test[0, 0] fills to ~10 (test mean of col 0).

    The P2 fix persists self.col_means_ at fit and reuses it at transform:
    -> X_test[0, 0] fills to ~1 (train mean of col 0).

    This test asserts (a) col_means_ is persisted, (b) persisted value
    differs materially from test-set col_means, (c) transform output is
    bitwise-identical to the fix-path reconstruction and bitwise-distinct
    from the pre-fix reconstruction. If any assertion fails, F01 has
    regressed — do not weaken this test.
    """
    rng = np.random.default_rng(SEED)

    train_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_mean = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    X_train = rng.standard_normal((100, 5)) + train_mean
    X_test = rng.standard_normal((20, 5)) + test_mean
    X_test[0, 0] = np.nan

    pca = LeakageFreePCA(n_components=3).fit(X_train)

    # (a) col_means_ persisted and matches training means.
    assert pca.col_means_ is not None, (
        "LeakageFreePCA.col_means_ was not persisted at fit-time. "
        "F01 regression."
    )
    expected_train_means = np.nanmean(X_train, axis=0)
    np.testing.assert_allclose(
        pca.col_means_,
        expected_train_means,
        atol=1e-12,
        err_msg=(
            "LeakageFreePCA.col_means_ does not match nanmean(X_train). "
            "fit() must persist the training-set column means."
        ),
    )

    # (b) persisted col_means differ materially from test-set col_means
    # (otherwise the fixture does not discriminate the F01 defect).
    test_col_means = np.nanmean(np.asarray(X_test), axis=0)
    col0_diff = abs(float(pca.col_means_[0]) - float(test_col_means[0]))
    assert col0_diff > 5.0, (
        f"Fixture error: train/test col_means[0] differ by only "
        f"{col0_diff:.4f} (need >5.0 to discriminate F01). Check that the "
        f"regime shift in the fixture is intact."
    )

    # (c) transform output == fix-path reconstruction.
    X_test_arr = np.asarray(X_test)
    nan_mask = np.isnan(X_test_arr)
    X_filled_fix = np.where(nan_mask, pca.col_means_, X_test_arr)
    X_filled_pre_fix = np.where(nan_mask, test_col_means, X_test_arr)

    # Sanity: fix-path and pre-fix-path produce DIFFERENT filled arrays
    # (fills the NaN position with different values).
    assert not np.allclose(X_filled_fix, X_filled_pre_fix), (
        "Fixture error: fix-path and pre-fix-path produce identical filled "
        "arrays. Check NaN injection and regime shift."
    )

    pc_output = np.asarray(pca.transform(X_test))
    pc_fix = pca._pca.transform(X_filled_fix)  # type: ignore[union-attr]
    pc_pre_fix = pca._pca.transform(X_filled_pre_fix)  # type: ignore[union-attr]

    assert np.allclose(pc_output, pc_fix, atol=1e-12), (
        "LeakageFreePCA.transform output does not match the fix-path "
        "reconstruction (fill NaN with persisted self.col_means_). "
        "F01 may have regressed: transform() is likely recomputing col_means "
        "from the arriving X. See transformers.py docstring in fit()."
    )
    assert not np.allclose(pc_output, pc_pre_fix, atol=1e-12), (
        "LeakageFreePCA.transform output matches the PRE-FIX reconstruction "
        "(fill NaN with recomputed test-set col_means). F01 HAS regressed — "
        "do not weaken this test; restore fit-time persistence of col_means_."
    )


# -----------------------------------------------------------------------------
# F-RP-006 PR2 — LeakageFreeRMTDenoiser + LeakageFreeLedoitWolfShrinkage canaries
# -----------------------------------------------------------------------------

# Discrimination thresholds pinned at 0.5 × measured_min across 5 seed
# pairs. Measured by quantcore/tests/spikes/_pr2_threshold_measurement.py
# (scratch script — deleted after thresholds locked here):
#   RMT Design A (volatility 1x→3x):
#     min rel_frob_diff = 7.52  →  pin > 3.76
#     (σ² and λ+ are scale-invariant — corr eigenvalues unchanged;
#      discrimination is entirely in cov_ via std_.)
#   LW Design B (correlation injection — uncorrelated train vs spike test):
#     min cov rel_frob_diff = 0.85  →  pin > 0.42
#     min |δ̂_diff|          = 0.64  →  pin > 0.32
_RMT_DISCRIMINATION_PIN = 3.76
_LW_COV_DISCRIMINATION_PIN = 0.42
_LW_DELTA_DISCRIMINATION_PIN = 0.32


def _build_volatility_regime_fixture(
    seed_train: int = 20260503,
    seed_test: int = 20260504,
    *,
    n_samples: int = 200,
    n_features: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Design A: train = N(0, I), test = N(0, 9·I) (volatility regime shift).

    RMT canary fixture. Discriminates on ``cov_`` via the std vector
    (test scale = 3× → cov_ scales by 9×). ``sigma2_`` and ``lambda_plus_``
    are scale-invariant artifacts of the correlation matrix and do NOT
    discriminate (both ≈ 1.0 and 1.7325 in train and test).
    """
    rng_train = np.random.default_rng(seed_train)
    rng_test = np.random.default_rng(seed_test)
    X_train = rng_train.standard_normal((n_samples, n_features)) * 1.0
    X_test = rng_test.standard_normal((n_samples, n_features)) * 3.0
    return X_train, X_test


def _build_correlation_injection_fixture(
    seed_train: int = 20260503,
    seed_test: int = 20260504,
    *,
    n_samples: int = 200,
    n_features: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Design B: train = uncorrelated, test = factor model with total spikes [5,3,2].

    LW canary fixture. Discriminates on ``shrinkage_intensity_`` via a
    structural sample-vs-target distance flip:

      δ̂_train ≈ 1.0  — uncorrelated train; sample S ≈ I and target
                       F = mean(diag(S)) · I ≈ I are nearly identical,
                       so γ̂ → 0 and δ̂ = π̂ / (T γ̂) saturates at 1.
      δ̂_test  ≈ 0.16 — correlated test with spike structure; F is a
                       poor target, γ̂ is large, sample is informative,
                       low shrinkage.

    LW δ̂ is *largely* scale-invariant — Design A would discriminate
    weakly and inconsistently on this metric (verified empirically in
    ``_pr2_threshold_measurement.py``: 3 of 5 seed pairs had
    |δ̂_diff|<0.05, 2 of 5 had it >0.10 due to clipping at δ̂=1).
    Design B's structural flip gives strong, consistent discrimination
    across all seeds.
    """
    rng_train = np.random.default_rng(seed_train)
    X_train = rng_train.standard_normal((n_samples, n_features))

    rng_test = np.random.default_rng(seed_test)
    spikes_added = [4.0, 2.0, 1.0]  # total spectrum [5,3,2] post +I lift
    K = len(spikes_added)
    A = rng_test.standard_normal((n_features, K))
    L, _ = np.linalg.qr(A)
    Sigma_test_true = L @ np.diag(spikes_added) @ L.T + np.eye(n_features)
    chol = np.linalg.cholesky(Sigma_test_true)
    Z = rng_test.standard_normal((n_samples, n_features))
    X_test = Z @ chol.T
    return X_train, X_test


def test_leakage_free_rmt_denoiser_persists_train_eigvals() -> None:
    """F-RP-006 contract: RMT cov_ persisted at fit; transform doesn't recompute.

    Three-clause canary structure mirrors F01
    (``test_leakage_free_pca_transform_reuses_train_col_means``):
      (a) Persistence: ``denoiser.cov_`` matches from-scratch
          ``denoise_covariance(X_train)`` byte-exact (atol=1e-12).
      (b) Discrimination: re-fit on X_test gives materially different
          ``cov_`` (relative Frobenius diff > 3.76 = 0.5 × measured min
          across 5 seed pairs).
      (c) Identity + artifact-byte-identical-after-transform:
          - ``transform(X_test)`` returns X_test unchanged.
          - ``cov_``, ``sigma2_``, ``eigvals_``, ``eigvecs_``, ``std_``,
            ``lambda_plus_``, ``eigvals_denoised_`` byte-identical
            before AND after a ``transform`` call (catches a buggy
            "transform recomputes from arriving X" defect).
    """
    X_train, X_test = _build_volatility_regime_fixture()
    denoiser = LeakageFreeRMTDenoiser().fit(X_train)

    # (a) Persistence
    expected_cov = denoise_covariance(X_train)
    np.testing.assert_allclose(
        denoiser.cov_,
        expected_cov,
        atol=1e-10,
        err_msg=(
            "LeakageFreeRMTDenoiser.cov_ does not match from-scratch "
            "denoise_covariance(X_train). F01-pattern persistence "
            "regression: see F-RP-006."
        ),
    )

    # (b) Discrimination
    denoiser_test_fit = LeakageFreeRMTDenoiser().fit(X_test)
    cov_diff = float(np.linalg.norm(denoiser.cov_ - denoiser_test_fit.cov_, ord="fro"))
    cov_norm = float(np.linalg.norm(denoiser.cov_, ord="fro"))
    rel_frob_diff = cov_diff / cov_norm
    assert rel_frob_diff > _RMT_DISCRIMINATION_PIN, (
        f"Fixture error: train-fit and test-fit cov_ differ by only "
        f"{rel_frob_diff:.4f} relative Frobenius (need > "
        f"{_RMT_DISCRIMINATION_PIN} to discriminate F-RP-006 leakage). "
        f"Check the volatility regime shift fixture (X_test scale should "
        f"be 3× X_train)."
    )

    # (c) Identity transform + artifact persistence under transform call
    cov_before = denoiser.cov_.copy()
    sigma2_before = denoiser.sigma2_
    q_train_before = denoiser.q_train_
    eigvals_before = denoiser.eigvals_.copy()
    eigvecs_before = denoiser.eigvecs_.copy()
    std_before = denoiser.std_.copy()
    lambda_plus_before = denoiser.lambda_plus_
    eigvals_denoised_before = denoiser.eigvals_denoised_.copy()

    out = denoiser.transform(X_test)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(X_test))

    np.testing.assert_array_equal(denoiser.cov_, cov_before)
    assert denoiser.sigma2_ == sigma2_before
    assert denoiser.q_train_ == q_train_before
    np.testing.assert_array_equal(denoiser.eigvals_, eigvals_before)
    np.testing.assert_array_equal(denoiser.eigvecs_, eigvecs_before)
    np.testing.assert_array_equal(denoiser.std_, std_before)
    assert denoiser.lambda_plus_ == lambda_plus_before
    np.testing.assert_array_equal(denoiser.eigvals_denoised_, eigvals_denoised_before)


def test_leakage_free_ledoit_wolf_persists_train_shrinkage() -> None:
    """F-RP-006 contract: LW cov_ + δ̂ persisted at fit; transform doesn't recompute.

    Three-clause canary structure mirrors F01:
      (a) Persistence: ``shrinker.cov_`` + ``shrinker.shrinkage_intensity_``
          match from-scratch ``ledoit_wolf_shrinkage(X_train)`` byte-exact.
      (b) Discrimination: re-fit on X_test (correlation-injected, vs
          uncorrelated train) gives materially different ``cov_`` AND δ̂.
          Mechanism: δ̂ flips from ~1.0 (uncorr train, F≈S → shrinkage
          saturates) to ~0.16 (correlated test, F poor target →
          shrinkage minimizes). Pins: rel_frob_diff > 0.42,
          delta_abs_diff > 0.32 (both 0.5 × measured min across 5 seed
          pairs).
      (c) Identity + artifact-byte-identical-after-transform:
          - ``transform(X_test)`` returns X_test unchanged.
          - ``cov_``, ``shrinkage_intensity_`` byte-identical before
            AND after a ``transform`` call.
    """
    X_train, X_test = _build_correlation_injection_fixture()
    shrinker = LeakageFreeLedoitWolfShrinkage().fit(X_train)

    # (a) Persistence
    expected_cov, expected_delta = ledoit_wolf_shrinkage(X_train)
    np.testing.assert_allclose(
        shrinker.cov_,
        expected_cov,
        atol=1e-10,
        err_msg=(
            "LeakageFreeLedoitWolfShrinkage.cov_ does not match from-"
            "scratch ledoit_wolf_shrinkage(X_train). F01-pattern "
            "persistence regression: see F-RP-006."
        ),
    )
    assert shrinker.shrinkage_intensity_ == pytest.approx(expected_delta, abs=1e-10)

    # (b) Discrimination
    shrinker_test_fit = LeakageFreeLedoitWolfShrinkage().fit(X_test)
    cov_diff = float(np.linalg.norm(shrinker.cov_ - shrinker_test_fit.cov_, ord="fro"))
    cov_norm = float(np.linalg.norm(shrinker.cov_, ord="fro"))
    rel_frob_diff = cov_diff / cov_norm
    delta_abs_diff = abs(shrinker.shrinkage_intensity_ - shrinker_test_fit.shrinkage_intensity_)
    assert rel_frob_diff > _LW_COV_DISCRIMINATION_PIN, (
        f"Fixture error: train-fit and test-fit cov_ differ by only "
        f"{rel_frob_diff:.4f} relative Frobenius (need > "
        f"{_LW_COV_DISCRIMINATION_PIN} to discriminate F-RP-006 leakage)."
    )
    assert delta_abs_diff > _LW_DELTA_DISCRIMINATION_PIN, (
        f"Fixture error: train-fit and test-fit δ̂ differ by only "
        f"{delta_abs_diff:.4f} absolute (need > "
        f"{_LW_DELTA_DISCRIMINATION_PIN} to discriminate F-RP-006 leakage)."
    )

    # (c) Identity transform + artifact persistence under transform call
    cov_before = shrinker.cov_.copy()
    delta_before = shrinker.shrinkage_intensity_

    out = shrinker.transform(X_test)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(X_test))

    np.testing.assert_array_equal(shrinker.cov_, cov_before)
    assert shrinker.shrinkage_intensity_ == delta_before


@pytest.mark.parametrize("seed", [20260503, 20260504, 20260505])
def test_leakage_free_rmt_denoiser_matches_pure_function(seed: int) -> None:
    """Regression-defense: adapter ``cov_`` matches ``denoise_covariance(X)`` byte-exact.

    The PR2 ``LeakageFreeRMTDenoiser`` duplicates the
    Σ→(D,corr)→eigh→denoise→recompose pipeline from
    ``denoise_covariance``. This test
    guards against drift between the two paths until the post-S19
    refactor that has the adapter delegate to a private helper.

    atol=1e-12 (tighter than the canary persistence tests at 1e-10)
    because ``adapter.cov_`` and ``denoise_covariance(X)`` are computed
    in the same process on identical data; any drift would indicate
    real implementation divergence, not platform/sklearn numerical
    noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, 20))
    adapter = LeakageFreeRMTDenoiser().fit(X)
    expected_cov = denoise_covariance(X)
    np.testing.assert_allclose(adapter.cov_, expected_cov, atol=1e-12)
