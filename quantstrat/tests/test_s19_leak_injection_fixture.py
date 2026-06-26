"""Unit tests for the S19 leak-injection fixture (PR4).

Seven tests. Six demanded by the PR4 design review plus
``test_label_sums_match_loop_form`` to pin the cumsum-vs-loop algebraic
equivalence after the vectorisation rewrite.

  1. ``test_leak_rate_matches_closed_form``   — r̄ = 2(h-1)(K-1)/(T-h+1)
  2. ``test_leak_rate_zero_with_purgedkfold`` — naming-distinction
                                                 sanity (rate on PurgedKFold
                                                 train_idx is 0)
  3. ``test_returns_have_factor_structure``   — sample eigenvalue shape
  4. ``test_t1_is_deterministic_vertical``    — t1 = t0 + h - 1
  5. ``test_knn_delta_r2_matches_reference``  — kNN ΔR² ∈ [0.85, 0.95]
  6. ``test_seed_determinism``                — byte-identical at the
                                                 canonical seed
  7. ``test_label_sums_match_loop_form``      — cumsum vs explicit loop
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

from quantcore.cv import PurgedKFold
from quantcore.uncertainty.conformal.finance import (
    ConformalAlphaModel,
    backtest_alpha_model,
)

# The fixture lives in tests/spikes/ (not auto-collected by pytest because
# the filename does not match test_*). Load it explicitly so the test
# module has stable imports.
_FIXTURE_PATH = Path(__file__).parent / "spikes" / "s19_leak_injection_fixture.py"
_spec = importlib.util.spec_from_file_location("s19_leak_injection_fixture", _FIXTURE_PATH)
assert _spec is not None and _spec.loader is not None, f"Cannot load {_FIXTURE_PATH}"
_module = importlib.util.module_from_spec(_spec)
sys.modules["s19_leak_injection_fixture"] = _module
_spec.loader.exec_module(_module)

build_leak_injection_panel = _module.build_leak_injection_panel
build_knn_on_monotone_panel = _module.build_knn_on_monotone_panel
_compute_leak_rate_at_embargo_0 = _module._compute_leak_rate_at_embargo_0
_count_overlapping_test_events = _module._count_overlapping_test_events


# ---------------------------------------------------------------------------
# 1. Closed-form leak-rate equivalence
# ---------------------------------------------------------------------------


def _closed_form_overlap_rate(T: int, h: int, K: int) -> float:
    """r̄ = 2(h-1)(K-1) / (T-h+1) — see Section 4 of the sprint plan."""
    return 2.0 * (h - 1) * (K - 1) / (T - h + 1)


@pytest.mark.parametrize(
    "T,h,K",
    [
        (2520, 11, 10),  # canonical
        (1260, 11, 10),  # half the sample
        (2520, 21, 10),  # double the horizon
    ],
)
def test_leak_rate_matches_closed_form(T: int, h: int, K: int) -> None:
    """Empirical structural overlap rate matches r̄ = 2(h-1)(K-1)/(T-h+1).

    With vertical-only barriers (``t1 = t0 + h - 1`` deterministic), the
    rate is purely structural — no dependence on returns or seed. Each
    case is even (n_events = T-h+1 is divisible by K), so the closed
    form is byte-exact, not approximate.
    """
    panel = build_leak_injection_panel(n_obs=T, horizon=h, seed=20260502)
    rate = _compute_leak_rate_at_embargo_0(panel["events"], n_splits=K)
    expected = _closed_form_overlap_rate(T, h, K)
    assert abs(rate["mean"] - expected) < 1e-12, (
        f"Empirical mean {rate['mean']!r} != closed form {expected!r} "
        f"(T={T}, h={h}, K={K}). Δ = {rate['mean'] - expected!r}."
    )


# ---------------------------------------------------------------------------
# 2. PurgedKFold(embargo_pct=0).split() returns purged train_idx — rate is 0
# ---------------------------------------------------------------------------


def test_leak_rate_zero_with_purgedkfold() -> None:
    """PurgedKFold(embargo_pct=0).split() purges the overlap; rate is 0.

    Pins the load-bearing distinction in
    ``_compute_leak_rate_at_embargo_0``'s docstring: the function
    measures the rate naive K-fold WOULD have, NOT the rate after
    PurgedKFold has already closed it. Recomputing the overlap rate
    on PurgedKFold's already-purged ``train_idx`` must yield 0 on
    every fold — confirming the naming concern is real.
    """
    panel = build_leak_injection_panel(seed=20260502)
    events = panel["events"]
    pk = PurgedKFold(n_splits=10, t1=events["t1"], embargo_pct=0.0)

    dummy_X = np.zeros((len(events), 1))
    per_fold_rates: list[float] = []
    for train_idx, test_idx in pk.split(dummy_X):
        leaked = _count_overlapping_test_events(events, train_idx, test_idx)
        per_fold_rates.append(float(leaked) / float(len(test_idx)))

    assert max(per_fold_rates) == 0.0, (
        f"PurgedKFold(embargo_pct=0) leaked: per-fold rates = {per_fold_rates}"
    )


# ---------------------------------------------------------------------------
# 3. Returns reproduce the spike + bulk eigenvalue structure
# ---------------------------------------------------------------------------


def test_returns_have_factor_structure() -> None:
    """Top-K sample eigenvalues track Λ + σ²; bulk sits within MP edges.

    With T=2520, N=20, q = N/T ≈ 0.0079, the MP edges are
    (1 ± √q)² ≈ [0.829, 1.186]. Spike eigenvalues (51, 31, 21) sit far
    above the bulk and are well-resolved at 10% relative tolerance.
    """
    panel = build_leak_injection_panel(seed=20260502)
    returns = panel["returns"]
    sample_cov = np.cov(returns, rowvar=False, ddof=1)
    sample_eigs = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]

    spikes = np.array(panel["metadata"]["spike_lambdas"])
    sigma2 = panel["metadata"]["sigma2"]
    expected_top = spikes + sigma2

    np.testing.assert_allclose(sample_eigs[: len(spikes)], expected_top, rtol=0.10)

    n_assets, n_obs = sample_cov.shape[0], returns.shape[0]
    q = n_assets / n_obs
    mp_lower = (1.0 - np.sqrt(q)) ** 2
    mp_upper = (1.0 + np.sqrt(q)) ** 2

    bulk = sample_eigs[len(spikes) :]
    # MP edges are population limits; finite-sample bulk extends slightly
    # past the edges. 30% slack on each edge is the conventional finite-T
    # tolerance and keeps the test robust to seed-level noise.
    assert bulk.min() >= 0.7 * mp_lower * sigma2, (
        f"Bulk min {bulk.min():.4f} below 0.7 × σ² × λ₋ = {0.7 * mp_lower * sigma2:.4f}"
    )
    assert bulk.max() <= 1.3 * mp_upper * sigma2, (
        f"Bulk max {bulk.max():.4f} above 1.3 × σ² × λ₊ = {1.3 * mp_upper * sigma2:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Vertical-only t1 invariant
# ---------------------------------------------------------------------------


def test_t1_is_deterministic_vertical() -> None:
    """t1[i] == t0[i] + h - 1 exactly for every event.

    Vertical-only barriers are doctrinal for the synthetic panel
    (Section 4). This invariant is what makes the leak rate a pure
    function of (T, h, K) and the closed-form prediction byte-exact.
    """
    panel = build_leak_injection_panel(seed=20260502)
    events = panel["events"]
    h = panel["metadata"]["horizon"]
    t0 = events.index.to_numpy()
    t1 = events["t1"].to_numpy()
    np.testing.assert_array_equal(t1, t0 + h - 1)


# ---------------------------------------------------------------------------
# 5. kNN ΔR² between embargo=0 and embargo=h+1 — F-RP-005 backup
# ---------------------------------------------------------------------------


def _r2_from_signals(results: dict[str, Any]) -> float:
    """OOS R² over the walk-forward predictions, matching the reference test
    at ``quantcore/tests/test_alpha_embargo.py:48-54``."""
    preds = np.array([float(s.expected_return[0]) for s in results["signals"]])
    actuals = np.asarray(results["returns"], dtype=np.float64)
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def test_knn_delta_r2_matches_reference() -> None:
    """ΔR² between embargo=0 and embargo=h+1 ≈ 0.30 on the production walk-forward.

    Calls the production path directly: ``backtest_alpha_model``
    (refit_frequency=21) wrapping ``ConformalAlphaModel(method="split",
    random_state=42)`` with a ``KNeighborsRegressor(n_neighbors=3)``
    base. This makes test 5 a regression on the F-RP-001 fix's
    continued embargo signal — not an independent re-derivation of
    walk-forward kNN semantics.

    Empirical anchor:

      * seed=20260502, n=2000, h=10:  ΔR² = 0.2998
      * seed=42, n=500, h=10 (matches the published reference test):
        ΔR² = 0.2507

    The 0.049 spread across seeds means a ±0.05 bracket would break on
    silent RNG / sklearn-default drift; the test brackets ±0.10 around
    the canonical seed=20260502 value to leave headroom for the
    seed-level variance while still failing on a meaningful semantic
    regression.

    Production-path version dependence (READ BEFORE TUNING THE BRACKET):
    ``backtest_alpha_model``'s embargo=0 branch applies the t1 filter
    at ``alpha.py:1027`` (admits events with ``t1 <= t - 1``);
    ``ConformalAlphaModel(method="split")`` halves the effective
    training set via the calibration holdout; both effects compound
    into the empirical ≈ 0.30 signal. Changes to either's internals
    (filter logic, split-fraction, sklearn KNeighborsRegressor
    defaults) require re-baselining via the decomposition probe in the
    decision doc — do NOT widen the bracket post-hoc to absorb a
    semantic regression.

    The plan's original "ΔR² ≈ 0.92" claim was uncalibrated and is
    superseded by this empirical anchor; recalibration audit trail in
    the decision doc.
    """
    panel = build_knn_on_monotone_panel(n_obs=2000, n_features=1, horizon=10, seed=20260502)
    features = panel["features"]
    labels = panel["labels"]
    t1_arr = panel["t1"]
    h = panel["metadata"]["horizon"]

    base = KNeighborsRegressor(n_neighbors=3)

    model_leak = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)
    res_leak = backtest_alpha_model(
        model_leak,
        features,
        labels,
        initial_train_size=100,
        refit_frequency=21,
        t1=t1_arr,
        embargo=0,
    )
    model_safe = ConformalAlphaModel(base, alpha=0.1, method="split", random_state=42)
    res_safe = backtest_alpha_model(
        model_safe,
        features,
        labels,
        initial_train_size=100,
        refit_frequency=21,
        t1=t1_arr,
        embargo=h + 1,
    )

    r2_leak = _r2_from_signals(res_leak)
    r2_safe = _r2_from_signals(res_safe)
    delta = r2_leak - r2_safe

    assert 0.20 <= delta <= 0.40, (
        f"ΔR² = {delta:.4f} not in [0.20, 0.40] "
        f"(r2_leak={r2_leak:.4f}, r2_safe={r2_safe:.4f}). "
        f"Production-path drift suspected — see docstring + decision doc."
    )


# ---------------------------------------------------------------------------
# 6. Seed determinism — canonical seed produces a stable byte stream
# ---------------------------------------------------------------------------


def test_seed_determinism() -> None:
    """Two builds with seed=20260502 produce byte-identical returns and true_cov.

    Defends against silent RNG-stream drift (numpy upgrades, factor-form
    re-orderings). The factor-form sampler used in build_leak_injection_panel
    is the canonical stream; if a future refactor changes the order of
    rng.standard_normal calls or the QR sign convention, this test fails
    and forces a conscious bytes-regen.
    """
    p1 = build_leak_injection_panel(seed=20260502)
    p2 = build_leak_injection_panel(seed=20260502)
    np.testing.assert_array_equal(p1["returns"], p2["returns"])
    np.testing.assert_array_equal(p1["true_cov"], p2["true_cov"])

    p3 = build_leak_injection_panel(seed=20260502 + 1)
    assert not np.array_equal(p1["returns"], p3["returns"]), (
        "Different seeds produced identical returns — RNG plumbing is broken."
    )


# ---------------------------------------------------------------------------
# 7. cumsum vectorisation matches the explicit loop (rtol=0, atol=1e-12)
# ---------------------------------------------------------------------------


def test_label_sums_match_loop_form() -> None:
    """Cumsum prefix-sum trick agrees with the explicit loop form bit-tight.

    Pins the algebraic equivalence after replacing
    ``np.array([noise[i:i+h].sum() for i in range(n)])`` with the
    cumsum form ``csum[h:n+h] - csum[:n]`` in build_knn_on_monotone_panel
    (and the analogous EW-returns variant in build_leak_injection_panel).
    rtol=0 / atol=1e-12 is bit-tight; any drift would be a real algebra
    bug, not floating-point rounding.
    """
    rng = np.random.default_rng(seed=20260502)
    h = 10
    n = 2000

    noise = rng.standard_normal(n + h)
    csum = np.concatenate(([0.0], np.cumsum(noise)))
    vec = csum[h : n + h] - csum[:n]
    loop = np.array([noise[i : i + h].sum() for i in range(n)], dtype=np.float64)

    np.testing.assert_allclose(vec, loop, rtol=0.0, atol=1e-12)

    knn_panel = build_knn_on_monotone_panel(n_obs=n, n_features=1, horizon=h, seed=20260502)
    np.testing.assert_allclose(knn_panel["labels"], loop, rtol=0.0, atol=1e-12)

    inj = build_leak_injection_panel(n_obs=512, horizon=h, seed=20260502)
    ew = inj["returns"].mean(axis=1)
    n_events = len(inj["events"])
    csum_ew = np.concatenate(([0.0], np.cumsum(ew)))
    cum_h_ret_vec = csum_ew[h : n_events + h] - csum_ew[:n_events]
    cum_h_ret_loop = np.array([ew[i : i + h].sum() for i in range(n_events)], dtype=np.float64)
    np.testing.assert_allclose(cum_h_ret_vec, cum_h_ret_loop, rtol=0.0, atol=1e-12)
    expected_labels = pd.Series(
        np.sign(cum_h_ret_loop), index=inj["events"].index, name="label", dtype=np.float64
    )
    pd.testing.assert_series_equal(inj["labels"], expected_labels)
