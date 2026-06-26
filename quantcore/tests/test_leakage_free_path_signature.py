"""Leakage-canary regression tests for :class:`PathSignatureTransformer`.

Three pins per S18 plan §9:

* **Pin A — per-level stat isolation (F01 closure).** Mirrors the
  :mod:`quantcore.tests.test_leakage_free_pca_coverage` precedent.
  Asserts that ``level_means_`` / ``level_stds_`` are computed from
  training data only AND that ``transform`` reuses those stats
  byte-exact on disjoint test data with a deliberately different
  volatility regime. The "leakage path" reconstruction (re-fitting
  stats on the test data) must produce a different output than the
  honest ``transform``.

* **Pin B — strict causality byte-exact.** Asserts that mutating
  ``X[event_t+1:]`` does not change the feature for the event at
  index ``event_t``. Three mutation cases (NaN-fill, random-replace,
  drop-tail) test different defect classes. ``np.array_equal`` (NOT
  ``np.allclose``) is the canonical comparator — any future-data
  leakage produces a non-zero diff at the bit level.

* **Pin C — lead-lag orientation.** Asserts the level-2 antisymmetric
  component ``sig_l2_c0_c1 - sig_l2_c1_c0`` is strictly positive when
  channel 0 leads channel 1 in the augmented path, and strictly
  negative under channel reversal. The ``1e-6`` magnitude floor guards
  against floating-point noise that could let a sign-only assertion
  pass spuriously. Catches one-bit-flip lead-lag wiring defects (e.g.,
  ``lag = X[t+1]`` instead of ``X[t-1]``) that don't show up in
  coverage or accuracy metrics. Tagged ``@pytest.mark.lead_lag_optional``
  because the S18 FB1 spike demoted ``"lead-lag"``
  from the production default; the test still runs by default but the
  marker allows ``pytest -m "not lead_lag_optional"`` filtering.

All three pins are failing-pre-fix and passing-post-fix by
construction:

* Swapping the F01-closure ``self.level_means_`` reference for a
  freshly-recomputed array would fail Pin A.
* Reading any ``X.iloc[event_t+1:]`` row inside transform would fail
  Pin B.
* Inverting lead-lag orientation (lag-as-future) would fail Pin C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from quantcore.cv.purged_kfold import PurgedKFold
from quantcore.importance.importance import (
    feature_importance_mda,
    importance_gate,
)
from quantcore.preprocessing.path_signature import PathSignatureTransformer
from quantcore.preprocessing.transformers import (
    LeakageFreeNaNHandler,
    LeakageFreePCA,
    LeakageFreePipeline,
    LeakageFreeStandardScaler,
)
from quantcore.uncertainty.conformal.finance.alpha import ConformalAlphaModel


# ----------------------------------------------------------------------
# Fixture constants
# ----------------------------------------------------------------------

WINDOW_SIZE = 64
DEPTH = 3
N_BARS = 200
SEED = 20260430


# ----------------------------------------------------------------------
# Synthesis helper
# ----------------------------------------------------------------------


def _synth_ohlcv(*, n: int, seed: int, vol: float = 0.005, ts_start: str) -> pd.DataFrame:
    """Synthesize an ``(n, 5)`` OHLCV bar fixture with controllable vol.

    Log-returns are drawn ``N(0, vol²)`` IID per channel; prices are
    cumulative-product GBM. Volume is uniform ``U(50, 500)``. Index is
    a 1-minute ``DatetimeIndex`` starting at ``ts_start``.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(ts_start, periods=n, freq="min")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, vol, size=(n, 4)), axis=0))
    return pd.DataFrame(
        {
            "open": prices[:, 0],
            "high": prices[:, 1],
            "low": prices[:, 2],
            "close": prices[:, 3],
            "volume": rng.uniform(50, 500, size=n),
        },
        index=ts,
    )


# ----------------------------------------------------------------------
# Pin A — per-level stat isolation (F01 closure)
# ----------------------------------------------------------------------


def test_pin_a_per_level_stat_isolation() -> None:
    """level_means_ / level_stds_ are train-fit and reused at transform.

    Four-step proof:

    1. ``level_means_`` / ``level_stds_`` populated by ``fit`` match a
       byte-exact from-scratch reconstruction using the same primitives
       as :meth:`PathSignatureTransformer._fit_rescaling`.
    2. Sanity: a parallel transformer fit on the high-vol test data
       produces materially different stats — the regimes are not
       coincidentally aligned.
    3. ``transform(test)`` is byte-exact equal to a manual
       reconstruction that applies train-fit stats to test signatures.
    4. Output is **not** close to a leakage-path reconstruction that
       applies test-fit stats to test signatures (the F01 defect mode).
    """
    df_train = _synth_ohlcv(n=N_BARS, seed=SEED, vol=0.005, ts_start="2026-01-02 09:30:00")
    df_test = _synth_ohlcv(n=N_BARS, seed=SEED + 1, vol=0.015, ts_start="2026-01-03 09:30:00")

    t = PathSignatureTransformer(window_size=WINDOW_SIZE, depth=DEPTH)
    t.fit(df_train)

    d_aug = len(t.augmented_channel_names_)
    offsets = t._level_offsets(d_aug)
    n_features_out = len(t.feature_names_out_)

    # --- Step 1: from-scratch reconstruction of train stats ----------------
    windows_train = t._extract_windows(df_train)
    sigs_train = t._compute_signatures(windows_train, n_features_out=n_features_out)
    expected_means = np.array(
        [float(sigs_train[:, offsets[k] : offsets[k + 1]].mean()) for k in range(DEPTH)],
        dtype=np.float64,
    )
    expected_stds = np.array(
        [
            max(float(sigs_train[:, offsets[k] : offsets[k + 1]].std(ddof=1)), 1e-12)
            for k in range(DEPTH)
        ],
        dtype=np.float64,
    )
    assert t.level_means_ is not None
    assert t.level_stds_ is not None
    assert np.array_equal(t.level_means_, expected_means), (
        "level_means_ deviates from from-scratch reconstruction — fit() and "
        "_fit_rescaling have drifted apart."
    )
    assert np.array_equal(t.level_stds_, expected_stds), (
        "level_stds_ deviates from from-scratch reconstruction — fit() and "
        "_fit_rescaling have drifted apart."
    )

    # --- Step 2: regimes produce materially different stats ----------------
    t_test_only = PathSignatureTransformer(window_size=WINDOW_SIZE, depth=DEPTH)
    t_test_only.fit(df_test)
    assert t_test_only.level_means_ is not None
    assert t_test_only.level_stds_ is not None
    assert not np.allclose(t.level_means_, t_test_only.level_means_), (
        "Train and test regimes produced near-identical level_means_ — "
        "the vol shift is not surfacing in the per-level statistics; the "
        "Pin A discrimination is inconclusive."
    )
    assert not np.allclose(t.level_stds_, t_test_only.level_stds_), (
        "Train and test regimes produced near-identical level_stds_ — "
        "the vol shift is not surfacing in the per-level statistics; the "
        "Pin A discrimination is inconclusive."
    )

    # --- Step 3: transform(test) uses train-fit stats ----------------------
    windows_test = t._extract_windows(df_test)
    sigs_test = t._compute_signatures(windows_test, n_features_out=n_features_out)
    honest_output = sigs_test.copy()
    for k in range(DEPTH):
        honest_output[:, offsets[k] : offsets[k + 1]] = (
            honest_output[:, offsets[k] : offsets[k + 1]] - t.level_means_[k]
        ) / t.level_stds_[k]
    actual_output = t.transform(df_test).to_numpy()
    assert np.array_equal(actual_output, honest_output), (
        "transform(test) is not byte-exact equal to manual reconstruction "
        "with train-fit stats — F01 closure violated."
    )

    # --- Step 4: output is NOT the leakage-path reconstruction -------------
    leakage_output = sigs_test.copy()
    for k in range(DEPTH):
        leakage_output[:, offsets[k] : offsets[k + 1]] = (
            leakage_output[:, offsets[k] : offsets[k + 1]] - t_test_only.level_means_[k]
        ) / t_test_only.level_stds_[k]
    assert not np.allclose(actual_output, leakage_output), (
        "transform(test) is suspiciously close to the leakage-path "
        "reconstruction (test-fit stats applied to test sigs). F01 closure "
        "may not be discriminating between honest and leaky outputs on "
        "this fixture."
    )


# ----------------------------------------------------------------------
# Pin B — strict causality byte-exact
# ----------------------------------------------------------------------


def test_pin_b_strict_causality_byte_exact() -> None:
    """Mutating ``X[event_t+1:]`` does not change the feature for ``event_t``.

    Three mutation cases test different leakage defect classes. The
    feature for the event at index ``event_t`` lives at output row
    ``event_t - window_size + 1`` and depends only on
    ``X[event_t - window_size + 1 : event_t + 1]``. Any path inside
    ``transform`` that reads ``X[event_t+1:]`` — through stale index
    references, lookahead-padded windows, post-event normalization,
    etc. — produces a non-zero diff at the bit level.

    ``np.array_equal`` (NOT ``allclose``) is intentional: a leak that
    shifts output by ``1e-15`` is still a leak and must fail the test.
    """
    n = 100  # smaller than Pin A's 200; gives 37 windows post-W=64
    event_t = 70
    output_row_idx = event_t - WINDOW_SIZE + 1  # 7

    df_full = _synth_ohlcv(n=n, seed=SEED, vol=0.005, ts_start="2026-01-02 09:30:00")

    t = PathSignatureTransformer(window_size=WINDOW_SIZE, depth=DEPTH)
    t.fit(df_full)

    # Reference feature for the event at event_t.
    feat_ref = t.transform(df_full).to_numpy()[output_row_idx].copy()

    rng = np.random.default_rng(SEED + 100)

    # --- Case 1: NaN-fill X[event_t+1:] -----------------------------------
    df_nan = df_full.copy()
    df_nan.iloc[event_t + 1 :] = np.nan
    feat_nan = t.transform(df_nan).to_numpy()[output_row_idx]
    assert np.array_equal(feat_ref, feat_nan), (
        f"Pin B failed under NaN-fill mutation: feature at event_t={event_t} "
        "changed when X[event_t+1:] was set to NaN. Some path inside "
        "transform is reading post-event rows."
    )

    # --- Case 2: random-replace X[event_t+1:] (large-magnitude noise) -----
    # Multiplied by 1000 so any leakage signal is far above floating-point
    # quantization noise — a small leak would still fail array_equal even
    # at modest magnitude, but the large value makes the failure mode
    # more diagnostic.
    df_rand = df_full.copy()
    n_tail = n - (event_t + 1)
    df_rand.iloc[event_t + 1 :] = rng.standard_normal(size=(n_tail, df_full.shape[1])) * 1000.0
    feat_rand = t.transform(df_rand).to_numpy()[output_row_idx]
    assert np.array_equal(feat_ref, feat_rand), (
        f"Pin B failed under random-replace mutation: feature at event_t="
        f"{event_t} changed when X[event_t+1:] was overwritten with "
        "large-magnitude random noise."
    )

    # --- Case 3: drop X[event_t+1:] entirely ------------------------------
    df_drop = df_full.iloc[: event_t + 1].copy()
    feat_drop = t.transform(df_drop).to_numpy()[output_row_idx]
    assert np.array_equal(feat_ref, feat_drop), (
        f"Pin B failed under drop-tail mutation: feature at event_t="
        f"{event_t} changed when X was truncated to .iloc[:event_t+1]. "
        "transform may have an off-by-one or stale-reference defect."
    )


# ----------------------------------------------------------------------
# Pin C — lead-lag orientation
# ----------------------------------------------------------------------


@pytest.mark.lead_lag_optional
def test_pin_c_lead_lag_orientation() -> None:
    """Lead-lag-augmented signatures encode lead direction at level 2.

    For a 2-channel path where channel 0 leads channel 1, the level-2
    antisymmetric component (Lévy area) ``sig_l2_c0_c1 - sig_l2_c1_c0``
    is strictly positive. Reversing the channels flips the sign exactly.

    Magnitude floor of ``1e-6`` defends against floating-point noise
    on near-zero values (matches the ``LeakageFreeStandardScaler``
    σ-floor pattern at signature scale). Sign-only assertion (``> 0``)
    can pass on noise that happens to land positive — a one-bit-flip
    in the lead-lag wiring (e.g., reading ``X[t+1]`` instead of
    ``X[t-1]``) would silently invert the sign without changing the
    coverage / accuracy metrics, which is the failure class this pin
    is built to catch.

    Path geometry: channel 0 ramps from 1.0 to 1.7 in the first half,
    plateau in the second half; channel 1 plateau in the first half,
    ramps from 1.0 to 1.7 in the second half. The L-shaped 2D path
    encloses a positive Lévy area on the right side (going right
    along the bottom edge, then up along the right edge).
    """
    W = 8
    half = W // 2

    c0 = np.empty(W, dtype=np.float64)
    c1 = np.empty(W, dtype=np.float64)
    for i in range(half):
        c0[i] = 1.0 + (i / max(half - 1, 1)) * 0.7
        c1[i] = 1.0
    for i in range(half, W):
        c0[i] = 1.7
        c1[i] = 1.0 + ((i - half) / max(W - half - 1, 1)) * 0.7

    ts = pd.date_range("2026-01-02 09:30:00", periods=W, freq="min")

    def _signed_levy_area(path_df: pd.DataFrame) -> float:
        # Use rescaling="none" so the absolute Lévy sign is read raw —
        # rescaling="post" on a single window standardizes to mean=0
        # which would erase the sign signal at the level block aggregate.
        transformer = PathSignatureTransformer(
            depth=2,
            augmentations=("basepoint", "addtime", "lead-lag"),
            rescaling="none",
            window_size=W,
            path_columns=("c0_raw", "c1_raw"),
        )
        transformer.fit(path_df)
        feats = transformer.transform(path_df).iloc[0]
        return float(feats["sig_l2_c0_c1"] - feats["sig_l2_c1_c0"])

    # --- c0 leads c1: Lévy area > 1e-6 -------------------------------------
    df = pd.DataFrame({"c0_raw": c0, "c1_raw": c1}, index=ts)
    levy = _signed_levy_area(df)
    assert levy > 1e-6, (
        f"Pin C failed (c0 leads c1): sig_l2_c0_c1 - sig_l2_c1_c0 = {levy}; "
        "expected > 1e-6 for a 2D path where channel 0 leads channel 1. "
        "Lead-lag may be wired with lag = X[t+1] instead of X[t-1] — a "
        "one-bit-flip silent failure that does not show up in coverage "
        "or accuracy metrics."
    )

    # --- c1 leads c0 (channels reversed): Lévy area < -1e-6 ----------------
    df_rev = pd.DataFrame({"c0_raw": c1, "c1_raw": c0}, index=ts)
    levy_rev = _signed_levy_area(df_rev)
    assert levy_rev < -1e-6, (
        f"Pin C failed (c1 leads c0): sig_l2_c0_c1 - sig_l2_c1_c0 = {levy_rev}; "
        "expected < -1e-6 with channels reversed. Sign-symmetry violation "
        "indicates an asymmetry in the lead-lag construction."
    )


# ----------------------------------------------------------------------
# End-to-end pipeline composition smoke
# ----------------------------------------------------------------------


def test_pipeline_composition_end_to_end() -> None:
    """End-to-end wiring smoke through the full S18 pipeline.

    Wires
    ``PathSignatureTransformer → LeakageFreeNaNHandler →
    LeakageFreeStandardScaler → LeakageFreePCA(0.95) → MDA →
    importance_gate → ConformalAlphaModel.fit → predict``
    on a 500-bar synthetic OHLCV fixture, asserting only that
    every component composes correctly: no exceptions, output
    shapes match contracts, no NaN/inf, ``signal.lower <=
    signal.upper`` everywhere. ``gate_passed`` is allowed to be
    either ``True`` or ``False`` — this is a wiring test, not a
    quality test.
    """
    n = 500
    train_frac = 0.7
    seed = 20260501

    df = _synth_ohlcv(n=n, seed=seed, vol=0.005, ts_start="2026-01-02 09:30:00")

    # Forward-1-bar log-return target — same alignment scheme as
    # tests/spikes/fb1_leadlag_fixture.build_fb1_targets but inlined here
    # to keep the smoke test self-contained and clear about its inputs.
    log_close = np.log(df["close"].to_numpy(dtype=np.float64))
    forward_log_ret = log_close[1:] - log_close[:-1]  # length n - 1
    target_idx = df.index[WINDOW_SIZE - 1 : -1]
    targets = pd.Series(forward_log_ret[WINDOW_SIZE - 1 :], index=target_idx, name="next_log_ret")

    # t1 series for PurgedKFold (event-end = next bar timestamp).
    t1 = pd.Series(df.index[WINDOW_SIZE:].to_numpy(), index=target_idx, name="t1")

    # Chronological train/test split on the EVENT axis.
    n_events = len(targets)
    split_idx = int(n_events * train_frac)
    train_event_idx = targets.index[:split_idx]
    test_event_idx = targets.index[split_idx:]

    # Map back to the bar slice each event needs (window_size - 1 bars
    # before the event for the lookback window).
    bars_train = df.loc[: train_event_idx[-1]]
    bars_test = df  # full path; test rows extracted via index slice below

    # Pipeline with the post-FB1 default augmentations (lead-lag opt-in
    # only; the smoke test exercises the production path).
    pipeline = LeakageFreePipeline(
        steps=[
            ("path_sig", PathSignatureTransformer(depth=2, window_size=WINDOW_SIZE)),
            ("nan", LeakageFreeNaNHandler(strategy="median")),
            ("scale", LeakageFreeStandardScaler(clip_outliers=5.0)),
            ("pca", LeakageFreePCA(n_components=0.95)),
        ]
    )
    pipeline.fit(bars_train)
    features_train_full = pipeline.transform(bars_train)
    features_full = pipeline.transform(bars_test)
    features_test = features_full.loc[test_event_idx]

    # Sanity: pipeline output is finite and aligned with the event index.
    assert np.isfinite(features_train_full.to_numpy()).all()
    assert np.isfinite(features_test.to_numpy()).all()
    assert (features_train_full.index == train_event_idx).all()
    assert (features_test.index == test_event_idx).all()

    # MDA importance + gate on the training features. Small RF +
    # n_repeats=2 for speed; the gate threshold is t_stat=2.0 (default).
    cv = PurgedKFold(n_splits=3, t1=t1.loc[train_event_idx], embargo_pct=0.01)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, n_jobs=-1, random_state=seed)
    mda = feature_importance_mda(
        rf,
        features_train_full,
        targets.loc[train_event_idx],
        cv,
        scoring="neg_mean_squared_error",
        n_repeats=2,
        random_state=seed,
    )
    selected, gate_passed = importance_gate({"mda": mda}, min_features=1, t_stat=2.0)

    # Wiring assertions — no quality assertions.
    assert isinstance(gate_passed, bool)
    assert isinstance(selected, list)

    # ConformalAlphaModel on the training features → predict on test.
    alpha_model = ConformalAlphaModel(rf, alpha=0.1, method="split", random_state=seed)
    alpha_model.fit(features_train_full.to_numpy(), targets.loc[train_event_idx].to_numpy())
    signal = alpha_model.predict(features_test.to_numpy())

    # AlphaSignal contract: shape + finiteness + interval ordering.
    assert signal.expected_return.shape == (len(features_test),)
    assert signal.lower.shape == (len(features_test),)
    assert signal.upper.shape == (len(features_test),)
    assert np.isfinite(signal.expected_return).all()
    assert np.isfinite(signal.lower).all()
    assert np.isfinite(signal.upper).all()
    assert (signal.lower <= signal.upper).all()
