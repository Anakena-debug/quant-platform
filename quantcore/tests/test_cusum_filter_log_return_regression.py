"""Regression tests for `labels/labelling.py::cusum_filter` under the
log-return semantic migration (S31, 2026-05-17).

Pins:

  - Threshold ≤ 0 raises `ValueError` (pre-S31: silently returned an
    empty index because no sum could ever cross a non-positive bar).
  - Non-positive / non-finite close prices raise `ValueError` (pre-S31:
    silently injected `NaN` / ±inf through `np.log` and accumulated
    non-finite state, never firing or firing pathologically).
  - Empty input returns an empty `DatetimeIndex` (degenerate-but-valid
    case used by streaming / windowed callers).
  - Non-monotonic index raises `ValueError` — events on a shuffled
    index are temporally meaningless. Matches `get_daily_vol`'s index
    contract (`_assert_daily_or_lower`).
  - Cumulative-surprise math is on log-returns (additive over a path)
    rather than `pct_change` (non-additive: `-50 %` then `+50 %` sums
    to zero but ends at `-25 %`). The path-additivity test below is the
    audit-grade discriminator vs. `pct_change`.
  - Scale invariance is preserved but is NOT a `pct_change`
    discriminator (simple returns are also scale-invariant) — it
    discriminates *log* and *pct* together against raw price-difference
    CUSUM (`close.diff()`), whose threshold would scale with price.
  - Strict `>` / `<` trigger preserved across the migration: a
    cumulative log-return exactly equal to `threshold` does NOT fire.
  - V1 composition-spike fixture pins event-count drift attributable
    to the migration: under `CUSUM_THRESHOLD = 0.01` on the V1 daily
    DGP, the per-seed event counts shift by at most ±2 (well inside
    the [350, 450] pin band in `test_pin2_event_count_bounded`).

§0 provenance — pinned 2026-05-17, deterministic re-execution in
`quantcore/.venv` (python 3.11.14, numpy 2.4.4, pandas 3.0.2):

    Pre-S31 (`close.pct_change()` semantics):
        seed=42: 412   seed=7:    406   seed=123: 408
        seed=2026: 422 seed=4321: 386
    Post-S31 (`np.log(close).diff()` semantics) — pinned below:
        seed=42: 413   seed=7:    405   seed=123: 407
        seed=2026: 420 seed=4321: 386
    Per-seed drift: [+1, -1, -1, -2, 0]. Pin band [350, 450] holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.labels.labelling import cusum_filter


# =====================================================================
# Fixture builders.
# =====================================================================


def _flat_close(n: int = 100, level: float = 100.0) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(np.full(n, level, dtype=np.float64), index=idx, name="close")


# =====================================================================
# Validation guards.
# =====================================================================


def test_threshold_zero_raises() -> None:
    close = _flat_close()
    with pytest.raises(ValueError, match="threshold must be > 0"):
        cusum_filter(close, threshold=0.0)


def test_threshold_negative_raises() -> None:
    close = _flat_close()
    with pytest.raises(ValueError, match="threshold must be > 0"):
        cusum_filter(close, threshold=-0.01)


def test_non_positive_price_raises_on_zero() -> None:
    close = _flat_close()
    close.iloc[10] = 0.0
    with pytest.raises(ValueError, match="strictly-positive finite prices"):
        cusum_filter(close, threshold=0.01)


def test_non_positive_price_raises_on_negative() -> None:
    close = _flat_close()
    close.iloc[10] = -1.0
    with pytest.raises(ValueError, match="strictly-positive finite prices"):
        cusum_filter(close, threshold=0.01)


def test_non_finite_price_raises_on_nan() -> None:
    close = _flat_close()
    close.iloc[10] = np.nan
    with pytest.raises(ValueError, match="strictly-positive finite prices"):
        cusum_filter(close, threshold=0.01)


def test_non_finite_price_raises_on_pos_inf() -> None:
    close = _flat_close()
    close.iloc[10] = np.inf
    with pytest.raises(ValueError, match="strictly-positive finite prices"):
        cusum_filter(close, threshold=0.01)


def test_non_finite_price_raises_on_neg_inf() -> None:
    close = _flat_close()
    close.iloc[10] = -np.inf
    with pytest.raises(ValueError, match="strictly-positive finite prices"):
        cusum_filter(close, threshold=0.01)


def test_non_monotonic_index_raises() -> None:
    idx = pd.DatetimeIndex(["2025-01-02", "2025-01-01", "2025-01-03"], dtype="datetime64[ns]")
    close = pd.Series([100.0, 100.1, 100.2], index=idx)
    with pytest.raises(ValueError, match="monotonic increasing"):
        cusum_filter(close, threshold=0.01)


# =====================================================================
# Degenerate-but-valid inputs.
# =====================================================================


def test_empty_series_returns_empty_index() -> None:
    """Empty input short-circuits before the finiteness check.
    Returns an empty `DatetimeIndex` with the same dtype as the
    input's index — important for streaming callers that splice
    per-window outputs together."""
    idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
    close = pd.Series([], index=idx, dtype=np.float64)
    events = cusum_filter(close, threshold=0.01)
    assert isinstance(events, pd.DatetimeIndex)
    assert len(events) == 0


def test_single_element_emits_no_event() -> None:
    """One bar yields a single zero-valued log-diff; cannot cross any
    positive threshold from a zero baseline."""
    idx = pd.date_range("2025-01-01", periods=1, freq="D")
    close = pd.Series([100.0], index=idx)
    events = cusum_filter(close, threshold=0.01)
    assert len(events) == 0


# =====================================================================
# Math contract — log-return surprise sum, strict trigger.
# =====================================================================


def test_flat_prices_emit_no_events() -> None:
    """Zero log-returns can never accumulate to a positive threshold."""
    close = _flat_close()
    events = cusum_filter(close, threshold=0.001)
    assert len(events) == 0


def test_ramp_fires_at_predicted_indices() -> None:
    """Linear log-price ramp with per-bar log-return = 0.01.

    At `threshold=0.045` the cumulative log-return crosses on the 5th
    bar after each reset (sum = 0.05 > 0.045). Predicted fire indices:
    [5, 10, 15, 20, 25, 30, 35] (n=40).

    `threshold` is set comfortably between integer-step boundaries
    (0.04 vs 0.05) to immunize the analytic prediction against
    ULP-level roundoff in `log(exp(x))`.
    """
    n = 40
    step = 0.01
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    log_prices = step * np.arange(n, dtype=np.float64)
    close = pd.Series(np.exp(log_prices), index=idx)
    events = cusum_filter(close, threshold=0.045)
    expected = pd.DatetimeIndex([idx[k] for k in (5, 10, 15, 20, 25, 30, 35)])
    pd.testing.assert_index_equal(events, expected)


def test_symmetric_trigger_on_downward_path() -> None:
    """Mirror of the upward ramp: same fire indices on the negative
    accumulator (`s_neg < -threshold`)."""
    n = 40
    step = -0.01
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    log_prices = step * np.arange(n, dtype=np.float64)
    close = pd.Series(np.exp(log_prices), index=idx)
    events = cusum_filter(close, threshold=0.045)
    expected = pd.DatetimeIndex([idx[k] for k in (5, 10, 15, 20, 25, 30, 35)])
    pd.testing.assert_index_equal(events, expected)


def test_strict_trigger_at_exact_threshold_does_not_fire() -> None:
    """`s_pos == threshold` does NOT fire — only `s_pos > threshold`.

    Constructed to be exact under IEEE-754: `close = [1, 2]` gives
    `log(close)[1] - log(close)[0] = ln(2) - 0 = ln(2)` exactly
    (subtraction of zero is exact). Setting `threshold = float(np.log(2.0))`
    means `s_pos` equals `threshold` bit-for-bit, so the strict `>`
    must hold and no event fires.
    """
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    close = pd.Series([1.0, 2.0], index=idx)
    threshold = float(np.log(2.0))
    events = cusum_filter(close, threshold=threshold)
    assert len(events) == 0


def test_just_above_threshold_fires() -> None:
    """Companion to the strict-boundary test: threshold nudged 1 ppt
    below `ln(2)` must fire (`s_pos > threshold` strictly)."""
    idx = pd.date_range("2025-01-01", periods=2, freq="D")
    close = pd.Series([1.0, 2.0], index=idx)
    threshold = float(np.log(2.0)) * (1.0 - 1e-12)
    events = cusum_filter(close, threshold=threshold)
    assert len(events) == 1
    assert events[0] == idx[1]


# =====================================================================
# Scale invariance — discriminator vs. raw-price-difference CUSUM.
# =====================================================================


@pytest.mark.parametrize("scale", [1e-3, 0.5, 2.0, 1e3])
def test_scale_invariance(scale: float) -> None:
    """Multiplying close by any positive constant must not move events.

    `log(c·p_t) − log(c·p_{t−1}) = log(p_t) − log(p_{t−1})`. This is a
    discriminator vs. raw-price-difference CUSUM (`close.diff()`),
    whose threshold would scale linearly with the price level and
    require re-tuning at every level shift. It is NOT a discriminator
    vs. `pct_change` (simple returns are also scale-invariant) — for
    that, see `test_path_additivity_distinguishes_log_from_pct`.
    """
    rng = np.random.default_rng(seed=42)
    idx = pd.date_range("2025-01-01", periods=500, freq="D")
    log_ret = 0.005 * rng.standard_normal(500)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    scaled = close * scale
    events_orig = cusum_filter(close, threshold=0.02)
    events_scaled = cusum_filter(scaled, threshold=0.02)
    assert list(events_orig) == list(events_scaled)


def test_path_additivity_distinguishes_log_from_pct() -> None:
    """Discriminator vs. pre-S31 `pct_change` semantics.

    Path `[100, 50, 75]` at `threshold = 0.6`:
      - Log-return CUSUM: `log(50/100) ≈ -0.693`, which crosses
        `-threshold` on bar 1 → fires at `idx[1]`. After reset,
        `log(75/50) ≈ +0.405` does not cross. Total: 1 event.
      - `pct_change` CUSUM (pre-S31): `pct_change = [NaN, -0.5,
        +0.5]`. `s_neg = -0.5`, which is NOT below `-0.6`. Then
        `s_neg` resets to `0` after the `+0.5` bar. Total: 0 events.

    Pinning a `len == 1` event at `idx[1]` makes any silent revert
    to `pct_change` loud.
    """
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    close = pd.Series([100.0, 50.0, 75.0], index=idx, name="close")
    events = cusum_filter(close, threshold=0.6)
    assert len(events) == 1
    assert events[0] == idx[1]


# =====================================================================
# Composition-spike fixture pin — bounds the migration's blast radius.
# =====================================================================


@pytest.mark.parametrize(
    "seed,expected_n_events",
    [(42, 413), (7, 405), (123, 407), (2026, 420), (4321, 386)],
)
def test_v1_fixture_per_seed_event_count_pin(seed: int, expected_n_events: int) -> None:
    """Per-seed `n_events` pin on the V1 daily DGP at the canonical
    `CUSUM_THRESHOLD = 0.01`. Pinned post-S31 (log-return semantics)
    so future drift in either the filter or the fixture DGP is caught
    even before the looser `test_pin2_event_count_bounded` band in
    `test_tml_composition.py` would notice.
    """
    from tests.fixtures.tml_composition_spike import (
        CUSUM_THRESHOLD,
        FIXTURE_N,
        build_fixture,
    )

    close, _ = build_fixture(seed=seed, n=FIXTURE_N)
    events = cusum_filter(close, threshold=CUSUM_THRESHOLD)
    assert len(events) == expected_n_events, (
        f"seed {seed}: n_events={len(events)} != expected {expected_n_events}"
    )
