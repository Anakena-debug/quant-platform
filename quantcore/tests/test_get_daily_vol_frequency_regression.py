"""P1.4 regression tests for `labels/labelling.py::get_daily_vol` frequency contract.

Pins:

  - `get_daily_vol` raises `ValueError` on sub-daily input (1-min, 30-min, …).
    Pre-P1.4 silently returned bar-horizon vol, under-sizing daily vol by
    ~1/sqrt(bars_per_day) — ~20× on SPY 1-min (audit §F29 repro).
  - Index contract enforced by `_assert_daily_or_lower`: must be
    `DatetimeIndex`, monotonic-increasing, unique, with median spacing ≥ 20h.
  - 20h median-delta threshold tolerates DST transitions, holiday-spanning
    gaps, and timestamp jitter while cleanly rejecting 30-min and below.
  - Daily and weekly inputs pass unchanged (body preserved bitwise).
  - Resample workaround (`close.resample("1B").last().pipe(get_daily_vol)`)
    is documented in the function docstring and verified end-to-end here.

Scope note: F29 is fixed as a contract-enforce (option (e) per user scope
decision 2026-04-21) — not a generalization to handle intraday. Call-site
grep confirmed zero production intraday callers. Example-tree follow-ups
(`examples/legacy/afml_omega_pipeline.py:56` latent, and
`examples/legacy/afml_complete_pipeline.py::get_daily_volatility` potentially
F29-class) are out of P1.4 scope and logged separately.

§0 provenance — pinned 2026-04-21, deterministic re-execution in
`quantcore/.venv` (python 3.11.14, numpy 2.4.4, pandas 3.0.2):

    inspect.getsource(quantcore.labels.labelling.get_daily_vol):
        def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
            ret = close.pct_change()
            return ret.ewm(span=span, adjust=False).std()

    §0 gate regime: pct_change (close.pct_change().ewm(...).std())
    `adjust=False` present in the .ewm(...) call (reference impl matched).

    Probe A  (daily identity, pct_change vs intent-preserving AFML, 250 aligned)
        max|diff| = 0.000e+00   → current impl IS pct_change regime
    Probe A' (verbatim AFML snippet off-by-one, 2520 bdays)
        ratio verbatim/current = 1.2863  (expected ~sqrt(1.8)=1.342; distinct finding,
        NOT F29-confounding — pinned for audit record only)
    Probe B  (1-min, n=3900, tail=500)
        cur_scale_err = 0.0484   (expected ~1/sqrt(390)=0.0506 under pct_change
        regime; observed within 5%)  → F29 confirmed: current impl under-sizes
        σ_daily by sqrt(N_bars/day) on intraday bars.
    Probe C  (30-min, n=780 bdays×13, tail=30)
        cur_scale_err = 0.2855   (expected ~1/sqrt(13)=0.2774; observed within 3%)

Caveat on Probe B/C `intent_scale_err`: closed-form prediction
`sqrt(2)/sqrt(N_b)` was wrong; ewm.std of overlapping moving-sum returns
has a distinct autocorrelation bias (observed ~0.337 @ 1-min and ~0.926
@ 30-min). Not used as a correctness claim; treated as presentation noise
in the probe output. The clean F29 diagnostic is `cur_scale_err` vs
`1/sqrt(N_b)` under the pct_change regime — confirmed.

Discriminator map: of 9 tests, 5 are F29 discriminators that FAIL on
`main@2ad69dc` (no `ValueError` raised on sub-daily / malformed index);
4 are non-regression baselines preserving the daily/weekly/resampled/DST
acceptance path. This mirrors the P1.2 `test_stats_degenerate_input.py`
ratio of discriminators to baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.labels.labelling import get_daily_vol


# =====================================================================
# Fixture builders — deterministic seeds, shared across tests.
# =====================================================================


def _daily_gbm_fixture(
    n_days: int = 252,
    daily_sigma: float = 0.01,
    seed: int = 0,
    tz: str | None = None,
    start: str = "2020-01-02",
) -> pd.Series:
    """Daily-spaced GBM close series. Optional tz makes it DST-sensitive."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, tz=tz)
    returns = rng.normal(0, daily_sigma, len(idx))
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)


def _weekly_fixture(n_weeks: int = 104, weekly_sigma: float = 0.02, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start="2020-01-03", periods=n_weeks, freq="W-FRI")
    returns = rng.normal(0, weekly_sigma, len(idx))
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)


def _intraday_fixture(
    n_days: int,
    bars_per_day: int,
    step_minutes: int,
    daily_sigma: float = 0.01,
    seed: int = 0,
) -> pd.Series:
    """bdate_range × intraday offsets Cartesian. Unique, monotonic, bdays only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start="2020-01-02", periods=n_days)
    session = pd.Timedelta(hours=9, minutes=30)
    offsets = pd.to_timedelta(np.arange(bars_per_day) * step_minutes, unit="m")
    idx_vals = (days.values[:, None] + session.to_numpy() + offsets.values[None, :]).ravel()
    idx = pd.DatetimeIndex(idx_vals)
    assert idx.is_unique and idx.is_monotonic_increasing, "fixture contract"
    bar_sigma = daily_sigma / np.sqrt(bars_per_day)
    returns = rng.normal(0, bar_sigma, len(idx))
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=idx)


# =====================================================================
# Non-regression baselines — pass on both main@2ad69dc and post-fix.
# =====================================================================


def test_daily_identity_preserved() -> None:
    """252-bday GBM with σ_daily=0.01, span=100. Tail mean within 5% of σ_daily.

    Calibrated on seed=0: pct_err=0.01% (tail_n=100). Pins that the gate
    insertion does NOT perturb the pct_change().ewm(...).std() body — a
    behaviour-preservation check on the accepted-frequency path.
    """
    close = _daily_gbm_fixture(n_days=252, daily_sigma=0.01, seed=0)
    out = get_daily_vol(close, span=100)
    tail_mean = float(out.dropna().iloc[-100:].mean())
    assert abs(tail_mean - 0.01) / 0.01 < 0.05, (
        f"tail_mean={tail_mean:.6e} diverges >5% from σ_daily=0.01"
    )


def test_weekly_accepted() -> None:
    """W-FRI frequency (~7d spacing) clears the 20h threshold. Output finite.

    Calibrated on seed=0, n_weeks=104, weekly_sigma=0.02, span=20: pct_err=4.11%.
    Use 10% tolerance to absorb small-sample EWMA noise.
    """
    close = _weekly_fixture(n_weeks=104, weekly_sigma=0.02, seed=0)
    out = get_daily_vol(close, span=20)
    finite = out.dropna()
    assert len(finite) > 0, "no finite values"
    assert np.isfinite(finite).all(), "non-finite values in output"
    tail_mean = float(finite.iloc[-50:].mean())
    assert abs(tail_mean - 0.02) / 0.02 < 0.10, (
        f"weekly tail_mean={tail_mean:.6e} diverges >10% from σ_weekly=0.02"
    )


def test_resample_workaround() -> None:
    """Docstring-promised escape hatch: intraday → .resample('1B').last() → get_daily_vol.

    30 bdays × 390 1-min bars → 30 daily bars via resample. σ_daily=0.01 by
    construction. Calibrated: pct_err=3.08%. Use 10% tolerance.
    """
    intraday = _intraday_fixture(n_days=30, bars_per_day=390, step_minutes=1, seed=0)
    daily = intraday.resample("1B").last().dropna()
    out = get_daily_vol(daily, span=10)
    tail_mean = float(out.dropna().iloc[-10:].mean())
    assert abs(tail_mean - 0.01) / 0.01 < 0.10, (
        f"resampled tail_mean={tail_mean:.6e} diverges >10% from σ_daily=0.01"
    )


def test_dst_transition_tolerated() -> None:
    """tz-aware daily bdate_range spanning 2020-03-08 US DST start.

    Three of the 20 business days straddle the DST boundary; at wall-clock
    midnight the diff is still structurally one business day. Median spacing
    remains ≥ 20h — the 20h threshold is tuned precisely to tolerate this.
    """
    close = _daily_gbm_fixture(
        n_days=20,
        daily_sigma=0.01,
        seed=0,
        tz="US/Eastern",
        start="2020-03-02",
    )
    # Verify fixture actually spans DST boundary (2020-03-08)
    assert close.index.min() < pd.Timestamp("2020-03-08", tz="US/Eastern")
    assert close.index.max() > pd.Timestamp("2020-03-08", tz="US/Eastern")
    # Should not raise:
    out = get_daily_vol(close, span=5)
    assert np.isfinite(out.dropna()).all()


# =====================================================================
# F29 discriminators — FAIL on main@2ad69dc (no ValueError raised),
# PASS post-fix.
# =====================================================================


def test_1min_rejected() -> None:
    """30 bdays × 390 min bars. Median spacing = 1 min ≪ 20h threshold.

    On main@2ad69dc, this returns silently with minute-horizon vol
    (scale_error ≈ 0.048, under-sizing daily σ by ~sqrt(390) per Probe B).
    Post-fix: raises ValueError naming the frequency contract.
    """
    close = _intraday_fixture(n_days=30, bars_per_day=390, step_minutes=1, seed=0)
    with pytest.raises(ValueError, match="daily-or-lower frequency"):
        get_daily_vol(close)


def test_30min_rejected() -> None:
    """60 bdays × 13 bars/day (30-min spacing). Median = 30 min < 20h.

    Calibrated Probe C scale_error=0.2855 (vs 0.2774 expected under
    pct_change regime) — under-sizing by sqrt(13).
    """
    close = _intraday_fixture(n_days=60, bars_per_day=13, step_minutes=30, seed=0)
    with pytest.raises(ValueError, match="daily-or-lower frequency"):
        get_daily_vol(close)


def test_non_monotonic_rejected() -> None:
    """Shuffled daily index. pct_change on shuffled data gives positional
    diffs that are semantically meaningless — pre-fix returned garbage silently.
    """
    close = _daily_gbm_fixture(n_days=252, seed=0)
    shuffled = close.sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match="monotonic"):
        get_daily_vol(shuffled)


def test_duplicate_rejected() -> None:
    """Daily index + appended duplicate of last timestamp.

    Precedence check: monotonic passes (is_monotonic_increasing allows
    equality), unique fails. Must raise on unique, not monotonic.
    """
    close = _daily_gbm_fixture(n_days=252, seed=0)
    dup_close = pd.concat([close, close.iloc[[-1]]])
    assert dup_close.index.is_monotonic_increasing, "fixture precedence precondition"
    assert not dup_close.index.is_unique, "fixture precedence precondition"
    with pytest.raises(ValueError, match="unique"):
        get_daily_vol(dup_close)


def test_non_datetime_rejected() -> None:
    """RangeIndex. Pre-fix silently computed positional pct_change (meaningless)."""
    close = pd.Series(np.linspace(100.0, 110.0, 252), index=pd.RangeIndex(252))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        get_daily_vol(close)
