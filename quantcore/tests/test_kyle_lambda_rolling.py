"""Parity pin for the vectorized kyle_lambda_rolling vs the original per-window linregress.

The vectorization (rolling cov/var) must reproduce the loop-of-``scipy.stats.linregress``
slope to fp tolerance AND gate windows with <10 valid pairs to NaN identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantcore.features.microstructure import kyle_lambda_rolling, tick_rule


def _reference(prices: pd.Series, volumes: pd.Series, window: int) -> pd.Series:
    """The original implementation: per-window linregress slope, <10 valid -> NaN."""
    signs = tick_rule(prices)
    signed_vol = signs * volumes
    dp = prices.diff()

    def compute(idx: range) -> float:
        sv = signed_vol.iloc[idx]
        pc = dp.iloc[idx]
        valid = ~(sv.isna() | pc.isna())
        if valid.sum() < 10:
            return np.nan
        slope, _, _, _, _ = stats.linregress(sv[valid], pc[valid])
        return float(slope)

    res = [compute(range(i - window, i)) for i in range(window, len(prices) + 1)]
    return pd.Series(res, index=prices.index[window - 1 :])


def _synth(n: int = 300, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    vol = pd.Series(rng.integers(1, 1000, n).astype(float), index=idx)
    return close, vol


def test_matches_reference_clean_data():
    close, vol = _synth()
    for window in (20, 50, 100):
        new = kyle_lambda_rolling(close, vol, window)
        ref = _reference(close, vol, window)
        assert list(new.index) == list(ref.index)
        np.testing.assert_allclose(
            new.to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
        )


def test_under_10_valid_pairs_gates_to_nan_like_reference():
    # Inject NaNs into volume so some windows fall below the 10-valid-pair floor.
    close, vol = _synth(n=120, seed=3)
    vol.iloc[::2] = np.nan  # every other obs invalid -> small windows drop below 10
    window = 16
    new = kyle_lambda_rolling(close, vol, window)
    ref = _reference(close, vol, window)
    assert list(new.index) == list(ref.index)
    # NaN gating must agree position-for-position, and values match where both are finite.
    assert np.array_equal(new.isna().to_numpy(), ref.isna().to_numpy())
    np.testing.assert_allclose(
        new.to_numpy(), ref.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_output_alignment():
    close, vol = _synth(n=60)
    window = 25
    out = kyle_lambda_rolling(close, vol, window)
    assert len(out) == len(close) - window + 1
    assert out.index[0] == close.index[window - 1]
    assert out.index[-1] == close.index[-1]
