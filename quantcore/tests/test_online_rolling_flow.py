"""S42 — OnlineRollingFlow byte-parity with build_flow's pandas .rolling.

OnlineRollingFlow (live, incremental) must reproduce, value-for-value, what
``alpha_research.features.build_flow`` computes with pandas ``.rolling`` over the
full bar history. The oracle below replicates build_flow's EXACT pandas ops
(``rolling(w).sum()`` + ``_safe_z``) inline — quantcore must not import alpha_R
(dependency inversion), and pinning the literal pandas expressions IS the
contract. Parity is asserted at atol=1e-12 including the warmup-NaN / warmup-0.0
asymmetry that a naive online impl would get wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.features.online_rolling import (
    SUM_WINDOWS,
    Z_WINDOW,
    OnlineRollingFlow,
)

ATOL = 1e-12


# --- oracle: EXACT copy of build_flow / _safe_z pandas math (the spec) -------


def _safe_z_oracle(series: pd.Series, window: int = 20) -> pd.Series:
    mu = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()  # ddof=1
    z = (series - mu) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_flow_oracle(values: list[float]) -> pd.DataFrame:
    s = pd.Series(values, dtype="float64")
    out = pd.DataFrame(index=s.index)
    for w in SUM_WINDOWS:
        out[f"sum{w}"] = s.rolling(w).sum()
    out["z20"] = _safe_z_oracle(s, Z_WINDOW)
    return out


def _run_online(values: list[float]) -> pd.DataFrame:
    r = OnlineRollingFlow()
    rows = [r.update(v) for v in values]
    return pd.DataFrame(rows)


def _assert_parity(values: list[float]) -> None:
    oracle = _build_flow_oracle(values)
    online = _run_online(values)
    assert list(online.columns) == [f"sum{w}" for w in SUM_WINDOWS] + ["z20"]
    for col in oracle.columns:
        o = oracle[col].to_numpy(dtype=float)
        n = online[col].to_numpy(dtype=float)
        # equal_nan so warmup NaNs (sum cols) must align position-for-position.
        np.testing.assert_allclose(n, o, rtol=0.0, atol=ATOL, equal_nan=True, err_msg=col)


# --- random parity -----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_parity_random_series(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # flow ratios live in [-1, 1]; sample that range.
    values = rng.uniform(-1.0, 1.0, size=100).tolist()
    _assert_parity(values)


# --- edge cases the warmup/cascade depends on --------------------------------


def test_warmup_sum_is_nan_z_is_zero() -> None:
    """sum{w} is NaN for the first (w-1) rows; z20 is 0.0 (not NaN) for the
    first 19 rows — the build_flow / _safe_z asymmetry."""
    values = [0.1 * (i % 7 - 3) for i in range(30)]
    online = _run_online(values)
    # sum5 NaN for rows 0..3, finite from row 4
    assert online["sum5"].iloc[:4].isna().all()
    assert online["sum5"].iloc[4:].notna().all()
    # z20 is 0.0 (NOT NaN) throughout warmup
    assert (online["z20"].iloc[: Z_WINDOW - 1] == 0.0).all()
    assert not online["z20"].isna().any()
    _assert_parity(values)


def test_constant_run_zero_std_z_is_zero() -> None:
    """A full window of identical values -> rolling std 0 -> z20 0.0 (cascade)."""
    values = [3.0] * 30
    online = _run_online(values)
    assert (online["z20"] == 0.0).all()
    _assert_parity(values)


def test_series_shorter_than_windows() -> None:
    """Fewer than 5 values: every sum NaN, z20 all 0.0."""
    values = [0.5, -0.5, 0.25]
    online = _run_online(values)
    for w in SUM_WINDOWS:
        assert online[f"sum{w}"].isna().all()
    assert (online["z20"] == 0.0).all()
    _assert_parity(values)


def test_nan_input_matches_pandas() -> None:
    """A NaN value propagates through pandas .rolling the same way online does."""
    values = [0.2, -0.3, float("nan"), 0.4, 0.1] * 6
    _assert_parity(values)


def test_zero_then_variance_transition() -> None:
    """Constant warmup (z20 0.0) then a jump introduces variance -> nonzero z."""
    values = [1.0] * 20 + [5.0] + [1.0] * 9
    online = _run_online(values)
    # the jump row has a full window with variance -> z20 != 0
    assert online["z20"].iloc[20] != 0.0
    _assert_parity(values)


# --- reset semantics ---------------------------------------------------------


def test_reset_clears_state() -> None:
    r = OnlineRollingFlow()
    for v in range(25):
        r.update(float(v))
    r.reset()
    out = r.update(1.0)
    # after reset, first update is warmup again
    assert np.isnan(out["sum5"])
    assert out["z20"] == 0.0


# --- real-data tie-in: closes the loop to the s41 byte-parity validation -----


def test_real_data_tie_in_via_bar_flow_ratios() -> None:
    """Build a synthetic-but-realistic flow-ratio series (as bar_flow_ratios
    would emit per bar), run it through OnlineRollingFlow, and confirm parity
    with the build_flow oracle. (Uses the s41 adapter to source the values, so
    the online rolling sits directly atop the verified base-ratio path.)"""
    from quantcore.features.top_of_book import bar_flow_ratios

    rng = np.random.default_rng(11)

    class _FakeBar:
        __slots__ = (
            "volume",
            "dollar_volume",
            "signed_volume_sum",
            "signed_dollar_sum",
            "signed_tick_imbalance",
        )

    ratios: list[float] = []
    for _ in range(60):
        b = _FakeBar()
        b.volume = float(rng.integers(100, 1000))
        b.dollar_volume = b.volume * float(rng.uniform(50, 150))
        # signed sums bounded by |sum| <= volume / dollar_volume (|dir|=1)
        b.signed_volume_sum = b.volume * float(rng.uniform(-1, 1))
        b.signed_dollar_sum = b.dollar_volume * float(rng.uniform(-1, 1))
        b.signed_tick_imbalance = float(rng.uniform(-1, 1))
        ratios.append(bar_flow_ratios(b)["signed_vol_imb"])

    _assert_parity(ratios)
