"""quantcore.factors — shape, orientation, and (the load-bearing property) point-in-time safety.

The PIT-cleanliness of each feature constructor is checked with the real leakage guard
(``quantcore.leakage.assert_no_lookahead``), not eyeballed — and ``forward_returns`` is asserted
to be *flagged* as lookahead, pinning the feature/label boundary. A small end-to-end run proves
the constructors feed straight into ``quantcore.factory.run_factory``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore import catalog
from quantcore.factors import (
    cross_sectional_illiquidity,
    cross_sectional_momentum,
    cross_sectional_reversal,
    cross_sectional_volatility,
    forward_returns,
)
from quantcore.factory import run_factory
from quantcore.leakage import LeakageError, assert_no_lookahead


def _panel(n_dates: int = 320, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic [dates x assets] close + volume panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    assets = ["A", "B", "C", "D", "E"]
    rets = rng.normal(0.0003, 0.02, size=(n_dates, len(assets)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=assets)
    volume = pd.DataFrame(
        rng.lognormal(12.0, 0.5, size=(n_dates, len(assets))), index=dates, columns=assets
    )
    return close, volume


def test_shapes_and_alignment_preserved():
    close, volume = _panel()
    for out in (
        cross_sectional_momentum(close),
        cross_sectional_reversal(close),
        cross_sectional_volatility(close),
        cross_sectional_illiquidity(close, volume),
        forward_returns(close),
    ):
        assert out.shape == close.shape
        assert out.index.equals(close.index)
        assert out.columns.equals(close.columns)


def test_momentum_is_point_in_time():
    close, _ = _panel()
    reports = assert_no_lookahead(lambda c: cross_sectional_momentum(c, lookback=60, skip=5), close)
    assert all(r.is_causal for r in reports)


def test_reversal_is_point_in_time():
    close, _ = _panel()
    reports = assert_no_lookahead(lambda c: cross_sectional_reversal(c, lookback=20), close)
    assert all(r.is_causal for r in reports)


def test_volatility_is_point_in_time():
    close, _ = _panel()
    reports = assert_no_lookahead(lambda c: cross_sectional_volatility(c, lookback=30), close)
    assert all(r.is_causal for r in reports)


def test_illiquidity_is_point_in_time():
    close, volume = _panel()
    # reindex volume onto the (possibly truncated) close index so the guard truncates both panels.
    reports = assert_no_lookahead(
        lambda c: cross_sectional_illiquidity(c, volume.reindex(c.index), lookback=20), close
    )
    assert all(r.is_causal for r in reports)


def test_momentum_and_reversal_have_opposite_orientation():
    # A rising name, a falling name, a flat name -> momentum ranks up>flat>down; reversal flips it.
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    close = pd.DataFrame(
        {
            "up": np.linspace(100.0, 200.0, 120),
            "flat": np.full(120, 150.0),
            "down": np.linspace(200.0, 100.0, 120),
        },
        index=dates,
    )
    mom = cross_sectional_momentum(close, lookback=60, skip=5).iloc[-1]
    rev = cross_sectional_reversal(close, lookback=60).iloc[-1]
    assert mom["up"] > mom["flat"] > mom["down"]
    assert rev["down"] > rev["flat"] > rev["up"]


def test_illiquidity_is_higher_for_thinner_volume():
    # Two names with identical price paths but 10x different volume: lower volume -> higher Amihud.
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    rng = np.random.default_rng(1)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    close = pd.DataFrame({"thin": px, "thick": px}, index=dates)
    volume = pd.DataFrame({"thin": np.full(80, 1e4), "thick": np.full(80, 1e5)}, index=dates)
    illiq = cross_sectional_illiquidity(close, volume, lookback=20).iloc[-1]
    assert illiq["thin"] > illiq["thick"]


def test_forward_returns_is_the_forward_aligned_label():
    close, _ = _panel(n_dates=50)
    fwd = forward_returns(close, horizon=1)
    t = 10
    assert fwd.iloc[t]["A"] == pytest.approx(close.iloc[t + 1]["A"] / close.iloc[t]["A"] - 1.0)
    assert fwd.iloc[-1].isna().all()  # last row has no future
    # The label is intentionally forward-looking: the leakage guard must flag it.
    with pytest.raises(LeakageError):
        assert_no_lookahead(lambda c: forward_returns(c, horizon=1), close)


def test_end_to_end_through_factory():
    close, _ = _panel()
    candidates = {
        "momentum": cross_sectional_momentum(close, lookback=60, skip=5),
        "reversal": cross_sectional_reversal(close, lookback=20),
    }
    verdicts = run_factory(candidates, forward_returns(close), min_days=20)
    assert {v.name for v in verdicts} == {"momentum", "reversal"}
    for v in verdicts:
        assert v.n_days > 0
        assert np.isfinite(v.mean_ic)


def test_constructors_are_registered_in_the_catalog():
    specs = {s.name: s for s in catalog.list_factors(category="cross_sectional")}
    assert set(specs) == {
        "cross_sectional_momentum",
        "cross_sectional_reversal",
        "cross_sectional_volatility",
        "cross_sectional_illiquidity",
    }
    for spec in specs.values():
        assert spec.module == "quantcore.factors"
        assert callable(spec.resolve())


def test_input_validation():
    with pytest.raises(TypeError):
        cross_sectional_momentum(pd.Series([1.0, 2.0, 3.0]))  # pyright: ignore[reportArgumentType]
    unsorted = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=[3, 2, 1])
    with pytest.raises(ValueError):
        cross_sectional_momentum(unsorted)
    with pytest.raises(ValueError):
        cross_sectional_momentum(_panel()[0], skip=300, lookback=60)
