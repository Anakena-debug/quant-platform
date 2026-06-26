"""Unit + parity tests for OnlineEWMAVolatility (S34 §3.AC6)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantcore.bars import OnlineEWMAVolatility
from quantcore.data import Bar, BarKind, Side, TradeEvent
from quantcore.labels.labelling import get_daily_vol

ATOL = 1e-10


def _bar(i: int, close: float) -> Bar:
    ts = pd.Timestamp("2026-01-02") + i * pd.Timedelta(days=1)
    return Bar(
        ts_event=int(ts.value),
        instrument_id=1,
        sequence=i,
        ts_open=int(ts.value),
        kind=BarKind.TICK,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
        vwap=close,
        tick_count=1,
        dollar_volume=0.0,
    )


def test_span_must_be_positive() -> None:
    with pytest.raises(ValueError, match="span"):
        OnlineEWMAVolatility(span=0)


def test_non_bar_events_are_no_op() -> None:
    vol = OnlineEWMAVolatility(span=10)
    t = TradeEvent(
        ts_event=0,
        instrument_id=1,
        sequence=0,
        price=100.0,
        size=1.0,
        aggressor_side=Side.BID,
    )
    assert vol.on_event(t) is None


def test_first_two_bars_return_none() -> None:
    """First bar: no return yet. Second bar: nobs=1, variance undefined."""
    vol = OnlineEWMAVolatility(span=5)
    assert vol.on_event(_bar(0, 100.0)) is None
    assert vol.on_event(_bar(1, 101.0)) is None
    out = vol.on_event(_bar(2, 100.5))
    assert out is not None and math.isfinite(out)


def test_reset_clears_state() -> None:
    vol = OnlineEWMAVolatility(span=5)
    for i, p in enumerate([100.0, 101.0, 102.0, 100.5, 101.5]):
        vol.on_event(_bar(i, p))
    assert vol.sigma is not None
    vol.reset()
    assert vol.sigma is None
    assert vol.on_event(_bar(10, 100.0)) is None


def test_online_volatility_parity_with_legacy() -> None:
    """AC6 parity: streaming sigma matches get_daily_vol at atol=1e-10."""
    span = 20
    n = 500
    rng = np.random.default_rng(13)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))
    idx = pd.date_range("2026-01-02", periods=n, freq="B")
    close = pd.Series(prices, index=idx)

    legacy = get_daily_vol(close, span=span).to_numpy()

    vol = OnlineEWMAVolatility(span=span)
    streaming = np.full(n, np.nan, dtype=np.float64)
    for i, (ts, p) in enumerate(zip(idx, prices)):
        out = vol.on_event(
            Bar(
                ts_event=int(ts.value),
                instrument_id=1,
                sequence=i,
                ts_open=int(ts.value),
                kind=BarKind.TICK,
                open=float(p),
                high=float(p),
                low=float(p),
                close=float(p),
                volume=0.0,
                vwap=float(p),
                tick_count=1,
                dollar_volume=0.0,
            )
        )
        if out is not None:
            streaming[i] = out

    # NaN masks must align (pandas emits NaN at i=0 and i=1)
    nan_legacy = np.isnan(legacy)
    nan_streaming = np.isnan(streaming)
    assert np.array_equal(nan_legacy, nan_streaming), "NaN masks diverged"

    # Finite values match at atol=1e-10
    finite = ~nan_legacy
    np.testing.assert_allclose(streaming[finite], legacy[finite], rtol=0.0, atol=ATOL)
