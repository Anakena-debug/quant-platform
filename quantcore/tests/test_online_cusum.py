"""Unit + parity tests for OnlineCUSUMFilter (S34 §3.AC5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.bars import OnlineCUSUMFilter
from quantcore.data import Bar, BarKind, Side, TradeEvent
from quantcore.labels.labelling import cusum_filter


def _bar(i: int, close: float) -> Bar:
    ts = pd.Timestamp("2026-01-02 09:30:00") + pd.Timedelta(seconds=i)
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


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threshold must be > 0"):
        OnlineCUSUMFilter(threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be > 0"):
        OnlineCUSUMFilter(threshold=-0.01)


def test_non_bar_events_are_no_op() -> None:
    flt = OnlineCUSUMFilter(threshold=0.01)
    t = TradeEvent(
        ts_event=0,
        instrument_id=1,
        sequence=0,
        price=100.0,
        size=1.0,
        aggressor_side=Side.BID,
    )
    assert flt.on_event(t) is None


def test_first_bar_returns_none() -> None:
    flt = OnlineCUSUMFilter(threshold=0.01)
    assert flt.on_event(_bar(0, 100.0)) is None


def test_reset_clears_state() -> None:
    flt = OnlineCUSUMFilter(threshold=0.01)
    flt.on_event(_bar(0, 100.0))
    flt.on_event(_bar(1, 200.0))  # huge jump triggers
    flt.reset()
    # After reset: another huge jump must still trigger as if from scratch
    assert flt.on_event(_bar(2, 200.0)) is None  # seeds prev_close
    out = flt.on_event(_bar(3, 400.0))
    assert out is not None


def test_invalid_close_raises() -> None:
    flt = OnlineCUSUMFilter(threshold=0.01)
    flt.on_event(_bar(0, 100.0))
    with pytest.raises(ValueError, match="strictly positive"):
        flt.on_event(_bar(1, 0.0))


def test_online_cusum_parity_with_legacy() -> None:
    """AC5 parity: streaming event timestamps match cusum_filter exactly."""
    rng = np.random.default_rng(7)
    n = 500
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n)))
    idx = pd.date_range("2026-01-02", periods=n, freq="min")
    close = pd.Series(prices, index=idx)
    threshold = 0.01

    legacy_events = cusum_filter(close, threshold=threshold)
    legacy_ts = np.asarray([t.value for t in legacy_events], dtype=np.int64)

    flt = OnlineCUSUMFilter(threshold=threshold)
    streaming_ts: list[int] = []
    for i, (ts, p) in enumerate(zip(idx, prices)):
        out = flt.on_event(
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
            streaming_ts.append(out)

    streaming_arr = np.asarray(streaming_ts, dtype=np.int64)
    assert np.array_equal(legacy_ts, streaming_arr), (
        f"event indices diverged: legacy={len(legacy_ts)} streaming={len(streaming_arr)}"
    )
