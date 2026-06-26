"""s83 F2a/F2b/F5/F16/F3 regressions — partial-bar contract + volatility fail-fast.

F2a: ``imbalance_bars(include_partial_last_bar=True)`` crashed with a
pandas k-vs-k+1 length mismatch whenever a partial tail existed (executed
s83 repro: "Length of values (34) does not match length of index (35)").
F2b: ``aggregate_to_ohlcv`` with zero completed bars silently dropped a
requested partial row for ALL bar families.
F5:  ``imbalance_bars`` accepted ``ewma_span_* == 0`` (α=2, unstable EWMA)
that ``runs_bars`` and the streaming engines reject.
F16: ``dollar_bars_with_microstructure`` inherited both F2 failure modes
through ``micro_agg.index = ohlcv.index``.
F3:  ``OnlineEWMAVolatility`` mutated ``_prev_close`` BEFORE validating,
so one NaN close destroyed two returns while the docstring claimed
"state unchanged"; now fails fast like ``OnlineCUSUMFilter``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.bars.bars import (
    ImbalanceConfig,
    aggregate_to_ohlcv,
    dollar_bars,
    imbalance_bars,
)
from quantcore.bars.volatility import OnlineEWMAVolatility
from quantcore.data import Bar, BarKind
from quantcore.features.top_of_book import dollar_bars_with_microstructure


def _tick_df(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "price": 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))),
            "volume": rng.integers(1, 20, n).astype(float),
        },
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="s"),
    )


class TestImbalancePartialGuard:
    def test_partial_flag_raises_not_implemented(self) -> None:
        cfg = ImbalanceConfig(exp_num_ticks_init=50, exp_imbalance_init=0.05)
        with pytest.raises(NotImplementedError, match="imbalance_bars"):
            imbalance_bars(_tick_df(), "tick", cfg, include_partial_last_bar=True)

    def test_span_zero_raises(self) -> None:
        cfg = ImbalanceConfig(exp_num_ticks_init=50, exp_imbalance_init=0.05, ewma_span_ticks=0)
        with pytest.raises(ValueError, match="ewma_span"):
            imbalance_bars(_tick_df(), "tick", cfg)


class TestAggregatePartialAtZeroBars:
    def test_partial_row_emitted_when_no_bar_completes(self) -> None:
        df = _tick_df(n=5)
        out = aggregate_to_ohlcv(df, np.array([], dtype=np.int64), include_partial_last_bar=True)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["open"] == pytest.approx(df["price"].iloc[0])
        assert row["close"] == pytest.approx(df["price"].iloc[-1])
        assert row["volume"] == pytest.approx(df["volume"].sum())
        assert row["tick_count"] == 5

    def test_no_partial_requested_stays_empty(self) -> None:
        out = aggregate_to_ohlcv(
            _tick_df(n=5), np.array([], dtype=np.int64), include_partial_last_bar=False
        )
        assert len(out) == 0

    def test_empty_df_with_partial_stays_empty(self) -> None:
        df = _tick_df(n=5).iloc[:0]
        out = aggregate_to_ohlcv(df, np.array([], dtype=np.int64), include_partial_last_bar=True)
        assert len(out) == 0

    def test_threshold_family_partial_tail_still_works(self) -> None:
        """The threshold wrappers legitimately support the flag (no
        diagnostic columns); the F2b fix must not disturb them."""
        df = _tick_df()
        full = dollar_bars(df, threshold=50_000.0, include_partial_last_bar=True)
        completed = dollar_bars(df, threshold=50_000.0, include_partial_last_bar=False)
        assert len(full) in (len(completed), len(completed) + 1)
        if len(full) == len(completed) + 1:
            assert full["close"].iloc[-1] == pytest.approx(df["price"].iloc[-1])


class TestMicrostructurePartialGuard:
    def test_partial_flag_raises_not_implemented(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        df = pd.DataFrame(
            {
                "price": prices,
                "size": rng.integers(1, 10, n).astype(float),
                "side": rng.choice(["A", "B"], size=n),
                "bid_px_00": prices - 0.05,
                "ask_px_00": prices + 0.05,
                "bid_sz_00": np.full(n, 100.0),
                "ask_sz_00": np.full(n, 100.0),
            },
            index=pd.date_range("2026-01-02 09:30:00", periods=n, freq="ms"),
        )
        with pytest.raises(NotImplementedError, match="dollar_bars_with_microstructure"):
            dollar_bars_with_microstructure(df, threshold=500.0, include_partial_last_bar=True)


def _bar(i: int, close: float) -> Bar:
    return Bar(
        ts_event=i,
        instrument_id=1,
        sequence=i,
        ts_open=i,
        kind=BarKind.TICK,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        vwap=close,
        tick_count=1,
        dollar_volume=close,
        spread_mean=0.0,
        spread_last=0.0,
        spread_std=0.0,
        imbalance_mean=0.0,
        imbalance_last=0.0,
        imbalance_std=0.0,
        microprice_dev_mean=0.0,
        microprice_dev_last=0.0,
        microprice_dev_std=0.0,
        signed_volume_sum=0.0,
        signed_dollar_sum=0.0,
        signed_tick_imbalance=0.0,
    )


class TestOnlineVolatilityFailFast:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_bad_close_raises(self, bad: float) -> None:
        v = OnlineEWMAVolatility(span=10)
        v.on_event(_bar(0, 100.0))
        with pytest.raises(ValueError, match="finite and > 0"):
            v.on_event(_bar(1, bad))

    def test_state_not_mutated_by_rejected_close(self) -> None:
        """Pre-s83 a NaN close was 'skipped' AFTER seeding _prev_close, so the
        NEXT return was destroyed too (executed repro: closes
        [100, 101, nan, 102, 103] yielded no sigma until the 5th bar). Post-fix
        the rejected event leaves the chain intact: the very next valid close
        produces the return vs the last GOOD close."""
        v = OnlineEWMAVolatility(span=10)
        assert v.on_event(_bar(0, 100.0)) is None  # seed
        with pytest.raises(ValueError):
            v.on_event(_bar(1, float("nan")))
        assert v.on_event(_bar(2, 101.0)) is None  # nobs=1 (vs 100, not NaN)
        out = v.on_event(_bar(3, 102.0))  # nobs=2 → first sigma
        assert out is not None and np.isfinite(out)
