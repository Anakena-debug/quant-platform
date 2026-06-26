"""Tests for top_of_book batch-path microstructure features.

Covers:
- signed_flow_features: exchange side priority, tick-rule fallback
- dollar_bars_with_microstructure: side_col wiring
- top_of_book_features: crossed-book NaN guard
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantcore.features.top_of_book import (
    dollar_bars_with_microstructure,
    signed_flow_features,
    top_of_book_features,
)


# ===================================================================
# signed_flow_features — side_col priority
# ===================================================================


def _tick_df(prices: list[float], sizes: list[float], sides: list[str]) -> pd.DataFrame:
    n = len(prices)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="ms")
    return pd.DataFrame(
        {"price": prices, "size": sizes, "side": sides},
        index=ts,
    )


class TestSignedFlowSideCol:
    def test_exchange_side_used_when_present(self) -> None:
        """'A' = ask-side (sell) aggressor → -1 per the DBN spec (s83 F11)."""
        df = _tick_df(
            prices=[100.0, 101.0, 102.0],
            sizes=[1.0, 1.0, 1.0],
            sides=["A", "A", "A"],
        )
        flow = signed_flow_features(df, side_col="side")
        np.testing.assert_array_equal(flow["tick_dir"].values, [-1.0, -1.0, -1.0])

    def test_tick_rule_would_disagree(self) -> None:
        """Prices increase → tick rule gives +1; exchange side 'A' (sell) → -1."""
        df = _tick_df(
            prices=[100.0, 101.0, 102.0],
            sizes=[1.0, 1.0, 1.0],
            sides=["A", "A", "A"],
        )
        flow_tick = signed_flow_features(df, side_col=None)
        flow_exch = signed_flow_features(df, side_col="side")
        assert flow_tick["tick_dir"].iloc[1] == 1.0
        assert flow_exch["tick_dir"].iloc[1] == -1.0

    def test_unknown_side_falls_back_to_tick_rule(self) -> None:
        df = _tick_df(
            prices=[100.0, 101.0, 100.5],
            sizes=[1.0, 1.0, 1.0],
            sides=["N", "N", "N"],
        )
        flow = signed_flow_features(df, side_col="side")
        flow_tick = signed_flow_features(df, side_col=None)
        np.testing.assert_array_equal(
            flow["tick_dir"].values,
            flow_tick["tick_dir"].values,
        )

    def test_mixed_known_and_unknown(self) -> None:
        df = _tick_df(
            prices=[100.0, 99.0, 101.0],
            sizes=[1.0, 2.0, 3.0],
            sides=["A", "N", "B"],
        )
        flow = signed_flow_features(df, side_col="side")
        assert flow["tick_dir"].iloc[0] == -1.0  # A (sell aggressor) → -1
        assert flow["tick_dir"].iloc[1] == -1.0  # N, price down → tick rule -1
        assert flow["tick_dir"].iloc[2] == 1.0  # B (buy aggressor) → +1

    def test_signed_volume_uses_direction(self) -> None:
        df = _tick_df(
            prices=[100.0, 99.0],
            sizes=[5.0, 10.0],
            sides=["B", "A"],
        )
        flow = signed_flow_features(df, side_col="side")
        assert flow["signed_volume"].iloc[0] == pytest.approx(5.0)  # B → +1
        assert flow["signed_volume"].iloc[1] == pytest.approx(-10.0)  # A → -1

    def test_none_side_col_is_pure_tick_rule(self) -> None:
        df = _tick_df([100.0, 101.0], [1.0, 1.0], ["B", "B"])
        flow = signed_flow_features(df, side_col=None)
        assert flow["tick_dir"].iloc[1] == 1.0  # price up → tick rule +1

    def test_missing_column_falls_back(self) -> None:
        df = _tick_df([100.0, 101.0], [1.0, 1.0], ["A", "A"])
        df = df.drop(columns=["side"])
        flow = signed_flow_features(df, side_col="side")
        assert flow["tick_dir"].iloc[1] == 1.0  # column absent → tick rule


# ===================================================================
# top_of_book_features — crossed-book guard
# ===================================================================


def _bbo_df(
    bids: list[float],
    asks: list[float],
    bid_szs: list[float],
    ask_szs: list[float],
) -> pd.DataFrame:
    n = len(bids)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="ms")
    return pd.DataFrame(
        {
            "bid_px_00": bids,
            "ask_px_00": asks,
            "bid_sz_00": bid_szs,
            "ask_sz_00": ask_szs,
        },
        index=ts,
    )


class TestCrossedBookGuard:
    def test_normal_book_produces_finite_spread(self) -> None:
        df = _bbo_df([99.0], [100.0], [10.0], [10.0])
        feat = top_of_book_features(df)
        assert feat["spread"].iloc[0] == pytest.approx(1.0)

    def test_crossed_book_produces_nan(self) -> None:
        df = _bbo_df([100.0], [99.0], [10.0], [10.0])
        feat = top_of_book_features(df)
        assert math.isnan(feat["spread"].iloc[0])
        assert math.isnan(feat["mid"].iloc[0])
        assert math.isnan(feat["microprice_dev"].iloc[0])

    def test_locked_book_produces_nan(self) -> None:
        df = _bbo_df([100.0], [100.0], [10.0], [10.0])
        feat = top_of_book_features(df)
        assert math.isnan(feat["spread"].iloc[0])

    def test_nan_bid_produces_nan(self) -> None:
        df = _bbo_df([float("nan")], [100.0], [10.0], [10.0])
        feat = top_of_book_features(df)
        assert math.isnan(feat["spread"].iloc[0])

    def test_mixed_rows(self) -> None:
        df = _bbo_df(
            [99.0, 101.0, 99.5],
            [100.0, 100.0, 100.5],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
        )
        feat = top_of_book_features(df)
        assert math.isfinite(feat["spread"].iloc[0])
        assert math.isnan(feat["spread"].iloc[1])  # crossed
        assert math.isfinite(feat["spread"].iloc[2])


# ===================================================================
# dollar_bars_with_microstructure — side_col wiring
# ===================================================================


class TestDollarBarsWithMicroSideWiring:
    def _make_tbbo_df(self, n: int = 100) -> pd.DataFrame:
        ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="ms")
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        return pd.DataFrame(
            {
                "price": prices,
                "size": rng.integers(1, 10, size=n).astype(float),
                "side": rng.choice(["A", "B"], size=n),
                "bid_px_00": prices - 0.05,
                "ask_px_00": prices + 0.05,
                "bid_sz_00": np.full(n, 100.0),
                "ask_sz_00": np.full(n, 100.0),
            },
            index=ts,
        )

    def test_side_col_default_reads_side(self) -> None:
        df = self._make_tbbo_df()
        bars = dollar_bars_with_microstructure(df, threshold=500.0)
        assert "signed_vol_imb" in bars.columns
        assert len(bars) > 0

    def test_side_col_none_uses_tick_rule(self) -> None:
        df = self._make_tbbo_df()
        bars_exch = dollar_bars_with_microstructure(df, threshold=500.0, side_col="side")
        bars_tick = dollar_bars_with_microstructure(df, threshold=500.0, side_col=None)
        assert not np.allclose(
            bars_exch["signed_vol_imb"].values,
            bars_tick["signed_vol_imb"].values,
            equal_nan=True,
        )


class TestTickRuleLeadingZeroFix:
    """S41 D4: _tick_rule_direction seeds +1 BEFORE carry-forward, so a leading
    run of zero-ticks (equal prices before the first move) resolves to +1 — not
    0. Pre-S41 the seed ran AFTER the loop, so leading zero-ticks kept 0 and
    silently zeroed that tick's signed flow (material on open/halt/illiquid)."""

    def test_leading_flat_run_resolves_to_plus_one(self) -> None:
        from quantcore.features.top_of_book import _tick_rule_direction

        # All examples: NO zero survives; leading flats become +1.
        np.testing.assert_array_equal(
            _tick_rule_direction(np.array([100.0, 100.0, 101.0])), [1.0, 1.0, 1.0]
        )
        np.testing.assert_array_equal(
            _tick_rule_direction(np.array([50.0, 50.0, 50.0])), [1.0, 1.0, 1.0]
        )
        # Real moves still classified correctly after a leading flat.
        np.testing.assert_array_equal(
            _tick_rule_direction(np.array([100.0, 100.0, 99.0])), [1.0, 1.0, -1.0]
        )

    def test_no_zero_direction_anywhere(self) -> None:
        """The tick rule must never emit 0 for a non-empty series (every tick is
        buy or sell once seeded) — the property the leading-zero bug violated."""
        from quantcore.features.top_of_book import _tick_rule_direction

        rng = np.random.default_rng(0)
        # price series with many repeats (rounded) incl. a leading flat run
        price = np.r_[np.full(5, 100.0), np.round(100.0 + np.cumsum(rng.normal(0, 0.3, 200)), 1)]
        tick_dir = _tick_rule_direction(price)
        assert set(np.unique(tick_dir)).issubset({-1.0, 1.0}), (
            f"tick rule emitted a 0 direction: {np.unique(tick_dir)}"
        )

    def test_signed_flow_no_longer_drops_leading_unknown_flat(self) -> None:
        """End-to-end: an unknown-side trade in a leading flat run now carries
        signed flow (was 0 pre-S41)."""
        df = pd.DataFrame(
            {
                "price": [100.0, 100.0, 100.0],
                "size": [10.0, 10.0, 10.0],
                "side": ["N", "N", "N"],
            }
        )
        flow = signed_flow_features(df, side_col="side")
        # all +1 → signed_volume == +size everywhere (none dropped to 0)
        np.testing.assert_array_equal(flow["tick_dir"].values, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(flow["signed_volume"].values, [10.0, 10.0, 10.0])
