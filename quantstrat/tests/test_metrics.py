"""Tests for quantstrat.metrics.performance.

Covers:

- Array primitives (annualised return / vol, Sortino, max-DD, Calmar, turnover)
  on crafted returns series with closed-form expected values.
- The structured ``DrawdownEvent`` — magnitude, peak / trough / recovery indices,
  including the no-recovery case.
- The public ``compute_performance`` entry point on a synthetic ``run_frames`` dict
  + price panel, verifying NAV reconstruction and that the ``PerformanceReport``
  is populated end-to-end.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantstrat.metrics.performance import (
    DrawdownEvent,
    PerformanceReport,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_performance,
    max_drawdown,
    sortino_ratio,
    turnover_from_weights,
)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


class TestAnnualizedReturn:
    def test_zero_returns_compound_to_zero(self):
        assert annualized_return(np.zeros(252)) == pytest.approx(0.0)

    def test_constant_daily_return(self):
        r = np.full(252, 0.001)
        expected = (1.001**252) - 1.0
        assert annualized_return(r) == pytest.approx(expected, rel=1e-9)

    def test_accepts_pandas_series(self):
        r = pd.Series(np.full(252, 0.001))
        assert annualized_return(r) == pytest.approx((1.001**252) - 1.0, rel=1e-9)

    def test_empty_returns_zero(self):
        assert annualized_return(np.array([])) == 0.0

    def test_full_wipeout_returns_neg_one(self):
        r = np.array([-1.0, 0.0, 0.0])
        assert annualized_return(r) == -1.0


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------


class TestAnnualizedVolatility:
    def test_constant_returns_zero_vol(self):
        assert annualized_volatility(np.full(252, 0.01)) == pytest.approx(0.0, abs=1e-15)

    def test_known_gaussian(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, size=10_000)
        expected = 0.01 * math.sqrt(252)
        assert annualized_volatility(r) == pytest.approx(expected, rel=0.05)

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            annualized_volatility(np.array([0.01]))


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_no_downside_raises(self):
        with pytest.raises(ValueError, match="downside deviation degenerate"):
            sortino_ratio(np.full(100, 0.01))

    def test_known_value(self):
        # r = 50 × -0.01 followed by 50 × +0.03 (length 100):
        #   mean = 0.01
        #   mean(min(r, 0) ** 2) = 0.5 · 1e-4 = 5e-5
        #   dd_dev = sqrt(5e-5)
        #   Sortino = 0.01 / sqrt(5e-5) · sqrt(252) = sqrt(2 · 252)
        r = np.concatenate([np.full(50, -0.01), np.full(50, 0.03)])
        assert sortino_ratio(r) == pytest.approx(math.sqrt(2 * 252), rel=1e-9)

    def test_too_few_obs_raises(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            sortino_ratio(np.array([0.01]))


# ---------------------------------------------------------------------------
# max_drawdown — structured DrawdownEvent
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_no_drawdown_returns_zero_magnitude(self):
        ev = max_drawdown(np.full(100, 0.01))
        assert isinstance(ev, DrawdownEvent)
        assert ev.magnitude == 0.0
        assert ev.recovery_index is None

    def test_known_drawdown_positional_indices(self):
        # +10%, -20%, +5%: equity = [1.10, 0.88, 0.924]
        # peak at 0 (1.10), trough at 1 (0.88), magnitude = 0.88/1.10 - 1 = -0.20.
        # no recovery within the series (0.924 < 1.10).
        r = np.array([0.10, -0.20, 0.05])
        ev = max_drawdown(r)
        assert ev.magnitude == pytest.approx(-0.20, rel=1e-9)
        assert ev.peak_index == 0
        assert ev.trough_index == 1
        assert ev.recovery_index is None

    def test_drawdown_with_recovery(self):
        # +10%, -30%, +50%, +10%:
        # equity = [1.10, 0.77, 1.155, 1.2705].
        # trough at 1 (0.77), magnitude ≈ -0.30, recovery first reached at idx 2.
        r = np.array([0.10, -0.30, 0.50, 0.10])
        ev = max_drawdown(r)
        assert ev.magnitude == pytest.approx(-0.30, rel=1e-9)
        assert ev.peak_index == 0
        assert ev.trough_index == 1
        assert ev.recovery_index == 2

    def test_preserves_series_timestamp_index(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="D")
        r = pd.Series([0.10, -0.20, 0.05], index=idx)
        ev = max_drawdown(r)
        assert ev.peak_index == idx[0]
        assert ev.trough_index == idx[1]
        assert ev.recovery_index is None

    def test_empty_returns_none_indices(self):
        ev = max_drawdown(np.array([]))
        assert ev.magnitude == 0.0
        assert ev.peak_index is None
        assert ev.trough_index is None
        assert ev.recovery_index is None


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_none_on_zero_drawdown(self):
        assert calmar_ratio(np.full(252, 0.001)) is None

    def test_matches_manual_computation(self):
        r = np.array([0.10, -0.20, 0.05] * 84)  # 252 obs
        expected = annualized_return(r) / abs(max_drawdown(r).magnitude)
        assert calmar_ratio(r) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# turnover_from_weights
# ---------------------------------------------------------------------------


class TestTurnoverFromWeights:
    def test_static_weights_zero(self):
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
        assert turnover_from_weights(w) == 0.0

    def test_full_flip_one_sided(self):
        w = pd.DataFrame({"A": [1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0]})
        assert turnover_from_weights(w) == pytest.approx(1.0)

    def test_two_sided(self):
        w = pd.DataFrame({"A": [1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0]})
        assert turnover_from_weights(w, one_sided=False) == pytest.approx(2.0)

    def test_single_row_returns_zero(self):
        w = pd.DataFrame({"A": [0.5], "B": [0.5]})
        assert turnover_from_weights(w) == 0.0

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            turnover_from_weights(np.array([[0.5, 0.5], [0.5, 0.5]]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_performance — public entry point, end-to-end
# ---------------------------------------------------------------------------


def _make_synthetic_run_frames() -> dict[str, pd.DataFrame]:
    """One-ticker round trip: buy 10 AAA @ 50 on day 0, sell 10 @ 55 on day 2."""
    run = pd.DataFrame(
        [
            {
                "run_id": "test-run",
                "initial_cash": 1000.0,
                "final_cash": 1050.0,
                "realized_pnl": 50.0,
                "total_commission": 0.0,
                "skipped_steps": 0,
                "n_events": 2,
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "run_id": "test-run",
                "seq": 0,
                "timestamp": "2026-01-01T00:00:00",
                "ticker": "AAA",
                "signed_quantity": 10,
                "price": 50.0,
                "commission": 0.0,
            },
            {
                "run_id": "test-run",
                "seq": 1,
                "timestamp": "2026-01-03T00:00:00",
                "ticker": "AAA",
                "signed_quantity": -10,
                "price": 55.0,
                "commission": 0.0,
            },
        ]
    )
    return {"run": run, "fills": fills}


def _make_synthetic_price_panel() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame({"AAA": [50.0, 52.0, 55.0, 55.0, 55.0], "BBB": [100.0] * 5}, index=idx)


class TestComputePerformance:
    def test_end_to_end_round_trip(self):
        run_frames = _make_synthetic_run_frames()
        panel = _make_synthetic_price_panel()

        report = compute_performance(run_frames, panel)

        assert isinstance(report, PerformanceReport)
        # Walked-forward NAV:
        #   day 0: buy 10 AAA @ 50 → cash = 500, pos = 10, NAV = 500 + 10·50 = 1000.
        #   day 1: NAV = 500 + 10·52 = 1020.
        #   day 2: sell 10 AAA @ 55 → cash = 1050, pos = 0, NAV = 1050.
        #   day 3–4: NAV stays at 1050.
        assert report.nav.iloc[0] == pytest.approx(1000.0)
        assert report.nav.iloc[1] == pytest.approx(1020.0)
        assert report.nav.iloc[2] == pytest.approx(1050.0)
        assert report.nav.iloc[-1] == pytest.approx(1050.0)
        # Four pct-change rows after dropna.
        assert report.returns.size == 4
        # Strictly positive compounded return: 1050 / 1000.
        assert report.ann_return > 0.0
        # No downside observation in the NAV path (all non-negative pct_change);
        # compute_performance swallows the ValueError and records NaN.
        assert math.isnan(report.sortino)
        # Equity curve is non-decreasing, so no drawdown and Calmar is undefined.
        assert report.max_drawdown.magnitude == 0.0
        assert report.calmar is None
        # Weights change when the position opens and closes → positive turnover.
        assert report.turnover > 0.0

    def test_missing_required_frame_raises(self):
        with pytest.raises(KeyError, match="missing required keys"):
            compute_performance({"run": pd.DataFrame()}, _make_synthetic_price_panel())

    def test_empty_run_frame_raises(self):
        run_frames = {
            "run": pd.DataFrame(columns=["run_id", "initial_cash"]),
            "fills": pd.DataFrame(),
        }
        with pytest.raises(ValueError, match="'run' frame is empty"):
            compute_performance(run_frames, _make_synthetic_price_panel())

    def test_missing_initial_cash_column_raises(self):
        run_frames = {
            "run": pd.DataFrame([{"run_id": "test-run", "final_cash": 1050.0}]),
            "fills": pd.DataFrame(),
        }
        with pytest.raises(KeyError, match="missing 'initial_cash' column"):
            compute_performance(run_frames, _make_synthetic_price_panel())
