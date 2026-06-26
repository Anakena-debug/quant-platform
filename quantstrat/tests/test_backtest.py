"""quantstrat — e2e backtest harness test (trivial strategy + synthetic price panel)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantengine.contracts.signal import build_alpha_signal
from quantengine.execution.cost_model import CostRealismWarning, LinearCostModel
from quantengine.execution.paper import PaperBroker
from quantengine.strategies.base import Strategy

from quantstrat.backtest import CostRealismError, run_backtest
from quantstrat.metrics.performance import RelativeMetrics


class _EqualWeightLong(Strategy):
    """Trivial frozen strategy: equal-weight long every priced name (for harness testing)."""

    def predict(self, market):  # noqa: ANN001 — quantengine MarketSnapshot
        n = len(market.tickers)
        w = [1.0 / n] * n
        return build_alpha_signal(
            tickers=market.tickers,
            expected_return=[1e-6] * n,
            lower=[1e-6] * n,
            upper=[1.0] * n,
            alpha=0.1,
            kelly_weights=w,
            timestamp=market.timestamp,
        )


def _price_panel(n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    data = {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days))) for t in tickers}
    return pd.DataFrame(data, index=dates)


def test_run_backtest_produces_a_sane_report():
    panel = _price_panel()
    res = run_backtest(
        _EqualWeightLong(), panel, initial_cash=1_000_000.0, allow_optimistic_costs=True
    )
    r = res.report
    # equity curve reconstructed over the panel
    assert len(r.nav) == len(panel)
    assert (r.nav > 0).all()
    assert len(r.returns) == len(r.nav) - 1
    assert np.isfinite(r.ann_return) and np.isfinite(r.ann_vol)
    assert res.skipped_steps == 0  # every panel date has data
    # the strategy actually traded (fills recorded)
    assert len(res.run_frames["fills"]) > 0


def test_run_backtest_accepts_parquet_path(tmp_path):
    panel = _price_panel(seed=1)
    pq = tmp_path / "prices.parquet"
    panel.to_parquet(pq)
    res = run_backtest(_EqualWeightLong(), pq, initial_cash=500_000.0, allow_optimistic_costs=True)
    assert len(res.report.nav) == len(panel)
    assert res.report.nav.iloc[0] > 0


def test_run_backtest_attaches_relative_metrics_when_benchmark_given():
    panel = _price_panel(seed=2)
    # no benchmark -> no relative block
    assert run_backtest(_EqualWeightLong(), panel, allow_optimistic_costs=True).relative is None

    # a benchmark return Series over the panel dates -> relative metrics attached
    bench = pd.Series(np.random.default_rng(5).normal(0.0004, 0.01, len(panel)), index=panel.index)
    res = run_backtest(_EqualWeightLong(), panel, benchmark=bench, allow_optimistic_costs=True)
    assert isinstance(res.relative, RelativeMetrics)
    assert np.isfinite(res.relative.beta)
    assert np.isfinite(res.relative.tracking_error)


def test_run_backtest_refuses_optimistic_costs_unless_opted_in():
    # Fail-closed: the default broker prices fills at the optimistic 1bp slippage, so a plain
    # run is REFUSED (cannot silently report net performance on under-charged costs). Opting in
    # explicitly downgrades the refusal to a loud warning and flags the result honestly.
    panel = _price_panel(seed=3)
    with pytest.raises(CostRealismError, match="optimistic execution costs"):
        run_backtest(_EqualWeightLong(), panel)
    with pytest.warns(CostRealismWarning, match="optimistic execution costs"):
        res = run_backtest(_EqualWeightLong(), panel, allow_optimistic_costs=True)
    assert res.cost_assumptions is not None
    assert res.cost_assumptions["optimistic"] is True
    assert res.cost_assumptions["slippage_bps"] == 1.0


def test_run_backtest_realistic_costs_are_silent_and_flagged_honest():
    # An honest cost model (realistic preset) must NOT warn and must be flagged non-optimistic.
    panel = _price_panel(seed=4)
    broker = PaperBroker(cost_model=LinearCostModel.realistic())
    with warnings.catch_warnings():
        warnings.simplefilter("error", CostRealismWarning)  # any such warning fails the test
        res = run_backtest(_EqualWeightLong(), panel, broker=broker)
    assert res.cost_assumptions is not None
    assert res.cost_assumptions["optimistic"] is False
    assert res.cost_assumptions["slippage_bps"] == 5.0
