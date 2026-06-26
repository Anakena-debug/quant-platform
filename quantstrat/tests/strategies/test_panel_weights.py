"""PanelWeightsStrategy (s91) — dual-surface consistency + handoff replay.

House rule (s60 lesson): a Strategy that mirrors another weight surface ships its
byte-equal consistency pin IN THE SAME COMMIT — predict() is pinned to
``weights.loc[asof]`` at decimal=10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantengine.contracts.market import MarketSnapshot

from quantstrat.backtest import run_backtest
from quantstrat.backtest.evaluate import deflated_evaluation
from quantstrat.strategies import PanelWeightsStrategy


def _weights_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    rng = np.random.default_rng(2)
    raw = pd.DataFrame(rng.normal(0, 1, (8, 4)), index=dates, columns=["AAA", "BBB", "CCC", "DDD"])
    pct = raw.rank(axis=1, pct=True)
    w = pct.where(pct > 0.75, 0.0).where(pct <= 0.75, 0.5)  # top quartile long 0.5
    w = w - w.mean(axis=1).to_numpy()[:, None] * 0.0  # keep simple long-tilt book
    return w


def _market(ts: pd.Timestamp, tickers=("AAA", "BBB", "CCC", "DDD")) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=ts.isoformat(),
        tickers=tuple(tickers),
        prices=np.full(len(tickers), 100.0),
    )


def test_predict_pins_to_weights_row_decimal_10():
    w = _weights_panel()
    strat = PanelWeightsStrategy(w)
    ts = w.index[3] + pd.Timedelta(hours=16)  # intraday timestamp -> asof = row 3
    sig = strat.predict(_market(ts))
    expected = w.loc[w.index[3]].reindex(list(sig.tickers)).fillna(0.0).to_numpy()
    np.testing.assert_array_almost_equal(sig.kelly_weights, expected, decimal=10)
    assert sig.metadata["weights_row"] == str(w.index[3])


def test_sign_encoding_matches_the_live_producer_contract():
    dates = pd.bdate_range("2024-01-01", periods=2)
    w = pd.DataFrame({"L": [0.5, 0.5], "S": [-0.5, -0.5], "F": [0.0, 0.0]}, index=dates)
    sig = PanelWeightsStrategy(w).predict(_market(dates[-1], tickers=("F", "L", "S")))
    i = {t: k for k, t in enumerate(sig.tickers)}
    # long: lower>0; short: upper<0; flat: interval brackets 0 (untradeable)
    assert sig.lower[i["L"]] > 0 and sig.upper[i["L"]] == 1.0
    assert sig.upper[i["S"]] < 0 and sig.lower[i["S"]] == -1.0
    assert sig.lower[i["F"]] < 0 < sig.upper[i["F"]]


def test_unknown_names_in_snapshot_are_flat():
    w = _weights_panel()
    sig = PanelWeightsStrategy(w).predict(_market(w.index[-1], tickers=("AAA", "ZZZ")))
    i = {t: k for k, t in enumerate(sig.tickers)}
    assert sig.kelly_weights[i["ZZZ"]] == 0.0


def test_predict_before_history_raises():
    w = _weights_panel()
    with pytest.raises(ValueError, match="no weights row"):
        PanelWeightsStrategy(w).predict(_market(w.index[0] - pd.Timedelta(days=9)))


def test_handoff_chain_panel_to_backtest_to_dsr():
    """The paved road in one test: weights panel -> run_backtest -> deflated_evaluation."""
    w = _weights_panel()
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (8, 4)), axis=0)),
        index=w.index,
        columns=w.columns,
    )
    res = run_backtest(PanelWeightsStrategy(w), prices, allow_optimistic_costs=True)
    assert len(res.run_frames["fills"]) > 0  # the book actually trades
    report = deflated_evaluation(res.report.returns, n_trials=18, n_windows=2)
    assert np.isfinite(report.dsr) and 0.0 <= report.dsr <= 1.0
    assert report.n_trials == 18
