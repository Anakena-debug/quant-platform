"""Smoke + invariant tests for `quantstrat.strategies.cs_alpha_nco`.

Synthetic data only — no parquet/network dependency. Tests pin three
contracts on the backtest function path:
  1. Config validation (ValueError on malformed inputs).
  2. n_clusters rule resolution (the ⌈√N⌉ rule we shipped to fix the
     ONC-at-high-N turnover blowup).
  3. Forward-leak invariant: signals dated > rebalance_date never affect
     positions established at or before that date.

Plus a smoke run on a 10-ticker, 500-bar synthetic panel that exercises
the full backtest function end-to-end and checks output shapes.

Plus ABC-conformance + ``predict()`` shape tests for the
``CSAlphaNCO(Strategy)`` class, parametrised over the cov_estimator axis
(sample / lw / rmt) — winner is regime-dependent per the cov_n100 sweep,
so no single fixed default; all three must produce well-formed signals.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal
from quantengine.strategies.base import Strategy

from quantstrat.portfolio.nco import nco_weights
from quantstrat.strategies.cs_alpha_nco import (
    CSAlphaNCO,
    CSAlphaNCOConfig,
    _fit_cov,
    _resolve_n_clusters,
    cs_alpha_nco_backtest,
)


# ─────────────────────────── fixtures ───────────────────────────────────


def _make_synthetic_panel(
    n_tickers: int = 10, n_days: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic returns + alpha signals.

    Returns are i.i.d. normal with a small positive drift. Alpha signals
    are published monthly; expected_return is the trailing 21-bar mean
    (a plausible-looking but uninformative forecast — fine for wiring tests).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets_arr = rng.normal(0.0005, 0.012, size=(n_days, n_tickers))
    returns = pd.DataFrame(rets_arr, index=dates, columns=tickers)

    rebal = pd.date_range(dates.min(), dates.max(), freq="BMS")
    rows: list[tuple] = []
    for rd in rebal:
        nxt = dates[dates >= rd]
        if len(nxt) == 0:
            continue
        rd_actual = nxt[0]
        loc = dates.get_loc(rd_actual)
        if loc < 21:
            continue
        trail = returns.iloc[loc - 21 : loc].mean()
        for tk in tickers:
            mu = float(trail[tk])
            sigma = 0.012
            rows.append((rd_actual, tk, mu, mu - 1.645 * sigma, mu + 1.645 * sigma))

    sig = (
        pd.DataFrame(rows, columns=["date", "ticker", "expected_return", "lower", "upper"])
        .set_index(["date", "ticker"])
        .sort_index()
    )
    return returns, sig


# ─────────────────────────── n_clusters rule ────────────────────────────


def test_sqrt_n_rule_returns_ceiling():
    assert _resolve_n_clusters("sqrt_n", 10, None) == 4  # ⌈√10⌉
    assert _resolve_n_clusters("sqrt_n", 100, None) == 10
    assert _resolve_n_clusters("sqrt_n", 4, None) == 2
    assert _resolve_n_clusters("sqrt_n", 1, None) == 1


def test_sqrt_n_rule_clamped_to_n_active():
    # n_active=2 → ⌈√2⌉=2 == n_active (no over-clamp)
    assert _resolve_n_clusters("sqrt_n", 2, None) == 2


def test_fixed_rule_clamped_to_n_active():
    assert _resolve_n_clusters("fixed", 5, 10) == 5
    assert _resolve_n_clusters("fixed", 100, 10) == 10


def test_onc_rule_returns_none():
    assert _resolve_n_clusters("onc", 100, None) is None


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown n_clusters_rule"):
        _resolve_n_clusters("bogus", 10, None)  # type: ignore[arg-type]


# ─────────────────────────── config validation ──────────────────────────


def test_config_kelly_fraction_bounds():
    with pytest.raises(ValueError, match="kelly_fraction"):
        CSAlphaNCOConfig(kelly_fraction=0.0)
    with pytest.raises(ValueError, match="kelly_fraction"):
        CSAlphaNCOConfig(kelly_fraction=2.0)
    # Valid edges
    CSAlphaNCOConfig(kelly_fraction=1.0)


def test_config_kelly_cap_bounds():
    with pytest.raises(ValueError, match="kelly_cap"):
        CSAlphaNCOConfig(kelly_cap=0.0)
    with pytest.raises(ValueError, match="kelly_cap"):
        CSAlphaNCOConfig(kelly_cap=10.0)


def test_config_cov_lookback_floor():
    with pytest.raises(ValueError, match="cov_lookback_days"):
        CSAlphaNCOConfig(cov_lookback_days=10)


def test_config_fixed_rule_requires_n_clusters_fixed():
    with pytest.raises(ValueError, match="n_clusters_fixed"):
        CSAlphaNCOConfig(n_clusters_rule="fixed")
    with pytest.raises(ValueError, match="n_clusters_fixed"):
        CSAlphaNCOConfig(n_clusters_rule="fixed", n_clusters_fixed=0)
    # Valid
    CSAlphaNCOConfig(n_clusters_rule="fixed", n_clusters_fixed=5)


def test_config_min_active_tickers_floor():
    with pytest.raises(ValueError, match="min_active_tickers"):
        CSAlphaNCOConfig(min_active_tickers=0)


# ─────────────────────────── input validation ───────────────────────────


def test_alpha_signals_must_have_multiindex():
    returns, _ = _make_synthetic_panel()
    flat = pd.DataFrame(
        {"expected_return": [0.01], "lower": [-0.01], "upper": [0.02]},
        index=pd.Index([pd.Timestamp("2020-02-03")], name="date"),
    )
    with pytest.raises(TypeError, match="MultiIndex"):
        cs_alpha_nco_backtest(
            alpha_signals=flat,
            panel_returns=returns,
            config=CSAlphaNCOConfig(),
        )


def test_alpha_signals_index_names():
    returns, sig = _make_synthetic_panel()
    sig_renamed = sig.copy()
    sig_renamed.index = sig_renamed.index.set_names(["bad", "name"])
    with pytest.raises(ValueError, match="index.names"):
        cs_alpha_nco_backtest(
            alpha_signals=sig_renamed,
            panel_returns=returns,
            config=CSAlphaNCOConfig(),
        )


def test_alpha_signals_required_columns():
    returns, sig = _make_synthetic_panel()
    sig_partial = sig.drop(columns=["lower"])
    with pytest.raises(ValueError, match="missing required columns"):
        cs_alpha_nco_backtest(
            alpha_signals=sig_partial,
            panel_returns=returns,
            config=CSAlphaNCOConfig(),
        )


def test_panel_returns_must_be_datetimeindex():
    _, sig = _make_synthetic_panel()
    bad = pd.DataFrame(np.random.randn(50, 10), columns=[f"T{i:02d}" for i in range(10)])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        cs_alpha_nco_backtest(
            alpha_signals=sig,
            panel_returns=bad,
            config=CSAlphaNCOConfig(),
        )


# ─────────────────────────── smoke + invariants ─────────────────────────


def test_smoke_n10_synthetic_returns_well_shaped_result():
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    cfg = CSAlphaNCOConfig(
        cov_estimator="lw",
        cov_lookback_days=120,
        n_clusters_rule="sqrt_n",
        kelly_fraction=0.5,
        kelly_cap=0.5,
        rebalance_freq="BMS",
    )
    res = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)

    assert isinstance(res.daily_pnl, pd.Series)
    assert len(res.daily_pnl) == len(returns)
    assert res.daily_pnl.index.equals(returns.index)
    assert len(res.rebalance_dates) >= 6  # ~24 months of data → ≥ 6 rebals
    assert res.weights_history.shape == (len(res.rebalance_dates), 10)
    assert res.turnover_history.index.equals(res.rebalance_dates)
    assert (res.n_active_history >= 0).all()
    # First-rebalance turnover is from 0 → first weights, so ≥ 0
    assert (res.turnover_history >= 0).all()


def test_cost_bps_charges_turnover_and_keeps_gross_separate():
    """cost_bps charges one-way cost per unit (two-way L1) turnover on rebalance
    dates; daily_pnl stays GROSS, daily_pnl_net subtracts the charge. Guards
    finding #4 (the backtest charged no cost yet was reported as net)."""
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    base = dict(
        cov_estimator="lw",
        cov_lookback_days=120,
        n_clusters_rule="sqrt_n",
        kelly_fraction=0.5,
        kelly_cap=0.5,
        rebalance_freq="BMS",
    )

    # Default (cost_bps=0): cost-free, net == gross.
    gross = cs_alpha_nco_backtest(
        alpha_signals=sig, panel_returns=returns, config=CSAlphaNCOConfig(**base)
    )
    assert (gross.daily_costs == 0.0).all()
    assert gross.daily_pnl_net.equals(gross.daily_pnl)

    # With a cost: daily_pnl stays gross; costs = cost_bps*1e-4*turnover on rebal dates.
    cost = 6.35  # bps one-way, ~ s86 measured standing-cost surface
    net = cs_alpha_nco_backtest(
        alpha_signals=sig, panel_returns=returns, config=CSAlphaNCOConfig(cost_bps=cost, **base)
    )
    assert net.daily_pnl.equals(gross.daily_pnl)  # gross is unchanged
    assert net.daily_costs.sum() > 0  # turnover was actually charged
    expected = cost * 1e-4 * net.turnover_history
    pd.testing.assert_series_equal(
        net.daily_costs.loc[net.rebalance_dates], expected, check_names=False
    )
    pd.testing.assert_series_equal(net.daily_pnl_net, net.daily_pnl - net.daily_costs)
    assert net.daily_pnl_net.sum() < gross.daily_pnl.sum()  # costs drag net below gross


def test_kelly_cap_enforced():
    """Portfolio leverage never exceeds cap (in absolute value)."""
    returns, sig = _make_synthetic_panel(seed=42)
    cfg = CSAlphaNCOConfig(kelly_cap=0.10)
    res = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)
    assert (res.portfolio_kelly_history.abs() <= cfg.kelly_cap + 1e-12).all()


def test_no_signal_no_position():
    """With empty alpha signals, the strategy holds cash forever."""
    returns, _ = _make_synthetic_panel()
    empty_idx = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    sig_empty = pd.DataFrame(columns=["expected_return", "lower", "upper"], index=empty_idx)
    res = cs_alpha_nco_backtest(
        alpha_signals=sig_empty,
        panel_returns=returns,
        config=CSAlphaNCOConfig(),
    )
    # Note: weights_history columns are tickers in panel order
    assert (res.weights_history.to_numpy() == 0.0).all()
    assert (res.daily_pnl == 0.0).all()
    # n_active = 0 at every rebalance
    assert (res.n_active_history == 0).all()


def test_no_forward_leak_via_future_dated_signal():
    """A future-dated signal must not affect rebalances dated ≤ its date."""
    returns, sig = _make_synthetic_panel()
    last_date = returns.index[-1]
    poison = pd.DataFrame(
        [[1e6, 1e6 - 1, 1e6 + 1]],
        columns=["expected_return", "lower", "upper"],
        index=pd.MultiIndex.from_tuples([(last_date, "T00")], names=["date", "ticker"]),
    )
    sig_poisoned = pd.concat([sig, poison]).sort_index()
    cfg = CSAlphaNCOConfig()
    res_clean = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)
    res_poisoned = cs_alpha_nco_backtest(
        alpha_signals=sig_poisoned, panel_returns=returns, config=cfg
    )
    early = res_clean.rebalance_dates[res_clean.rebalance_dates < last_date]
    np.testing.assert_array_almost_equal(
        res_clean.weights_history.loc[early].to_numpy(),
        res_poisoned.weights_history.loc[early].to_numpy(),
    )


def test_max_signal_age_days_drops_stale_signals():
    """A ticker with only a stale signal (older than max_signal_age_days
    before the rebalance) is excluded from the active universe."""
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    # Drop all signals after a chosen cutoff for ticker T00 to make its most
    # recent signal stale at later rebalances.
    cutoff = returns.index[60]
    sig_filtered = sig.copy()
    is_t00 = sig_filtered.index.get_level_values("ticker") == "T00"
    is_late = sig_filtered.index.get_level_values("date") > cutoff
    sig_filtered = sig_filtered[~(is_t00 & is_late)]

    cfg_no_filter = CSAlphaNCOConfig(max_signal_age_days=None)
    cfg_strict = CSAlphaNCOConfig(max_signal_age_days=30)

    res_no = cs_alpha_nco_backtest(
        alpha_signals=sig_filtered, panel_returns=returns, config=cfg_no_filter
    )
    res_strict = cs_alpha_nco_backtest(
        alpha_signals=sig_filtered, panel_returns=returns, config=cfg_strict
    )
    # Without the filter, T00 carries its (now-stale) cutoff-date signal forward
    # at every later rebalance — nonzero weights.
    later = res_no.rebalance_dates[res_no.rebalance_dates > cutoff + pd.Timedelta(days=120)]
    assert (res_no.weights_history.loc[later, "T00"].abs() > 0).any()
    # With the strict 30-day filter, T00 is dropped at every rebalance
    # > cutoff + 30 days.
    far_later = res_strict.rebalance_dates[
        res_strict.rebalance_dates > cutoff + pd.Timedelta(days=60)
    ]
    assert (res_strict.weights_history.loc[far_later, "T00"] == 0).all()


def test_max_signal_age_days_validation():
    with pytest.raises(ValueError, match="max_signal_age_days"):
        CSAlphaNCOConfig(max_signal_age_days=0)
    with pytest.raises(ValueError, match="max_signal_age_days"):
        CSAlphaNCOConfig(max_signal_age_days=-5)
    # None is valid (no filter)
    CSAlphaNCOConfig(max_signal_age_days=None)


def test_min_active_tickers_holds_cash():
    """When fewer than min_active_tickers have signals, positions go to 0."""
    returns, _ = _make_synthetic_panel(n_tickers=10)
    # Build sig where only T00 has signals
    rebal = pd.date_range(returns.index.min(), returns.index.max(), freq="BMS")
    rows = []
    for rd in rebal:
        nxt = returns.index[returns.index >= rd]
        if len(nxt) == 0:
            continue
        rd_actual = nxt[0]
        rows.append((rd_actual, "T00", 0.001, -0.01, 0.01))
    sig_one = (
        pd.DataFrame(rows, columns=["date", "ticker", "expected_return", "lower", "upper"])
        .set_index(["date", "ticker"])
        .sort_index()
    )
    cfg = CSAlphaNCOConfig(min_active_tickers=2)
    res = cs_alpha_nco_backtest(alpha_signals=sig_one, panel_returns=returns, config=cfg)
    # n_active is at most 1 at every rebalance, so we always hold cash
    assert (res.n_active_history <= 1).all()
    assert (res.weights_history.to_numpy() == 0.0).all()


# ─────────────────────────── CSAlphaNCO(Strategy) ABC adapter ───────────


def _make_market_snapshot(returns: pd.DataFrame, asof: pd.Timestamp) -> MarketSnapshot:
    """Build a MarketSnapshot for the universe at ``asof`` using cumulative
    return-derived prices (positive by construction). The snapshot is the
    rebalance trigger, not a price source for the strategy itself — the
    strategy reads its own ``panel_returns`` for cov estimation.
    """
    tickers = tuple(returns.columns)
    base = 100.0 + np.arange(len(tickers))  # tickers get distinct positive bases
    return MarketSnapshot(
        timestamp=asof.isoformat(),
        tickers=tickers,
        prices=base.astype(np.float64),
    )


def test_csalphanco_inherits_strategy():
    assert issubclass(CSAlphaNCO, Strategy)
    assert Strategy in CSAlphaNCO.__mro__


def test_csalphanco_construction_validates_inputs():
    returns, sig = _make_synthetic_panel()
    cfg = CSAlphaNCOConfig()

    # Bad alpha_signals index
    flat = pd.DataFrame(
        {"expected_return": [0.0], "lower": [-1e-3], "upper": [1e-3]},
        index=pd.Index([returns.index[0]], name="date"),
    )
    with pytest.raises(TypeError, match="MultiIndex"):
        CSAlphaNCO(alpha_signals=flat, panel_returns=returns, config=cfg)

    # Missing required column
    sig_partial = sig.drop(columns=["upper"])
    with pytest.raises(ValueError, match="missing required columns"):
        CSAlphaNCO(alpha_signals=sig_partial, panel_returns=returns, config=cfg)

    # Bad panel_returns index
    bad = pd.DataFrame(np.zeros((10, 10)), columns=[f"T{i:02d}" for i in range(10)])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        CSAlphaNCO(alpha_signals=sig, panel_returns=bad, config=cfg)

    # Miscoverage out of range
    with pytest.raises(ValueError, match="miscoverage"):
        CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg, miscoverage=0.0)
    with pytest.raises(ValueError, match="miscoverage"):
        CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg, miscoverage=1.5)


@pytest.mark.parametrize("cov_estimator", ["sample", "lw", "rmt"])
def test_csalphanco_predict_returns_well_formed_alpha_signal(cov_estimator):
    """predict() must return an AlphaSignal aligned to market.tickers,
    with kelly_weights bounded by the configured kelly_cap, for every
    cov_estimator on the {sample, lw, rmt} axis (winner is regime-
    dependent — all three must produce well-formed signals)."""
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    cfg = CSAlphaNCOConfig(
        cov_estimator=cov_estimator,
        cov_lookback_days=120,
        n_clusters_rule="sqrt_n",
        kelly_fraction=0.5,
        kelly_cap=0.5,
    )
    strat = CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg)

    asof = cast(pd.Timestamp, returns.index[400])  # well past the cov-lookback floor
    market = _make_market_snapshot(returns, asof)
    out = strat.predict(market)

    assert isinstance(out, AlphaSignal)
    assert out.tickers == market.tickers
    assert out.expected_return.shape == (len(market.tickers),)
    assert out.lower.shape == (len(market.tickers),)
    assert out.upper.shape == (len(market.tickers),)
    assert out.kelly_weights is not None
    assert out.kelly_weights.shape == (len(market.tickers),)
    assert np.all(out.lower <= out.upper)
    # NCO weights sum to 1 (per quantstrat.portfolio.nco contract); leverage
    # is the kelly_fraction · f* scalar clipped to [-kelly_cap, kelly_cap].
    # So net exposure (signed sum of kelly_weights) is the leverage, capped.
    # Gross exposure (L1 sum) is unbounded — long-short MV portfolios can
    # have arbitrary L1 norm.
    assert abs(float(np.sum(out.kelly_weights))) <= cfg.kelly_cap + 1e-9
    assert out.metadata.get("cov_estimator") == cov_estimator


def test_csalphanco_predict_no_forward_leak():
    """A future-dated signal must not affect a predict() call dated
    ≤ that signal's date. Mirrors the backtest forward-leak test against
    the live predict() pathway."""
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    cfg = CSAlphaNCOConfig(cov_lookback_days=120)
    asof = cast(pd.Timestamp, returns.index[300])
    market = _make_market_snapshot(returns, asof)

    # Poison: a far-future, extreme signal that should be ignored at asof
    poison_date = cast(pd.Timestamp, returns.index[-1])
    poison = pd.DataFrame(
        [[1e6, 1e6 - 1, 1e6 + 1]],
        columns=["expected_return", "lower", "upper"],
        index=pd.MultiIndex.from_tuples([(poison_date, "T00")], names=["date", "ticker"]),
    )
    sig_poisoned = pd.concat([sig, poison]).sort_index()

    strat_clean = CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg)
    strat_poisoned = CSAlphaNCO(alpha_signals=sig_poisoned, panel_returns=returns, config=cfg)
    out_clean = strat_clean.predict(market)
    out_poisoned = strat_poisoned.predict(market)
    assert out_clean.kelly_weights is not None
    assert out_poisoned.kelly_weights is not None
    np.testing.assert_array_almost_equal(out_clean.kelly_weights, out_poisoned.kelly_weights)


def test_csalphanco_predict_holds_cash_when_universe_starved():
    """With empty alpha signals the strategy must emit a no-trade signal
    (kelly_weights all zero, all tickers non-tradeable)."""
    returns, _ = _make_synthetic_panel(n_tickers=10)
    empty_idx = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    sig_empty = pd.DataFrame(columns=["expected_return", "lower", "upper"], index=empty_idx)
    cfg = CSAlphaNCOConfig()
    strat = CSAlphaNCO(alpha_signals=sig_empty, panel_returns=returns, config=cfg)
    market = _make_market_snapshot(returns, cast(pd.Timestamp, returns.index[400]))
    out = strat.predict(market)

    assert out.kelly_weights is not None
    assert np.all(out.kelly_weights == 0.0)
    assert not out.tradeable.any()  # no interval excludes zero
    assert out.metadata.get("n_active") == 0


def test_csalphanco_update_is_noop():
    """Default Strategy.update() should be a no-op for this composition-only
    strategy — calibration lives in the upstream quantcore pipeline."""
    returns, sig = _make_synthetic_panel()
    cfg = CSAlphaNCOConfig()
    strat = CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg)
    # Should not raise, returns None.
    assert strat.update({"T00": 0.01}) is None


def test_predict_matches_backtest_at_rebalance_date():
    """predict() at rebalance_dates[i] must produce the same kelly_weights
    as cs_alpha_nco_backtest() records for that same date.

    Pins the two surfaces (live + backtest) against parallel-orchestration
    drift: any future change that touches one path but not the other will
    break this test rather than silently diverge the harness numbers from
    the live signal stream.
    """
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    cfg = CSAlphaNCOConfig(cov_lookback_days=120)
    res = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)
    rd = cast(pd.Timestamp, res.rebalance_dates[5])  # past warmup

    strat = CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg)
    market = _make_market_snapshot(returns, rd)
    out = strat.predict(market)

    bt_w = res.weights_history.loc[rd].reindex(list(market.tickers)).fillna(0).to_numpy()
    assert out.kelly_weights is not None
    np.testing.assert_array_almost_equal(out.kelly_weights, bt_w, decimal=10)


# ─────────────────────────── detone (s77) ───────────────────────────────
def test_config_detone_validation():
    assert CSAlphaNCOConfig().detone is False  # off by default (back-compat)
    assert CSAlphaNCOConfig(detone=True).detone is True
    with pytest.raises(ValueError, match="n_market_factors must be"):
        CSAlphaNCOConfig(detone=True, n_market_factors=0)


def test_detone_de_concentrates_nco_book():
    """Audit mechanism on a controlled fixture: a market factor with heterogeneous betas and
    similar idiosyncratic vol gives a market-mode-dominated cov → NCO concentrates in the low-beta
    names; detoning strips the common mode → the (near-uncorrelated, similar-vol) idiosyncratic
    structure → NCO spreads out. Exercises the wired _fit_cov(detone=...) path."""
    rng = np.random.default_rng(7)
    n_obs, n = 400, 6
    mkt = rng.normal(0.0, 0.02, (n_obs, 1))
    betas = np.linspace(0.2, 2.0, n)
    rets = betas[None, :] * mkt + 0.004 * rng.standard_normal((n_obs, n))

    cov_toned = _fit_cov("sample", rets, detone=False)
    cov_detoned = _fit_cov("sample", rets, detone=True, n_market_factors=1)
    w_toned = nco_weights(cov_toned)
    w_detoned = nco_weights(cov_detoned)

    for w in (w_toned, w_detoned):  # both valid books
        assert np.isclose(w.sum(), 1.0) and np.all(np.isfinite(w))
    assert np.any(np.abs(w_toned - w_detoned) > 1e-6)  # detone changed the allocation

    def _mean_abs_offdiag_corr(cov: np.ndarray) -> float:
        s = np.sqrt(np.diag(cov))
        c = cov / np.outer(s, s)
        off = ~np.eye(cov.shape[0], dtype=bool)
        return float(np.mean(np.abs(c[off])))

    # mechanism: detoning collapses the common market mode (off-diagonal correlation drops) …
    assert _mean_abs_offdiag_corr(cov_detoned) < _mean_abs_offdiag_corr(cov_toned)
    # … and the resulting NCO book is less concentrated (lower Herfindahl).
    assert float(np.sum(w_detoned**2)) < float(np.sum(w_toned**2))


def test_predict_matches_backtest_with_detone():
    """The dual-surface byte-identity holds with detone=True — both surfaces detone through the
    shared _fit_cov, so they cannot drift (the s43/s44 dual-surface contract, extended to detone)."""
    returns, sig = _make_synthetic_panel(n_tickers=10, n_days=500, seed=42)
    cfg = CSAlphaNCOConfig(cov_lookback_days=120, detone=True)
    res = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)
    rd = cast(pd.Timestamp, res.rebalance_dates[5])

    strat = CSAlphaNCO(alpha_signals=sig, panel_returns=returns, config=cfg)
    market = _make_market_snapshot(returns, rd)
    out = strat.predict(market)

    bt_w = res.weights_history.loc[rd].reindex(list(market.tickers)).fillna(0).to_numpy()
    assert out.kelly_weights is not None
    np.testing.assert_array_almost_equal(out.kelly_weights, bt_w, decimal=10)
