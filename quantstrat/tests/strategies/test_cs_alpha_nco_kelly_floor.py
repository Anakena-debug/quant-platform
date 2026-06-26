"""Regression: cs_alpha_nco floors portfolio leverage at 0 on negative edge (s49).

Audit finding (re-verified 2026-05-31): the continuous-Kelly leverage
``f* = (w'μ)/(w'Σw)`` was clipped SYMMETRICALLY to ``[-kelly_cap, kelly_cap]``
at both the backtest and predict call sites. ``nco_weights`` returns a sum-to-1,
long-biased μ-tilted vector; when aggregate edge ``w'μ < 0`` the negative
leverage scalar multiplied through the whole vector, INVERTING the entire
cross-section into a nonsensical net-short book instead of holding cash.

The fix floors the clip lower bound to ``0.0`` at both sites (kept byte-identical
so predict() cannot diverge from the backtest). The primitive
``_portfolio_kelly_leverage`` is deliberately left sign-preserving — flooring is
a caller policy, so any other consumer still gets the true signed Kelly fraction.

Construction note: with all alpha signals carrying the SAME negative
expected_return ``c``, ``w'μ = c · Σw_i = c`` exactly (NCO weights sum to 1), so
``f* = c / (w'Σw) < 0`` at every active rebalance — a deterministic trigger.
Pre-fix these rebalances carried negative leverage and a net-short book; post-fix
they floor to 0 (hold cash). Each assertion below fails on the pre-fix code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantstrat.strategies.cs_alpha_nco import (
    CSAlphaNCOConfig,
    _portfolio_kelly_leverage,
    cs_alpha_nco_backtest,
)


def _negative_edge_panel(
    n_tickers: int = 6, n_days: int = 400, seed: int = 7, edge: float = -0.02
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic returns + monthly alpha signals with uniform NEGATIVE edge.

    Every signal is tradeable (interval strictly below zero) so the active set is
    non-empty, but the aggregate edge w'μ == edge < 0, forcing f* < 0.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.011, size=(n_days, n_tickers)), index=dates, columns=tickers
    )
    rebal = pd.date_range(dates.min(), dates.max(), freq="BMS")
    rows: list[tuple] = []
    for rd in rebal:
        nxt = dates[dates >= rd]
        if len(nxt) == 0:
            continue
        rd_actual = nxt[0]
        if dates.get_loc(rd_actual) < 21:
            continue
        for tk in tickers:
            rows.append((rd_actual, tk, edge, edge - 0.01, edge + 0.005))  # interval < 0
    sig = (
        pd.DataFrame(rows, columns=["date", "ticker", "expected_return", "lower", "upper"])
        .set_index(["date", "ticker"])
        .sort_index()
    )
    return returns, sig


def test_negative_aggregate_edge_floors_leverage_to_zero():
    returns, sig = _negative_edge_panel()
    cfg = CSAlphaNCOConfig(
        cov_estimator="lw",
        cov_lookback_days=120,
        kelly_fraction=0.5,
        kelly_cap=0.5,
        min_active_tickers=2,
    )
    res = cs_alpha_nco_backtest(alpha_signals=sig, panel_returns=returns, config=cfg)

    # Invariant: leverage is never negative (the fix). Pre-fix it went < 0.
    assert (res.portfolio_kelly_history >= -1e-12).all()

    # Discriminator: the panel DOES produce active rebalances (signals are
    # tradeable), and on every one the negative edge floored leverage to 0 —
    # pre-fix these carried strictly-negative leverage.
    active = res.n_active_history >= cfg.min_active_tickers
    assert active.any(), "panel did not produce active rebalances; test is vacuous"
    assert np.allclose(res.portfolio_kelly_history[active].to_numpy(), 0.0)

    # No net-short book: net exposure (signed weight sum) is >= 0 everywhere.
    net_exposure = res.weights_history.sum(axis=1)
    assert (net_exposure >= -1e-12).all()


def test_portfolio_kelly_leverage_remains_sign_preserving():
    """The primitive is intentionally NOT floored — it returns the true signed
    Kelly fraction; flooring is the strategy's caller-side policy. Guards against
    a future 'fix' that moves the floor into the primitive and breaks any other
    consumer that needs the real sign."""
    w = np.array([0.5, 0.5])
    cov = np.eye(2) * 0.01
    assert _portfolio_kelly_leverage(w, np.array([-0.02, -0.02]), cov) < 0  # negative edge
    assert _portfolio_kelly_leverage(w, np.array([0.02, 0.02]), cov) > 0  # positive edge
