"""Shared pytest fixtures across S24 + S25+ lifecycle smoke tests.

S24 (sealed) defines its fixtures in ``test_s24_e2e_smoke.py``. To
share them with S25+ test files without modifying the sealed file,
this conftest re-exports them at the directory-conftest level so
pytest's fixture-collection mechanism makes them available to every
test in this directory.

If you add new shared fixtures, prefer defining them here directly
rather than re-exporting.
"""

from __future__ import annotations

from test_s24_e2e_smoke import (  # noqa: F401  (re-exported for pytest discovery)
    alpha_signal,
    alpha_signals_mi,
    closes_panel,
    cov_lookback_window,
    cs_alpha_nco_result,
    dj30_tickers,
    latest_close_prices,
    latest_rebalance_date,
    latest_target_weights,
    market_snapshot,
    orders_empty,
    orders_pre_staged,
    portfolio_state_empty,
    portfolio_state_pre_staged,
    rebalance_engine,
    returns_panel_wide,
    signals_panel,
)
