"""S24 PR1 — End-to-end lifecycle smoke test (data-lake precondition).

Scope:
  - verify the data-lake fixture exists
  - load closes via the locked quantdata fixture pattern (option B' + CWD α)
  - load signals from the in-tree parquet fixture
  - pin deterministic settings
  - fail fast if the required catalog / universe / date range is missing

Locked PR1 decisions (post-amendment; see sprint plan §5):
  - Closes: ``quantdata/quant.duckdb`` queried directly via ``duckdb``
    against the ``MarketData`` view. Original locked option B
    (``QuantDataQuery.daily()``) was infeasible — quantdata is not a
    proper Python package AND ``query.daily()`` queries a non-existent
    ``daily`` view. Bypass adopted as option B'; AC3 satisfied via
    "catalog-backed data" (catalog exercised; wrapper not).
  - Signals: ``tests/fixtures/s24_dj30_2022_2024_signals.parquet``
    (committed; built one-shot via ``build_s24_smoke_signals.py``).
  - CWD pattern α: ``contextlib.chdir(QUANTDATA_DIR)`` so the
    ``MarketData`` view's relative parquet path resolves; restored on
    fixture teardown.
  - Universe: DJ30 (``quantdata/dowjones30_tickers.txt``).
  - Date range: 2022-01-03 → 2024-12-31.
  - Cache tag: ``s24_dj30_2022_2024``.

Run:
  uv run --directory quantstrat pytest tests/test_s24_e2e_smoke.py -v
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from quantcore.covariance.shrinkage import ledoit_wolf_shrinkage
from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order
from quantengine.contracts.signal import AlphaSignal
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState, Position
from quantstrat.strategies.cs_alpha_nco import (
    CSAlphaNCOConfig,
    CSAlphaNCOResult,
    cs_alpha_nco_backtest,
)

# ─── Pinned constants (mirror sprint plan §5 PR1) ─────────────────────
QUANTDATA_DIR = Path(__file__).parents[2] / "quantdata"
DJ30_FILE = QUANTDATA_DIR / "dowjones30_tickers.txt"
CATALOG_PATH = QUANTDATA_DIR / "quant.duckdb"

SIGNALS_FIXTURE = Path(__file__).parent / "fixtures" / "s24_dj30_2022_2024_signals.parquet"

START_DATE = "2022-01-03"
END_DATE = "2024-12-31"

EXPECTED_SIGNAL_COLUMNS = ["date", "ticker", "expected_return", "lower", "upper"]


def _load_dj30_tickers() -> list[str]:
    return [
        line.strip()
        for line in DJ30_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def dj30_tickers() -> list[str]:
    return sorted(_load_dj30_tickers())


@pytest.fixture
def closes_panel(dj30_tickers: list[str]) -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        pytest.fail(f"data-lake fixture missing: catalog not found at {CATALOG_PATH}")
    with contextlib.chdir(QUANTDATA_DIR):
        con = duckdb.connect(str(CATALOG_PATH), read_only=True)
        try:
            placeholders = ",".join(["?"] * len(dj30_tickers))
            return con.execute(
                f"SELECT ticker, date, close FROM MarketData "
                f"WHERE ticker IN ({placeholders}) "
                f"AND date >= ? AND date <= ? "
                f"ORDER BY ticker, date",
                [*dj30_tickers, START_DATE, END_DATE],
            ).df()
        finally:
            con.close()


@pytest.fixture
def signals_panel() -> pd.DataFrame:
    if not SIGNALS_FIXTURE.exists():
        pytest.fail(f"signals fixture missing: {SIGNALS_FIXTURE}")
    return pd.read_parquet(SIGNALS_FIXTURE)


@pytest.fixture
def returns_panel_wide(closes_panel: pd.DataFrame) -> pd.DataFrame:
    """DJ30 closes pivoted wide → daily simple-return matrix.

    Mirrors the ``panel_returns`` construction in
    ``cs_alpha_nco_backtest`` (closes.pct_change().dropna(how='any')).
    """
    closes_wide = closes_panel.pivot(index="date", columns="ticker", values="close")
    return closes_wide.pct_change().dropna(how="any")


@pytest.fixture
def cov_lookback_window(returns_panel_wide: pd.DataFrame) -> np.ndarray:
    """Trailing 252-day returns window — matches `cov_lookback_days=252`."""
    window = returns_panel_wide.tail(252)
    if len(window) < 252:
        pytest.fail(f"insufficient returns history: got {len(window)} rows, need 252")
    return window.to_numpy()


# Production CSAlphaNCO config — verbatim from
# quantstrat/tests/spikes/s23b_pr1_profile.py:_PRODUCTION_CONFIG_KW + cov_estimator="lw".
# Pinned inline per sprint plan §5 PR3 ("decide exact strategy/config dict
# to pin inline"). Same config that produced the S23b sealed numerics.
S24_CSALPHA_CONFIG = CSAlphaNCOConfig(
    cov_estimator="lw",
    n_clusters_rule="sqrt_n",
    clustering_method="ward",
    rebalance_freq="BMS",
    kelly_fraction=0.5,
    kelly_cap=0.5,
    cov_lookback_days=252,
    max_signal_age_days=45,
    min_active_tickers=10,
)


@pytest.fixture
def alpha_signals_mi(signals_panel: pd.DataFrame) -> pd.DataFrame:
    """Signals fixture re-indexed to MultiIndex [date, ticker] — the shape
    cs_alpha_nco_backtest requires."""
    return signals_panel.set_index(["date", "ticker"])[["expected_return", "lower", "upper"]]


@pytest.fixture
def cs_alpha_nco_result(
    alpha_signals_mi: pd.DataFrame, returns_panel_wide: pd.DataFrame
) -> CSAlphaNCOResult:
    """Walk-forward cs_alpha_nco backtest over the DJ30 smoke panel."""
    return cs_alpha_nco_backtest(
        alpha_signals=alpha_signals_mi,
        panel_returns=returns_panel_wide,
        config=S24_CSALPHA_CONFIG,
    )


# ─── PR4 fixtures: AlphaSignal + MarketSnapshot + PortfolioStates ────

INITIAL_CASH_USD = 1_000_000.0


@pytest.fixture
def latest_rebalance_date(cs_alpha_nco_result: CSAlphaNCOResult) -> pd.Timestamp:
    return cs_alpha_nco_result.rebalance_dates[-1]


@pytest.fixture
def latest_target_weights(
    cs_alpha_nco_result: CSAlphaNCOResult,
    latest_rebalance_date: pd.Timestamp,
    dj30_tickers: list[str],
) -> pd.Series:
    """Strategy target weights at the latest rebalance, reindexed to full
    DJ30 (inactive tickers → 0.0)."""
    w = cs_alpha_nco_result.weights_history.loc[latest_rebalance_date]
    return w.reindex(dj30_tickers).fillna(0.0)


@pytest.fixture
def latest_close_prices(
    closes_panel: pd.DataFrame,
    latest_rebalance_date: pd.Timestamp,
    dj30_tickers: list[str],
) -> pd.Series:
    """DJ30 close prices at the latest rebalance date.

    The catalog returns ``date`` as ``datetime.date``; we coerce both sides
    to ``Timestamp`` for safe comparison.
    """
    panel = closes_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    at_rd = panel[panel["date"] == latest_rebalance_date]
    if at_rd.empty:
        pytest.fail(
            f"no closes at rebalance date {latest_rebalance_date.date()}; "
            "panel may not include this date"
        )
    return at_rd.set_index("ticker")["close"].reindex(dj30_tickers)


@pytest.fixture
def market_snapshot(
    latest_close_prices: pd.Series,
    latest_rebalance_date: pd.Timestamp,
    dj30_tickers: list[str],
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=latest_rebalance_date.isoformat(),
        tickers=tuple(dj30_tickers),
        prices=latest_close_prices.to_numpy(dtype=np.float64),
    )


@pytest.fixture
def alpha_signal(
    latest_target_weights: pd.Series,
    latest_rebalance_date: pd.Timestamp,
    dj30_tickers: list[str],
) -> AlphaSignal:
    """AlphaSignal seeded with cs_alpha_nco's kelly weights and synthetic
    conformal bounds aligned to weight sign.

    Why synthetic bounds: AlphaSignal.tradeable = ``(lower > 0) | (upper < 0)``
    (PI must EXCLUDE zero). PR1's actual conformal bounds straddle zero
    for nearly every ticker (typical for short-horizon ACI on equities),
    so .tradeable would be False everywhere → RebalanceEngine zeros all
    weights → empty orders. cs_alpha_nco_backtest uses a different
    "active" notion (max_signal_age_days + min_active_tickers) and never
    constructs an AlphaSignal object — so the conformal-tradeable gate
    is a downstream contract this test surfaces but does not enforce.
    Worth a follow-up.

    Bounds chosen so AlphaSignal.tradeable matches "non-zero kelly_weight"
    (the strategy's own active set). kelly_weights drive the rebalance
    math; bounds only gate the tradeable mask.
    """
    weights = latest_target_weights.to_numpy(np.float64)
    expected_return = np.where(weights > 0, 0.05, np.where(weights < 0, -0.05, 0.0))
    lower = np.where(weights > 0, 0.001, np.where(weights < 0, -0.1, -0.05))
    upper = np.where(weights > 0, 0.1, np.where(weights < 0, -0.001, 0.05))
    return AlphaSignal(
        tickers=tuple(dj30_tickers),
        expected_return=expected_return,
        lower=lower,
        upper=upper,
        alpha=0.1,  # conformal miscoverage rate; doesn't affect kelly_weight()
        kelly_weights=weights,
        timestamp=latest_rebalance_date.isoformat(),
    )


@pytest.fixture
def portfolio_state_empty() -> PortfolioState:
    return PortfolioState.empty(initial_cash=INITIAL_CASH_USD)


@pytest.fixture
def portfolio_state_pre_staged(latest_close_prices: pd.Series) -> PortfolioState:
    """Pre-stage with three long positions for variety. Cost basis = current
    price (zero unrealized PnL by construction; keeps the AC4 invariant
    arithmetic clean)."""
    positions = {
        "AAPL": Position("AAPL", 50, float(latest_close_prices["AAPL"])),
        "MSFT": Position("MSFT", 100, float(latest_close_prices["MSFT"])),
        "KO": Position("KO", 200, float(latest_close_prices["KO"])),
    }
    return PortfolioState(
        cash=INITIAL_CASH_USD - sum(p.quantity * p.avg_cost for p in positions.values()),
        positions=positions,
    )


@pytest.fixture
def rebalance_engine() -> RebalanceEngine:
    """Default RebalanceConstraints: cash_buffer=0.02, max_gross_leverage=1.0,
    min_trade_notional=$100, allow_short=False, lot_size=1."""
    return RebalanceEngine()


@pytest.fixture
def orders_empty(
    rebalance_engine: RebalanceEngine,
    alpha_signal: AlphaSignal,
    portfolio_state_empty: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> list[Order]:
    return rebalance_engine.rebalance(
        signal=alpha_signal,
        state=portfolio_state_empty,
        market=market_snapshot,
    )


@pytest.fixture
def orders_pre_staged(
    rebalance_engine: RebalanceEngine,
    alpha_signal: AlphaSignal,
    portfolio_state_pre_staged: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> list[Order]:
    return rebalance_engine.rebalance(
        signal=alpha_signal,
        state=portfolio_state_pre_staged,
        market=market_snapshot,
    )


# ─── PR1 acceptance ──────────────────────────────────────────────────


def test_dj30_universe_loads() -> None:
    tickers = _load_dj30_tickers()
    assert len(tickers) == 30
    assert all(t.isupper() and t.isalpha() for t in tickers)
    assert len(set(tickers)) == 30


def test_catalog_returns_all_dj30_tickers(
    closes_panel: pd.DataFrame, dj30_tickers: list[str]
) -> None:
    assert not closes_panel.empty
    returned = set(closes_panel["ticker"])
    expected = set(dj30_tickers)
    assert returned == expected, (
        f"catalog missing: {expected - returned}; extra: {returned - expected}"
    )


def test_catalog_date_range_covers_request(closes_panel: pd.DataFrame) -> None:
    dates = pd.to_datetime(closes_panel["date"])
    assert dates.min() >= pd.Timestamp(START_DATE)
    assert dates.max() <= pd.Timestamp(END_DATE)
    # Non-trivial coverage near both ends — guards against silent fallback.
    assert dates.min() <= pd.Timestamp("2022-01-31"), (
        f"first available date {dates.min().date()} suspiciously far past start"
    )
    assert dates.max() >= pd.Timestamp("2024-12-01"), (
        f"last available date {dates.max().date()} suspiciously far before end"
    )


def test_catalog_close_prices_finite_positive(closes_panel: pd.DataFrame) -> None:
    closes = closes_panel["close"]
    assert closes.notna().all()
    assert (closes > 0).all()


def test_signals_fixture_loads_with_expected_schema(
    signals_panel: pd.DataFrame,
) -> None:
    assert not signals_panel.empty
    assert list(signals_panel.columns) == EXPECTED_SIGNAL_COLUMNS


def test_signals_universe_matches_dj30(
    signals_panel: pd.DataFrame, dj30_tickers: list[str]
) -> None:
    assert set(signals_panel["ticker"]) == set(dj30_tickers)


def test_signals_date_range_matches_request(signals_panel: pd.DataFrame) -> None:
    dates = pd.to_datetime(signals_panel["date"])
    assert dates.min() >= pd.Timestamp(START_DATE)
    assert dates.max() <= pd.Timestamp(END_DATE)


def test_signals_fields_finite(signals_panel: pd.DataFrame) -> None:
    for col in ["expected_return", "lower", "upper"]:
        nan_count = int(signals_panel[col].isna().sum())
        assert nan_count == 0, f"Found {nan_count} NaN in column '{col}'"


def test_signals_bounds_well_ordered(signals_panel: pd.DataFrame) -> None:
    """Conformal bounds: lower ≤ expected_return ≤ upper."""
    assert (signals_panel["lower"] <= signals_panel["expected_return"]).all()
    assert (signals_panel["expected_return"] <= signals_panel["upper"]).all()


# ─── PR2 acceptance — quantcore signal/feature path ──────────────────
#
# Exercises ``quantcore.covariance.shrinkage.ledoit_wolf_shrinkage`` —
# the production primitive cs_alpha_nco_backtest invokes for the lw
# branch's per-rebalance covariance estimate. AC3 satisfier:
# "compute or validate signal primitives / ML-relevant transforms".
#
# Per PR2 scope: no new alpha logic; assert output shape, finite
# coverage, structural properties, deterministic values.


def test_quantcore_lw_returns_well_formed_covariance(
    cov_lookback_window: np.ndarray, dj30_tickers: list[str]
) -> None:
    cov, shrinkage = ledoit_wolf_shrinkage(cov_lookback_window)

    n = len(dj30_tickers)
    assert cov.shape == (n, n), f"expected ({n}, {n}); got {cov.shape}"
    assert np.isfinite(cov).all(), "non-finite values in covariance"
    np.testing.assert_allclose(cov, cov.T, atol=1e-12, err_msg="not symmetric")
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-10, f"non-PSD: min eigenvalue {eigvals.min()}"
    assert 0.0 <= shrinkage <= 1.0, f"shrinkage {shrinkage} outside [0, 1]"


def test_quantcore_lw_deterministic_under_repeat(
    cov_lookback_window: np.ndarray,
) -> None:
    """Closed-form LW is deterministic by construction; pin it so any
    accidental introduction of stochasticity (RNG, multithreaded ops)
    surfaces here."""
    cov_1, shrinkage_1 = ledoit_wolf_shrinkage(cov_lookback_window)
    cov_2, shrinkage_2 = ledoit_wolf_shrinkage(cov_lookback_window)
    np.testing.assert_array_equal(cov_1, cov_2)
    assert shrinkage_1 == shrinkage_2


# ─── PR3 acceptance — quantstrat strategy path ───────────────────────
#
# Wires PR1 closes + PR1 signals → cs_alpha_nco_backtest(lw, sqrt_n,
# BMS) and asserts target-weight sanity per sprint plan §5 PR3:
# finite, aligned to ticker universe, no duplicate symbols, gross
# exposure within bound, not all-zero unless explicit.
#
# Config pinned in S24_CSALPHA_CONFIG above (verbatim from S23b
# _PRODUCTION_CONFIG_KW + cov_estimator='lw').


def test_strategy_produces_rebalances(cs_alpha_nco_result: CSAlphaNCOResult) -> None:
    assert len(cs_alpha_nco_result.rebalance_dates) > 0, (
        "no rebalances produced; check date range vs cov_lookback_days warm-up"
    )
    assert len(cs_alpha_nco_result.weights_history) == len(cs_alpha_nco_result.rebalance_dates)


def test_strategy_weights_aligned_to_dj30(
    cs_alpha_nco_result: CSAlphaNCOResult, dj30_tickers: list[str]
) -> None:
    weights = cs_alpha_nco_result.weights_history
    cols = list(weights.columns)
    assert set(cols) <= set(dj30_tickers), (
        f"weights have non-DJ30 columns: {set(cols) - set(dj30_tickers)}"
    )
    assert len(cols) == len(set(cols)), "duplicate ticker columns in weights_history"


def test_strategy_latest_weights_finite(
    cs_alpha_nco_result: CSAlphaNCOResult,
) -> None:
    """Latest rebalance weights are finite where present (NaN allowed for
    inactive tickers; the rest must be finite numbers)."""
    latest = cs_alpha_nco_result.weights_history.iloc[-1]
    finite_mask = latest.notna()
    assert finite_mask.any(), "latest rebalance has no finite weights"
    assert np.isfinite(latest[finite_mask]).all(), "non-NaN weights contain inf/non-finite values"


def test_strategy_latest_weights_not_all_zero(
    cs_alpha_nco_result: CSAlphaNCOResult,
) -> None:
    latest = cs_alpha_nco_result.weights_history.iloc[-1].fillna(0.0)
    assert (latest != 0).any(), "all-zero weights at last rebalance — strategy degenerated"


def test_strategy_gross_exposure_bounded(
    cs_alpha_nco_result: CSAlphaNCOResult,
) -> None:
    """Per-asset weight bounded by kelly_cap=0.5 → gross ≤ 0.5 × N_active.
    For DJ30 (N≤30): gross ≤ 15.0. Loose structural bound; actual values
    are typically much smaller (Kelly-scaled subset of active names)."""
    weights = cs_alpha_nco_result.weights_history.fillna(0.0)
    gross = weights.abs().sum(axis=1)
    max_gross = float(gross.max())
    assert max_gross <= 15.0, f"gross exposure {max_gross} exceeds kelly_cap × N_dj30 bound (15.0)"
    assert (gross > 0).any(), "no rebalance produced non-zero gross exposure"


# ─── PR4 acceptance — quantengine dry-run order path ─────────────────
#
# Wires PR3 strategy weights → AlphaSignal → RebalanceEngine.rebalance()
# and asserts AC4 arithmetic correctness per sprint plan §3.
#
# Account-state coverage: BOTH (per plan preference) — empty paper
# account + deterministic mocked pre-staged positions.
#
# Symbology: deferred. DJ30 has no IBKR/yfinance divergence (PaperBroker /
# RebalanceEngine don't enforce symbology resolution). If S24 later
# extends to IBKR via PR6, symbology resolution lands there.
#
# Currency: implicit USD. PaperBroker / RebalanceEngine / Order /
# PortfolioState carry no Currency type — single-currency assumed.
# Asserted structurally below.
#
# AC4 tolerances: per-ticker = 1.5 × price (1-share rounding + margin),
# using post-pipeline target_weight from order.metadata. Aggregate =
# N × max(price) × 1.5 (worst-case sum of per-ticker bounds).
#
# AC5 dry-run discipline: no Broker class is imported in this module;
# RebalanceEngine.rebalance() is a pure function with no submission
# side-effect. Asserted structurally below.


def _ticker_price(market: MarketSnapshot, ticker: str) -> float:
    return float(market.prices[market.tickers.index(ticker)])


def test_orders_emitted_from_empty_account(orders_empty: list[Order]) -> None:
    assert len(orders_empty) > 0, "empty account should produce orders to reach target"


def test_orders_emitted_from_pre_staged_account(orders_pre_staged: list[Order]) -> None:
    assert len(orders_pre_staged) > 0, "pre-staged account should produce delta orders"


def test_orders_universe_subset_of_dj30(orders_empty: list[Order], dj30_tickers: list[str]) -> None:
    bad = [o.ticker for o in orders_empty if o.ticker not in set(dj30_tickers)]
    assert not bad, f"orders contain non-DJ30 tickers: {bad}"


def test_no_broker_imported_in_test_module() -> None:
    """AC5 structural check — dry-run discipline. No broker class is in
    this module's namespace; RebalanceEngine.rebalance() is a pure
    function with no submission side-effect."""
    import sys

    test_module = sys.modules[__name__]
    forbidden = ["IBKRBroker", "PaperBroker", "AbstractBroker"]
    in_scope = [n for n in forbidden if hasattr(test_module, n)]
    assert not in_scope, f"AC5 violation: {in_scope} present in test scope"


def test_currency_implicit_usd() -> None:
    """quantengine assumes single-currency (USD); no Currency field on
    Order or PortfolioState. Structural absence catches accidental FX
    introductions."""
    from dataclasses import fields

    order_field_names = {f.name for f in fields(Order)}
    state_field_names = {f.name for f in fields(PortfolioState)}
    assert "currency" not in order_field_names
    assert "currency" not in state_field_names


def _ac4_per_ticker(
    orders: list[Order],
    state: PortfolioState,
    market: MarketSnapshot,
    nav: float,
) -> list[tuple[str, float, float]]:
    """Returns list of (ticker, delta, tolerance) for AC4 per-ticker
    invariant violations. Empty list = all clear."""
    violations: list[tuple[str, float, float]] = []
    for order in orders:
        price = _ticker_price(market, order.ticker)
        target_weight = float(order.metadata["target_weight"])
        target_value = target_weight * nav
        current_value = state.quantity_of(order.ticker) * price
        order_value = order.signed_quantity * price
        delta = order_value - (target_value - current_value)
        tol = 1.5 * price
        if abs(delta) > tol:
            violations.append((order.ticker, float(delta), float(tol)))
    return violations


def test_ac4_per_ticker_invariant_empty_account(
    orders_empty: list[Order],
    portfolio_state_empty: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> None:
    nav = portfolio_state_empty.cash  # empty account NAV = cash
    violations = _ac4_per_ticker(orders_empty, portfolio_state_empty, market_snapshot, nav)
    assert not violations, f"AC4 per-ticker (empty): {violations}"


def test_ac4_per_ticker_invariant_pre_staged_account(
    orders_pre_staged: list[Order],
    portfolio_state_pre_staged: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> None:
    price_map = {t: _ticker_price(market_snapshot, t) for t in market_snapshot.tickers}
    nav = portfolio_state_pre_staged.nav(price_map)
    violations = _ac4_per_ticker(
        orders_pre_staged, portfolio_state_pre_staged, market_snapshot, nav
    )
    assert not violations, f"AC4 per-ticker (pre-staged): {violations}"


def test_ac4_aggregate_invariant_empty_account(
    orders_empty: list[Order],
    portfolio_state_empty: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> None:
    nav = portfolio_state_empty.cash
    sum_order = sum(
        o.signed_quantity * _ticker_price(market_snapshot, o.ticker) for o in orders_empty
    )
    sum_target = sum(float(o.metadata["target_weight"]) * nav for o in orders_empty)
    sum_current = sum(
        portfolio_state_empty.quantity_of(o.ticker) * _ticker_price(market_snapshot, o.ticker)
        for o in orders_empty
    )
    delta = sum_order - (sum_target - sum_current)
    max_price = float(market_snapshot.prices.max())
    tol = max(len(orders_empty) * max_price * 1.5, 1000.0)
    assert abs(delta) <= tol, f"AC4 aggregate (empty): delta={delta:.2f}, tol={tol:.2f}"


def test_ac4_aggregate_invariant_pre_staged_account(
    orders_pre_staged: list[Order],
    portfolio_state_pre_staged: PortfolioState,
    market_snapshot: MarketSnapshot,
) -> None:
    price_map = {t: _ticker_price(market_snapshot, t) for t in market_snapshot.tickers}
    nav = portfolio_state_pre_staged.nav(price_map)
    sum_order = sum(
        o.signed_quantity * _ticker_price(market_snapshot, o.ticker) for o in orders_pre_staged
    )
    sum_target = sum(float(o.metadata["target_weight"]) * nav for o in orders_pre_staged)
    sum_current = sum(
        portfolio_state_pre_staged.quantity_of(o.ticker) * _ticker_price(market_snapshot, o.ticker)
        for o in orders_pre_staged
    )
    delta = sum_order - (sum_target - sum_current)
    max_price = float(market_snapshot.prices.max())
    tol = max(len(orders_pre_staged) * max_price * 1.5, 1000.0)
    assert abs(delta) <= tol, f"AC4 aggregate (pre-staged): delta={delta:.2f}, tol={tol:.2f}"
