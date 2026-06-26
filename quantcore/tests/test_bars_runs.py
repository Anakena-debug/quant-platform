"""Test suite for `quantcore.bars.runs_bars` (canonical AFML run bars,
§2.3.2.2 — TRB / VRB / DRB).

Run bars differ from imbalance bars by tracking buy-side and sell-side
weighted totals INDEPENDENTLY rather than as a single signed sum:

    theta_buy  = sum_{t | b_t=+1} w_t
    theta_sell = sum_{t | b_t=-1} w_t
    theta      = max(theta_buy, theta_sell)
    close when theta >= E[T] * max{P_buy * E[w|buy], P_sell * E[w|sell]}

The two-sided accumulation is what makes run bars close on
alternating buy/sell tapes where imbalance bars (single signed
sum) get stuck near zero — see test #3 (tug-of-war).

# Test scope (8 tests)

  * `test_buy_dominated_tape_closes_on_buy_run` — 95%-buy tape via
    explicit side_col; every closed bar has theta_buy > theta_sell.
  * `test_sell_dominated_tape_closes_on_sell_run` — mirror.
  * `test_tug_of_war_alternating_closes_run_bars_not_imbalance_bars` —
    THE behavioral fingerprint: alternating ±V tape closes run bars
    (one-sided totals grow monotonically) but produces ZERO imbalance
    bars (net signed sum oscillates below threshold).
  * `test_explicit_side_col_matches_tick_rule_on_monotonic_tape` —
    on a strictly increasing price tape the tick rule yields all +1;
    passing an explicit all-+1 side_col must produce byte-equal output.
  * `test_tick_runs_bars_deterministic_close_pattern_all_buy` —
    pinned config (E[T]=5, all weights=1, all sides=+1) → deterministic
    close indices [4, 9, 14, 19, 24] with theta_buy=5, theta_sell=0,
    threshold=5, ticks_in_bar=5. Catches off-by-one in the >=
    comparison and accumulator-reset timing.
  * `test_runs_bars_no_sells_in_warmup_uses_fallback_weight` —
    warmup window with zero sell ticks must not crash and must use
    sign-blind mean weight as E[w|sell] init (EWMA corrects later).
  * `test_all_runs_bars_wrappers_return_bars_on_toy_tape` — smoke
    test that tick / volume / dollar wrappers all wire through and
    return non-empty outputs with the expected diagnostic columns.
  * `test_runs_bars_input_validation` — all error contracts:
    missing/invalid columns, NaN/negative inputs, invalid side values,
    include_partial_last_bar NotImplementedError, span < 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from quantcore.bars.bars import (
    ImbalanceConfig,
    RunsConfig,
    dollar_runs_bars,
    tick_runs_bars,
    volume_imbalance_bars,
    volume_runs_bars,
)


def _toy_df(
    prices: NDArray[np.floating[Any]],
    volumes: NDArray[np.floating[Any]],
    sides: NDArray[np.integer[Any]] | None = None,
) -> pd.DataFrame:
    """Standard fixture shape: float64 prices/volumes, deterministic
    second-frequency timestamps, optional int8 side column."""
    n = len(prices)
    data: dict[str, Any] = {
        "price": prices.astype(np.float64),
        "volume": volumes.astype(np.float64),
        "timestamp": pd.date_range("2026-01-02", periods=n, freq="s"),
    }
    if sides is not None:
        data["side"] = sides.astype(np.int8)
    return pd.DataFrame(data)


# =============================================================================
# 1. Buy-dominated tape closes on buy run
# =============================================================================


def test_buy_dominated_tape_closes_on_buy_run() -> None:
    """95%-buy tape via explicit side_col: every closed bar must have
    `theta_buy > theta_sell`, since the only side accumulating mass
    fast enough to cross the threshold is the buy side.

    Pre-emission walk: with P_buy ≈ 0.95, E[w|buy] ≈ 50, the threshold
    is dominated by the buy term (P_buy * E[w|buy] = 47.5 vs. P_sell *
    E[w|sell] = 2.5). theta_buy grows ~47.5/tick on average while
    theta_sell grows ~2.5/tick — theta_sell never overtakes theta_buy
    on a 5000-tick fixture.
    """
    rng = np.random.default_rng(seed=7)
    n = 5000
    sides = np.where(rng.uniform(size=n) < 0.95, 1, -1).astype(np.int8)
    prices = 100.0 + np.cumsum(sides.astype(np.float64) * 0.01)
    volumes = np.full(n, 50.0)
    df = _toy_df(prices, volumes, sides)

    bars = volume_runs_bars(
        df,
        config=RunsConfig(exp_num_ticks_init=500.0, warmup_ticks=200),
        side_col="side",
    )

    assert len(bars) > 0, "must produce bars on a 95%-buy tape"
    buy_dominated = (bars["theta_buy"] > bars["theta_sell"]).to_numpy()
    assert buy_dominated.all(), (
        f"every bar on a 95%-buy tape should close on a buy run; "
        f"got {int((~buy_dominated).sum())} sell-dominated closes"
    )


# =============================================================================
# 2. Sell-dominated tape closes on sell run (mirror of #1)
# =============================================================================


def test_sell_dominated_tape_closes_on_sell_run() -> None:
    """Mirror of the buy-dominated test: 95%-sell tape, every closed
    bar must have `theta_sell > theta_buy`."""
    rng = np.random.default_rng(seed=8)
    n = 5000
    sides = np.where(rng.uniform(size=n) < 0.95, -1, 1).astype(np.int8)
    prices = 100.0 + np.cumsum(sides.astype(np.float64) * 0.01)
    volumes = np.full(n, 50.0)
    df = _toy_df(prices, volumes, sides)

    bars = volume_runs_bars(
        df,
        config=RunsConfig(exp_num_ticks_init=500.0, warmup_ticks=200),
        side_col="side",
    )

    assert len(bars) > 0
    sell_dominated = (bars["theta_sell"] > bars["theta_buy"]).to_numpy()
    assert sell_dominated.all(), (
        f"every bar on a 95%-sell tape should close on a sell run; "
        f"got {int((~sell_dominated).sum())} buy-dominated closes"
    )


# =============================================================================
# 3. Tug-of-war: runs bars close, imbalance bars do not
# =============================================================================


def test_tug_of_war_alternating_closes_run_bars_not_imbalance_bars() -> None:
    """THE behavioral fingerprint that distinguishes run bars from
    imbalance bars. Alternating buy/sell ticks at constant volume:
    one-sided run totals grow monotonically (theta_buy and theta_sell
    each receive every other tick) so run bars close at predictable
    intervals; the signed-sum imbalance theta oscillates within a
    bounded range and cannot cross its threshold.

    Pre-emission walk:
      - Prices alternate 100.0 / 100.1 → tick rule:
        b = [+1, +1, -1, +1, -1, +1, -1, ...]
        (b[0]=+1 from initializer; from b[1] onwards strict alternation)
      - Volumes = 100 constant. signed_increments = b * 100.
      - Warmup over first 100 ticks: 51 buys, 49 sells.
        mean(signed_increments[:100]) = (51 - 49) * 100 / 100 = +2.
        IMBALANCE threshold = exp_T * |exp_x| = 200 * 2 = 400.
        IMBALANCE theta oscillates in {100, 200} forever → NEVER
        reaches 400 → 0 bars.
      - RUNS: P_buy ≈ 0.51, E[w|buy] = E[w|sell] = 100.
        threshold = 200 * max(0.51*100, 0.49*100) = 10_200.
        theta_buy grows by 100 every odd tick (~50/tick avg);
        first close around tick i=201; subsequent every ~200 ticks
        → roughly 10 runs bars across 2000 ticks.
    """
    n = 2000
    prices_alt = np.tile([100.0, 100.1], n // 2)
    volumes = np.full(n, 100.0)
    df = _toy_df(prices_alt, volumes)

    runs = volume_runs_bars(
        df,
        config=RunsConfig(exp_num_ticks_init=200.0, warmup_ticks=100),
    )
    imb = volume_imbalance_bars(
        df,
        config=ImbalanceConfig(exp_num_ticks_init=200.0, warmup_ticks=100),
    )

    assert len(runs) >= 5, (
        f"run bars must close repeatedly on a tape with one-sided runs "
        f"accumulating; got {len(runs)} bars"
    )
    assert len(imb) == 0, (
        f"imbalance bars must NOT close on net-cancelling tug-of-war "
        f"(theta_net bounded < threshold); got {len(imb)} bars"
    )


# =============================================================================
# 4. Explicit side_col equivalent to tick rule on monotonic tape
# =============================================================================


def test_explicit_side_col_matches_tick_rule_on_monotonic_tape() -> None:
    """On a strictly increasing price tape, the tick rule yields b_t=+1
    for every t (b[0]=+1 from init, and every subsequent price is
    higher than the previous). An explicit all-+1 side_col must produce
    byte-equal output — pinning the equivalence of the two side-source
    paths through `_resolve_sides`."""
    n = 500
    prices = (100.0 + np.arange(n, dtype=np.float64) * 0.1).astype(np.float64)
    volumes = np.full(n, 50.0)
    sides = np.ones(n, dtype=np.int8)

    df_no_side = _toy_df(prices, volumes)
    df_with_side = _toy_df(prices, volumes, sides)

    cfg = RunsConfig(exp_num_ticks_init=100.0, warmup_ticks=50)

    bars_tick_rule = volume_runs_bars(df_no_side, config=cfg)
    bars_explicit = volume_runs_bars(df_with_side, config=cfg, side_col="side")

    pd.testing.assert_frame_equal(bars_tick_rule, bars_explicit, check_exact=True)


# =============================================================================
# 5. Deterministic close pattern: all buys, weights=1
# =============================================================================


def test_tick_runs_bars_deterministic_close_pattern_all_buy() -> None:
    """Pinned config with all sides=+1 and weights=1 (tick runs) gives
    a fully deterministic close pattern. With E[T]=5, P_buy=1,
    E[w|buy]=1 (all explicit, slow EWMAs so realized matches init),
    threshold = 5 * (1 * 1) = 5; theta_buy grows by 1/tick; bars must
    close at indices [4, 9, 14, 19, 24] with:
        theta_buy = 5, theta_sell = 0, threshold = 5, ticks_in_bar = 5.

    This pins three regression-prone properties simultaneously:
      (a) the >= comparison in the close condition (off-by-one would
          shift the close to index 5 instead of 4),
      (b) per-bar accumulator reset (a leak would make subsequent
          theta_buy exceed 5),
      (c) one-sided E[w|sell] preservation when n_sell=0 (a degenerate
          NaN/0 here would make threshold drift across bars).
    """
    n = 25
    prices = (100.0 + np.arange(n, dtype=np.float64) * 0.001).astype(np.float64)
    volumes = np.full(n, 50.0)
    sides = np.ones(n, dtype=np.int8)
    df = _toy_df(prices, volumes, sides)

    cfg = RunsConfig(
        exp_num_ticks_init=5.0,
        exp_prob_buy_init=1.0,
        exp_w_buy_init=1.0,
        exp_w_sell_init=1.0,
        ewma_span_ticks=1_000,
        ewma_span_weights=1_000,
        ewma_span_prob=1_000,
        exp_num_ticks_min=1.0,
        exp_num_ticks_max=100_000.0,
        warmup_ticks=1,
    )

    bars = tick_runs_bars(df, config=cfg, side_col="side")

    assert len(bars) == 5
    assert (bars["ticks_in_bar"].to_numpy() == 5).all()
    assert (bars["theta_buy"].to_numpy() == 5.0).all()
    assert (bars["theta_sell"].to_numpy() == 0.0).all()
    assert (bars["theta_buy"].to_numpy() > bars["theta_sell"].to_numpy()).all()
    # Threshold stays at 5 across bars (slow EWMA + realized==init = no drift).
    np.testing.assert_allclose(bars["threshold"].to_numpy(), 5.0, atol=1e-9)


# =============================================================================
# 6. Warmup with zero sell ticks uses fallback weight
# =============================================================================


def test_runs_bars_no_sells_in_warmup_uses_fallback_weight() -> None:
    """When the warmup window contains zero sell ticks, exp_w_sell_init
    must fall back to the sign-blind warmup-mean weight (rather than
    NaN, 0, or a degenerate value carried into the numba kernel).
    Subsequent mixed-side activity must produce well-defined run bars.

    Pre-emission walk: first 100 ticks all +1 (n_sell_warmup=0) →
    fallback path sets w_sell0 = mean(weights[:100]) = 100.0. The
    kernel receives four finite floats; mixed activity later closes
    valid bars."""
    rng = np.random.default_rng(seed=11)
    warmup_n = 100
    n = 1000
    sides_warmup = np.ones(warmup_n, dtype=np.int8)
    sides_rest = np.where(rng.uniform(size=n - warmup_n) < 0.5, 1, -1).astype(np.int8)
    sides = np.concatenate([sides_warmup, sides_rest])

    prices = 100.0 + np.cumsum(sides.astype(np.float64) * 0.01)
    volumes = np.full(n, 100.0)
    df = _toy_df(prices, volumes, sides)

    bars = volume_runs_bars(
        df,
        config=RunsConfig(exp_num_ticks_init=50.0, warmup_ticks=warmup_n),
        side_col="side",
    )

    assert len(bars) > 0, "must produce bars on mixed-activity tape"
    assert bars["threshold"].isna().sum() == 0, "thresholds must not be NaN"
    assert (bars["threshold"].to_numpy() > 0.0).all(), "thresholds must be positive"
    # Same for the realized one-sided totals
    assert bars["theta_buy"].isna().sum() == 0
    assert bars["theta_sell"].isna().sum() == 0


# =============================================================================
# 7. All three wrappers smoke test
# =============================================================================


def test_all_runs_bars_wrappers_return_bars_on_toy_tape() -> None:
    """tick / volume / dollar wrappers all return non-empty bars on
    the same toy tape. Confirms the three weight schemes wire through
    `_standard_increments` cleanly, the public surface works end-to-end,
    and the expected attrs/diagnostic-columns shape holds for each.
    """
    rng = np.random.default_rng(seed=13)
    n = 2000
    sides = np.where(rng.uniform(size=n) < 0.6, 1, -1).astype(np.int8)
    prices = 100.0 + np.cumsum(sides.astype(np.float64) * 0.005)
    volumes = rng.integers(1, 50, size=n).astype(np.float64)
    df = _toy_df(prices, volumes, sides)
    cfg = RunsConfig(exp_num_ticks_init=50.0, warmup_ticks=200)

    for fn, label in (
        (tick_runs_bars, "tick"),
        (volume_runs_bars, "volume"),
        (dollar_runs_bars, "dollar"),
    ):
        bars = fn(df, config=cfg, side_col="side")
        assert len(bars) > 0, f"{label}_runs_bars produced no bars"
        assert (bars["theta_buy"].to_numpy() >= 0.0).all(), f"{label}: theta_buy must be >= 0"
        assert (bars["theta_sell"].to_numpy() >= 0.0).all(), f"{label}: theta_sell must be >= 0"
        assert (bars["threshold"].to_numpy() > 0.0).all(), f"{label}: threshold must be > 0"
        assert (bars["ticks_in_bar"].to_numpy() >= 1).all(), f"{label}: ticks_in_bar must be >= 1"
        assert bars.attrs["bar_type"] == f"{label}_runs_bars"


# =============================================================================
# 8. Input validation — all error contracts
# =============================================================================


def test_runs_bars_input_validation() -> None:
    """Comprehensive boundary contracts. Each branch is hit
    individually so a regression in any validator surfaces with a
    precise failure rather than a generic InvalidInput."""
    base = _toy_df(
        np.array([100.0, 100.1, 100.2]),
        np.array([10.0, 10.0, 10.0]),
    )

    # NaN price → ValueError (from _extract_arrays)
    bad = base.copy()
    bad.loc[1, "price"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        volume_runs_bars(bad)

    # NaN volume → ValueError
    bad = base.copy()
    bad.loc[1, "volume"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        volume_runs_bars(bad)

    # Negative volume → ValueError
    bad = base.copy()
    bad.loc[1, "volume"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        volume_runs_bars(bad)

    # Empty df → ValueError
    with pytest.raises(ValueError, match="empty"):
        volume_runs_bars(pd.DataFrame({"price": [], "volume": []}))

    # Missing price column → KeyError
    with pytest.raises(KeyError):
        volume_runs_bars(base.drop(columns=["price"]))

    # Missing volume column → KeyError
    with pytest.raises(KeyError):
        volume_runs_bars(base.drop(columns=["volume"]))

    # Missing side_col when requested → KeyError
    with pytest.raises(KeyError, match="side"):
        volume_runs_bars(base, side_col="nonexistent")

    # Invalid side values: 0 → ValueError
    bad = base.assign(side=[1, 0, -1])
    with pytest.raises(ValueError, match="must contain only"):
        volume_runs_bars(bad, side_col="side")

    # Invalid side values: 2 → ValueError
    bad = base.assign(side=[1, 2, -1])
    with pytest.raises(ValueError, match="must contain only"):
        volume_runs_bars(bad, side_col="side")

    # NaN side → ValueError
    bad = base.assign(side=[1.0, np.nan, -1.0])
    with pytest.raises(ValueError, match="NaN"):
        volume_runs_bars(bad, side_col="side")

    # Non-numeric side dtype → TypeError
    bad = base.assign(side=["B", "A", "B"])
    with pytest.raises(TypeError, match="numeric"):
        volume_runs_bars(bad, side_col="side")

    # include_partial_last_bar=True → NotImplementedError
    with pytest.raises(NotImplementedError, match="include_partial_last_bar"):
        volume_runs_bars(base, include_partial_last_bar=True)

    # ewma_span < 1 → ValueError
    with pytest.raises(ValueError, match="ewma_span"):
        volume_runs_bars(base, config=RunsConfig(ewma_span_ticks=0))
    with pytest.raises(ValueError, match="ewma_span"):
        volume_runs_bars(base, config=RunsConfig(ewma_span_prob=0))
    with pytest.raises(ValueError, match="ewma_span"):
        volume_runs_bars(base, config=RunsConfig(ewma_span_weights=0))
