"""s19b b12: smoke-test suite for `quantcore.bars.bars`.

Closes the s19a known-unknowns finding b12: `bars/bars.py` (589 LOC,
12 public symbols implementing AFML Ch. 2 information-driven bars)
had no dedicated test file. Source unchanged (locked under s19b
`forbidden_actions`).

# Why test if S19 doesn't currently exercise this?

A read-pass over the codebase confirmed S19's harness consumes pre-aggregated daily
OHLCV via `harness/data.py:load_ohlcv_frame` rather than info-driven
bars from `quantcore.bars`. The s19a-era enumeration's claim "S19's
input bars come from here" was extrapolation, not a verified
import-chain fact.

The b12 rationale that survives import-chain re-grounding: foundational
modules with zero dedicated tests are a liability regardless of whether
the *next* sprint exercises them. The repo has a longer horizon than
S19; a future RMT/NCO migration to dollar bars or a microstructure
feature pivot would activate this surface, and undetected
bar-construction bugs would propagate into the covariance matrix
silently. Forward-looking insurance, not present-day load-bearing.

# Test scope (6 tests on the public API)

  * `test_tick_bars_threshold_one_per_tick` — saturation at minimum
    threshold (1 tick per bar).
  * `test_volume_bars_threshold_larger_than_total_returns_empty` —
    saturation at maximum threshold (no bars closed; empty output
    preserves column shape).
  * `test_dollar_bars_zero_volume_rows_merged_into_neighboring_bars` —
    rows with volume=0 don't raise and don't create degenerate bars.
  * `test_tick_bars_single_row_input_yields_single_bar_OHLC_equal` —
    single-row input → 1-row output with O=H=L=C.
  * `test_aggregate_to_ohlcv_preserves_within_bar_OHLC_invariant` —
    the algebraic invariant `low ≤ open ≤ high` AND
    `low ≤ close ≤ high` for every output bar across a
    randomized 500-row fixture. Cheapest possible test that catches
    the hardest possible bug class (column-swap, sub-window
    computation error). Equivalent to c1 Phase-2's
    `test_quantile_score_negative_inside_interval` algebraic-invariant
    pin.
  * `test_dollar_bars_monotonic_prices_threshold_at_each_crossing` —
    monotonic prices, threshold sized so each row triggers a close.
    Pins threshold-crossing logic without entangling tick-direction
    reversals.

# Silent-fallback audit (per s19b plan Implementation notes)

Phase-equivalent read-pass scanned bars.py for `+ 1e-`, `eps`, `EPS`,
`except: pass`, `except Exception:`, `or 0` patterns. Found 1 match:
line 26 `min_abs_exp_imbalance: float = 1e-12` on the
`ImbalanceConfig` dataclass — a public configurable field, caller
can override. Class-(a) regularization at the API surface, same shape
as the `epsilon: float = 1e-8` parameter in `normalized_residual_score`
(c1 Phase-1 finding). **No new F-RP-006 finding logged.** No silent
exception-swallowing, no `or N` defaults, no bare divide-by-zero.

# Pre-emission verification

  * `tick_bars(df_5_rows, threshold=1)` → shape (5, 8); columns
    `[open, high, low, close, volume, vwap, tick_count, dollar_volume]`;
    `index.name='timestamp'`; `attrs={'bar_type': 'tick_bars', 'threshold': 1.0}`.
  * `volume_bars(df, threshold=1000, partial=False|True)` → shape
    (0, 8) — partial-last-bar is NOT included when zero closed bars
    exist (surprising; pin the actual behavior).
  * `tick_bars(df_1_row, threshold=1)` → shape (1, 8) with
    O=H=L=C=price.
  * `dollar_bars` on a 500-row randomized fixture: 0 OHLC-invariant
    violations across 456 output bars. Algebraic invariant holds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.bars.bars import (
    dollar_bars,
    tick_bars,
    volume_bars,
)

# Canonical 8-column shape for every bars.py output.
_BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "tick_count",
    "dollar_volume",
]


def _toy_df(prices: list[float], volumes: list[float]) -> pd.DataFrame:
    """Helper: ticks fixture with deterministic timestamps."""
    n = len(prices)
    return pd.DataFrame(
        {
            "price": np.asarray(prices, dtype=np.float64),
            "volume": np.asarray(volumes, dtype=np.float64),
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="s"),
        }
    )


# =============================================================================
# 1. Tick bars — saturation at minimum threshold
# =============================================================================


def test_tick_bars_threshold_one_per_tick() -> None:
    """``threshold=1`` makes every input row close its own bar
    (minimum-threshold saturation). Pins the 8-column shape, the
    column names + order, the index name, and the attrs metadata.
    """
    df = _toy_df([100.0, 100.7, 101.3, 102.0, 102.8], [10, 10, 10, 10, 10])
    bars = tick_bars(df, threshold=1)

    assert bars.shape == (5, 8)
    assert list(bars.columns) == _BAR_COLUMNS
    assert bars.index.name == "timestamp"
    # attrs metadata pinned (callers downstream of S19 may inspect).
    assert bars.attrs["bar_type"] == "tick_bars"
    assert float(bars.attrs["threshold"]) == 1.0
    # Each bar covers exactly one tick → tick_count = 1 everywhere.
    assert (bars["tick_count"] == 1).all()


# =============================================================================
# 2. Volume bars — saturation at maximum threshold
# =============================================================================


def test_volume_bars_threshold_larger_than_total_returns_empty() -> None:
    """``threshold > sum(volume)`` produces zero closed bars.

    s83 F2b: the pre-s83 version of this pin documented that the partial
    bar was NOT included when zero closed bars exist — i.e. it pinned the
    silent drop of an explicitly requested partial row. Corrected
    contract: ``include_partial_last_bar=False`` → empty (0, 8) with the
    canonical column shape; ``True`` with data present → the whole tape IS
    the partial bar (one row)."""
    df = _toy_df([100.0, 100.7, 101.3, 102.0, 102.8], [10, 10, 10, 10, 10])
    # Total volume = 50; threshold=1000 cannot be crossed.
    bars = volume_bars(df, threshold=1000.0, include_partial_last_bar=False)
    assert bars.shape == (0, 8)
    assert list(bars.columns) == _BAR_COLUMNS

    partial = volume_bars(df, threshold=1000.0, include_partial_last_bar=True)
    assert len(partial) == 1, "requested partial row must not be silently dropped"
    assert partial["close"].iloc[0] == 102.8
    assert partial["volume"].iloc[0] == 50.0
    assert partial["tick_count"].iloc[0] == 5


# =============================================================================
# 3. Dollar bars — zero-volume rows merge into neighboring bars
# =============================================================================


def test_dollar_bars_zero_volume_rows_merged_into_neighboring_bars() -> None:
    """Rows with ``volume=0`` contribute zero dollars to the threshold
    accumulator. They neither raise (volume=0 ≥ 0 passes
    ``_extract_arrays``'s non-negative check) nor produce their own bar.
    Verified pre-emission: a 5-row fixture with ``volume=[10,0,10,0,10]``
    and prices [100, 100.5, 101, 101.5, 102] at threshold=1000 produces
    3 bars (each crossing the 1000-dollar boundary)."""
    df = _toy_df(
        [100.0, 100.5, 101.0, 101.5, 102.0],
        [10, 0, 10, 0, 10],
    )
    bars = dollar_bars(df, threshold=1000.0, include_partial_last_bar=True)

    assert bars.shape == (3, 8)
    # Volumes are sums of non-zero contributions per bar.
    assert (bars["volume"] >= 0).all()
    # OHLC must be finite for every bar — no NaN propagation from
    # zero-volume rows. Use `isna().sum() == 0` (returns plain int)
    # rather than `notna().all()` (returns Series | bool — basedpyright
    # objects to the latter as a conditional operand).
    for col in ("open", "high", "low", "close"):
        assert bars[col].isna().sum() == 0, f"{col} has NaN entries"


# =============================================================================
# 4. Single-row input — degenerate-but-valid output
# =============================================================================


def test_tick_bars_single_row_input_yields_single_bar_OHLC_equal() -> None:
    """Single-row input → single-row output where O=H=L=C=price. The
    degenerate case where one tick is one bar; pinning the OHLC-equal
    invariant prevents a future regression that would compute high
    or low against a non-existent neighboring tick."""
    df = _toy_df([100.0], [10])
    bars = tick_bars(df, threshold=1)

    assert bars.shape == (1, 8)
    row = bars.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 100.0
    assert row["low"] == 100.0
    assert row["close"] == 100.0
    assert row["volume"] == 10.0
    assert row["tick_count"] == 1


# =============================================================================
# 5. OHLC algebraic invariant (cheapest test, highest leverage)
# =============================================================================


def test_aggregate_to_ohlcv_preserves_within_bar_OHLC_invariant() -> None:
    """For every output bar across a randomized 500-row fixture, the
    OHLC algebraic invariant holds:

        low ≤ open ≤ high
        low ≤ close ≤ high
        low ≤ high

    Equivalent to c1 Phase-2's
    ``test_quantile_score_negative_inside_interval`` — algebraic
    invariant pin, no fixture-shape-dependence, fails immediately
    on the bug class it's designed to catch (column-swap, sub-window
    computation error, dimension mixup). Cheapest test in the b12
    suite; highest leverage if a future regression touches the
    aggregate path.
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    prices: NDArray[np.floating[Any]] = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    volumes: NDArray[np.floating[Any]] = rng.integers(1, 100, n).astype(np.float64)
    df = pd.DataFrame({"price": prices, "volume": volumes})

    bars = dollar_bars(df, threshold=1000.0, include_partial_last_bar=False)
    assert bars.shape[0] > 0, "fixture must produce at least one bar"

    # Per-bar invariants. Each side of the invariant gets its own
    # assertion + diagnostic message so a future regression points
    # at exactly which side tripped (open-out-of-range vs.
    # close-out-of-range vs. low > high).
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low_ = bars["low"].to_numpy()
    c = bars["close"].to_numpy()

    assert (low_ <= h).all(), (
        f"OHLC invariant violation: low > high in {int((low_ > h).sum())} bars"
    )
    assert (low_ <= o).all() and (o <= h).all(), (
        f"OHLC invariant violation: open out of [low, high] in "
        f"{int(((o < low_) | (o > h)).sum())} bars"
    )
    assert (low_ <= c).all() and (c <= h).all(), (
        f"OHLC invariant violation: close out of [low, high] in "
        f"{int(((c < low_) | (c > h)).sum())} bars"
    )


# =============================================================================
# 6. Threshold-crossing on monotonic prices (no tick-direction reversal)
# =============================================================================


def test_dollar_bars_monotonic_prices_threshold_at_each_crossing() -> None:
    """Monotonic prices with threshold sized so each row triggers a
    close. Pre-emission verification: prices [100, 101, 102, 103, 104]
    at volume=10 each — dollar values are [1000, 1010, 1020, 1030,
    1040] (≥1010 per row except the first) — at threshold=1010
    produces 4 bars (first bar covers rows 0+1 because the first row's
    1000 < 1010 alone but 1000+1010=2010 ≥ 1010 closes at row 1;
    subsequent rows each individually cross the threshold).

    Pins the threshold-crossing logic without entangling tick-direction
    reversals (which would mostly test imbalance_bars logic, out of
    b12 scope).
    """
    df = _toy_df(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [10, 10, 10, 10, 10],
    )
    bars = dollar_bars(df, threshold=1010.0, include_partial_last_bar=False)

    assert bars.shape == (4, 8)
    # Each bar's dollar_volume crosses or equals the threshold.
    assert (bars["dollar_volume"] >= 1000.0).all()
    # Monotonic prices ⇒ monotonic bar closes (close prices strictly
    # non-decreasing across bars).
    closes = bars["close"].to_numpy()
    assert np.all(np.diff(closes) >= 0), (
        f"monotonic-input bars must produce non-decreasing close prices; got closes={closes}"
    )
