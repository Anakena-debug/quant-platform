"""S41 — live-inference flow-ratio parity (streaming Bar → batch ratios).

The streaming ``Bar`` emits signed-flow SUMS; the models train on the batch
RATIOS (``signed_vol_imb`` / ``signed_dollar_imb`` / ``signed_tick_imb`` from
``aggregate_signed_flow_by_close_indices``). :func:`bar_flow_ratios` is the
canonical adapter. This suite pins, on adversarial + property-based tick
streams, that for every bar:

    bar_flow_ratios(streaming_bar)  ==  batch signed_flow over the SAME ticks

INCLUDING the corner cases the s40 review flagged: mixed known/unknown aggressor
sides, flat-price runs (zero-ticks), and thin bars.

Oracle note (s40 finding): batch ``signed_flow_features``/``_tick_rule_direction``
emits direction 0 on a LEADING flat-price run before the first move
(``[1,0,0,0]``), which would drop that tick's volume. The streaming path resolves
the same run to ``[1,1,1,1]`` (the correct tick rule, seed +1 carried forward).
To compare apples to apples we feed BOTH paths the SAME synthetic tick stream
built with the SAME dollar-bar close indices and assert per-bar equality; we
avoid constructing fixtures whose FIRST tick is an unknown-side zero-tick (the
single 1-in-300k batch wart), which is out of s41 scope and tracked separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.bars import DollarBarBuilder
from quantcore.bars.bars import threshold_bar_close_indices
from quantcore.data import Side, TradeEvent
from quantcore.features.top_of_book import (
    aggregate_signed_flow_by_close_indices,
    bar_flow_ratios,
    signed_flow_features,
)

ATOL = 1e-10


def _build_df_and_trades(prices, sizes, side_codes):
    """Build the batch tick DataFrame and the streaming TradeEvent list from
    the same arrays. side_code: 0=A(-1, sell aggressor), 1=B(+1, buy
    aggressor), 2=unknown. (s83 F11: DBN side = side of the aggressor.)"""
    n = len(prices)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="s")
    side_str = np.where(
        np.asarray(side_codes) == 0, "A", np.where(np.asarray(side_codes) == 1, "B", "N")
    )
    df = pd.DataFrame(
        {
            "price": np.asarray(prices, dtype=float),
            "size": np.asarray(sizes, dtype=float),
            "side": side_str,
        },
        index=ts,
    )
    side_val = [int(Side.ASK) if c == 0 else (int(Side.BID) if c == 1 else 0) for c in side_codes]
    trades = [
        TradeEvent(
            ts_event=int(ts[i].value),
            instrument_id=1,
            sequence=i,
            price=float(prices[i]),
            size=float(sizes[i]),
            aggressor_side=side_val[i],  # type: ignore[arg-type]  # 0 = unknown sentinel
        )
        for i in range(n)
    ]
    return df, trades


def _batch_ratios_per_bar(df, threshold):
    """Batch reference: dollar-bar close indices → per-bar flow ratios."""
    close_indices = threshold_bar_close_indices(
        df, threshold, kind="dollar", price_col="price", volume_col="size"
    )
    flow = signed_flow_features(df, price_col="price", volume_col="size", side_col="side")
    agg = aggregate_signed_flow_by_close_indices(flow, close_indices)
    return close_indices, agg


def _stream_ratios_per_bar(trades, threshold):
    builder = DollarBarBuilder(threshold=threshold)
    out = []
    for t in trades:
        bar = builder.on_event(t)
        if bar is not None:
            out.append(bar_flow_ratios(bar))
    return out


def _assert_parity(prices, sizes, side_codes, threshold):
    df, trades = _build_df_and_trades(prices, sizes, side_codes)
    close_indices, batch = _batch_ratios_per_bar(df, threshold)
    stream = _stream_ratios_per_bar(trades, threshold)

    assert len(stream) == len(batch), (
        f"bar count diverged: stream={len(stream)} batch={len(batch)} "
        f"(close_indices={list(close_indices)})"
    )
    if len(batch) == 0:
        # No bars formed (e.g. all-zero sizes never cross the dollar threshold): stream and
        # batch agree vacuously, and the empty batch frame has no columns to index below.
        return
    for k in ("signed_vol_imb", "signed_dollar_imb", "signed_tick_imb"):
        s = np.array([r[k] for r in stream], dtype=float)
        b = batch[k].to_numpy().astype(float)
        np.testing.assert_allclose(s, b, rtol=0.0, atol=ATOL, err_msg=k)


# ---------------------------------------------------------------------------
# Adversarial hand-picked scenarios
# ---------------------------------------------------------------------------


def test_all_known_sides() -> None:
    prices = [100.0 + 0.1 * i for i in range(60)]
    sizes = [50.0] * 60
    side_codes = [i % 2 for i in range(60)]  # alternating A/B
    _assert_parity(prices, sizes, side_codes, threshold=20_000.0)


def test_all_unknown_sides_tick_rule() -> None:
    rng = np.random.default_rng(1)
    prices = (100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 80)))).tolist()
    sizes = rng.integers(1, 40, 80).astype(float).tolist()
    side_codes = [2] * 80  # all unknown → both paths tick-rule
    _assert_parity(prices, sizes, side_codes, threshold=15_000.0)


def test_mixed_sides_with_flat_runs() -> None:
    # Interleave flat-price runs (zero-ticks) with moves, mixed known/unknown.
    # First tick is a known side, so the leading-flat batch wart cannot trigger.
    prices = [100.0, 100.0, 100.0, 101.0, 101.0, 100.5, 100.5, 100.5, 102.0, 101.0] * 4
    sizes = [30.0, 70.0, 20.0, 90.0, 10.0, 60.0, 40.0, 25.0, 80.0, 55.0] * 4
    side_codes = [0, 2, 2, 1, 2, 0, 2, 2, 1, 2] * 4  # tick 0 is known (A)
    _assert_parity(prices, sizes, side_codes, threshold=25_000.0)


def test_thin_bars_small_threshold() -> None:
    # Tiny threshold → many bars of 1-2 ticks each (thin-bar regime).
    prices = [100.0, 101.0, 100.0, 102.0, 99.0, 103.0, 98.0, 104.0]
    sizes = [500.0] * 8  # each tick ~ its own bar at threshold below one tick's $
    side_codes = [0, 1, 2, 2, 1, 0, 2, 1]
    _assert_parity(prices, sizes, side_codes, threshold=40_000.0)


def test_single_unknown_after_known_then_flat() -> None:
    # condition-4 stress: known trade, then a flat-price unknown run.
    prices = [100.0, 101.0, 101.0, 101.0, 101.0, 102.0]
    sizes = [40.0] * 6
    side_codes = [0, 0, 2, 2, 2, 1]
    _assert_parity(prices, sizes, side_codes, threshold=12_000.0)


# ---------------------------------------------------------------------------
# Property-based fuzz (Hypothesis) — known FIRST tick to dodge the leading-flat
# batch wart (out of s41 scope; tracked separately).
# ---------------------------------------------------------------------------

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - hypothesis is a dev dep
    _HAS_HYPOTHESIS = False


@pytest.mark.skipif(not _HAS_HYPOTHESIS, reason="hypothesis not installed")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_parity_property_fuzz(data) -> None:
    if not _HAS_HYPOTHESIS:
        return
    n = data.draw(st.integers(min_value=3, max_value=60))
    # prices: positive, with deliberate repeats to force zero-ticks.
    prices = data.draw(
        st.lists(
            st.floats(min_value=50.0, max_value=150.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    # round to 1 decimal so equal-price runs actually occur
    prices = [round(p, 1) for p in prices]
    sizes = data.draw(st.lists(st.integers(min_value=1, max_value=100), min_size=n, max_size=n))
    sizes = [float(s) for s in sizes]
    side_codes = data.draw(st.lists(st.integers(min_value=0, max_value=2), min_size=n, max_size=n))
    # Force the FIRST tick to a KNOWN side so the leading-flat-run batch wart
    # (s40 finding, 1-in-300k, out of s41 scope) cannot trigger.
    side_codes[0] = data.draw(st.sampled_from([0, 1]))
    # threshold small enough to yield >=1 bar.
    total_dollar = sum(p * s for p, s in zip(prices, sizes))
    threshold = data.draw(
        st.floats(
            min_value=max(1.0, total_dollar / max(n, 1) * 0.5),
            max_value=max(2.0, total_dollar),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    _assert_parity(prices, sizes, side_codes, threshold)
