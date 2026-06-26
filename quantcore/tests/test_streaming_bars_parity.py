"""Parity tests: streaming BarBuilders vs legacy functional API.

S34 §3.AC8 protocol:
  - Integer / count / timestamp fields: np.array_equal (exact).
  - Float fields: np.testing.assert_allclose(rtol=0.0, atol=1e-10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcore.bars import (
    DollarBarBuilder,
    DollarImbalanceBarBuilder,
    DollarRunsBarBuilder,
    ImbalanceConfig,
    RunsConfig,
    TickBarBuilder,
    TickImbalanceBarBuilder,
    TickRunsBarBuilder,
    VolumeBarBuilder,
    VolumeImbalanceBarBuilder,
    VolumeRunsBarBuilder,
    dollar_bars,
    dollar_imbalance_bars,
    dollar_runs_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_runs_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_runs_bars,
)
from quantcore.data import Bar, Side, TradeEvent
from quantcore.features.top_of_book import dollar_bars_with_microstructure
from tests.fixtures.synthetic_tick_stream import (
    TickRuleTradeEvent,
    _make_trades_tick_rule,
    gbm_stream,
    trending_stream,
)

ATOL = 1e-10
RTOL = 0.0


def _run_streaming(builder, trades: list[TradeEvent] | list[TickRuleTradeEvent]) -> list[Bar]:
    out: list[Bar] = []
    for t in trades:
        bar = builder.on_event(t)
        if bar is not None:
            out.append(bar)
    return out


def _assert_parity(streaming: list[Bar], legacy: pd.DataFrame) -> None:
    assert len(streaming) == len(legacy), (
        f"bar count diverged: streaming={len(streaming)} legacy={len(legacy)}"
    )
    if not streaming:
        return

    # Integer / count / timestamp fields — exact
    s_ts_event = np.asarray([b.ts_event for b in streaming], dtype=np.int64)
    l_ts_event = np.asarray([t.value for t in legacy.index], dtype=np.int64)
    assert np.array_equal(s_ts_event, l_ts_event), "ts_event diverged"

    s_tc = np.asarray([b.tick_count for b in streaming], dtype=np.int64)
    l_tc = legacy["tick_count"].to_numpy().astype(np.int64)
    assert np.array_equal(s_tc, l_tc), "tick_count diverged"

    # Float fields — atol parity
    for field in ("open", "high", "low", "close", "volume", "vwap", "dollar_volume"):
        s = np.asarray([getattr(b, field) for b in streaming], dtype=np.float64)
        ll = legacy[field].to_numpy().astype(np.float64)
        np.testing.assert_allclose(s, ll, rtol=RTOL, atol=ATOL, err_msg=field)


# ---------------------------------------------------------------------------
# Plain bars (Tick / Volume / Dollar)
# ---------------------------------------------------------------------------


def test_tick_bar_builder_parity_with_legacy() -> None:
    df, trades = gbm_stream(n=1_000)
    streaming = _run_streaming(TickBarBuilder(threshold=10), trades)
    legacy = tick_bars(df, threshold=10)
    _assert_parity(streaming, legacy)


def test_volume_bar_builder_parity_with_legacy() -> None:
    df, trades = gbm_stream(n=1_000)
    streaming = _run_streaming(VolumeBarBuilder(threshold=200.0), trades)
    legacy = volume_bars(df, threshold=200.0)
    _assert_parity(streaming, legacy)


def test_dollar_bar_builder_parity_with_legacy() -> None:
    df, trades = gbm_stream(n=1_000)
    streaming = _run_streaming(DollarBarBuilder(threshold=50_000.0), trades)
    legacy = dollar_bars(df, threshold=50_000.0)
    _assert_parity(streaming, legacy)


# ---------------------------------------------------------------------------
# Imbalance bars (explicit init, both fixture variants)
# ---------------------------------------------------------------------------

ICFG = ImbalanceConfig(
    exp_num_ticks_init=50.0,
    exp_imbalance_init=0.0,
    ewma_span_ticks=20,
    ewma_span_imbalance=20,
    exp_num_ticks_min=5.0,
    exp_num_ticks_max=500.0,
)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_tick_imbalance_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(TickImbalanceBarBuilder(config=ICFG), tick_rule_trades)
    legacy = tick_imbalance_bars(df, config=ICFG)
    _assert_parity(streaming, legacy)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_volume_imbalance_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(VolumeImbalanceBarBuilder(config=ICFG), tick_rule_trades)
    legacy = volume_imbalance_bars(df, config=ICFG)
    _assert_parity(streaming, legacy)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_dollar_imbalance_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(DollarImbalanceBarBuilder(config=ICFG), tick_rule_trades)
    legacy = dollar_imbalance_bars(df, config=ICFG)
    _assert_parity(streaming, legacy)


# ---------------------------------------------------------------------------
# Runs bars (explicit init, both fixture variants)
# ---------------------------------------------------------------------------

RCFG = RunsConfig(
    exp_num_ticks_init=50.0,
    exp_prob_buy_init=0.5,
    exp_w_buy_init=1.0,
    exp_w_sell_init=1.0,
    ewma_span_ticks=20,
    ewma_span_prob=20,
    ewma_span_weights=20,
    exp_num_ticks_min=5.0,
    exp_num_ticks_max=500.0,
)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_tick_runs_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(TickRunsBarBuilder(config=RCFG), tick_rule_trades)
    legacy = tick_runs_bars(df, config=RCFG)
    _assert_parity(streaming, legacy)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_volume_runs_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(VolumeRunsBarBuilder(config=RCFG), tick_rule_trades)
    legacy = volume_runs_bars(df, config=RCFG)
    _assert_parity(streaming, legacy)


@pytest.mark.parametrize("stream", [gbm_stream, trending_stream])
def test_dollar_runs_bar_builder_parity_with_legacy(stream) -> None:
    df, _trades = stream(n=1_000)
    tick_rule_trades = _make_trades_tick_rule(df)
    streaming = _run_streaming(DollarRunsBarBuilder(config=RCFG), tick_rule_trades)
    legacy = dollar_runs_bars(df, config=RCFG)
    _assert_parity(streaming, legacy)


# ---------------------------------------------------------------------------
# S40 — TBBO microstructure train/serve parity (F2/F3)
#
# The OHLCV parity tests above prove bar BOUNDARIES + OHLCV agree. S40 closes
# the *microstructure feature* seam: the streaming Bar (live) and the batch
# dollar_bars_with_microstructure (what the model trains on) must agree on the
# fields a flow model consumes, INCLUDING on unknown-aggressor-side ticks where
# D1 added the tick-rule fallback to the plain-dollar-bar streaming path.
#
# Schema note: streaming Bar emits signed-flow SUMS (signed_volume_sum,
# signed_dollar_sum) while batch emits RATIOS (signed_vol_imb = Σdir·vol/Σ|dir·vol|
# = signed_volume_sum / volume since vol>0). signed_tick_imbalance is emitted
# identically by both (Σdir/Σ|dir|) and is the field that most directly
# exercises the D1 direction-resolution fix. We compare the directly-equal
# fields and reconstruct the ratio fields from the streaming sums.
# ---------------------------------------------------------------------------

_MICRO_THRESHOLD = 50_000.0


def _tbbo_stream(n: int = 1_500, seed: int = 7):
    """Synthetic TBBO tick stream in BOTH representations.

    Returns (df, trades):
      - df: tick DataFrame for the BATCH path — columns price, size, side
        ('A' sell-aggressor / 'B' buy-aggressor / 'N' unknown; DBN aggressor
        convention, s83 F11), bid_px_00, ask_px_00, bid_sz_00, ask_sz_00.
      - trades: list[TradeEvent] for the STREAMING path — same ticks with
        aggressor_side = Side.ASK (-1) for 'A' / Side.BID (+1) for 'B' / 0 (unknown sentinel,
        the value _extract_bbo + _direction treat as "fall back to tick rule"),
        and matching bid_px/ask_px/bid_sz/ask_sz.

    ~1/3 of ticks carry an UNKNOWN side (the regime where D1's tick-rule
    fallback matters): batch resolves these via the tick rule, and post-D1 the
    streaming plain-dollar-bar path does too — so the two must now agree.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-02 09:30:00", periods=n, freq="s")
    returns = rng.normal(0.0, 0.0007, size=n)
    price = 100.0 * np.exp(np.cumsum(returns))
    size = rng.integers(1, 60, size=n).astype(np.float64)
    half_spread = rng.uniform(0.005, 0.03, size=n)
    bid_px = price - half_spread
    ask_px = price + half_spread
    bid_sz = rng.integers(1, 500, size=n).astype(np.float64)
    ask_sz = rng.integers(1, 500, size=n).astype(np.float64)
    # side code per tick: 0=A(-1, sell aggr), 1=B(+1, buy aggr), 2=N(unknown).
    # ~1/3 unknown. (s83 F11: DBN side = side of the aggressor.)
    side_code = rng.integers(0, 3, size=n)

    side_str = np.where(side_code == 0, "A", np.where(side_code == 1, "B", "N"))
    df = pd.DataFrame(
        {
            "price": price,
            "size": size,
            "side": side_str,
            "bid_px_00": bid_px,
            "ask_px_00": ask_px,
            "bid_sz_00": bid_sz,
            "ask_sz_00": ask_sz,
        },
        index=ts,
    )

    # Unknown side is the int sentinel 0 (NOT None): _extract_bbo does
    # int(aggressor_side) and _direction treats 0 as "tick-rule fallback".
    # Side.BID==+1 / Side.ASK==-1 are IntEnums so they pass through int(...).
    side_val: list[int] = [
        int(Side.ASK) if c == 0 else (int(Side.BID) if c == 1 else 0) for c in side_code
    ]
    trades = [
        TradeEvent(
            ts_event=int(ts[i].value),
            instrument_id=1,
            sequence=i,
            price=float(price[i]),
            size=float(size[i]),
            aggressor_side=side_val[i],  # type: ignore[arg-type]  # int sentinel 0 = unknown
            bid_px=float(bid_px[i]),
            ask_px=float(ask_px[i]),
            bid_sz=float(bid_sz[i]),
            ask_sz=float(ask_sz[i]),
        )
        for i in range(n)
    ]
    return df, trades


def test_streaming_microstructure_parity_with_batch() -> None:
    """Streaming Bar microstructure + signed-flow match the batch aggregation
    on a TBBO stream with mixed known/unknown aggressor sides (S40 F2/F3).

    Pre-D1 this FAILS: the streaming plain-dollar-bar path stored 0 for
    unknown-side signed flow while the batch path tick-ruled, so
    signed_tick_imbalance diverged on unknown-side bars.
    """
    df, trades = _tbbo_stream()
    streaming = _run_streaming(DollarBarBuilder(threshold=_MICRO_THRESHOLD), trades)
    batch = dollar_bars_with_microstructure(df, threshold=_MICRO_THRESHOLD, side_col="side")

    assert len(streaming) == len(batch), (
        f"bar count diverged: streaming={len(streaming)} batch={len(batch)}"
    )
    assert len(streaming) > 5, "fixture too small to be meaningful"

    # Microstructure fields the streaming Bar emits (derive from BBO only).
    for s_attr, b_col in (
        ("spread_mean", "spread_mean"),
        ("spread_last", "spread_last"),
        ("spread_std", "spread_std"),
        ("microprice_dev_mean", "microprice_dev_mean"),
        ("microprice_dev_last", "microprice_dev_last"),
        ("microprice_dev_std", "microprice_dev_std"),
        ("imbalance_mean", "quoted_imbalance_mean"),  # name differs, semantics same
    ):
        s = np.asarray([getattr(b, s_attr) for b in streaming], dtype=np.float64)
        bb = batch[b_col].to_numpy().astype(np.float64)
        np.testing.assert_allclose(s, bb, rtol=RTOL, atol=ATOL, err_msg=f"{s_attr} vs {b_col}")

    # THE D1 pin: signed_tick_imbalance == batch signed_tick_imb (both Σdir/Σ|dir|),
    # equal ONLY if streaming resolves unknown-side direction via the tick rule.
    s_tick = np.asarray([b.signed_tick_imbalance for b in streaming], dtype=np.float64)
    b_tick = batch["signed_tick_imb"].to_numpy().astype(np.float64)
    np.testing.assert_allclose(
        s_tick, b_tick, rtol=RTOL, atol=ATOL, err_msg="signed_tick_imbalance vs signed_tick_imb"
    )

    # Reconstruct the batch RATIO fields from the streaming SUMS:
    #   batch signed_vol_imb = Σ(dir·vol)/Σ|dir·vol| = signed_volume_sum / volume (vol>0).
    s_vol = np.asarray([b.signed_volume_sum for b in streaming], dtype=np.float64)
    s_volume = np.asarray([b.volume for b in streaming], dtype=np.float64)
    s_vol_imb = s_vol / s_volume
    b_vol_imb = batch["signed_vol_imb"].to_numpy().astype(np.float64)
    np.testing.assert_allclose(
        s_vol_imb, b_vol_imb, rtol=RTOL, atol=ATOL, err_msg="reconstructed signed_vol_imb"
    )

    #   batch signed_dollar_imb = signed_dollar_sum / dollar_volume.
    s_dol = np.asarray([b.signed_dollar_sum for b in streaming], dtype=np.float64)
    s_dolvol = np.asarray([b.dollar_volume for b in streaming], dtype=np.float64)
    s_dol_imb = s_dol / s_dolvol
    b_dol_imb = batch["signed_dollar_imb"].to_numpy().astype(np.float64)
    np.testing.assert_allclose(
        s_dol_imb, b_dol_imb, rtol=RTOL, atol=ATOL, err_msg="reconstructed signed_dollar_imb"
    )


def test_streaming_microstructure_fixture_has_unknown_sides() -> None:
    """Guard: the parity fixture genuinely exercises unknown-side ticks (the
    regime D1 fixes). If this regresses to all-known sides, the parity test
    above would pass even with the D1 fix reverted."""
    df, _trades = _tbbo_stream()
    n_unknown = int((df["side"] == "N").sum())
    assert n_unknown > 0
    assert 0.2 < n_unknown / len(df) < 0.5  # ~1/3 by construction
