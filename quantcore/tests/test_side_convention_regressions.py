"""s83 F11 + F1 regressions — Databento aggressor-side sign convention.

F11 (CRITICAL): DBN ``side`` is the side of the AGGRESSOR for trades —
'A' (Ask) = sell aggressor → -1, 'B' (Bid) = buy aggressor → +1, 'N' =
none → tick-rule fallback (vendor: databento_dbn.Side, "the side of the
aggressor for trades"). The pre-s83 ``_SIDE_MAP`` (and the quantengine
adapter ``_side_to_aggressor``) inverted this together, so every
exchange-side signed-flow feature was sign-flipped on A/B-tagged trades,
consistently in research and live. These tests pin the corrected
convention against the in-repo ``Side`` enum so any future drift fails
loudly.

F1 (HIGH): the streaming imbalance/runs engines wrote the RAW side into
``_OHLCVBuffer`` (0 for unknowns) instead of the tick-rule-resolved
direction the S40-fixed threshold engine writes — ``signed_volume_sum``,
``signed_dollar_sum`` and ``signed_tick_imbalance`` were identically zero
on side-less tapes. These tests pin per-bar sum-level parity against a
resolved-direction oracle for threshold, imbalance, and runs families on
both side-less and mixed-side tapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quantcore.bars.streaming import (
    ImbalanceConfig,
    RunsConfig,
    TickImbalanceBarBuilder,
    TickRunsBarBuilder,
    VolumeBarBuilder,
)
from quantcore.data.events import Side
from quantcore.features.top_of_book import (
    _SIDE_MAP,
    _tick_rule_direction,
    signed_flow_features,
)

# ===================================================================
# F11 — sign-convention pins
# ===================================================================


class TestSideMapConvention:
    def test_side_map_pins_dbn_aggressor_convention(self) -> None:
        assert _SIDE_MAP == {"B": 1.0, "A": -1.0}

    def test_side_map_agrees_with_events_side_enum(self) -> None:
        """The batch map and the typed Side enum must express the same
        convention — BID/buy aggressor = +1, ASK/sell aggressor = -1."""
        assert _SIDE_MAP["B"] == float(int(Side.BID)) == 1.0
        assert _SIDE_MAP["A"] == float(int(Side.ASK)) == -1.0

    def test_buy_sweep_has_positive_signed_flow(self) -> None:
        """A rising tape of exchange-marked buy aggressors ('B') must show
        POSITIVE signed volume. Pre-s83 this returned -40 (the executed
        F11 repro): the exchange tag made the feature worse than no tag."""
        df = pd.DataFrame(
            {
                "price": [100.0, 100.01, 100.02, 100.03],
                "size": [10.0, 10.0, 10.0, 10.0],
                "side": ["B", "B", "B", "B"],
            }
        )
        flow = signed_flow_features(df, side_col="side")
        assert flow["signed_volume"].sum() == pytest.approx(40.0)

    def test_sell_sweep_has_negative_signed_flow(self) -> None:
        df = pd.DataFrame(
            {
                "price": [100.03, 100.02, 100.01, 100.0],
                "size": [10.0, 10.0, 10.0, 10.0],
                "side": ["A", "A", "A", "A"],
            }
        )
        flow = signed_flow_features(df, side_col="side")
        assert flow["signed_volume"].sum() == pytest.approx(-40.0)

    def test_exchange_side_overrides_tick_rule_with_correct_sign(self) -> None:
        """Sell aggressors on an uptick ('A' while price rises): exchange
        truth (-1) must win over the tick rule (+1) — with the CORRECT
        sign, not its inverse."""
        df = pd.DataFrame(
            {
                "price": [100.0, 100.05],
                "size": [1.0, 1.0],
                "side": ["A", "A"],
            }
        )
        flow = signed_flow_features(df, side_col="side")
        assert flow["tick_dir"].iloc[1] == -1.0


# ===================================================================
# F1 — streaming signed-flow parity vs resolved-direction oracle
# ===================================================================


@dataclass
class _Trade:
    """Structurally satisfies the streaming ``_TradeEventLike`` protocol."""

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int = 0


def _stream_with_spans(builder, prices, sizes, sides):
    """Feed the tape; return (bars, [(start, end)] inclusive tick spans)."""
    bars, spans, start = [], [], 0
    for i, (p, v, a) in enumerate(zip(prices, sizes, sides)):
        bar = builder.on_event(_Trade(i, 1, i, float(p), float(v), int(a)))
        if bar is not None:
            bars.append(bar)
            spans.append((start, i))
            start = i + 1
    return bars, spans


def _resolved_directions(prices, sides) -> np.ndarray:
    """Spec oracle: exchange side where reported, else tick rule — the
    convention shared by batch ``_resolve_side_column``, the streaming
    ``_TickRule._direction``, and (post-F1) the buffer write."""
    tick = _tick_rule_direction(np.asarray(prices, dtype=np.float64))
    a = np.asarray(sides, dtype=np.float64)
    return np.where(a != 0.0, a, tick)


def _tape(n: int = 3_000, seed: int = 3, side_mode: str = "none"):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 5e-4, n)))
    prices[::97] = prices[1::97][: len(prices[::97])]  # exercise zero-diff carry
    sizes = rng.integers(1, 30, n).astype(np.float64)
    if side_mode == "none":
        sides = np.zeros(n, dtype=int)
    else:
        sides = rng.choice([-1, 0, 1], size=n, p=[0.4, 0.2, 0.4])
    return prices, sizes, sides


def _builders():
    return [
        TickImbalanceBarBuilder(ImbalanceConfig(exp_num_ticks_init=150, exp_imbalance_init=0.05)),
        TickRunsBarBuilder(
            RunsConfig(
                exp_num_ticks_init=150,
                exp_prob_buy_init=0.5,
                exp_w_buy_init=1.0,
                exp_w_sell_init=1.0,
            )
        ),
        VolumeBarBuilder(threshold=2_000.0),
    ]


@pytest.mark.parametrize("side_mode", ["none", "mixed"])
def test_signed_flow_matches_resolved_directions(side_mode: str) -> None:
    """Per-bar signed-flow sums must equal the resolved-direction oracle
    for ALL engine families. Pre-s83 the imbalance/runs engines emitted
    identically-zero signed flows on a side-less tape (executed F1 repro)
    while the threshold engine — fixed in S40 — was correct."""
    prices, sizes, sides = _tape(side_mode=side_mode)
    d = _resolved_directions(prices, sides)

    for builder in _builders():
        bars, spans = _stream_with_spans(builder, prices, sizes, sides)
        assert bars, f"{type(builder).__name__}: fixture emitted no bars"
        for bar, (s, e) in zip(bars, spans):
            seg = slice(s, e + 1)
            assert bar.signed_volume_sum == pytest.approx(
                float((d[seg] * sizes[seg]).sum()), rel=1e-9, abs=1e-9
            ), f"{type(builder).__name__} signed_volume_sum, span {s}:{e}"
            assert bar.signed_dollar_sum == pytest.approx(
                float((d[seg] * prices[seg] * sizes[seg]).sum()), rel=1e-9
            ), f"{type(builder).__name__} signed_dollar_sum, span {s}:{e}"
            # |d| == 1 for every tick => signed_tick_imbalance == mean(d).
            assert bar.signed_tick_imbalance == pytest.approx(
                float(d[seg].mean()), rel=1e-9, abs=1e-12
            ), f"{type(builder).__name__} signed_tick_imbalance, span {s}:{e}"


def test_sideless_tape_signed_flow_is_not_degenerate() -> None:
    """The exact pre-s83 symptom: every imbalance/runs bar carried
    signed_volume_sum == 0.0 on a side-less tape. At least one bar must
    now carry nonzero signed flow."""
    prices, sizes, sides = _tape(side_mode="none")
    for builder in _builders():
        bars, _ = _stream_with_spans(builder, prices, sizes, sides)
        assert any(b.signed_volume_sum != 0.0 for b in bars), (
            f"{type(builder).__name__}: all signed_volume_sum are zero "
            "(raw side written to buffer instead of resolved direction)"
        )
