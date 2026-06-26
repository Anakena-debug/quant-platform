"""S45 — production bar-builder wiring guard (the cli.py train/serve skew fix).

s44 discovered that cli.py's former ``_QuantcoreBarAdapter`` round-tripped each
feed event through quantcore ``TradeEvent`` (a 2-state ``Side`` enum with no zero)
and so collapsed ``aggressor_side == 0`` ('N'/no-aggressor — 90.91% of TXN trades)
to ``+1`` instead of the tick-rule fallback, and dropped BBO (→ ``spread_last``
NaN → dead spread gate). The research batch pipeline
(``top_of_book._resolve_side_column``) tick-rule-resolves unknown side and keeps
BBO, so the live path diverged from research on the signed-flow features.

s45 replaced the adapter with ``_make_quantcore_bar_builder``, which returns the
quantcore builder UNWRAPPED so feed events reach it directly (it duck-types the
event and reads ``aggressor_side`` + BBO via getattr). These tests guard that fix
against regression:

* unit — a fed ``aggressor_side=0`` on a descending price resolves to the TICK
  RULE (negative signed flow), not a forced ``+1``, and ``spread_last`` survives;
* data parity — driving the REAL TXN.dbn through the PRODUCTION factory builder
  reproduces the s43 frozen research replay byte-for-byte (closes the gap the s44
  test left open: s44 drove a bare ``DollarBarBuilder``, this drives exactly the
  object cli.py constructs). Skips cleanly when the research outputs are absent.

NOT a profitability claim — the realistic-exit replay of this alpha is negative.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from quantengine.runtime.streaming.cli import _make_quantcore_bar_builder
from quantengine.runtime.streaming.databento_feed import (
    DatabentoTradeFeed,
    _DatabentoTradeEvent,
)

_REPO = Path(__file__).resolve().parents[2]
_ARTIFACT = _REPO / "alpha_R" / "outputs" / "artifacts" / "txn_flow_regime_v1"
_PARITY_CSV = _REPO / "alpha_R" / "outputs" / "diagnostics" / "txn_alpha_replay_signals_frozen.csv"
_DBN = _REPO / "quantengine" / "data" / "backtest" / "TXN.dbn"
_THRESHOLD = 500_000.0


# ---------------------------------------------------------------------------
# Unit — the decisive, data-free regression guard
# ---------------------------------------------------------------------------
def test_factory_builder_resolves_unknown_side_via_tick_rule_and_keeps_bbo() -> None:
    """aggressor_side=0 ('N') -> TICK RULE (not forced +1); BBO survives.

    Descending price + all-zero aggressor side: tick rule seeds +1 on the first
    tick then marks every down-tick -1, so signed flow is NEGATIVE. The old
    _QuantcoreBarAdapter forced 0 -> Side.BID(+1) (signed flow positive) and
    dropped BBO (spread_last NaN) — both are asserted absent here.
    """
    builder = _make_quantcore_bar_builder("dollar", 10_000_000.0)  # high -> no mid-close
    assert builder is not None
    on_event = cast(Any, builder).on_event

    prices = [100.0, 99.0, 98.0, 97.0, 96.0]  # strictly descending
    for i, px in enumerate(prices):
        ev = _DatabentoTradeEvent(
            ts_event=1_000 + i,
            instrument_id=7,
            sequence=i,
            price=px,
            size=10.0,
            aggressor_side=0,  # 'N' / no-aggressor — must fall back to the tick rule
            bid_px=px - 0.01,
            ask_px=px + 0.01,
            bid_sz=5.0,
            ask_sz=5.0,
        )
        assert on_event(ev) is None  # 4,900 dollar volume << threshold

    bar = cast(Any, builder).flush()
    assert bar is not None

    # Tick directions: [+1, -1, -1, -1, -1] -> signed_volume_sum = (1-4)*10 = -30.
    # Under the old bug (force +1) it would be +50.
    assert bar.signed_volume_sum < 0.0, (
        f"signed_volume_sum={bar.signed_volume_sum} >= 0 — 'N' wrongly forced to +1"
    )
    assert bar.signed_volume_sum == pytest.approx(-30.0)
    assert bar.signed_tick_imbalance == pytest.approx(-0.6)  # (1-4)/5
    # BBO survived the direct feed (old adapter dropped it -> NaN spread).
    assert math.isfinite(bar.spread_last), "spread_last NaN — BBO dropped"
    assert bar.spread_last == pytest.approx(0.02)


def test_factory_returns_none_for_non_quantcore_bar_type() -> None:
    """Non-quantcore bar types (e.g. the demo 'simple') get no quantcore builder."""
    assert _make_quantcore_bar_builder("simple", _THRESHOLD) is None
    assert _make_quantcore_bar_builder("nonsense", _THRESHOLD) is None


def test_factory_builds_the_expected_quantcore_types() -> None:
    """dollar/volume/tick map to the three quantcore threshold builders."""
    assert type(_make_quantcore_bar_builder("dollar", _THRESHOLD)).__name__ == "DollarBarBuilder"
    assert type(_make_quantcore_bar_builder("volume", _THRESHOLD)).__name__ == "VolumeBarBuilder"
    assert type(_make_quantcore_bar_builder("tick", _THRESHOLD)).__name__ == "TickBarBuilder"


# ---------------------------------------------------------------------------
# Data parity — the production factory builder reproduces research on real ticks
# ---------------------------------------------------------------------------
def _sync_iter(feed: object) -> list[object]:
    """Pull all events from an async DataFeed synchronously (replay file = no IO wait)."""
    import asyncio

    async def _collect() -> list[object]:
        out: list[object] = []
        async for ev in cast(Any, feed):
            out.append(ev)
        return out

    return asyncio.new_event_loop().run_until_complete(_collect())


class _RecordingBroker:
    """Minimal SyncBrokerFacade: tracks net position, fills MARKET fully."""

    def __init__(self) -> None:
        from quantengine.portfolio.state import Position

        self._Position = Position
        self._qty = 0

    def get_position(self, ticker: str, timeout: float | None = None):
        return self._Position(ticker, self._qty, 0.0) if self._qty != 0 else None

    def submit_order(self, order, timeout: float | None = None):
        self._qty += order.quantity if order.side.value == "BUY" else -order.quantity
        return []

    def cancel_order(self, order_id, timeout: float | None = None) -> bool:
        return True

    def get_account_state(self, timeout: float | None = None):
        from quantengine.portfolio.state import PortfolioState

        return PortfolioState(cash=0.0, positions={})


def _artifact_is_pre_s83() -> bool:
    """True when the frozen artifact predates the s83 F11 aggressor-side fix
    (no ``side_convention`` stamp in manifest.json) — the frozen replay CSV was
    generated under the inverted A/B mapping and cannot byte-match the corrected
    pipeline. Regenerate the s43 outputs to re-arm this test."""
    import json

    from quantcore.models.frozen_strategy import SIDE_CONVENTION

    try:
        manifest = json.loads((_ARTIFACT / "manifest.json").read_text())
    except (OSError, ValueError):
        return True
    return manifest.get("side_convention") != SIDE_CONVENTION


@pytest.mark.skipif(
    not (_ARTIFACT.exists() and _PARITY_CSV.exists() and _DBN.exists()),
    reason="s43 artifact / parity CSV / TXN.dbn absent (regenerable research outputs)",
)
@pytest.mark.skipif(
    _artifact_is_pre_s83(),
    reason=(
        "s83 F11: frozen artifact + parity CSV predate the aggressor-side "
        "sign fix (no side_convention stamp); regenerate the s43 outputs"
    ),
)
def test_cli_factory_builder_reproduces_frozen_replay_on_txn() -> None:
    """The PRODUCTION factory builder (cli._make_quantcore_bar_builder) -> the
    frozen strategy reproduces the s43 research replay byte-for-byte on real
    TXN.dbn.

    This is the gap the s44 parity test could not close: s44 drove a bare
    DollarBarBuilder, so it proved the strategy faithful given correct bars but did
    not exercise the cli.py wiring. Here the builder is constructed by the SAME
    factory cli.py uses, and events are fed to it the SAME way (directly), so a
    regression of the s45 fix (re-introducing a lossy conversion) would fail here.
    """
    from quantengine.strategies import FrozenFlowRegimeStrategy

    frozen = pd.read_csv(_PARITY_CSV)
    event_idx = {int(i) for i in frozen["bar_index"].unique()}

    strat = FrozenFlowRegimeStrategy(
        _ARTIFACT, ticker="TXN", shares_per_unit=100, record=True, score_bar_indices=event_idx
    )

    # PRODUCTION wiring: the builder comes from the cli.py factory, unwrapped.
    builder = _make_quantcore_bar_builder("dollar", _THRESHOLD)
    assert builder is not None
    broker = _RecordingBroker()
    for event in _sync_iter(DatabentoTradeFeed.from_dbn_file(_DBN)):
        bar = cast(Any, builder).on_event(event)
        if bar is not None:
            strat.on_bar(int(cast(Any, bar).ts_event), cast(Any, bar), cast(Any, None), broker)

    live = pd.DataFrame(strat.records).set_index("bar_index")
    assert len(live) > 500, f"too few scored bars: {len(live)}"

    # Collapse the frozen CSV's per-(bar_index, H) rows to one target via the live
    # tight>moderate proxy gate (same construction as the s44 parity test).
    by_idx: dict[int, dict[str, Any]] = {}
    for bar_index, grp in frozen.groupby("bar_index"):
        h100 = cast(Any, grp[grp["H"] == 100])
        h50 = cast(Any, grp[grp["H"] == 50])
        regime, fp = "none", 0.0
        if len(h100) and h100["regime_proxy"].iloc[0] == "tight":
            regime, fp = "tight", float(h100["filtered_position_proxy"].iloc[0])
        elif len(h50) and h50["regime_proxy"].iloc[0] == "moderate":
            regime, fp = "moderate", float(h50["filtered_position_proxy"].iloc[0])
        by_idx[int(cast(Any, bar_index))] = {"regime": regime, "fp": fp}

    matched = regime_mismatch = pos_mismatch = 0
    for idx, tgt in by_idx.items():
        if idx not in live.index:
            continue
        row = live.loc[idx]
        matched += 1
        if row["regime"] != tgt["regime"]:
            regime_mismatch += 1
        if abs(float(row["target_unit"]) - tgt["fp"]) > 1e-9:
            pos_mismatch += 1

    assert matched > 500, f"too few matched bars: {matched}"
    assert regime_mismatch == 0, f"regime mismatched on {regime_mismatch}/{matched} bars"
    assert pos_mismatch == 0, f"position mismatched on {pos_mismatch}/{matched} bars"
