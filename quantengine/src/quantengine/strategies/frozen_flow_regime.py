"""Live streaming strategy for the frozen flow_only TXN alpha (s44).

Runs the s43-frozen dual-horizon GradientBoostingClassifier package online through
the quantengine StreamingEngine: on each closed dollar bar it reproduces the
research feature vector (s41 ``bar_flow_ratios`` + s42 ``OnlineRollingFlow``),
scores both horizons, runs the per-horizon expected-return position FSM, applies
the spread-regime gate, and submits integer-share MARKET orders against whatever
broker the engine wraps.

This is the consumer side of the architecture invariant — it NEVER trains; it
loads a frozen package via ``quantcore.models.FrozenFlowRegimeArtifact.read`` and
calls ``predict_proba``. It satisfies the ``StreamingStrategy`` Protocol
structurally (three sync callbacks, no base class).

FRAMING: the realistic-exit replay of this alpha is NEGATIVE. This strategy
exists to prove TRAIN/SERVE FIDELITY (the live signal/position path reproduces the
frozen research replay byte-for-byte) — NOT as a profitability claim.

Spread gate (operator-locked, s44): the research gate uses ``rel_spread_last``
(relative, per-tick spread/mid). The streaming ``Bar`` carries only the absolute
``spread_last`` and no last-tick mid, so the live gate reconstructs
``rel_spread_bps = spread_last / close * 1e4``. On real TXN data this proxy
reproduces the research regime classification on 0/1810 bars flipped (s43 report);
the s44 parity test compares against a proxy-computed target so equivalence is
exact by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np

from quantcore.features.online_rolling import OnlineRollingFlow
from quantcore.features.top_of_book import bar_flow_ratios
from quantcore.models import FrozenFlowRegimeArtifact
from quantengine.contracts.orders import Order, OrderType
from quantengine.runtime.streaming.protocols import (
    BarLike,
    CUSUMEvent,
    StreamContext,
    SyncBrokerFacade,
)

_ROLL_SUFFIXES = ("_sum5", "_sum10", "_sum20", "_z20")
_ROLL_KEYS = ("sum5", "sum10", "sum20", "z20")


class _FlowSpreadBar(Protocol):
    """The flow + spread fields ``BarLike`` omits but the concrete ``Bar`` carries.

    ``BarLike`` (protocols.py) lists only OHLCV; the live ``Bar`` from the s40
    streaming builder also has these microstructure fields. We cast to this
    Protocol at the single read site so static typing stays honest without
    editing the locked streaming substrate.
    """

    close: float
    volume: float
    dollar_volume: float
    spread_last: float
    signed_volume_sum: float
    signed_dollar_sum: float
    signed_tick_imbalance: float


def _position_step(er: float, entry: float, exit_th: float, flip: float, prev: float) -> float:
    """One iteration of alpha_research.walk_forward._position_logic (byte-identical).

    flat -> long if er>entry; flat -> short if er<-entry; long -> short if
    er<-flip; long -> flat if er<exit_th; short -> long if er>flip; short -> flat
    if er>-exit_th. Otherwise hold.
    """
    if prev == 0.0:
        if er > entry:
            prev = 1.0
        elif er < -entry:
            prev = -1.0
    elif prev == 1.0:
        if er < -flip:
            prev = -1.0
        elif er < exit_th:
            prev = 0.0
    elif prev == -1.0:
        if er > flip:
            prev = 1.0
        elif er > -exit_th:
            prev = 0.0
    return prev


class FrozenFlowRegimeStrategy:
    """Dual-horizon flow_only live strategy. Satisfies ``StreamingStrategy``.

    Parameters
    ----------
    artifact_dir : Path | str
        Directory of the s43 ``FrozenFlowRegimeArtifact`` (manifest + joblib models).
    ticker : str
        Instrument to trade (must match the engine's single instrument).
    shares_per_unit : int
        Integer shares per unit position; live ``target_shares = round(unit) *
        shares_per_unit`` (research positions are unit {-1,0,+1}).
    submit_timeout_s : float
        Per-order broker timeout.
    record : bool
        When True, append a per-bar signal dict to ``self.records`` (used by the
        parity test). No effect on order flow.
    """

    def __init__(
        self,
        artifact_dir: Path | str,
        *,
        ticker: str = "TXN",
        shares_per_unit: int = 100,
        submit_timeout_s: float = 5.0,
        record: bool = False,
        score_bar_indices: set[int] | None = None,
    ) -> None:
        self._artifact, self._models = FrozenFlowRegimeArtifact.read(artifact_dir)
        self._ticker = ticker
        self._shares_per_unit = int(shares_per_unit)
        self._timeout = float(submit_timeout_s)
        self._record = record
        # CUSUM-event gating (s44 v1): research scores/steps the FSM ONLY at
        # cusum_filter event bars, not every bar. The position FSM is path-
        # dependent on which bars it sees, so to reproduce research the strategy
        # must score on the SAME event grid. When `score_bar_indices` is given,
        # the strategy predicts + steps the FSM only at those bar ordinals
        # (rolling state still updates every bar, matching build_flow). When
        # None, it scores every bar — the live default, which requires the engine
        # OnlineCUSUMFilter to reproduce research's cusum_filter event set (a
        # documented follow-up; the two CUSUM implementations are not yet
        # event-equivalent). See s44 plan decision 5.E.
        self._score_idx = score_bar_indices
        self._bar_count = -1  # 0-based ordinal of bars seen (matches frozen bar_index)
        self.records: list[dict[str, float | str]] = []

        # One OnlineRollingFlow per raw flow column (s42).
        self._roll = {col: OnlineRollingFlow() for col in self._artifact.raw_flow_cols}

        # Per-horizon prev FSM position; advanced EVERY scored bar (research
        # advances both horizons' FSMs and masks the emitted target post-hoc).
        self._prev: dict[str, float] = {h.name: 0.0 for h in self._artifact.horizons}

        # Cache per-horizon scoring scalars + the proba-column index for each class.
        self._spec = {}
        for h in self._artifact.horizons:
            col_idx = {c: i for i, c in enumerate(h.classes)}
            self._spec[h.name] = {
                "model": self._models[h.name],
                "mu": h.mu,
                "col_idx": col_idx,
                "entry": h.entry_th,
                "exit": h.exit_th,
                "flip": h.flip_th,
                "lo": h.spread_lo_bps,
                "hi": h.spread_hi_bps,
            }
        # regime_priority is the order we resolve the gate (tight before moderate).
        self._priority = list(self._artifact.regime_priority)

    # ------------------------------------------------------------------
    # StreamingStrategy callbacks
    # ------------------------------------------------------------------
    def on_bar(
        self,
        ts: int,
        bar: BarLike,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        fb = cast(_FlowSpreadBar, cast(object, bar))
        self._bar_count += 1  # ordinal of this bar in the full series (join key)
        ratios = bar_flow_ratios(fb)  # {signed_vol_imb, signed_dollar_imb, signed_tick_imb}

        # Update rolling state for EVERY bar (warmup must accrue) and assemble the
        # 15-vector in the artifact's exact feature_order.
        roll_out = {
            col: self._roll[col].update(ratios[col]) for col in self._artifact.raw_flow_cols
        }

        # CUSUM-event gate: rolling state above accrues EVERY bar (matches
        # build_flow), but prediction + FSM step happen only at scored events.
        if self._score_idx is not None and self._bar_count not in self._score_idx:
            return

        feat_by_name: dict[str, float] = {}
        for col in self._artifact.raw_flow_cols:
            feat_by_name[col] = ratios[col]
            for suffix, key in zip(_ROLL_SUFFIXES, _ROLL_KEYS):
                feat_by_name[col + suffix] = roll_out[col][key]
        x = np.array(
            [feat_by_name[name] for name in self._artifact.feature_order], dtype=np.float64
        )

        # Research drops any row with a NaN feature (X.notna().all(axis=1)); sum{w}
        # is NaN during warmup. Mirror exactly: hold FSM state, emit nothing.
        if not np.all(np.isfinite(x)):
            if self._record:
                self._record_bar(ts, fb, regime=None, target_unit=0.0, ers={})
            return

        xrow = x.reshape(1, -1)
        ers: dict[str, float] = {}
        for name in self._spec:
            sp = self._spec[name]
            proba = sp["model"].predict_proba(xrow)[0]
            er = 0.0
            for c in (-1, 0, 1):
                if c in sp["col_idx"]:
                    er += float(proba[sp["col_idx"][c]]) * float(sp["mu"].get(c, 0.0))
            ers[name] = er
            # Advance THIS horizon's FSM every scored bar.
            self._prev[name] = _position_step(
                er, sp["entry"], sp["exit"], sp["flip"], self._prev[name]
            )

        # Spread gate (proxy): pick the first priority regime whose bounds contain
        # the proxy rel-spread; emit that horizon's FSM position, else flat.
        s_bps = self._rel_spread_bps(fb)
        regime = self._regime_for(s_bps)
        target_unit = self._prev[regime] if regime is not None else 0.0

        if self._record:
            self._record_bar(ts, fb, regime=regime, target_unit=target_unit, ers=ers)

        # Map unit -> integer shares; diff against broker truth; submit MARKET.
        target_shares = int(round(target_unit)) * self._shares_per_unit
        pos = broker.get_position(self._ticker)
        cur = pos.quantity if pos is not None else 0
        delta = target_shares - cur
        if delta != 0:
            broker.submit_order(
                Order.new(self._ticker, delta, OrderType.MARKET),
                timeout=self._timeout,
            )

    def on_cusum(
        self, ts: int, event: CUSUMEvent, ctx: StreamContext, broker: SyncBrokerFacade
    ) -> None:
        # v1 scores every bar in on_bar; CUSUM gating is a documented follow-up.
        return None

    def on_vol(self, ts: int, sigma: float, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rel_spread_bps(self, bar: _FlowSpreadBar) -> float:
        """Proxy relative spread in bps: absolute spread_last / close * 1e4.

        The streaming Bar has no last-tick mid; close (last trade price) is the
        available denominator. Validated on TXN: 0/1810 regime flips vs the true
        per-tick rel_spread_last (s43 report)."""
        close = float(bar.close)
        sl = float(bar.spread_last)
        if not (close > 0.0) or not np.isfinite(sl):
            return float("nan")
        return sl / close * 10_000.0

    def _regime_for(self, s_bps: float) -> str | None:
        if not np.isfinite(s_bps):
            return None
        for name in self._priority:
            sp = self._spec[name]
            if sp["lo"] < s_bps <= sp["hi"]:
                return name
        return None

    def _record_bar(
        self,
        ts: int,
        bar: _FlowSpreadBar,
        *,
        regime: str | None,
        target_unit: float,
        ers: dict[str, float],
    ) -> None:
        rec: dict[str, float | str] = {
            "ts": int(ts),
            "ts_event": int(cast(BarLike, cast(object, bar)).ts_event),
            "bar_index": int(self._bar_count),
            "regime": regime if regime is not None else "none",
            "target_unit": float(target_unit),
            "spread_proxy_bps": self._rel_spread_bps(bar),
        }
        for name, er in ers.items():
            rec[f"er_{name}"] = float(er)
        self.records.append(rec)


__all__ = ["FrozenFlowRegimeStrategy"]
