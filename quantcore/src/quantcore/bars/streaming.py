"""Nine concrete streaming BarBuilder subclasses (S34 §3.AC3).

Each public class inherits directly from ``BarBuilder`` and delegates
to an internal engine that holds state and arithmetic. The engines
mirror the legacy numba kernels (``_threshold_bars_core``,
``_imbalance_bars_core``, ``_runs_bars_core``) one-to-one — same
recurrence, same operation order, same IEEE-754 result.

Live-deployable per S33 §5.D5: consumes only ``TradeEvent``; never
touches the research-only L3 book / order-state surface (the
``quantcore.book`` package and its order-state event variant).

Imbalance and runs builders REQUIRE explicit init values in their
config (``exp_imbalance_init``, ``exp_prob_buy_init``,
``exp_w_buy_init``, ``exp_w_sell_init``). The legacy's warmup-mean
fallback is lookahead (research-only per S33 §5.D7); the streaming
constructor raises ``ValueError`` if any required init is ``None``.
"""

# Suppressions tied to the file-internal composition pattern:
#   - reportImplicitOverride: typing.override is 3.12+; project pins 3.11.
#   - reportUnsafeMultipleInheritance: the two mixins (_InstrumentLock,
#     _TickRule) are explicitly initialised in each engine's __init__;
#     basedpyright cannot prove this statically.
#   - reportAny / reportUnannotatedClassAttribute: numpy reductions
#     surface as Any in strict mode; __slots__ declarations on
#     non-@final classes trigger the warning.
# pyright: reportImplicitOverride=false, reportUnsafeMultipleInheritance=false, reportAny=false, reportUnannotatedClassAttribute=false, reportMissingSuperCall=false

from __future__ import annotations

import math
from typing import Protocol, TypeGuard, runtime_checkable

import numpy as np

from quantcore.bars._streaming_abc import BarBuilder
from quantcore.bars.bars import ImbalanceConfig, RunsConfig
from quantcore.data import Bar, BarKind, BaseEvent


@runtime_checkable
class _TradeEventLike(Protocol):
    """Minimal structural contract for a trade event.

    Defined locally so quantcore stays independent of quantengine's
    ``protocols.py``.  Mirrors the five base attributes every trade
    event must carry plus the optional BBO / side enrichment.
    """

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float


# ===========================================================================
# OHLCV buffer
# ===========================================================================


class _OHLCVBuffer:
    """Accumulates per-bar OHLCV + microstructure fields; emits via
    numpy reductions matching the legacy ``aggregate_to_ohlcv``
    byte-for-byte for the OHLCV subset."""

    __slots__ = (
        "_ts",
        "_prices",
        "_volumes",
        "_last_sequence",
        "_spreads",
        "_imbalances",
        "_microprice_devs",
        "_signed_vols",
        "_signed_dollars",
        "_tick_dirs",
    )

    def __init__(self) -> None:
        self._ts: list[int] = []
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._last_sequence: int = 0
        self._spreads: list[float] = []
        self._imbalances: list[float] = []
        self._microprice_devs: list[float] = []
        self._signed_vols: list[float] = []
        self._signed_dollars: list[float] = []
        self._tick_dirs: list[float] = []

    def add(
        self,
        ts: int,
        price: float,
        volume: float,
        sequence: int,
        bid_px: float = float("nan"),
        ask_px: float = float("nan"),
        bid_sz: float = float("nan"),
        ask_sz: float = float("nan"),
        aggressor_side: int = 0,
    ) -> None:
        self._ts.append(ts)
        self._prices.append(price)
        self._volumes.append(volume)
        self._last_sequence = sequence

        has_bbo = math.isfinite(bid_px) and math.isfinite(ask_px)

        if has_bbo and ask_px > bid_px:
            mid = (bid_px + ask_px) * 0.5
            self._spreads.append(ask_px - bid_px)
            total_sz = bid_sz + ask_sz
            if total_sz > 0.0:
                imb = (bid_sz - ask_sz) / total_sz
                microprice = (ask_px * bid_sz + bid_px * ask_sz) / total_sz
                mpd = (microprice - mid) / mid if mid > 0.0 else float("nan")
                self._imbalances.append(imb)
                self._microprice_devs.append(mpd)
            else:
                self._imbalances.append(float("nan"))
                self._microprice_devs.append(float("nan"))
        else:
            self._spreads.append(float("nan"))
            self._imbalances.append(float("nan"))
            self._microprice_devs.append(float("nan"))

        side_f = float(aggressor_side)
        self._signed_vols.append(side_f * volume)
        self._signed_dollars.append(side_f * price * volume)
        self._tick_dirs.append(side_f)

    def reset(self) -> None:
        self._ts.clear()
        self._prices.clear()
        self._volumes.clear()
        self._last_sequence = 0
        self._spreads.clear()
        self._imbalances.clear()
        self._microprice_devs.clear()
        self._signed_vols.clear()
        self._signed_dollars.clear()
        self._tick_dirs.clear()

    def is_empty(self) -> bool:
        return not self._ts

    @staticmethod
    def _finite_stats(vals: list[float]) -> tuple[float, float, float]:
        """Return (mean, last, std) over finite elements; NaN if empty."""
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return float("nan"), float("nan"), float("nan")
        n = len(finite)
        mean = sum(finite) / n
        last = finite[-1]
        if n > 1:
            var = sum((v - mean) ** 2 for v in finite) / (n - 1)
            std = math.sqrt(var) if var > 0.0 else 0.0
        else:
            std = 0.0
        return mean, last, std

    def to_bar(self, instrument_id: int, kind: BarKind) -> Bar:
        prices = np.asarray(self._prices, dtype=np.float64)
        volumes = np.asarray(self._volumes, dtype=np.float64)
        total_volume = float(volumes.sum())
        dollar_volume = float((prices * volumes).sum())
        close_px = float(prices[-1])
        vwap = dollar_volume / total_volume if total_volume > 0.0 else close_px

        sp_mean, sp_last, sp_std = self._finite_stats(self._spreads)
        imb_mean, imb_last, imb_std = self._finite_stats(self._imbalances)
        mpd_mean, mpd_last, mpd_std = self._finite_stats(self._microprice_devs)

        sv_sum = sum(self._signed_vols)
        sd_sum = sum(self._signed_dollars)
        td_abs = sum(abs(v) for v in self._tick_dirs)
        td_sum = sum(self._tick_dirs)
        signed_tick_imb = td_sum / td_abs if td_abs > 0.0 else 0.0

        return Bar(
            ts_event=self._ts[-1],
            instrument_id=instrument_id,
            sequence=self._last_sequence,
            ts_open=self._ts[0],
            kind=kind,
            open=float(prices[0]),
            high=float(prices.max()),
            low=float(prices.min()),
            close=close_px,
            volume=total_volume,
            vwap=vwap,
            tick_count=len(prices),
            dollar_volume=dollar_volume,
            spread_mean=sp_mean,
            spread_last=sp_last,
            spread_std=sp_std,
            imbalance_mean=imb_mean,
            imbalance_last=imb_last,
            imbalance_std=imb_std,
            microprice_dev_mean=mpd_mean,
            microprice_dev_last=mpd_last,
            microprice_dev_std=mpd_std,
            signed_volume_sum=sv_sum,
            signed_dollar_sum=sd_sum,
            signed_tick_imbalance=signed_tick_imb,
        )


def _kind_weight(trade: object, kind: BarKind) -> float:
    """Unsigned per-tick weight matching `_standard_increments`.

    Accepts any object with ``price`` and ``size`` attributes (duck-typed
    to support structurally compatible feed events alongside concrete
    ``TradeEvent``).
    """
    if kind == BarKind.TICK:
        return 1.0
    size = float(getattr(trade, "size", 0.0))
    if kind == BarKind.VOLUME:
        return size
    return float(getattr(trade, "price", 0.0)) * size


# ===========================================================================
# Engines (internal, mirror the numba kernels 1:1)
# ===========================================================================


class _InstrumentLock:
    """Mixin-like helper: enforce single-instrument input.

    Intentionally no ``__slots__`` so it can coexist with ``_TickRule``
    in multiple inheritance (CPython forbids combining two slotted
    parent classes).
    """

    def __init__(self) -> None:
        self._instrument_id: int | None = None

    def _check_instrument(self, event_instrument: int) -> int:
        if self._instrument_id is None:
            self._instrument_id = int(event_instrument)
        elif event_instrument != self._instrument_id:
            raise ValueError(
                f"trade instrument_id {event_instrument} does not match "
                + f"builder instrument_id {self._instrument_id}"
            )
        return self._instrument_id


class _TickRule:
    """Mixin-like helper: cumulative tick-rule direction.

    Matches `_compute_tick_direction`:
        b[0] = +1
        b[i] = sign(p[i] - p[i-1]) if changed, else b[i-1]

    No ``__slots__`` — see ``_InstrumentLock`` note.
    """

    def __init__(self) -> None:
        self._last_price: float | None = None
        self._last_direction: int = 1

    def _tick_direction(self, price: float) -> int:
        """Pure tick-rule inference. Always updates internal state."""
        if self._last_price is None:
            d = 1
        else:
            diff = price - self._last_price
            if diff > 0.0:
                d = 1
            elif diff < 0.0:
                d = -1
            else:
                d = self._last_direction
        self._last_price = float(price)
        self._last_direction = d
        return d

    def _direction(self, price: float, aggressor_side: int = 0) -> int:
        """Resolve trade direction: prefer exchange side, tick-rule fallback.

        ``aggressor_side`` ∈ {+1, -1, 0}. Non-zero means the exchange
        reported the aggressor; use it directly. Zero (unknown / synthetic)
        falls back to the tick rule. Tick-rule state is always advanced so
        it stays warm for the next fallback.
        """
        tick_d = self._tick_direction(price)
        if aggressor_side == 1 or aggressor_side == -1:
            return aggressor_side
        return tick_d


def _is_trade_event(event: object) -> TypeGuard[_TradeEventLike]:
    """Duck-type check for the five attributes every trade event must carry.

    Accepts both quantcore ``TradeEvent`` instances and structurally
    compatible objects from external feeds (e.g. Databento's
    ``_DatabentoTradeEvent``).  Returns a ``TypeGuard`` so the
    narrowed ``_TradeEventLike`` type flows to callers without
    ``type: ignore``.
    """
    return isinstance(event, _TradeEventLike)


def _extract_bbo(event: object) -> tuple[float, float, float, float, int]:
    """Extract BBO + aggressor_side from an event; NaN/0 fallback.

    The ``or 0`` guard (S41 D2) resolves a ``None`` aggressor_side to 0
    (unknown → tick-rule fallback downstream) instead of raising
    ``TypeError`` on ``int(None)``. ``TradeEvent`` types the field as ``Side``,
    but a malformed / replay / second-vendor feed could yield ``None``; this
    keeps the LIVE path degrading gracefully rather than crashing. ``Side``
    IntEnums (±1) and an explicit 0 pass through unchanged (``x or 0`` returns
    ``x`` for any truthy ±1 and 0 for both ``0`` and ``None``).
    """
    return (
        getattr(event, "bid_px", float("nan")),
        getattr(event, "ask_px", float("nan")),
        getattr(event, "bid_sz", float("nan")),
        getattr(event, "ask_sz", float("nan")),
        int(getattr(event, "aggressor_side", 0) or 0),
    )


class _ThresholdEngine(_InstrumentLock, _TickRule):
    """Plain threshold bars — matches `_threshold_bars_core`.

    State:
        cum: running cumulative increment
        buffer: per-bar OHLCV accumulator
        (+ _TickRule state for the unknown-aggressor-side fallback)
    """

    __slots__ = ("_threshold", "_kind", "_cum", "_buffer")

    def __init__(self, threshold: float, kind: BarKind) -> None:
        _InstrumentLock.__init__(self)
        # S40 D1 (F3): the plain-dollar-bar path now mixes in _TickRule so an
        # unknown aggressor side (0) falls back to the tick rule, matching the
        # batch top_of_book._resolve_side_column behaviour. Without this the
        # streaming path stored 0 for unknown-side signed flow while batch used
        # the tick rule — a train/serve skew on the signed-flow features.
        _TickRule.__init__(self)
        if threshold <= 0.0:
            raise ValueError(f"threshold must be > 0; got {threshold}")
        self._threshold: float = float(threshold)
        self._kind: BarKind = kind
        self._cum: float = 0.0
        self._buffer: _OHLCVBuffer = _OHLCVBuffer()

    def feed(self, event: object) -> Bar | None:
        if not _is_trade_event(event):
            return None
        iid = self._check_instrument(event.instrument_id)

        self._cum += _kind_weight(event, self._kind)
        bid_px, ask_px, bid_sz, ask_sz, side = _extract_bbo(event)
        # Resolve direction with the tick-rule fallback (S40 D1) — exchange
        # side when known (±1), tick rule when unknown (0). Mirrors the
        # imbalance/runs engines, which already call self._direction(...).
        direction = self._direction(event.price, side)
        self._buffer.add(
            event.ts_event,
            event.price,
            event.size,
            event.sequence,
            bid_px=bid_px,
            ask_px=ask_px,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
            aggressor_side=direction,
        )

        if self._cum >= self._threshold:
            bar = self._buffer.to_bar(instrument_id=iid, kind=self._kind)
            self._buffer.reset()
            self._cum = 0.0
            return bar
        return None

    def flush(self) -> Bar | None:
        if self._buffer.is_empty() or self._instrument_id is None:
            return None
        bar = self._buffer.to_bar(instrument_id=self._instrument_id, kind=self._kind)
        self._buffer.reset()
        self._cum = 0.0
        return bar


class _ImbalanceEngine(_InstrumentLock, _TickRule):
    """Imbalance bars — matches `_imbalance_bars_core` + the
    `imbalance_bars` wrapper preparation step."""

    __slots__ = (
        "_kind",
        "_alpha_ticks",
        "_alpha_imb",
        "_min_abs",
        "_exp_T_min",
        "_exp_T_max",
        "_exp_T",
        "_exp_x",
        "_theta",
        "_ticks_in_bar",
        "_buffer",
    )

    def __init__(self, config: ImbalanceConfig, kind: BarKind) -> None:
        _InstrumentLock.__init__(self)
        _TickRule.__init__(self)

        if config.exp_imbalance_init is None:
            raise ValueError(
                "streaming imbalance bar builders require explicit "
                + "config.exp_imbalance_init (warmup-mean is lookahead "
                + "and research-only per S34 §5.D7)"
            )
        if config.ewma_span_ticks < 1 or config.ewma_span_imbalance < 1:
            raise ValueError("ewma_span_* must be >= 1")

        exp_x0 = float(config.exp_imbalance_init)
        if abs(exp_x0) < config.min_abs_exp_imbalance:
            exp_x0 = (
                float(config.min_abs_exp_imbalance)
                if exp_x0 >= 0.0
                else -float(config.min_abs_exp_imbalance)
            )

        self._kind: BarKind = kind
        self._alpha_ticks: float = 2.0 / (config.ewma_span_ticks + 1.0)
        self._alpha_imb: float = 2.0 / (config.ewma_span_imbalance + 1.0)
        self._min_abs: float = float(config.min_abs_exp_imbalance)
        self._exp_T_min: float = float(config.exp_num_ticks_min)
        self._exp_T_max: float = float(config.exp_num_ticks_max)

        self._exp_T: float = float(config.exp_num_ticks_init)
        self._exp_x: float = exp_x0

        self._theta: float = 0.0
        self._ticks_in_bar: int = 0
        self._buffer: _OHLCVBuffer = _OHLCVBuffer()

    def _signed_increment(self, trade: _TradeEventLike, direction: int) -> float:
        if self._kind == BarKind.TICK:
            return float(direction)
        if self._kind == BarKind.VOLUME:
            return float(direction) * float(trade.size)
        return float(direction) * float(trade.price) * float(trade.size)

    def feed(self, event: object) -> Bar | None:
        if not _is_trade_event(event):
            return None
        iid = self._check_instrument(event.instrument_id)

        bid_px, ask_px, bid_sz, ask_sz, side = _extract_bbo(event)
        direction = self._direction(event.price, side)
        x = self._signed_increment(event, direction)
        self._theta += x
        self._ticks_in_bar += 1

        self._buffer.add(
            event.ts_event,
            event.price,
            event.size,
            event.sequence,
            bid_px=bid_px,
            ask_px=ask_px,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
            aggressor_side=direction,  # s83 F1: resolved side (S40-D1), not raw
        )

        threshold = self._exp_T * abs(self._exp_x)
        if threshold < self._min_abs:
            threshold = self._min_abs

        if abs(self._theta) >= threshold:
            bar = self._buffer.to_bar(instrument_id=iid, kind=self._kind)

            realized_avg_x = self._theta / self._ticks_in_bar
            self._exp_T = (
                self._alpha_ticks * self._ticks_in_bar + (1.0 - self._alpha_ticks) * self._exp_T
            )
            if self._exp_T < self._exp_T_min:
                self._exp_T = self._exp_T_min
            elif self._exp_T > self._exp_T_max:
                self._exp_T = self._exp_T_max

            self._exp_x = self._alpha_imb * realized_avg_x + (1.0 - self._alpha_imb) * self._exp_x
            if abs(self._exp_x) < self._min_abs:
                self._exp_x = self._min_abs if self._exp_x >= 0.0 else -self._min_abs

            self._theta = 0.0
            self._ticks_in_bar = 0
            self._buffer.reset()
            return bar
        return None

    def flush(self) -> Bar | None:
        if self._buffer.is_empty() or self._instrument_id is None:
            return None
        bar = self._buffer.to_bar(instrument_id=self._instrument_id, kind=self._kind)
        self._buffer.reset()
        self._theta = 0.0
        self._ticks_in_bar = 0
        return bar


class _RunsEngine(_InstrumentLock, _TickRule):
    """Runs bars — matches `_runs_bars_core` + `_resolve_runs_initial_state`."""

    __slots__ = (
        "_kind",
        "_alpha_ticks",
        "_alpha_prob",
        "_alpha_weights",
        "_min_w",
        "_exp_T_min",
        "_exp_T_max",
        "_exp_T",
        "_exp_prob_buy",
        "_exp_w_buy",
        "_exp_w_sell",
        "_theta_buy",
        "_theta_sell",
        "_ticks_in_bar",
        "_n_buy",
        "_n_sell",
        "_sum_w_buy_in_bar",
        "_sum_w_sell_in_bar",
        "_buffer",
    )

    def __init__(self, config: RunsConfig, kind: BarKind) -> None:
        _InstrumentLock.__init__(self)
        _TickRule.__init__(self)

        if config.exp_prob_buy_init is None:
            raise ValueError(
                "streaming runs bar builders require explicit "
                + "config.exp_prob_buy_init (warmup-mean is lookahead "
                + "and research-only per S34 §5.D7)"
            )
        if config.exp_w_buy_init is None:
            raise ValueError(
                "streaming runs bar builders require explicit " + "config.exp_w_buy_init"
            )
        if config.exp_w_sell_init is None:
            raise ValueError(
                "streaming runs bar builders require explicit " + "config.exp_w_sell_init"
            )
        if config.ewma_span_ticks < 1 or config.ewma_span_prob < 1 or config.ewma_span_weights < 1:
            raise ValueError("ewma_span_* must be >= 1")

        # Mirror _resolve_runs_initial_state for explicit-init path.
        prob_buy = float(config.exp_prob_buy_init)
        if prob_buy < 0.0:
            prob_buy = 0.0
        elif prob_buy > 1.0:
            prob_buy = 1.0

        w_buy = float(config.exp_w_buy_init)
        if w_buy < config.min_abs_exp_weight:
            w_buy = float(config.min_abs_exp_weight)

        w_sell = float(config.exp_w_sell_init)
        if w_sell < config.min_abs_exp_weight:
            w_sell = float(config.min_abs_exp_weight)

        exp_T = float(config.exp_num_ticks_init)
        if exp_T < config.exp_num_ticks_min:
            exp_T = float(config.exp_num_ticks_min)
        elif exp_T > config.exp_num_ticks_max:
            exp_T = float(config.exp_num_ticks_max)

        self._kind: BarKind = kind
        self._alpha_ticks: float = 2.0 / (config.ewma_span_ticks + 1.0)
        self._alpha_prob: float = 2.0 / (config.ewma_span_prob + 1.0)
        self._alpha_weights: float = 2.0 / (config.ewma_span_weights + 1.0)
        self._min_w: float = float(config.min_abs_exp_weight)
        self._exp_T_min: float = float(config.exp_num_ticks_min)
        self._exp_T_max: float = float(config.exp_num_ticks_max)

        self._exp_T: float = exp_T
        self._exp_prob_buy: float = prob_buy
        self._exp_w_buy: float = w_buy
        self._exp_w_sell: float = w_sell

        self._theta_buy: float = 0.0
        self._theta_sell: float = 0.0
        self._ticks_in_bar: int = 0
        self._n_buy: int = 0
        self._n_sell: int = 0
        self._sum_w_buy_in_bar: float = 0.0
        self._sum_w_sell_in_bar: float = 0.0
        self._buffer: _OHLCVBuffer = _OHLCVBuffer()

    def feed(self, event: object) -> Bar | None:
        if not _is_trade_event(event):
            return None
        iid = self._check_instrument(event.instrument_id)

        bid_px, ask_px, bid_sz, ask_sz, side = _extract_bbo(event)
        b = self._direction(event.price, side)
        w = _kind_weight(event, self._kind)

        if b == 1:
            self._theta_buy += w
            self._sum_w_buy_in_bar += w
            self._n_buy += 1
        else:
            self._theta_sell += w
            self._sum_w_sell_in_bar += w
            self._n_sell += 1

        self._ticks_in_bar += 1
        self._buffer.add(
            event.ts_event,
            event.price,
            event.size,
            event.sequence,
            bid_px=bid_px,
            ask_px=ask_px,
            bid_sz=bid_sz,
            ask_sz=ask_sz,
            aggressor_side=b,  # s83 F1: resolved side (S40-D1), not raw
        )

        theta = self._theta_buy if self._theta_buy >= self._theta_sell else self._theta_sell

        side_term_buy = self._exp_prob_buy * self._exp_w_buy
        side_term_sell = (1.0 - self._exp_prob_buy) * self._exp_w_sell
        threshold = (
            self._exp_T * side_term_buy
            if side_term_buy >= side_term_sell
            else self._exp_T * side_term_sell
        )
        if threshold < self._min_w:
            threshold = self._min_w

        if theta >= threshold:
            bar = self._buffer.to_bar(instrument_id=iid, kind=self._kind)

            self._exp_T = (
                self._alpha_ticks * self._ticks_in_bar + (1.0 - self._alpha_ticks) * self._exp_T
            )
            if self._exp_T < self._exp_T_min:
                self._exp_T = self._exp_T_min
            elif self._exp_T > self._exp_T_max:
                self._exp_T = self._exp_T_max

            realized_prob_buy = self._n_buy / self._ticks_in_bar
            self._exp_prob_buy = (
                self._alpha_prob * realized_prob_buy + (1.0 - self._alpha_prob) * self._exp_prob_buy
            )
            if self._exp_prob_buy < 0.0:
                self._exp_prob_buy = 0.0
            elif self._exp_prob_buy > 1.0:
                self._exp_prob_buy = 1.0

            if self._n_buy > 0:
                realized_w_buy = self._sum_w_buy_in_bar / self._n_buy
                self._exp_w_buy = (
                    self._alpha_weights * realized_w_buy
                    + (1.0 - self._alpha_weights) * self._exp_w_buy
                )
                if self._exp_w_buy < self._min_w:
                    self._exp_w_buy = self._min_w

            if self._n_sell > 0:
                realized_w_sell = self._sum_w_sell_in_bar / self._n_sell
                self._exp_w_sell = (
                    self._alpha_weights * realized_w_sell
                    + (1.0 - self._alpha_weights) * self._exp_w_sell
                )
                if self._exp_w_sell < self._min_w:
                    self._exp_w_sell = self._min_w

            self._theta_buy = 0.0
            self._theta_sell = 0.0
            self._ticks_in_bar = 0
            self._n_buy = 0
            self._n_sell = 0
            self._sum_w_buy_in_bar = 0.0
            self._sum_w_sell_in_bar = 0.0
            self._buffer.reset()
            return bar
        return None

    def flush(self) -> Bar | None:
        if self._buffer.is_empty() or self._instrument_id is None:
            return None
        bar = self._buffer.to_bar(instrument_id=self._instrument_id, kind=self._kind)
        self._buffer.reset()
        self._theta_buy = 0.0
        self._theta_sell = 0.0
        self._ticks_in_bar = 0
        self._n_buy = 0
        self._n_sell = 0
        self._sum_w_buy_in_bar = 0.0
        self._sum_w_sell_in_bar = 0.0
        return bar


# ===========================================================================
# Public bar builders — direct BarBuilder subclasses (AC3 grep contract)
# ===========================================================================


class TickBarBuilder(BarBuilder):
    def __init__(self, threshold: int) -> None:
        self._engine: _ThresholdEngine = _ThresholdEngine(float(threshold), BarKind.TICK)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class VolumeBarBuilder(BarBuilder):
    def __init__(self, threshold: float) -> None:
        self._engine: _ThresholdEngine = _ThresholdEngine(float(threshold), BarKind.VOLUME)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class DollarBarBuilder(BarBuilder):
    def __init__(self, threshold: float) -> None:
        self._engine: _ThresholdEngine = _ThresholdEngine(float(threshold), BarKind.DOLLAR)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class TickImbalanceBarBuilder(BarBuilder):
    def __init__(self, config: ImbalanceConfig) -> None:
        self._engine: _ImbalanceEngine = _ImbalanceEngine(config, BarKind.TICK)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class VolumeImbalanceBarBuilder(BarBuilder):
    def __init__(self, config: ImbalanceConfig) -> None:
        self._engine: _ImbalanceEngine = _ImbalanceEngine(config, BarKind.VOLUME)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class DollarImbalanceBarBuilder(BarBuilder):
    def __init__(self, config: ImbalanceConfig) -> None:
        self._engine: _ImbalanceEngine = _ImbalanceEngine(config, BarKind.DOLLAR)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class TickRunsBarBuilder(BarBuilder):
    def __init__(self, config: RunsConfig) -> None:
        self._engine: _RunsEngine = _RunsEngine(config, BarKind.TICK)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class VolumeRunsBarBuilder(BarBuilder):
    def __init__(self, config: RunsConfig) -> None:
        self._engine: _RunsEngine = _RunsEngine(config, BarKind.VOLUME)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()


class DollarRunsBarBuilder(BarBuilder):
    def __init__(self, config: RunsConfig) -> None:
        self._engine: _RunsEngine = _RunsEngine(config, BarKind.DOLLAR)

    def on_event(self, event: BaseEvent) -> Bar | None:
        return self._engine.feed(event)

    def flush(self) -> Bar | None:
        return self._engine.flush()
