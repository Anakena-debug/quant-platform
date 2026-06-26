"""Top-of-book (L1/TBBO) microstructure features aggregated to bars.

Computes tick-level BBO features from TBBO data and aggregates them
to bar boundaries defined by close_indices (from threshold_bar_close_indices).

Public API:
    top_of_book_features(df, ...) -> pd.DataFrame (tick-level)
    aggregate_microstructure_by_close_indices(features, close_indices) -> pd.DataFrame
    dollar_bars_with_microstructure(df, threshold, ...) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from quantcore.bars.bars import (
    aggregate_to_ohlcv,
    threshold_bar_close_indices,
)


def top_of_book_features(
    df: pd.DataFrame,
    bid_px_col: str = "bid_px_00",
    ask_px_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
) -> pd.DataFrame:
    """Compute tick-level top-of-book microstructure features.

    Returns a DataFrame aligned with df's index containing:
    mid, spread, rel_spread, quoted_imbalance, microprice, microprice_dev.
    Rows with invalid BBO (zero mid or zero total size) are NaN.
    """
    bid = df[bid_px_col].astype(np.float64).values
    ask = df[ask_px_col].astype(np.float64).values
    bid_sz = df[bid_sz_col].astype(np.float64).values
    ask_sz = df[ask_sz_col].astype(np.float64).values

    valid_book = (ask > bid) & np.isfinite(bid) & np.isfinite(ask)

    mid = np.where(valid_book, (bid + ask) / 2.0, np.nan)
    spread = np.where(valid_book, ask - bid, np.nan)
    total_sz = bid_sz + ask_sz

    safe_mid = np.where(mid > 0, mid, np.nan)
    safe_total = np.where((total_sz > 0) & valid_book, total_sz, np.nan)

    rel_spread = spread / safe_mid
    imbalance = (bid_sz - ask_sz) / safe_total
    microprice = (ask * bid_sz + bid * ask_sz) / safe_total
    microprice_dev = (microprice - mid) / safe_mid

    return pd.DataFrame(
        {
            "mid": mid,
            "spread": spread,
            "rel_spread": rel_spread,
            "quoted_imbalance": imbalance,
            "microprice": microprice,
            "microprice_dev": microprice_dev,
        },
        index=df.index,
    )


def _tick_rule_direction(price: np.ndarray) -> np.ndarray:
    """Tick rule: sign(Δp), carry forward on zero, first tick = +1.

    S41 D4 fix: the seed ``tick_dir[0] = +1`` is applied BEFORE the carry-forward
    loop. The pre-S41 code seeded it AFTER, so a LEADING run of zero-ticks (equal
    prices before the first move) carried forward the original ``0`` instead of
    the seed — e.g. ``[100,100,101]`` -> ``[1,0,1]`` (tick 1 wrongly 0) and
    ``[50,50,50]`` -> ``[1,0,0]``. A direction of 0 zeroes that tick's signed
    flow. Seeding first makes a leading zero-run resolve to +1, matching the
    streaming ``_TickRule`` and this docstring's stated contract. (The Hypothesis
    fuzz in test_bar_flow_ratios_parity surfaced this; it is material on
    market-open / post-halt / illiquid sequences, not the 1/300k first-tick-only
    case originally assumed.)
    """
    tick_dir = np.sign(np.diff(price, prepend=price[0]))
    if len(tick_dir) > 0 and tick_dir[0] == 0:
        tick_dir[0] = 1.0  # seed BEFORE carry-forward so a leading zero-run -> +1
    for i in range(1, len(tick_dir)):
        if tick_dir[i] == 0:
            tick_dir[i] = tick_dir[i - 1]
    return tick_dir


# DBN spec: ``side`` is the side of the AGGRESSOR for trades. 'B' (bid side)
# = buy aggressor → +1; 'A' (ask side) = sell aggressor → -1; 'N' → tick-rule
# fallback. Matches ``quantcore.data.events.Side`` (BID=+1, ASK=-1) and the
# quantengine adapter ``databento_feed._side_to_aggressor``. (s83 F11: the
# pre-s83 map was inverted — 'A' mis-read as "lifted offer = buyer".)
_SIDE_MAP = {"B": 1.0, "A": -1.0}


def _resolve_side_column(
    df: pd.DataFrame,
    side_col: str,
    price: np.ndarray,
) -> np.ndarray:
    """Map exchange side to ±1, tick-rule fallback for unknowns.

    Databento DBN ``side`` is the aggressor side: 'B' (bid) = buy
    aggressor → +1; 'A' (ask) = sell aggressor → -1. Any other value
    (e.g. 'N', NaN, empty) falls back to the tick rule for that row.
    """
    raw = df[side_col]
    mapped = raw.map(_SIDE_MAP)
    tick_dir = _tick_rule_direction(price)
    result = np.where(mapped.notna(), mapped.astype(np.float64).values, tick_dir)
    return result


def signed_flow_features(
    df: pd.DataFrame,
    price_col: str = "price",
    volume_col: str = "size",
    side_col: str | None = None,
) -> pd.DataFrame:
    """Compute tick-level signed-flow features.

    When ``side_col`` is provided and present in ``df``, the exchange-
    reported aggressor side is used (±1).  Unknown / missing values
    fall back to the tick rule per row.  When ``side_col`` is ``None``
    or the column is absent, pure tick rule is used throughout.
    """
    price = df[price_col].astype(np.float64).values
    volume = df[volume_col].astype(np.float64).values

    if side_col is not None and side_col in df.columns:
        direction = _resolve_side_column(df, side_col, price)
    else:
        direction = _tick_rule_direction(price)

    return pd.DataFrame(
        {
            "tick_dir": direction,
            "signed_volume": direction * volume,
            "signed_dollar": direction * price * volume,
        },
        index=df.index,
    )


def aggregate_signed_flow_by_close_indices(
    flow: pd.DataFrame,
    close_indices: np.ndarray,
) -> pd.DataFrame:
    """Aggregate tick-level signed-flow features to bar boundaries.

    Returns per-bar: signed_vol_imb, signed_dollar_imb, signed_tick_imb.
    Each is the signed sum divided by absolute sum (range [-1, +1]).
    """
    if len(close_indices) == 0:
        return pd.DataFrame()

    timestamps = flow.index
    rows: list[dict] = []
    start = 0

    for end in close_indices:
        sl = slice(start, end + 1)
        sv = flow["signed_volume"].values[sl]
        sd = flow["signed_dollar"].values[sl]
        td = flow["tick_dir"].values[sl]

        sv_abs = np.abs(sv).sum()
        sd_abs = np.abs(sd).sum()
        td_abs = np.abs(td).sum()

        rows.append(
            {
                "timestamp": timestamps[end],
                "signed_vol_imb": float(sv.sum() / sv_abs) if sv_abs > 0 else 0.0,
                "signed_dollar_imb": float(sd.sum() / sd_abs) if sd_abs > 0 else 0.0,
                "signed_tick_imb": float(td.sum() / td_abs) if td_abs > 0 else 0.0,
            }
        )
        start = end + 1

    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    return out


class _FlowBar(Protocol):
    """Structural type for :func:`bar_flow_ratios` — the streaming ``Bar``
    fields it reads. ``quantcore.data.Bar`` satisfies this duck-typically."""

    volume: float
    dollar_volume: float
    signed_volume_sum: float
    signed_dollar_sum: float
    signed_tick_imbalance: float


def bar_flow_ratios(bar: _FlowBar) -> dict[str, float]:
    """Convert a streaming ``Bar``'s signed-flow SUMS to the batch RATIOS.

    The streaming ``Bar`` (``quantcore.data.Bar``) emits per-bar signed-flow
    SUMS — ``signed_volume_sum``, ``signed_dollar_sum`` — plus the already-ratio
    ``signed_tick_imbalance``. The models trained in research consume the batch
    RATIOS produced by :func:`aggregate_signed_flow_by_close_indices`:
    ``signed_vol_imb``, ``signed_dollar_imb``, ``signed_tick_imb``. This is the
    single canonical Bar→ratio adapter, co-located with the batch definition so
    the two cannot drift (S41 D1).

    Mapping::

        signed_vol_imb    = signed_volume_sum / volume          (0.0 if volume       <= 0)
        signed_dollar_imb = signed_dollar_sum / dollar_volume   (0.0 if dollar_volume<= 0)
        signed_tick_imb   = signed_tick_imbalance               (already a ratio)

    Exactness (load-bearing, S40-dependent)
    ---------------------------------------
    Batch computes ``signed_vol_imb = Σ(dir·vol) / Σ|dir·vol|``. The adapter uses
    ``signed_volume_sum / volume``. These are EQUAL iff ``|dir| == 1`` for every
    tick — i.e. no zero-direction ticks — because then ``Σ|dir·vol| = Σvol =
    volume`` (vol ≥ 0) and likewise ``Σ|dir·price·vol| = Σprice·vol =
    dollar_volume`` (price > 0). The S40 fix guarantees this: the streaming
    ``_ThresholdEngine`` tick-rule-resolves every side to ±1, never 0. The
    parity test (``test_bar_flow_ratios_parity``) enforces the equality and
    therefore the invariant.

    Empty / thin bar convention
    ---------------------------
    A non-positive denominator yields ``0.0``, matching batch
    ``aggregate_signed_flow_by_close_indices`` (``... if sv_abs > 0 else 0.0``).
    Dollar bars always close with ``volume > 0``; a time bar that closes with no
    trades (``volume == 0``) yields ``0.0`` here exactly as batch would. Callers
    wanting a NaN sentinel for "no flow" must special-case empty bars upstream.

    Parameters
    ----------
    bar : object
        Any object exposing ``signed_volume_sum``, ``signed_dollar_sum``,
        ``signed_tick_imbalance``, ``volume``, ``dollar_volume`` (the streaming
        ``Bar`` dataclass; duck-typed so a structurally-compatible row works).

    Returns
    -------
    dict[str, float]
        ``{"signed_vol_imb", "signed_dollar_imb", "signed_tick_imb"}``.
    """
    volume = float(bar.volume)
    dollar_volume = float(bar.dollar_volume)
    signed_volume_sum = float(bar.signed_volume_sum)
    signed_dollar_sum = float(bar.signed_dollar_sum)
    signed_tick_imbalance = float(bar.signed_tick_imbalance)

    return {
        "signed_vol_imb": (signed_volume_sum / volume) if volume > 0 else 0.0,
        "signed_dollar_imb": (signed_dollar_sum / dollar_volume) if dollar_volume > 0 else 0.0,
        "signed_tick_imb": signed_tick_imbalance,
    }


def aggregate_microstructure_by_close_indices(
    features: pd.DataFrame,
    close_indices: np.ndarray,
) -> pd.DataFrame:
    """Aggregate tick-level microstructure features to bar boundaries.

    Uses the same close_indices that aggregate_to_ohlcv consumes.
    For each bar interval [start, end], computes:
        mean, last, std, min, max for spread/rel_spread/imbalance/microprice_dev
        change (last - first) for spread/imbalance/microprice_dev
    """
    if len(close_indices) == 0:
        return pd.DataFrame()

    agg_cols = ["spread", "rel_spread", "quoted_imbalance", "microprice_dev"]
    change_cols = ["spread", "quoted_imbalance", "microprice_dev"]

    timestamps = features.index
    rows: list[dict] = []
    start = 0

    for end in close_indices:
        row: dict[str, float] = {"timestamp": timestamps[end]}
        for col in agg_cols:
            vals = features[col].values[start : end + 1]
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                for sfx in ("_mean", "_last", "_std", "_min", "_max"):
                    row[f"{col}{sfx}"] = np.nan
            else:
                row[f"{col}_mean"] = float(np.mean(finite))
                row[f"{col}_last"] = float(finite[-1])
                row[f"{col}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
                row[f"{col}_min"] = float(np.min(finite))
                row[f"{col}_max"] = float(np.max(finite))

        for col in change_cols:
            vals = features[col].values[start : end + 1]
            finite = vals[np.isfinite(vals)]
            if len(finite) >= 2:
                row[f"{col}_change"] = float(finite[-1] - finite[0])
            else:
                row[f"{col}_change"] = 0.0

        rows.append(row)
        start = end + 1

    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out = out.set_index("timestamp")
    return out


def dollar_bars_with_microstructure(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = "price",
    volume_col: str = "size",
    bid_px_col: str = "bid_px_00",
    ask_px_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
    side_col: str | None = "side",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    """Build dollar bars with microstructure features in one pass.

    When ``side_col`` names a column present in ``df`` (Databento TBBO
    provides ``"side"`` with values ``'A'``/``'B'``), exchange-reported
    aggressor side is used for signed-flow features.  Unknown values
    fall back to the tick rule per row.  Pass ``None`` to force pure
    tick rule.
    """
    # s83 F16 (F2 family): the micro/flow aggregations emit one row per
    # COMPLETED bar, so a partial tail desynced `micro_agg.index = ohlcv.index`
    # (k vs k+1 ValueError) and the zero-completed-bars case silently dropped
    # the partial row. Refuse loudly until the aggregators emit a padded
    # partial row.
    if include_partial_last_bar:
        raise NotImplementedError(
            "include_partial_last_bar=True is not supported for "
            "dollar_bars_with_microstructure: microstructure/flow aggregates "
            "exist only for completed bars."
        )
    close_indices = threshold_bar_close_indices(
        df,
        threshold,
        kind="dollar",
        price_col=price_col,
        volume_col=volume_col,
    )

    ohlcv = aggregate_to_ohlcv(
        df,
        close_indices,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )

    tick_features = top_of_book_features(
        df,
        bid_px_col=bid_px_col,
        ask_px_col=ask_px_col,
        bid_sz_col=bid_sz_col,
        ask_sz_col=ask_sz_col,
    )
    micro_agg = aggregate_microstructure_by_close_indices(tick_features, close_indices)
    micro_agg.index = ohlcv.index

    flow = signed_flow_features(
        df,
        price_col=price_col,
        volume_col=volume_col,
        side_col=side_col,
    )
    flow_agg = aggregate_signed_flow_by_close_indices(flow, close_indices)
    flow_agg.index = ohlcv.index

    return pd.concat([ohlcv, micro_agg, flow_agg], axis=1)


__all__ = [
    "aggregate_microstructure_by_close_indices",
    "aggregate_signed_flow_by_close_indices",
    "bar_flow_ratios",
    "dollar_bars_with_microstructure",
    "signed_flow_features",
    "top_of_book_features",
]
