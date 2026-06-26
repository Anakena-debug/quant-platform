"""quantlake — corporate-action (split) detection + back-adjustment.

Raw market-data feeds (Databento, exchange feeds) ship UNADJUSTED prices and no
corporate-action feed, so splits are detected from the raw OHLCV itself and the price series
is back-adjusted. A split applies at the OPEN, so its signature is a clean overnight gap
``prev_close/open ≈ ratio`` followed by a NORMAL intraday move (``|close/open - 1| <
INTRADAY_MAX``) — this separates a split from a crash, which moves intraday. Forward splits
are volume-confirmed (shares jump ≈ ratio). Back-adjustment divides historical prices by the
product of all FUTURE split ratios, leaving recent prices unchanged (the standard convention).

Re-homed from quantdata in s79 (block item 7); quantdata.corp_actions is now a pure re-export
shim. Pure functions, no network.

LIMITATION (disclosed): detection is heuristic — there is no authoritative split calendar, so
a residual extreme-return rate remains (the adjustment-error bound). A clean ~50% overnight
crash can be mis-flagged as a 2:1 split, and exotic ratios outside ``SPLIT_RATIOS`` are missed.
The 1.5/2.5 ratios are deliberately excluded (genuine 3:2 splits are rare; that bin was
dominated by false positives).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Common forward-split ratios (reverse splits handled via the inverse).
SPLIT_RATIOS: tuple[float, ...] = (
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    100.0,
)
RATIO_TOL = 0.025  # |observed/candidate - 1| must be within this
MIN_VOL_SCALE = 0.30  # forward split: share volume jumps ≈ ratio; loose lower bound
INTRADAY_MAX = 0.20  # a split day gaps at the open then trades normally intraday (vs a crash)


def detect_splits(
    open_: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> list[tuple[int, float]]:
    """Detect (index, ratio) split events in one instrument's raw OHLCV series.

    ``ratio`` is the factor the price DROPPED on a forward split (N:1 → N) or the fractional
    rise on a reverse split (1:N → 1/N).
    """
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    events: list[tuple[int, float]] = []
    for t in range(1, len(close)):
        pc, op, cl = close[t - 1], open_[t], close[t]
        if pc <= 0 or op <= 0 or cl <= 0:
            continue
        if abs(cl / op - 1.0) > INTRADAY_MAX:  # split day trades flat intraday after the gap
            continue
        gap = pc / op  # >1 forward (open is the lower, split-adjusted price); <1 reverse
        lo = max(0, t - 6)
        vbase = np.median(volume[lo:t]) if t > lo else volume[t - 1]
        for cand in SPLIT_RATIOS:
            if abs(gap / cand - 1.0) < RATIO_TOL:  # forward split
                if vbase <= 0 or (volume[t] / vbase) > cand * MIN_VOL_SCALE:
                    events.append((t, cand))
                break
            if abs(gap * cand - 1.0) < RATIO_TOL:  # reverse split (price rises at the open)
                events.append((t, 1.0 / cand))
                break
    return events


def adjust_splits(
    panel: pd.DataFrame, *, key: str = "instrument_id", date_col: str = "date"
) -> pd.DataFrame:
    """Add ``adj_factor`` + ``adj_close`` (back-adjusted) + ``split_ratio`` per instrument.

    Back-adjustment convention: recent prices unchanged; a price on date t is divided by
    ``adj_factor`` = product of all split ratios with event date > t. Apply ``adj_factor`` to
    open/high/low too for a consistent adjusted bar. Input needs ``open``/``close``/``volume``
    + the ``key`` and ``date_col`` columns.
    """
    out: list[pd.DataFrame] = []
    for _, g in panel.groupby(key, sort=False):
        g = g.sort_values(date_col).copy()
        ev = detect_splits(g["open"].to_numpy(), g["close"].to_numpy(), g["volume"].to_numpy())
        n = len(g)
        cum = np.ones(n)
        ratio_col = np.ones(n)
        for t, ratio in ev:
            cum[:t] *= ratio  # all dates BEFORE the split get divided down
            ratio_col[t] = ratio
        g["adj_factor"] = cum
        g["adj_close"] = g["close"].to_numpy() / cum
        g["split_ratio"] = ratio_col
        out.append(g)
    return pd.concat(out, ignore_index=True)


def residual_extreme_return_rate(
    panel_adj: pd.DataFrame,
    *,
    key: str = "instrument_id",
    date_col: str = "date",
    threshold: float = 0.5,
) -> float:
    """Fraction of adjusted daily returns whose magnitude exceeds ``threshold`` — the disclosed
    adjustment-error bound (a mix of genuine crashes + a few missed/exotic splits)."""
    g = panel_adj.sort_values([key, date_col])
    ret = g.groupby(key)["adj_close"].pct_change()
    n = int(ret.notna().sum())  # pyright: ignore[reportArgumentType]  # pandas-stub Series.sum()
    if n == 0:
        return float("nan")
    return float((ret.abs() > threshold).sum() / n)


__all__ = [
    "INTRADAY_MAX",
    "MIN_VOL_SCALE",
    "RATIO_TOL",
    "SPLIT_RATIOS",
    "adjust_splits",
    "detect_splits",
    "residual_extreme_return_rate",
]
