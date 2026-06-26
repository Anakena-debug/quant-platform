# Legacy module — pre-S34. The S34 plan locks behaviour here except
# for the privatised research-only helper. The pandas typing surface
# raises strict-mode basedpyright errors that pre-date S34;
# suppress at file scope so the S34 triad gate (which adds bars/ to
# the basedpyright target list) runs clean. Drop these suppressions
# when a future sprint refactors the legacy pandas signatures.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numba import njit


# ============================================================================
# Types / config
# ============================================================================

BarKind = Literal["tick", "volume", "dollar"]


@dataclass(frozen=True)
class ImbalanceConfig:
    exp_num_ticks_init: float = 1_000.0
    exp_imbalance_init: float | None = None
    ewma_span_ticks: int = 100
    ewma_span_imbalance: int = 100
    exp_num_ticks_min: float = 10.0
    exp_num_ticks_max: float = 100_000.0
    min_abs_exp_imbalance: float = 1e-12
    warmup_ticks: int = 1_000


@dataclass(frozen=True)
class RunsConfig:
    exp_num_ticks_init: float = 1_000.0
    exp_prob_buy_init: float | None = None
    exp_w_buy_init: float | None = None
    exp_w_sell_init: float | None = None
    ewma_span_ticks: int = 100
    ewma_span_weights: int = 100
    ewma_span_prob: int = 100
    exp_num_ticks_min: float = 10.0
    exp_num_ticks_max: float = 100_000.0
    min_abs_exp_weight: float = 1e-12
    warmup_ticks: int = 1_000


# ============================================================================
# Validation / extraction
# ============================================================================


def _validate_input(
    df: pd.DataFrame,
    price_col: str,
    volume_col: str,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if price_col not in df.columns:
        raise KeyError(f"Missing price column: {price_col}")
    if volume_col not in df.columns:
        raise KeyError(f"Missing volume column: {volume_col}")
    if len(df) == 0:
        raise ValueError("df is empty")


def _extract_arrays(
    df: pd.DataFrame,
    price_col: str,
    volume_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_input(df, price_col, volume_col)

    prices = df[price_col].to_numpy(dtype=np.float64, copy=False)
    volumes = df[volume_col].to_numpy(dtype=np.float64, copy=False)

    if np.isnan(prices).any():
        raise ValueError("prices contain NaN")
    if np.isnan(volumes).any():
        raise ValueError("volumes contain NaN")
    if (volumes < 0).any():
        raise ValueError("volumes must be non-negative")

    return prices, volumes


# ============================================================================
# Numba kernels
# ============================================================================


@njit(cache=True)
def _compute_tick_direction(prices: np.ndarray) -> np.ndarray:
    """
    Tick rule:
        b_t = sign(p_t - p_{t-1}) if changed, else b_{t-1}
    """
    n = len(prices)
    b = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0.0:
            b[i] = 1
        elif diff < 0.0:
            b[i] = -1
        else:
            b[i] = b[i - 1]

    return b


@njit(cache=True)
def _threshold_bars_core(increments: np.ndarray, threshold: float) -> np.ndarray:
    """
    Generic fixed-threshold bar sampler.

    Close a bar at the first index i such that cumulative increment >= threshold.
    """
    n = len(increments)
    out = np.empty(n, dtype=np.int64)
    k = 0
    cum = 0.0

    for i in range(n):
        cum += increments[i]
        if cum >= threshold:
            out[k] = i
            k += 1
            cum = 0.0

    return out[:k]


@njit(cache=True)
def _imbalance_bars_core(
    signed_increments: np.ndarray,
    exp_num_ticks_init: float,
    exp_imb_init: float,
    alpha_ticks: float,
    alpha_imb: float,
    exp_num_ticks_min: float,
    exp_num_ticks_max: float,
    min_abs_exp_imbalance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic imbalance-bar sampler.

    signed_increments:
        TIB:  b_t
        VIB:  b_t * v_t
        DIB:  b_t * p_t * v_t

    Stop when:
        |theta| >= E[T] * |E[x_t]|
    """
    n = len(signed_increments)

    out_idx = np.empty(n, dtype=np.int64)
    out_theta = np.empty(n, dtype=np.float64)
    out_threshold = np.empty(n, dtype=np.float64)
    out_ticks = np.empty(n, dtype=np.int64)

    k = 0
    theta = 0.0
    ticks_in_bar = 0

    exp_T = exp_num_ticks_init
    exp_x = exp_imb_init

    if abs(exp_x) < min_abs_exp_imbalance:
        exp_x = min_abs_exp_imbalance if exp_x >= 0.0 else -min_abs_exp_imbalance

    for i in range(n):
        x = signed_increments[i]
        theta += x
        ticks_in_bar += 1

        threshold = exp_T * abs(exp_x)
        if threshold < min_abs_exp_imbalance:
            threshold = min_abs_exp_imbalance

        if abs(theta) >= threshold:
            out_idx[k] = i
            out_theta[k] = theta
            out_threshold[k] = threshold
            out_ticks[k] = ticks_in_bar
            k += 1

            realized_avg_x = theta / ticks_in_bar

            exp_T = alpha_ticks * ticks_in_bar + (1.0 - alpha_ticks) * exp_T
            if exp_T < exp_num_ticks_min:
                exp_T = exp_num_ticks_min
            elif exp_T > exp_num_ticks_max:
                exp_T = exp_num_ticks_max

            exp_x = alpha_imb * realized_avg_x + (1.0 - alpha_imb) * exp_x
            if abs(exp_x) < min_abs_exp_imbalance:
                exp_x = min_abs_exp_imbalance if exp_x >= 0.0 else -min_abs_exp_imbalance

            theta = 0.0
            ticks_in_bar = 0

    return (
        out_idx[:k],
        out_theta[:k],
        out_threshold[:k],
        out_ticks[:k],
    )


@njit(cache=True)
def _runs_bars_core(
    sides: np.ndarray,
    weights: np.ndarray,
    exp_num_ticks_init: float,
    exp_prob_buy_init: float,
    exp_w_buy_init: float,
    exp_w_sell_init: float,
    alpha_ticks: float,
    alpha_prob: float,
    alpha_weights: float,
    exp_num_ticks_min: float,
    exp_num_ticks_max: float,
    min_abs_exp_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    AFML run-bar sampler (TRB / VRB / DRB).

    weights[i]:
        TRB:  1
        VRB:  v_t
        DRB:  p_t * v_t

    Buy-side and sell-side run totals accumulated independently:
        theta_buy  = sum_{t | b_t=+1} weights[t]
        theta_sell = sum_{t | b_t=-1} weights[t]

    Close when:
        max(theta_buy, theta_sell) >=
            E[T] * max{ P_buy * E[w | buy], (1 - P_buy) * E[w | sell] }

    Independent EWMA state for E[T], P[b=+1], E[w | buy], E[w | sell].
    One-sided weight EWMAs do NOT update on bars that saw zero ticks
    of that side (no 0/0; prior estimate carried forward).
    """
    n = len(sides)

    out_idx = np.empty(n, dtype=np.int64)
    out_theta_buy = np.empty(n, dtype=np.float64)
    out_theta_sell = np.empty(n, dtype=np.float64)
    out_threshold = np.empty(n, dtype=np.float64)
    out_ticks = np.empty(n, dtype=np.int64)

    k = 0
    theta_buy = 0.0
    theta_sell = 0.0
    ticks_in_bar = 0
    n_buy = 0
    n_sell = 0
    sum_w_buy_in_bar = 0.0
    sum_w_sell_in_bar = 0.0

    exp_T = exp_num_ticks_init
    exp_prob_buy = exp_prob_buy_init
    exp_w_buy = exp_w_buy_init
    exp_w_sell = exp_w_sell_init

    if exp_w_buy < min_abs_exp_weight:
        exp_w_buy = min_abs_exp_weight
    if exp_w_sell < min_abs_exp_weight:
        exp_w_sell = min_abs_exp_weight

    for i in range(n):
        b = sides[i]
        w = weights[i]

        if b == 1:
            theta_buy += w
            sum_w_buy_in_bar += w
            n_buy += 1
        else:
            theta_sell += w
            sum_w_sell_in_bar += w
            n_sell += 1

        ticks_in_bar += 1

        if theta_buy >= theta_sell:
            theta = theta_buy
        else:
            theta = theta_sell

        side_term_buy = exp_prob_buy * exp_w_buy
        side_term_sell = (1.0 - exp_prob_buy) * exp_w_sell
        if side_term_buy >= side_term_sell:
            threshold = exp_T * side_term_buy
        else:
            threshold = exp_T * side_term_sell
        if threshold < min_abs_exp_weight:
            threshold = min_abs_exp_weight

        if theta >= threshold:
            out_idx[k] = i
            out_theta_buy[k] = theta_buy
            out_theta_sell[k] = theta_sell
            out_threshold[k] = threshold
            out_ticks[k] = ticks_in_bar
            k += 1

            exp_T = alpha_ticks * ticks_in_bar + (1.0 - alpha_ticks) * exp_T
            if exp_T < exp_num_ticks_min:
                exp_T = exp_num_ticks_min
            elif exp_T > exp_num_ticks_max:
                exp_T = exp_num_ticks_max

            realized_prob_buy = n_buy / ticks_in_bar
            exp_prob_buy = alpha_prob * realized_prob_buy + (1.0 - alpha_prob) * exp_prob_buy
            if exp_prob_buy < 0.0:
                exp_prob_buy = 0.0
            elif exp_prob_buy > 1.0:
                exp_prob_buy = 1.0

            if n_buy > 0:
                realized_w_buy = sum_w_buy_in_bar / n_buy
                exp_w_buy = alpha_weights * realized_w_buy + (1.0 - alpha_weights) * exp_w_buy
                if exp_w_buy < min_abs_exp_weight:
                    exp_w_buy = min_abs_exp_weight

            if n_sell > 0:
                realized_w_sell = sum_w_sell_in_bar / n_sell
                exp_w_sell = alpha_weights * realized_w_sell + (1.0 - alpha_weights) * exp_w_sell
                if exp_w_sell < min_abs_exp_weight:
                    exp_w_sell = min_abs_exp_weight

            theta_buy = 0.0
            theta_sell = 0.0
            ticks_in_bar = 0
            n_buy = 0
            n_sell = 0
            sum_w_buy_in_bar = 0.0
            sum_w_sell_in_bar = 0.0

    return (
        out_idx[:k],
        out_theta_buy[:k],
        out_theta_sell[:k],
        out_threshold[:k],
        out_ticks[:k],
    )


# ============================================================================
# Aggregation
# ============================================================================


def aggregate_to_ohlcv(
    df: pd.DataFrame,
    close_indices: np.ndarray,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    """
    Aggregate ticks into OHLCV bars using close indices.

    Each close index is inclusive.
    """
    # s83 F2b: zero completed bars + data present + partial requested used to
    # early-return empty, silently dropping the requested partial row. Let the
    # partial-bar block below run in that case.
    if len(close_indices) == 0 and not (include_partial_last_bar and len(df) > 0):
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "tick_count",
                "dollar_volume",
            ]
        )

    prices = df[price_col].to_numpy(dtype=np.float64, copy=False)
    volumes = df[volume_col].to_numpy(dtype=np.float64, copy=False)
    timestamps = df.index

    bars: list[dict[str, float | int | pd.Timestamp]] = []
    start = 0

    for end in close_indices:
        bar_prices = prices[start : end + 1]
        bar_volumes = volumes[start : end + 1]

        total_volume = float(bar_volumes.sum())
        dollar_volume = float((bar_prices * bar_volumes).sum())

        bars.append(
            {
                "timestamp": timestamps[end],
                "open": float(bar_prices[0]),
                "high": float(bar_prices.max()),
                "low": float(bar_prices.min()),
                "close": float(bar_prices[-1]),
                "volume": total_volume,
                "vwap": dollar_volume / total_volume
                if total_volume > 0.0
                else float(bar_prices[-1]),
                "tick_count": int(len(bar_prices)),
                "dollar_volume": dollar_volume,
            }
        )
        start = end + 1

    if include_partial_last_bar and start < len(df):
        bar_prices = prices[start:]
        bar_volumes = volumes[start:]
        total_volume = float(bar_volumes.sum())
        dollar_volume = float((bar_prices * bar_volumes).sum())

        bars.append(
            {
                "timestamp": timestamps[-1],
                "open": float(bar_prices[0]),
                "high": float(bar_prices.max()),
                "low": float(bar_prices.min()),
                "close": float(bar_prices[-1]),
                "volume": total_volume,
                "vwap": dollar_volume / total_volume
                if total_volume > 0.0
                else float(bar_prices[-1]),
                "tick_count": int(len(bar_prices)),
                "dollar_volume": dollar_volume,
            }
        )

    out = pd.DataFrame(bars).set_index("timestamp")
    return out


# ============================================================================
# Increment builders
# ============================================================================


def _standard_increments(
    prices: np.ndarray,
    volumes: np.ndarray,
    kind: BarKind,
) -> np.ndarray:
    if kind == "tick":
        return np.ones(len(prices), dtype=np.float64)
    if kind == "volume":
        return volumes
    if kind == "dollar":
        return prices * volumes
    raise ValueError(f"Unknown bar kind: {kind}")


def _signed_increments(
    prices: np.ndarray,
    volumes: np.ndarray,
    kind: BarKind,
) -> tuple[np.ndarray, np.ndarray]:
    b = _compute_tick_direction(prices)

    if kind == "tick":
        x = b.astype(np.float64)
    elif kind == "volume":
        x = b.astype(np.float64) * volumes
    elif kind == "dollar":
        x = b.astype(np.float64) * prices * volumes
    else:
        raise ValueError(f"Unknown bar kind: {kind}")

    return b, x


# ============================================================================
# Standard bars
# ============================================================================


def bars_by_threshold(
    df: pd.DataFrame,
    threshold: float,
    kind: BarKind,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")

    prices, volumes = _extract_arrays(df, price_col, volume_col)
    increments = _standard_increments(prices, volumes, kind)
    close_idx = _threshold_bars_core(increments, float(threshold))

    out = aggregate_to_ohlcv(
        df,
        close_idx,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )
    out.attrs["bar_type"] = f"{kind}_bars"
    out.attrs["threshold"] = float(threshold)
    return out


def threshold_bar_close_indices(
    df: pd.DataFrame,
    threshold: float,
    kind: BarKind,
    price_col: str = "price",
    volume_col: str = "volume",
) -> np.ndarray:
    """Return tick-level close indices for threshold bars.

    Reuses _extract_arrays, _standard_increments, and _threshold_bars_core.
    The returned array contains the positional index of the last tick in
    each bar — the same indices that aggregate_to_ohlcv consumes.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")
    prices, volumes = _extract_arrays(df, price_col, volume_col)
    increments = _standard_increments(prices, volumes, kind)
    return _threshold_bars_core(increments, float(threshold))


def tick_bars(
    df: pd.DataFrame,
    threshold: int,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return bars_by_threshold(
        df=df,
        threshold=float(threshold),
        kind="tick",
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def volume_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return bars_by_threshold(
        df=df,
        threshold=threshold,
        kind="volume",
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def dollar_bars(
    df: pd.DataFrame,
    threshold: float,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return bars_by_threshold(
        df=df,
        threshold=threshold,
        kind="dollar",
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


# ============================================================================
# Imbalance bars
# ============================================================================


def imbalance_bars(
    df: pd.DataFrame,
    kind: BarKind,
    config: ImbalanceConfig = ImbalanceConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    # s83 F2a: with a partial tail, the diagnostic-column assignment below
    # (theta/threshold/expected_ticks_proxy, one value per COMPLETED bar)
    # crashed with a k-vs-k+1 length mismatch. Same explicit contract as
    # runs_bars until the core emits padded partial diagnostics.
    if include_partial_last_bar:
        raise NotImplementedError(
            "include_partial_last_bar=True is not supported for imbalance_bars: "
            "diagnostic columns (theta, threshold, expected_ticks_proxy) exist "
            "only for completed bars."
        )
    # s83 F5: parity with runs_bars and the streaming engines (span=0 ⇒ α=2,
    # an unstable EWMA, previously accepted silently).
    if config.ewma_span_ticks < 1 or config.ewma_span_imbalance < 1:
        raise ValueError("ewma_span_* must be >= 1")

    prices, volumes = _extract_arrays(df, price_col, volume_col)
    _, signed_x = _signed_increments(prices, volumes, kind)

    warmup = min(config.warmup_ticks, len(signed_x))
    if warmup <= 0:
        raise ValueError("No data available for warmup")

    if config.exp_imbalance_init is None:
        exp_x0 = float(np.mean(signed_x[:warmup]))
    else:
        exp_x0 = float(config.exp_imbalance_init)

    if abs(exp_x0) < config.min_abs_exp_imbalance:
        exp_x0 = config.min_abs_exp_imbalance if exp_x0 >= 0.0 else -config.min_abs_exp_imbalance

    alpha_ticks = 2.0 / (config.ewma_span_ticks + 1.0)
    alpha_imb = 2.0 / (config.ewma_span_imbalance + 1.0)

    close_idx, theta, threshold, ticks = _imbalance_bars_core(
        signed_increments=signed_x,
        exp_num_ticks_init=float(config.exp_num_ticks_init),
        exp_imb_init=exp_x0,
        alpha_ticks=float(alpha_ticks),
        alpha_imb=float(alpha_imb),
        exp_num_ticks_min=float(config.exp_num_ticks_min),
        exp_num_ticks_max=float(config.exp_num_ticks_max),
        min_abs_exp_imbalance=float(config.min_abs_exp_imbalance),
    )

    out = aggregate_to_ohlcv(
        df,
        close_idx,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )

    if len(out) > 0:
        out["theta"] = theta
        out["threshold"] = threshold
        out["expected_ticks_proxy"] = ticks

    out.attrs["bar_type"] = f"{kind}_imbalance_bars"
    out.attrs["config"] = config
    return out


def tick_imbalance_bars(
    df: pd.DataFrame,
    config: ImbalanceConfig = ImbalanceConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return imbalance_bars(
        df=df,
        kind="tick",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def volume_imbalance_bars(
    df: pd.DataFrame,
    config: ImbalanceConfig = ImbalanceConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return imbalance_bars(
        df=df,
        kind="volume",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def dollar_imbalance_bars(
    df: pd.DataFrame,
    config: ImbalanceConfig = ImbalanceConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return imbalance_bars(
        df=df,
        kind="dollar",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )


# ============================================================================
# Runs bars (AFML §2.3.2.2, TRB / VRB / DRB)
# ============================================================================


def _resolve_sides(
    df: pd.DataFrame,
    prices: np.ndarray,
    side_col: str | None,
) -> np.ndarray:
    """
    Resolve aggressor side as int8 array of strict ±1 values.

    side_col is None → tick rule via ``_compute_tick_direction``
        (b[0]=+1, zero-diff carries forward; never produces 0).
    side_col is a column name → ground truth (e.g. Databento trades
        normalised by the caller to ±1). Strict validation: numeric
        dtype, no NaN, all values exactly ±1.
    """
    if side_col is None:
        return _compute_tick_direction(prices)

    if side_col not in df.columns:
        raise KeyError(f"Missing side column: {side_col}")

    b_raw = df[side_col].to_numpy()
    if not np.issubdtype(b_raw.dtype, np.number):
        raise TypeError(f"side_col '{side_col}' must be numeric, got dtype {b_raw.dtype}")
    if np.issubdtype(b_raw.dtype, np.floating) and np.isnan(b_raw).any():
        raise ValueError(f"side_col '{side_col}' contains NaN")
    if not np.all((b_raw == 1) | (b_raw == -1)):
        raise ValueError(f"side_col '{side_col}' must contain only ±1 values")
    return b_raw.astype(np.int8)


def _resolve_runs_initial_state(
    sides: np.ndarray,
    weights: np.ndarray,
    config: RunsConfig,
) -> tuple[float, float, float, float]:
    """
    Resolve kernel initial state from the warmup window.

    Returns ``(exp_T, prob_buy, w_buy, w_sell)``, all guaranteed
    finite floats:
      * ``exp_T`` clamped to ``[exp_num_ticks_min, exp_num_ticks_max]``
      * ``prob_buy`` clamped to [0, 1]
      * one-sided weight fallbacks: if a side is absent in warmup,
        use the sign-blind warmup mean (EWMA corrects once that side
        appears); both floored at ``min_abs_exp_weight``.

    Pre-resolution here keeps ``_runs_bars_core`` pure-numeric: no
    None ever enters the numba kernel.
    """
    n = len(sides)
    warmup = min(config.warmup_ticks, n)
    if warmup <= 0:
        raise ValueError("No data available for warmup")

    sides_w = sides[:warmup]
    weights_w = weights[:warmup]
    buy_mask = sides_w == 1
    sell_mask = sides_w == -1
    n_buy = int(buy_mask.sum())
    n_sell = int(sell_mask.sum())

    if config.exp_prob_buy_init is None:
        prob_buy = n_buy / warmup
    else:
        prob_buy = float(config.exp_prob_buy_init)
    if prob_buy < 0.0:
        prob_buy = 0.0
    elif prob_buy > 1.0:
        prob_buy = 1.0

    fallback_w = float(weights_w.mean())

    if config.exp_w_buy_init is None:
        w_buy = float(weights_w[buy_mask].mean()) if n_buy > 0 else fallback_w
    else:
        w_buy = float(config.exp_w_buy_init)

    if config.exp_w_sell_init is None:
        w_sell = float(weights_w[sell_mask].mean()) if n_sell > 0 else fallback_w
    else:
        w_sell = float(config.exp_w_sell_init)

    if w_buy < config.min_abs_exp_weight:
        w_buy = config.min_abs_exp_weight
    if w_sell < config.min_abs_exp_weight:
        w_sell = config.min_abs_exp_weight

    exp_T = float(config.exp_num_ticks_init)
    if exp_T < config.exp_num_ticks_min:
        exp_T = config.exp_num_ticks_min
    elif exp_T > config.exp_num_ticks_max:
        exp_T = config.exp_num_ticks_max

    return exp_T, prob_buy, w_buy, w_sell


def runs_bars(
    df: pd.DataFrame,
    kind: BarKind,
    config: RunsConfig = RunsConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    side_col: str | None = None,
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    """
    Canonical AFML run bars (§2.3.2.2). See ``_runs_bars_core`` for math.

    ``side_col`` is None → infer via tick rule (default for OHLCV tapes).
    ``side_col`` is a column → ground-truth aggressor side, e.g.
    Databento's ``B``/``A`` mapped to ±1 by the caller. Strict ±1
    validation; no implicit fallback.

    Output adds ``theta_buy``, ``theta_sell``, ``threshold``, and
    ``ticks_in_bar`` (realized tick count per completed bar) to the
    standard OHLCV columns.

    ``include_partial_last_bar=True`` is not supported yet — the
    diagnostic columns are kernel-emitted only for completed bars and
    would misalign with an appended partial bar.
    """
    if include_partial_last_bar:
        raise NotImplementedError(
            "include_partial_last_bar=True is not supported for runs_bars: "
            "diagnostic columns (theta_buy/sell, threshold, ticks_in_bar) "
            "exist only for completed bars and would misalign with an "
            "appended partial bar."
        )
    if config.ewma_span_ticks < 1 or config.ewma_span_prob < 1 or config.ewma_span_weights < 1:
        raise ValueError("ewma_span_* must be >= 1")

    prices, volumes = _extract_arrays(df, price_col, volume_col)
    sides = _resolve_sides(df, prices, side_col)
    weights = _standard_increments(prices, volumes, kind)

    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values (NaN or inf)")
    if (weights < 0.0).any():
        raise ValueError("weights must be non-negative")

    exp_T_init, prob_buy_init, w_buy_init, w_sell_init = _resolve_runs_initial_state(
        sides, weights, config
    )

    alpha_ticks = 2.0 / (config.ewma_span_ticks + 1.0)
    alpha_prob = 2.0 / (config.ewma_span_prob + 1.0)
    alpha_weights = 2.0 / (config.ewma_span_weights + 1.0)

    close_idx, theta_buy, theta_sell, threshold, ticks = _runs_bars_core(
        sides=sides,
        weights=weights,
        exp_num_ticks_init=exp_T_init,
        exp_prob_buy_init=prob_buy_init,
        exp_w_buy_init=w_buy_init,
        exp_w_sell_init=w_sell_init,
        alpha_ticks=alpha_ticks,
        alpha_prob=alpha_prob,
        alpha_weights=alpha_weights,
        exp_num_ticks_min=float(config.exp_num_ticks_min),
        exp_num_ticks_max=float(config.exp_num_ticks_max),
        min_abs_exp_weight=float(config.min_abs_exp_weight),
    )

    out = aggregate_to_ohlcv(
        df,
        close_idx,
        price_col=price_col,
        volume_col=volume_col,
        include_partial_last_bar=include_partial_last_bar,
    )

    if len(out) > 0:
        out["theta_buy"] = theta_buy
        out["theta_sell"] = theta_sell
        out["threshold"] = threshold
        out["ticks_in_bar"] = ticks

    out.attrs["bar_type"] = f"{kind}_runs_bars"
    out.attrs["config"] = config
    return out


def tick_runs_bars(
    df: pd.DataFrame,
    config: RunsConfig = RunsConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    side_col: str | None = None,
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return runs_bars(
        df=df,
        kind="tick",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        side_col=side_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def volume_runs_bars(
    df: pd.DataFrame,
    config: RunsConfig = RunsConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    side_col: str | None = None,
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return runs_bars(
        df=df,
        kind="volume",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        side_col=side_col,
        include_partial_last_bar=include_partial_last_bar,
    )


def dollar_runs_bars(
    df: pd.DataFrame,
    config: RunsConfig = RunsConfig(),
    price_col: str = "price",
    volume_col: str = "volume",
    side_col: str | None = None,
    include_partial_last_bar: bool = False,
) -> pd.DataFrame:
    return runs_bars(
        df=df,
        kind="dollar",
        config=config,
        price_col=price_col,
        volume_col=volume_col,
        side_col=side_col,
        include_partial_last_bar=include_partial_last_bar,
    )


# ============================================================================
# Calibration / diagnostics
# ============================================================================


def _calibrate_threshold_research(
    df: pd.DataFrame,
    kind: BarKind,
    target_bars_per_day: int = 50,
    price_col: str = "price",
    volume_col: str = "volume",
) -> float:
    """Research-only threshold calibrator (S33 audit CHK2, S34 §3.AC7).

    Consumes the **full** input DataFrame to compute
    ``avg_daily_increment = sum(increments) / num_days`` and divides by
    ``target_bars_per_day``. The full-sample sum makes this a
    lookahead estimator at the first event time — production-blocking
    per S33 §5.D7.

    Privatised in S34: the function is reachable only via
    ``quantcore.bars.bars._calibrate_threshold_research`` (no public
    re-export from ``quantcore.bars``). For production live use,
    supply a constant threshold or a trailing-window estimator.
    """
    if target_bars_per_day <= 0:
        raise ValueError("target_bars_per_day must be > 0")

    prices, volumes = _extract_arrays(df, price_col, volume_col)
    increments = _standard_increments(prices, volumes, kind)

    idx = pd.to_datetime(df.index)
    num_days = max(idx.normalize().nunique(), 1)
    avg_daily_increment = float(increments.sum()) / num_days
    return avg_daily_increment / float(target_bars_per_day)
