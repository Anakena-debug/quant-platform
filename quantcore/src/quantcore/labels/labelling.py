from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# numba is a soft runtime dependency (pending S1.4 pin). When installed,
# @njit delivers the designed JIT acceleration on _triple_barrier_core.
# When absent, @njit degrades to a pass-through decorator so pure-Python
# /NumPy semantics remain reachable. Behaviour is identical; only speed
# degrades. Parallels the P0.3 shim in validation/bootstrap.py.
try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator


@dataclass(frozen=True)
class TripleBarrierConfig:
    vertical_bars: int
    pt_sl: tuple[float, float] = (1.0, 1.0)
    min_ret: float = 0.0

    def __post_init__(self) -> None:
        # vertical_bars=None / 0 / negative all fail here: None fails first via
        # the `<= 0` comparison (TypeError: NoneType vs int), ints at or below
        # zero raise ValueError.
        if self.vertical_bars <= 0:
            raise ValueError(
                f"TripleBarrierConfig.vertical_bars must be > 0; "
                f"got {self.vertical_bars}."
            )


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """EWMA std of close-to-close returns on a daily-or-lower frequency index.

    Intraday input is rejected. On
    intraday bars this function would return bar-horizon vol, under-sizing
    true daily vol by ~1/sqrt(bars_per_day).

    To compute daily vol from intraday bars, resample first:
        close.resample("1B").last().pipe(get_daily_vol, span=span)
    """
    _assert_daily_or_lower(close.index)
    ret = close.pct_change()
    return ret.ewm(span=span, adjust=False).std()


def _assert_daily_or_lower(idx: pd.Index) -> None:
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"get_daily_vol requires DatetimeIndex, got {type(idx).__name__}")
    if not idx.is_monotonic_increasing:
        raise ValueError("get_daily_vol requires monotonic index")
    if not idx.is_unique:
        raise ValueError("get_daily_vol requires unique timestamps")
    if len(idx) < 2:
        return
    # 20h (not 24h) tolerates DST transitions, holiday-spanning gaps, and
    # timestamp jitter while cleanly rejecting 30-min and below.
    median_delta = pd.Series(idx).diff().dropna().median()
    if median_delta < pd.Timedelta(hours=20):
        raise ValueError(
            f"get_daily_vol requires daily-or-lower frequency "
            f"(median spacing >= 20h); got {median_delta}. On "
            f"intraday bars this returns bar-horizon vol, not "
            f"daily. Resample first:\n"
            f"    close.resample('1B').last().pipe(get_daily_vol, span=span)"
        )


@njit(cache=True)
def _triple_barrier_core(
    prices: np.ndarray, t0: np.ndarray, t1: np.ndarray, up: np.ndarray, dn: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t0)
    exit_idx = np.empty(n, dtype=np.int64)
    label = np.empty(n, dtype=np.int8)
    ret = np.empty(n, dtype=np.float64)
    for i in range(n):
        e0, e1 = t0[i], t1[i]
        ex = e1
        lb = 0
        p0 = prices[e0]
        for j in range(e0 + 1, e1 + 1):
            p = prices[j]
            if p >= up[i]:
                ex = j
                lb = 1
                break
            if p <= dn[i]:
                ex = j
                lb = -1
                break
        exit_idx[i] = ex
        label[i] = lb
        ret[i] = prices[ex] / p0 - 1.0
    return exit_idx, label, ret


@njit(cache=True)
def _cusum_core(log_diffs: np.ndarray, threshold: float) -> np.ndarray:
    n = log_diffs.shape[0]
    out_idx = np.empty(n, dtype=np.int64)
    k = 0
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        x = log_diffs[i]
        s_pos = s_pos + x
        if s_pos < 0.0:
            s_pos = 0.0
        s_neg = s_neg + x
        if s_neg > 0.0:
            s_neg = 0.0
        if s_pos > threshold or s_neg < -threshold:
            out_idx[k] = i
            k += 1
            s_pos = 0.0
            s_neg = 0.0
    return out_idx[:k]


def cusum_filter(close: pd.Series, threshold: float) -> pd.DatetimeIndex:
    r"""Symmetric CUSUM filter on log-returns (AFML §2.5.2.1).

    Emits an event timestamp whenever the running sum of log-returns
    (reset at each event) crosses ``+threshold`` or ``-threshold``.
    Trigger uses strict inequality (``>`` / ``<``) — exact boundary
    values do not fire. Both accumulators reset to zero on every event.

    Parameters
    ----------
    close : pd.Series
        Strictly-positive close prices on a monotonic index. Non-
        positive values raise ``ValueError`` because ``log`` is
        otherwise ill-defined and would silently inject NaNs.
    threshold : float
        Cumulative log-return threshold (e.g. ``0.02`` ≈ a 2 %
        cumulative move). Must be strictly positive.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps at which the symmetric CUSUM crossed
        ``±threshold``. Order-preserving, no duplicates.

    Notes
    -----
    Prior to S31 (2026-05-17) this filter used ``close.pct_change()``.
    Simple returns are non-additive over a path (e.g. ``-50 %`` then
    ``+50 %`` sums to zero but ends at ``-25 %``), which violates the
    AFML §2.5.2.1 cumulative-surprise interpretation. Log-returns are
    additive and scale-invariant, matching López de Prado's reference
    snippet (``gRaw.diff()`` on log-prices). Migration impact:
    historical event counts may shift for the same threshold.

    The hot loop is JIT-compiled via ``_cusum_core``; when numba is
    absent the ``@njit`` shim at module top degrades to pure Python
    with identical semantics.
    """
    if threshold <= 0:
        raise ValueError(
            f"cusum_filter: threshold must be > 0; got {threshold!r}. "
            "Threshold is in cumulative log-return units (e.g. 0.02 "
            "≈ 2% cumulative move)."
        )
    arr = close.to_numpy(np.float64, copy=False)
    if arr.size == 0:
        return pd.DatetimeIndex(close.index[:0])
    valid_mask = np.isfinite(arr) & (arr > 0)
    if not bool(np.all(valid_mask)):
        n_bad = int((~valid_mask).sum())
        raise ValueError(
            "cusum_filter: close must contain strictly-positive finite "
            f"prices (log transform is otherwise ill-defined); found "
            f"{n_bad} non-conforming value(s) (NaN / ±inf / non-positive)."
        )
    if not close.index.is_monotonic_increasing:
        raise ValueError(
            "cusum_filter: close.index must be monotonic increasing; "
            "events on a shuffled index are temporally meaningless."
        )
    log_prices = np.log(arr)
    log_diffs = np.empty(arr.size, dtype=np.float64)
    log_diffs[0] = 0.0
    if arr.size > 1:
        log_diffs[1:] = log_prices[1:] - log_prices[:-1]
    event_idx = _cusum_core(log_diffs, float(threshold))
    return pd.DatetimeIndex(close.index[event_idx])


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    config: TripleBarrierConfig,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    target = target.reindex(t_events, method="ffill").dropna()
    idx = target.index
    side = pd.Series(1.0, index=idx) if side is None else side.reindex(idx).fillna(1.0)
    events = pd.DataFrame(index=idx)
    events["target"] = target
    events["side"] = side
    events = events[events["target"] > config.min_ret]
    pt, sl = config.pt_sl
    events["pt"] = events["target"] * pt if pt > 0 else np.nan
    events["sl"] = events["target"] * sl if sl > 0 else np.nan
    loc = close.index.searchsorted(events.index)
    end_loc = np.minimum(loc + config.vertical_bars, len(close) - 1)
    events["t1"] = close.index[end_loc]
    return events


def _get_events_legacy_unbounded(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    min_ret: float = 0.0,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    r"""DO NOT USE. Reproduces the pre-P1.1 ``get_events`` behaviour when
    ``TripleBarrierConfig.vertical_bars`` was ``None``: every un-touched
    event's ``t1`` equals ``close.index[-1]``.

    The legacy default produced pathological concurrency pile-up at the
    series end, silently breaking AFML §4.8 sample-weight invariants and
    collapsing the ``PurgedKFold`` training set on every fold touching
    the tail.

    Preserved as a **private numerical oracle** for regression testing
    only. Underscore prefix + ``_legacy_unbounded`` suffix + module-
    private; no production code should import this. Takes primitive args
    (not a ``TripleBarrierConfig``) so the oracle still works after the
    config has been locked down post-P1.1. Removal anchored to the
    conformal-integration sprint (S6+) alongside
    ``_get_sample_weights_legacy_broken`` (P0.3) and the P0.1
    structural-breaks shims.
    """
    warnings.warn(
        "_get_events_legacy_unbounded is a regression oracle preserving "
        "the pre-P1.1 pathological behaviour (t1 = close.index[-1] for "
        "every un-touched event). Use get_events with "
        "TripleBarrierConfig(vertical_bars=N) for production.",
        DeprecationWarning,
        stacklevel=2,
    )
    target = target.reindex(t_events, method="ffill").dropna()
    idx = target.index
    side = pd.Series(1.0, index=idx) if side is None else side.reindex(idx).fillna(1.0)
    events = pd.DataFrame(index=idx)
    events["target"] = target
    events["side"] = side
    events = events[events["target"] > min_ret]
    pt, sl = pt_sl
    events["pt"] = events["target"] * pt if pt > 0 else np.nan
    events["sl"] = events["target"] * sl if sl > 0 else np.nan
    events["t1"] = close.index[-1]
    return events


def apply_triple_barrier(
    close: pd.Series, events: pd.DataFrame, side: pd.Series | None = None
) -> pd.DataFrame:
    prices = close.to_numpy(np.float64, copy=False)
    idx = close.index
    t0 = idx.searchsorted(events.index)
    t1 = idx.searchsorted(events["t1"])
    event_side = (
        events["side"].to_numpy(np.float64)
        if side is None
        # Mirror get_events: a missing (NaN) explicit side defaults long (+1).
        else side.reindex(events.index).fillna(1.0).to_numpy(np.float64)
    )
    # Side must be directional (±1). A 0.0/NaN side silently forces
    # ``bin = lb * sign(side) == 0`` for EVERY outcome — masking real PT/SL
    # touches as the "vertical timeout" label (lb == 0), which is a legitimate
    # value, so the spurious row survives the consumers' ``dropna(subset=["bin"])``
    # and pollutes the training set. The 'N'/no-aggressor (0) state must be
    # resolved to ±1 upstream (e.g. the tick rule) BEFORE labeling. Fail loud.
    if not np.all(np.isin(np.sign(event_side), (-1.0, 1.0))):
        raise ValueError(
            "apply_triple_barrier: event side must be ±1 (got 0.0/NaN for some "
            "events). A 0.0/NaN side collapses bin to 0 for every outcome; "
            "resolve no-aggressor sides to ±1 upstream before labeling."
        )
    entry = prices[t0]
    pt = events["pt"].fillna(np.inf).to_numpy(np.float64)
    sl = events["sl"].fillna(np.inf).to_numpy(np.float64)
    up = np.where(event_side > 0, entry * (1 + pt), entry * (1 + sl))
    dn = np.where(event_side > 0, entry * (1 - sl), entry * (1 - pt))
    ex, lb, ret = _triple_barrier_core(prices, t0, t1, up, dn)
    out = pd.DataFrame(index=events.index)
    out["t1"] = idx[ex]
    out["ret"] = ret * event_side
    out["bin"] = lb * np.sign(event_side).astype(np.int8)
    return out
