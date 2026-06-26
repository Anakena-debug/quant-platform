from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# numba is a *soft* runtime dependency. When installed, @njit kernels deliver
# the designed JIT acceleration. When absent (e.g. a minimal test sandbox),
# @njit degrades to a pass-through decorator so pure-Python/NumPy semantics
# remain reachable. Behaviour is identical; only speed degrades.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — import-time branch, trivially covered by either path
    from numba import njit as _njit  # type: ignore[import-not-found]

    def njit(*args: Any, **kwargs: Any) -> Any:
        return _njit(*args, **kwargs)
except ImportError:  # pragma: no cover

    def njit(*args: Any, **kwargs: Any) -> Any:
        # Bare decorator form: @njit
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]

        # Parametrised form: @njit(cache=True) or @njit("sig")
        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _wrap


@dataclass(frozen=True)
class BootstrapConfig:
    normalize_weights_to_n: bool = True
    min_weight: float = 1e-12


def _intervals_from_t1(
    close_idx: pd.DatetimeIndex, t1: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    if not close_idx.is_monotonic_increasing:
        raise ValueError("close_idx must be sorted")
    if not isinstance(t1.index, pd.DatetimeIndex):
        raise TypeError("t1.index must be DatetimeIndex")
    t1 = t1.dropna().sort_index()
    close_ns = close_idx.view("i8")
    t0_ns = t1.index.view("i8")
    t1_ns = pd.DatetimeIndex(t1.values).view("i8")
    start = np.searchsorted(close_ns, t0_ns, side="left")
    end = np.searchsorted(close_ns, t1_ns, side="right") - 1
    valid = (start < len(close_idx)) & (end >= 0) & (start <= end)
    return start[valid].astype(np.int64), end[valid].astype(np.int64), t1.index[valid]


@njit(cache=True)
def _concurrency(n: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    diff = np.zeros(n + 1, dtype=np.int64)
    for i in range(len(start)):
        diff[start[i]] += 1
        if end[i] + 1 < len(diff):
            diff[end[i] + 1] -= 1
    out = np.empty(n, dtype=np.int64)
    run = 0
    for i in range(n):
        run += diff[i]
        out[i] = run
    return out


@njit(cache=True)
def _avg_uniqueness(start: np.ndarray, end: np.ndarray, conc: np.ndarray) -> np.ndarray:
    out = np.empty(len(start), dtype=np.float64)
    for i in range(len(start)):
        s, e = start[i], end[i]
        total = 0.0
        n = 0
        for t in range(s, e + 1):
            c = conc[t]
            if c > 0:
                total += 1.0 / c
                n += 1
        out[i] = total / n if n else 1.0
    return out


@njit(cache=True)
def _abs_returns(close_values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Legacy helper: |p_end / p_start - 1| per event.

    Retained because ``_get_sample_weights_legacy_broken`` depends on it.
    Not used by the canonical AFML ``get_sample_weights`` path.
    """
    out = np.empty(len(start), dtype=np.float64)
    for i in range(len(start)):
        p0 = close_values[start[i]]
        p1 = close_values[end[i]]
        out[i] = abs(p1 / p0 - 1.0) if p0 > 0 else 0.0
    return out


@njit(cache=True)
def _afml_weights(
    ret: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    conc: np.ndarray,
) -> np.ndarray:
    r"""AFML snippet 4.10 — return-attribution sample weights.

    .. math::
        w_i \;=\; \left| \sum_{t = \mathrm{start}_i}^{\mathrm{end}_i}
            \frac{r_t}{c_t} \right|

    Parameters
    ----------
    ret : np.ndarray, shape (N,)
        Per-bar log-returns. ``ret[t] = log(close[t]) - log(close[t-1])``.
        ``ret[0]`` must be finite (the caller sets it to 0 since there is
        no prior bar); events that span bar 0 include that zero term.
    start, end : np.ndarray, shape (M,), int64
        Inclusive event bar-index interval. ``start[i] <= end[i]``.
    conc : np.ndarray, shape (N,), int64
        Concurrent-event count per bar. Bars with ``conc[t] == 0`` contribute
        nothing (vacuously outside every event — but the loop protects
        against divide-by-zero regardless).

    Returns
    -------
    np.ndarray, shape (M,), float64
        ``w_i`` raw (pre-clamp, pre-normalisation).

    Complexity
    ----------
    :math:`O\!\left(\sum_i (\mathrm{end}_i - \mathrm{start}_i + 1)\right)` —
    linear in the total event-coverage footprint. No allocations inside
    the loop.
    """
    n = len(start)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s, e = start[i], end[i]
        acc = 0.0
        for t in range(s, e + 1):
            c = conc[t]
            if c > 0:
                acc += ret[t] / c
        out[i] = abs(acc)
    return out


@njit(cache=True)
def _inc_avg_uniqueness(start: np.ndarray, end: np.ndarray, occ: np.ndarray, i: int) -> float:
    s, e = start[i], end[i]
    total = 0.0
    n = 0
    for t in range(s, e + 1):
        total += 1.0 / (occ[t] + 1.0)
        n += 1
    return total / n if n else 1.0


@njit(cache=True)
def _seq_bootstrap(
    start: np.ndarray, end: np.ndarray, sample_length: int, u: np.ndarray
) -> np.ndarray:
    n_labels = len(start)
    n_bars = int(end.max()) + 1
    occ = np.zeros(n_bars, dtype=np.int64)
    out = np.empty(sample_length, dtype=np.int64)
    for k in range(sample_length):
        probs = np.empty(n_labels, dtype=np.float64)
        total = 0.0
        for i in range(n_labels):
            p = _inc_avg_uniqueness(start, end, occ, i)
            probs[i] = p
            total += p
        threshold = u[k] * total
        csum = 0.0
        choice = 0
        for i in range(n_labels):
            csum += probs[i]
            if csum >= threshold:
                choice = i
                break
        out[k] = choice
        for t in range(start[choice], end[choice] + 1):
            occ[t] += 1
    return out


def get_num_concurrent_events(close_idx: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    start, end, _ = _intervals_from_t1(close_idx, t1)
    return pd.Series(
        _concurrency(len(close_idx), start, end), index=close_idx, name="num_concurrent"
    )


def get_sample_uniqueness(close_idx: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    start, end, idx = _intervals_from_t1(close_idx, t1)
    conc = _concurrency(len(close_idx), start, end)
    return pd.Series(_avg_uniqueness(start, end, conc), index=idx, name="avg_uniqueness")


def get_sample_weights(
    close: pd.Series, t1: pd.Series, config: BootstrapConfig = BootstrapConfig()
) -> pd.Series:
    r"""Return-attribution sample weights per AFML snippet 4.10.

    Canonical formula [López de Prado 2018, §4.8, p.64]:

    .. math::
        w_i \;\propto\; \left| \sum_{t \in [t_{0,i},\, t_{1,i}]}
                        \frac{r_t}{c_t} \right|

    where :math:`r_t = \ln(p_t / p_{t-1})` is the log-return of bar ``t``
    and :math:`c_t` is the number of events concurrent at bar ``t``.

    The signed sum-then-absolute-value ordering means mean-reverting
    events (signed returns cancelling within the event horizon) receive
    near-zero weight, and each contribution is down-weighted by its
    per-bar concurrency. This is the correctness fix the P0.3 spec
    mandated.

    Parameters
    ----------
    close : pd.Series with DatetimeIndex
        Monotonic-increasing close prices.
    t1 : pd.Series with DatetimeIndex
        Event-end labels: ``t1.index`` is the event-start DatetimeIndex,
        ``t1.values`` is the event-end DatetimeIndex.
    config : BootstrapConfig
        ``min_weight`` floors each weight post-formula (guards
        division-by-zero-like downstream paths, not the formula itself).
        ``normalize_weights_to_n`` rescales so ``sum(w) == N``.

    Returns
    -------
    pd.Series
        Weights indexed by event-start timestamp.

    Raises
    ------
    ValueError
        If every raw weight is zero (signed-sum cancellation across all
        events). Normalisation is then impossible and silent handling
        would mask pathological event/return data. Treat as an upstream
        data-construction bug.
    """
    close_idx = pd.DatetimeIndex(close.index)
    start, end, idx = _intervals_from_t1(close_idx, t1)
    conc = _concurrency(len(close_idx), start, end)

    close_values = close.to_numpy(np.float64, copy=False)
    # Log-returns: ret[t] = log(close[t]) - log(close[t-1]). ret[0] is
    # undefined (no prior bar) and set to 0.0 by convention; callers
    # whose events include bar 0 thereby include a zero contribution,
    # which is the only correct behaviour given no prior reference.
    ret = np.empty(len(close_values), dtype=np.float64)
    ret[0] = 0.0
    if len(close_values) > 1:
        ret[1:] = np.log(close_values[1:]) - np.log(close_values[:-1])

    w = _afml_weights(ret, start, end, conc)

    # All-zero edge case: every event's signed-weighted sum cancels
    # exactly. Pathological; fail loudly rather than divide-by-zero or
    # silently emit uniform weights. Note: individual zero-weight events
    # are a valid AFML outcome (e.g. mean-reverting events) and are NOT
    # clamped — preserving ``w_i == 0`` exactly is an invariant.
    # ``config.min_weight`` is intentionally *not* applied on the AFML
    # path; it remains a parameter of the legacy (broken) path only.
    if float(w.sum()) <= 0.0:
        raise ValueError(
            "get_sample_weights: all events sum to zero weight; "
            "check event construction or returns data"
        )

    if config.normalize_weights_to_n:
        w = w * (len(w) / w.sum())
    return pd.Series(w, index=idx, name="sample_weight")


def _get_sample_weights_legacy_broken(
    close: pd.Series, t1: pd.Series, config: BootstrapConfig = BootstrapConfig()
) -> pd.Series:
    r"""DO NOT USE. Legacy (incorrect) implementation of AFML snippet 4.10.

    Uses :math:`\text{uniq}_i \cdot |p_{\text{end}} / p_{\text{start}} - 1|`
    instead of the canonical :math:`|\sum_t r_t / c_t|`. The two formulas
    differ on magnitude (A), zigzag paths (B), mean-reverting events
    (C: broken gives non-zero, correct gives 0), and round-trip prices
    (D: broken gives 0, correct is non-zero). See AFML §4.8, p.64.

    Preserved as a numerical oracle for regression testing only. The
    private underscore prefix and the ``_legacy_broken`` suffix are
    deliberate — no production code should import this. Will be removed
    after the conformal-integration sprint (S6+) alongside the P0.1
    structural-breaks shims.

    Emits ``DeprecationWarning`` so accidental imports surface in CI.
    """
    warnings.warn(
        "_get_sample_weights_legacy_broken is not a public API. "
        "Use get_sample_weights (AFML snippet 4.10) for production.",
        DeprecationWarning,
        stacklevel=2,
    )
    close_idx = pd.DatetimeIndex(close.index)
    start, end, idx = _intervals_from_t1(close_idx, t1)
    conc = _concurrency(len(close_idx), start, end)
    uniq = _avg_uniqueness(start, end, conc)
    rets = _abs_returns(close.to_numpy(np.float64, copy=False), start, end)
    w = np.maximum(uniq * rets, config.min_weight)
    if config.normalize_weights_to_n:
        w = w * (len(w) / w.sum())
    return pd.Series(w, index=idx, name="sample_weight")


def seq_bootstrap(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    sample_length: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    start, end, _ = _intervals_from_t1(close_idx, t1)
    if sample_length is None:
        sample_length = len(start)
    rng = np.random.default_rng(random_state)
    return _seq_bootstrap(start, end, int(sample_length), rng.random(int(sample_length)))
