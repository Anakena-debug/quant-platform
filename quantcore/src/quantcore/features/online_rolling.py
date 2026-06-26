"""Online (incremental) rolling flow features — byte-parity with build_flow.

The flow_only model consumes, per raw flow column (``signed_vol_imb`` /
``signed_dollar_imb`` / ``signed_tick_imb``), the rolling transforms that
``alpha_research.features.build_flow`` computes with pandas ``.rolling`` over the
full bar history::

    f[col_sum5]  = col.rolling(5).sum()      # min_periods=5  -> NaN warmup
    f[col_sum10] = col.rolling(10).sum()
    f[col_sum20] = col.rolling(20).sum()
    f[col_z20]   = _safe_z(col, 20)          # rolling mean/std(ddof=1) + cascade

where ``_safe_z`` is::

    mu  = col.rolling(20, min_periods=20).mean()
    std = col.rolling(20, min_periods=20).std()             # ddof=1 (sample)
    z   = (col - mu) / std.replace(0.0, np.nan)
    z   = z.replace([inf, -inf], nan).fillna(0.0)           # whole series

The live inference loop must compute these INCREMENTALLY per closed bar.
``OnlineRollingFlow`` reproduces the above exactly, in O(1) amortized per
update.

Native overlay (s50)
--------------------
The public ``OnlineRollingFlow`` is the Rust/PyO3 kernel
``quantcore_native.OnlineRollingFlow`` when it is installed (an opt-in dev/prod
overlay), and falls back to the pure-Python ``_OnlineRollingFlowPy`` otherwise.
Both are byte-parity (atol 1e-12) with ``build_flow``; the pure-Python class is
the permanent fallback AND the parity oracle. The public import path
(``from quantcore.features.online_rolling import OnlineRollingFlow``), the
``SUM_WINDOWS`` / ``Z_WINDOW`` exports, and the ``update`` dict contract are
unchanged, so live consumers need no edits. Set ``QUANTCORE_NATIVE=0`` to force
the pure-Python path (A/B benchmarking, incident rollback) without uninstalling
the native module.

Parity-critical design choices
------------------------------
1. **Recompute from a fixed window, NOT incremental sum-of-squares.** Windows
   are <= 20, so recomputing the trailing-window sum/mean/std each update is
   O(1); crucially it matches numpy/pandas float results to atol 1e-12, whereas
   an incremental SoS drifts from pandas' summation over thousands of bars and
   would break byte-parity.
2. **Sample std (ddof=1)** to match ``pandas.Series.rolling.std``.
3. **Warmup asymmetry (verified against pandas):**
   - ``sum{w}`` is ``NaN`` for the first ``w-1`` updates (min_periods=w).
   - ``z20`` is ``0.0`` (not NaN) during warmup, because ``_safe_z``'s final
     ``.fillna(0.0)`` covers the whole series including the warmup rows. A
     zero-variance full window also yields ``0.0`` (std 0 -> NaN -> fillna 0.0).
   Reproducing this asymmetry is the whole point: a naive implementation that
   NaN-fills z20 warmup would NOT match ``build_flow``'s column.
4. **NaN inputs.** pandas ``.rolling`` treats a NaN value as present-but-NaN: a
   window containing a NaN yields NaN for sum (min_periods met but NaN
   propagates) and the _safe_z cascade ultimately 0.0-fills it. We mirror pandas
   by keeping NaN in the window and letting numpy nan-propagate, then applying
   the same z cascade. (Flow ratios from bar_flow_ratios are always finite, so
   this is defensive; the parity test pins it anyway.)
"""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

# Rolling-sum windows produced by build_flow, in output order.
SUM_WINDOWS: tuple[int, ...] = (5, 10, 20)
Z_WINDOW: int = 20
_MAXLEN: int = max(Z_WINDOW, *SUM_WINDOWS)


class _OnlineRollingFlowPy:
    """Pure-Python streaming rolling transforms for ONE flow feature.

    Byte-parity (atol 1e-12) with ``build_flow``. This is the permanent fallback
    when the native kernel is unavailable AND the oracle the native kernel is
    pinned against (s50 ``test_online_rolling_native_parity.py``). The logic is
    unchanged since s42 — only the class name (was ``OnlineRollingFlow``).

    Construct one per raw flow column. Call :meth:`update` with each closed
    bar's raw value; it returns a dict with the latest
    ``{sum5, sum10, sum20, z20}`` exactly as ``build_flow`` would compute for
    that row given all values seen so far.
    """

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        # Holds the last _MAXLEN values (newest at the right).
        self._buf: deque[float] = deque(maxlen=_MAXLEN)

    def reset(self) -> None:
        """Clear all state (e.g. at a session boundary)."""
        self._buf.clear()

    def update(self, value: float) -> dict[str, float]:
        """Ingest one bar's raw flow value; return the latest rolling features.

        Returns ``{"sum5", "sum10", "sum20", "z20"}``. ``sum{w}`` is ``NaN``
        until ``w`` values have been seen; ``z20`` is ``0.0`` until the 20-value
        window is full (and ``0.0`` on a zero-variance window), matching
        ``build_flow`` / ``_safe_z`` exactly.
        """
        v = float(value)
        self._buf.append(v)
        buf = self._buf
        n = len(buf)

        out: dict[str, float] = {}

        # Rolling sums: trailing w values; NaN until the window is full
        # (pandas min_periods=w). Recompute over exactly the last w.
        for w in SUM_WINDOWS:
            if n < w:
                out[f"sum{w}"] = math.nan
            else:
                # last w elements of the deque
                window = list(buf)[-w:]
                out[f"sum{w}"] = float(np.sum(window))

        # z20 via the _safe_z cascade.
        out["z20"] = self._z20()
        return out

    def _z20(self) -> float:
        """(value - rolling_mean) / rolling_std(ddof=1) with the _safe_z cascade.

        Warmup (n < 20) and the zero-std / non-finite branches all resolve to
        0.0 — exactly ``_safe_z``'s ``.fillna(0.0)`` over the whole series.
        """
        buf = self._buf
        if len(buf) < Z_WINDOW:
            return 0.0  # _safe_z fillna(0.0) covers warmup rows
        window = np.asarray(list(buf)[-Z_WINDOW:], dtype=np.float64)
        latest = window[-1]
        mu = float(np.mean(window))
        std = float(np.std(window, ddof=1))  # pandas rolling.std is ddof=1
        if std == 0.0:
            return 0.0  # std.replace(0.0, nan) -> nan -> fillna(0.0)
        z = (latest - mu) / std
        if not math.isfinite(z):
            return 0.0  # replace([inf,-inf], nan) -> fillna(0.0)
        return z


# --- native overlay selection ------------------------------------------------
# Import the compiled kernel if present. `is not None` narrowing below keeps the
# selected class type free of `| None` so basedpyright stays clean at the
# assignment site.
try:
    from quantcore_native import OnlineRollingFlow as _OnlineRollingFlowNative
except ImportError:
    _OnlineRollingFlowNative = None

#: True when the Rust/PyO3 kernel is importable. The s50 parity test asserts
#: this (fail-closed) so CI cannot go green without the native build.
_NATIVE_AVAILABLE: bool = _OnlineRollingFlowNative is not None

if _OnlineRollingFlowNative is not None and os.environ.get("QUANTCORE_NATIVE") != "0":
    OnlineRollingFlow = _OnlineRollingFlowNative
else:
    OnlineRollingFlow = _OnlineRollingFlowPy


__all__ = ["OnlineRollingFlow", "SUM_WINDOWS", "Z_WINDOW"]
