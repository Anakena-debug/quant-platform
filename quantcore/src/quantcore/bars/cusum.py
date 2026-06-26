"""Online symmetric CUSUM filter on bar-close log-returns (S34 §3.AC5).

Reproduces ``quantcore.labels.labelling._cusum_core`` semantics one
event at a time: symmetric ``s_pos`` / ``s_neg`` accumulator with
reset on every threshold crossing. Threshold is caller-supplied
(causal per S33 §5.D7); the filter does NOT calibrate.
"""

# pyright: reportImplicitOverride=false

from __future__ import annotations

import math

from quantcore.data import Bar, BaseEvent


class OnlineCUSUMFilter:
    """Streaming variant of ``cusum_filter``.

    Accepts ``Bar`` events; uses ``Bar.close`` as the input series.
    Returns ``event.ts_event`` on threshold crossing, else ``None``.
    Non-Bar events are no-ops.
    """

    def __init__(self, threshold: float) -> None:
        if threshold <= 0.0:
            raise ValueError(
                f"threshold must be > 0; got {threshold!r}. "
                + "Threshold is in cumulative log-return units (e.g. 0.02 "
                + "≈ 2% cumulative move)."
            )
        self._threshold: float = float(threshold)
        self._prev_log_close: float | None = None
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0

    def on_event(self, event: BaseEvent) -> int | None:
        if not isinstance(event, Bar):
            return None

        close = float(event.close)
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError(
                "OnlineCUSUMFilter: Bar.close must be strictly positive "
                + f"and finite; got {close!r}"
            )

        log_close = math.log(close)
        if self._prev_log_close is None:
            # First sample: log_diff = 0 (matches legacy seed).
            x = 0.0
        else:
            x = log_close - self._prev_log_close
        self._prev_log_close = log_close

        self._s_pos = self._s_pos + x
        if self._s_pos < 0.0:
            self._s_pos = 0.0
        self._s_neg = self._s_neg + x
        if self._s_neg > 0.0:
            self._s_neg = 0.0

        if self._s_pos > self._threshold or self._s_neg < -self._threshold:
            self._s_pos = 0.0
            self._s_neg = 0.0
            return int(event.ts_event)
        return None

    def reset(self) -> None:
        self._prev_log_close = None
        self._s_pos = 0.0
        self._s_neg = 0.0
