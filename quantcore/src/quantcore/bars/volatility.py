"""Online EWMA volatility on bar-close pct-change (S34 §3.AC6).

Reproduces ``close.pct_change().ewm(span=span, adjust=False).std()``
— the exact recurrence used by ``quantcore.labels.labelling.get_daily_vol``
— in O(1) per event.

Recurrence (verified against pandas 2.x at atol < 1e-12 on a hand
fixture):

    α          = 2 / (span + 1)
    ω          = 1 − α
    sum_wt²_t  = ω² · sum_wt²_{t−1} + α²        (with sum_wt²_1 = 1)
    μ_t        = ω · μ_{t−1} + α · r_t          (adjust=False mean)
    cov_t      = ω · (cov_{t−1} + α · (r_t − μ_{t−1})²)   (biased var)
    var_t      = cov_t / (1 − sum_wt²_t)        (bias-corrected; sum_wt=1)
    σ_t        = sqrt(var_t)                    if var_t > 0

The first sample (nobs=1) returns None — pandas convention for
single-sample variance. Non-finite or non-positive closes raise
ValueError (s83 F3): the previous "skip" path mutated ``_prev_close``
BEFORE validating, so one bad close silently destroyed TWO returns while
the docstring claimed "state unchanged". Fail-fast matches the
OnlineCUSUMFilter convention — a corrupt bar close in the live loop is a
data-integrity event, not something to average over. See S34 §6.R3 / R7.
"""

# pyright: reportImplicitOverride=false

from __future__ import annotations

import math

from quantcore.data import Bar, BaseEvent


class OnlineEWMAVolatility:
    """Streaming variant of ``get_daily_vol`` (bar-close-based)."""

    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError(f"span must be >= 1; got {span}")
        self._span: int = int(span)
        alpha = 2.0 / (span + 1.0)
        self._alpha: float = alpha
        self._omega: float = 1.0 - alpha
        self._prev_close: float | None = None
        self._nobs: int = 0
        self._sum_wt2: float = 0.0
        self._mu: float = 0.0
        self._cov: float = 0.0
        self._sigma: float | None = None

    @property
    def sigma(self) -> float | None:
        return self._sigma

    def on_event(self, event: BaseEvent) -> float | None:
        if not isinstance(event, Bar):
            return None

        close = float(event.close)
        # s83 F3: validate BEFORE any state mutation. The old path seeded /
        # advanced _prev_close first and then skipped on non-finite pct-change,
        # so a single bad close corrupted the chain (two returns lost) while
        # the docstring claimed the state was unchanged.
        if not math.isfinite(close) or close <= 0.0:
            raise ValueError(
                f"OnlineEWMAVolatility: Bar.close must be finite and > 0; got {close!r}"
            )
        if self._prev_close is None:
            # First bar: pct_change is undefined; state seeded.
            self._prev_close = close
            return None

        ret = (close - self._prev_close) / self._prev_close
        self._prev_close = close

        self._nobs += 1
        if self._nobs == 1:
            # First finite return: single-sample variance undefined.
            self._mu = ret
            self._cov = 0.0
            self._sum_wt2 = 1.0
            self._sigma = None
            return None

        # Recurrence
        self._sum_wt2 = self._omega * self._omega * self._sum_wt2 + self._alpha * self._alpha
        old_mean = self._mu
        self._mu = self._omega * self._mu + self._alpha * ret
        delta = ret - old_mean
        self._cov = self._omega * (self._cov + self._alpha * delta * delta)

        denom = 1.0 - self._sum_wt2
        if denom <= 0.0:
            self._sigma = None
            return None
        var = self._cov / denom
        if var <= 0.0:
            self._sigma = 0.0 if var == 0.0 else None
            return self._sigma
        self._sigma = math.sqrt(var)
        return self._sigma

    def reset(self) -> None:
        self._prev_close = None
        self._nobs = 0
        self._sum_wt2 = 0.0
        self._mu = 0.0
        self._cov = 0.0
        self._sigma = None
