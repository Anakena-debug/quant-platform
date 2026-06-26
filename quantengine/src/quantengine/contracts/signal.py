"""AlphaSignal mirror — structural handoff from quantcore.

quantcore defines the authoritative AlphaSignal. We redeclare a *structurally
identical* frozen dataclass here so quantengine can be imported standalone
(e.g., inside unit tests without quantcore installed). Duck-typing at the
adapter boundary: any object with these fields satisfies us.

Math (see ARCHITECTURE.md):
    tradeable_i  = 1{ 0 ∉ [lower_i, upper_i] }
    direction_i  = sign(expected_return_i) * tradeable_i
    kelly_weight = Kelly fraction clipped to [-max_leverage, max_leverage],
                   computed by quantcore (NOT recomputed here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """Cross-sectional return forecast with prediction intervals.

    Attributes
    ----------
    tickers        : list of symbols, length N. Order is authoritative.
    expected_return: point forecast r_hat in R^N (e.g., next-period log return).
    lower, upper   : prediction interval bounds at miscoverage `alpha`.
    alpha          : conformal miscoverage rate in (0, 1).
    kelly_weights  : quantcore-computed target weights in [-L, L], length N.
                     If None, quantengine treats expected_return as the target
                     (NOT recommended — quantcore should always size).
    timestamp      : pandas-compatible timestamp at which signal was produced.
    metadata       : free-form dict for traceability (model version, run id,
                     mlflow run uuid, calibration state hash, etc.).
    """

    tickers: tuple[str, ...]
    expected_return: FloatArray
    lower: FloatArray
    upper: FloatArray
    alpha: float
    kelly_weights: FloatArray | None = None
    timestamp: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    # --- invariants -------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.tickers)
        for name, arr in (
            ("expected_return", self.expected_return),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name}.shape={arr.shape}, expected ({n},)")
        if self.kelly_weights is not None and self.kelly_weights.shape != (n,):
            raise ValueError(f"kelly_weights.shape={self.kelly_weights.shape}, expected ({n},)")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if np.any(self.lower > self.upper):
            raise ValueError("lower > upper at some index; interval ill-formed")

    # --- derived ----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.tickers)

    @property
    def tradeable(self) -> npt.NDArray[np.bool_]:
        """True where the prediction interval excludes zero."""
        return (self.lower > 0.0) | (self.upper < 0.0)

    @property
    def direction(self) -> IntArray:
        """{-1, 0, +1} per ticker."""
        d = np.sign(self.expected_return).astype(np.int64)
        return d * self.tradeable.astype(np.int64)

    def kelly_weight(self, max_leverage: float = 1.0) -> FloatArray:
        """Return Kelly weights, falling back to scaled sign(r_hat) if absent.

        If quantcore did not attach kelly_weights we return a crude proxy:
        direction * min(|r_hat| / sigma_interval, max_leverage / N). This is
        NOT a substitute for quantcore's sizing — it exists so quantengine is
        runnable in isolation for testing.
        """
        if self.kelly_weights is not None:
            return np.clip(self.kelly_weights, -max_leverage, max_leverage)
        # Fallback proxy. quantcore should always supply kelly_weights.
        sigma = np.maximum((self.upper - self.lower) / 2.0, 1e-9)
        raw = self.direction.astype(np.float64) * np.minimum(
            np.abs(self.expected_return) / sigma, max_leverage / max(self.n, 1)
        )
        return np.clip(raw, -max_leverage, max_leverage)

    @classmethod
    def from_quantcore(cls, obj: object) -> "AlphaSignal":
        """Structural copy from a quantcore.AlphaSignal duck-typed object."""
        return cls(
            tickers=tuple(getattr(obj, "tickers")),
            expected_return=np.asarray(getattr(obj, "expected_return"), dtype=np.float64),
            lower=np.asarray(getattr(obj, "lower"), dtype=np.float64),
            upper=np.asarray(getattr(obj, "upper"), dtype=np.float64),
            alpha=float(getattr(obj, "alpha")),
            kelly_weights=(
                np.asarray(getattr(obj, "kelly_weights"), dtype=np.float64)
                if getattr(obj, "kelly_weights", None) is not None
                else None
            ),
            timestamp=getattr(obj, "timestamp", None),
            metadata=dict(getattr(obj, "metadata", {})),
        )


def build_alpha_signal(
    tickers: Sequence[str],
    expected_return: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    alpha: float,
    kelly_weights: Sequence[float] | None = None,
    timestamp: str | None = None,
    metadata: dict[str, object] | None = None,
) -> AlphaSignal:
    """Convenience constructor accepting Python sequences."""
    return AlphaSignal(
        tickers=tuple(tickers),
        expected_return=np.asarray(expected_return, dtype=np.float64),
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        alpha=alpha,
        kelly_weights=(
            np.asarray(kelly_weights, dtype=np.float64) if kelly_weights is not None else None
        ),
        timestamp=timestamp,
        metadata=metadata or {},
    )
