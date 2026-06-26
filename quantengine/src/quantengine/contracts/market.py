"""Market snapshot passed to RebalanceEngine / Broker at each event."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Reference prices at rebalance time.

    For Phase 1 we use a single reference price per ticker (close, mid, or
    VWAP estimate). Phase 3 may extend with bid/ask/size.
    """

    timestamp: str
    tickers: tuple[str, ...]
    prices: FloatArray
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prices.shape != (len(self.tickers),):
            raise ValueError(f"prices.shape={self.prices.shape}, expected ({len(self.tickers)},)")
        if np.any(self.prices <= 0):
            raise ValueError("prices must be strictly positive")

    def price_of(self, ticker: str) -> float:
        i = self.tickers.index(ticker)
        return float(self.prices[i])
