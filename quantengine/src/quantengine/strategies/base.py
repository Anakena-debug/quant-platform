"""Strategy ABC.

A concrete Strategy wraps a frozen quantcore package (pipelines + primary +
meta + conformal calibrator) and exposes two methods:

    predict(t, market) -> AlphaSignal
    update(y_realized)  -> None     # online calibration only

No fitting. No CV. No feature engineering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal


class Strategy(ABC):
    @abstractmethod
    def predict(self, market: MarketSnapshot) -> AlphaSignal:
        """Produce an AlphaSignal for the given market snapshot."""

    def update(self, realized: dict[str, float]) -> None:
        """Feed realized returns back into an online conformal calibrator.

        Default no-op. Strategies backed by quantcore's ACI should override.
        """
        return None
