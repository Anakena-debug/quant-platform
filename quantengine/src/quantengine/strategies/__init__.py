"""Strategy adapters.

A Strategy produces an AlphaSignal for a given timestamp + market snapshot.
quantengine does not train or fit — it only calls `.predict()` on frozen
quantcore objects.
"""

from quantengine.strategies.base import Strategy
from quantengine.strategies.frozen_flow_regime import FrozenFlowRegimeStrategy

__all__ = ["FrozenFlowRegimeStrategy", "Strategy"]
