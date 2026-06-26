"""Data contracts shared at the quantcore ↔ quantengine boundary.

Every class here is a frozen dataclass or pydantic model — no behaviour beyond
pure derivation from stored fields. Keep this module dependency-light:
numpy + stdlib only.
"""

from quantengine.contracts.signal import AlphaSignal
from quantengine.contracts.orders import Order, OrderSide, OrderType, Fill, Trade
from quantengine.contracts.market import MarketSnapshot

__all__ = [
    "AlphaSignal",
    "Order",
    "OrderSide",
    "OrderType",
    "Fill",
    "Trade",
    "MarketSnapshot",
]
