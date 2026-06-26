"""Internal: PriceLevel — resting orders at a single price tick.

Composition member of ``OrderBook``. Tracks ``order_id -> size`` for
O(1) add/remove/modify. Arrival order is preserved by Python 3.7+ dict
insertion-order semantics (queue position will be exposed as a
research-only feature in S35; not part of the S33 public contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantcore.data.events import Side


@dataclass(slots=True)
class PriceLevel:
    price: float
    side: Side
    _orders: dict[int, float] = field(default_factory=dict)

    @property
    def total_size(self) -> float:
        return sum(self._orders.values())

    @property
    def is_empty(self) -> bool:
        return not self._orders

    @property
    def order_count(self) -> int:
        return len(self._orders)

    def add(self, order_id: int, size: float) -> None:
        self._orders[order_id] = size

    def remove(self, order_id: int) -> None:
        del self._orders[order_id]

    def get_size(self, order_id: int) -> float:
        return self._orders[order_id]

    def set_size(self, order_id: int, size: float) -> None:
        self._orders[order_id] = size

    def has(self, order_id: int) -> bool:
        return order_id in self._orders
