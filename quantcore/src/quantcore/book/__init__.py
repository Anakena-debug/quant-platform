"""quantcore.book — research-only L3 order-book state machines (S33).

Per S33 §5.D5–D6: NOT on the live L1 inference path. Downstream
production code (S34 streaming bars, S35 deployable features, model
inference) MUST NOT import from this package.

Public surface:
- ``BookStateMachine`` — abstract base for any L3 book implementation.
- ``OrderBook`` — canonical sorted-price-level implementation.
- ``BookCrossedError`` / ``UnknownOrderError`` / ``DuplicateOrderError``
  — sprint-local exceptions raised by ``OrderBook.apply``.
"""

from quantcore.book._abc import (
    BookCrossedError,
    BookStateMachine,
    DuplicateOrderError,
    UnknownOrderError,
)
from quantcore.book.order_book import OrderBook

__all__ = [
    "BookCrossedError",
    "BookStateMachine",
    "DuplicateOrderError",
    "OrderBook",
    "UnknownOrderError",
]
