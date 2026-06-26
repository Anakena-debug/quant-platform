"""quantcore.data — typed market-event hierarchy (S33 + S34).

Deployment posture (S33 §5.D5–D6):

- ``TradeEvent`` is the canonical *production* event type.
- ``OrderEvent`` is research/simulation only. Live L1 feeds do not
  produce ``OrderEvent``; production code MUST NOT depend on it.
- ``BookSnapshot`` is research/simulation only (see ``quantcore.book``).

S34 adds the typed ``Bar`` shape emitted by streaming bar builders.
``Bar`` inherits from ``BaseEvent`` so it flows through the same
``on_event(BaseEvent)`` plumbing as TradeEvent.
"""

from quantcore.data.bars import Bar, BarKind
from quantcore.data.events import (
    Action,
    BaseEvent,
    BookSnapshot,
    OrderEvent,
    Side,
    TradeEvent,
)

__all__ = [
    "Action",
    "Bar",
    "BarKind",
    "BaseEvent",
    "BookSnapshot",
    "OrderEvent",
    "Side",
    "TradeEvent",
]
