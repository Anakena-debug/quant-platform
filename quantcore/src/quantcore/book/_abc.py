"""BookStateMachine ABC + sprint-local exceptions (S33 §3.AC2, §5.D4).

Research/simulation only per S33 §5.D5 — never imported into
production-path modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantcore.data.events import BaseEvent, BookSnapshot


class BookCrossedError(Exception):
    """An ``ADD`` would produce ``best_bid >= best_ask``."""


class UnknownOrderError(Exception):
    """``CANCEL`` / ``MODIFY`` / ``FILL`` references an unknown ``order_id``."""


class DuplicateOrderError(Exception):
    """``ADD`` reuses an ``order_id`` already resting on either side."""


class BookStateMachine(ABC):
    """Abstract base for L3 order-book state machines.

    Subclasses implement event-driven mutation (``apply``) and snapshot
    emission (``snapshot``). The uniform ``apply(BaseEvent)`` signature
    lets any future book implementation drop in without call-site
    changes (S33 §3.AC3, AC4).

    ``best_bid`` / ``best_ask`` expose prices only; size at the top
    level is reachable via ``snapshot(depth=1)``.
    """

    @abstractmethod
    def apply(self, event: BaseEvent) -> None: ...

    @abstractmethod
    def snapshot(self, depth: int | None = None) -> BookSnapshot: ...

    @property
    @abstractmethod
    def best_bid(self) -> float | None: ...

    @property
    @abstractmethod
    def best_ask(self) -> float | None: ...
