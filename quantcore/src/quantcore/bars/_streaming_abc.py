"""BarBuilder ABC (S34 §3.AC2).

Streaming bar builders consume ``TradeEvent`` instances and emit
``Bar`` instances when their threshold / imbalance / runs condition is
met. The uniform ``on_event(BaseEvent) -> Bar | None`` signature
(S33 §3.AC3 plug-and-play contract) means swapping one builder for
another is a single-line change at the call site.
"""

# `typing.override` is 3.12+; project pins 3.11 and pyproject is locked
# for S34. Suppress reportImplicitOverride at file scope on every
# module that subclasses BarBuilder so the basedpyright triad gate
# stays clean. Drop these once the project floor lifts to 3.12+.
# pyright: reportImplicitOverride=false

from __future__ import annotations

from abc import ABC, abstractmethod

from quantcore.data import Bar, BaseEvent


class BarBuilder(ABC):
    """Abstract base for streaming information-driven bar samplers.

    Subclasses consume ``TradeEvent`` and emit ``Bar`` when their
    threshold trigger fires. Non-trade events are silently ignored.

    Per S33 §5.D5, the live-deployable path uses only ``TradeEvent``.
    No dependency on the research-only L3 book / order-state surface.
    """

    @abstractmethod
    def on_event(self, event: BaseEvent) -> Bar | None:
        """Apply event. Returns Bar on threshold trigger, else None."""

    @abstractmethod
    def flush(self) -> Bar | None:
        """Close any open partial bar (end-of-stream).

        Returns the partial bar if any trades have been buffered since
        the last emission, else None. Idempotent: a second call after
        a successful flush returns None.
        """
