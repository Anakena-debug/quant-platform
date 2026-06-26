"""Single-producer single-consumer async queue (S36b D3).

Replaces ``asyncio.Queue`` in the engine's feed → consumer hand-off.
The S35 engine has exactly one producer (``_feed_loop``) and one
consumer (``_consumer_loop``); ``asyncio.Queue``'s multi-producer /
multi-consumer machinery (Future allocation per put/get + getter-
waiter list management) is unused overhead — measured at ~8µs/event
aggregate in the 2026-05-22 cProfile session.

``SpscQueue`` is backed by ``collections.deque`` (append/popleft are
atomic in CPython per the GIL contract) plus a single
``asyncio.Event`` for non-empty signalling. The Event's Future is
amortised across the burst of events between empty-states (typically
100s–1000s of events per signal under load) vs ``asyncio.Queue``'s
per-event Future cost.

Drop-newest backpressure (mirrors ``asyncio.Queue``):
    ``put_nowait`` raises ``asyncio.QueueFull`` when at maxsize. The
    engine's existing ``except asyncio.QueueFull`` clause catches it
    and increments ``backpressure_drops`` (S35 D8 policy unchanged
    at the protocol surface).

Race-safety notes (SPSC is critical here):

  The Event-clear / append-set ordering matters. The producer MUST
  append THEN set; any consumer that wakes up sees the data.
  The consumer's get() does:

    1. if buf is non-empty → popleft, return
    2. else: clear the event (idempotent if already cleared)
    3. re-check buf: producer may have appended + set between
       the empty-check and the clear; if so, loop back to step 1
    4. await event.wait(); on wake, loop back to step 1

  The double-check around clear() closes the race window where a
  producer's set() happens between step 1's empty-check and step 2's
  clear() — without the re-check, the consumer would clear the just-
  set signal and then wait forever on an Event that no producer is
  about to set again.

SPSC invariant — NOT enforced at runtime:
    Multiple producers or multiple consumers will work but lose the
    fast-path guarantees and the race-analysis above. Drop-in
    compatibility with ``asyncio.Queue`` is limited to the
    ``put_nowait`` / ``get`` / ``qsize`` / ``empty`` surface; async
    ``put`` is intentionally NOT implemented — the engine's drop-
    newest is the explicit policy on full, and a blocking ``put``
    would re-introduce the Future-allocation cost SpscQueue exists
    to avoid.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Generic, TypeVar

_T = TypeVar("_T")


class SpscQueue(Generic[_T]):
    """Bounded single-producer single-consumer async queue.

    Construct with ``SpscQueue(maxsize=N)``. The bound is enforced by
    ``put_nowait``; ``get`` is unbounded-wait. See module docstring
    for the race-safety analysis.
    """

    __slots__ = ("_buf", "_maxsize", "_event")

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be > 0; got {maxsize}")
        self._buf: deque[_T] = deque()
        self._maxsize = maxsize
        self._event = asyncio.Event()

    def put_nowait(self, item: _T) -> None:
        """Enqueue without blocking; raise ``asyncio.QueueFull`` at maxsize.

        Producer-side path: append THEN set the event. Any consumer
        that wakes up sees the data.
        """
        if len(self._buf) >= self._maxsize:
            raise asyncio.QueueFull
        self._buf.append(item)
        self._event.set()

    async def get(self) -> _T:
        """Dequeue; await a producer signal when empty.

        Consumer-side path: see module docstring's race-safety notes.
        Cancellation-safe: ``CancelledError`` propagates cleanly
        without leaving the queue in a partial state (deque hasn't
        been mutated; event state is consistent with buffer state).
        """
        while True:
            if self._buf:
                item = self._buf.popleft()
                if not self._buf:
                    # Buffer drained; clear the event so the next
                    # empty-state get() blocks. If a producer set the
                    # event between popleft and clear, the next call
                    # to get() will see the non-empty buf at step 1
                    # and skip the wait.
                    self._event.clear()
                return item
            # Buffer empty; clear the event (idempotent) and re-check.
            self._event.clear()
            if self._buf:
                # Producer appended + set between our empty-check and
                # clear; loop back to pop it without waiting.
                continue
            await self._event.wait()

    def qsize(self) -> int:
        """Return the current queue length."""
        return len(self._buf)

    def empty(self) -> bool:
        """Return True iff the queue has no items."""
        return not self._buf


__all__ = ["SpscQueue"]
