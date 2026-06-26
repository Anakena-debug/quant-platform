"""Streaming runtime protocol contracts (S35 §3 / D9, D10).

This module defines protocol surfaces that make the streaming runtime
swappable while keeping ``quantengine`` package-independent of
``quantcore`` (matching the existing ``quantengine.contracts.signal``
pattern: structural typing at the boundary, no Python-level dependency).

Public protocols:

- ``StreamingStrategy`` — sync, user-authored; receives ``on_bar`` /
  ``on_cusum`` / ``on_vol`` callbacks (NO ``on_tick`` per D9).
- ``AsyncBrokerProtocol`` — async broker contract; ``DemoBroker``
  (S35) and ``AsyncIBKRBroker`` (S36) satisfy it.
- ``SyncBrokerFacade`` — what the strategy sees through
  ``ThreadSafeBrokerWrapper`` (sync over async).
- ``DataFeedProtocol`` — async iterator yielding ``TradeEventLike``.
- ``Clock`` (+ ``EventClock``, ``WallClock``) — abstracts "now".
- ``BarBuilderProtocol`` — streaming bar builder (e.g. quantcore's
  ``BarBuilder`` subclasses) injected into the engine.
- ``OnlineCUSUMFilterProtocol`` — streaming CUSUM detector (e.g.
  quantcore's ``OnlineCUSUMFilter``).
- ``OnlineEWMAVolatilityProtocol`` — streaming volatility estimator
  (e.g. quantcore's ``OnlineEWMAVolatility``).

Structural value types:

- ``TradeEventLike`` — duck-typed shape of ``quantcore.data.events.TradeEvent``.
- ``BarLike`` — duck-typed shape of ``quantcore.data.bars.Bar``.
- ``CUSUMEvent`` — wraps the bar-close + ts_event that
  ``OnlineCUSUMFilter`` reports on threshold crossing.
- ``StreamContext`` — per-callback metadata.
- ``BrokerTimeoutError`` — raised on D12 timeout.

Concrete ``quantcore`` classes satisfy these protocols at runtime
without ``quantengine`` ever importing ``quantcore``. The engine
accepts the protocols via constructor injection; the integration
seam lives one layer above this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable
from uuid import UUID

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.state import PortfolioState, Position


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BrokerTimeoutError(TimeoutError):
    """Raised by ``SyncBrokerFacade`` when a sync-over-async call exceeds
    the per-call timeout (S35 D12). Subclass of ``TimeoutError`` so that
    callers can catch either.
    """


# ---------------------------------------------------------------------------
# Structural value-type protocols (duck-typed shape of quantcore types)
# ---------------------------------------------------------------------------
@runtime_checkable
class TradeEventLike(Protocol):
    """Structural shape of ``quantcore.data.events.TradeEvent``.

    The streaming engine consumes events that look like a quantcore
    ``TradeEvent``: ``ts_event``, ``instrument_id``, ``sequence`` (from
    ``BaseEvent``) plus ``price``, ``size``, ``aggressor_side``.
    quantcore's concrete ``TradeEvent`` satisfies this Protocol without
    quantengine importing quantcore at all.

    The ``aggressor_side`` attribute is typed ``int`` here (it's a
    ``Side`` IntEnum on the quantcore side; ``int(Side.BID) == 1``,
    ``int(Side.ASK) == -1``). Treating it as ``int`` keeps quantengine
    independent of the quantcore enum while preserving signed semantics.
    """

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int


@runtime_checkable
class BarLike(Protocol):
    """Structural shape of ``quantcore.data.bars.Bar``.

    Mirrors the attributes the streaming engine reads off a bar
    (``ts_open``, ``open``/``high``/``low``/``close``, ``volume``,
    ``vwap``, ``tick_count``, ``dollar_volume``, plus the base-event
    triple). quantcore's ``Bar`` satisfies this Protocol structurally;
    quantengine never imports it.

    ``kind`` is typed ``int`` (quantcore's ``BarKind`` IntEnum). Cross-
    package boundary preserves the integer value, not the enum identity.
    """

    ts_event: int
    instrument_id: int
    sequence: int
    ts_open: int
    kind: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    dollar_volume: float


# ---------------------------------------------------------------------------
# Support types defined in quantengine (no upstream equivalent)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CUSUMEvent:
    """Emitted when an ``OnlineCUSUMFilterProtocol`` crosses its
    threshold.

    quantcore's ``OnlineCUSUMFilter.on_event`` returns the triggering
    bar's ``ts_event`` (or ``None``). The engine wraps the non-None case
    into this dataclass before dispatching ``on_cusum``, so the strategy
    receives a structured argument matching the D9 callback signature.

    Attributes
    ----------
    ts_event : int
        Bar timestamp at which the CUSUM accumulator crossed
        ``±threshold``.
    instrument_id : int
        Instrument identifier propagated from the triggering bar.
    bar_close : float
        Closing price of the bar that triggered the crossing.
    """

    ts_event: int
    instrument_id: int
    bar_close: float


@dataclass(frozen=True, slots=True)
class StreamContext:
    """Per-callback metadata.

    Carries enough information for the strategy to correlate the callback
    with engine state without grabbing it from globals.

    Attributes
    ----------
    sequence : int
        Engine-wide monotone event counter at the moment of dispatch.
    clock_ns : int
        ``Clock.now_ns()`` reading at dispatch time (event-time for
        replay/synthetic feeds, wall-clock for live feeds; S35 D10).
    instrument_id : int
        Identifier of the instrument the callback concerns.
    """

    sequence: int
    clock_ns: int
    instrument_id: int


# ---------------------------------------------------------------------------
# Clock protocol + reference implementations (S35 D10)
# ---------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    """Abstract source of ``now``. Engine, watchdog, and journal all
    consume this rather than calling ``time.time()`` directly so the
    same engine code drives replay and live modes.
    """

    def now_ns(self) -> int:
        """Return the current nanosecond timestamp under this clock."""
        ...


class EventClock:
    """Clock whose "now" advances with the event stream.

    The engine calls :meth:`tick` for each consumed event passing the
    event's ``ts_event``; :meth:`now_ns` returns the most recent value.
    Default clock for ``SyntheticTradeFeed`` and replay mode.

    Initial state returns ``0`` until the first event ticks the clock.
    """

    def __init__(self) -> None:
        self._now_ns: int = 0

    def tick(self, ts_event_ns: int) -> None:
        """Advance the clock to ``ts_event_ns``.

        Does NOT enforce monotonicity — out-of-order replays would
        otherwise raise here; the engine is responsible for ordering its
        input stream.
        """
        self._now_ns = int(ts_event_ns)

    def now_ns(self) -> int:
        return self._now_ns


class WallClock:
    """Clock backed by ``time.monotonic_ns()``.

    Default clock for real-time feeds (S36 onward). Monotonic across
    process lifetime; not affected by NTP adjustments.
    """

    def now_ns(self) -> int:
        return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Pipeline-primitive protocols (quantcore concrete classes satisfy these)
# ---------------------------------------------------------------------------
@runtime_checkable
class BarBuilderProtocol(Protocol):
    """Streaming bar builder.

    Matches quantcore's ``BarBuilder`` ABC: consumes a ``TradeEventLike``
    via :meth:`on_event`, returns a ``BarLike`` on bar-close trigger or
    ``None`` between bars. :meth:`flush` returns any partial bar at
    end-of-stream.

    The streaming engine accepts an instance of any class satisfying
    this Protocol; quantcore's ``TickBarBuilder`` /
    ``VolumeImbalanceBarBuilder`` / etc. satisfy it structurally.
    """

    def on_event(self, event: TradeEventLike) -> BarLike | None: ...

    def flush(self) -> BarLike | None: ...


@runtime_checkable
class OnlineCUSUMFilterProtocol(Protocol):
    """Streaming CUSUM detector consuming bar-close prices.

    Matches quantcore's ``OnlineCUSUMFilter`` API: :meth:`on_event`
    accepts a ``BarLike`` and returns the bar's ``ts_event`` on threshold
    crossing, ``None`` otherwise. :meth:`reset` clears accumulator state.
    """

    def on_event(self, event: BarLike) -> int | None: ...

    def reset(self) -> None: ...


@runtime_checkable
class OnlineEWMAVolatilityProtocol(Protocol):
    """Streaming EWMA volatility estimator consuming bar-close pct-changes.

    Matches quantcore's ``OnlineEWMAVolatility`` API: :meth:`on_event`
    returns the current sigma estimate (``float``) or ``None`` while
    warming up. :meth:`reset` clears the recurrence state.
    """

    def on_event(self, event: BarLike) -> float | None: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Broker protocols (S35 D2, D3, D12)
# ---------------------------------------------------------------------------
@runtime_checkable
class AsyncBrokerProtocol(Protocol):
    """Async broker contract.

    Adapters (``DemoBroker`` in S35; ``AsyncIBKRBroker`` in S36)
    implement this. Return semantics mirror
    ``quantengine.execution.broker.AbstractBroker`` cross-implementation
    invariants (CI-1 through CI-6): one parent order MAY produce
    multiple fills, callers MUST associate via ``Fill.order_id``, etc.
    """

    async def submit_order(self, order: Order) -> list[Fill]:
        """Submit a single order; return resulting fills (possibly empty)."""
        ...

    async def cancel_order(self, order_id: UUID) -> bool:
        """Cancel one open order by id. Return True iff cancellation
        was acknowledged by the broker.
        """
        ...

    async def get_position(self, ticker: str) -> Position | None:
        """Return the broker's authoritative position for ``ticker``,
        or ``None`` if the broker reports no position on that
        instrument.
        """
        ...

    async def get_account_state(self) -> PortfolioState:
        """Snapshot the broker's authoritative ``PortfolioState``."""
        ...


@runtime_checkable
class SyncBrokerFacade(Protocol):
    """Synchronous broker interface as seen by strategy code.

    ``ThreadSafeBrokerWrapper`` is the canonical implementation: it
    wraps an ``AsyncBrokerProtocol`` and bridges via
    ``asyncio.run_coroutine_threadsafe`` with per-call timeouts
    (D12 defaults: 5s for submit, 1s for get_position, 2s for
    get_account_state). Any call that exceeds its timeout raises
    :class:`BrokerTimeoutError`.

    Protocol typing exists so test code can inject a synchronous mock
    without instantiating the real wrapper.
    """

    def submit_order(self, order: Order, timeout: float | None = None) -> list[Fill]: ...

    def cancel_order(self, order_id: UUID, timeout: float | None = None) -> bool: ...

    def get_position(self, ticker: str, timeout: float | None = None) -> Position | None: ...

    def get_account_state(self, timeout: float | None = None) -> PortfolioState: ...


# ---------------------------------------------------------------------------
# Data feed (S35 D9)
# ---------------------------------------------------------------------------
class DataFeedProtocol(Protocol):
    """Async iterator yielding ``TradeEventLike``.

    Concrete implementations:

    - ``SyntheticTradeFeed`` (S35 PR4): deterministic synthetic feed.
    - ``DatabentoTradeFeed`` (S36): WSS-backed live feed.

    The engine consumes the feed under ``async for event in feed: ...``.
    ``StopAsyncIteration`` signals end-of-stream; for live feeds, the
    iterator should resume rather than stop on transient disconnects
    (S36 D7 backpressure + reconnection policy).
    """

    def __aiter__(self) -> AsyncIterator[TradeEventLike]: ...

    async def __anext__(self) -> TradeEventLike: ...


# ---------------------------------------------------------------------------
# Strategy (S35 D9: bar/cusum/vol callbacks, NO on_tick)
# ---------------------------------------------------------------------------
@runtime_checkable
class StreamingStrategy(Protocol):
    """User-authored alpha layer; receives meaningful summaries of the
    event stream and decides whether to place orders.

    Strategies never see raw L1 ticks (D9). The engine processes every
    tick internally — building bars, updating CUSUM, updating EWMA
    volatility — and only dispatches the resulting summaries.

    Parameters
    ----------
    ts : int
        Event timestamp (per ``StreamContext.clock_ns`` semantics).
    bar / event / sigma : payload
        The summary the callback fired on.
    ctx : StreamContext
        Per-callback metadata.
    broker : SyncBrokerFacade
        Synchronous broker interface; strategy may call
        ``broker.submit_order(order)`` and block on the result up to
        the D12 timeout.

    Implementations are sync. The engine dispatches them via
    ``asyncio.to_thread`` from its async loop (S35 D2 / AC4).
    """

    def on_bar(
        self,
        ts: int,
        bar: BarLike,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None: ...

    def on_cusum(
        self,
        ts: int,
        event: CUSUMEvent,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None: ...

    def on_vol(
        self,
        ts: int,
        sigma: float,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None: ...


__all__ = [
    "AsyncBrokerProtocol",
    "BarBuilderProtocol",
    "BarLike",
    "BrokerTimeoutError",
    "CUSUMEvent",
    "Clock",
    "DataFeedProtocol",
    "EventClock",
    "OnlineCUSUMFilterProtocol",
    "OnlineEWMAVolatilityProtocol",
    "StreamContext",
    "StreamingStrategy",
    "SyncBrokerFacade",
    "TradeEventLike",
    "WallClock",
]
