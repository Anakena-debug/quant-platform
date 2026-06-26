"""DatabentoTradeFeed — satisfies DataFeedProtocol against Databento Live.

Two transport paths share the same record→TradeEvent mapping (amended
D5 of s36-real-feeds-and-async-broker):

- *Live transport.* ``databento.Live(key=...)`` opens a TCP session;
  callbacks fire in an SDK-owned worker thread. The callback enqueues
  records onto a bounded ``asyncio.Queue`` via
  ``loop.call_soon_threadsafe(self._enqueue, record)``.
  ``asyncio.Queue`` is NOT threadsafe across threads;
  ``call_soon_threadsafe`` is load-bearing — NOT
  ``asyncio.run_coroutine_threadsafe``, which has fire-and-forget-
  wrong semantics for enqueue (returns ``concurrent.futures.Future``
  with cancellation machinery, ~10x cost per call). The async iterator
  awaits ``queue.get()``.

- *Replay transport.* ``DatabentoTradeFeed.from_dbn_file(path)``
  bypasses the network. ``databento.read_dbn(path)`` decodes the
  committed fixture via the SDK decoder and emits the SAME
  ``TradeMsg`` instances the live callback would deliver. The
  record→TradeEvent mapping is shared.

**Backpressure (D5 + D7).** When the bounded queue is full, the
callback catches ``asyncio.QueueFull``, increments
``feed.metrics.backpressure_drops``, and returns immediately — the
dropped record is the *newest* (the one being enqueued). Symmetric
with S35 D8 engine-side drop-newest.

**Reconnection (D7).** SDK-native via ``reconnect_policy='reconnect'``
+ ``slow_reader_behavior='skip'`` on ``databento.Live``. The adapter
does NOT wrap the SDK with handrolled retry logic; that path was
killed in the D7 amendment.

**TradeEvent shape.** Yields ``_DatabentoTradeEvent`` instances that
structurally satisfy ``TradeEventLike`` (S35 protocols.py). Defined
locally so quantengine does not import a concrete quantcore type;
mirrors the S35 ``_demo.py`` ``_FakeTradeEvent`` precedent.

**L1 firewall (S33 D5 / S35 AC2).** No ``quantcore.book`` imports.
No ``quantcore.data.events`` imports either — the local dataclass
satisfies the structural protocol without binding to the concrete
quantcore enum.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import databento as db
import databento_dbn as ddbn

from quantengine.runtime.streaming.databento_config import DatabentoConfig
from quantengine.runtime.streaming.protocols import DataFeedProtocol, TradeEventLike

if TYPE_CHECKING:
    from collections.abc import Iterator

_FIXED_PRICE_SCALE: Final[int] = 1_000_000_000
_DEFAULT_QUEUE_SIZE: Final[int] = 4096
_DEFAULT_RECONNECT_POLICY: Final[str] = "reconnect"
_DEFAULT_SLOW_READER: Final[str] = "skip"
_DEFAULT_SCHEMA: Final[str] = "tbbo"

# DBN spec: ``side`` is the side of the AGGRESSOR for trades ('A'=Ask, 'B'=Bid,
# 'N'=None). Ask-side aggressor = SELLER (-1); bid-side aggressor = BUYER (+1).
# Matches quantcore.data.events.Side (BID=+1, ASK=-1) and the batch
# quantcore.features.top_of_book._SIDE_MAP. (s83 F11: pre-s83 both sites were
# inverted together — 'A' mis-read as "lifted ask = buyer".)
_SIDE_A: Final[int] = -1  # 'A' = ask-side aggressor = seller (ASK, -1)
_SIDE_B: Final[int] = 1  # 'B' = bid-side aggressor = buyer (BID, +1)


@dataclass(frozen=True, slots=True)
class _DatabentoTradeEvent:
    """Local TradeEvent shape; structurally satisfies ``TradeEventLike``.

    Defined here rather than importing from quantcore so the adapter
    does not bind to a concrete quantcore enum (per S35 _demo.py
    precedent). ``aggressor_side`` is typed ``int`` per the structural
    protocol — +1 (buyer aggressor), -1 (seller aggressor), or 0
    (no-aggressor; rare in equities, e.g., crosses).

    BBO fields carry the top-of-book state at trade time (TBBO
    contract). NaN when the source schema is ``trades`` (no BBO).
    """

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    bid_sz: float = float("nan")
    ask_sz: float = float("nan")


@dataclass(slots=True)
class FeedMetrics:
    """Counters and gauges observable on a running feed.

    ``backpressure_drops`` increments per ``asyncio.QueueFull`` drop
    in the SDK-callback → queue bridge (amended D5 + D7).
    ``reconnects_total`` increments per SDK reconnect callback (D7).
    ``last_disconnect_duration_s`` is a placeholder — populated when
    the disconnect/reconnect timestamp pair is wired through the SDK
    reconnect callback (post-PR2 hardening).
    """

    backpressure_drops: int = 0
    reconnects_total: int = 0
    last_disconnect_duration_s: float = 0.0


def _side_to_aggressor(side: object) -> int:
    """Map ``TradeMsg.side`` ('A'/'B'/'N' as str/bytes/int) → signed int.

    Convention (DBN: side = side of the aggressor):
        'B' (bid-side aggressor; buyer)      →  +1
        'A' (ask-side aggressor; seller)     →  -1
        anything else / 'N'                  →   0
    """
    if isinstance(side, bytes):
        side_char: str = side.decode("ascii", errors="replace")
    elif isinstance(side, int):
        try:
            side_char = chr(side)
        except (ValueError, OverflowError):
            return 0
    else:
        side_char = str(side)
    if side_char == "A":
        return _SIDE_A
    if side_char == "B":
        return _SIDE_B
    return 0


def _record_to_trade_event(record: object) -> _DatabentoTradeEvent | None:
    """Map a ``databento_dbn`` record to a local TradeEvent; return None for non-trades.

    Non-trade records (heartbeats, symbol mappings, error frames) are
    silently filtered. The adapter's iterator simply advances to the
    next record without yielding.

    For MBP1Msg (TBBO schema), BBO fields are extracted from
    ``record.levels[0]``. For TradeMsg (trades schema), BBO defaults
    to NaN.
    """
    if not isinstance(record, (ddbn.TradeMsg, ddbn.MBP1Msg)):
        return None

    bid_px = float("nan")
    ask_px = float("nan")
    bid_sz = float("nan")
    ask_sz = float("nan")

    if isinstance(record, ddbn.MBP1Msg) and hasattr(record, "levels") and record.levels:
        lvl = record.levels[0]
        raw_bid = int(lvl.bid_px)
        raw_ask = int(lvl.ask_px)
        if raw_bid != 0x7FFF_FFFF_FFFF_FFFF:
            bid_px = raw_bid / _FIXED_PRICE_SCALE
        if raw_ask != 0x7FFF_FFFF_FFFF_FFFF:
            ask_px = raw_ask / _FIXED_PRICE_SCALE
        bid_sz = float(lvl.bid_sz)
        ask_sz = float(lvl.ask_sz)

    return _DatabentoTradeEvent(
        ts_event=int(record.ts_event),
        instrument_id=int(record.instrument_id),
        sequence=int(record.sequence),
        price=int(record.price) / _FIXED_PRICE_SCALE,
        size=float(record.size),
        aggressor_side=_side_to_aggressor(record.side),
        bid_px=bid_px,
        ask_px=ask_px,
        bid_sz=bid_sz,
        ask_sz=ask_sz,
    )


class DatabentoTradeFeed:
    """``DataFeedProtocol`` adapter for the Databento Live trades schema.

    Construct for live mode by passing a ``DatabentoConfig`` and
    subscription parameters. Construct for replay mode (hermetic
    tests) via the ``from_dbn_file`` classmethod.

    The class is an async iterator; the engine consumes it as
    ``async for event in feed:``.
    """

    def __init__(
        self,
        config: DatabentoConfig,
        dataset: str,
        symbols: Sequence[str],
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        reconnect_policy: str = _DEFAULT_RECONNECT_POLICY,
        slow_reader_behavior: str = _DEFAULT_SLOW_READER,
        schema: str = _DEFAULT_SCHEMA,
    ) -> None:
        self._metrics = FeedMetrics()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_size)
        self._closed = False
        self._started = False
        self._replay_iter: Iterator[object] | None = None
        self._client: db.Live | None = db.Live(
            key=config.api_key,
            reconnect_policy=reconnect_policy,
            slow_reader_behavior=slow_reader_behavior,
        )
        self._client.subscribe(
            dataset=dataset,
            schema=schema,
            symbols=list(symbols),
        )
        self._client.add_callback(self._on_record)
        self._client.add_reconnect_callback(self._on_reconnect)

    @classmethod
    def from_dbn_file(cls, path: Path | str) -> DatabentoTradeFeed:
        """Build a replay-mode feed from a DBN file (amended D5)."""
        self = cls.__new__(cls)
        self._metrics = FeedMetrics()
        self._loop = None
        self._queue = asyncio.Queue()
        self._closed = False
        self._started = True  # replay starts on construction
        self._client = None
        store = db.read_dbn(str(path))
        self._replay_iter = iter(store)
        return self

    @property
    def metrics(self) -> FeedMetrics:
        return self._metrics

    def _on_record(self, record: object) -> None:
        """SDK callback (runs in SDK-owned worker thread).

        Bridges to the loop via ``call_soon_threadsafe``. See module
        docstring + amended D5 for the call_soon_threadsafe vs
        run_coroutine_threadsafe rationale.
        """
        if self._loop is None or self._closed:
            return
        self._loop.call_soon_threadsafe(self._enqueue, record)

    def _enqueue(self, record: object) -> None:
        """Loop-thread enqueue helper: count QueueFull as backpressure drop."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._metrics.backpressure_drops += 1

    def _on_reconnect(self, *args: object, **kwargs: object) -> None:
        """SDK reconnect callback.

        Argument signature varies across SDK versions; accept anything
        and just bump the counter. last_disconnect_duration_s is left
        at 0.0 in PR2; populating it requires capturing the
        disconnect-side timestamp, which is post-PR2 hardening.
        """
        del args, kwargs
        self._metrics.reconnects_total += 1

    def __aiter__(self) -> AsyncIterator[TradeEventLike]:
        return self

    async def __anext__(self) -> TradeEventLike:
        if self._replay_iter is not None:
            return await self._next_replay()
        return await self._next_live()

    async def _next_replay(self) -> TradeEventLike:
        assert self._replay_iter is not None
        while True:
            try:
                record = next(self._replay_iter)
            except StopIteration:
                raise StopAsyncIteration
            event = _record_to_trade_event(record)
            if event is not None:
                # Cast through object: frozen dataclass + Protocol-with-implicit-
                # writable-attr is a known type-system variance gap; mirrors the
                # S35 _demo.py _FakeTradeEvent return pattern.
                return cast(TradeEventLike, cast(object, event))

    async def _next_live(self) -> TradeEventLike:
        assert self._client is not None
        if not self._started:
            self._loop = asyncio.get_running_loop()
            self._client.start()
            self._started = True
        while True:
            record = await self._queue.get()
            event = _record_to_trade_event(record)
            if event is not None:
                return cast(TradeEventLike, cast(object, event))

    async def aclose(self) -> None:
        """Graceful shutdown: stop the SDK worker (live mode) or drop refs (replay)."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            await asyncio.to_thread(self._client.stop)


def _runtime_protocol_check() -> None:
    """Static structural check: DatabentoTradeFeed satisfies DataFeedProtocol.

    Called from tests via ``runtime_checkable``-based isinstance check.
    Existence here ensures the protocol import isn't dead-code-pruned
    by future refactors.
    """
    _ = DataFeedProtocol


__all__ = [
    "DatabentoTradeFeed",
    "FeedMetrics",
]
