"""OrderBook — canonical L3 state machine (S33 §3.AC5).

Research/simulation only per S33 §5.D5 — never imported into
production-path modules. Live L1 inference does not need a book; the
``TradeEvent`` stream is sufficient.
"""

# `typing.override` is 3.12+; project pins 3.11 and pyproject is locked
# for S33 (cannot pull `typing_extensions`). Suppress the
# implicit-override warning at file scope so the basedpyright triad
# gate runs clean. Drop this once the project floor lifts to 3.12+ and
# the ABC overrides can carry `@override`.
# pyright: reportImplicitOverride=false

from __future__ import annotations

import numpy as np

from quantcore.book._abc import (
    BookCrossedError,
    BookStateMachine,
    DuplicateOrderError,
    UnknownOrderError,
)
from quantcore.book._price_level import PriceLevel
from quantcore.data.events import (
    Action,
    BaseEvent,
    BookSnapshot,
    OrderEvent,
    Side,
    TradeEvent,
)


class OrderBook(BookStateMachine):
    """Sorted-price-level L3 book.

    Composition members:
    - ``_bids`` / ``_asks``: ``dict[price, PriceLevel]`` sides
    - ``_order_index``: ``dict[order_id, (side, price)]`` for O(1)
      CANCEL / MODIFY / FILL lookup
    - ``_last_ts``: timestamp of the most recently *successfully*
      applied book-mutating ``OrderEvent``; stamped onto every
      ``BookSnapshot``. ``TradeEvent`` and ``OrderEvent(action=TRADE)``
      do not advance this (AC7 Property 5).
    """

    def __init__(self, instrument_id: int, max_depth: int = 50) -> None:
        self._instrument_id: int = int(instrument_id)
        self._max_depth: int = int(max_depth)
        self._bids: dict[float, PriceLevel] = {}
        self._asks: dict[float, PriceLevel] = {}
        self._order_index: dict[int, tuple[Side, float]] = {}
        self._last_ts: int = 0

    # ------------------------------------------------------------------
    # ABC contract
    # ------------------------------------------------------------------

    def apply(self, event: BaseEvent) -> None:
        if event.instrument_id != self._instrument_id:
            msg = (
                f"event instrument_id {event.instrument_id} does not match "
                f"book instrument_id {self._instrument_id}"
            )
            raise ValueError(msg)
        if isinstance(event, TradeEvent):
            return
        if isinstance(event, OrderEvent):
            if event.action == Action.TRADE:
                return
            self._apply_order(event)
            self._last_ts = event.ts_event
            return

    @property
    def best_bid(self) -> float | None:
        if not self._bids:
            return None
        return max(self._bids)

    @property
    def best_ask(self) -> float | None:
        if not self._asks:
            return None
        return min(self._asks)

    def snapshot(self, depth: int | None = None) -> BookSnapshot:
        bid_prices = sorted(self._bids.keys(), reverse=True)
        ask_prices = sorted(self._asks.keys())
        if depth is not None:
            if depth < 0:
                raise ValueError(f"depth must be >= 0; got {depth}")
            bid_prices = bid_prices[:depth]
            ask_prices = ask_prices[:depth]
        bid_px = np.asarray(bid_prices, dtype=np.float64)
        bid_sz = np.asarray([self._bids[p].total_size for p in bid_prices], dtype=np.float64)
        ask_px = np.asarray(ask_prices, dtype=np.float64)
        ask_sz = np.asarray([self._asks[p].total_size for p in ask_prices], dtype=np.float64)
        return BookSnapshot(
            ts_event=self._last_ts,
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_order(self, event: OrderEvent) -> None:
        action = event.action
        if action == Action.ADD:
            self._add(event)
        elif action == Action.CANCEL:
            self._cancel(event)
        elif action == Action.MODIFY:
            self._modify(event)
        elif action == Action.FILL:
            self._fill(event)
        elif action == Action.CLEAR:
            self._clear()

    def _add(self, e: OrderEvent) -> None:
        if e.size <= 0:
            raise ValueError(f"ADD size must be > 0; got {e.size}")
        if e.order_id in self._order_index:
            raise DuplicateOrderError(f"order_id {e.order_id} already resting")
        if e.side == Side.BID:
            ba = self.best_ask
            if ba is not None and e.price >= ba:
                raise BookCrossedError(f"bid {e.price} >= best_ask {ba}")
            levels = self._bids
        else:
            bb = self.best_bid
            if bb is not None and e.price <= bb:
                raise BookCrossedError(f"ask {e.price} <= best_bid {bb}")
            levels = self._asks
        level = levels.get(e.price)
        if level is None:
            level = PriceLevel(price=e.price, side=e.side)
            levels[e.price] = level
        level.add(e.order_id, e.size)
        self._order_index[e.order_id] = (e.side, e.price)

    def _cancel(self, e: OrderEvent) -> None:
        side, price = self._lookup(e.order_id)
        levels = self._bids if side == Side.BID else self._asks
        level = levels[price]
        level.remove(e.order_id)
        if level.is_empty:
            del levels[price]
        del self._order_index[e.order_id]

    def _modify(self, e: OrderEvent) -> None:
        if e.size <= 0:
            raise ValueError(f"MODIFY size must be > 0; got {e.size}")
        side, price = self._lookup(e.order_id)
        if e.side != side:
            raise ValueError(f"MODIFY side {e.side} != resting side {side} for order {e.order_id}")
        if e.price != price:
            # s83 F12: price-changing MODIFY = cancel + add (loses queue
            # priority, per ITCH semantics). Previously ``e.price`` was
            # IGNORED — the order kept its stale level with the new size,
            # silently corrupting book state on every MBO replay carrying
            # price modifies. Residual (documented, queue-position scope):
            # an in-place size INCREASE should also lose priority;
            # ``set_size`` preserves dict position — irrelevant until
            # queue-position features ship.
            levels = self._bids if side == Side.BID else self._asks
            level = levels[price]
            level.remove(e.order_id)
            if level.is_empty:
                del levels[price]
            del self._order_index[e.order_id]
            self._add(e)  # re-validates duplicate + crossing at the new price
            return
        levels = self._bids if side == Side.BID else self._asks
        levels[price].set_size(e.order_id, e.size)

    def _fill(self, e: OrderEvent) -> None:
        if e.size <= 0:
            raise ValueError(f"FILL size must be > 0; got {e.size}")
        side, price = self._lookup(e.order_id)
        levels = self._bids if side == Side.BID else self._asks
        level = levels[price]
        resting = level.get_size(e.order_id)
        if e.size >= resting:
            level.remove(e.order_id)
            if level.is_empty:
                del levels[price]
            del self._order_index[e.order_id]
        else:
            level.set_size(e.order_id, resting - e.size)

    def _clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._order_index.clear()

    def _lookup(self, order_id: int) -> tuple[Side, float]:
        try:
            return self._order_index[order_id]
        except KeyError:
            raise UnknownOrderError(f"order_id {order_id} not resting") from None
