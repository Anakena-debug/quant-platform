"""Demo broker + synthetic feed for self-contained S35 testing (D5).

Two reference implementations of the S35 protocols, providing an
end-to-end stack with no external network or broker:

- ``DemoBroker`` satisfies ``AsyncBrokerProtocol``. Holds an internal
  ``VirtualPortfolio`` (per D5: the ml4t-live "demo broker without
  position tracking causes infinite-buy-loop" pitfall is avoided
  precisely because we DO track positions). Fills at the configured
  reference price (constructor-provided or last-known per ticker).
  After each fill, the internal portfolio is updated synchronously
  inside ``submit_order`` before returning — AC10 atomicity invariant.
- ``SyntheticTradeFeed`` satisfies ``DataFeedProtocol``. Deterministic
  ``seed``-driven random walk yielding ``_FakeTradeEvent`` (a frozen
  dataclass that structurally satisfies ``TradeEventLike``). Supports
  a fixed ``n_events`` count or unbounded mode with caller-triggered
  stop.

quantcore-independence: no quantcore imports. The synthetic trade
type is defined inline.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from typing import cast

from quantengine.contracts.orders import Fill, Order, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.protocols import TradeEventLike


# ---------------------------------------------------------------------------
# Synthetic trade event (structurally satisfies TradeEventLike)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SyntheticTrade:
    """Inline trade-event shape for the demo feed.

    Satisfies ``TradeEventLike`` structurally (same attribute set as
    quantcore's ``TradeEvent``). Defined locally so the streaming
    package stays quantcore-independent.

    BBO fields default to NaN (synthetic feed has no book).
    """

    ts_event: int
    instrument_id: int
    sequence: int
    price: float
    size: float
    aggressor_side: int  # 1 = BID, -1 = ASK; matches quantcore Side.value
    bid_px: float = float("nan")
    ask_px: float = float("nan")
    bid_sz: float = float("nan")
    ask_sz: float = float("nan")


# ---------------------------------------------------------------------------
# SyntheticTradeFeed (S35 D5, AC9 determinism)
# ---------------------------------------------------------------------------
class SyntheticTradeFeed:
    """Deterministic seeded ``DataFeedProtocol`` for self-contained smoke tests.

    Parameters
    ----------
    seed : int
        ``random.Random(seed)`` is the only randomness source; two
        feeds constructed with the same seed yield byte-identical
        event sequences (AC9: test_determinism_seed_42).
    instrument_id : int
        Single instrument the feed covers in S35 MVP.
    n_events : int | None
        Total events to yield; ``None`` for unbounded. Default 1000.
    start_price : float
        Initial mid-price.
    start_ts_ns : int
        Initial event timestamp; subsequent events advance by
        ``tick_interval_ns``.
    tick_interval_ns : int
        Inter-event spacing on the synthetic clock. Default 1 ms.
    volatility : float
        Per-step log-return standard deviation (Gaussian increments).
    """

    def __init__(
        self,
        seed: int,
        instrument_id: int = 1,
        n_events: int | None = 1000,
        start_price: float = 100.0,
        start_ts_ns: int = 0,
        tick_interval_ns: int = 1_000_000,  # 1 ms
        volatility: float = 0.0005,  # ~0.05% per tick
    ) -> None:
        if start_price <= 0:
            raise ValueError(f"start_price must be > 0; got {start_price}")
        if volatility < 0:
            raise ValueError(f"volatility must be >= 0; got {volatility}")
        self._rng = random.Random(seed)
        self._instrument_id = instrument_id
        self._n_events = n_events
        self._start_price = start_price
        self._start_ts_ns = start_ts_ns
        self._tick_interval_ns = tick_interval_ns
        self._volatility = volatility
        self._index = 0
        self._price = start_price

    def __aiter__(self) -> "SyntheticTradeFeed":
        return self

    async def __anext__(self) -> TradeEventLike:
        if self._n_events is not None and self._index >= self._n_events:
            raise StopAsyncIteration
        # Yield control briefly so an external shutdown can preempt.
        await asyncio.sleep(0)
        # Geometric random walk on price.
        step = self._rng.gauss(0.0, self._volatility)
        self._price = max(0.01, self._price * (1.0 + step))
        side = 1 if self._rng.random() < 0.5 else -1
        size = float(self._rng.randint(1, 100))
        ev = _SyntheticTrade(
            ts_event=self._start_ts_ns + self._index * self._tick_interval_ns,
            instrument_id=self._instrument_id,
            sequence=self._index,
            price=round(self._price, 4),
            size=size,
            aggressor_side=side,
        )
        self._index += 1
        # why: basedpyright doesn't infer structural Protocol conformance
        # for frozen+slotted dataclasses; _SyntheticTrade satisfies
        # TradeEventLike by construction (runtime-checked by isinstance
        # in tests/test_streaming_protocols.py). Two-stage cast through
        # object is basedpyright's documented escape hatch for
        # reportInvalidCast.
        return cast(TradeEventLike, cast(object, ev))


# ---------------------------------------------------------------------------
# DemoBroker (S35 D5)
# ---------------------------------------------------------------------------
class DemoBroker:
    """Async ``AsyncBrokerProtocol`` implementation; fills at reference
    price; updates internal ``VirtualPortfolio`` synchronously per fill.

    The reference price comes from an injected ``price_lookup`` callable
    (typically a closure over the engine's last-price map). MARKET
    orders fill at the lookup result; LIMIT orders fill at
    ``order.limit_price`` if a lookup price is also available (no
    price gating in MVP — limit price is used as the trade price).

    State (cash + positions) is updated inside ``submit_order`` before
    the fill is returned, satisfying AC10 (position atomicity after
    submit).

    Parameters
    ----------
    starting_cash : float
        Initial portfolio cash. Default 1_000_000.0.
    price_lookup : Callable[[str], float | None]
        Returns the current reference price for a ticker, or ``None``
        if unknown. If ``None``, ``submit_order`` returns an empty
        fill list (no price -> can't construct a fill).
    commission_per_share : float
        Flat per-share commission. Default 0.0 (paper-style).
    """

    def __init__(
        self,
        starting_cash: float = 1_000_000.0,
        price_lookup: "PriceLookup | None" = None,
        commission_per_share: float = 0.0,
    ) -> None:
        if starting_cash < 0:
            raise ValueError(f"starting_cash must be >= 0; got {starting_cash}")
        if commission_per_share < 0:
            raise ValueError(f"commission_per_share must be >= 0; got {commission_per_share}")
        self._state = PortfolioState(cash=starting_cash, positions={})
        self._price_lookup = price_lookup if price_lookup is not None else (lambda t: None)
        self._commission_per_share = commission_per_share
        self._open_orders: dict[UUID, Order] = {}
        self._lock = asyncio.Lock()

    @property
    def state(self) -> PortfolioState:
        """Synchronous accessor for the current portfolio state.

        Used by SafeBroker's ``state_provider`` to take a snapshot for
        pre-trade risk checks. Read-only — callers MUST NOT mutate the
        returned object (PortfolioState is frozen-slotted; the contract
        is enforced structurally)."""
        return self._state

    async def submit_order(self, order: Order) -> list[Fill]:
        async with self._lock:
            # Determine fill price.
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                fill_price = float(order.limit_price)
            else:
                price = self._price_lookup(order.ticker)
                if price is None:
                    return []
                fill_price = float(price)

            commission = abs(order.quantity) * self._commission_per_share
            fill = Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=fill_price,
                commission=commission,
                timestamp="demo",
            )
            # Apply atomically before returning -> AC10 invariant.
            self._state = self._state.apply(fill)
            return [fill]

    async def cancel_order(self, order_id: UUID) -> bool:
        async with self._lock:
            if order_id in self._open_orders:
                del self._open_orders[order_id]
                return True
            return False

    async def get_position(self, ticker: str) -> Position | None:
        return self._state.positions.get(ticker)

    async def get_account_state(self) -> PortfolioState:
        return self._state


# ---------------------------------------------------------------------------
# Type alias for the price lookup closure
# ---------------------------------------------------------------------------

PriceLookup = Callable[[str], "float | None"]


__all__ = [
    "DemoBroker",
    "PriceLookup",
    "SyntheticTradeFeed",
]
