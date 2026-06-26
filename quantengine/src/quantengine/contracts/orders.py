"""Order / Fill / Trade contracts.

Deliberately minimal: these objects flow from RebalanceEngine → Broker and
Broker → Ledger. Broker-specific fields (e.g., IBKR permId) live in
`metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    MOC = "MOC"  # market-on-close
    LOO = "LOO"  # limit-on-open
    STOP = "STOP"  # stop / stop-loss: market once the stop price is touched
    STOP_LIMIT = "STOP_LIMIT"  # becomes a LIMIT once the stop price is touched
    TRAIL = "TRAIL"  # trailing stop: trigger trails the favorable extreme
    TRAIL_LIMIT = "TRAIL_LIMIT"  # becomes a LIMIT once the trailing trigger is touched


class OrderStatus(str, Enum):
    """Order lifecycle states.

    Transition graph (see ``quantengine.execution.order_state.OrderTracker``
    for the canonical legal-transition table):

        PENDING     ── submit ──▶  SUBMITTED
        PENDING     ── reject ──▶  REJECTED
        SUBMITTED   ── ack    ──▶  WORKING
        SUBMITTED   ── reject ──▶  REJECTED
        SUBMITTED   ── fill   ──▶  PARTIALLY_FILLED | FILLED   (sync broker shortcut)
        SUBMITTED   ── cancel ──▶  CANCELLED
        WORKING     ── fill   ──▶  PARTIALLY_FILLED | FILLED
        WORKING     ── cancel ──▶  CANCELLED
        PARTIALLY_FILLED ── fill   ──▶  PARTIALLY_FILLED | FILLED
        PARTIALLY_FILLED ── cancel ──▶  CANCELLED

    FILLED, CANCELLED, REJECTED are terminal.

    ``WORKING`` exists to distinguish "broker has acknowledged but no fill
    yet" from "submitted to broker but no ack received". The synchronous
    ``PaperBroker`` collapses SUBMITTED → FILLED (skipping WORKING); the
    asynchronous IBKR adapter in Phase 3 will use all states.
    """

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class Order:
    """Discrete trade instruction; always integer shares for US equities."""

    order_id: UUID
    ticker: str
    side: OrderSide
    quantity: int  # absolute, always > 0
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None  # trigger price for STOP / STOP_LIMIT
    trail_amount: Optional[float] = None  # TRAIL: absolute trail distance ($/share)
    trail_percent: Optional[float] = (
        None  # TRAIL: trail distance as a percent (IBKR conv: 2.0 = 2%)
    )
    limit_offset: Optional[float] = None  # TRAIL_LIMIT: limit = effective stop -/+ this offset
    timestamp: Optional[str] = None
    parent_signal_ts: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order.quantity must be > 0, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP order requires stop_price")
        if self.order_type == OrderType.STOP_LIMIT and (
            self.stop_price is None or self.limit_price is None
        ):
            raise ValueError("STOP_LIMIT order requires both stop_price and limit_price")
        if self.order_type in (OrderType.TRAIL, OrderType.TRAIL_LIMIT):
            amt, pct = self.trail_amount, self.trail_percent
            if (amt is None) == (pct is None):
                raise ValueError(
                    f"{self.order_type.value} order requires exactly one of "
                    "trail_amount or trail_percent"
                )
            if amt is not None and amt <= 0:
                raise ValueError("trail_amount must be > 0")
            if pct is not None and pct <= 0:
                raise ValueError("trail_percent must be > 0")
            if self.order_type == OrderType.TRAIL_LIMIT and (
                self.limit_offset is None or self.limit_offset < 0
            ):
                raise ValueError("TRAIL_LIMIT order requires limit_offset >= 0")

    @property
    def signed_quantity(self) -> int:
        """Signed shares: + for BUY, - for SELL. Used in state reducer."""
        return self.quantity if self.side == OrderSide.BUY else -self.quantity

    @classmethod
    def new(
        cls,
        ticker: str,
        signed_quantity: int,
        order_type: OrderType = OrderType.MARKET,
        **kwargs: object,
    ) -> "Order":
        if signed_quantity == 0:
            raise ValueError("Cannot build an Order for zero shares")
        side = OrderSide.BUY if signed_quantity > 0 else OrderSide.SELL
        return cls(
            order_id=uuid4(),
            ticker=ticker,
            side=side,
            quantity=abs(signed_quantity),
            order_type=order_type,
            **kwargs,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )


@dataclass(frozen=True, slots=True)
class Fill:
    """A confirmed execution slice for a given Order."""

    fill_id: UUID
    order_id: UUID
    ticker: str
    signed_quantity: int  # + for BUY fill, - for SELL fill
    price: float  # execution price per share
    commission: float  # absolute, always >= 0
    timestamp: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.signed_quantity * self.price

    @property
    def cash_delta(self) -> float:
        """Cash change from this fill: -(notional) - commission."""
        return -self.notional - self.commission


@dataclass(frozen=True, slots=True)
class Trade:
    """Round-trip view: one order + its (possibly multiple) fills."""

    order: Order
    fills: tuple[Fill, ...]

    @property
    def filled_quantity(self) -> int:
        return sum(f.signed_quantity for f in self.fills)

    @property
    def avg_fill_price(self) -> float:
        q = sum(abs(f.signed_quantity) for f in self.fills)
        if q == 0:
            return 0.0
        return sum(f.price * abs(f.signed_quantity) for f in self.fills) / q

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)
