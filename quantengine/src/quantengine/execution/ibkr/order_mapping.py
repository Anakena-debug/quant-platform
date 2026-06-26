"""Order ↔ ib_async {Contract, Order} mapping; ib_async fill → Fill.

The ib_async Order created by ``order_to_ib_order`` leaves
``orderId = 0`` so that ``IB.placeOrder`` auto-assigns it via
``client.getReqId()``. Client-side orderId generation is a known
source of duplicate-order rejections at the IBKR side.

Order-type mapping:

- ``MARKET`` → ``"MKT"``
- ``LIMIT`` → ``"LMT"`` (requires ``limit_price``; sets ``lmtPrice``)
- ``MOC`` → ``"MOC"`` (market-on-close)
- ``LOO`` → ``"LMT"`` + ``tif="OPG"`` (limit-on-open; requires
  ``limit_price``; sets ``lmtPrice`` and ``tif="OPG"``). IBKR rejects
  the literal ``"LOO"`` orderType — the gateway expects a LIMIT order
  with the OPG time-in-force tag, per the IBKR API order-type table.
  Verified via paper-account REPL 2026-05-11.
- ``STOP`` → ``"STP"`` (requires ``stop_price``; sets ``auxPrice``, the
  IBKR stop-trigger field).
- ``STOP_LIMIT`` → ``"STP LMT"`` (requires both ``stop_price`` and
  ``limit_price``; sets ``auxPrice`` = stop trigger and ``lmtPrice`` =
  the post-trigger limit), per the IBKR API order-type table.
- ``TRAIL`` → ``"TRAIL"`` (trailing stop; sets ``auxPrice`` = absolute
  trail distance OR ``trailingPercent`` = the trail percent — exactly
  one is present).
- ``TRAIL_LIMIT`` → ``"TRAIL LIMIT"`` (as ``TRAIL`` plus ``lmtPriceOffset``
  = ``limit_offset``, the offset of the limit from the trailing trigger),
  per the IBKR API order-type table.

Fill conversion preserves the signed-quantity polarity (BUY → positive,
SELL → negative). IBKR's Execution.shares is always positive; the side
is encoded in the parent Order, so we apply the sign here. Audit-trail
metadata records ``ib_perm_id``, ``ib_exec_id``, and ``ib_order_id``
for forensic traceability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType

if TYPE_CHECKING:
    from ib_async import Contract
    from ib_async import Fill as IBFill
    from ib_async import Order as IBOrder
    from ib_async import Trade


_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.MOC: "MOC",
    # LOO is a Limit order with tif="OPG"; "LOO" is NOT a valid IBKR
    # orderType string and is rejected at the gateway. The OPG tag is
    # appended downstream in order_to_ib_order().
    OrderType.LOO: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
    OrderType.TRAIL: "TRAIL",
    OrderType.TRAIL_LIMIT: "TRAIL LIMIT",
}


def order_to_ib_order(order: Order) -> tuple[Contract, IBOrder]:
    """Map our ``Order`` → ``(ib_async.Stock, ib_async.Order)``.

    The ib_async Order's ``orderId`` is left as 0 so that
    ``IB.placeOrder`` auto-assigns it via ``client.getReqId()``.

    Raises ``ValueError`` if a LIMIT/LOO order lacks a ``limit_price``, a
    STOP order lacks a ``stop_price``, or a STOP_LIMIT order lacks either.
    """
    from ib_async import Order as IBOrder
    from ib_async import Stock

    contract = Stock(order.ticker, "SMART", "USD")

    ib_order = IBOrder()
    ib_order.action = order.side.value  # "BUY" or "SELL"
    ib_order.totalQuantity = order.quantity  # always positive (abs)
    ib_order.orderType = _ORDER_TYPE_MAP[order.order_type]

    if order.order_type == OrderType.LIMIT:
        if order.limit_price is None:
            raise ValueError("LIMIT order requires limit_price; got None")
        ib_order.lmtPrice = order.limit_price
    elif order.order_type == OrderType.LOO:
        if order.limit_price is None:
            raise ValueError("LOO (limit-on-open) order requires limit_price; got None")
        ib_order.lmtPrice = order.limit_price
        ib_order.tif = "OPG"
    elif order.order_type == OrderType.STOP:
        if order.stop_price is None:
            raise ValueError("STOP order requires stop_price; got None")
        ib_order.auxPrice = order.stop_price  # IBKR stop-trigger field
    elif order.order_type == OrderType.STOP_LIMIT:
        if order.stop_price is None or order.limit_price is None:
            raise ValueError("STOP_LIMIT order requires stop_price and limit_price; got None")
        ib_order.lmtPrice = order.limit_price  # post-trigger limit
        ib_order.auxPrice = order.stop_price  # stop trigger
    elif order.order_type in (OrderType.TRAIL, OrderType.TRAIL_LIMIT):
        # Trailing distance is either absolute (auxPrice) or a percent
        # (trailingPercent), per the IBKR API order-type table.
        if order.trail_amount is not None:
            ib_order.auxPrice = order.trail_amount
        elif order.trail_percent is not None:
            ib_order.trailingPercent = order.trail_percent
        else:
            raise ValueError(
                f"{order.order_type.value} order requires trail_amount or trail_percent; got neither"
            )
        if order.order_type == OrderType.TRAIL_LIMIT:
            if order.limit_offset is None:
                raise ValueError("TRAIL_LIMIT order requires limit_offset; got None")
            ib_order.lmtPriceOffset = order.limit_offset  # limit trails the trigger by this offset

    # ib_order.orderId stays at 0 — IB.placeOrder assigns via getReqId().
    return contract, ib_order


def ib_trade_to_fill(
    trade: Trade,
    order: Order,
    fill_event: IBFill,
) -> Fill:
    """Convert an ``ib_async`` fill event into our ``Fill`` dataclass.

    Preserves signed-quantity polarity (BUY positive, SELL negative).
    IBKR's ``execution.shares`` is always positive; the sign comes
    from the parent ``order.side``.

    Records IBKR-side identifiers (``permId``, ``execId``,
    ``orderId``) in metadata for audit-trail traceability.
    """
    execution: Any = fill_event.execution
    commission_report: Any = fill_event.commissionReport

    sign = 1 if order.side == OrderSide.BUY else -1
    signed_quantity = sign * int(execution.shares)

    return Fill(
        fill_id=uuid4(),
        order_id=order.order_id,
        ticker=order.ticker,
        signed_quantity=signed_quantity,
        price=float(execution.price),
        commission=float(commission_report.commission),
        timestamp=str(execution.time),
        metadata={
            "ib_perm_id": int(execution.permId),
            "ib_exec_id": str(execution.execId),
            "ib_order_id": int(trade.order.orderId),
        },
    )


__all__ = ["ib_trade_to_fill", "order_to_ib_order"]
