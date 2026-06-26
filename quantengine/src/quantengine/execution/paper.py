"""PaperBroker — synchronous paper / replay broker.

Order-type-aware fills against the single reference price: MARKET/MOC fill unconditionally at
the cost-model price; LIMIT/LOO fill only when MARKETABLE (ref on the right side of the limit),
CAPPED at the limit, otherwise they REST in ``open_orders``. Phase 2 replay can subclass and
delay fills to next-open.

Resting STOP-family orders (STOP / STOP_LIMIT / TRAIL / TRAIL_LIMIT) are RE-EVALUATED every
``submit_orders`` call: each new snapshot sweeps the resting book before processing the fresh
batch, so a stop that was untouched on its submit bar can still trigger on a later one. TRAIL
orders trail a per-order water-mark (the favorable extreme since they began resting) held in
``_trail_state``. LIMIT/LOO remain submit-once — they are excluded from the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
from uuid import UUID, uuid4

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.cost_model import CostModel, LinearCostModel

# Order types whose triggers are re-checked against every later snapshot.
_STOP_FAMILY = (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAIL, OrderType.TRAIL_LIMIT)


@dataclass
class PaperBroker(AbstractBroker):
    cost_model: CostModel = field(default_factory=LinearCostModel)
    _open: list[Order] = field(default_factory=list)
    _trail_state: dict[UUID, float] = field(default_factory=dict)  # order_id → trailing water-mark

    def submit_orders(self, orders: Sequence[Order], market: MarketSnapshot) -> list[Fill]:
        price_map = {t: float(p) for t, p in zip(market.tickers, market.prices)}
        # 1. Re-evaluate already-resting stop-family orders against the new market.
        fills = self._sweep_resting(price_map, market)
        # 2. Process the freshly-submitted orders as today.
        for order in orders:
            ref = price_map.get(order.ticker)
            if ref is None:
                # Can't fill what we can't price. Mark as open (Phase 2 will
                # expose rejection reasons).
                self._open.append(order)
                continue
            fp = self._resolve_fill_price(order, ref)
            if fp is None:
                # LIMIT/LOO not marketable, or stop not triggered → rest the order.
                self._open.append(order)
                continue
            fills.append(self._make_fill(order, fp, ref, market))
        return fills

    def _sweep_resting(self, price_map: dict[str, float], market: MarketSnapshot) -> list[Fill]:
        """Re-check resting STOP-family orders against the new snapshot.

        STOP/STOP_LIMIT/TRAIL/TRAIL_LIMIT orders resting in ``_open`` are
        re-evaluated each bar: a TRAIL updates its water-mark, every stop
        recomputes its (possibly trailing) trigger, and any now-triggered order
        fills and is dropped from ``_open`` (and ``_trail_state``). LIMIT/LOO and
        unpriceable orders are submit-once and pass through untouched, preserving
        the s66 resting semantics byte-for-byte.
        """
        if not self._open:
            return []
        fills: list[Fill] = []
        still_open: list[Order] = []
        for order in self._open:
            ref = price_map.get(order.ticker)
            if order.order_type not in _STOP_FAMILY or ref is None:
                still_open.append(order)
                continue
            fp = self._resolve_stop_family(order, ref)
            if fp is None:
                still_open.append(order)  # still untriggered → keep resting
                continue
            fills.append(self._make_fill(order, fp, ref, market))
            self._trail_state.pop(order.order_id, None)
        self._open = still_open
        return fills

    def _make_fill(self, order: Order, fp: float, ref: float, market: MarketSnapshot) -> Fill:
        """Build a Fill at price ``fp`` for ``order`` against reference ``ref``."""
        return Fill(
            fill_id=uuid4(),
            order_id=order.order_id,
            ticker=order.ticker,
            signed_quantity=order.signed_quantity,
            price=fp,
            commission=self.cost_model.commission(order, fp),
            timestamp=market.timestamp,
            metadata={"reference_price": ref},
        )

    def _resolve_fill_price(self, order: Order, ref: float) -> float | None:
        """Order-type-aware fill price; ``None`` means "not marketable, rest the order".

        MARKET/MOC fill at the cost-model price. LIMIT/LOO fill only when the reference price is
        on the right side of the limit (BUY: ref<=limit, SELL: ref>=limit) and are capped at the
        limit (never pay above / sell below it). A LOO without a limit_price behaves as MARKET.

        STOP/STOP_LIMIT/TRAIL/TRAIL_LIMIT are delegated to ``_resolve_stop_family`` (trigger on
        the reference touching the — possibly trailing — stop, then market-style or LIMIT-capped).
        A fresh TRAIL only initialises its water-mark here and rests; it can never trigger on its
        own submit bar.
        """
        if order.order_type in _STOP_FAMILY:
            return self._resolve_stop_family(order, ref)
        cost_fill = self.cost_model.fill_price(order, ref)
        if order.order_type in (OrderType.MARKET, OrderType.MOC):
            return cost_fill
        if order.order_type in (OrderType.LIMIT, OrderType.LOO):
            limit = order.limit_price
            if limit is None:
                return cost_fill  # LOO with no limit → market-style (defensive)
            return self._limit_fill(order.side, cost_fill, ref, limit)
        return cost_fill  # unknown type → market-style (defensive)

    def _resolve_stop_family(self, order: Order, ref: float) -> float | None:
        """Trigger logic shared by STOP / STOP_LIMIT / TRAIL / TRAIL_LIMIT.

        Returns a fill price when the order's (possibly trailing) stop is triggered at ``ref``,
        else ``None`` (the order keeps resting). The trigger reuses the s71 side convention —
        BUY: ``ref >= stop`` (breakout / short-cover); SELL: ``ref <= stop`` (stop-loss). A
        triggered STOP/TRAIL fills market-style; a STOP_LIMIT/TRAIL_LIMIT applies the s66 LIMIT
        marketable+cap rule. For TRAIL/TRAIL_LIMIT the trailing water-mark is advanced FIRST (the
        sole ``_trail_state`` mutation, in ``_advance_trail_mark``), then the stop is derived purely
        from it. A misconfigured order (a ``None`` trigger — only reachable by bypassing
        ``__post_init__``, e.g. a recovery path that skips validation) RESTS rather than firing a
        market exit: resting is fail-closed, a market exit is the most destructive default.
        """
        if order.order_type in (OrderType.TRAIL, OrderType.TRAIL_LIMIT):
            mark = self._advance_trail_mark(order, ref)  # the one trailing-state mutation
            stop = self._trail_stop_from_mark(order, mark)  # pure
        else:  # STOP / STOP_LIMIT use the static trigger
            stop = order.stop_price
        if stop is None:
            return None  # misconfigured (no trigger) → rest, never market-fill
        triggered = ref >= stop if order.side == OrderSide.BUY else ref <= stop
        if not triggered:
            return None  # stop not touched → rest
        cost_fill = self.cost_model.fill_price(order, ref)
        if order.order_type in (OrderType.STOP, OrderType.TRAIL):
            return cost_fill  # triggered → market-style fill
        # STOP_LIMIT / TRAIL_LIMIT: post-trigger LIMIT (marketable + capped).
        if order.order_type == OrderType.STOP_LIMIT:
            limit = order.limit_price
        else:  # TRAIL_LIMIT: the limit sits limit_offset beyond the trailing trigger.
            off = order.limit_offset if order.limit_offset is not None else 0.0
            limit = stop - off if order.side == OrderSide.SELL else stop + off
        if limit is None:
            return None  # misconfigured (no limit) → rest, never market-fill
        return self._limit_fill(order.side, cost_fill, ref, limit)

    def _advance_trail_mark(self, order: Order, ref: float) -> float:
        """Advance and return the per-order trailing water-mark.

        The favorable extreme seen since the order began resting: the running HIGH for a SELL
        trail (protecting a long), the running LOW for a BUY trail (protecting a short). This is
        the SINGLE place ``_trail_state`` is mutated — ``_resolve_stop_family`` calls it as the
        explicit "advance trailing state" step before deriving the (pure) stop.
        """
        prev = self._trail_state.get(order.order_id)
        if order.side == OrderSide.SELL:  # protect a long: trail the running HIGH
            mark = ref if prev is None else max(prev, ref)
        else:  # protect a short: trail the running LOW
            mark = ref if prev is None else min(prev, ref)
        self._trail_state[order.order_id] = mark
        return mark

    @staticmethod
    def _trail_stop_from_mark(order: Order, mark: float) -> float | None:
        """Pure: the trailing trigger derived from a water-mark.

        Sits ``offset`` away from ``mark`` — ``offset`` is ``trail_amount`` or
        ``mark * trail_percent / 100`` — below it for a SELL trail, above for a BUY trail.
        Returns ``None`` only for a misconfigured trail (neither distance set; ``__post_init__``
        forbids that on a real order).
        """
        if order.trail_amount is not None:
            offset = order.trail_amount
        elif order.trail_percent is not None:
            offset = mark * order.trail_percent / 100.0
        else:
            return None
        return mark - offset if order.side == OrderSide.SELL else mark + offset

    @staticmethod
    def _limit_fill(side: OrderSide, cost_fill: float, ref: float, limit: float) -> float | None:
        """LIMIT marketable+cap: fill capped at ``limit`` when ref is on the right side, else None."""
        if side == OrderSide.BUY:
            return min(cost_fill, limit) if ref <= limit else None
        return max(cost_fill, limit) if ref >= limit else None

    def cancel_all(self) -> int:
        n = len(self._open)
        self._open.clear()
        self._trail_state.clear()
        return n

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a single resting order by id; ``True`` if it was open, else ``False``.

        Removes the order from ``_open`` and drops any trailing water-mark it held. Lets a caller
        retire the prior bar's protective stop before regenerating, so re-running
        ``RebalanceEngine.protective_stops`` each bar is idempotent rather than accumulating one
        duplicate resting stop per bar (a drawdown would otherwise fire all of them).
        """
        for i, o in enumerate(self._open):
            if o.order_id == order_id:
                del self._open[i]
                self._trail_state.pop(order_id, None)
                return True
        return False

    def open_orders(self) -> Iterable[Order]:
        return tuple(self._open)
