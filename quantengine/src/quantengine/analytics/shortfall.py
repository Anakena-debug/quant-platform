"""Implementation shortfall (IS) analytics.

Theory (Perold 1988; Almgren & Chriss 2001)
-------------------------------------------
Given a parent decision: at decision-time :math:`t_0`, reference price
:math:`p_0`, signed target quantity :math:`q^\\star`; and a set of
realized fills :math:`\\{(q_k, p_k, c_k)\\}_{k=1}^{K}` (signed quantity,
execution price, commission), the dollar implementation shortfall is

.. math::

    \\mathrm{IS}_{\\$} \\;=\\; \\sum_{k=1}^{K} q_k \\big(p_k - p_0\\big)
                              + \\sum_{k=1}^{K} c_k
                              + \\underbrace{(q^\\star - Q)\\,
                                (p_T - p_0)}_{\\text{missed-trade cost}}

where :math:`Q = \\sum_k q_k` is the total filled quantity and
:math:`p_T` is the reference price at the final fill (or session close).

The IS in basis points of notional is

.. math::

    \\mathrm{IS}_{\\mathrm{bps}} \\;=\\;
        10\\,000 \\cdot
        \\frac{\\mathrm{IS}_{\\$}}{|q^\\star|\\,p_0}

Decomposition into buckets (research convention):

=====================  =========================================================
Bucket                 Formula (signed $ cost)
=====================  =========================================================
``price_impact``       :math:`\\sum_k q_k (p_k - p_0)`  (execution drift)
``commission``         :math:`\\sum_k c_k`               (explicit fees)
``missed``             :math:`(q^\\star - Q)\\,(p_T - p_0)` (opportunity cost of
                        unfilled residual, approximated by final ref price)
=====================  =========================================================

(Spread vs impact vs delay separation requires bid/ask snapshots we
don't carry in ``MarketSnapshot`` yet — that decomposition lands with
P1-#8, the snapshot-v2 work. For now, ``price_impact`` lumps spread +
impact + delay together.)

Usage
-----
.. code-block:: python

    from quantengine.analytics import compute_shortfall

    reports = compute_shortfall(
        ledger=ledger,
        decision_prices={
            # map: order_id -> reference price at decision time
            order_id: ref_price,
            ...
        },
        final_prices=None,  # optional: final close per ticker for missed-trade
    )
    # reports: list[ImplementationShortfall], one per parent order with fills

The caller supplies ``decision_prices`` because the ledger alone does
not know which price was the "decision-time reference" — that's a
strategy-time quantity captured upstream. In the common case where the
``PaperBroker`` records ``reference_price`` in ``Fill.metadata``, a
helper (``decision_prices_from_metadata``) extracts them automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.ledger import Ledger


# ---------------------------------------------------------------------------
# Report type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImplementationShortfall:
    """IS report for one parent order.

    Attributes
    ----------
    order_id          : parent order UUID.
    ticker            : ticker traded.
    target_quantity   : signed shares the order intended to trade.
    filled_quantity   : signed shares actually filled.
    decision_price    : reference price at rebalance/decision time.
    avg_fill_price    : weighted-avg execution price (signed-qty weighted;
                        undefined if filled_quantity == 0).
    final_price       : reference price at report time (close / last ref),
                        used for missed-trade cost. ``None`` means skip.
    price_impact_usd  : $ cost from execution drift vs decision_price.
    commission_usd    : $ commission paid across all fills.
    missed_usd        : $ cost of unfilled residual at final_price.
    total_usd         : sum of the three buckets.
    total_bps         : total_usd / (|target_quantity| * decision_price) * 1e4;
                        NaN if |target|*price == 0.
    """

    order_id: UUID
    ticker: str
    target_quantity: int
    filled_quantity: int
    decision_price: float
    avg_fill_price: float
    final_price: float | None
    price_impact_usd: float
    commission_usd: float
    missed_usd: float
    total_usd: float
    total_bps: float

    @property
    def fill_rate(self) -> float:
        if self.target_quantity == 0:
            return 0.0
        return self.filled_quantity / self.target_quantity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decision_prices_from_metadata(ledger: Ledger) -> dict[UUID, float]:
    """Best-effort extraction of decision prices from fill metadata.

    ``PaperBroker`` stores ``reference_price`` (the mid/close used at
    decision time) in ``Fill.metadata``. For each order_id we take the
    reference of the *first* fill — that's the snapshot price the
    rebalance saw. If the broker you plug in doesn't record this, you
    must supply ``decision_prices`` manually to ``compute_shortfall``.
    """
    out: dict[UUID, float] = {}
    for e in ledger.events():
        if e.kind != "ORDER_FILLED" or not isinstance(e.payload, Fill):
            continue
        f = e.payload
        if f.order_id in out:
            continue  # first-fill wins
        ref = f.metadata.get("reference_price") if f.metadata else None
        if ref is not None:
            out[f.order_id] = float(ref)
    return out


def compute_shortfall(
    ledger: Ledger,
    *,
    decision_prices: Mapping[UUID, float] | None = None,
    final_prices: Mapping[str, float] | None = None,
) -> list[ImplementationShortfall]:
    """Compute per-order implementation shortfall.

    Parameters
    ----------
    ledger          : source of orders + fills.
    decision_prices : order_id → decision-time reference price. If
                      omitted, falls back to
                      ``decision_prices_from_metadata(ledger)``.
    final_prices    : ticker → closing/final ref price for the missed-
                      trade calculation. If absent, ``missed_usd = 0``
                      and the bucket is not computed (caveat in
                      ``ImplementationShortfall.missed_usd``).

    Returns
    -------
    One ``ImplementationShortfall`` per parent order observed in the
    ledger (both fully and partially filled). Orders with zero fills
    are skipped (no execution drift to measure; use fill-rate reports
    instead).
    """
    if decision_prices is None:
        decision_prices = decision_prices_from_metadata(ledger)
    final_prices = final_prices or {}

    # Index orders and fills by order_id.
    orders: dict[UUID, Order] = {}
    fills_by_order: dict[UUID, list[Fill]] = defaultdict(list)
    for e in ledger.events():
        if e.kind == "ORDER_SUBMITTED" and isinstance(e.payload, Order):
            orders[e.payload.order_id] = e.payload
        elif e.kind == "ORDER_FILLED" and isinstance(e.payload, Fill):
            fills_by_order[e.payload.order_id].append(e.payload)

    reports: list[ImplementationShortfall] = []
    for oid, fills in fills_by_order.items():
        if not fills:
            continue
        order = orders.get(oid)
        ticker = order.ticker if order is not None else fills[0].ticker
        target = (
            order.signed_quantity if order is not None else sum(f.signed_quantity for f in fills)
        )
        p0 = decision_prices.get(oid)
        if p0 is None or p0 <= 0:
            # Without a decision price there's nothing to compare against.
            # Skip — callers can see this by the absence of an entry for
            # the order_id in the returned list.
            continue

        filled = sum(f.signed_quantity for f in fills)
        total_abs = sum(abs(f.signed_quantity) for f in fills)
        # Signed-qty-weighted average (what you actually paid / received per share,
        # with sign preserved so BUY shows a positive notional out).
        avg_fill_price = (
            sum(f.price * abs(f.signed_quantity) for f in fills) / total_abs
            if total_abs > 0
            else 0.0
        )

        # Price-impact: Σ q_k (p_k - p0)
        price_impact = sum(f.signed_quantity * (f.price - p0) for f in fills)
        commission = sum(f.commission for f in fills)

        # Missed-trade: (target - filled) * (p_T - p0), only if we have p_T.
        p_final = final_prices.get(ticker)
        if p_final is not None and p_final > 0:
            missed = (target - filled) * (p_final - p0)
        else:
            missed = 0.0

        total_usd = price_impact + commission + missed
        notional_decision = abs(target) * p0
        total_bps = (
            10_000.0 * total_usd / notional_decision if notional_decision > 0 else float("nan")
        )

        reports.append(
            ImplementationShortfall(
                order_id=oid,
                ticker=ticker,
                target_quantity=int(target),
                filled_quantity=int(filled),
                decision_price=float(p0),
                avg_fill_price=float(avg_fill_price),
                final_price=(float(p_final) if p_final is not None else None),
                price_impact_usd=float(price_impact),
                commission_usd=float(commission),
                missed_usd=float(missed),
                total_usd=float(total_usd),
                total_bps=float(total_bps),
            )
        )
    return reports


__all__ = [
    "ImplementationShortfall",
    "compute_shortfall",
    "decision_prices_from_metadata",
]
