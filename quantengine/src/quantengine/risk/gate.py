"""Pre-trade risk gate.

Role in the pipeline
--------------------
The ``RebalanceEngine`` projects a set of orders that *respects*
``RebalanceConstraints``. Those constraints are declarative knobs baked
into the rebalance math. The ``RiskGate`` is a *second* layer —
independent of rebalance — that inspects the actual ``list[Order]``
against the current ``PortfolioState`` and a ``MarketSnapshot`` and can
veto any order before it reaches the broker.

Two reasons for defence-in-depth:

1. A bug in ``RebalanceEngine`` (or in an upstream factor / screen /
   Kelly weight) can produce an order set that technically passes the
   rebalance math but still violates *portfolio-level* limits. The
   gate catches that.

2. External ground-truth state drift (manual adjustments, corporate
   actions, filled orders from a prior session we haven't reconciled
   yet) can make an order set unsafe at the moment of submit even if
   it was safe when computed. The gate re-validates against the
   *current* state, not the rebalance-time state.

Architecture
------------
A ``RiskCheck`` is a callable of signature
``(orders, state, market) -> list[RiskRejection]``. The gate composes
any sequence of checks. On each iteration, orders already rejected by a
prior check are excluded from subsequent checks — so portfolio-level
checks see only the surviving set (projection stays correct).

US-equities scope
-----------------
These built-ins assume whole-share integer quantities, USD
denomination, and a single account. Multi-currency, fractional shares,
and options-specific checks (margin-requirement computation, assignment
risk) are out of scope.

See also
--------
- ``quantengine.execution.order_state.OrderTracker.reject`` : the
  *mechanical* recorder of a rejection once the gate flags it.
- ``quantengine.portfolio.constraints.RebalanceConstraints`` : the
  *declarative* knobs embedded in the rebalance math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order
from quantengine.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskRejection:
    """One order rejected by one check.

    ``check`` identifies *which* rule tripped; ``reason`` is a
    human-readable message suitable for logging and for
    ``OrderTracker.reject(..., reason=...)``.
    """

    order: Order
    check: str
    reason: str


@runtime_checkable
class RiskCheck(Protocol):
    """Structural contract: any callable that flags orders.

    Implementations are free to veto zero, one, or many orders per
    invocation. They *must not* mutate ``state`` or ``market``.
    """

    name: str

    def __call__(
        self,
        orders: Sequence[Order],
        state: PortfolioState,
        market: MarketSnapshot,
    ) -> list[RiskRejection]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _price_map(market: MarketSnapshot) -> Mapping[str, float]:
    return {t: float(p) for t, p in zip(market.tickers, market.prices)}


def _project_quantities(orders: Sequence[Order], state: PortfolioState) -> dict[str, int]:
    """Signed share counts after hypothetically filling every order."""
    projected: dict[str, int] = {t: p.quantity for t, p in state.positions.items()}
    for o in orders:
        projected[o.ticker] = projected.get(o.ticker, 0) + o.signed_quantity
    return projected


def _project_cash(
    orders: Sequence[Order], state: PortfolioState, prices: Mapping[str, float]
) -> float:
    """Cash after hypothetically filling every order at reference price.

    Commission is *not* subtracted — the gate's cash check operates on
    notional-only; commission is small-change relative to the intended
    safety margin. If you need commission-exact accounting here, plug in
    a custom ``non_negative_cash_check`` that passes a per-share cost
    estimate.
    """
    out = state.cash
    for o in orders:
        px = prices.get(o.ticker)
        if px is None:
            continue  # priced-out orders are flagged by known_ticker_check
        out += -o.signed_quantity * px  # BUY: subtract; SELL: add
    return out


# ---------------------------------------------------------------------------
# Built-in checks (closures so they can carry parameters)
# ---------------------------------------------------------------------------
def known_ticker_check() -> RiskCheck:
    """Reject any order whose ticker is not priced in the snapshot."""

    class _KnownTicker:
        name = "known_ticker"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            known = set(market.tickers)
            return [
                RiskRejection(
                    order=o,
                    check=self.name,
                    reason=f"ticker {o.ticker!r} not in market snapshot",
                )
                for o in orders
                if o.ticker not in known
            ]

    return _KnownTicker()


def max_order_notional_check(max_notional: float) -> RiskCheck:
    """Fat-finger: reject any single order with notional > max_notional."""
    if max_notional <= 0:
        raise ValueError("max_notional must be > 0")

    class _OrderNotional:
        name = "max_order_notional"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            px = _price_map(market)
            out: list[RiskRejection] = []
            for o in orders:
                p = px.get(o.ticker)
                if p is None:
                    continue
                notional = abs(o.signed_quantity) * p
                if notional > max_notional:
                    out.append(
                        RiskRejection(
                            order=o,
                            check=self.name,
                            reason=(f"order notional ${notional:,.2f} > cap ${max_notional:,.2f}"),
                        )
                    )
            return out

    return _OrderNotional()


def non_negative_cash_check(min_cash: float = 0.0) -> RiskCheck:
    """Reject buy orders that, in aggregate, would drive cash below min.

    Greedy drop: sort surviving buys by notional descending, keep dropping
    the biggest one until projected cash >= min. This prioritizes many
    small trades over one big one — the usual "don't blow the account on
    one fat order" heuristic. Sells are never rejected by this check.
    """

    class _NonNegativeCash:
        name = "non_negative_cash"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            prices = _price_map(market)
            if _project_cash(orders, state, prices) >= min_cash:
                return []
            # Only buys consume cash; iterate and drop biggest buys first.
            orders_list = list(orders)
            buys = sorted(
                [o for o in orders_list if o.signed_quantity > 0],
                key=lambda o: o.signed_quantity * prices.get(o.ticker, 0.0),
                reverse=True,
            )
            rejected: list[RiskRejection] = []
            for b in buys:
                # Recompute projection with already-rejected buys excluded.
                rejected_ids = {r.order.order_id for r in rejected}
                survivors = [o for o in orders_list if o.order_id not in rejected_ids]
                if _project_cash(survivors, state, prices) >= min_cash:
                    break
                rejected.append(
                    RiskRejection(
                        order=b,
                        check=self.name,
                        reason=(
                            f"would drive cash below ${min_cash:,.2f}; "
                            f"projected={_project_cash(survivors, state, prices):,.2f}"
                        ),
                    )
                )
            return rejected

    return _NonNegativeCash()


def max_gross_leverage_check(max_leverage: float) -> RiskCheck:
    """Reject the order set if projected gross exposure / NAV > max.

    Drops orders greedily (largest absolute notional first) until the
    projected gross falls back within cap. This is a last-line defense:
    in normal operation ``RebalanceEngine`` already targets a gross well
    below the cap.
    """
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")

    class _MaxGross:
        name = "max_gross_leverage"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            prices = _price_map(market)
            orders_list = list(orders)

            def projected_gross(surviving: Sequence[Order]) -> float:
                qty = _project_quantities(surviving, state)
                return sum(abs(q) * prices[t] for t, q in qty.items() if t in prices)

            def projected_nav(surviving: Sequence[Order]) -> float:
                qty = _project_quantities(surviving, state)
                cash = _project_cash(surviving, state, prices)
                equity = sum(q * prices[t] for t, q in qty.items() if t in prices)
                return cash + equity

            gross = projected_gross(orders_list)
            nav = projected_nav(orders_list)
            if nav <= 0:
                # Insolvent projection — bail: reject all.
                return [RiskRejection(o, self.name, "projected NAV <= 0") for o in orders_list]
            if gross / nav <= max_leverage:
                return []

            rejected: list[RiskRejection] = []
            # Greedy drop by absolute notional.
            candidates = sorted(
                orders_list,
                key=lambda o: abs(o.signed_quantity) * prices.get(o.ticker, 0.0),
                reverse=True,
            )
            for o in candidates:
                rejected_ids = {r.order.order_id for r in rejected}
                surviving = [x for x in orders_list if x.order_id not in rejected_ids]
                g = projected_gross(surviving)
                nv = projected_nav(surviving)
                if nv > 0 and g / nv <= max_leverage:
                    break
                rejected.append(
                    RiskRejection(
                        order=o,
                        check=self.name,
                        reason=(
                            f"projected gross/NAV = {g / max(nv, 1e-12):.3f} > "
                            f"cap {max_leverage:.3f}"
                        ),
                    )
                )
            return rejected

    return _MaxGross()


def max_position_weight_check(max_weight: float) -> RiskCheck:
    """Reject any order whose *projected* position weight |q·p|/NAV > cap."""
    if not (0.0 < max_weight <= 1.0):
        raise ValueError("max_weight must be in (0, 1]")

    class _PositionWeight:
        name = "max_position_weight"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            prices = _price_map(market)
            orders_list = list(orders)
            # Per-ticker projected weight — identify which tickers exceed cap
            # and reject the orders on those tickers.
            qty = _project_quantities(orders_list, state)
            cash = _project_cash(orders_list, state, prices)
            equity = sum(q * prices[t] for t, q in qty.items() if t in prices)
            nav = cash + equity
            if nav <= 0:
                return [RiskRejection(o, self.name, "projected NAV <= 0") for o in orders_list]
            rejected: list[RiskRejection] = []
            for o in orders_list:
                p = prices.get(o.ticker)
                if p is None:
                    continue
                projected_q = qty.get(o.ticker, 0)
                w = abs(projected_q) * p / nav
                if w > max_weight:
                    rejected.append(
                        RiskRejection(
                            order=o,
                            check=self.name,
                            reason=(f"projected |w_{o.ticker}| = {w:.4f} > cap {max_weight:.4f}"),
                        )
                    )
            return rejected

    return _PositionWeight()


def max_participation_check(
    adv_dollars: Mapping[str, float], max_participation: float
) -> RiskCheck:
    """Reject any order whose notional exceeds ``max_participation`` of the name's ADV (s91).

    ``adv_dollars`` maps ticker -> average daily dollar volume (the caller computes it from
    its panel; the engine stays data-source-agnostic). Names ABSENT from the map are
    rejected too — an unknown-liquidity name must not be sized silently (fail-closed, the
    F24 lesson applied to capacity).
    """
    if not (0.0 < max_participation <= 1.0):
        raise ValueError("max_participation must be in (0, 1]")

    class _Participation:
        name = "max_participation"

        def __call__(
            self,
            orders: Sequence[Order],
            state: PortfolioState,
            market: MarketSnapshot,
        ) -> list[RiskRejection]:
            prices = _price_map(market)
            rejected: list[RiskRejection] = []
            for o in orders:
                p = prices.get(o.ticker)
                if p is None:
                    continue
                adv = adv_dollars.get(o.ticker)
                if adv is None or adv <= 0:
                    rejected.append(
                        RiskRejection(
                            order=o,
                            check=self.name,
                            reason=f"no ADV for {o.ticker} — refusing to size blind",
                        )
                    )
                    continue
                frac = o.quantity * p / adv
                if frac > max_participation:
                    rejected.append(
                        RiskRejection(
                            order=o,
                            check=self.name,
                            reason=(
                                f"order = {frac:.2%} of {o.ticker} ADV "
                                f"(${adv:,.0f}) > cap {max_participation:.2%}"
                            ),
                        )
                    )
            return rejected

    return _Participation()


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
@dataclass
class RiskGate:
    """Composes a sequence of ``RiskCheck``s.

    Usage:

        gate = RiskGate.default_us_equities(
            max_order_notional=500_000,
            max_gross_leverage=1.5,
            max_position_weight=0.10,
        )
        accepted, rejected = gate.validate(orders, state, market)
        for rj in rejected:
            tracker.reject(rj.order.order_id, market.timestamp, rj.reason)
        fills = broker.submit_orders(accepted, market)
    """

    checks: list[RiskCheck] = field(default_factory=list)

    def validate(
        self,
        orders: Sequence[Order],
        state: PortfolioState,
        market: MarketSnapshot,
    ) -> tuple[list[Order], list[RiskRejection]]:
        surviving = list(orders)
        rejected: list[RiskRejection] = []
        for check in self.checks:
            new_rejs = check(surviving, state, market)
            if not new_rejs:
                continue
            rejected_ids = {r.order.order_id for r in new_rejs}
            surviving = [o for o in surviving if o.order_id not in rejected_ids]
            rejected.extend(new_rejs)
        return surviving, rejected

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------
    @classmethod
    def default_us_equities(
        cls,
        *,
        max_order_notional: float = 1_000_000.0,
        max_gross_leverage: float = 1.5,
        max_position_weight: float = 0.20,
        min_cash: float = 0.0,
    ) -> "RiskGate":
        """Sensible defaults for a single-account US cash-equities book.

        Order matters: cheap structural checks first (known-ticker,
        fat-finger), then cash, then portfolio-level limits.
        """
        return cls(
            checks=[
                known_ticker_check(),
                max_order_notional_check(max_order_notional),
                non_negative_cash_check(min_cash),
                max_gross_leverage_check(max_gross_leverage),
                max_position_weight_check(max_position_weight),
            ]
        )


__all__ = [
    "RiskGate",
    "RiskCheck",
    "RiskRejection",
    "known_ticker_check",
    "max_gross_leverage_check",
    "max_order_notional_check",
    "max_participation_check",
    "max_position_weight_check",
    "non_negative_cash_check",
]
