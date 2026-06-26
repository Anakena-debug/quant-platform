"""Runner — the event loop skeleton.

Phase 1 synchronous single-process loop:

    for t in clock:
        market  = data_source.snapshot(t)
        signal  = strategy.predict(t, market)       # <- quantcore
        orders  = rebalance.rebalance(signal, state, market)
        orders  = risk_gate.validate(orders, ...)   # optional; defence-in-depth
        fills   = broker.submit_orders(orders, market)
        for f in fills:
            state = state.apply(f)
            ledger.append(t, "ORDER_FILLED", f)

Phase 2+ (optional): attach an ``OrderTracker`` to enforce the order
lifecycle state machine. When ``tracker`` is ``None`` (default), the
Phase-1 direct-ledger path is preserved — bit-for-bit identical events,
same replay parity.

A ``RiskGate`` can also be attached. Rejected orders are routed to the
tracker (if present) so they surface as ``ORDER_REJECTED`` events; when
no tracker is attached, rejections still filter the submitted set but
are not separately journaled.

This file contains the glue. Logic belongs in the classes, not the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.signal import AlphaSignal
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.order_state import OrderTracker
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate


@dataclass
class Runner:
    state: PortfolioState
    rebalance: RebalanceEngine
    broker: AbstractBroker
    ledger: Ledger
    tracker: OrderTracker | None = None
    risk_gate: RiskGate | None = None

    def step(self, signal: AlphaSignal, market: MarketSnapshot) -> PortfolioState:
        orders = self.rebalance.rebalance(signal, self.state, market)

        # Defence-in-depth: pre-trade risk gate.
        if self.risk_gate is not None:
            orders, rejections = self.risk_gate.validate(orders, self.state, market)
            # Route rejections. If a tracker is present we need to register
            # the rejected order first (PENDING → SUBMITTED → REJECTED is the
            # only legal path) so that reject() has a record to transition.
            for rj in rejections:
                if self.tracker is not None:
                    self.tracker.submit(rj.order, market.timestamp)
                    self.tracker.reject(rj.order.order_id, market.timestamp, reason=rj.reason)
                else:
                    # No tracker: record a dict payload for forensic replay.
                    self.ledger.append(
                        market.timestamp,
                        "ORDER_REJECTED",
                        {
                            "order_id": str(rj.order.order_id),
                            "ticker": rj.order.ticker,
                            "check": rj.check,
                            "reason": rj.reason,
                        },
                    )

        if self.tracker is not None:
            for o in orders:
                self.tracker.submit(o, market.timestamp)
        else:
            for o in orders:
                self.ledger.append(market.timestamp, "ORDER_SUBMITTED", o)

        fills = self.broker.submit_orders(orders, market)

        if self.tracker is not None:
            for f in fills:
                self.tracker.on_fill(f)
                self.state = self.state.apply(f)
        else:
            for f in fills:
                self.state = self.state.apply(f)
                self.ledger.append(market.timestamp, "ORDER_FILLED", f)

        return self.state

    def run(
        self,
        events: Iterable[tuple[AlphaSignal, MarketSnapshot]],
    ) -> PortfolioState:
        for signal, market in events:
            self.state = self.step(signal, market)
        return self.state
