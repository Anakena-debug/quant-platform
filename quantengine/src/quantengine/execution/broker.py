"""AbstractBroker — the single interface all broker adapters implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order


class AbstractBroker(ABC):
    """Minimal broker interface. Expand per phase.

    Phase 1 (synchronous paper): submit + immediate fill.
    Phase 2 (replay): submit at t, fill at t+1 (next-open) or t+close.
    Phase 3 (IBKR):  submit → async callbacks for fills + status.

    Cross-implementation invariants (S25 PR2 prerequisite)
    ------------------------------------------------------
    Callers writing code against this interface — rather than against a
    specific implementation — MUST treat the following as the authoritative
    contract. Implementation-specific behaviour beyond these invariants
    (exact fill price, exact timestamp, exact commission, return ordering
    of fills, etc.) belongs in implementation-specific test modules — never
    in cross-broker tests.

    (CI-1) ``submit_orders`` returns zero or more **execution-level**
        ``Fill`` objects for orders that received broker executions.
        A single parent order MAY produce multiple ``Fill`` objects
        when the broker reports partial fills (PaperBroker collapses
        each order to a single fill; IBKRBroker forwards each
        execution as it arrives). The returned list is NOT guaranteed
        to match the input order, nor is it guaranteed to contain at
        most one fill per order. Callers MUST associate fills to
        orders via ``Fill.order_id``; when order-level quantities are
        needed, aggregate ``Fill.signed_quantity`` by
        ``Fill.order_id`` (the sum equals the parent order's
        ``signed_quantity`` iff the order reached terminal-FILLED).

    (CI-2) ``Fill.timestamp`` is non-canonical operational metadata.
        PaperBroker copies ``market.timestamp`` (deterministic);
        IBKRBroker uses ``str(execution.time)`` (broker-reported
        wallclock, non-deterministic). Callers MUST NOT equality-assert
        ``Fill.timestamp`` across implementations.

    (CI-3) ``Fill.commission`` is implementation-defined and NOT
        comparable across implementations. PaperBroker derives it from
        the configured ``CostModel``; IBKRBroker uses the live
        ``commissionReport.commission`` (paper commissions are typically
        zero but the test should not rely on that). Callers MAY assert
        ``commission >= 0``; they MUST NOT compare absolute magnitudes
        across implementations.

    (CI-4) ``Fill.cash_delta == -(signed_quantity × price) - commission``
        holds for both implementations by construction (the property is
        defined on ``Fill`` in ``quantengine/contracts/orders.py``). This
        is the only AC4-style bookkeeping identity safe to assert across
        implementations.

    (CI-5) ``open_orders()`` is truthful in PaperBroker (returns orders
        that arrived without a matching market price) but is a Phase 3
        placeholder in IBKRBroker (returns an empty tuple regardless of
        actual open trades on the IBKR side). Callers MUST NOT rely on
        ``open_orders()`` as a source of truth for the IBKR
        implementation until Phase 4 OrderTracker persistence lands.

    (CI-6) ``Order.order_id`` (UUID, client-side) is preserved on
        ``Fill.order_id`` by both implementations. The IBKR-side integer
        ``orderId`` (auto-assigned by ``IB.placeOrder`` via
        ``client.getReqId()``) is recorded in
        ``Fill.metadata["ib_order_id"]`` and is a SEPARATE identity.
        Callers MUST NOT confuse the two — passing an IBKR int as a UUID
        will fail; vice versa silently breaks duplicate-order detection
        on the IBKR side.

    The cross-broker contract test
    ``quantengine/tests/test_broker_contract_equivalence.py`` exercises
    both implementations against assertions restricted to (CI-1) … (CI-6).
    """

    @abstractmethod
    def submit_orders(self, orders: Sequence[Order], market: MarketSnapshot) -> list[Fill]:
        """Submit orders and return the resulting fills (possibly empty)."""

    @abstractmethod
    def cancel_all(self) -> int:
        """Cancel any open orders. Return number cancelled."""

    @abstractmethod
    def open_orders(self) -> Iterable[Order]: ...
