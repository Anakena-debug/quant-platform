"""IBKR adapter package — Phase 3 of quantengine.

Implementations land across S22 PR1-PR4. Public API:

- ``IBKRConfig``, ``TimeoutPolicy``, ``PAPER_PORTS`` (PR1) — connection
  configuration and the layer-1 paper-account gate.
- ``IBKRConnection``, ``IBKRCircuitOpen``, ``derive_client_id``,
  ``assert_paper_account`` (PR1) — lifecycle wrapper over
  ``ib_async.IB()`` plus the layer-2 paper-account cross-check.
"""

from __future__ import annotations

from quantengine.execution.ibkr.broker import IBKRBroker
from quantengine.execution.ibkr.config import (
    PAPER_PORTS,
    IBKRConfig,
    TimeoutPolicy,
)
from quantengine.execution.ibkr.connection import (
    IBKRCircuitOpen,
    IBKRConnection,
    assert_paper_account,
    derive_client_id,
)
from quantengine.execution.ibkr.order_mapping import (
    ib_trade_to_fill,
    order_to_ib_order,
)
from quantengine.execution.ibkr.positions import pull_broker_snapshot

__all__ = [
    "IBKRBroker",
    "IBKRCircuitOpen",
    "IBKRConfig",
    "IBKRConnection",
    "PAPER_PORTS",
    "TimeoutPolicy",
    "assert_paper_account",
    "derive_client_id",
    "ib_trade_to_fill",
    "order_to_ib_order",
    "pull_broker_snapshot",
]
