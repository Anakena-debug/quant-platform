"""Single-worker serialization invariant for AsyncIBKRBroker (S36 AC3b).

This test pins the load-bearing D2 contract: every wrapped call MUST
execute on a thread whose name begins with ``ibkr-sync``. ``max_workers=1``
makes the executor a FIFO serializer matching ib_async's single-
connection invariant.

Without this pin, a future "modernization" sweep that replaces
``ThreadPoolExecutor(max_workers=1, thread_name_prefix='ibkr-sync')``
with ``asyncio.to_thread`` re-introduces the defect silently:
``asyncio.to_thread`` dispatches to the default pool (min(32, cpu+4)
workers) named ``asyncio_X``, and concurrent threads racing on
ib_async's request-id counter / pending-future map / socket write
buffer produce wrong-reqId dispatches and silent drops.

What this test does:
  - Constructs an AsyncIBKRBroker around a mock sync broker whose
    ``submit_orders`` records ``threading.current_thread().name``.
  - Concurrently launches N submit_order tasks via ``asyncio.gather``.
  - Asserts every recorded thread name starts with ``ibkr-sync`` and
    there is exactly one unique recorded name (the FIFO serializer
    has one worker).

What this test does NOT do:
  - Verify ib_async behaviour. The reach-through to ib_async is
    mock-substituted; the contract under pin is the executor
    configuration.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.runtime.streaming.ibkr_async import AsyncIBKRBroker


def _make_order() -> Order:
    return Order(
        order_id=uuid4(),
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )


def test_all_submit_orders_run_on_ibkr_sync_thread() -> None:
    """N concurrent submits share one ``ibkr-sync``-prefixed worker thread."""
    recorded_names: list[str] = []
    lock = threading.Lock()

    def recording_submit_orders(*args: Any, **kwargs: Any) -> list[Any]:
        # Block briefly to force overlap if the executor weren't a
        # FIFO serializer — under max_workers=1 the calls run in
        # sequence on the same worker thread.
        with lock:
            recorded_names.append(threading.current_thread().name)
        return []

    mock_ib = MagicMock()
    mock_ib.portfolio.return_value = []
    mock_ib.accountValues.return_value = []
    mock_connection = MagicMock()
    mock_connection.ib = mock_ib
    mock_broker = MagicMock()
    mock_broker.connection = mock_connection
    mock_broker.submit_orders.side_effect = recording_submit_orders

    n_tasks = 16

    async def run() -> None:
        broker = AsyncIBKRBroker(sync_broker=mock_broker, account="DU1")
        try:
            await asyncio.gather(*(broker.submit_order(_make_order()) for _ in range(n_tasks)))
        finally:
            await broker.aclose()

    asyncio.run(run())

    assert len(recorded_names) == n_tasks, (
        f"expected {n_tasks} recordings, got {len(recorded_names)}"
    )
    unique_names = set(recorded_names)
    assert len(unique_names) == 1, (
        f"expected all calls on one worker thread; got {len(unique_names)} "
        f"distinct names: {unique_names}"
    )
    (name,) = unique_names
    assert name.startswith("ibkr-sync"), (
        f"thread name {name!r} does not start with 'ibkr-sync'; the "
        "single-worker pool is the load-bearing serializer for ib_async — "
        "if you see this assertion fail, you likely replaced the executor "
        "with asyncio.to_thread or changed the prefix. Restore D2."
    )
