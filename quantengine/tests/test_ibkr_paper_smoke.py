"""Opt-in IBKR paper-account smoke test (PR5).

This is the **only** test in the quantengine suite that opens a real
socket to TWS / IB Gateway. It is gated behind a dual condition so
that a default ``pytest`` invocation never attempts to connect:

1. The ``ibkr_paper`` pytest marker (registered in ``pyproject.toml``
   under ``[tool.pytest.ini_options].markers``). Use
   ``pytest -m ibkr_paper`` to select the smoke explicitly.
2. The ``IBKR_PAPER_SMOKE`` environment variable must equal ``"1"``.
   When unset (or set to anything else), ``pytest.mark.skipif`` skips
   the test before any IBKR import path is exercised.

Both gates are independent: the marker is metadata for the test
runner; the env-var skipif is the load-bearing skip. The acceptance
gate ``uv run --directory quantengine pytest tests/test_ibkr_paper_smoke.py -x``
must therefore exit zero with the test SKIPPED in the default
environment (no env var).

The smoke is the load-bearing seal artifact for S22: the
``manual_checks`` block
requires recording ``order_id``, ``fill_price``, ``fill_latency``,
``reconcile.ok``, and the actual ``managedAccounts()`` list in the
session report before the sprint is sealed.

Required environment when running the smoke:

    IBKR_PAPER_SMOKE=1
    IBKR_HOST=127.0.0.1            # default in IBKRConfig.from_env
    IBKR_PORT=7497                 # 7497 = TWS paper, 4002 = Gateway paper
    IBKR_CLIENT_ID=11              # any free integer in [1, 999]
    IBKR_ACCOUNT=DU…               # paper-prefix; see runbook §1
"""

from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest

# Dual gate: marker (metadata) + skipif on env var (load-bearing).
pytestmark = [
    pytest.mark.ibkr_paper,
    pytest.mark.skipif(
        os.environ.get("IBKR_PAPER_SMOKE") != "1",
        reason="opt-in smoke; set IBKR_PAPER_SMOKE=1 to run (also ensure TWS/Gateway is up)",
    ),
]


def test_ibkr_paper_smoke_one_share_aapl_round_trip() -> None:
    """Live round-trip: BUY 1 AAPL → wait for fill → SELL 1 AAPL → wait for fill.

    The position is opened and closed within the same test so the
    paper account ends net-flat on AAPL (modulo commission, which on
    paper accounts is zero).

    Sequence (matches the manual_check seal-artifact requirements):
        1. Build ``IBKRConfig.from_env()`` and connect.
        2. ``assert_paper_account`` — Layer-2 ``managedAccounts()``
           cross-check. Capture the actual list for the seal record.
        3. ``pull_broker_snapshot`` (pre-trade).
        4. Submit BUY 1 AAPL MARKET via ``IBKRBroker``. Capture
           ``order_id`` and fill latency. Assert exactly one fill.
        5. Submit SELL 1 AAPL MARKET via ``IBKRBroker``. Same capture.
        6. ``pull_broker_snapshot`` (post-trade) for reconcile context.
        7. Disconnect (via ``with connection:`` context).

    Failure modes are reported via the standard pytest assertion path;
    consult the runbook §6 (failure-mode triage) when this test fails.
    """
    # Imports are inside the test body so that the module loads cleanly
    # when the smoke is skipped (the IBKR import path is heavy and
    # would pull ib_async even on a default `pytest` run).
    import numpy as np

    from quantengine.contracts.market import MarketSnapshot
    from quantengine.contracts.orders import Order, OrderSide, OrderType
    from quantengine.execution.ibkr.broker import IBKRBroker
    from quantengine.execution.ibkr.config import IBKRConfig, TimeoutPolicy
    from quantengine.execution.ibkr.connection import (
        IBKRConnection,
        assert_paper_account,
    )
    from quantengine.execution.ibkr.positions import pull_broker_snapshot
    from quantengine.execution.order_state import OrderTracker
    from quantengine.portfolio.ledger import Ledger

    # 1. Build config + connect.
    cfg = IBKRConfig.from_env()
    connection = IBKRConnection()
    connection.connect(cfg)
    assert connection.is_connected(), (
        "connect() returned without raising but is_connected() is False"
    )

    try:
        # 2. Layer-2 paper-account assertion.
        assert_paper_account(connection.ib, cfg.account)
        managed = list(connection.ib.managedAccounts())
        assert cfg.account in managed, (
            f"configured account {cfg.account} not in managedAccounts() = {managed}"
        )
        # Print for the seal record (pytest -s captures this).
        print(f"\n[smoke] managedAccounts() = {managed}")

        # 3. Pre-trade snapshot.
        as_of_pre = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        snap_pre = pull_broker_snapshot(connection.ib, cfg.account, as_of=as_of_pre)
        print(f"[smoke] pre-trade cash={snap_pre.cash:.2f} positions={len(snap_pre.positions)}")

        # 4-5. Round-trip BUY then SELL.
        ledger = Ledger()
        tracker = OrderTracker(ledger=ledger)
        broker = IBKRBroker(connection=connection, tracker=tracker, timeouts=TimeoutPolicy())
        # IBKRBroker requires a MarketSnapshot for tracker timestamps; the
        # actual prices are unused (real fills come from IB, not from this
        # snapshot). MarketSnapshot enforces strictly-positive prices, so
        # use 1.0 as a meaningless placeholder.
        market = MarketSnapshot(timestamp=as_of_pre, tickers=("AAPL",), prices=np.array([1.0]))

        for side in (OrderSide.BUY, OrderSide.SELL):
            order = Order(
                order_id=uuid4(),
                ticker="AAPL",
                side=side,
                quantity=1,
                order_type=OrderType.MARKET,
            )
            t_start = time.monotonic()
            fills = broker.submit_orders([order], market)
            elapsed = time.monotonic() - t_start
            assert len(fills) == 1, f"{side.value} expected 1 fill, got {len(fills)}: {fills}"
            f = fills[0]
            msg = (
                f"[smoke] {side.value} 1 AAPL → order_id={order.order_id}"
                f" fill_price={f.price:.2f} signed_qty={f.signed_quantity}"
                f" latency={elapsed:.2f}s"
            )
            print(msg)

        # 6. Post-trade snapshot (informational; round-trip should be net-flat).
        as_of_post = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        snap_post = pull_broker_snapshot(connection.ib, cfg.account, as_of=as_of_post)
        print(f"[smoke] post-trade cash={snap_post.cash:.2f} positions={len(snap_post.positions)}")

    finally:
        # 7. Always disconnect, even on assertion failure.
        connection.disconnect()
