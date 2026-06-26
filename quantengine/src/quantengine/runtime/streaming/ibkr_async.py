"""AsyncIBKRBroker — AsyncBrokerProtocol over S22 sync IBKRBroker.

S22's ``IBKRBroker`` is sync-with-timeout. The streaming runtime's
``AsyncBrokerProtocol`` is async. This module bridges the two via a
**dedicated single-worker** ``ThreadPoolExecutor``, NOT
``asyncio.to_thread``.

Rationale (load-bearing — DO NOT replace with asyncio.to_thread).
``asyncio.to_thread`` dispatches to the default thread pool, which is
sized ``min(32, os.cpu_count() + 4)``. ``ib_async``'s ``IB`` object
owns a single internal event loop and a single TCP connection to
TWS/Gateway. Concurrent calls from multiple threads race on:

  - the request-id counter (``client.getReqId()``)
  - the pending-future map keyed by reqId
  - the socket write buffer

Symptoms: wrong-reqId dispatches (a fill arrives at the future for an
unrelated order), callback delivery to wrong futures, silent drops.
``max_workers=1`` makes the executor a FIFO serializer that matches
ib_async's single-connection invariant. The dedicated pool is torn
down at ``aclose()`` so engine shutdown does not leak threads.

The thread-name prefix ``ibkr-sync`` is a hard contract: a separate
serialization test in ``tests/test_ibkr_async_serialization.py``
asserts that all wrapped calls execute on a thread whose name begins
with ``ibkr-sync``. Without that pin, a future "modernization" sweep
that replaces the executor with ``asyncio.to_thread`` re-introduces
the defect silently.

S22 wrap-not-rewrite (D2).
Construction takes a pre-connected ``IBKRBroker`` (the S22 sync
adapter). Methods that S22 exposes (``submit_orders``) are
trampolined. Methods that S22 does NOT expose at the
public surface — per-order ``cancel_order(UUID)``,
``get_position(ticker)``, ``get_account_state()`` — reach through
``sync_broker.connection.ib`` (the underlying ``ib_async.IB``
instance). All such reach-throughs run inside the same single-worker
executor, preserving the serialization invariant.

S37 boundary.
``cancel_order(UUID)`` returns ``False`` in S36 — per-order
cancellation by client UUID requires the UUID↔ib_order_id index that
S37's ``OrderTracker`` persistence work introduces. Operators needing
to cancel in flight call the non-protocol method ``cancel_all_sync``
(which trampolines S22's ``cancel_all``) or use the TWS UI directly.
This limitation is documented in the runbook (PR5).

Paper-account two-layer gate.
``from_env()`` reads ``IBKR_PAPER_ACCOUNT`` (the streaming-runtime
env-var name; S22's batch path uses ``IBKR_ACCOUNT``). The S22
``managedAccounts()`` cross-check via
``execution.ibkr.connection.assert_paper_account`` runs at construct
time — the wrapper does not weaken this gate, only surfaces the S22
exception path through the async boundary. ``ib.managedAccounts()``
returns DU-prefixed IDs for paper TWS instances; mismatch with
``IBKR_PAPER_ACCOUNT`` raises and refuses to construct.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Final
from uuid import UUID

import numpy as np

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Fill, Order
from quantengine.execution.ibkr.broker import IBKRBroker
from quantengine.execution.ibkr.config import IBKRConfig
from quantengine.execution.ibkr.connection import IBKRConnection, assert_paper_account
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.protocols import AsyncBrokerProtocol

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

_THREAD_NAME_PREFIX: Final[str] = "ibkr-sync"
_PLACEHOLDER_PRICE: Final[float] = (
    1.0  # market.prices validator requires > 0; routing ignores this value
)


def _now_iso8601_z() -> str:
    """Return the current UTC time as an ISO-8601 string with trailing Z."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AsyncIBKRBroker(AsyncBrokerProtocol):
    """``AsyncBrokerProtocol`` over a sync S22 ``IBKRBroker`` (D2).

    Construction is cheap: the caller provides an already-connected
    sync broker. The dedicated single-worker executor is created here;
    the caller must invoke ``aclose()`` to tear it down (or use
    ``async with`` if a future PR adds the context manager).

    See module docstring for the asyncio.to_thread rejection rationale
    and the S37 cancel_order limitation.
    """

    def __init__(self, sync_broker: IBKRBroker, account: str) -> None:
        """Wrap an already-connected sync IBKRBroker.

        ``account`` is the paper-account ID (the value of
        ``IBKR_PAPER_ACCOUNT`` env var, or the same string the
        caller passed to ``IBKRConfig`` when constructing
        ``sync_broker``). It's plumbed explicitly because
        ``IBKRConnection`` discards the config after ``connect()`` —
        the broker needs the account ID for ``ib.portfolio(account)``
        and ``ib.accountValues(account)`` lookups.
        """
        self._sync_broker = sync_broker
        self._account = account
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=_THREAD_NAME_PREFIX,
        )
        self._closed = False

    @classmethod
    def from_env(cls) -> AsyncIBKRBroker:
        """Build an AsyncIBKRBroker from environment variables.

        Reads ``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_CLIENT_ID``,
        ``IBKR_PAPER_ACCOUNT``. Constructs IBKRConfig directly
        (bypassing IBKRConfig.from_env, which reads IBKR_ACCOUNT —
        a deliberate env-var naming distinction between the streaming
        runtime and the S22 batch path; see module docstring).

        Connects the IB session and asserts paper-account ownership
        via ``assert_paper_account``. Mismatch raises and refuses to
        construct.
        """
        try:
            host = os.environ["IBKR_HOST"]
            port = int(os.environ["IBKR_PORT"])
            client_id = int(os.environ["IBKR_CLIENT_ID"])
            account = os.environ["IBKR_PAPER_ACCOUNT"]
        except KeyError as e:
            raise KeyError(
                f"missing IBKR streaming env var: {e.args[0]!r}. "
                "Expected: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_PAPER_ACCOUNT. "
                "(The S22 batch path uses IBKR_ACCOUNT instead of IBKR_PAPER_ACCOUNT; "
                "see s36 runbook §IBKR env-var naming.)"
            ) from e

        config = IBKRConfig(host=host, port=port, client_id=client_id, account=account)
        connection = IBKRConnection()
        connection.connect(config)
        assert_paper_account(connection.ib, config.account)
        sync_broker = IBKRBroker(connection=connection)
        return cls(sync_broker=sync_broker, account=account)

    async def submit_order(self, order: Order) -> list[Fill]:
        """Submit a single order via S22's sync ``submit_orders``.

        Trampolined through the single-worker executor. The
        ``MarketSnapshot`` is a thin envelope: ``submit_orders`` only
        consumes ``market.timestamp`` (verified by grep against S22's
        broker.py); the price array is structurally required by
        MarketSnapshot's __post_init__ but not used for IBKR routing.
        """
        loop = asyncio.get_running_loop()
        market = MarketSnapshot(
            timestamp=_now_iso8601_z(),
            tickers=(order.ticker,),
            prices=np.array([_PLACEHOLDER_PRICE]),
        )
        return await loop.run_in_executor(
            self._executor,
            self._sync_broker.submit_orders,
            [order],
            market,
        )

    async def cancel_order(self, order_id: UUID) -> bool:
        """PR3 stub: returns False with a warning.

        Per-order cancellation by client UUID requires the
        UUID↔ib_order_id index introduced by S37's OrderTracker
        persistence work. Operators needing to cancel in flight use
        ``cancel_all_sync`` or the TWS UI directly. See module
        docstring for the S37 boundary rationale.
        """
        _log.warning(
            "AsyncIBKRBroker.cancel_order(%s) returning False — per-order "
            "cancellation by UUID is implemented by S37 OrderTracker "
            "persistence. Use cancel_all_sync() or TWS UI for now.",
            order_id,
        )
        return False

    async def cancel_all_sync(self) -> int:
        """Non-protocol: trampoline S22's ``cancel_all`` for operator use.

        Returns the count of cancellations acknowledged. NOT part of
        AsyncBrokerProtocol — exposed for runbook-driven cancel-all
        operations only.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._sync_broker.cancel_all)

    async def get_position(self, ticker: str) -> Position | None:
        """Return the broker's authoritative position for ``ticker``.

        Reaches through ``sync_broker.connection.ib.portfolio()`` —
        S22 does not expose a per-ticker accessor at the public
        surface. Runs inside the executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_position_sync, ticker)

    def _get_position_sync(self, ticker: str) -> Position | None:
        ib = self._sync_broker.connection.ib
        for item in ib.portfolio(self._account):
            contract = item.contract
            if getattr(contract, "symbol", None) != ticker:
                continue
            if getattr(contract, "secType", "") != "STK":
                continue
            qty = int(item.position)
            if qty == 0:
                return None
            return Position(ticker=ticker, quantity=qty, avg_cost=float(item.averageCost))
        return None

    async def get_account_state(self) -> PortfolioState:
        """Snapshot the broker's authoritative ``PortfolioState``.

        Reaches through ``sync_broker.connection.ib`` to pull cash +
        positions, then builds a ``PortfolioState`` via the same
        type-system the engine consumes. Runs inside the executor.

        Note: ``realized_pnl`` and ``total_commission`` are returned
        as 0.0 — IBKR does not report cumulative session PnL through
        the surfaces we consult here. The engine's in-process
        VirtualPortfolio is authoritative for those fields; the
        broker snapshot is for *position* truth-checking only.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._get_account_state_sync)

    def _get_account_state_sync(self) -> PortfolioState:
        ib = self._sync_broker.connection.ib

        # Cash: USD TotalCashValue when present, else fall through to 0.0.
        cash = 0.0
        for av in ib.accountValues(self._account):
            if getattr(av, "tag", "") == "TotalCashValue" and getattr(av, "currency", "") == "USD":
                try:
                    cash = float(av.value)
                except (TypeError, ValueError):
                    cash = 0.0
                break

        positions: dict[str, Position] = {}
        for item in ib.portfolio(self._account):
            contract = item.contract
            if getattr(contract, "secType", "") != "STK":
                continue
            ticker = str(getattr(contract, "symbol", ""))
            if not ticker:
                continue
            qty = int(item.position)
            if qty == 0:
                continue
            positions[ticker] = Position(
                ticker=ticker, quantity=qty, avg_cost=float(item.averageCost)
            )

        return PortfolioState(cash=cash, positions=positions)

    async def aclose(self) -> None:
        """Graceful shutdown: tear down the executor.

        Idempotent. Does NOT disconnect the underlying IBKR session —
        that lifecycle belongs to the caller who supplied the
        connected sync broker (or to the ``from_env`` invocation path,
        which is documented in the runbook).
        """
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


__all__ = ["AsyncIBKRBroker"]
