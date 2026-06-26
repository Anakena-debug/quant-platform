"""ThreadSafeBrokerWrapper — sync facade over AsyncBrokerProtocol (S35 D3, D12, D13).

Strategy code is sync (S35 D2). Brokers are async. This module bridges
the boundary without touching ``nest_asyncio`` (D13: explicitly rejected;
the streaming runtime is CLI-launched and notebook-driven use is not
supported).

The wrapper accepts a *running* ``asyncio.AbstractEventLoop`` (owned by
the engine, typically running in a separate thread) plus an
``AsyncBrokerProtocol``. Each sync call schedules a coroutine onto the
loop via :func:`asyncio.run_coroutine_threadsafe` and blocks the
calling thread on ``future.result(timeout=...)``. On timeout, the
future is cancelled and :class:`BrokerTimeoutError` is raised — a
subclass of ``TimeoutError`` (per S35 D12) so callers can catch
either.

Default per-call timeouts (S35 D12):

- ``submit_order``      : 5.0 s
- ``cancel_order``      : 5.0 s
- ``get_position``      : 1.0 s
- ``get_account_state`` : 2.0 s

Defaults eliminate the ``result(timeout=None)`` deadlock failure mode
that strategy code would otherwise have to guard against on every
call. Per-call override via the ``timeout`` kwarg.

Lifecycle contract
------------------
The caller owns the event loop and its thread. The wrapper does NOT
start, stop, or take ownership of the loop. Pre-conditions on every
call:

- ``self._loop`` is running (``loop.is_running()`` truthy).
- ``self._loop`` is on a thread *other than* the caller's thread.

Violating either yields ``RuntimeError`` from
``asyncio.run_coroutine_threadsafe`` (this module does NOT mask it;
caller misuse should surface).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar
from uuid import UUID

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.protocols import (
    AsyncBrokerProtocol,
    BrokerTimeoutError,
)


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class WrapperTimeouts:
    """Per-call timeout policy (S35 D12 defaults).

    ``timeout=None`` on a wrapper call resolves to the matching field
    here; an explicit numeric ``timeout`` on a call overrides per-
    invocation.

    All values are seconds (float). Construct with positional or
    keyword overrides:

        WrapperTimeouts(submit_order_s=10.0, get_position_s=0.5)

    Negative or zero values are rejected at construction time:
    ``result(timeout=0)`` returns immediately whether the coroutine
    completed or not, which would defeat the gate.
    """

    submit_order_s: float = 5.0
    cancel_order_s: float = 5.0
    get_position_s: float = 1.0
    get_account_state_s: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("submit_order_s", self.submit_order_s),
            ("cancel_order_s", self.cancel_order_s),
            ("get_position_s", self.get_position_s),
            ("get_account_state_s", self.get_account_state_s),
        ):
            if not (value > 0):
                raise ValueError(f"WrapperTimeouts.{name} must be > 0; got {value!r}")


class ThreadSafeBrokerWrapper:
    """Sync facade over an ``AsyncBrokerProtocol``.

    Satisfies the ``SyncBrokerFacade`` Protocol (see
    ``quantengine.runtime.streaming.protocols``). Implementation note:
    we do not inherit from ``SyncBrokerFacade`` (it is a Protocol,
    structural) — concrete type identity is unimportant; method
    signatures are.

    Parameters
    ----------
    broker : AsyncBrokerProtocol
        The async broker implementation (``DemoBroker`` in S35;
        ``AsyncIBKRBroker`` in S36) whose coroutines this wrapper
        schedules onto the loop.
    loop : asyncio.AbstractEventLoop
        Loop running on a thread *other than* the strategy thread.
        The engine owns this loop; the wrapper never modifies it.
    timeouts : WrapperTimeouts | None
        Per-call timeout policy. ``None`` uses D12 defaults.
    """

    def __init__(
        self,
        broker: AsyncBrokerProtocol,
        loop: asyncio.AbstractEventLoop,
        timeouts: WrapperTimeouts | None = None,
    ) -> None:
        self._broker = broker
        self._loop = loop
        self._timeouts: WrapperTimeouts = timeouts if timeouts is not None else WrapperTimeouts()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run(
        self,
        coro: Coroutine[Any, Any, _T],
        timeout: float,
        op_name: str,
    ) -> _T:
        """Schedule ``coro`` on the loop; block until result or timeout.

        Uses ``asyncio.run_coroutine_threadsafe`` (AC5 grep target).
        On ``concurrent.futures.TimeoutError`` the future is cancelled
        (best-effort; if the coroutine has already started a blocking
        await, cancellation may be delayed until the next await point
        on the loop) and we raise :class:`BrokerTimeoutError`.

        Any other exception raised by the coroutine propagates
        unchanged to the caller — including ``RiskRejection``-style
        domain exceptions raised by ``SafeBroker``.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise BrokerTimeoutError(f"{op_name} timed out after {timeout:.3f}s") from e

    # ------------------------------------------------------------------
    # Public sync facade — satisfies SyncBrokerFacade Protocol
    # ------------------------------------------------------------------
    def submit_order(self, order: Order, timeout: float | None = None) -> list[Fill]:
        t = timeout if timeout is not None else self._timeouts.submit_order_s
        return self._run(self._broker.submit_order(order), t, "submit_order")

    def cancel_order(self, order_id: UUID, timeout: float | None = None) -> bool:
        t = timeout if timeout is not None else self._timeouts.cancel_order_s
        return self._run(self._broker.cancel_order(order_id), t, "cancel_order")

    def get_position(self, ticker: str, timeout: float | None = None) -> Position | None:
        t = timeout if timeout is not None else self._timeouts.get_position_s
        return self._run(self._broker.get_position(ticker), t, "get_position")

    def get_account_state(self, timeout: float | None = None) -> PortfolioState:
        t = timeout if timeout is not None else self._timeouts.get_account_state_s
        return self._run(self._broker.get_account_state(), t, "get_account_state")


__all__ = [
    "ThreadSafeBrokerWrapper",
    "WrapperTimeouts",
]
