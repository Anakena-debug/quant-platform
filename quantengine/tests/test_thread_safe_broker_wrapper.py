"""Tests for quantengine.runtime.streaming.wrappers.ThreadSafeBrokerWrapper.

Pins the sync-over-async bridge contract (S35 D3, D12, D13):

- ``asyncio.run_coroutine_threadsafe`` is the only bridge mechanism (AC5).
- Per-call timeouts default to the S35 D12 values; on timeout, the
  wrapper raises :class:`BrokerTimeoutError` (subclass of
  ``TimeoutError``).
- Non-timeout exceptions from coroutines propagate unchanged.
- ``WrapperTimeouts`` rejects non-positive values at construction.

quantcore-independence pattern: this test file does not import
quantcore; the mock broker satisfies ``AsyncBrokerProtocol``
structurally, mirroring the way ``DemoBroker``/``AsyncIBKRBroker``
will at runtime.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from typing import Callable
from uuid import UUID, uuid4

import pytest

from quantengine.contracts.orders import (
    Fill,
    Order,
    OrderSide,
    OrderType,
)
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.protocols import (
    AsyncBrokerProtocol,
    BrokerTimeoutError,
    SyncBrokerFacade,
)
from quantengine.runtime.streaming.wrappers import (
    ThreadSafeBrokerWrapper,
    WrapperTimeouts,
)


# ---------------------------------------------------------------------------
# Test fixtures — a running event loop on a worker thread
# ---------------------------------------------------------------------------
@pytest.fixture
def loop_thread() -> Iterator[asyncio.AbstractEventLoop]:
    """Spawn an asyncio loop on a daemon thread; yield the loop.

    The wrapper's contract requires a *running* loop on a thread OTHER
    than the caller's. This fixture provides exactly that.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_runner, name="test-loop", daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0), "event loop thread failed to start"

    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


# ---------------------------------------------------------------------------
# Mock async broker — controllable for timeouts and exceptions
# ---------------------------------------------------------------------------
class _MockAsyncBroker:
    """AsyncBrokerProtocol implementation whose responses + delays can
    be steered per-test.

    Parameters control the artificial delay applied per call (use to
    force timeouts) and a one-shot exception (use to verify
    propagation).
    """

    def __init__(self) -> None:
        self.submit_delay_s: float = 0.0
        self.cancel_delay_s: float = 0.0
        self.get_position_delay_s: float = 0.0
        self.get_account_state_delay_s: float = 0.0
        self.next_exception: BaseException | None = None
        self.submitted_orders: list[Order] = []
        self.cancelled_ids: list[UUID] = []

    def _maybe_raise(self) -> None:
        if self.next_exception is not None:
            exc, self.next_exception = self.next_exception, None
            raise exc

    async def submit_order(self, order: Order) -> list[Fill]:
        await asyncio.sleep(self.submit_delay_s)
        self._maybe_raise()
        self.submitted_orders.append(order)
        return [
            Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=100.0,
                commission=0.0,
                timestamp="2026-05-21T16:00:00Z",
            )
        ]

    async def cancel_order(self, order_id: UUID) -> bool:
        await asyncio.sleep(self.cancel_delay_s)
        self._maybe_raise()
        self.cancelled_ids.append(order_id)
        return True

    async def get_position(self, ticker: str) -> Position | None:
        await asyncio.sleep(self.get_position_delay_s)
        self._maybe_raise()
        if ticker == "AAPL":
            return Position(ticker="AAPL", quantity=100, avg_cost=150.0)
        return None

    async def get_account_state(self) -> PortfolioState:
        await asyncio.sleep(self.get_account_state_delay_s)
        self._maybe_raise()
        return PortfolioState(cash=1_000_000.0, positions={})


@pytest.fixture
def mock_broker() -> _MockAsyncBroker:
    return _MockAsyncBroker()


@pytest.fixture
def make_order() -> Callable[..., Order]:
    def _make(ticker: str = "AAPL", quantity: int = 10) -> Order:
        return Order(
            order_id=uuid4(),
            ticker=ticker,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

    return _make


# ---------------------------------------------------------------------------
# WrapperTimeouts construction
# ---------------------------------------------------------------------------
def test_wrapper_timeouts_defaults_match_d12_spec() -> None:
    """D12: submit=5s, cancel=5s, get_position=1s, get_account_state=2s."""
    t = WrapperTimeouts()
    assert t.submit_order_s == 5.0
    assert t.cancel_order_s == 5.0
    assert t.get_position_s == 1.0
    assert t.get_account_state_s == 2.0


def test_wrapper_timeouts_rejects_zero_or_negative() -> None:
    """timeout=0 makes future.result return immediately whether done or
    not — that's a footgun, not a valid policy."""
    with pytest.raises(ValueError, match="must be > 0"):
        WrapperTimeouts(submit_order_s=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        WrapperTimeouts(get_position_s=-1.0)


# ---------------------------------------------------------------------------
# Wrapper construction and Protocol conformance
# ---------------------------------------------------------------------------
def test_wrapper_satisfies_sync_broker_facade(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    assert isinstance(w, SyncBrokerFacade)


def test_mock_async_broker_satisfies_async_broker_protocol(
    mock_broker: _MockAsyncBroker,
) -> None:
    """Sanity: the mock used here must satisfy the upstream Protocol."""
    assert isinstance(mock_broker, AsyncBrokerProtocol)


# ---------------------------------------------------------------------------
# Happy path: each sync facade method returns the coroutine's result
# ---------------------------------------------------------------------------
def test_submit_order_returns_fill_synchronously(
    mock_broker: _MockAsyncBroker,
    loop_thread: asyncio.AbstractEventLoop,
    make_order: Callable[..., Order],
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    order = make_order("AAPL", 10)
    fills = w.submit_order(order)
    assert len(fills) == 1
    assert fills[0].order_id == order.order_id
    assert fills[0].signed_quantity == 10
    assert mock_broker.submitted_orders == [order]


def test_cancel_order_returns_bool(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    oid = uuid4()
    assert w.cancel_order(oid) is True
    assert mock_broker.cancelled_ids == [oid]


def test_get_position_returns_position(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    pos = w.get_position("AAPL")
    assert pos is not None
    assert pos.ticker == "AAPL"
    assert pos.quantity == 100


def test_get_position_returns_none_for_unknown(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    assert w.get_position("ZZZZ") is None


def test_get_account_state_returns_portfolio_state(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    s = w.get_account_state()
    assert isinstance(s, PortfolioState)
    assert s.cash == 1_000_000.0


# ---------------------------------------------------------------------------
# Timeout behavior — the D12 invariant
# ---------------------------------------------------------------------------
def test_submit_order_raises_broker_timeout_when_delay_exceeds_budget(
    mock_broker: _MockAsyncBroker,
    loop_thread: asyncio.AbstractEventLoop,
    make_order: Callable[..., Order],
) -> None:
    """Coroutine takes 1s, per-call timeout is 0.05s -> BrokerTimeoutError."""
    mock_broker.submit_delay_s = 1.0
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    with pytest.raises(BrokerTimeoutError) as exc:
        w.submit_order(make_order(), timeout=0.05)
    assert "submit_order" in str(exc.value)
    assert isinstance(exc.value, TimeoutError)  # subclass invariant


def test_get_position_raises_broker_timeout(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    mock_broker.get_position_delay_s = 1.0
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    with pytest.raises(BrokerTimeoutError, match="get_position"):
        w.get_position("AAPL", timeout=0.05)


def test_explicit_timeout_overrides_default(
    mock_broker: _MockAsyncBroker,
    loop_thread: asyncio.AbstractEventLoop,
    make_order: Callable[..., Order],
) -> None:
    """submit_order default is 5s; we override with 0.05s and force a 1s
    delay — must time out (proving the override took effect)."""
    mock_broker.submit_delay_s = 1.0
    w = ThreadSafeBrokerWrapper(
        mock_broker, loop_thread, timeouts=WrapperTimeouts(submit_order_s=10.0)
    )
    with pytest.raises(BrokerTimeoutError):
        w.submit_order(make_order(), timeout=0.05)


def test_no_timeout_when_call_completes_within_budget(
    mock_broker: _MockAsyncBroker,
    loop_thread: asyncio.AbstractEventLoop,
    make_order: Callable[..., Order],
) -> None:
    mock_broker.submit_delay_s = 0.01
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    # Default submit_order_s = 5.0 >> 0.01; no timeout.
    fills = w.submit_order(make_order())
    assert len(fills) == 1


# ---------------------------------------------------------------------------
# Non-timeout exceptions propagate unchanged
# ---------------------------------------------------------------------------
def test_non_timeout_exception_propagates(
    mock_broker: _MockAsyncBroker,
    loop_thread: asyncio.AbstractEventLoop,
    make_order: Callable[..., Order],
) -> None:
    """A domain error (e.g., RiskRejection from SafeBroker) must not be
    masked as BrokerTimeoutError."""
    mock_broker.next_exception = ValueError("simulated risk rejection")
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    with pytest.raises(ValueError, match="simulated risk rejection"):
        w.submit_order(make_order())
    # Mock state confirms the coroutine reached the raise point.
    assert mock_broker.next_exception is None


# ---------------------------------------------------------------------------
# Custom timeout policy — wrapper-wide override
# ---------------------------------------------------------------------------
def test_custom_timeouts_apply_when_per_call_kwarg_omitted(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    """If WrapperTimeouts overrides get_position_s = 0.05s and the
    coroutine takes 1s, we must time out without specifying timeout=."""
    mock_broker.get_position_delay_s = 1.0
    w = ThreadSafeBrokerWrapper(
        mock_broker,
        loop_thread,
        timeouts=WrapperTimeouts(get_position_s=0.05),
    )
    with pytest.raises(BrokerTimeoutError):
        w.get_position("AAPL")  # no timeout= -> uses 0.05s policy


# ---------------------------------------------------------------------------
# AC5 / AC5b grep targets are pinned by source greps in the build pipeline.
# Here we verify the runtime semantics that those greps protect.
# ---------------------------------------------------------------------------
def test_wrapper_uses_run_coroutine_threadsafe_runtime(
    mock_broker: _MockAsyncBroker, loop_thread: asyncio.AbstractEventLoop
) -> None:
    """Indirect proof: the wrapper actually crosses thread boundaries.
    If it used asyncio.run() (single-threaded), get_account_state in a
    nested context would deadlock; here we call it from the test thread
    while the loop runs in a separate thread."""
    w = ThreadSafeBrokerWrapper(mock_broker, loop_thread)
    # Call N times to exercise the bridge consistently.
    for _ in range(5):
        s = w.get_account_state()
        assert isinstance(s, PortfolioState)
