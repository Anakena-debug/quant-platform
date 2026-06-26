"""Hermetic tests for ``AsyncIBKRBroker`` (S36 PR3).

Coverage map:

- ``TestProtocolConformance``: AsyncIBKRBroker satisfies AsyncBrokerProtocol
  structurally (isinstance via runtime_checkable).
- ``TestExecutorPolicy``: single-worker + ibkr-sync prefix invariants
  (the FIFO-serialization pin is in test_ibkr_async_serialization.py).
- ``TestSubmitOrderTrampoline``: submit_order routes the order through
  the sync broker's submit_orders with a minimal MarketSnapshot.
- ``TestCancelOrderS37Boundary``: cancel_order returns False with a
  warning per the S37 boundary documented in ibkr_async.py.
- ``TestGetPosition``: reaches through connection.ib.portfolio + filters
  to STK + matching ticker.
- ``TestGetAccountState``: reaches through connection.ib.accountValues +
  ib.portfolio, builds PortfolioState.
- ``TestFromEnv``: env-driven construction; missing var → KeyError;
  IBKR_PAPER_ACCOUNT (not IBKR_ACCOUNT) is the required env var.
- ``TestAclose``: executor shutdown idempotent.
- ``TestLiveSmoke``: IBKR_PAPER_SMOKE-gated opt-in (AC9).

The sync IBKRBroker + IBKRConnection + ib_async surface are mocked
end-to-end — no network. The single-worker thread-name invariant
(AC3b) is pinned by tests/test_ibkr_async_serialization.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.ibkr_async import AsyncIBKRBroker
from quantengine.runtime.streaming.protocols import AsyncBrokerProtocol

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_mock_broker(*, fills: list[Any] | None = None) -> MagicMock:
    """Build a mock sync IBKRBroker with .connection.ib, .submit_orders,
    .cancel_all, .config-like .account flow."""
    mock_ib = MagicMock(name="ib_async.IB")
    mock_ib.portfolio.return_value = []
    mock_ib.accountValues.return_value = []

    mock_connection = MagicMock(name="IBKRConnection")
    mock_connection.ib = mock_ib

    mock_broker = MagicMock(name="IBKRBroker")
    mock_broker.connection = mock_connection
    mock_broker.submit_orders.return_value = fills if fills is not None else []
    mock_broker.cancel_all.return_value = 0
    return mock_broker


@pytest.fixture
def mock_sync_broker() -> MagicMock:
    return _make_mock_broker()


@pytest.fixture
def async_broker(mock_sync_broker: MagicMock) -> AsyncIBKRBroker:
    return AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU123456")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_async_ibkr_broker_satisfies_protocol(self, async_broker: AsyncIBKRBroker) -> None:
        assert isinstance(async_broker, AsyncBrokerProtocol)


# ---------------------------------------------------------------------------
# Executor policy
# ---------------------------------------------------------------------------


class TestExecutorPolicy:
    def test_executor_is_single_worker(self, async_broker: AsyncIBKRBroker) -> None:
        # _max_workers is the public-private attr name on ThreadPoolExecutor
        assert async_broker._executor._max_workers == 1

    def test_executor_thread_name_prefix_ibkr_sync(self, async_broker: AsyncIBKRBroker) -> None:
        prefix = async_broker._executor._thread_name_prefix
        assert prefix is not None
        assert prefix.startswith("ibkr-sync")


# ---------------------------------------------------------------------------
# submit_order trampoline
# ---------------------------------------------------------------------------


def _make_order(ticker: str = "AAPL", qty: int = 100) -> Order:
    return Order(
        order_id=uuid4(),
        ticker=ticker,
        side=OrderSide.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
    )


class TestSubmitOrderTrampoline:
    def test_calls_sync_submit_orders_with_single_order_list(
        self, async_broker: AsyncIBKRBroker, mock_sync_broker: MagicMock
    ) -> None:
        order = _make_order()

        async def run() -> list[Any]:
            return await async_broker.submit_order(order)

        result = asyncio.run(run())
        mock_sync_broker.submit_orders.assert_called_once()
        call_args = mock_sync_broker.submit_orders.call_args
        # First positional arg is the [order] list
        assert call_args.args[0] == [order]
        # Second positional arg is a MarketSnapshot for the ticker
        market = call_args.args[1]
        assert market.tickers == (order.ticker,)
        assert market.prices.shape == (1,)
        assert result == []  # mock returns empty fills

    def test_passes_fills_back_unchanged(self, mock_sync_broker: MagicMock) -> None:
        fake_fill = object()
        mock_sync_broker.submit_orders.return_value = [fake_fill]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")
        order = _make_order()

        async def run() -> list[Any]:
            return await broker.submit_order(order)

        result = asyncio.run(run())
        assert result == [fake_fill]


# ---------------------------------------------------------------------------
# cancel_order S37 boundary
# ---------------------------------------------------------------------------


class TestCancelOrderS37Boundary:
    def test_cancel_order_returns_false_without_calling_broker(
        self, async_broker: AsyncIBKRBroker, mock_sync_broker: MagicMock
    ) -> None:
        order_id = uuid4()

        async def run() -> bool:
            return await async_broker.cancel_order(order_id)

        result = asyncio.run(run())
        assert result is False
        # S22 sync broker not touched — limitation is at the wrapper level
        mock_sync_broker.cancel_all.assert_not_called()

    def test_cancel_all_sync_trampolines(
        self, async_broker: AsyncIBKRBroker, mock_sync_broker: MagicMock
    ) -> None:
        mock_sync_broker.cancel_all.return_value = 3

        async def run() -> int:
            return await async_broker.cancel_all_sync()

        result = asyncio.run(run())
        assert result == 3
        mock_sync_broker.cancel_all.assert_called_once()


# ---------------------------------------------------------------------------
# get_position
# ---------------------------------------------------------------------------


def _make_portfolio_item(
    ticker: str, qty: int, avg_cost: float, sec_type: str = "STK"
) -> MagicMock:
    item = MagicMock()
    item.contract.symbol = ticker
    item.contract.secType = sec_type
    item.position = qty
    item.averageCost = avg_cost
    return item


class TestGetPosition:
    def test_returns_position_for_matching_stk_ticker(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.portfolio.return_value = [
            _make_portfolio_item("AAPL", 100, 150.5),
            _make_portfolio_item("MSFT", 200, 300.0),
        ]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> Position | None:
            return await broker.get_position("AAPL")

        pos = asyncio.run(run())
        assert pos is not None
        assert pos.ticker == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_cost == 150.5

    def test_returns_none_when_ticker_absent(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.portfolio.return_value = [
            _make_portfolio_item("MSFT", 200, 300.0),
        ]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> Position | None:
            return await broker.get_position("AAPL")

        assert asyncio.run(run()) is None

    def test_skips_non_stk_contracts(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.portfolio.return_value = [
            _make_portfolio_item("AAPL", 50, 100.0, sec_type="OPT"),
        ]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> Position | None:
            return await broker.get_position("AAPL")

        assert asyncio.run(run()) is None

    def test_returns_none_when_zero_position(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.portfolio.return_value = [
            _make_portfolio_item("AAPL", 0, 0.0),
        ]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> Position | None:
            return await broker.get_position("AAPL")

        assert asyncio.run(run()) is None


# ---------------------------------------------------------------------------
# get_account_state
# ---------------------------------------------------------------------------


def _make_account_value(tag: str, value: str, currency: str = "USD") -> MagicMock:
    av = MagicMock()
    av.tag = tag
    av.value = value
    av.currency = currency
    return av


class TestGetAccountState:
    def test_builds_portfolio_state_from_ib_surfaces(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.accountValues.return_value = [
            _make_account_value("TotalCashValue", "50000.0"),
            _make_account_value("NetLiquidation", "75000.0"),
        ]
        mock_sync_broker.connection.ib.portfolio.return_value = [
            _make_portfolio_item("AAPL", 100, 150.0),
            _make_portfolio_item("MSFT", -50, 300.0),
        ]
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> PortfolioState:
            return await broker.get_account_state()

        state = asyncio.run(run())
        assert state.cash == 50000.0
        assert state.realized_pnl == 0.0  # not surfaced by IBKR
        assert state.total_commission == 0.0
        assert state.positions["AAPL"].quantity == 100
        assert state.positions["MSFT"].quantity == -50
        assert state.positions["AAPL"].avg_cost == 150.0

    def test_zero_cash_when_total_cash_value_missing(self, mock_sync_broker: MagicMock) -> None:
        mock_sync_broker.connection.ib.accountValues.return_value = []
        mock_sync_broker.connection.ib.portfolio.return_value = []
        broker = AsyncIBKRBroker(sync_broker=mock_sync_broker, account="DU1")

        async def run() -> PortfolioState:
            return await broker.get_account_state()

        state = asyncio.run(run())
        assert state.cash == 0.0
        assert dict(state.positions) == {}


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_missing_paper_account_raises_keyerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
        monkeypatch.setenv("IBKR_PORT", "7497")
        monkeypatch.setenv("IBKR_CLIENT_ID", "1")
        monkeypatch.delenv("IBKR_PAPER_ACCOUNT", raising=False)
        with pytest.raises(KeyError, match="IBKR_PAPER_ACCOUNT"):
            AsyncIBKRBroker.from_env()

    def test_missing_host_raises_keyerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IBKR_HOST", raising=False)
        monkeypatch.setenv("IBKR_PORT", "7497")
        monkeypatch.setenv("IBKR_CLIENT_ID", "1")
        monkeypatch.setenv("IBKR_PAPER_ACCOUNT", "DU123456")
        with pytest.raises(KeyError, match="IBKR_HOST"):
            AsyncIBKRBroker.from_env()

    def test_does_not_read_ibkr_account(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # IBKR_ACCOUNT is the S22 batch env var. AsyncIBKRBroker reads
        # IBKR_PAPER_ACCOUNT explicitly to keep the streaming-runtime
        # layer's naming distinct.
        monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
        monkeypatch.setenv("IBKR_PORT", "7497")
        monkeypatch.setenv("IBKR_CLIENT_ID", "1")
        monkeypatch.setenv("IBKR_ACCOUNT", "DU123456")  # set the WRONG one
        monkeypatch.delenv("IBKR_PAPER_ACCOUNT", raising=False)
        with pytest.raises(KeyError, match="IBKR_PAPER_ACCOUNT"):
            AsyncIBKRBroker.from_env()


# ---------------------------------------------------------------------------
# aclose lifecycle
# ---------------------------------------------------------------------------


class TestAclose:
    def test_aclose_shuts_down_executor(self, async_broker: AsyncIBKRBroker) -> None:
        async def run() -> None:
            await async_broker.aclose()

        asyncio.run(run())
        # Re-running aclose is idempotent
        asyncio.run(run())
        # Submitting after aclose should fail (executor shut down)
        with pytest.raises(RuntimeError):
            async_broker._executor.submit(lambda: None)


# ---------------------------------------------------------------------------
# Live smoke (opt-in via IBKR_PAPER_SMOKE; AC9)
# ---------------------------------------------------------------------------


class TestLiveSmoke:
    @pytest.mark.skipif(
        os.environ.get("IBKR_PAPER_SMOKE") != "1",
        reason="IBKR_PAPER_SMOKE=1 not set; skipping live test.",
    )
    def test_live_construct_and_account_state(self) -> None:
        async def run() -> None:
            broker = AsyncIBKRBroker.from_env()
            try:
                state = await broker.get_account_state()
                assert state is not None
                assert isinstance(state, PortfolioState)
            finally:
                await broker.aclose()

        asyncio.run(run())
