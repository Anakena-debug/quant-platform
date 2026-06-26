"""Tests for StreamingReconciler (S37 PR2).

Coverage:

- ``TestConstruction``: tolerance/debounce validation.
- ``TestWithinTolerance``: silent pass on diff ≤ tolerance.
- ``TestExceedsTolerance``: halt_callback fires with reason; sticky
  halted state; subsequent fills ignored.
- ``TestPositionEventFastPath``: on_position_event cancels the timer
  and reconciles using the pushed position (no get_position call).
- ``TestTimerFallback``: when no position-event arrives, the timer
  fires and the reconciler calls broker.get_position.
- ``TestBurstFills``: burst-of-fills extends the timer; single
  reconciliation runs after the burst.
- ``TestAvgPriceDiff``: avg-price divergence with matching quantity
  logs WARNING but does NOT halt.
- ``TestUnsolicitedPositionEvent``: on_position_event with no
  pending timer is ignored.
- ``TestNonePositions``: vp empty + broker has position, vice versa,
  both empty.
- ``TestAclose``: pending timers + tasks cleaned up.

All async; uses asyncio.run + tight debounce_s for fast tests.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Fill
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming.reconciler import StreamingReconciler


def _make_fill(ticker: str, signed_qty: int = 10) -> Fill:
    return Fill(
        fill_id=uuid4(),
        order_id=uuid4(),
        ticker=ticker,
        signed_quantity=signed_qty,
        price=100.0,
        commission=0.0,
        timestamp="2026-05-22T15:00:00Z",
    )


def _state(positions: dict[str, Position] | None = None) -> PortfolioState:
    return PortfolioState(cash=0.0, positions=positions or {})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_negative_tolerance_raises(self) -> None:
        broker = AsyncMock()
        with pytest.raises(ValueError, match="tolerance"):
            StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(),
                halt_callback=AsyncMock(),
                tolerance=-1,
            )

    def test_zero_debounce_raises(self) -> None:
        broker = AsyncMock()
        with pytest.raises(ValueError, match="debounce_s"):
            StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(),
                halt_callback=AsyncMock(),
                debounce_s=0,
            )

    def test_defaults(self) -> None:
        r = StreamingReconciler(
            broker=AsyncMock(),
            virtual_portfolio_provider=lambda: _state(),
            halt_callback=AsyncMock(),
        )
        assert r.tolerance == 1
        assert r.halted is False


# ---------------------------------------------------------------------------
# Within tolerance — silent pass
# ---------------------------------------------------------------------------


class TestWithinTolerance:
    def test_matching_quantities_no_halt(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position(ticker="AAPL", quantity=10, avg_cost=100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 10, 100.0)}),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)  # let debounce fire
            assert r.halted is False
            halt.assert_not_awaited()

        asyncio.run(run())

    def test_diff_at_tolerance_no_halt(self) -> None:
        """diff == tolerance (default 1) is within bounds, no halt."""

        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(
                    {"AAPL": Position("AAPL", 11, 100.0)}  # diff=1, tolerance=1
                ),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is False
            halt.assert_not_awaited()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Exceeds tolerance — halt + sticky
# ---------------------------------------------------------------------------


class TestExceedsTolerance:
    def test_diff_exceeds_tolerance_triggers_halt(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(
                    {"AAPL": Position("AAPL", 13, 100.0)}  # diff=3, tolerance=1
                ),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is True
            halt.assert_awaited_once()
            reason = halt.call_args.args[0]
            assert "diff=3" in reason
            assert "AAPL" in reason

        asyncio.run(run())

    def test_halt_is_sticky_subsequent_fills_ignored(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 13, 100.0)}),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is True
            assert halt.await_count == 1
            # Subsequent fills go nowhere
            r.on_fill(_make_fill("AAPL"))
            r.on_fill(_make_fill("MSFT"))
            await asyncio.sleep(0.05)
            assert halt.await_count == 1  # still one
            # broker.get_position never called for the post-halt fills
            assert broker.get_position.await_count == 1  # only the initial halt's fallback

        asyncio.run(run())

    def test_custom_tolerance(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(
                    {"AAPL": Position("AAPL", 14, 100.0)}  # diff=4
                ),
                halt_callback=halt,
                tolerance=5,  # 4 < 5, no halt
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is False

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Position-event fast path
# ---------------------------------------------------------------------------


class TestPositionEventFastPath:
    def test_position_event_cancels_timer_and_reconciles(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            # Configure get_position to raise; if the fast path works,
            # get_position should NEVER be called.
            broker.get_position.side_effect = AssertionError(
                "get_position must not be called on fast path"
            )
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 10, 100.0)}),
                halt_callback=halt,
                debounce_s=0.1,  # generous; we'll fire position event before it
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.01)  # less than debounce
            # Broker pushed a position — should cancel the timer
            await r.on_position_event("AAPL", Position("AAPL", 10, 100.0))
            await asyncio.sleep(0.15)  # past the original debounce; timer should be dead
            assert r.halted is False
            halt.assert_not_awaited()
            broker.get_position.assert_not_called()

        asyncio.run(run())

    def test_position_event_triggers_halt_when_diverged(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 100, 100.0)}),
                halt_callback=halt,
                debounce_s=0.1,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.01)
            # Broker pushes a position with big quantity divergence
            await r.on_position_event("AAPL", Position("AAPL", 10, 100.0))
            assert r.halted is True
            halt.assert_awaited_once()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Timer fallback
# ---------------------------------------------------------------------------


class TestTimerFallback:
    def test_timer_fires_get_position(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 10, 100.0)}),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            broker.get_position.assert_awaited_once_with("AAPL")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Burst-of-fills extends timer
# ---------------------------------------------------------------------------


class TestBurstFills:
    def test_burst_extends_timer_single_reconciliation(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 10, 100.0)}),
                halt_callback=halt,
                debounce_s=0.05,
            )
            # 5 fills back-to-back within the debounce window
            for _ in range(5):
                r.on_fill(_make_fill("AAPL"))
                await asyncio.sleep(0.01)  # 5x10ms = 50ms total, just at debounce
            # The earliest fills' timer was cancelled by the later ones.
            # Wait for the final timer to fire.
            await asyncio.sleep(0.10)
            # Only one reconciliation (one get_position call) should have fired.
            assert broker.get_position.await_count == 1

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Avg-price WARNING (no halt)
# ---------------------------------------------------------------------------


class TestAvgPriceDiff:
    def test_avg_price_diff_logs_warning_no_halt(self, caplog: pytest.LogCaptureFixture) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 101.50)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(
                    # quantity matches but avg differs by $1.50
                    {"AAPL": Position("AAPL", 10, 100.0)}
                ),
                halt_callback=halt,
                debounce_s=0.02,
            )
            with caplog.at_level(logging.WARNING):
                r.on_fill(_make_fill("AAPL"))
                await asyncio.sleep(0.05)
            assert r.halted is False
            halt.assert_not_awaited()
            # Warning logged with both prices in the message
            warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
            assert len(warnings) >= 1
            assert "avg-price divergence" in warnings[0].message

        asyncio.run(run())

    def test_avg_price_match_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state({"AAPL": Position("AAPL", 10, 100.0)}),
                halt_callback=halt,
                debounce_s=0.02,
            )
            with caplog.at_level(logging.WARNING):
                r.on_fill(_make_fill("AAPL"))
                await asyncio.sleep(0.05)
            warnings = [
                rec
                for rec in caplog.records
                if rec.levelno == logging.WARNING and "avg-price" in rec.message
            ]
            assert len(warnings) == 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Unsolicited position event
# ---------------------------------------------------------------------------


class TestUnsolicitedPositionEvent:
    def test_no_pending_timer_means_ignored(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(),
                halt_callback=halt,
                debounce_s=0.02,
            )
            # Broker pushed without our having any pending Fill
            await r.on_position_event("AAPL", Position("AAPL", 999, 100.0))
            # Ignored: no halt, no get_position
            assert r.halted is False
            halt.assert_not_awaited()
            broker.get_position.assert_not_called()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# None-position cases
# ---------------------------------------------------------------------------


class TestNonePositions:
    def test_vp_empty_broker_empty(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = None
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(),  # empty
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is False  # 0 vs 0

        asyncio.run(run())

    def test_vp_has_broker_empty_exceeds_tolerance(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = None
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(
                    {"AAPL": Position("AAPL", 100, 100.0)}  # vp has 100, broker has 0
                ),
                halt_callback=halt,
                debounce_s=0.02,
            )
            r.on_fill(_make_fill("AAPL"))
            await asyncio.sleep(0.05)
            assert r.halted is True

        asyncio.run(run())


# ---------------------------------------------------------------------------
# aclose
# ---------------------------------------------------------------------------


class TestAclose:
    def test_aclose_cancels_pending_timers(self) -> None:
        async def run() -> None:
            broker = AsyncMock()
            broker.get_position.return_value = Position("AAPL", 10, 100.0)
            halt = AsyncMock()
            r = StreamingReconciler(
                broker=broker,
                virtual_portfolio_provider=lambda: _state(),
                halt_callback=halt,
                debounce_s=1.0,  # long debounce so we close before it fires
            )
            r.on_fill(_make_fill("AAPL"))
            r.on_fill(_make_fill("MSFT"))
            await asyncio.sleep(0.01)
            await r.aclose()
            # Wait long enough to verify the timers really were cancelled
            await asyncio.sleep(0.05)
            broker.get_position.assert_not_called()

        asyncio.run(run())
