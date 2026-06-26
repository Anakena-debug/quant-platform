"""End-to-end smoke for the S35 streaming runtime.

Exercises the full stack: ``SyntheticTradeFeed`` -> ``StreamingEngine``
-> pipeline primitives -> ``StreamingStrategy`` -> ``ThreadSafeBrokerWrapper``
-> ``SafeBroker`` (with ``RiskGate`` + JSONL journal) -> ``DemoBroker``.

Pins:

- AC6: full smoke runs to completion without external network.
- AC9: ``SyntheticTradeFeed(seed=42)`` is reproducible across two
  independent constructions (test_determinism_seed_42).

quantcore-independence: no quantcore imports; the pipeline primitives
are local stubs that satisfy the structural Protocols.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming import (
    BarLike,
    DemoBroker,
    EngineConfig,
    EventClock,
    SafeBroker,
    StreamContext,
    StreamingEngine,
    SyncBrokerFacade,
    SyntheticTradeFeed,
    ThreadSafeBrokerWrapper,
    TradeEventLike,
)


# ---------------------------------------------------------------------------
# Pipeline stubs (production code injects quantcore equivalents)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Bar:
    ts_event: int
    instrument_id: int
    sequence: int
    ts_open: int
    kind: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    dollar_volume: float


class _EveryNBars:
    def __init__(self, every: int = 5) -> None:
        self._every = every
        self._n = 0

    def on_event(self, event: TradeEventLike) -> BarLike | None:
        self._n += 1
        if self._n % self._every == 0:
            return _Bar(
                ts_event=event.ts_event,
                instrument_id=event.instrument_id,
                sequence=self._n,
                ts_open=event.ts_event - 1,
                kind=1,
                open=event.price,
                high=event.price,
                low=event.price,
                close=event.price,
                volume=event.size,
                vwap=event.price,
                tick_count=1,
                dollar_volume=event.price * event.size,
            )
        return None

    def flush(self) -> BarLike | None:
        return None


class _NoCUSUM:
    def on_event(self, event: BarLike) -> int | None:
        return None

    def reset(self) -> None:
        pass


class _NoVol:
    def on_event(self, event: BarLike) -> float | None:
        return None

    def reset(self) -> None:
        pass


class _TradingStrategy:
    """Submits one BUY on first bar; later bars are no-op."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.fired = False
        self.last_fills: list[int] = []

    def on_bar(self, ts: int, bar: BarLike, ctx: StreamContext, broker: SyncBrokerFacade) -> None:
        if self.fired:
            return
        self.fired = True
        order = Order(
            order_id=uuid4(),
            ticker=self.ticker,
            side=OrderSide.BUY,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=100.0,
        )
        fills = broker.submit_order(order, timeout=2.0)
        self.last_fills.extend(f.signed_quantity for f in fills)

    def on_cusum(self, ts, event, ctx, broker) -> None:
        pass

    def on_vol(self, ts, sigma, ctx, broker) -> None:
        pass


# ---------------------------------------------------------------------------
# Loop-on-thread fixture (same pattern as PR2/PR4 tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def loop_thread() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_runner, name="e2e-loop", daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def _run_coro(loop: asyncio.AbstractEventLoop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=15.0)


# ---------------------------------------------------------------------------
# AC6: full e2e smoke, no external network
# ---------------------------------------------------------------------------
def test_e2e_synthetic_to_demo_broker(
    loop_thread: asyncio.AbstractEventLoop, tmp_path: Path
) -> None:
    """Feed -> engine -> strategy submits one order -> SafeBroker journals
    submit + fill -> DemoBroker updates portfolio."""
    feed = SyntheticTradeFeed(seed=11, instrument_id=1, n_events=30)
    builder = _EveryNBars(every=5)
    cusum = _NoCUSUM()
    vol = _NoVol()
    strat = _TradingStrategy(ticker="AAPL")
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities(max_order_notional=1_000_000.0)
    journal = tmp_path / "e2e.jsonl"

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        clock = EventClock()
        sb = SafeBroker(
            demo,
            gate,
            journal,
            price_provider=lambda: {"AAPL": 100.0},
            state_provider=lambda: demo.state,
            clock=clock,
        )
        wrapper = ThreadSafeBrokerWrapper(sb, loop)
        config = EngineConfig(instrument_id=1, ticker="AAPL")
        engine = StreamingEngine(feed, builder, cusum, vol, strat, wrapper, clock, config)
        await engine.run()
        await sb.aclose()

    _run_coro(loop_thread, _run())

    # Strategy fired exactly once.
    assert strat.fired is True
    assert strat.last_fills == [5]
    # Broker shows the position.
    assert demo.state.positions["AAPL"].quantity == 5
    # Journal recorded both submit and fill (and possibly more).
    records = [json.loads(line) for line in journal.read_text().splitlines() if line]
    event_types = [r["event_type"] for r in records]
    assert "submit" in event_types
    assert "fill" in event_types
    # All ts_ns values > 0 (shared-clock fix: SafeBroker's clock IS ticked by engine).
    assert all(r["ts_ns"] > 0 for r in records), event_types


# ---------------------------------------------------------------------------
# AC9: determinism (test_determinism_seed_42)
# ---------------------------------------------------------------------------
def test_determinism_seed_42() -> None:
    """Two SyntheticTradeFeed(seed=42) constructions must yield byte-
    identical event sequences (AC9)."""

    async def _collect(seed: int, n: int) -> list[tuple]:
        feed = SyntheticTradeFeed(seed=seed, instrument_id=1, n_events=n)
        out: list[tuple] = []
        async for ev in feed:
            out.append(
                (
                    ev.ts_event,
                    ev.instrument_id,
                    ev.sequence,
                    ev.price,
                    ev.size,
                    ev.aggressor_side,
                )
            )
        return out

    seq_a = asyncio.run(_collect(42, 100))
    seq_b = asyncio.run(_collect(42, 100))
    assert seq_a == seq_b, "AC9 violation: seed=42 not reproducible"


def test_different_seeds_diverge() -> None:
    """Sanity: different seeds must yield different sequences (otherwise
    the AC9 test is vacuous)."""

    async def _collect(seed: int, n: int) -> list[float]:
        feed = SyntheticTradeFeed(seed=seed, instrument_id=1, n_events=n)
        return [ev.price async for ev in feed]

    a = asyncio.run(_collect(42, 50))
    b = asyncio.run(_collect(43, 50))
    assert a != b, "different seeds must produce different sequences"
