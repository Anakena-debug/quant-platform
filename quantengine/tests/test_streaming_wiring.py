"""REC-003/004 wiring primitives (S39): SafeBroker on_fill hook + kill-switch.

The CLI bootstrap (`cli._run`, live branch) wires these into a
`RecoveryCoordinator.replay()` + `StreamingReconciler` + signal handlers;
that wiring needs a live Databento feed so it is grep-asserted (AC3/AC5).
These tests pin the unit-testable mechanisms the wiring depends on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

from quantengine.contracts.orders import Fill, Order, OrderSide, OrderType
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.risk.gate import RiskGate
from quantengine.runtime.streaming.protocols import EventClock
from quantengine.runtime.streaming.safe_broker import SafeBroker


class _FillingBroker:
    def __init__(self) -> None:
        self.calls = 0

    async def submit_order(self, order: Order) -> list[Fill]:
        self.calls += 1
        return [
            Fill(
                fill_id=uuid4(),
                order_id=order.order_id,
                ticker=order.ticker,
                signed_quantity=order.signed_quantity,
                price=150.0,
                commission=0.0,
                timestamp="2026-05-21T16:00:00Z",
            )
        ]

    async def cancel_order(self, order_id: UUID) -> bool:
        return True

    async def get_position(self, ticker: str) -> Position | None:
        return None

    async def get_account_state(self) -> PortfolioState:
        return PortfolioState(cash=1_000_000.0, positions={})


def _gate() -> RiskGate:
    return RiskGate.default_us_equities(
        max_order_notional=200_000.0,
        max_gross_leverage=2.0,
        max_position_weight=0.25,
        min_cash=0.0,
    )


def _clock() -> EventClock:
    c = EventClock()
    c.tick(1_700_000_000_000_000_000)
    return c


def _sb(tmp_path: Path, mock: _FillingBroker, **kw) -> SafeBroker:
    return SafeBroker(
        mock,
        _gate(),
        tmp_path / "j.jsonl",
        price_provider=lambda: {"AAPL": 150.0},
        state_provider=lambda: PortfolioState(cash=1_000_000.0, positions={}),
        clock=_clock(),
        **kw,
    )


def _order() -> Order:
    return Order(
        order_id=uuid4(),
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )


def test_on_fill_subscriber_invoked_after_journal(tmp_path: Path) -> None:
    seen: list[Fill] = []
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock, on_fill=seen.append)
    fills = asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())
    assert len(fills) == 1
    assert len(seen) == 1
    assert seen[0].ticker == "AAPL"
    assert seen[0].signed_quantity == 10


def test_set_fill_subscriber_post_construction(tmp_path: Path) -> None:
    seen: list[Fill] = []
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock)  # no on_fill at construction
    sb.set_fill_subscriber(seen.append)
    asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())
    assert len(seen) == 1


def test_no_subscriber_is_backward_compatible(tmp_path: Path) -> None:
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock)  # default on_fill=None
    fills = asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())
    assert len(fills) == 1  # unchanged behaviour


def test_kill_switch_rejects_presubmit(tmp_path: Path) -> None:
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock)
    assert sb.killed is False
    sb.trip_kill_switch(reason="watchdog_feed_silence")
    assert sb.killed is True

    fills = asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())

    assert fills == []  # zero fills
    assert mock.calls == 0  # broker was NEVER reached
    recs = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert recs[-1]["event_type"] == "reject"
    assert recs[-1]["check"] == "kill_switch"
    assert recs[-1]["reason"] == "watchdog_feed_silence"


def test_kill_switch_blocks_on_fill_subscriber(tmp_path: Path) -> None:
    """A tripped kill-switch returns before any fill, so the reconciler hook
    must NOT fire (no fill happened)."""
    seen: list[Fill] = []
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock, on_fill=seen.append)
    sb.trip_kill_switch()
    asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())
    assert seen == []


# ---------------------------------------------------------------------------
# D4: engine watchdog alert -> SafeBroker kill-switch (REC-004)
# ---------------------------------------------------------------------------


def test_watchdog_alert_callback_trips_kill_switch(tmp_path: Path) -> None:
    """The CLI wires EngineConfig.on_watchdog_alert to SafeBroker.trip_kill_switch.

    Pin the seam contract: invoking the callback (as the engine watchdog does
    via _fire_watchdog_alert) trips the kill-switch with a watchdog-prefixed
    reason, after which submits are rejected pre-broker."""
    mock = _FillingBroker()
    sb = _sb(tmp_path, mock)
    # This is exactly the lambda cli._run installs for the live feed.
    on_watchdog_alert = lambda reason: sb.trip_kill_switch(reason=f"watchdog_{reason}")  # noqa: E731

    assert sb.killed is False
    on_watchdog_alert("feed_silence_31.0s")  # engine fires this on silence
    assert sb.killed is True

    fills = asyncio.run(sb.submit_order(_order()))
    asyncio.run(sb.aclose())
    assert fills == []
    assert mock.calls == 0
    recs = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert recs[-1]["reason"] == "watchdog_feed_silence_31.0s"


def test_engine_config_carries_watchdog_alert_callback() -> None:
    """EngineConfig exposes the on_watchdog_alert injection point (D4)."""
    from quantengine.runtime.streaming.engine import EngineConfig

    assert "on_watchdog_alert" in EngineConfig.__dataclass_fields__
    # default is None (log-only; demo/replay unchanged)
    cfg = EngineConfig(instrument_id=0, ticker="AAPL")
    assert cfg.on_watchdog_alert is None


def test_fire_watchdog_alert_swallows_callback_errors() -> None:
    """_fire_watchdog_alert must never propagate a callback exception (a
    watchdog tick crashing would take down the background task)."""
    from quantengine.runtime.streaming.engine import EngineConfig, StreamingEngine

    def _boom(_reason: str) -> None:
        raise RuntimeError("callback blew up")

    cfg = EngineConfig(instrument_id=0, ticker="AAPL", on_watchdog_alert=_boom)
    # Build a bare engine instance without running it; call the private
    # dispatcher directly. __new__ + manual _config set avoids the full
    # protocol-typed constructor (we only exercise _fire_watchdog_alert).
    engine = StreamingEngine.__new__(StreamingEngine)
    engine._config = cfg  # type: ignore[attr-defined]
    # Should NOT raise despite the callback raising.
    engine._fire_watchdog_alert("feed_silence_99s")  # type: ignore[attr-defined]
