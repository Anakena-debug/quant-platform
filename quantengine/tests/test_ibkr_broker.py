"""Tests for IBKRBroker — the AbstractBroker implementation.

All tests use mocked ib_async.IB and a hand-rolled FakeEvent class to
simulate ib_async's eventkit callbacks. No real network calls.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from quantengine.contracts.market import MarketSnapshot
from quantengine.contracts.orders import Order
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.ibkr import connection as connection_mod
from quantengine.execution.ibkr.broker import IBKRBroker
from quantengine.execution.ibkr.config import IBKRConfig, TimeoutPolicy
from quantengine.execution.ibkr.connection import IBKRConnection
from quantengine.execution.order_state import OrderTracker
from quantengine.portfolio.ledger import Ledger


# ---- Fake event helper -----------------------------------------------


class FakeEvent:
    """Minimal eventkit-Event stand-in: collects callbacks and emits."""

    def __init__(self) -> None:
        self.callbacks: list[Callable[..., None]] = []

    def __iadd__(self, cb: Callable[..., None]) -> FakeEvent:
        self.callbacks.append(cb)
        return self

    def __isub__(self, cb: Callable[..., None]) -> FakeEvent:
        self.callbacks.remove(cb)
        return self

    def emit(self, *args: Any) -> None:
        for cb in list(self.callbacks):
            cb(*args)


# ---- Test fixtures ---------------------------------------------------


def _make_market() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-05-07T16:00:00Z",
        tickers=("AAPL", "MSFT"),
        prices=np.array([200.0, 400.0]),
    )


def _make_connected_broker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tracker: OrderTracker | None = None,
    timeouts: TimeoutPolicy | None = None,
) -> tuple[IBKRBroker, MagicMock]:
    """Build a connected IBKRBroker with a mocked ib_async.IB.

    Returns ``(broker, mock_ib)``. The mock IB has ``isConnected`` =
    True; ``placeOrder`` returns a fresh MagicMock Trade (with
    ``fillEvent`` and ``statusEvent`` as ``FakeEvent`` instances and
    ``isDone`` returning False); ``sleep`` is a no-op.
    """
    mock_ib = MagicMock()
    mock_ib.isConnected.return_value = True
    mock_ib.sleep.return_value = None

    monkeypatch.setattr(connection_mod, "_new_ib", lambda: mock_ib)

    cfg = IBKRConfig(host="127.0.0.1", port=7497, client_id=42, account="DU123")
    conn = IBKRConnection()
    conn.connect(cfg)

    broker = IBKRBroker(
        connection=conn,
        tracker=tracker,
        timeouts=timeouts or TimeoutPolicy(),
    )
    return broker, mock_ib


def _new_mock_trade(orderId: int = 100) -> MagicMock:
    """Make a mock ib_async.Trade with FakeEvent fillEvent/statusEvent."""
    trade = MagicMock()
    trade.fillEvent = FakeEvent()
    trade.statusEvent = FakeEvent()
    trade.orderStatus.status = "PendingSubmit"
    trade.orderStatus.whyHeld = ""
    trade.isDone.return_value = False
    trade.order.orderId = orderId
    return trade


def _emit_fill(trade: MagicMock, qty: int, price: float = 200.0) -> None:
    fill_event = MagicMock()
    fill_event.execution.shares = qty
    fill_event.execution.price = price
    fill_event.execution.permId = 12345
    fill_event.execution.execId = f"exec.{qty}"
    fill_event.execution.time = "2026-05-07T16:00:00Z"
    fill_event.commissionReport.commission = 1.0
    trade.fillEvent.emit(trade, fill_event)


def _emit_status(trade: MagicMock, status: str) -> None:
    trade.orderStatus.status = status
    trade.statusEvent.emit(trade)
    # IBKR terminal statuses that flip Trade.isDone() True. "Inactive"
    # is the IBKR side's terminal state for orders rejected at the
    # gateway (contract validation, account permissions, etc.).
    if status in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
        trade.isDone.return_value = True


# ---- AbstractBroker interface ----------------------------------------


def test_ibkr_broker_implements_abstract_broker_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IBKRBroker is an AbstractBroker with the documented method signatures."""
    broker, _ = _make_connected_broker(monkeypatch)
    assert isinstance(broker, AbstractBroker)
    # Methods exist and are callable
    assert callable(broker.submit_orders)
    assert callable(broker.cancel_all)
    assert callable(broker.open_orders)


def test_ibkr_broker_empty_order_list_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_orders([], market) → []; placeOrder never invoked."""
    broker, mock_ib = _make_connected_broker(monkeypatch)
    fills = broker.submit_orders([], _make_market())
    assert fills == []
    mock_ib.placeOrder.assert_not_called()
    mock_ib.sleep.assert_not_called()


def test_ibkr_broker_uses_ib_assigned_order_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IBOrder passed to placeOrder has orderId=0 (delegate to IB)."""
    broker, mock_ib = _make_connected_broker(monkeypatch)
    trade = _new_mock_trade(orderId=999)
    trade.isDone.return_value = True  # immediately done so loop exits
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    broker.submit_orders([order], _make_market())

    mock_ib.placeOrder.assert_called_once()
    contract, ib_order = mock_ib.placeOrder.call_args.args
    assert contract.symbol == "AAPL"
    # Critical: orderId is 0 → IB.placeOrder assigns via client.getReqId().
    assert ib_order.orderId == 0


# ---- Tracker routing ------------------------------------------------


def test_ibkr_broker_submit_orders_routes_through_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tracker.submit + ack + on_fill all called for a happy-path order."""
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    # Drive: first sleep emits Submitted, second emits fill, third emits Filled.
    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Submitted")
        elif state[0] == 1:
            _emit_fill(trade, 10)
        elif state[0] == 2:
            _emit_status(trade, "Filled")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)

    # tracker.submit was called once (PENDING → SUBMITTED before placeOrder).
    # tracker.ack moved SUBMITTED → WORKING after Submitted statusEvent.
    # tracker.on_fill moved WORKING → FILLED after the fill_event.
    assert len(fills) == 1
    assert fills[0].signed_quantity == 10
    # Final status is FILLED.
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.FILLED


def test_ibkr_broker_handles_partial_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two fills (50+50 of a 100-share parent) → tracker reaches FILLED only after the second."""
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=100)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Submitted")
        elif state[0] == 1:
            _emit_fill(trade, 50)
        elif state[0] == 2:
            # Before the second fill, status should be PARTIALLY_FILLED.
            from quantengine.contracts.orders import OrderStatus

            assert tracker.status(order.order_id) == OrderStatus.PARTIALLY_FILLED
            _emit_fill(trade, 50)
        elif state[0] == 3:
            _emit_status(trade, "Filled")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert len(fills) == 2
    assert sum(f.signed_quantity for f in fills) == 100
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.FILLED


def test_ibkr_broker_handles_broker_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelled status (no prior ack) → tracker.cancel; no Fill returned.

    The order is in SUBMITTED state when the broker emits Cancelled
    (e.g. pre-trade gate rejection). Both ``cancel`` and ``reject`` are
    legal from SUBMITTED, but the broker treats every "Cancelled"
    status as a cancellation in the audit trail — REJECTED is reserved
    for explicit pre-trade refusals where the IBKR side actively
    refuses to place the order. The whyHeld text is preserved as the
    cancel ``reason`` for forensic analysis.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    trade.orderStatus.whyHeld = "insufficient_buying_power"
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Cancelled")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert fills == []
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 1
    payload = cancel_events[0].payload
    assert isinstance(payload, dict)
    assert payload["reason"] == "insufficient_buying_power"


def test_ibkr_broker_handles_cancellation_after_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitted → Cancelled sequence: order reaches WORKING, then cancels.

    Regression: the previous implementation routed Cancelled → tracker.reject,
    which raises OrderStateError because the legal transition table forbids
    WORKING → REJECTED. Verified runtime crash before the fix; this test
    pins the contract that WORKING → CANCELLED is the correct path and that
    submit_orders returns cleanly with an empty fills list.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    trade.orderStatus.whyHeld = "wash_trade_gate"
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            # Broker acks → SUBMITTED → WORKING.
            _emit_status(trade, "Submitted")
        elif state[0] == 1:
            # Broker cancels in-flight → must transition WORKING → CANCELLED.
            _emit_status(trade, "Cancelled")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    # If the bug were still present this would raise OrderStateError.
    fills = broker.submit_orders([order], market)
    assert fills == []
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 1
    assert cancel_events[0].payload["reason"] == "wash_trade_gate"


# ---- Timeouts --------------------------------------------------------


def test_ibkr_broker_cancels_on_per_order_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-order timeout fires → ib.cancelOrder called → tracker.cancel(per_order_timeout).

    The per-order timeout path uses ``tracker.cancel`` (not ``reject``)
    because the order may have been acked into WORKING before the
    timeout fires — and ``WORKING → REJECTED`` is illegal per the
    state-machine table. See order_state.py:_LEGAL.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(
        monkeypatch,
        tracker=tracker,
        timeouts=TimeoutPolicy(per_order_seconds=0.05, batch_ceiling_seconds=10.0),
    )
    trade = _new_mock_trade(orderId=100)
    # isDone stays False forever (will only flip if cancel→Cancelled status).
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    # Each ib.sleep increments mock time past per_order deadline.
    cancel_called = [False]

    def cancel_side_effect(_ib_order: Any) -> None:
        cancel_called[0] = True
        _emit_status(trade, "Cancelled")  # broker confirms cancel

    mock_ib.cancelOrder.side_effect = cancel_side_effect

    # Real-time delays would slow the test; per_order=0.05 fires after the
    # first sleep slice (0.25s) since we let real time pass during sleep.
    # We use real time by NOT mocking time.sleep — but mock_ib.sleep is a no-op.
    # The broker's loop calls time.monotonic() between iterations; after 0.05s
    # of real time the per-order deadline has fired. Since ib.sleep is mocked
    # to no-op, we need to inject real sleep via side_effect.
    import time as time_mod

    def slow_sleep(_: float) -> None:
        time_mod.sleep(0.06)  # > per_order_seconds

    mock_ib.sleep.side_effect = slow_sleep

    fills = broker.submit_orders([order], market)
    assert fills == []
    assert cancel_called[0]
    # tracker.cancel was called with reason="per_order_timeout"
    # The order should be in CANCELLED state.
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    # Verify the reason in the ledger
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 1
    payload = cancel_events[0].payload
    assert isinstance(payload, dict)
    assert payload["reason"] == "per_order_timeout"


def test_ibkr_broker_per_order_timeout_after_ack_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-order timeout AFTER ack: SUBMITTED → WORKING → CANCELLED.

    Regression: the previous implementation routed timeouts to
    tracker.reject, which raises OrderStateError because the legal
    transition table forbids WORKING → REJECTED. This test exercises
    the previously-uncovered path where an order is acked before the
    timeout fires.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(
        monkeypatch,
        tracker=tracker,
        timeouts=TimeoutPolicy(per_order_seconds=0.05, batch_ceiling_seconds=10.0),
    )
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    cancel_called = [False]

    def cancel_side_effect(_ib_order: Any) -> None:
        cancel_called[0] = True
        _emit_status(trade, "Cancelled")

    mock_ib.cancelOrder.side_effect = cancel_side_effect

    import time as time_mod

    sleep_count = [0]

    def driver(_: float) -> None:
        # First slice: emit Submitted → tracker advances to WORKING.
        # Then let real time pass through the deadline.
        if sleep_count[0] == 0:
            _emit_status(trade, "Submitted")
        time_mod.sleep(0.06)  # > per_order_seconds
        sleep_count[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert fills == []
    assert cancel_called[0]
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 1
    assert cancel_events[0].payload["reason"] == "per_order_timeout"


def test_ibkr_broker_cancels_on_batch_ceiling_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch ceiling fires → all remaining orders cancelled with batch_ceiling_timeout reason.

    Same state-machine rationale as per-order timeout: routes through
    ``tracker.cancel`` (not ``reject``) because orders may have been
    acked before the ceiling fires.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(
        monkeypatch,
        tracker=tracker,
        # per_order well above batch_ceiling so batch fires first
        timeouts=TimeoutPolicy(per_order_seconds=10.0, batch_ceiling_seconds=0.05),
    )
    trade1 = _new_mock_trade(orderId=100)
    trade2 = _new_mock_trade(orderId=101)
    placed = [trade1, trade2]

    def place_order_side_effect(_c: Any, _o: Any) -> MagicMock:
        return placed.pop(0)

    mock_ib.placeOrder.side_effect = place_order_side_effect

    order1 = Order.new(ticker="AAPL", signed_quantity=10)
    order2 = Order.new(ticker="MSFT", signed_quantity=20)
    market = _make_market()

    cancelled_orders: list[Any] = []

    def cancel_side_effect(ib_order: Any) -> None:
        cancelled_orders.append(ib_order)
        # Emit Cancelled for each — this flips isDone via _emit_status.
        if ib_order is trade1.order:
            _emit_status(trade1, "Cancelled")
        elif ib_order is trade2.order:
            _emit_status(trade2, "Cancelled")

    mock_ib.cancelOrder.side_effect = cancel_side_effect

    import time as time_mod

    def slow_sleep(_: float) -> None:
        time_mod.sleep(0.06)  # > batch_ceiling_seconds

    mock_ib.sleep.side_effect = slow_sleep

    fills = broker.submit_orders([order1, order2], market)
    assert fills == []
    assert len(cancelled_orders) == 2
    # Both orders should have been routed via tracker.cancel with batch_ceiling_timeout.
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 2
    for ev in cancel_events:
        assert isinstance(ev.payload, dict)
        assert ev.payload["reason"] == "batch_ceiling_timeout"


def test_ibkr_broker_batch_ceiling_drains_via_continued_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-ceiling cancellations drain through additional ib.sleep slices.

    Regression: the previous implementation called ``break`` immediately
    after issuing the batch-ceiling ``cancelOrder`` calls, so the
    asyncio loop never got to flush cancel bytes and the IBKR side
    could be left with orphaned working orders. The fix replaced the
    break with a ``batch_cancel_issued`` flag so the loop continues
    running additional ``ib.sleep`` slices until each Trade.isDone
    flips True (or until callers' own outer timeouts fire).

    This test pins that behavior: the broker's "Cancelled" callback
    only flips isDone on the *second* ib.sleep iteration after the
    batch-ceiling fires, so a working drain requires at least two
    additional sleep calls beyond the cancellation point. With the
    old `break`, those calls would not happen and isDone would remain
    False, hanging the test (caught by submit_orders never returning).
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(
        monkeypatch,
        tracker=tracker,
        timeouts=TimeoutPolicy(per_order_seconds=10.0, batch_ceiling_seconds=0.05),
    )
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    # Cancel does NOT immediately flip isDone — instead the broker
    # confirms the cancellation on a later ib.sleep slice (simulating
    # network latency between cancelOrder and the Cancelled callback).
    cancel_request_count = [0]
    sleeps_after_cancel = [0]
    cancel_observed_at_sleep_idx = [-1]

    def cancel_side_effect(_ib_order: Any) -> None:
        cancel_request_count[0] += 1
        cancel_observed_at_sleep_idx[0] = sleeps_after_cancel[0]

    mock_ib.cancelOrder.side_effect = cancel_side_effect

    import time as time_mod

    def driver(_: float) -> None:
        if cancel_request_count[0] > 0:
            sleeps_after_cancel[0] += 1
            # Two slices after the cancel request, the broker confirms.
            if sleeps_after_cancel[0] >= 2:
                _emit_status(trade, "Cancelled")
        time_mod.sleep(0.06)  # ensure batch_ceiling fires on first slice

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert fills == []
    # cancelOrder issued exactly once (idempotent flag prevents re-issue).
    assert cancel_request_count[0] == 1
    # The loop ran AT LEAST 2 more sleep slices after issuing cancel —
    # which is what flushes the cancel through the asyncio socket and
    # lets the Cancelled callback land.
    assert sleeps_after_cancel[0] >= 2, (
        f"loop did not drain after batch_ceiling — only {sleeps_after_cancel[0]} "
        "sleep slices after cancellation; expected ≥ 2"
    )
    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED


# ---- Race-safety, Inactive mapping, post-loop reconciliation --------


def test_ibkr_broker_fill_after_cancel_request_is_ledgered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race-safety regression: late fill between cancelOrder and Cancelled status.

    Sequence: ack → per-order timeout fires → ib.cancelOrder issued (but
    broker has NOT yet confirmed cancellation) → partial fill arrives
    BEFORE the Cancelled status callback → broker confirms cancellation.

    Old behavior: timeout marked tracker CANCELLED immediately, so
    on_fill_cb saw a terminal tracker and silently dropped the fill from
    the tracker/ledger while still appending to collected_fills. Result:
    ledger and returned-fills divergence (a true correctness bug).

    Race-safe behavior: timeout only requests cancellation; tracker stays
    non-terminal (PARTIALLY_FILLED after the late fill). Cancelled status
    callback then transitions PARTIALLY_FILLED → CANCELLED with the
    per_order_timeout reason and the cumulative fill quantity preserved.
    Returned fills list and ledger remain consistent.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(
        monkeypatch,
        tracker=tracker,
        timeouts=TimeoutPolicy(per_order_seconds=0.05, batch_ceiling_seconds=10.0),
    )
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=100)
    market = _make_market()

    cancel_call_count = [0]

    def cancel_side_effect(_ib_order: Any) -> None:
        # Simulate socket-flushed cancel request that has NOT yet been
        # confirmed by the broker.
        cancel_call_count[0] += 1

    mock_ib.cancelOrder.side_effect = cancel_side_effect

    import time as time_mod

    sleep_count = [0]

    def driver(_: float) -> None:
        if sleep_count[0] == 0:
            # Slice 1: ack the order so it reaches WORKING before the
            # timeout fires.
            _emit_status(trade, "Submitted")
        elif cancel_call_count[0] > 0 and sleep_count[0] == 2:
            # Two slices after the cancel was requested, IBKR delivers a
            # partial fill that crossed the cancel on the wire.
            _emit_fill(trade, 30)
        elif cancel_call_count[0] > 0 and sleep_count[0] == 3:
            # Then the broker confirms the cancellation.
            _emit_status(trade, "Cancelled")
        time_mod.sleep(0.06)  # > per_order_seconds
        sleep_count[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)

    # The late fill IS in collected_fills (race-safe path preserved it).
    assert len(fills) == 1
    assert fills[0].signed_quantity == 30

    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    assert tracker.cumulative_filled(order.order_id) == 30
    assert tracker.remaining(order.order_id) == 70

    fill_events = [e for e in ledger.events() if e.kind == "ORDER_FILLED"]
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(fill_events) == 1, (
        "(race-safety) the late fill MUST be ledgered; with the old "
        "timeout-marks-terminal code it was silently dropped"
    )
    assert len(cancel_events) == 1
    payload = cancel_events[0].payload
    assert isinstance(payload, dict)
    assert payload["reason"] == "per_order_timeout"
    assert payload["cumulative_filled"] == 30
    assert payload["remaining"] == 70


def test_ibkr_broker_inactive_status_from_submitted_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inactive status from SUBMITTED → tracker.reject (legal pre-trade refusal).

    IBKR sends "Inactive" for orders that fail at the gateway (contract
    validation, permissions, etc.). When the order is still SUBMITTED in
    our tracker, this is a pre-trade rejection and routes to reject.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    trade.orderStatus.whyHeld = "contract_validation_failed"
    trade.advancedError = "ESecCodeInvalid: invalid symbol"
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Inactive")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert fills == []

    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.REJECTED
    reject_events = [e for e in ledger.events() if e.kind == "ORDER_REJECTED"]
    assert len(reject_events) == 1
    payload = reject_events[0].payload
    assert isinstance(payload, dict)
    # Reason concatenates the inactive marker, whyHeld, and advancedError.
    assert "inactive" in payload["reason"]
    assert "whyHeld=contract_validation_failed" in payload["reason"]
    assert "advancedError=ESecCodeInvalid: invalid symbol" in payload["reason"]


def test_ibkr_broker_inactive_status_from_working_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inactive status from WORKING → tracker.cancel (REJECTED would crash).

    Once an order has been acked into WORKING, the legal transition table
    forbids WORKING → REJECTED. An Inactive that arrives post-ack is
    routed to tracker.cancel instead, preserving the audit trail and the
    inactive reason without crashing the event loop.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    trade.orderStatus.whyHeld = "post_ack_invalidation"
    trade.advancedError = ""
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Submitted")
        elif state[0] == 1:
            _emit_status(trade, "Inactive")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)
    assert fills == []

    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.CANCELLED
    cancel_events = [e for e in ledger.events() if e.kind == "ORDER_CANCELLED"]
    assert len(cancel_events) == 1
    payload = cancel_events[0].payload
    assert isinstance(payload, dict)
    assert "inactive" in payload["reason"]
    assert "whyHeld=post_ack_invalidation" in payload["reason"]


def test_ibkr_broker_done_trade_with_non_terminal_tracker_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-loop reconciliation: isDone=True with non-terminal tracker → RuntimeError.

    Simulates an unmapped IBKR status: trade.isDone() flips True (e.g.,
    via a future status string we don't yet handle in on_status_cb), but
    the tracker is never advanced. Without post-loop reconciliation this
    would leave the ledger silently inconsistent with the broker; with
    it, submit_orders raises with a forensic message naming the
    unmapped IBKR status.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=10)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        # Flip isDone without emitting a known terminal status — this
        # simulates an unmapped IBKR status string. The orderStatus
        # field is set so the RuntimeError can name what it found.
        if state[0] == 0:
            trade.orderStatus.status = "FuturisticUnknownStatus"
            trade.isDone.return_value = True
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    with pytest.raises(RuntimeError, match="OrderTracker remains non-terminal"):
        broker.submit_orders([order], market)


def test_ibkr_broker_multiple_partial_fills_for_one_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI-1 execution-level: one order may produce multiple Fill objects.

    Pins the contract documented on AbstractBroker (see CI-1 in
    quantengine/execution/broker.py): callers must aggregate fills by
    Fill.order_id; the returned list does NOT contain at most one fill
    per order. Two partial fills for one parent order is the canonical
    multi-fill case.
    """
    ledger = Ledger()
    tracker = OrderTracker(ledger=ledger)
    broker, mock_ib = _make_connected_broker(monkeypatch, tracker=tracker)
    trade = _new_mock_trade(orderId=100)
    mock_ib.placeOrder.return_value = trade

    order = Order.new(ticker="AAPL", signed_quantity=100)
    market = _make_market()

    state = [0]

    def driver(_: float) -> None:
        if state[0] == 0:
            _emit_status(trade, "Submitted")
        elif state[0] == 1:
            _emit_fill(trade, 30)
        elif state[0] == 2:
            _emit_fill(trade, 70)
        elif state[0] == 3:
            _emit_status(trade, "Filled")
        state[0] += 1

    mock_ib.sleep.side_effect = driver

    fills = broker.submit_orders([order], market)

    # Two fills returned for ONE submitted order — explicit CI-1 violation
    # of the old (now-corrected) "one fill per terminally-filled order"
    # wording, and explicit confirmation of the new execution-level wording.
    assert len(fills) == 2
    by_order_id: dict[Any, int] = {}
    for f in fills:
        by_order_id[f.order_id] = by_order_id.get(f.order_id, 0) + f.signed_quantity
    # Aggregate matches the parent order's signed_quantity for terminal-FILLED.
    assert by_order_id[order.order_id] == order.signed_quantity == 100

    from quantengine.contracts.orders import OrderStatus

    assert tracker.status(order.order_id) == OrderStatus.FILLED


# ---- cancel_all ------------------------------------------------------


def test_ibkr_broker_cancel_all_returns_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cancel_all returns count (before - after) of openTrades."""
    broker, mock_ib = _make_connected_broker(monkeypatch)
    # Three open trades before; zero after.
    states = [
        [MagicMock(), MagicMock(), MagicMock()],  # before
        [],  # after
    ]

    def open_trades_side_effect() -> list[MagicMock]:
        return states.pop(0)

    mock_ib.openTrades.side_effect = open_trades_side_effect

    count = broker.cancel_all()
    assert count == 3
    mock_ib.reqGlobalCancel.assert_called_once()
