"""Hermetic + live-gated tests for DatabentoTradeFeed.

Coverage map:

- ``TestSideToAggressor``: pure mapping function, no SDK dependency.
- ``TestRecordToTradeEvent``: non-trade records map to None. Mapping of
  a real ``ddbn.TradeMsg`` is exercised by ``TestReplay`` against the
  committed fixture (skipped at CI without fixture; re-run pre-seal
  by the operator per amended D5 of s36 plan).
- ``TestEnqueueBackpressure``: ``put_nowait`` + ``QueueFull`` path
  bumps ``feed.metrics.backpressure_drops`` (amended D5 + D7).
- ``TestReconnectMetric``: ``_on_reconnect`` increments
  ``feed.metrics.reconnects_total`` (amended D7).
- ``TestOnRecordBridge``: ``_on_record`` schedules ``_enqueue`` on the
  captured event loop via ``call_soon_threadsafe``.
- ``TestReplay``: end-to-end replay path against the captured fixture.
  Guarded by ``pytest.mark.skipif`` on missing fixture (amended D5).
- ``TestLiveSmoke``: opt-in live test against real Databento. Guarded
  by ``DATABENTO_LIVE=1`` (AC9). Not run in CI.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import databento as _db_real
import pytest

from quantengine.runtime.streaming.databento_config import DatabentoConfig
from quantengine.runtime.streaming.databento_feed import (
    DatabentoTradeFeed,
    FeedMetrics,
    _record_to_trade_event,
    _side_to_aggressor,
)
from quantengine.runtime.streaming.protocols import TradeEventLike

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "databento_dbn" / "sample_trades.dbn"


# ---------------------------------------------------------------------------
# Stub SDK Live client + monkeypatch fixture
# ---------------------------------------------------------------------------


class _StubLive:
    """Records SDK lifecycle calls for verification without network I/O."""

    instances: list[_StubLive] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.subscriptions: list[dict[str, object]] = []
        self.callback: Any = None
        self.reconnect_callback: Any = None
        self.started = False
        self.stopped = False
        _StubLive.instances.append(self)

    def subscribe(self, dataset: str, schema: str, symbols: list[str]) -> None:
        self.subscriptions.append({"dataset": dataset, "schema": schema, "symbols": symbols})

    def add_callback(self, fn: Any) -> None:
        self.callback = fn

    def add_reconnect_callback(self, fn: Any) -> None:
        self.reconnect_callback = fn

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _DbProxy:
    """Module-namespace shim: stubs Live, leaves read_dbn real for replay tests."""

    Live = _StubLive
    read_dbn = staticmethod(_db_real.read_dbn)


@pytest.fixture
def stub_live(monkeypatch: pytest.MonkeyPatch) -> type[_StubLive]:
    _StubLive.instances = []
    monkeypatch.setattr("quantengine.runtime.streaming.databento_feed.db", _DbProxy)
    return _StubLive


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestSideToAggressor:
    # DBN spec (s83 F11): side = side of the AGGRESSOR for trades.
    # 'A' (Ask) = sell aggressor -> -1; 'B' (Bid) = buy aggressor -> +1.
    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            ("A", -1),
            ("B", 1),
            ("N", 0),
            ("", 0),
            ("X", 0),
            (65, -1),
            (66, 1),
            (78, 0),
            (b"A", -1),
            (b"B", 1),
            (b"N", 0),
        ],
    )
    def test_maps_correctly(self, side: object, expected: int) -> None:
        assert _side_to_aggressor(side) == expected

    def test_agrees_with_quantcore_side_enum_and_batch_map(self) -> None:
        """Three-way s83 F11 pin: adapter ints == quantcore Side enum ==
        batch _SIDE_MAP. A drift in any one of the three is a silent
        train/serve sign skew on every signed-flow feature."""
        from quantcore.data.events import Side
        from quantcore.features.top_of_book import _SIDE_MAP

        assert _side_to_aggressor("B") == int(Side.BID) == int(_SIDE_MAP["B"]) == 1
        assert _side_to_aggressor("A") == int(Side.ASK) == int(_SIDE_MAP["A"]) == -1
        assert _side_to_aggressor("N") == 0

    def test_out_of_range_int_returns_zero(self) -> None:
        # chr() raises ValueError above 0x10FFFF
        assert _side_to_aggressor(0x110000) == 0


class TestRecordToTradeEvent:
    def test_non_trademsg_returns_none(self) -> None:
        assert _record_to_trade_event(object()) is None
        assert _record_to_trade_event("not a record") is None
        assert _record_to_trade_event(42) is None
        assert _record_to_trade_event(None) is None


# ---------------------------------------------------------------------------
# Backpressure + reconnect counters
# ---------------------------------------------------------------------------


class TestEnqueueBackpressure:
    def test_queue_full_increments_drops(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"], queue_size=2)
            feed._enqueue("rec1")
            feed._enqueue("rec2")
            assert feed.metrics.backpressure_drops == 0
            feed._enqueue("rec3")
            assert feed.metrics.backpressure_drops == 1
            feed._enqueue("rec4")
            assert feed.metrics.backpressure_drops == 2

        asyncio.run(run())


class TestReconnectMetric:
    def test_on_reconnect_increments_counter(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"])
            assert feed.metrics.reconnects_total == 0
            feed._on_reconnect()
            assert feed.metrics.reconnects_total == 1
            feed._on_reconnect("any", "args", kw="ignored")
            assert feed.metrics.reconnects_total == 2

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Callback → loop bridge
# ---------------------------------------------------------------------------


class TestOnRecordBridge:
    """``_on_record`` schedules ``_enqueue`` on the captured loop."""

    def test_on_record_enqueues_via_call_soon_threadsafe(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"], queue_size=4)
            # Capture the loop the way live-mode __anext__ would
            feed._loop = asyncio.get_running_loop()
            feed._on_record("rec-via-callback")
            # Allow the call_soon_threadsafe scheduled task to run
            await asyncio.sleep(0)
            # _enqueue should have placed the record on the queue
            queued = feed._queue.get_nowait()
            assert queued == "rec-via-callback"
            assert feed.metrics.backpressure_drops == 0

        asyncio.run(run())

    def test_on_record_without_loop_is_noop(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"])
            # Loop not captured (__anext__ never called); callback should drop
            assert feed._loop is None
            feed._on_record("rec-early")
            # Queue should still be empty
            assert feed._queue.qsize() == 0

        asyncio.run(run())

    def test_on_record_after_close_is_noop(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"])
            feed._loop = asyncio.get_running_loop()
            feed._closed = True
            feed._on_record("rec-after-close")
            await asyncio.sleep(0)
            assert feed._queue.qsize() == 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# SDK lifecycle verification
# ---------------------------------------------------------------------------


class TestLiveLifecycle:
    def test_subscribes_with_configured_params(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY", "AAPL"])
            assert len(stub_live.instances) == 1
            stub = stub_live.instances[0]
            assert stub.kwargs["key"] == "db-fake"
            assert stub.kwargs["reconnect_policy"] == "reconnect"
            assert stub.kwargs["slow_reader_behavior"] == "skip"
            assert len(stub.subscriptions) == 1
            assert stub.subscriptions[0] == {
                "dataset": "EQUS.MINI",
                # TBBO is the production default schema (_DEFAULT_SCHEMA="tbbo");
                # the TBBO train/serve-skew commit switched the live feed to
                # trades+BBO but left this assertion on the old "trades" value.
                # Stale-assertion fix (pre-existing, surfaced by the s39 AC8 gate).
                "schema": "tbbo",
                "symbols": ["SPY", "AAPL"],
            }
            assert stub.callback is not None
            assert stub.reconnect_callback is not None

        asyncio.run(run())

    def test_aclose_stops_the_client(self, stub_live: type[_StubLive]) -> None:
        async def run() -> None:
            cfg = DatabentoConfig(api_key="db-fake")
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"])
            stub = stub_live.instances[-1]
            await feed.aclose()
            assert stub.stopped is True
            assert feed._closed is True
            # Second close is idempotent
            await feed.aclose()
            assert stub.stopped is True

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Hermetic replay test (guarded by fixture presence per amended D5)
# ---------------------------------------------------------------------------


class TestReplay:
    @pytest.mark.skipif(
        not _FIXTURE_PATH.exists(),
        reason=(
            f"Databento DBN fixture not present at {_FIXTURE_PATH}. "
            "Operator must run quantengine/scripts/capture_dbn_fixture.py "
            "pre-seal (amended D5)."
        ),
    )
    def test_replay_yields_trade_events_from_fixture(self) -> None:
        async def run() -> None:
            feed = DatabentoTradeFeed.from_dbn_file(_FIXTURE_PATH)
            events: list[TradeEventLike] = []
            async for ev in feed:
                events.append(ev)
                if len(events) >= 10:
                    break
            assert len(events) >= 1
            for ev in events:
                assert isinstance(ev.aggressor_side, int)
                assert ev.aggressor_side in {-1, 0, 1}
                assert ev.size >= 0
                assert ev.price > 0

        asyncio.run(run())

    @pytest.mark.skipif(
        not _FIXTURE_PATH.exists(),
        reason="DBN fixture not present; see TestReplay docstring.",
    )
    def test_replay_full_stream_exits_cleanly(self) -> None:
        async def run() -> None:
            feed = DatabentoTradeFeed.from_dbn_file(_FIXTURE_PATH)
            count = 0
            async for _ev in feed:
                count += 1
                if count > 100_000:  # safety
                    break
            assert count > 0

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Live smoke test (opt-in via DATABENTO_LIVE; AC9)
# ---------------------------------------------------------------------------


class TestLiveSmoke:
    @pytest.mark.skipif(
        os.environ.get("DATABENTO_LIVE") != "1",
        reason="DATABENTO_LIVE=1 not set; skipping network test.",
    )
    def test_live_yields_at_least_one_trade(self) -> None:
        async def run() -> None:
            cfg = DatabentoConfig.from_env()
            feed = DatabentoTradeFeed(cfg, "EQUS.MINI", ["SPY"])
            try:
                event = await asyncio.wait_for(anext(aiter(feed)), timeout=30.0)
                assert hasattr(event, "ts_event")
                assert hasattr(event, "instrument_id")
                # Smoke-test floor: no backpressure drops during the brief window
                assert feed.metrics.backpressure_drops == 0
            finally:
                await feed.aclose()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Smoke: FeedMetrics dataclass is constructible
# ---------------------------------------------------------------------------


def test_feed_metrics_defaults() -> None:
    m = FeedMetrics()
    assert m.backpressure_drops == 0
    assert m.reconnects_total == 0
    assert m.last_disconnect_duration_s == 0.0
