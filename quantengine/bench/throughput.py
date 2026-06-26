"""S36 PR4: streaming throughput benchmark + bridge microbench.

Per amended D3 (commit 5fc3589) the benchmark drives the
single-instrument SyntheticTradeFeed at consumer-drain speed (no
artificial rate-limit) and records the per-event dispatch latency
from feed emission to ``on_bar`` callback entry. The 500-instrument
multi-symbol workload from the original D3 is deferred — building a
MultiInstrumentFeed merger is post-S36 work (S36b candidate / S37).

Per amended D6 (commit 92a9a1f) ``bridge_cost_us_p99`` is reported
as a named line item — separate from the pipeline cost — so a future
adapter-code regression cannot silently absorb into the pipeline
budget. The bridge microbench is a standalone measurement of the
``loop.call_soon_threadsafe → asyncio.Queue.put_nowait → queue.get``
round-trip, the same primitive the Databento adapter uses (PR2).

D6 margin policy (interpretation at seal time, NOT enforced here):

  | p99_us in       | S36b status                                    |
  |-----------------|------------------------------------------------|
  | > 30            | precondition for live cutover                  |
  | [15, 30]        | contingency; S37 carries watchpoint metric     |
  | < 15            | removed from backlog                           |

D6 also marks `bridge_cost_us_p99` independently regressable; the
seal report classifies the pipeline against the margin table and
records the bridge cost as a separate line.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final, cast
from uuid import UUID

import numpy as np

from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.state import PortfolioState, Position
from quantengine.runtime.streaming import (
    EngineConfig,
    EventClock,
    StreamingEngine,
)
from quantengine.runtime.streaming._demo import SyntheticTradeFeed
from quantengine.runtime.streaming.protocols import (
    BarLike,
    CUSUMEvent,
    StreamContext,
    SyncBrokerFacade,
    TradeEventLike,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS_PER_US: Final[int] = 1_000
_DEFAULT_N_EVENTS: Final[int] = 300_000
_DEFAULT_BRIDGE_CALLS: Final[int] = 10_000
_DEFAULT_TARGET_QPS: Final[float] = 5000.0  # per amended D3 input rate
_BENCH_INSTRUMENT_ID: Final[int] = 1
_BENCH_TICKER: Final[str] = "BENCH"
_BENCH_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Pipeline stubs (mirror the e2e test pattern; no quantcore imports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BenchBar:
    """Minimal BarLike for the benchmark pipeline."""

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


class _EveryTickBarBuilder:
    """Fires a bar on every event — gives one latency sample per tick.

    The plan D3 talks about per-tick cost; with a multi-tick bar
    builder we'd only sample at bar boundaries, which is a sparse
    fraction of the pipeline-cost surface. Every-tick bar emission
    is the right shape for regression-detecting the per-event cost
    of the engine + dispatch path.
    """

    def __init__(self) -> None:
        self._n = 0

    def on_event(self, event: TradeEventLike) -> BarLike | None:
        self._n += 1
        bar = _BenchBar(
            ts_event=event.ts_event,
            instrument_id=event.instrument_id,
            sequence=self._n,
            ts_open=event.ts_event,
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
        # Frozen-dataclass vs implicit-writable Protocol variance gap — mirrors
        # S35 _demo.py + S36 PR2 databento_feed.py cast pattern.
        return cast(BarLike, cast(object, bar))

    def flush(self) -> BarLike | None:
        return None


class _NoCUSUM:
    def on_event(self, event: BarLike) -> int | None:
        del event
        return None

    def reset(self) -> None:
        pass


class _NoVol:
    def on_event(self, event: BarLike) -> float | None:
        del event
        return None

    def reset(self) -> None:
        pass


class _NoopBroker:
    """SyncBrokerFacade stub — no orders submitted in the benchmark."""

    def submit_order(self, order: Order, timeout: float | None = None) -> list[Fill]:
        del order, timeout
        return []

    def cancel_order(self, order_id: UUID, timeout: float | None = None) -> bool:
        del order_id, timeout
        return False

    def get_position(self, ticker: str, timeout: float | None = None) -> Position | None:
        del ticker, timeout
        return None

    def get_account_state(self, timeout: float | None = None) -> PortfolioState:
        del timeout
        return PortfolioState(cash=0.0, positions={})


# ---------------------------------------------------------------------------
# Benchmark feed + strategy
# ---------------------------------------------------------------------------


class _BenchmarkFeed:
    """Wraps SyntheticTradeFeed; records emission perf_counter_ns by sequence.

    Rate-limited to ``target_qps`` (default 5000 per amended D3): the
    feed sleeps between yields to keep the input rate at the target,
    so queue-wait does not dominate the per-event latency measurement.
    Without rate-limiting the feed outruns the consumer and the
    measured p99 reflects queue depth, not pipeline cost — defeating
    the point of the benchmark.

    The emission timestamp array is pre-allocated; no list append in
    the hot path. ``__anext__`` writes ``perf_counter_ns()`` to
    ``emission_ns[sequence]`` before returning the event.
    """

    def __init__(
        self,
        n_events: int,
        emission_ns: np.ndarray,
        target_qps: float = _DEFAULT_TARGET_QPS,
    ) -> None:
        assert emission_ns.shape == (n_events,)
        assert emission_ns.dtype == np.int64
        self._feed = SyntheticTradeFeed(
            seed=_BENCH_SEED,
            instrument_id=_BENCH_INSTRUMENT_ID,
            n_events=n_events,
        )
        self._emission_ns = emission_ns
        # Inter-event interval in nanoseconds. 0 disables rate-limiting.
        self._interval_ns: int = int(1e9 / target_qps) if target_qps > 0 else 0
        self._next_emit_ns: int = 0  # set on first call

    def __aiter__(self) -> _BenchmarkFeed:
        return self

    async def __anext__(self) -> TradeEventLike:
        # Rate-limit to target_qps. perf_counter_ns is monotonic;
        # asyncio.sleep yields the loop so the consumer can drain.
        if self._interval_ns > 0:
            now = time.perf_counter_ns()
            if self._next_emit_ns == 0:
                self._next_emit_ns = now
            wait_ns = self._next_emit_ns - now
            if wait_ns > 0:
                await asyncio.sleep(wait_ns / 1e9)
            self._next_emit_ns += self._interval_ns

        event = await self._feed.__anext__()
        # Sequence is monotonic from 0; emission_ns is pre-allocated to size.
        self._emission_ns[event.sequence] = time.perf_counter_ns()
        return event


class _BenchmarkStrategy:
    """Records latency at on_bar entry into a pre-allocated array.

    No list append, no dict mutation in the hot path. Latency =
    callback_entry_ns - emission_ns[bar.sequence]. ``_EveryTickBarBuilder``
    pegs bar.sequence to the bar count, so we need the triggering
    event's sequence: that arrives as ``bar.sequence`` (same index since
    builder fires every tick — see _BenchBar.sequence = self._n).
    """

    __slots__ = ("_latencies_ns", "_emission_ns", "_n_recorded", "_queue_depths", "_engine_ref")

    def __init__(
        self,
        latencies_ns: np.ndarray,
        emission_ns: np.ndarray,
        queue_depths: np.ndarray,
    ) -> None:
        self._latencies_ns = latencies_ns
        self._emission_ns = emission_ns
        self._n_recorded = 0
        self._queue_depths = queue_depths
        # Set by the harness before run starts so we can peek queue depth.
        self._engine_ref: StreamingEngine | None = None

    def attach_engine(self, engine: StreamingEngine) -> None:
        self._engine_ref = engine

    def on_bar(
        self,
        ts: int,
        bar: BarLike,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        del ts, ctx, broker
        now_ns = time.perf_counter_ns()
        # Map bar.sequence (1-based bar count == event index + 1) back to
        # emission index. The every-tick builder increments sequence
        # synchronously with each event, so emission_ns[bar.sequence - 1]
        # is the triggering event's emission timestamp.
        idx = bar.sequence - 1
        if 0 <= idx < self._emission_ns.size and self._n_recorded < self._latencies_ns.size:
            self._latencies_ns[self._n_recorded] = now_ns - self._emission_ns[idx]
            if self._engine_ref is not None:
                # Engine queue size is internal but readable via private attr;
                # this is benchmark-only instrumentation.
                self._queue_depths[self._n_recorded] = self._engine_ref._queue.qsize()
            self._n_recorded += 1

    def on_cusum(
        self,
        ts: int,
        event: CUSUMEvent,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        del ts, event, ctx, broker

    def on_vol(
        self,
        ts: int,
        sigma: float,
        ctx: StreamContext,
        broker: SyncBrokerFacade,
    ) -> None:
        del ts, sigma, ctx, broker

    @property
    def n_recorded(self) -> int:
        return self._n_recorded


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    """Reported metrics per amended D3 + D6.

    AC6 names four required keys: ``p99_us``, ``sustained_qps``,
    ``queue_depth_p99``, ``bridge_cost_us_p99``. The remaining
    fields support the seal report's hot-path classification.
    """

    n_samples: int
    wall_time_s: float
    p50_us: float
    p95_us: float
    p99_us: float
    p999_us: float
    max_us: float
    sustained_qps: float
    queue_depth_p50: int
    queue_depth_p99: int
    queue_depth_max: int
    backpressure_drops_total: int
    gc_pause_count: int
    bridge_cost_us_p99: float
    bridge_cost_us_p50: float
    bridge_cost_us_max: float
    workload: str = field(default="single_instrument_full_drain")  # per amended D3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pipeline benchmark
# ---------------------------------------------------------------------------


async def _run_pipeline_async(
    n_events: int,
    queue_maxsize: int,
    target_qps: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Drive the engine for n_events; return (latencies_ns, queue_depths, drops, gc_count, wall_s)."""
    emission_ns = np.empty(n_events, dtype=np.int64)
    latencies_ns = np.empty(n_events, dtype=np.int64)
    queue_depths = np.empty(n_events, dtype=np.int64)

    feed = _BenchmarkFeed(n_events=n_events, emission_ns=emission_ns, target_qps=target_qps)
    builder = _EveryTickBarBuilder()
    cusum = _NoCUSUM()
    vol = _NoVol()
    strategy = _BenchmarkStrategy(latencies_ns, emission_ns, queue_depths)
    broker = _NoopBroker()
    clock = EventClock()
    config = EngineConfig(
        instrument_id=_BENCH_INSTRUMENT_ID,
        ticker=_BENCH_TICKER,
        queue_maxsize=queue_maxsize,
    )
    engine = StreamingEngine(feed, builder, cusum, vol, strategy, broker, clock, config)
    strategy.attach_engine(engine)

    gc_pauses_before = _gc_pause_count()
    t0 = time.perf_counter()
    await engine.run()
    wall_s = time.perf_counter() - t0
    gc_pauses_after = _gc_pause_count()

    return (
        latencies_ns[: strategy.n_recorded],
        queue_depths[: strategy.n_recorded],
        engine.metrics.backpressure_drops,
        gc_pauses_after - gc_pauses_before,
        wall_s,
    )


def _gc_pause_count() -> int:
    """Sum of gc collection counts across all generations."""
    stats = gc.get_stats()
    return int(sum(s.get("collections", 0) for s in stats))


# ---------------------------------------------------------------------------
# Bridge microbench (amended D6)
# ---------------------------------------------------------------------------


async def _run_bridge_microbench(n_calls: int) -> np.ndarray:
    """Measure call_soon_threadsafe → put_nowait → get round-trip latency.

    Mimics the DatabentoTradeFeed pattern from PR2: a worker thread
    schedules an enqueue onto the running loop via
    ``call_soon_threadsafe``; the loop coroutine awaits ``queue.get``
    and computes the per-call latency.

    Returns an int64 array of latencies in nanoseconds.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[int] = asyncio.Queue()
    latencies_ns = np.empty(n_calls, dtype=np.int64)

    def _producer() -> None:
        for _ in range(n_calls):
            t0 = time.perf_counter_ns()
            loop.call_soon_threadsafe(queue.put_nowait, t0)

    producer_thread = threading.Thread(target=_producer, name="bridge-bench-producer")
    producer_thread.start()

    for i in range(n_calls):
        t0 = await queue.get()
        latencies_ns[i] = time.perf_counter_ns() - t0

    producer_thread.join()
    return latencies_ns


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_full_benchmark(
    n_events: int = _DEFAULT_N_EVENTS,
    bridge_calls: int = _DEFAULT_BRIDGE_CALLS,
    queue_maxsize: int = 1000,
    target_qps: float = _DEFAULT_TARGET_QPS,
) -> ThroughputResult:
    """Run the pipeline benchmark + the bridge microbench; assemble result."""
    pipeline_latencies, queue_depths, drops, gc_pauses, wall_s = asyncio.run(
        _run_pipeline_async(n_events, queue_maxsize, target_qps)
    )
    bridge_latencies = asyncio.run(_run_bridge_microbench(bridge_calls))

    p_q = np.quantile(pipeline_latencies, [0.5, 0.95, 0.99, 0.999, 1.0])
    qd_q = np.quantile(queue_depths, [0.5, 0.99, 1.0])
    b_q = np.quantile(bridge_latencies, [0.5, 0.99, 1.0])

    n_samples = int(pipeline_latencies.size)
    sustained_qps = float(n_samples / wall_s) if wall_s > 0 else 0.0

    return ThroughputResult(
        n_samples=n_samples,
        wall_time_s=float(wall_s),
        p50_us=float(p_q[0]) / _NS_PER_US,
        p95_us=float(p_q[1]) / _NS_PER_US,
        p99_us=float(p_q[2]) / _NS_PER_US,
        p999_us=float(p_q[3]) / _NS_PER_US,
        max_us=float(p_q[4]) / _NS_PER_US,
        sustained_qps=sustained_qps,
        queue_depth_p50=int(qd_q[0]),
        queue_depth_p99=int(qd_q[1]),
        queue_depth_max=int(qd_q[2]),
        backpressure_drops_total=int(drops),
        gc_pause_count=int(gc_pauses),
        bridge_cost_us_p99=float(b_q[1]) / _NS_PER_US,
        bridge_cost_us_p50=float(b_q[0]) / _NS_PER_US,
        bridge_cost_us_max=float(b_q[2]) / _NS_PER_US,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _output_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S36 PR4 throughput benchmark (per amended D3 + D6)."
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=_DEFAULT_N_EVENTS,
        help=f"Pipeline event count (default: {_DEFAULT_N_EVENTS}).",
    )
    parser.add_argument(
        "--bridge-calls",
        type=int,
        default=_DEFAULT_BRIDGE_CALLS,
        help=f"Bridge microbench call count (default: {_DEFAULT_BRIDGE_CALLS}).",
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=1000,
        help="Engine queue maxsize (default: 1000, matches EngineConfig).",
    )
    parser.add_argument(
        "--target-qps",
        type=float,
        default=_DEFAULT_TARGET_QPS,
        help=f"Target input rate (default: {_DEFAULT_TARGET_QPS} per amended D3).",
    )
    args = parser.parse_args(argv)

    result = run_full_benchmark(
        n_events=args.n_events,
        bridge_calls=args.bridge_calls,
        queue_maxsize=args.queue_maxsize,
        target_qps=args.target_qps,
    )

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"throughput-{timestamp}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2))

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nResult written to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
