"""CLI entry point for the streaming runtime (S35 D13).

Subcommands:

- ``start``  — Launch the engine. Two feed modes: ``--feed synthetic``
  (default, self-contained demo) and ``--feed databento`` (live market
  data via Databento TCP stream, requires ``DATABENTO_API_KEY``).
- ``replay`` — Forensic replay of a ``SafeBroker`` JSONL journal.
  S35 ships a minimal record-printer; full replay-with-state-rebuild
  lives in S37 (``RecoveryCoordinator``).
- ``status`` — Print engine status from a running pid file. S35 ships
  a placeholder; the full hot-attach lives in S37.

D13: notebook-driven engine invocation is explicitly not supported
(see the rejection of the loop-monkeypatching library in
``wrappers.py``). Scripts for execution, notebooks for research.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from quantengine.risk.gate import RiskGate
from typing import Any, cast

from quantengine.runtime.streaming import (
    BarBuilderProtocol,
    DemoBroker,
    EngineConfig,
    EventClock,
    OnlineCUSUMFilterProtocol,
    OnlineEWMAVolatilityProtocol,
    SafeBroker,
    ShutdownMode,
    StreamingEngine,
    SyntheticTradeFeed,
    ThreadSafeBrokerWrapper,
)

# RecoveryCoordinator / StreamingReconciler are not re-exported from the
# streaming package __init__ (S35 substrate is locked); import directly.
from quantengine.runtime.streaming.reconciler import StreamingReconciler
from quantengine.runtime.streaming.recovery import RecoveryCoordinator


# ---------------------------------------------------------------------------
# Demo strategy — does nothing but proves the wiring
# ---------------------------------------------------------------------------
class _DemoStrategy:
    """Records callback counts; no orders. Used by ``start`` subcommand
    so the CLI can prove the engine runs end-to-end without external
    services."""

    def __init__(self) -> None:
        self.bars = 0
        self.cusums = 0
        self.vols = 0

    def on_bar(self, ts, bar, ctx, broker) -> None:
        self.bars += 1

    def on_cusum(self, ts, event, ctx, broker) -> None:
        self.cusums += 1

    def on_vol(self, ts, sigma, ctx, broker) -> None:
        self.vols += 1


# ---------------------------------------------------------------------------
# Minimal pipeline primitives for the CLI demo path
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _DemoBar:
    """Module-level bar dataclass for the CLI demo path; satisfies BarLike."""

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


class _SimpleBarBuilder:
    """Emits one bar every 10 events."""

    def __init__(self, every: int = 10) -> None:
        self._every = every
        self._n = 0
        self._last_price = 100.0

    def on_event(self, event):
        self._n += 1
        self._last_price = event.price
        if self._n % self._every == 0:
            return _DemoBar(
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

    def flush(self):
        return None


class _NoopCUSUM:
    def on_event(self, event):
        return None

    def reset(self):
        pass


class _NoopVol:
    def on_event(self, event):
        return None

    def reset(self):
        pass


def _make_quantcore_bar_builder(bar_type: str, threshold: float) -> object | None:
    """Construct the quantcore streaming bar builder for ``bar_type`` (or None).

    Returned UNWRAPPED: the streaming feed's trade events are fed to it DIRECTLY.
    quantcore's ``BarBuilder.on_event`` duck-types the structural trade event and
    reads ``aggressor_side`` (∈ {+1,-1,0}; 0 → tick-rule fallback) + BBO via
    getattr, so the direct path preserves both the no-aggressor 0 state — the
    dominant signed-flow driver on real equity TBBO (≈91% of TXN trades are 'N')
    — and the top-of-book the spread features need.

    s45: replaces the former ``_QuantcoreBarAdapter``, which round-tripped the
    event through quantcore ``TradeEvent`` and so collapsed 0 → Side.BID(+1) (the
    ``Side`` enum has no zero state) and dropped BBO — a train/serve skew vs the
    research batch pipeline (``top_of_book._resolve_side_column``), which
    tick-rule-resolves unknown side and keeps BBO. ``int(threshold)`` for tick
    bars matches ``TickBarBuilder``'s integer trade-count contract.
    """
    if bar_type == "dollar":
        from quantcore.bars.streaming import DollarBarBuilder

        return DollarBarBuilder(threshold=threshold)
    if bar_type == "volume":
        from quantcore.bars.streaming import VolumeBarBuilder

        return VolumeBarBuilder(threshold=threshold)
    if bar_type == "tick":
        from quantcore.bars.streaming import TickBarBuilder

        return TickBarBuilder(threshold=int(threshold))
    return None


class _LiveLogStrategy:
    """Prints live trade activity to the terminal. Used with ``--feed databento``."""

    def __init__(self) -> None:
        self.bars = 0
        self.cusums = 0
        self.vols = 0
        self._start = time.monotonic()
        self._prices: dict[int, float] = {}
        self._event_counts: dict[int, int] = defaultdict(int)
        self._last_sigma: float | None = None

    def on_bar(self, ts, bar, ctx, broker) -> None:
        self.bars += 1
        iid = bar.instrument_id
        self._prices[iid] = bar.close
        self._event_counts[iid] += 1
        elapsed = time.monotonic() - self._start
        sigma_str = f"{self._last_sigma:.6f}" if self._last_sigma is not None else "    n/a"
        print(
            f"  bar #{self.bars:>5d} | iid={iid:<8d} | "
            f"close={bar.close:>10.4f} | vol={bar.volume:>8.0f} | "
            f"ticks={bar.tick_count:>4d} | σ={sigma_str} | t={elapsed:>7.1f}s"
        )

    def on_cusum(self, ts, event, ctx, broker) -> None:
        self.cusums += 1
        elapsed = time.monotonic() - self._start
        bar_close = getattr(event, "bar_close", "?")
        print(
            f"  *** CUSUM #{self.cusums:>3d} | bar_close={bar_close:>10} | t={elapsed:>7.1f}s ***"
        )

    def on_vol(self, ts, sigma, ctx, broker) -> None:
        self.vols += 1
        self._last_sigma = sigma

    def summary(self) -> str:
        elapsed = time.monotonic() - self._start
        n_instruments = len(self._prices)
        return (
            f"bars={self.bars} cusums={self.cusums} vols={self.vols} "
            f"instruments={n_instruments} elapsed={elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> int:
    """Launch the engine on a synthetic or live Databento feed."""
    is_live = args.feed == "databento"
    is_replay = args.feed == "replay"

    if is_live:
        from quantengine.runtime.streaming.databento_config import DatabentoConfig
        from quantengine.runtime.streaming.databento_feed import DatabentoTradeFeed

        db_config = DatabentoConfig.from_env()
        symbols = [s.strip() for s in args.symbols.split(",")]
        feed = DatabentoTradeFeed(
            db_config,
            dataset=args.dataset,
            symbols=symbols,
            schema=args.schema,
        )
        strat: _DemoStrategy | _LiveLogStrategy = _LiveLogStrategy()
        print(
            f"[live] Databento feed: dataset={args.dataset} "
            f"symbols={symbols} duration={args.duration}s"
        )
        print("[live] Waiting for market data (Ctrl+C to stop)...")
    elif is_replay:
        from quantengine.runtime.streaming.databento_feed import DatabentoTradeFeed

        replay_path = Path(args.replay_file)
        if not replay_path.exists():
            print(f"[replay] file not found: {replay_path}", file=sys.stderr)
            return 1
        feed = DatabentoTradeFeed.from_dbn_file(replay_path)
        strat = _LiveLogStrategy()
        size_kb = replay_path.stat().st_size / 1024
        print(f"[replay] Replaying {replay_path} ({size_kb:.1f} KB)")
    else:
        feed = SyntheticTradeFeed(seed=args.seed, instrument_id=1, n_events=args.n_events)
        strat = _DemoStrategy()
        print(f"[demo] starting streaming runtime (n_events={args.n_events}, seed={args.seed})")

    use_quantcore = args.bar_type in ("dollar", "volume", "tick")

    # s45: feed events DIRECTLY to the quantcore builder — no TradeEvent
    # round-trip (see _make_quantcore_bar_builder). The engine consumes it via
    # BarBuilderProtocol (cast below); the quantcore builders satisfy it
    # structurally and read aggressor_side + BBO off the feed event with getattr.
    quantcore_builder = _make_quantcore_bar_builder(args.bar_type, args.bar_threshold)
    if quantcore_builder is not None:
        builder = quantcore_builder
        print(
            f"[bars] {type(builder).__name__} threshold={args.bar_threshold:,.0f} ({args.bar_type})"
        )
    else:
        builder = _SimpleBarBuilder(every=10)

    if use_quantcore and args.cusum_threshold > 0:
        from quantcore.bars.cusum import OnlineCUSUMFilter

        cusum = OnlineCUSUMFilter(threshold=args.cusum_threshold)
        print(f"[cusum] OnlineCUSUMFilter threshold={args.cusum_threshold} (log-return)")
    else:
        cusum = _NoopCUSUM()

    if use_quantcore and args.vol_span > 0:
        from quantcore.bars.volatility import OnlineEWMAVolatility

        vol = OnlineEWMAVolatility(span=args.vol_span)
        print(f"[vol] OnlineEWMAVolatility span={args.vol_span} bars")
    else:
        vol = _NoopVol()
    demo = DemoBroker(price_lookup=lambda t: 100.0)
    gate = RiskGate.default_us_equities()
    journal = Path(args.journal) if args.journal else Path("safe_broker.jsonl")

    async def _run() -> _DemoStrategy | _LiveLogStrategy:
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
        queue_size = 50_000 if is_replay else 1000
        config = EngineConfig(
            instrument_id=0,
            ticker="MULTI" if is_live else "AAPL",
            queue_maxsize=queue_size,
            # REC-004 (S39 D4): on the live feed, a watchdog alert (feed
            # silence / health-probe failure) trips the SafeBroker kill-switch
            # so no further order is submitted until restart. Demo/replay keep
            # the log-only default (None).
            on_watchdog_alert=(
                (lambda reason: sb.trip_kill_switch(reason=f"watchdog_{reason}"))
                if is_live
                else None
            ),
        )
        engine = StreamingEngine(
            feed,
            cast(BarBuilderProtocol, cast(object, builder)),
            cast(OnlineCUSUMFilterProtocol, cast(object, cusum)),
            cast(OnlineEWMAVolatilityProtocol, cast(object, vol)),
            strat,
            wrapper,
            clock,
            config,
        )

        # REC-003/004 (S39) — live-readiness wiring. Demo/replay paths are
        # unchanged; this block only runs for the live feed.
        reconciler: StreamingReconciler | None = None
        if is_live:
            # Restart recovery: replay the journal, refuse to start on a
            # divergence the operator must resolve.
            recovery = RecoveryCoordinator(journal_path=journal, broker=demo)
            rec_result = await recovery.replay()
            if rec_result.records_replayed:
                print(f"[live] recovery: replayed {rec_result.records_replayed} journal record(s)")
            if rec_result.halted:
                print(
                    f"[live] RECOVERY HALT — refusing to start: {rec_result.halt_reason}",
                    file=sys.stderr,
                )
                return strat

            # Streaming reconciliation: compare engine state vs broker truth on
            # every durable fill; halt (CANCEL) on divergence beyond tolerance.
            async def _reconcile_halt(reason: str) -> None:
                print(f"[live] RECONCILE HALT: {reason}", file=sys.stderr)
                await engine.shutdown(ShutdownMode.CANCEL, timeout_s=10.0)

            reconciler = StreamingReconciler(demo, lambda: demo.state, _reconcile_halt)
            sb.set_fill_subscriber(reconciler.on_fill)

            # Signal handlers: SIGTERM from a process manager now DRAINS instead
            # of being silently ignored, and trips the kill-switch so no new
            # order is submitted during the drain.
            def _handle_signal(signame: str) -> None:
                print(f"\n[live] {signame} received — kill-switch + drain...", file=sys.stderr)
                sb.trip_kill_switch(reason=signame)
                asyncio.create_task(
                    engine.shutdown(ShutdownMode.DRAIN, timeout_s=10.0), name="signal-drain"
                )

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _handle_signal, sig.name)
                except (NotImplementedError, RuntimeError):
                    # Unavailable on some platforms (e.g. Windows); the
                    # KeyboardInterrupt path below remains the fallback.
                    pass

        if is_live and args.duration:

            async def _auto_stop() -> None:
                await asyncio.sleep(args.duration)
                print(f"\n[live] Duration reached ({args.duration}s), shutting down...")
                await engine.shutdown(ShutdownMode.DRAIN, timeout_s=10.0)

            asyncio.create_task(_auto_stop(), name="auto-stop")

        try:
            await engine.run()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("\n[live] Ctrl+C received, shutting down...")
            await engine.shutdown(ShutdownMode.DRAIN, timeout_s=10.0)
        finally:
            if reconciler is not None:
                await reconciler.aclose()
            if hasattr(feed, "aclose"):
                await cast(Any, feed).aclose()
            await sb.aclose()
        return strat

    s = asyncio.run(_run())
    if isinstance(s, _LiveLogStrategy):
        print(f"[live] done: {s.summary()}")
    else:
        print(f"[demo] done: bars={s.bars} cusums={s.cusums} vols={s.vols}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Print SafeBroker journal records. Full state-rebuild is S37."""
    p = Path(args.journal)
    if not p.exists():
        print(f"[s35] journal not found: {p}", file=sys.stderr)
        return 1
    n = 0
    for line in p.read_text().splitlines():
        if not line:
            continue
        record = json.loads(line)
        print(json.dumps(record))
        n += 1
    print(f"[s35] {n} records replayed (S37 will add state-rebuild)", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Status placeholder — S37 will add live engine introspection."""
    p = Path(args.pidfile) if args.pidfile else None
    if p is None or not p.exists():
        print("[s35] no pid file — engine not detected", file=sys.stderr)
        return 1
    print(f"[s35] pid file present at {p}: {p.read_text().strip()}")
    return 0


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``argparse`` parser.

    Subcommands: ``start``, ``replay``, ``status``."""
    parser = argparse.ArgumentParser(
        prog="quantengine.runtime.streaming",
        description="Streaming runtime CLI. Feeds: synthetic (demo) or databento (live).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_start = sub.add_parser("start", help="run the engine")
    s_start.add_argument(
        "--feed",
        choices=["synthetic", "databento", "replay"],
        default="synthetic",
        help="data feed source (default: synthetic)",
    )
    s_start.add_argument(
        "--replay-file", default=None, help="path to .dbn file (used with --feed replay)"
    )
    s_start.add_argument(
        "--dataset", default="EQUS.MINI", help="Databento dataset (default: EQUS.MINI)"
    )
    s_start.add_argument(
        "--schema", default="tbbo", help="Databento schema: trades, tbbo, mbp-1 (default: tbbo)"
    )
    s_start.add_argument(
        "--symbols", default="AAPL", help="comma-separated ticker symbols (default: AAPL)"
    )
    s_start.add_argument(
        "--duration", type=int, default=60, help="seconds to run live feed (default: 60)"
    )
    s_start.add_argument(
        "--bar-type",
        choices=["simple", "dollar", "volume", "tick"],
        default="simple",
        help="bar builder type (default: simple — every 10 events)",
    )
    s_start.add_argument(
        "--bar-threshold",
        type=float,
        default=500_000,
        help="threshold for dollar/volume/tick bars (default: 500000)",
    )
    s_start.add_argument(
        "--cusum-threshold",
        type=float,
        default=0.02,
        help="CUSUM threshold in cumulative log-return (default: 0.02 = ~2%% move)",
    )
    s_start.add_argument(
        "--vol-span",
        type=int,
        default=20,
        help="EWMA volatility span in bars (default: 20)",
    )
    s_start.add_argument("--config", default=None, help="config file path (future)")
    s_start.add_argument("--n-events", type=int, default=100, help="synthetic event count")
    s_start.add_argument("--seed", type=int, default=42, help="synthetic feed seed")
    s_start.add_argument("--journal", default=None, help="SafeBroker journal output path")
    s_start.set_defaults(func=cmd_start)

    s_replay = sub.add_parser("replay", help="print SafeBroker journal records")
    s_replay.add_argument("journal", help="path to a SafeBroker JSONL journal")
    s_replay.set_defaults(func=cmd_replay)

    s_status = sub.add_parser("status", help="print engine status from a pidfile")
    s_status.add_argument("pidfile", nargs="?", default=None, help="path to engine pidfile")
    s_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args))


__all__ = ["build_parser", "cmd_replay", "cmd_start", "cmd_status", "main"]
