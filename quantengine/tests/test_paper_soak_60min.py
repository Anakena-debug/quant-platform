"""60-minute paper soak harness (S37 PR4).

Gated by ``PAPER_SOAK_60MIN=1``. Not run in CI. The pre-seal
operator runs this once + commits the resulting reconciliation
event log at seal time per the S37 manual
review items.

The soak drives a DemoBroker for ``PAPER_SOAK_DURATION_S`` seconds
(default 3600 = 60 minutes), submitting small orders across three
tickers, feeding fills into a ``StreamingReconciler``, and
asserting that the reconciler does NOT halt during the run. A
JSONL event log is written to ``tmp_path``; the seal-time operator
copies it into the report.

Configuration via env vars (all optional except the gate):

  ``PAPER_SOAK_60MIN``         gate: must == "1" to run
  ``PAPER_SOAK_DURATION_S``    duration in seconds (default 3600)
  ``PAPER_SOAK_PERIOD_S``      inter-order period (default 5.0)
  ``PAPER_SOAK_TICKERS``       comma-sep tickers (default "AAPL,MSFT,NVDA")

For harness self-verification without the full 60-minute wait::

    PAPER_SOAK_60MIN=1 PAPER_SOAK_DURATION_S=5 PAPER_SOAK_PERIOD_S=0.2 \\
        uv run --directory quantengine pytest tests/test_paper_soak_60min.py -v

IBKR paper-account variant (post-S37 integration; not implemented here):
when ``IBKR_PAPER_SMOKE=1`` is also set, future PRs may swap the
DemoBroker for an ``AsyncIBKRBroker``. The cutover-gate runbook
(PR5) documents the full operator procedure.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from quantengine.contracts.orders import Order, OrderSide, OrderType
from quantengine.runtime.streaming._demo import DemoBroker
from quantengine.runtime.streaming.reconciler import StreamingReconciler

_DEFAULT_DURATION_S = 60 * 60  # 60 minutes
_DEFAULT_PERIOD_S = 5.0
_DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA")
_DEFAULT_REFERENCE_PRICE = 100.0


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _read_env_tickers(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return tuple(t.strip() for t in raw.split(",") if t.strip())


@pytest.mark.skipif(
    os.environ.get("PAPER_SOAK_60MIN") != "1",
    reason="PAPER_SOAK_60MIN=1 not set; skipping paper-soak harness.",
)
def test_paper_soak_60min(tmp_path: Path) -> None:
    """Drive DemoBroker for ~60min, reconcile every fill, assert no halts."""
    duration_s = _read_env_int("PAPER_SOAK_DURATION_S", _DEFAULT_DURATION_S)
    period_s = _read_env_float("PAPER_SOAK_PERIOD_S", _DEFAULT_PERIOD_S)
    tickers = _read_env_tickers("PAPER_SOAK_TICKERS", _DEFAULT_TICKERS)

    log_path = tmp_path / f"paper-soak-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    asyncio.run(_run_soak(duration_s, period_s, tickers, log_path))

    # Surface log path so the seal-time operator can locate it
    print(f"\nPaper soak event log: {log_path}")
    assert log_path.exists(), "event log was not written"


async def _run_soak(
    duration_s: int,
    period_s: float,
    tickers: tuple[str, ...],
    log_path: Path,
) -> None:
    """Internal: construct broker + reconciler; loop until duration_s expires."""
    broker = DemoBroker(
        starting_cash=1_000_000.0,
        price_lookup=lambda _t: _DEFAULT_REFERENCE_PRICE,
        commission_per_share=0.0,
    )

    halt_state: dict[str, Any] = {"halted": False, "reason": None, "halt_ts": None}

    async def halt_callback(reason: str) -> None:
        halt_state["halted"] = True
        halt_state["reason"] = reason
        halt_state["halt_ts"] = time.time()

    reconciler = StreamingReconciler(
        broker=broker,
        virtual_portfolio_provider=lambda: broker.state,
        halt_callback=halt_callback,
        tolerance=1,
        debounce_s=0.1,
    )

    try:
        with log_path.open("w", encoding="utf-8") as log:
            _write_log(
                log,
                {
                    "ts": time.time(),
                    "event": "soak_start",
                    "duration_s": duration_s,
                    "period_s": period_s,
                    "tickers": list(tickers),
                },
            )

            deadline = time.monotonic() + duration_s
            submit_count = 0
            fill_count = 0
            ticker_cycle_idx = 0
            # Alternate BUY / SELL to keep positions oscillating without
            # accumulating unbounded exposure.
            side_toggle = True

            while time.monotonic() < deadline:
                if halt_state["halted"]:
                    _write_log(
                        log,
                        {
                            "ts": time.time(),
                            "event": "halt_detected",
                            "reason": halt_state["reason"],
                        },
                    )
                    break

                ticker = tickers[ticker_cycle_idx % len(tickers)]
                ticker_cycle_idx += 1
                side = OrderSide.BUY if side_toggle else OrderSide.SELL
                side_toggle = not side_toggle

                order = Order(
                    order_id=uuid4(),
                    ticker=ticker,
                    side=side,
                    quantity=1,
                    order_type=OrderType.MARKET,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                fills = await broker.submit_order(order)
                submit_count += 1
                for fill in fills:
                    reconciler.on_fill(fill)
                    fill_count += 1

                vp_pos = broker.state.positions.get(ticker)
                vp_qty = vp_pos.quantity if vp_pos is not None else 0
                _write_log(
                    log,
                    {
                        "ts": time.time(),
                        "event": "order_submitted",
                        "ticker": ticker,
                        "side": side.value,
                        "submit_count": submit_count,
                        "fill_count": fill_count,
                        "reconciler_halted": reconciler.halted,
                        "vp_position_qty": vp_qty,
                    },
                )

                # Sleep before the next submit; clamp to avoid asyncio jitter
                # eating into long runs. period_s of 0 means tight loop —
                # only use that for harness-verification mode.
                if period_s > 0:
                    await asyncio.sleep(period_s)

            # Let any pending debounce timer fire
            await asyncio.sleep(0.5)

            _write_log(
                log,
                {
                    "ts": time.time(),
                    "event": "soak_end",
                    "submit_count": submit_count,
                    "fill_count": fill_count,
                    "reconciler_halted": reconciler.halted,
                    "halt_reason": halt_state["reason"],
                },
            )
    finally:
        await reconciler.aclose()

    # Final assertions: zero halts during steady-state
    assert not halt_state["halted"], (
        f"reconciler halted during soak: reason={halt_state['reason']!r}"
    )
    assert submit_count > 0, "no orders submitted — harness configuration error"


def _write_log(fh: Any, record: dict[str, Any]) -> None:
    """Append one JSONL record + flush so the operator can tail the file."""
    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    fh.flush()
