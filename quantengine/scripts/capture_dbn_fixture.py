"""Operator-runnable capture of real Databento Live traffic → DBN fixture.

Required artifact of S36 (per amended D5). The captured ``.dbn`` file
is NOT itself a required artifact at PR2-merge time — see the manual
review item in the s36 sprint plan; the operator runs this script
once before sprint seal.

Coverage intent (documented here so a test reader can verify without
re-running the capture):

1. Window: 5-10 minute capture during US RTH (default 600 seconds).
   Long enough to include at least one sequence gap — Databento
   sequence numbers are monotonic per instrument; a multi-minute
   window on a liquid name reliably sees one. The hermetic replay
   test exercises sequence-gap handling against the captured bytes.

2. Symbols: three liquid US equities (SPY + AAPL + NVDA) to exercise
   the adapter's per-instrument dispatch path. A single-symbol
   capture would not catch per-instrument-state bugs.

3. Schema: ``trades`` specifically. The Databento Live API supports
   many schemas (bbo, cbbo, mbp, ...); S36 PR2's DatabentoTradeFeed
   targets the trades schema only. Mixing schemas would expand the
   adapter's parse surface beyond what S36 ships.

4. Output: ``quantengine/tests/fixtures/databento_dbn/sample_trades.dbn``.
   Future captures REPLACE this file, never append — the fixture is
   the single source of truth for replay tests.

Usage:

    DATABENTO_API_KEY=db-... uv run python quantengine/scripts/capture_dbn_fixture.py

Optional flags::

    --duration 300       (shorter window for quick re-captures)
    --dataset XNAS.ITCH  (override the default EQUS.MINI dataset)

Not run in CI. Network + credentials required.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import databento as db

_DEFAULT_DURATION_SECONDS = 600
_SYMBOLS = ("SPY", "AAPL", "NVDA")
_SCHEMA = "trades"
_DEFAULT_DATASET = "EQUS.MINI"
_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "databento_dbn"
    / "sample_trades.dbn"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a Databento Live trades fixture for S36 PR2 replay tests."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=_DEFAULT_DURATION_SECONDS,
        help=f"Capture duration in seconds (default: {_DEFAULT_DURATION_SECONDS} = 10 min).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help=f"Databento dataset name (default: {_DEFAULT_DATASET}).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var is not set.", file=sys.stderr)
        return 1

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Capturing {args.duration}s of {_SCHEMA} on {', '.join(_SYMBOLS)} "
        f"(dataset={args.dataset}) → {_OUTPUT_PATH}",
        file=sys.stderr,
    )

    client = db.Live(key=api_key)
    client.subscribe(dataset=args.dataset, schema=_SCHEMA, symbols=list(_SYMBOLS))
    with _OUTPUT_PATH.open("wb") as f:
        client.add_stream(f)
        client.start()
        client.block_for_close(timeout=args.duration)

    size_kb = _OUTPUT_PATH.stat().st_size / 1024
    print(f"Captured {size_kb:.1f} KB → {_OUTPUT_PATH}", file=sys.stderr)
    if size_kb < 1.0:
        print(
            "WARNING: captured file is suspiciously small; re-run during RTH "
            "or with a longer --duration to ensure trades land in the window.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
