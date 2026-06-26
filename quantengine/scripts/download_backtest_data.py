"""Download 12 months of TBBO historical data for backtest research.

Pulls per-ticker DBN files from Databento's EQUS.MINI dataset.
Output goes to quantengine/data/backtest/<TICKER>.dbn — one file per ticker.

Usage:

    uv run python scripts/download_backtest_data.py

Optional flags::

    --tickers AAPL,MSFT,NVDA    (override default 10 mega-cap names)
    --months 12                  (lookback in months, default: 12)
    --schema tbbo                (default: tbbo)
    --output-dir data/backtest   (default: data/backtest)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import databento as db

_DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "JPM",
    "V",
    "UNH",
]

_DEFAULT_DATASET = "EQUS.MINI"
_DEFAULT_SCHEMA = "tbbo"


def _months_ago(months: int, ref: date | None = None) -> date:
    d = ref or date.today()
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, min(d.day, 28))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download backtest TBBO data from Databento.")
    parser.add_argument(
        "--tickers", default=",".join(_DEFAULT_TICKERS), help="comma-separated tickers"
    )
    parser.add_argument("--months", type=int, default=12, help="lookback months (default: 12)")
    parser.add_argument(
        "--schema", default=_DEFAULT_SCHEMA, help="Databento schema (default: tbbo)"
    )
    parser.add_argument("--dataset", default=_DEFAULT_DATASET, help="Databento dataset")
    parser.add_argument(
        "--output-dir", default=None, help="output directory (default: data/backtest)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var is not set.", file=sys.stderr)
        return 1

    tickers = [t.strip() for t in args.tickers.split(",")]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent.parent / "data" / "backtest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = _months_ago(args.months)
    end_date = date.today() - timedelta(days=3)

    print("Download plan:", file=sys.stderr)
    print(f"  Tickers:  {tickers}", file=sys.stderr)
    print(f"  Range:    {start_date} → {end_date} ({args.months} months)", file=sys.stderr)
    print(f"  Schema:   {args.schema}", file=sys.stderr)
    print(f"  Dataset:  {args.dataset}", file=sys.stderr)
    print(f"  Output:   {output_dir}/", file=sys.stderr)
    print(file=sys.stderr)

    client = db.Historical(key=api_key)
    total_bytes = 0
    total_start = time.monotonic()

    for i, ticker in enumerate(tickers, 1):
        out_path = output_dir / f"{ticker}.dbn"
        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(
                f"  [{i}/{len(tickers)}] {ticker}: SKIP (already exists, {size_mb:.1f} MB)",
                file=sys.stderr,
            )
            total_bytes += out_path.stat().st_size
            continue

        t0 = time.monotonic()
        print(
            f"  [{i}/{len(tickers)}] {ticker}: downloading...", end="", file=sys.stderr, flush=True
        )

        try:
            data = client.timeseries.get_range(
                dataset=args.dataset,
                symbols=[ticker],
                schema=args.schema,
                start=str(start_date),
                end=f"{end_date}T23:59",
            )
            data.to_file(str(out_path))

            size_mb = out_path.stat().st_size / (1024 * 1024)
            elapsed = time.monotonic() - t0
            total_bytes += out_path.stat().st_size
            print(f" {size_mb:.1f} MB in {elapsed:.0f}s", file=sys.stderr)

        except Exception as e:
            print(f" FAILED: {e}", file=sys.stderr)
            if out_path.exists():
                out_path.unlink()
            continue

    total_mb = total_bytes / (1024 * 1024)
    total_elapsed = time.monotonic() - total_start
    print(file=sys.stderr)
    print(
        f"Done: {total_mb:.1f} MB total across {len(tickers)} tickers in {total_elapsed:.0f}s",
        file=sys.stderr,
    )
    print(f"Files at: {output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
