"""Download a historical trades sample from Databento for replay testing.

Uses the Historical API (works anytime, no market hours needed).
Requires DATABENTO_API_KEY in the environment.

Usage:

    DATABENTO_API_KEY=db-... uv run python quantengine/scripts/download_historical_sample.py

Optional flags::

    --symbols AAPL,MSFT,NVDA   (comma-separated, default: AAPL)
    --date 2026-05-23           (trading day, default: last Friday)
    --dataset EQUS.MINI         (default: EQUS.MINI)
    --output path/to/file.dbn   (default: quantengine/data/replay/sample.dbn)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import databento as db


def _last_weekday(ref: date | None = None) -> date:
    d = ref or date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "replay" / "sample.dbn"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical Databento trades for replay.")
    parser.add_argument("--symbols", default="AAPL", help="comma-separated symbols (default: AAPL)")
    parser.add_argument(
        "--date", default=None, help="trading date YYYY-MM-DD (default: last weekday)"
    )
    parser.add_argument("--dataset", default="EQUS.MINI", help="Databento dataset")
    parser.add_argument("--schema", default="tbbo", help="Databento schema (default: tbbo)")
    parser.add_argument(
        "--output", default=None, help=f"output .dbn path (default: {_DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    api_key = os.environ.get("DATABENTO_API_KEY", "")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY env var is not set.", file=sys.stderr)
        return 1

    symbols = [s.strip() for s in args.symbols.split(",")]
    trade_date = args.date or str(_last_weekday())
    output = Path(args.output) if args.output else _DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading {args.schema}: {symbols} on {trade_date} (dataset={args.dataset}) → {output}",
        file=sys.stderr,
    )

    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=symbols,
        schema=args.schema,
        start=trade_date,
        end=f"{trade_date}T23:59",
    )
    data.to_file(str(output))

    size_kb = output.stat().st_size / 1024
    n_records = len(data.to_df()) if size_kb > 0 else 0
    print(f"Done: {size_kb:.1f} KB, ~{n_records} records → {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
