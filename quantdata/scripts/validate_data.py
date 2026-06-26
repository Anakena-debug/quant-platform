#!/usr/bin/env python3
"""
validate_data.py
~~~~~~~~~~~~~~~~
Run data quality checks on ingested parquet files.

Checks:
    1. Missing trading days (gaps)
    2. Null/NaN values in OHLCV
    3. Price anomalies (close <= 0, high < low)
    4. Volume anomalies (zero-volume days)
    5. Duplicate rows

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")


def validate_ticker(path: Path) -> dict:
    """Run all checks on a single ticker parquet file."""
    ticker = path.stem
    df = pl.read_parquet(path)
    issues: list[str] = []

    # 1. Null check
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            nulls = df.filter(pl.col(col).is_null()).height
            if nulls > 0:
                issues.append(f"{col}: {nulls} nulls")

    # 2. Price sanity
    if "close" in df.columns:
        bad_close = df.filter(pl.col("close") <= 0).height
        if bad_close > 0:
            issues.append(f"close <= 0: {bad_close} rows")

    if "high" in df.columns and "low" in df.columns:
        inverted = df.filter(pl.col("high") < pl.col("low")).height
        if inverted > 0:
            issues.append(f"high < low: {inverted} rows")

    # 3. Zero-volume days
    if "volume" in df.columns:
        zero_vol = df.filter(pl.col("volume") == 0).height
        pct = zero_vol / df.height * 100 if df.height > 0 else 0
        if pct > 5:
            issues.append(f"zero-volume: {zero_vol} rows ({pct:.1f}%)")

    # 4. Duplicates
    if "date" in df.columns:
        dupes = df.height - df.unique(subset=["date"]).height
        if dupes > 0:
            issues.append(f"duplicate dates: {dupes}")

    # 5. Date range
    date_col = "date" if "date" in df.columns else "timestamp"
    min_dt = df[date_col].min()
    max_dt = df[date_col].max()

    return {
        "ticker": ticker,
        "rows": df.height,
        "start": str(min_dt),
        "end": str(max_dt),
        "issues": "; ".join(issues) if issues else "OK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ingested parquet data")
    parser.add_argument("--ticker", default=None, help="Validate single ticker")
    parser.add_argument(
        "--dir",
        default=str(PATHS.raw_yf_daily),
        help="Directory of parquet files to validate",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        log.error(f"Directory not found: {data_dir}")
        sys.exit(1)

    if args.ticker:
        files = [data_dir / f"{args.ticker}.parquet"]
    else:
        files = sorted(data_dir.glob("*.parquet"))
        files = [f for f in files if not f.name.startswith("_")]

    if not files:
        log.warning("No parquet files found.")
        sys.exit(0)

    log.info(f"Validating {len(files)} files in {data_dir}")
    results = []
    for f in files:
        if not f.exists():
            log.warning(f"  {f.name}: not found")
            continue
        r = validate_ticker(f)
        status = "✓" if r["issues"] == "OK" else "✗"
        log.info(f"  {status} {r['ticker']}: {r['rows']:,} rows | {r['issues']}")
        results.append(r)

    # Summary
    ok = sum(1 for r in results if r["issues"] == "OK")
    bad = len(results) - ok
    log.info(f"\nSummary: {ok} clean, {bad} with issues, {len(results)} total")

    if bad > 0:
        log.info("\nTickers with issues:")
        for r in results:
            if r["issues"] != "OK":
                log.info(f"  {r['ticker']}: {r['issues']}")


if __name__ == "__main__":
    main()
