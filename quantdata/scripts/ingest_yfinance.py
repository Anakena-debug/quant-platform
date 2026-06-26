#!/usr/bin/env python3
"""
ingest_yfinance.py
~~~~~~~~~~~~~~~~~~
Download US equity OHLCV bars from Yahoo Finance → partitioned Parquet files.

Modes:
    daily     : Full history (period="max"), one parquet per ticker
    intraday  : Rolling window (7d for 1m, 60d for 5m, 730d for 1h)

Usage:
    # Smoke test with 10 tickers
    python scripts/ingest_yfinance.py --mode daily --scope test

    # Full S&P 500 daily bars
    python scripts/ingest_yfinance.py --mode daily --scope sp500

    # Intraday 1m bars (last 7 days) for test universe
    python scripts/ingest_yfinance.py --mode intraday --interval 1m --scope test

    # Custom ticker file
    python scripts/ingest_yfinance.py --mode daily --scope file --file tickers.txt

Author: Jules (quantdata pipeline)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import yfinance as yf

# Add parent dir so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import INGESTION, PATHS, UNIVERSE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_yfinance")


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def download_daily(tickers: list[str]) -> dict[str, pl.DataFrame]:
    """
    Download full daily history for a list of tickers.

    Returns dict of {ticker: polars.DataFrame} with columns:
        [date, open, high, low, close, volume, dividends, stock_splits, ticker]
    """
    results: dict[str, pl.DataFrame] = {}
    total = len(tickers)

    for batch_start in range(0, total, INGESTION.yf_batch_size):
        batch = tickers[batch_start : batch_start + INGESTION.yf_batch_size]
        batch_end = min(batch_start + INGESTION.yf_batch_size, total)
        log.info(
            f"Batch {batch_start // INGESTION.yf_batch_size + 1}: "
            f"tickers {batch_start + 1}–{batch_end} of {total}"
        )

        for ticker in batch:
            try:
                t = yf.Ticker(ticker)
                df_pd = t.history(period=INGESTION.yf_daily_period, auto_adjust=True)

                if df_pd.empty:
                    log.warning(f"  {ticker}: no data returned, skipping")
                    continue

                # Reset index to get Date as column
                df_pd = df_pd.reset_index()

                # Convert to Polars
                df = pl.from_pandas(df_pd)

                # Normalize column names
                df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})

                # Add ticker column
                df = df.with_columns(pl.lit(ticker).alias("ticker"))

                # Ensure date is Date type (not Datetime)
                if "date" in df.columns:
                    df = df.with_columns(pl.col("date").cast(pl.Date))

                rows = df.height
                start = df["date"].min()
                end = df["date"].max()
                log.info(f"  {ticker}: {rows:,} rows ({start} → {end})")
                results[ticker] = df

            except Exception as e:
                log.error(f"  {ticker}: FAILED — {e}")

        # Rate-limit courtesy
        if batch_end < total:
            log.info(f"  Sleeping {INGESTION.yf_sleep}s between batches...")
            time.sleep(INGESTION.yf_sleep)

    return results


def download_intraday(
    tickers: list[str], interval: str = "1m"
) -> dict[str, pl.DataFrame]:
    """
    Download intraday bars for the maximum lookback window.

    yfinance limits:
        1m  → 7 days
        5m  → 60 days
        1h  → 730 days
    """
    max_days = INGESTION.yf_intraday_intervals.get(interval)
    if max_days is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Choose from: {list(INGESTION.yf_intraday_intervals.keys())}"
        )

    results: dict[str, pl.DataFrame] = {}
    total = len(tickers)

    for batch_start in range(0, total, INGESTION.yf_batch_size):
        batch = tickers[batch_start : batch_start + INGESTION.yf_batch_size]
        batch_end = min(batch_start + INGESTION.yf_batch_size, total)
        log.info(
            f"Batch {batch_start // INGESTION.yf_batch_size + 1}: "
            f"tickers {batch_start + 1}–{batch_end} of {total} "
            f"(interval={interval}, lookback={max_days}d)"
        )

        for ticker in batch:
            try:
                t = yf.Ticker(ticker)
                df_pd = t.history(
                    period=f"{max_days}d",
                    interval=interval,
                    auto_adjust=True,
                )

                if df_pd.empty:
                    log.warning(f"  {ticker}: no data returned, skipping")
                    continue

                df_pd = df_pd.reset_index()
                df = pl.from_pandas(df_pd)
                df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})
                df = df.with_columns(pl.lit(ticker).alias("ticker"))

                # Rename 'datetime' → 'timestamp' if present
                if "datetime" in df.columns:
                    df = df.rename({"datetime": "timestamp"})

                rows = df.height
                log.info(f"  {ticker}: {rows:,} rows")
                results[ticker] = df

            except Exception as e:
                log.error(f"  {ticker}: FAILED — {e}")

        if batch_end < total:
            log.info(f"  Sleeping {INGESTION.yf_sleep}s between batches...")
            time.sleep(INGESTION.yf_sleep)

    return results


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_parquet(
    data: dict[str, pl.DataFrame],
    output_dir: Path,
    compression: str = INGESTION.parquet_compression,
) -> None:
    """
    Write one Parquet file per ticker: {output_dir}/{TICKER}.parquet

    Uses zstd compression for excellent ratio + speed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for ticker, df in data.items():
        path = output_dir / f"{ticker}.parquet"
        df.write_parquet(path, compression=compression)
        size_mb = path.stat().st_size / 1_048_576
        log.info(f"  Wrote {path.name} ({df.height:,} rows, {size_mb:.2f} MB)")
        written += 1

    log.info(f"Total: {written} files written to {output_dir}")


# ---------------------------------------------------------------------------
# Manifest — log what was ingested and when
# ---------------------------------------------------------------------------


def write_manifest(output_dir: Path, data: dict[str, pl.DataFrame], mode: str) -> None:
    """Write a simple JSON manifest for auditability."""
    import json

    manifest = {
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "source": "yfinance",
        "ticker_count": len(data),
        "tickers": sorted(data.keys()),
        "total_rows": sum(df.height for df in data.values()),
    }

    path = output_dir / "_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_tickers(scope: str, file: str | None = None) -> list[str]:
    if scope == "test":
        return list(UNIVERSE.test)
    elif scope == "sp500":
        return UNIVERSE.sp500()
    elif scope == "file":
        if not file:
            raise ValueError("--file required when --scope=file")
        return UNIVERSE.from_file(file)
    else:
        raise ValueError(f"Unknown scope: {scope}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest US equity data from Yahoo Finance → Parquet"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "intraday"],
        required=True,
        help="daily = full history EOD bars, intraday = rolling window",
    )
    parser.add_argument(
        "--scope",
        choices=["test", "sp500", "file"],
        default="test",
        help="Which ticker universe to ingest",
    )
    parser.add_argument(
        "--interval",
        default="1m",
        help="Intraday interval (1m, 5m, 1h). Ignored for daily mode.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to ticker file (one per line). Used with --scope=file",
    )
    args = parser.parse_args()

    PATHS.ensure_all()
    tickers = resolve_tickers(args.scope, args.file)
    log.info(f"Universe: {len(tickers)} tickers ({args.scope})")

    if args.mode == "daily":
        log.info("=" * 60)
        log.info("DAILY BAR INGESTION")
        log.info("=" * 60)
        data = download_daily(tickers)
        out = PATHS.raw_yf_daily
        write_parquet(data, out)
        write_manifest(out, data, "daily")

    elif args.mode == "intraday":
        log.info("=" * 60)
        log.info(f"INTRADAY BAR INGESTION (interval={args.interval})")
        log.info("=" * 60)
        data = download_intraday(tickers, interval=args.interval)
        out = PATHS.raw_yf_intraday / args.interval
        write_parquet(data, out)
        write_manifest(out, data, f"intraday_{args.interval}")

    # Summary
    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Tickers ingested: {len(data)}")
    log.info(f"  Total rows: {sum(df.height for df in data.values()):,}")
    log.info(f"  Output: {out}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
