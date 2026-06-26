#!/usr/bin/env python3
"""
ingest_fundamentals.py
~~~~~~~~~~~~~~~~~~~~~~
Download quarterly & annual fundamental data from Yahoo Finance → Parquet.

Data types:
    balance_sheet   : Assets, liabilities, equity
    income_stmt     : Revenue, EBITDA, net income, EPS
    cash_flow       : Operating, investing, financing cash flows
    earnings        : EPS actual vs estimate, surprise %, dates

Usage:
    # Smoke test with 10 tickers
    python3 scripts/ingest_fundamentals.py --scope test

    # Full universe from ticker file
    python3 scripts/ingest_fundamentals.py --scope file --file us_large_cap_tickers.txt

    # Single ticker debug
    python3 scripts/ingest_fundamentals.py --scope file --file <(echo "AAPL")

Author: Jules (quantdata pipeline)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import yfinance as yf

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
log = logging.getLogger("ingest_fundamentals")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

FUND_ROOT = PATHS.root / "raw" / "yfinance" / "fundamentals"

STMT_DIRS = {
    "balance_sheet": FUND_ROOT / "balance_sheet",
    "income_stmt": FUND_ROOT / "income_stmt",
    "cash_flow": FUND_ROOT / "cash_flow",
    "earnings": FUND_ROOT / "earnings",
}


def ensure_dirs() -> None:
    for d in STMT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Statement extraction helpers
# ---------------------------------------------------------------------------


def _stmt_to_polars(
    df_pd,
    ticker: str,
    freq: str,
) -> pl.DataFrame | None:
    """
    Convert a yfinance statement (pd.DataFrame with dates as columns,
    line items as rows) into a tidy Polars DataFrame.

    Output schema:
        ticker | date | metric | value
    """
    if df_pd is None or df_pd.empty:
        return None

    # yfinance returns metrics as index, dates as columns
    # Transpose → dates as index, metrics as columns
    df_t = df_pd.T
    df_t.index.name = "date"
    df_t = df_t.reset_index()

    # Melt to long format
    df = pl.from_pandas(df_t)

    # Normalize column names
    date_col = df.columns[0]
    metric_cols = df.columns[1:]

    df = df.unpivot(
        index=date_col,
        on=metric_cols,
        variable_name="metric",
        value_name="value",
    )

    # Rename date column and add metadata
    df = df.rename({date_col: "date"})
    df = df.with_columns(
        pl.lit(ticker).alias("ticker"),
        pl.lit(freq).alias("freq"),
    )

    # Cast date to Date type
    df = df.with_columns(pl.col("date").cast(pl.Date))

    # Cast value to Float64 (some come as int/object)
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))

    # Reorder
    df = df.select(["ticker", "date", "freq", "metric", "value"])

    return df


def _earnings_to_polars(ticker_obj: yf.Ticker, ticker: str) -> pl.DataFrame | None:
    """
    Extract earnings dates with EPS estimate/actual/surprise.

    yfinance .earnings_dates returns:
        EPS Estimate | Reported EPS | Surprise(%)
    indexed by earnings date.
    """
    try:
        df_pd = ticker_obj.earnings_dates
        if df_pd is None or df_pd.empty:
            return None

        df_pd = df_pd.reset_index()
        df = pl.from_pandas(df_pd)

        # Normalize column names
        df = df.rename(
            {
                c: c.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
                for c in df.columns
            }
        )

        df = df.with_columns(pl.lit(ticker).alias("ticker"))

        # Ensure date column exists
        date_col = df.columns[0]
        if date_col != "earnings_date":
            df = df.rename({date_col: "earnings_date"})

        return df

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def download_fundamentals(tickers: list[str]) -> dict[str, dict[str, pl.DataFrame]]:
    """
    Download all fundamental data for a list of tickers.

    Returns:
        {
            "balance_sheet": {ticker: DataFrame, ...},
            "income_stmt": {ticker: DataFrame, ...},
            "cash_flow": {ticker: DataFrame, ...},
            "earnings": {ticker: DataFrame, ...},
        }
    """
    results: dict[str, dict[str, pl.DataFrame]] = {
        "balance_sheet": {},
        "income_stmt": {},
        "cash_flow": {},
        "earnings": {},
    }

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
                counts = []

                # --- Balance Sheet (quarterly + annual) ---
                frames = []
                for freq, accessor in [
                    ("Q", t.quarterly_balance_sheet),
                    ("A", t.balance_sheet),
                ]:
                    df = _stmt_to_polars(accessor, ticker, freq)
                    if df is not None:
                        frames.append(df)
                if frames:
                    combined = pl.concat(frames)
                    results["balance_sheet"][ticker] = combined
                    counts.append(f"BS:{combined.height}")

                # --- Income Statement (quarterly + annual) ---
                frames = []
                for freq, accessor in [
                    ("Q", t.quarterly_income_stmt),
                    ("A", t.income_stmt),
                ]:
                    df = _stmt_to_polars(accessor, ticker, freq)
                    if df is not None:
                        frames.append(df)
                if frames:
                    combined = pl.concat(frames)
                    results["income_stmt"][ticker] = combined
                    counts.append(f"IS:{combined.height}")

                # --- Cash Flow (quarterly + annual) ---
                frames = []
                for freq, accessor in [("Q", t.quarterly_cashflow), ("A", t.cashflow)]:
                    df = _stmt_to_polars(accessor, ticker, freq)
                    if df is not None:
                        frames.append(df)
                if frames:
                    combined = pl.concat(frames)
                    results["cash_flow"][ticker] = combined
                    counts.append(f"CF:{combined.height}")

                # --- Earnings ---
                earn = _earnings_to_polars(t, ticker)
                if earn is not None:
                    results["earnings"][ticker] = earn
                    counts.append(f"E:{earn.height}")

                summary = " | ".join(counts) if counts else "no data"
                log.info(f"  {ticker}: {summary}")

            except Exception as e:
                log.error(f"  {ticker}: FAILED — {e}")

        # Rate-limit
        if batch_end < total:
            log.info(f"  Sleeping {INGESTION.yf_sleep}s between batches...")
            time.sleep(INGESTION.yf_sleep)

    return results


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


def write_fundamentals(
    data: dict[str, dict[str, pl.DataFrame]],
    compression: str = INGESTION.parquet_compression,
) -> None:
    """Write one parquet per ticker per statement type."""

    ensure_dirs()

    for stmt_type, ticker_data in data.items():
        out_dir = STMT_DIRS[stmt_type]
        written = 0

        for ticker, df in ticker_data.items():
            path = out_dir / f"{ticker}.parquet"
            df.write_parquet(path, compression=compression)
            written += 1

        log.info(f"  {stmt_type}: {written} files → {out_dir}")


def write_manifest(data: dict[str, dict[str, pl.DataFrame]]) -> None:
    """Write manifest for auditability."""
    manifest = {
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "source": "yfinance_fundamentals",
        "statements": {},
    }
    for stmt_type, ticker_data in data.items():
        manifest["statements"][stmt_type] = {
            "ticker_count": len(ticker_data),
            "total_rows": sum(df.height for df in ticker_data.values()),
        }

    path = FUND_ROOT / "_manifest.json"
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
        description="Ingest fundamental data from Yahoo Finance → Parquet"
    )
    parser.add_argument(
        "--scope",
        choices=["test", "sp500", "file"],
        default="test",
    )
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    PATHS.ensure_all()
    tickers = resolve_tickers(args.scope, args.file)
    log.info(f"Universe: {len(tickers)} tickers ({args.scope})")
    log.info("=" * 60)
    log.info("FUNDAMENTAL DATA INGESTION")
    log.info("=" * 60)

    data = download_fundamentals(tickers)
    write_fundamentals(data)
    write_manifest(data)

    # Summary
    log.info("=" * 60)
    log.info("DONE")
    for stmt_type, ticker_data in data.items():
        rows = sum(df.height for df in ticker_data.values())
        log.info(f"  {stmt_type}: {len(ticker_data)} tickers, {rows:,} rows")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
