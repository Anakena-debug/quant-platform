#!/usr/bin/env python3
"""
build_catalog.py
~~~~~~~~~~~~~~~~
Build a DuckDB catalog with views over all ingested Parquet files.

Covers:
    - daily price bars
    - intraday price bars
    - fundamental statements (balance_sheet, income_stmt, cash_flow)
    - earnings (EPS estimates, actuals, surprise)

Usage:
    python3 scripts/build_catalog.py

Then query:
    duckdb catalog/quantdata.duckdb

    -- Price data
    SELECT ticker, count(*) as bars FROM daily GROUP BY ticker ORDER BY bars DESC;

    -- Fundamentals: AAPL revenue over time
    SELECT date, value FROM income_stmt
    WHERE ticker = 'AAPL' AND metric = 'Total Revenue' AND freq = 'Q'
    ORDER BY date;

    -- Join price + earnings
    SELECT e.earnings_date, e.eps_estimate, e.reported_eps, e.surprisepct,
           d.close as price_at_earnings
    FROM earnings e
    JOIN daily d ON e.ticker = d.ticker AND d.date = e.earnings_date::DATE
    WHERE e.ticker = 'NVDA'
    ORDER BY e.earnings_date;
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_catalog")


def _create_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    parquet_dir: Path,
) -> bool:
    """Create a view if parquet files exist. Returns True if created."""
    files = [f for f in parquet_dir.glob("*.parquet") if not f.name.startswith("_")]
    if not files:
        return False

    glob = str(parquet_dir / "*.parquet")
    con.execute(f"DROP VIEW IF EXISTS {view_name}")
    con.execute(
        f"""
        CREATE VIEW {view_name} AS
        SELECT * FROM read_parquet('{glob}', union_by_name=true)
        """
    )
    return True


def build_catalog() -> None:
    PATHS.ensure_all()
    db_path = str(PATHS.duckdb_path)
    log.info(f"Building catalog at {db_path}")

    con = duckdb.connect(db_path)
    views_created = []

    # ------------------------------------------------------------------
    # 1. Daily bars
    # ------------------------------------------------------------------
    if _create_view(con, "daily", PATHS.raw_yf_daily):
        stats = con.sql("SELECT count(DISTINCT ticker), count(*) FROM daily").fetchone()
        log.info(f"  daily: {stats[0]} tickers, {stats[1]:,} rows")
        views_created.append("daily")

    # ------------------------------------------------------------------
    # 2. Intraday bars (one view per interval)
    # ------------------------------------------------------------------
    intraday_base = PATHS.raw_yf_intraday
    if intraday_base.exists():
        for interval_dir in sorted(intraday_base.iterdir()):
            if not interval_dir.is_dir():
                continue
            view_name = f"intraday_{interval_dir.name}"
            if _create_view(con, view_name, interval_dir):
                stats = con.sql(
                    f"SELECT count(DISTINCT ticker), count(*) FROM {view_name}"
                ).fetchone()
                log.info(f"  {view_name}: {stats[0]} tickers, {stats[1]:,} rows")
                views_created.append(view_name)

    # ------------------------------------------------------------------
    # 3. Fundamental statements (long: ticker|date|freq|metric|value)
    # ------------------------------------------------------------------
    fund_base = PATHS.root / "raw" / "yfinance" / "fundamentals"
    for stmt_type in ["balance_sheet", "income_stmt", "cash_flow"]:
        stmt_dir = fund_base / stmt_type
        if stmt_dir.exists() and _create_view(con, stmt_type, stmt_dir):
            stats = con.sql(
                f"SELECT count(DISTINCT ticker), count(*) FROM {stmt_type}"
            ).fetchone()
            log.info(f"  {stmt_type}: {stats[0]} tickers, {stats[1]:,} rows")
            views_created.append(stmt_type)

    # ------------------------------------------------------------------
    # 4. Earnings (wide: ticker|earnings_date|eps_estimate|reported_eps|surprise)
    # ------------------------------------------------------------------
    earnings_dir = fund_base / "earnings"
    if earnings_dir.exists() and _create_view(con, "earnings", earnings_dir):
        stats = con.sql(
            "SELECT count(DISTINCT ticker), count(*) FROM earnings"
        ).fetchone()
        log.info(f"  earnings: {stats[0]} tickers, {stats[1]:,} rows")
        views_created.append("earnings")

    # ------------------------------------------------------------------
    # 5. Processed daily (if exists)
    # ------------------------------------------------------------------
    if _create_view(con, "daily_clean", PATHS.processed_daily):
        stats = con.sql(
            "SELECT count(DISTINCT ticker), count(*) FROM daily_clean"
        ).fetchone()
        log.info(f"  daily_clean: {stats[0]} tickers, {stats[1]:,} rows")
        views_created.append("daily_clean")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info(f"\nCatalog ready: {len(views_created)} views")
    for v in views_created:
        log.info(f"  • {v}")

    con.close()
    log.info(f"\nOpen with: duckdb {db_path}")


if __name__ == "__main__":
    build_catalog()
