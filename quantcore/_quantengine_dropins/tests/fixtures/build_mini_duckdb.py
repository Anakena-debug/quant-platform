"""
Build a tiny DuckDB fixture exercising PIT snapshot and universe resolution.

Drop location
-------------
    quantengine/tests/fixtures/build_mini_duckdb.py

What it writes
--------------
Two DuckDB tables with deliberate PIT edge cases::

    daily_bars_adj(ticker TEXT, session_date DATE, close DOUBLE, volume BIGINT)
    universe_membership(ticker TEXT, session_date DATE, member BOOLEAN)

Fixture universe (5 tickers spanning 2026-01-01 .. 2026-02-01):

    AAPL  : member throughout, full price history
    MSFT  : member throughout, full price history
    TSLA  : delisted 2026-01-20 → member=False from 2026-01-20 onward
    NVDA  : added 2026-01-15 → member=True from 2026-01-15 onward
    GE    : member throughout, but has a price GAP on 2026-01-10..2026-01-12

Known-answer expectations (asserted in the companion test module):

    * pit_filter(..., as_of=2026-01-10) for GE returns the 2026-01-09 row
      (most-recent before gap). Width of stale-price tolerance defaults to
      3 days — verify in the reader.
    * DuckDBUniverseResolver(2026-01-19) includes TSLA but NOT NVDA.
    * DuckDBUniverseResolver(2026-01-20) excludes TSLA (delisted).
    * DuckDBUniverseResolver(2026-01-14) excludes NVDA (not yet listed).
    * PIT at 2026-01-15 never returns a 2026-01-16 close (no future leak).

CLI
---
    python -m tests.fixtures.build_mini_duckdb out.duckdb

Returns the path written.

Limits
------
* Fixture is intentionally tiny (~150 rows) to keep tests fast.
* Adjustments (splits/dividends) are NOT materialised here — production uses
  a separate ``corp_actions`` pipeline. Scaffolded for future extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import duckdb
import pandas as pd


FIXTURE_START: Final[pd.Timestamp] = pd.Timestamp("2026-01-01")
FIXTURE_END: Final[pd.Timestamp] = pd.Timestamp("2026-02-01")


def _business_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def build_price_frame() -> pd.DataFrame:
    """Deterministic price frame with documented edge cases."""
    dates = _business_days(FIXTURE_START, FIXTURE_END)
    rows: list[dict] = []
    base = {"AAPL": 180.0, "MSFT": 400.0, "TSLA": 250.0, "NVDA": 500.0, "GE": 160.0}
    for tkr, p0 in base.items():
        for i, d in enumerate(dates):
            # TSLA stops trading on 2026-01-20
            if tkr == "TSLA" and d >= pd.Timestamp("2026-01-20"):
                continue
            # NVDA begins trading 2026-01-15
            if tkr == "NVDA" and d < pd.Timestamp("2026-01-15"):
                continue
            # GE gap: skip 2026-01-10..12
            if tkr == "GE" and pd.Timestamp("2026-01-10") <= d <= pd.Timestamp("2026-01-12"):
                continue
            rows.append(
                {
                    "ticker": tkr,
                    "session_date": d.date(),
                    "close": float(p0 + 0.1 * i),
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows)


def build_universe_frame() -> pd.DataFrame:
    """Point-in-time membership; one row per (ticker, session_date)."""
    dates = _business_days(FIXTURE_START, FIXTURE_END)
    rows: list[dict] = []
    for d in dates:
        rows.append({"ticker": "AAPL", "session_date": d.date(), "member": True})
        rows.append({"ticker": "MSFT", "session_date": d.date(), "member": True})
        rows.append({"ticker": "GE", "session_date": d.date(), "member": True})
        rows.append(
            {
                "ticker": "TSLA",
                "session_date": d.date(),
                "member": bool(d < pd.Timestamp("2026-01-20")),
            }
        )
        rows.append(
            {
                "ticker": "NVDA",
                "session_date": d.date(),
                "member": bool(d >= pd.Timestamp("2026-01-15")),
            }
        )
    return pd.DataFrame(rows)


def build_mini_duckdb(out_path: str | Path) -> Path:
    """Materialise the fixture DB at ``out_path``. Overwrites if extant."""
    out = Path(out_path)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    price_df = build_price_frame()
    univ_df = build_universe_frame()

    con = duckdb.connect(str(out))
    try:
        con.execute(
            """
            CREATE TABLE daily_bars_adj (
                ticker       TEXT NOT NULL,
                session_date DATE NOT NULL,
                close        DOUBLE NOT NULL,
                volume       BIGINT NOT NULL,
                PRIMARY KEY (ticker, session_date)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE universe_membership (
                ticker       TEXT NOT NULL,
                session_date DATE NOT NULL,
                member       BOOLEAN NOT NULL,
                PRIMARY KEY (ticker, session_date)
            )
            """
        )
        con.register("_prices_df", price_df)
        con.register("_universe_df", univ_df)
        con.execute("INSERT INTO daily_bars_adj SELECT * FROM _prices_df")
        con.execute("INSERT INTO universe_membership SELECT * FROM _universe_df")
        # Indices to exercise realistic query plans
        con.execute("CREATE INDEX idx_prices_ticker_date ON daily_bars_adj(ticker, session_date)")
        con.execute("CREATE INDEX idx_univ_date ON universe_membership(session_date)")
    finally:
        con.close()

    return out


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="Output .duckdb path")
    ns = ap.parse_args()
    p = build_mini_duckdb(ns.out)
    print(f"wrote {p}")


if __name__ == "__main__":
    _cli()
