"""
quantdata.query
~~~~~~~~~~~~~~~
Zero-copy DuckDB → Polars query interface over the parquet catalog.

Price data:
    qd.daily("AAPL", start="2015-01-01")
    qd.daily(["AAPL", "MSFT"], columns=["date", "close", "volume"])

Fundamental data:
    qd.balance_sheet("AAPL")
    qd.income("AAPL", freq="Q")
    qd.cash_flow("AAPL", metric="Free Cash Flow")
    qd.earnings("NVDA")

Cross-dataset joins:
    qd.price_earnings("AAPL")   # price + EPS aligned by date
    qd.fundamentals_wide("AAPL", metrics=["Total Revenue", "Net Income"])

Raw SQL:
    qd.sql("SELECT * FROM income_stmt WHERE ticker='AAPL' AND metric LIKE '%Revenue%'")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import duckdb
import polars as pl

from config import PATHS


class QuantDataQuery:
    """Zero-copy DuckDB → Polars query interface."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = str(db_path or PATHS.duckdb_path)
        self._con = duckdb.connect(self._db_path, read_only=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> QuantDataQuery:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ==================================================================
    # Core
    # ==================================================================

    def sql(self, query: str) -> pl.DataFrame:
        """Run arbitrary SQL, return Polars DataFrame via Arrow (zero-copy)."""
        return self._con.sql(query).pl()

    # ==================================================================
    # Price data
    # ==================================================================

    def daily(
        self,
        tickers: str | Sequence[str],
        start: str | date | None = None,
        end: str | date | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        cols = ", ".join(columns) if columns else "*"
        where = self._build_where(tickers, start, end, date_col="date")
        return self.sql(f"SELECT {cols} FROM daily {where} ORDER BY ticker, date")

    def intraday(
        self,
        tickers: str | Sequence[str],
        interval: str = "1m",
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pl.DataFrame:
        view = f"intraday_{interval}"
        where = self._build_where(tickers, start, end, date_col="timestamp")
        return self.sql(f"SELECT * FROM {view} {where} ORDER BY ticker, timestamp")

    # ==================================================================
    # Fundamental statements (long format)
    # ==================================================================

    def balance_sheet(
        self,
        tickers: str | Sequence[str],
        freq: str | None = None,
        metric: str | None = None,
    ) -> pl.DataFrame:
        """Query balance sheet data. freq='Q' or 'A'. metric filters by name."""
        return self._query_statement("balance_sheet", tickers, freq, metric)

    def income(
        self,
        tickers: str | Sequence[str],
        freq: str | None = None,
        metric: str | None = None,
    ) -> pl.DataFrame:
        """Query income statement. Common metrics: Total Revenue, Net Income, EBITDA."""
        return self._query_statement("income_stmt", tickers, freq, metric)

    def cash_flow(
        self,
        tickers: str | Sequence[str],
        freq: str | None = None,
        metric: str | None = None,
    ) -> pl.DataFrame:
        """Query cash flow statement. Common: Free Cash Flow, Operating Cash Flow."""
        return self._query_statement("cash_flow", tickers, freq, metric)

    def _query_statement(
        self,
        view: str,
        tickers: str | Sequence[str],
        freq: str | None,
        metric: str | None,
    ) -> pl.DataFrame:
        clauses = self._ticker_clause(tickers)
        if freq:
            clauses.append(f"freq = '{freq}'")
        if metric:
            clauses.append(f"metric = '{metric}'")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self.sql(f"SELECT * FROM {view} {where} ORDER BY ticker, date, metric")

    # ==================================================================
    # Earnings
    # ==================================================================

    def earnings(
        self,
        tickers: str | Sequence[str],
    ) -> pl.DataFrame:
        """Get earnings dates with EPS estimate, actual, and surprise."""
        clauses = self._ticker_clause(tickers)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self.sql(
            f"SELECT * FROM earnings {where} ORDER BY ticker, earnings_date"
        )

    # ==================================================================
    # Cross-dataset joins (the ML feature builders)
    # ==================================================================

    def price_earnings(
        self,
        tickers: str | Sequence[str],
        start: str | date | None = None,
    ) -> pl.DataFrame:
        """
        Join daily price with earnings data.
        Returns: date, ticker, close, volume, eps_estimate, reported_eps, surprise_pct
        """
        tick_clause = " AND ".join(self._ticker_clause(tickers, prefix="d"))
        start_clause = f"AND d.date >= '{start}'" if start else ""

        return self.sql(f"""
            SELECT
                d.date,
                d.ticker,
                d.close,
                d.volume,
                e.eps_estimate,
                e.reported_eps,
                e.surprisepct as surprise_pct
            FROM daily d
            LEFT JOIN earnings e
                ON d.ticker = e.ticker
                AND d.date = e.earnings_date::DATE
            WHERE {tick_clause} {start_clause}
            ORDER BY d.ticker, d.date
        """)

    def fundamentals_wide(
        self,
        tickers: str | Sequence[str],
        metrics: list[str],
        view: str = "income_stmt",
        freq: str = "Q",
    ) -> pl.DataFrame:
        """
        Pivot fundamental data from long to wide format.

        Example:
            qd.fundamentals_wide("AAPL", ["Total Revenue", "Net Income"])

        Returns: ticker | date | Total Revenue | Net Income
        """
        metric_list = ", ".join(f"'{m}'" for m in metrics)
        pivot_cols = ", ".join(
            f"MAX(CASE WHEN metric = '{m}' THEN value END) AS \"{m}\"" for m in metrics
        )
        tick_clause = " AND ".join(self._ticker_clause(tickers))

        return self.sql(f"""
            SELECT ticker, date, {pivot_cols}
            FROM {view}
            WHERE {tick_clause}
              AND freq = '{freq}'
              AND metric IN ({metric_list})
            GROUP BY ticker, date
            ORDER BY ticker, date
        """)

    def price_fundamentals(
        self,
        tickers: str | Sequence[str],
        metrics: list[str],
        freq: str = "Q",
        start: str | date | None = None,
    ) -> pl.DataFrame:
        """
        Join daily prices with pivoted fundamentals (forward-filled).

        This is the core ML feature table: daily price rows enriched
        with the most recent quarterly fundamental values.

        Example:
            qd.price_fundamentals(
                "AAPL",
                metrics=["Total Revenue", "Net Income", "Total Assets"],
                start="2015-01-01"
            )
        """
        # Get wide fundamentals
        fund = self.fundamentals_wide(tickers, metrics, "income_stmt", freq)

        # Get daily prices
        prices = self.daily(
            tickers, start=start, columns=["ticker", "date", "close", "volume"]
        )

        # Polars join + forward fill (as-of join)
        # Sort both by ticker, date
        fund = fund.sort(["ticker", "date"])
        prices = prices.sort(["ticker", "date"])

        # Join and forward-fill fundamental values
        joined = prices.join_asof(
            fund,
            on="date",
            by="ticker",
            strategy="backward",  # use most recent fundamental <= price date
        )

        return joined

    # ==================================================================
    # Metadata
    # ==================================================================

    def tickers(self, view: str = "daily") -> list[str]:
        return (
            self.sql(f"SELECT DISTINCT ticker FROM {view} ORDER BY ticker")
            .get_column("ticker")
            .to_list()
        )

    def date_range(self, ticker: str, view: str = "daily") -> tuple[date, date]:
        date_col = "date" if view in ("daily", "daily_clean") else "timestamp"
        if view in ("balance_sheet", "income_stmt", "cash_flow"):
            date_col = "date"
        row = self.sql(
            f"SELECT min({date_col}), max({date_col}) FROM {view} "
            f"WHERE ticker = '{ticker}'"
        ).row(0)
        return row[0], row[1]

    def metrics(
        self, view: str = "income_stmt", ticker: str | None = None
    ) -> list[str]:
        """List all available metric names in a fundamental view."""
        where = f"WHERE ticker = '{ticker}'" if ticker else ""
        return (
            self.sql(f"SELECT DISTINCT metric FROM {view} {where} ORDER BY metric")
            .get_column("metric")
            .to_list()
        )

    def summary(self) -> pl.DataFrame:
        views = self._con.sql(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).fetchall()

        rows = []
        for (view_name,) in views:
            cols = self._con.sql(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{view_name}'"
            ).fetchall()
            col_names = [c[0] for c in cols]

            # Detect date column
            if "date" in col_names:
                date_col = "date"
            elif "timestamp" in col_names:
                date_col = "timestamp"
            elif "earnings_date" in col_names:
                date_col = "earnings_date"
            else:
                date_col = None

            try:
                date_range = (
                    f"min({date_col}), max({date_col})" if date_col else "NULL, NULL"
                )
                stats = self._con.sql(
                    f"SELECT count(DISTINCT ticker), count(*), "
                    f"{date_range} FROM {view_name}"
                ).fetchone()
                rows.append(
                    {
                        "view": view_name,
                        "tickers": stats[0],
                        "rows": stats[1],
                        "min_date": str(stats[2]) if stats[2] else "",
                        "max_date": str(stats[3]) if stats[3] else "",
                    }
                )
            except Exception:
                rows.append(
                    {
                        "view": view_name,
                        "tickers": 0,
                        "rows": 0,
                        "min_date": "",
                        "max_date": "",
                    }
                )

        return pl.DataFrame(rows)

    # ==================================================================
    # Internals
    # ==================================================================

    @staticmethod
    def _ticker_clause(
        tickers: str | Sequence[str], prefix: str | None = None
    ) -> list[str]:
        col = f"{prefix}.ticker" if prefix else "ticker"
        if isinstance(tickers, str):
            return [f"{col} = '{tickers}'"]
        tick_list = ", ".join(f"'{t}'" for t in tickers)
        return [f"{col} IN ({tick_list})"]

    @staticmethod
    def _build_where(
        tickers: str | Sequence[str],
        start: str | date | None,
        end: str | date | None,
        date_col: str = "date",
    ) -> str:
        clauses: list[str] = []
        if isinstance(tickers, str):
            clauses.append(f"ticker = '{tickers}'")
        else:
            tick_list = ", ".join(f"'{t}'" for t in tickers)
            clauses.append(f"ticker IN ({tick_list})")
        if start is not None:
            clauses.append(f"{date_col} >= '{start}'")
        if end is not None:
            clauses.append(f"{date_col} <= '{end}'")
        return "WHERE " + " AND ".join(clauses) if clauses else ""
