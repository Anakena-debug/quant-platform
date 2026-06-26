"""
quantdata.config
~~~~~~~~~~~~~~~~
Single source of truth for paths, ticker universe, and ingestion settings.

Usage:
    from config import PATHS, UNIVERSE, INGESTION
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("QUANTDATA_ROOT", "~/dev/quantdata")
        ).expanduser()
    )

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def raw_yf_daily(self) -> Path:
        return self.raw / "yfinance" / "daily"

    @property
    def raw_yf_intraday(self) -> Path:
        return self.raw / "yfinance" / "intraday"

    @property
    def raw_polygon(self) -> Path:
        return self.raw / "polygon" / "minute"

    @property
    def raw_lseg(self) -> Path:
        return self.raw / "lseg" / "intraday"

    @property
    def processed_daily(self) -> Path:
        return self.root / "processed" / "daily"

    @property
    def processed_minute(self) -> Path:
        return self.root / "processed" / "minute"

    @property
    def catalog(self) -> Path:
        return self.root / "catalog"

    @property
    def duckdb_path(self) -> Path:
        return self.catalog / "quantdata.duckdb"

    def ensure_all(self) -> None:
        """Create every directory in the tree."""
        for attr in [
            "raw_yf_daily",
            "raw_yf_intraday",
            "raw_polygon",
            "raw_lseg",
            "processed_daily",
            "processed_minute",
            "catalog",
        ]:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# ---------------------------------------------------------------------------
# Universe — US equities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Universe:
    """
    Ticker lists for different ingestion scopes.
    Start small (test), scale to full S&P 500, then total market.
    """

    # 10 liquid names for smoke-testing the pipeline
    test: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "JPM",
        "V",
        "UNH",
    )

    @staticmethod
    def sp500() -> list[str]:
        """
        Fetch live S&P 500 constituents from Wikipedia.
        Falls back to a cached snapshot if offline.
        """

        try:
            import pandas as pd

            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            tickers = sorted(
                tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
            )
            return tickers
        except Exception:
            # Offline fallback: return test universe
            return list(Universe().test)

    @staticmethod
    def from_file(path: str | Path) -> list[str]:
        """Load tickers from a plain-text file (one per line)."""
        p = Path(path)
        return [
            line.strip()
            for line in p.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]


UNIVERSE = Universe()


# ---------------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ingestion:
    # yfinance daily: how far back
    yf_daily_period: str = "max"  # full history

    # yfinance intraday: max lookback per interval
    # 1m → 7 days, 5m → 60 days, 1h → 730 days
    yf_intraday_intervals: dict[str, int] = field(
        default_factory=lambda: {
            "1m": 7,
            "5m": 60,
            "1h": 730,
        }
    )

    # Batch size for yfinance downloads (avoid rate limits)
    yf_batch_size: int = 50

    # Sleep between batches (seconds)
    yf_sleep: float = 2.0

    # Parquet compression
    parquet_compression: str = "zstd"

    # Polygon.io API key (set via env)
    polygon_api_key: str = field(
        default_factory=lambda: os.environ.get("POLYGON_API_KEY", "")
    )


INGESTION = Ingestion()
