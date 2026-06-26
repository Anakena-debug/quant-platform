# quantdata

Market data infrastructure for the `quantcore` research pipeline.

**Architecture:** yfinance/Polygon/LSEG → Parquet (on disk) → DuckDB (catalog) → Polars (in RAM) → quantcore

## Quickstart

```bash
cd ~/dev/quantdata

# Install deps
make setup

# Smoke test: 10 tickers, daily bars, full history
make test

# Query interactively
make query
# > SELECT ticker, count(*) as bars, min(date), max(date)
#   FROM daily GROUP BY ticker ORDER BY bars DESC;

# Full S&P 500
make sp500
```

## From quantcore

```python
# In quantcore, set QUANTDATA_ROOT and import
import sys; sys.path.insert(0, "path/to/quantdata")
from query import QuantDataQuery

qd = QuantDataQuery()
df = qd.daily("AAPL", start="2015-01-01")               # → Polars DataFrame
df = qd.daily(["AAPL", "MSFT"], columns=["date", "close", "volume"])
df = qd.sql("SELECT * FROM daily WHERE ticker = 'NVDA' AND date >= '2023-01-01'")
print(qd.summary())
```

## Directory layout

```
quantdata/
├── raw/                    # Immutable ingested data
│   ├── yfinance/daily/     # One parquet per ticker
│   ├── yfinance/intraday/  # Partitioned by interval
│   ├── polygon/minute/     # (when subscribed)
│   └── lseg/intraday/      # Tick History exports
├── processed/              # Cleaned, adjusted, merged
├── scripts/
│   ├── ingest_yfinance.py  # Yahoo Finance → parquet
│   ├── build_catalog.py    # Parquet → DuckDB views
│   └── validate_data.py    # Data quality checks
├── catalog/
│   └── quantdata.duckdb    # SQL views over parquet
├── config.py               # Paths, universe, settings
├── query.py                # DuckDB+Polars query API
└── Makefile
```

## Data source limits

| Source    | Daily   | Intraday        | Notes                        |
|-----------|---------|-----------------|------------------------------|
| yfinance  | 40+ yrs | 1m=7d, 5m=60d  | Free, fragile, rate-limited  |
| Polygon   | 5+ yrs  | Full 1m history | $29/mo, best indie option    |
| LSEG      | 20+ yrs | Tick-level      | Academic only, use CodeBook  |

## Adding Polygon (when ready)

```bash
export POLYGON_API_KEY="your_key"
python scripts/ingest_polygon.py --scope sp500
```
