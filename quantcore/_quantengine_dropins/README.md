# quantengine drop-ins (staged from quantcore session)

Staged under `quantcore/_quantengine_dropins/` for delivery. Copy these
files into the quantengine repo at the indicated paths.

## Files

| Source (here) | Target (quantengine repo) |
|---|---|
| `tests/fixtures/build_mini_duckdb.py` | `tests/fixtures/build_mini_duckdb.py` |
| `tests/test_duckdb_loader_roundtrip.py` | `tests/test_duckdb_loader_roundtrip.py` |

Create an empty `tests/fixtures/__init__.py` if one doesn't exist.

## Run

```bash
cd <quantengine-repo>
pip install -e '.[dev]'          # needs duckdb, pandas, pyarrow, pytest
pytest tests/test_duckdb_loader_roundtrip.py -v
```

## What it validates

1. `DuckDBUniverseResolver` correctly handles entries (NVDA IPO 2026-01-15)
   and exits (TSLA delisting 2026-01-20) — survivorship-bias safe.
2. `DuckDBSnapshotLoader` never leaks `session_date > as_of`.
3. Stale-tolerance: GE price gap 2026-01-10..12 resolves to the 01-09 close.
4. DuckDB backend ↔ pandas `pit_filter` parity on a shared fixture.
5. Universe ∩ snapshot drops de-listed tickers even when prior prices exist.

## Contract assumptions (verify before running)

Constructors and function signatures this scaffold assumes:

```python
DuckDBSnapshotLoader(db_path: str, price_table: str = "daily_bars_adj",
                     price_field: str = "close", ...)
DuckDBUniverseResolver(db_path: str, table: str = "universe_membership",
                       ticker_col: str = "ticker",
                       date_col: str = "session_date",
                       member_col: str = "member")
pit_filter(df, *, as_of, universe,
           price_col="price", ticker_col="ticker", date_col="session_date")
```

If the reader's defaults have drifted, override explicitly at call sites in
`test_duckdb_loader_roundtrip.py`. No other scaffold logic depends on the
defaults.

## Why the fixture is tiny

~150 rows across 5 tickers × 23 business days. Keeps test runtime <1s while
still exercising every PIT edge case (IPO, delisting, price gap, stale
tolerance, future-leak guardrail).
