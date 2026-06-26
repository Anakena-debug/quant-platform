"""quantdata — Databento daily OHLCV ingestion (EQUS.MINI ``ohlcv-1d``).

Pulls the consolidated US-equity daily bars (``ALL_SYMBOLS`` includes delisted instruments →
survivorship-free) into a tidy frame. Prices are RAW/unadjusted — corp-action adjustment is
``corp_actions.adjust_splits``. The API key is read from ``DATABENTO_API_KEY`` (env) ONLY —
never hardcoded. Promoted from the s61 alpha_R prototype.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASET = "EQUS.MINI"
SCHEMA = "ohlcv-1d"
DEFAULT_START = "2023-03-28"  # EQUS.MINI inception (verified via get_dataset_range)

_OHLCV_COLS = ["instrument_id", "raw_symbol", "date", "open", "high", "low", "close", "volume"]


def _historical_client():
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is not set (export it; never hardcode the key).")
    return db.Historical(key)


def fetch_ohlcv_1d(
    symbols: str | list[str] = "ALL_SYMBOLS",
    start: str = DEFAULT_START,
    end: str | None = None,
    dataset: str = DATASET,
) -> pd.DataFrame:
    """Fetch raw daily OHLCV → tidy frame with columns :data:`_OHLCV_COLS`.

    Survivorship-free with ``symbols='ALL_SYMBOLS'`` (delisted instruments simply stop having
    bars). Drops non-positive closes; sorted by (instrument_id, date).
    """
    client = _historical_client()
    data = client.timeseries.get_range(
        dataset=dataset, schema=SCHEMA, symbols=symbols, start=start, end=end
    )
    df = data.to_df().reset_index()
    df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
    sym_col = "symbol" if "symbol" in df.columns else None
    out = pd.DataFrame(
        {
            "instrument_id": df["instrument_id"].astype("int64"),
            "raw_symbol": (df[sym_col].astype(str) if sym_col else df["instrument_id"].astype(str)),
            "date": df["date"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
        }
    )
    out = out[out["close"] > 0]
    return out.sort_values(["instrument_id", "date"]).reset_index(drop=True)


def cache_ohlcv_1d(
    out_path: str | Path,
    symbols: str | list[str] = "ALL_SYMBOLS",
    start: str = DEFAULT_START,
    end: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch (or load cached) raw OHLCV parquet at ``out_path``. Idempotent unless ``refresh``."""
    out_path = Path(out_path)
    if out_path.exists() and not refresh:
        return pd.read_parquet(out_path)
    raw = fetch_ohlcv_1d(symbols, start=start, end=end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(out_path, index=False)
    return raw


__all__ = ["DATASET", "DEFAULT_START", "SCHEMA", "cache_ohlcv_1d", "fetch_ohlcv_1d"]
