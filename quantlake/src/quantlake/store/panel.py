"""quantlake — survivorship-free point-in-time daily panel + delisting metadata + validation.

Turns raw (split-unadjusted) Databento OHLCV into a split-adjusted, instrument_id-keyed daily
panel that is survivorship-free BY CONSTRUCTION — delisted instruments are retained (their bars
simply stop), so a per-day liquidity screen downstream yields the true point-in-time universe.
Emits the column contract the pandas consumers (quantcore / alpha_R xs_panel) expect:
``ticker, raw_symbol, date, open, high, low, close, volume, dividends, stock_splits`` with
``close`` split-adjusted and div/splits zeroed (already adjusted; never re-applied downstream).
Dividends are absent from ohlcv-1d → PRICE return (disclosed).

Re-homed from quantdata in s79 (block item 7); quantdata.panel is now a pure re-export shim.
"""

from __future__ import annotations

import pandas as pd

from quantlake.store.corp_actions import adjust_splits, residual_extreme_return_rate


def build_panel(raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Raw OHLCV → split-adjusted, survivorship-free PIT panel (s59 column contract).

    OHLC are all divided by the cumulative ``adj_factor`` for a consistent adjusted bar.
    """
    adj = adjust_splits(raw_ohlcv, key="instrument_id", date_col="date")
    f = adj["adj_factor"].to_numpy()
    out = pd.DataFrame(
        {
            "ticker": adj["instrument_id"].astype("int64"),
            "raw_symbol": adj["raw_symbol"],
            "date": adj["date"],
            "open": adj["open"].to_numpy() / f,
            "high": adj["high"].to_numpy() / f,
            "low": adj["low"].to_numpy() / f,
            "close": adj["adj_close"],  # split-adjusted close
            "volume": adj["volume"],
            "dividends": 0.0,  # ohlcv-1d has no dividends → price return (disclosed)
            "stock_splits": 0.0,  # already adjusted; never re-applied downstream
        }
    )
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def delisting_metadata(panel: pd.DataFrame, *, gap_days: int = 20) -> pd.DataFrame:
    """Per-instrument lifespan + a ``delisted`` flag (last bar > ``gap_days`` before panel end).

    Returns columns ``ticker, raw_symbol, first_date, last_date, n_days, delisted``.
    """
    panel_end = pd.Timestamp(panel["date"].max())  # pyright: ignore[reportArgumentType]  # pandas-stub
    g = panel.groupby("ticker")
    meta = pd.DataFrame(
        {
            "raw_symbol": g["raw_symbol"].last(),
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
            "n_days": g["date"].count(),
        }
    ).reset_index()
    meta["delisted"] = meta["last_date"] < (panel_end - pd.Timedelta(days=gap_days))
    return meta


def validate_panel(
    panel: pd.DataFrame, *, extreme_threshold: float = 0.5, gap_days: int = 20
) -> dict[str, object]:
    """PIT / survivorship / adjustment sanity checks. Returns a report dict (raises nothing).

    - ``n_delisted`` > 0 confirms the panel is survivorship-free (delisted names retained);
      ``gap_days`` is the lag past which a stopped instrument counts as delisted.
    - ``residual_extreme_rate`` is the split-adjustment-error bound (mostly genuine crashes).
    - ``all_prices_positive`` / ``no_future_dates`` / ``unique_ticker_date`` are basic integrity.
    """
    meta = delisting_metadata(panel, gap_days=gap_days)
    # panel["close"] is already split-adjusted (build_panel output) -> measure the residual
    # extreme-return rate on it directly (the adjustment-error bound).
    rate = residual_extreme_return_rate(
        panel.assign(adj_close=panel["close"]),
        key="ticker",
        date_col="date",
        threshold=extreme_threshold,
    )
    return {
        "n_instruments": int(panel["ticker"].nunique()),  # pyright: ignore[reportArgumentType]
        "n_delisted": int(meta["delisted"].sum()),  # pyright: ignore[reportArgumentType]
        "survivorship_free": bool(meta["delisted"].sum() > 0),
        "residual_extreme_rate": rate,
        "all_prices_positive": bool((panel["close"] > 0).all()),
        "no_future_dates": bool(panel["date"].max() <= pd.Timestamp.today().normalize()),
        "unique_ticker_date": bool(not panel.duplicated(subset=["ticker", "date"]).any()),
        "date_range": (str(panel["date"].min().date()), str(panel["date"].max().date())),
    }


__all__ = ["build_panel", "delisting_metadata", "validate_panel"]
