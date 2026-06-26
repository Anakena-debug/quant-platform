"""Point-in-time market snapshot loader.

Role
----
Every execution decision at session close :math:`T` must be driven by a
``MarketSnapshot`` that is *strictly* dated :math:`\\tau \\le T`. This module
is the single crossing point between ``quantdata`` (DuckDB) and
``quantengine`` (execution). Any look-ahead leak here silently
contaminates every downstream analytic.

Design
------
Two concrete loaders share a single PIT-filter function so the
correctness invariants are proven once and reused:

- ``DuckDBSnapshotLoader`` — production path, lazily imports ``duckdb``.
- ``DataFrameSnapshotLoader`` — testing / in-memory replay path; accepts a
  plain pandas frame with columns ``{ticker, session_date, price}``.

Both implement the ``SnapshotSource`` Protocol.

Contract (inputs → output)
--------------------------
Given:
    - ``as_of``:   pd.Timestamp at session close.
    - ``universe``: tuple[str, ...] of candidate tickers.

Return a ``MarketSnapshot`` where:
    1. Every included ticker has exactly one price row with
       ``session_date <= as_of``, the *latest* such row.
    2. Every included price is strictly positive.
    3. Ordering is lexicographic in ticker (deterministic).
    4. Tickers without any qualifying row are dropped (not errored) —
       the caller's risk gate will handle the missing residual.

Adjustment policy
-----------------
We assume the price table is *split/dividend adjusted* to ``as_of``. This
means ``CorpActionHandler`` must NOT reapply splits at replay time
(double-count). If your source table is unadjusted, either:

- Adjust upstream in quantdata and point this loader at the adjusted
  view, or
- Set the ``price_table`` to the unadjusted table and feed a
  ``CorpActionHandler`` the split stream explicitly during replay.

This module does not enforce the choice — pick one and document it in
your replay recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from quantengine.contracts.market import MarketSnapshot


# ---------------------------------------------------------------------------
# Shared PIT filter (the single source of truth for look-ahead correctness)
# ---------------------------------------------------------------------------
def pit_filter(
    df: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    universe: tuple[str, ...] | list[str],
    price_col: str = "price",
    ticker_col: str = "ticker",
    date_col: str = "session_date",
) -> pd.DataFrame:
    """PIT-filter a price dataframe down to one row per ticker.

    Parameters
    ----------
    df         : dataframe with at minimum ``ticker_col``, ``date_col``,
                 ``price_col`` columns. May contain extra columns.
    as_of      : inclusive upper bound on ``session_date``. No row with
                 ``session_date > as_of`` is ever returned.
    universe   : candidate tickers. Tickers absent from ``df`` (or whose
                 price is NaN / non-positive) are silently dropped.
    price_col / ticker_col / date_col : column-name overrides.

    Returns
    -------
    A dataframe with one row per surviving ticker, columns
    ``{ticker_col, date_col, price_col}``, sorted lexicographically by
    ticker. The returned frame is a fresh copy (no aliasing).

    Invariants enforced
    -------------------
    >>> out = pit_filter(df, as_of=T, universe=U)
    >>> assert (out[date_col] <= T).all()
    >>> assert (out[price_col] > 0).all()
    >>> assert out[ticker_col].is_unique
    >>> assert set(out[ticker_col]).issubset(set(U))
    """
    if df.empty:
        return pd.DataFrame(columns=[ticker_col, date_col, price_col])
    as_of_ts = pd.Timestamp(as_of).normalize() if _is_date_like(as_of) else pd.Timestamp(as_of)
    u = set(universe)
    # Vectorized PIT filter.
    w = df[df[ticker_col].isin(u)].copy()
    w[date_col] = pd.to_datetime(w[date_col])
    w = w[w[date_col] <= as_of_ts]
    w = w[w[price_col].notna() & (w[price_col] > 0)]
    if w.empty:
        return pd.DataFrame(columns=[ticker_col, date_col, price_col])
    # Latest row per ticker (within the PIT window).
    w = w.sort_values([ticker_col, date_col])
    idx = w.groupby(ticker_col, sort=False)[date_col].idxmax()
    out = w.loc[idx, [ticker_col, date_col, price_col]].copy()
    out = out.sort_values(ticker_col).reset_index(drop=True)
    return out


def _is_date_like(x: Any) -> bool:
    import datetime as _dt

    return isinstance(x, (_dt.date,)) and not isinstance(x, _dt.datetime)


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class SnapshotSource(Protocol):
    """Anything that can produce a ``MarketSnapshot`` at a given session.

    Concrete impls: ``DuckDBSnapshotLoader``, ``DataFrameSnapshotLoader``.
    User code takes a ``SnapshotSource`` parameter and stays backend-
    agnostic.
    """

    def load(
        self,
        as_of: pd.Timestamp,
        universe: tuple[str, ...],
    ) -> MarketSnapshot: ...


# ---------------------------------------------------------------------------
# DuckDB loader (production)
# ---------------------------------------------------------------------------
@dataclass
class DuckDBSnapshotLoader:
    """Production loader over a ``quantdata`` DuckDB file.

    Expected table shape (configurable column names)::

        CREATE TABLE daily_bars_adj (
            ticker        VARCHAR,
            session_date  DATE,
            open          DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            close         DOUBLE,
            volume        BIGINT,
            ...
        );

    The loader issues one parameterized query per ``load`` call. For
    large universes (≥ 10k tickers) consider pre-materializing a narrow
    PIT view in ``quantdata`` and pointing this loader at it.
    """

    db_path: str | Path
    price_table: str = "daily_bars_adj"
    price_field: str = "close"
    ticker_col: str = "ticker"
    date_col: str = "session_date"
    read_only: bool = True

    _con: Any = field(default=None, init=False, repr=False)
    _duckdb: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError as e:  # pragma: no cover — exercised only when dep missing
            raise ImportError(
                "DuckDBSnapshotLoader requires the `duckdb` package. "
                "Install via `pip install duckdb`."
            ) from e
        self._duckdb = duckdb
        self.db_path = Path(self.db_path)

    def _connect(self) -> Any:
        if self._con is None:
            self._con = self._duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "DuckDBSnapshotLoader":
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def load(
        self,
        as_of: pd.Timestamp,
        universe: tuple[str, ...],
    ) -> MarketSnapshot:
        if not universe:
            raise ValueError("universe must be non-empty.")
        con = self._connect()
        # CTE + row_number() picks the latest session_date per ticker within
        # the PIT window, then filters to the frontier row.
        placeholders = ",".join(["?"] * len(universe))
        q = f"""
        WITH win AS (
          SELECT {self.ticker_col} AS ticker,
                 {self.price_field} AS price,
                 CAST({self.date_col} AS DATE) AS session_date,
                 row_number() OVER (
                     PARTITION BY {self.ticker_col}
                     ORDER BY {self.date_col} DESC
                 ) AS rn
            FROM {self.price_table}
           WHERE {self.ticker_col} IN ({placeholders})
             AND {self.date_col} <= ?
             AND {self.price_field} IS NOT NULL
             AND {self.price_field} > 0
        )
        SELECT ticker, session_date, price
          FROM win
         WHERE rn = 1
         ORDER BY ticker;
        """
        params = [*universe, pd.Timestamp(as_of).date()]
        df = con.execute(q, params).fetch_df()
        return _dataframe_to_snapshot(df, as_of=as_of, source=str(self.db_path))


# ---------------------------------------------------------------------------
# In-memory loader (testing / small-replay)
# ---------------------------------------------------------------------------
@dataclass
class DataFrameSnapshotLoader:
    """Backend-free loader over a pre-built pandas DataFrame.

    Accepts any frame with columns ``{ticker, session_date, price}`` (or
    user-provided overrides). The PIT-filter logic is shared with
    ``DuckDBSnapshotLoader`` via ``pit_filter``, so test coverage here
    proves the production code's correctness.
    """

    prices: pd.DataFrame
    price_col: str = "price"
    ticker_col: str = "ticker"
    date_col: str = "session_date"

    def load(
        self,
        as_of: pd.Timestamp,
        universe: tuple[str, ...],
    ) -> MarketSnapshot:
        if not universe:
            raise ValueError("universe must be non-empty.")
        df = pit_filter(
            self.prices,
            as_of=as_of,
            universe=universe,
            price_col=self.price_col,
            ticker_col=self.ticker_col,
            date_col=self.date_col,
        )
        # Normalize column names for the shared materializer.
        df = df.rename(
            columns={
                self.ticker_col: "ticker",
                self.date_col: "session_date",
                self.price_col: "price",
            }
        )
        return _dataframe_to_snapshot(df, as_of=as_of, source="DataFrame")


# ---------------------------------------------------------------------------
# Shared materialization
# ---------------------------------------------------------------------------
def _dataframe_to_snapshot(df: pd.DataFrame, *, as_of: pd.Timestamp, source: str) -> MarketSnapshot:
    """Turn a validated 3-column frame into a ``MarketSnapshot``.

    Raises ``ValueError`` if zero rows survived — better to fail loudly
    than return an empty book the rebalance engine will silently no-op.
    """
    if df.empty:
        raise ValueError(
            f"SnapshotLoader: no priced tickers at as_of={pd.Timestamp(as_of)}. "
            "Check PIT window, universe overlap, and table name."
        )
    tickers = tuple(df["ticker"].tolist())
    prices = df["price"].to_numpy(dtype=np.float64)
    return MarketSnapshot(
        timestamp=pd.Timestamp(as_of).isoformat(),
        tickers=tickers,
        prices=prices,
        metadata={
            "source": source,
            "n_tickers": int(len(tickers)),
            "as_of": pd.Timestamp(as_of).isoformat(),
        },
    )


__all__ = [
    "DataFrameSnapshotLoader",
    "DuckDBSnapshotLoader",
    "SnapshotSource",
    "pit_filter",
]
