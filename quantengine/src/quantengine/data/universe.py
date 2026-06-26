"""Session-date-indexed universe resolution.

Why this exists
---------------
A universe defined as "today's S&P 500" carries survivorship bias: the
backtest pretends you knew which firms would still be in the index at
session :math:`T`, when you only knew the membership on ``T``'s date.

Correct practice: maintain a ``universe_membership`` table in
``quantdata`` with rows ``(session_date, ticker, member=TRUE/FALSE)``,
then resolve the universe as ``{ticker : session_date = T, member}``.
This module is the thin adapter that performs that lookup.

Contract
--------
Given a ``session_date``, return the ``tuple[str, ...]`` of tickers
considered eligible at that session. Ordering is lexicographic and
deterministic — the same input always returns the same tuple, which
matters because the tuple flows into ``MarketSnapshot.tickers`` and
``AlphaSignal.tickers`` downstream.

Sources
-------
Two implementations share the ``UniverseSource`` Protocol:

- ``DuckDBUniverseResolver`` — queries a membership table in quantdata.
- ``DataFrameUniverseResolver`` — in-memory, for tests and small replays.

Point-in-time correctness
-------------------------
As with prices, we insist on ``membership_date <= session_date``. For a
daily-snapshot table this is a trivial equality; for an
effective-range-style table you can extend the resolver to filter with
``valid_from <= T < valid_to``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class UniverseSource(Protocol):
    """Resolves ``session_date → tuple[str, ...]`` of eligible tickers."""

    def resolve(self, as_of: pd.Timestamp) -> tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# DataFrame backend
# ---------------------------------------------------------------------------
@dataclass
class DataFrameUniverseResolver:
    """In-memory universe membership.

    Accepts any dataframe with columns ``{session_date, ticker}`` — each
    row means "``ticker`` is a member on ``session_date``". An optional
    ``member`` column (bool) lets you model deletions without dropping
    rows; absent, every row is assumed to be ``member=True``.
    """

    membership: pd.DataFrame
    ticker_col: str = "ticker"
    date_col: str = "session_date"
    member_col: str = "member"

    def resolve(self, as_of: pd.Timestamp) -> tuple[str, ...]:
        df = self.membership
        if df.empty:
            return ()
        as_of_ts = pd.Timestamp(as_of).normalize()
        dates = pd.to_datetime(df[self.date_col]).dt.normalize()
        mask = dates == as_of_ts
        if self.member_col in df.columns:
            mask &= df[self.member_col].astype(bool)
        tickers = sorted(set(df.loc[mask, self.ticker_col].astype(str).tolist()))
        return tuple(tickers)


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------
@dataclass
class DuckDBUniverseResolver:
    """Resolves universe membership from a DuckDB table.

    Expected table shape::

        CREATE TABLE universe_membership (
            session_date DATE,
            ticker       VARCHAR,
            member       BOOLEAN DEFAULT TRUE
        );

    Column names are configurable. A ``member`` column is optional;
    absent, all rows are treated as members.
    """

    db_path: str | Path
    table: str = "universe_membership"
    ticker_col: str = "ticker"
    date_col: str = "session_date"
    member_col: str = "member"  # nullable — we feature-detect below
    read_only: bool = True

    _duckdb: Any = field(default=None, init=False, repr=False)
    _con: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import duckdb  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("DuckDBUniverseResolver requires the `duckdb` package.") from e
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

    def __enter__(self) -> "DuckDBUniverseResolver":
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _has_member_col(self) -> bool:
        cols = self._connect().execute(f"SELECT * FROM {self.table} LIMIT 0").description
        return any(c[0] == self.member_col for c in cols)

    def resolve(self, as_of: pd.Timestamp) -> tuple[str, ...]:
        con = self._connect()
        member_pred = f" AND {self.member_col} = TRUE " if self._has_member_col() else ""
        q = f"""
        SELECT DISTINCT {self.ticker_col} AS ticker
          FROM {self.table}
         WHERE {self.date_col} = ?
         {member_pred}
         ORDER BY ticker
        """
        df = con.execute(q, [pd.Timestamp(as_of).date()]).fetch_df()
        return tuple(df["ticker"].astype(str).tolist())


__all__ = [
    "DataFrameUniverseResolver",
    "DuckDBUniverseResolver",
    "UniverseSource",
]
