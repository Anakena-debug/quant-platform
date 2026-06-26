"""Bitemporal store — the PIT spine of quantlake (s78).

Every table is ``(entity_keys…, event_date, knowledge_date, _ingest_seq, _schema_version, value…)``,
**append-only**. As-of reads resolve on KNOWLEDGE time: the latest ``knowledge_date <= as_of`` per
``(entity_keys, event_date)``, tie-broken by ``_ingest_seq`` (last-ingested wins on a same-kd
restatement). ``knowledge_date`` is RECONSTRUCTED historical availability (per :data:`TEMPORAL_POLICY`),
never system-arrival; ``_ingest_seq`` is system-arrival and is used ONLY as the same-kd tie-break.

The invariant (``tests/test_asof_property.py``), upstream of PurgedKFold: perturbing or appending rows
with ``knowledge_date > T`` never changes ``as_of(T)`` — and neither does it change ``adjusted_prices(T)``
or ``as_of_join(…, T)`` (those are the leak vectors a single-table check misses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

_SCHEMA_VERSION = 1

# Item 1 — per-table temporal policy. `reconstruction` = how kd is stamped for BACKFILLED history
# (reconstructed availability, NOT system-arrival). System-arrival is `_ingest_seq`.
TEMPORAL_POLICY: dict[str, dict[str, str]] = {
    "prices": {
        "event_date": "session date",
        "knowledge_date": "session-close date (vendor correction -> later kd)",
        "reconstruction": "backfill kd = the bar's session-close date",
    },
    "corp_actions": {
        "event_date": "ex-date",
        "knowledge_date": "detection date (a split is known once prices show it; kd >= ex)",
        "reconstruction": "backfill kd = first session the adjustment is detectable from raw prices",
    },
    "universe_membership": {
        "event_date": "effective date",
        "knowledge_date": "announcement date",
        "reconstruction": "backfill kd = announcement date; fallback effective_date - lag (disclosed)",
    },
    "instrument_status": {
        "event_date": "effective delist/suspension date",
        "knowledge_date": "announcement date (or = effective if unsourced)",
        "reconstruction": "backfill kd = announcement date; fallback = effective_date",
    },
    "security_master": {
        "event_date": "identifier valid_from",
        "knowledge_date": "date the mapping became known",
        "reconstruction": "backfill kd = valid_from",
    },
    "features": {
        "event_date": "the feature's reference date (the date the value describes)",
        "knowledge_date": "the as_of the feature was computed at",
        "reconstruction": "kd = as_of; computable at T from data visible at T (inherits the no-leak invariant)",
    },
    "lseg_fundamentals": {
        "event_date": "fiscal Period End Date",
        "knowledge_date": "Report Date (release proxy, NOT legal filing date)",
        "reconstruction": "kd = LSEG Report Date (max metric-specific report date per Instrument+Period)",
    },
    "lseg_earnings_surprise": {
        "event_date": "Report Date (PEAD — the announcement IS the event; intraday timestamp kept)",
        "knowledge_date": "Report Date (event = kd; SUE built as-of Snapshot <= Report)",
        "reconstruction": "kd = LSEG Report Date timestamp",
    },
    "lseg_ibes_consensus": {
        "event_date": "Snapshot Date (monthly Calc Date)",
        "knowledge_date": "Snapshot Date (event = kd = the date the consensus existed)",
        "reconstruction": "kd = LSEG Snapshot/Calc Date",
    },
    "lseg_daily_panel": {
        "event_date": "session date",
        "knowledge_date": "session date (daily availability)",
        "reconstruction": "kd = session date",
    },
}

# As-of partition = ENTITY_KEYS[table] + ["event_date"].
ENTITY_KEYS: dict[str, list[str]] = {
    "prices": ["quantlake_id"],
    "corp_actions": ["quantlake_id", "type"],
    "universe_membership": ["quantlake_id", "universe_name"],
    "instrument_status": ["quantlake_id"],
    "security_master": ["symbol", "symbol_type"],
    # s80: materialized features ARE bitemporal rows (kd = as_of) — inherit the no-leak invariant.
    "features": ["quantlake_id", "feature_name", "feature_version"],
    # s81: LSEG PIT overlays (current-SPX universe; see SOURCES.universe_basis = current_constituents).
    "lseg_fundamentals": ["quantlake_id"],
    "lseg_earnings_surprise": ["quantlake_id"],
    "lseg_ibes_consensus": ["quantlake_id"],
    "lseg_daily_panel": ["quantlake_id"],
}

_STAMP_COLS = ("_ingest_seq", "_schema_version")


@dataclass
class BitemporalStore:
    """DuckDB-backed bitemporal store. In-memory by default; pass a path for on-disk Parquet/DuckDB."""

    con: duckdb.DuckDBPyConnection = field(default_factory=lambda: duckdb.connect(":memory:"))

    # ---- write -----------------------------------------------------------
    def append(self, table: str, df: pd.DataFrame) -> None:
        """Append rows to a bitemporal ``table`` (created on first append).

        ``df`` carries the entity keys + ``event_date`` + ``knowledge_date`` + value columns. The store
        stamps ``_ingest_seq`` (monotonic system-arrival) and ``_schema_version``; callers must not.
        """
        if table not in ENTITY_KEYS:
            raise ValueError(
                f"unknown table {table!r}; register it in ENTITY_KEYS + TEMPORAL_POLICY"
            )
        if any(c in df.columns for c in _STAMP_COLS):
            raise ValueError(f"{table}: caller must not set {_STAMP_COLS} (store-stamped)")
        required = set(ENTITY_KEYS[table]) | {"event_date", "knowledge_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{table}: missing required columns {sorted(missing)}")
        if df.empty:
            return
        df = df.copy()
        for c in ("event_date", "knowledge_date"):
            df[c] = pd.to_datetime(df[c])
        # _ingest_seq is globally monotonic, derived from the PERSISTED max (not an in-memory counter),
        # so it survives close/reopen — the same-kd tie-break (and the PK) must never reset across
        # sessions (s79 commit 1; the bug an in-memory counter hid from single-process tests).
        start = self._next_ingest_seq(table)
        df["_ingest_seq"] = range(start, start + len(df))
        df["_schema_version"] = _SCHEMA_VERSION

        if table not in self._tables():
            self.con.register("_schema_df", df.head(0))
            self.con.execute(f"CREATE TABLE {table} AS SELECT * FROM _schema_df")
            self.con.unregister("_schema_df")
            pk = ", ".join([*ENTITY_KEYS[table], "event_date", "knowledge_date", "_ingest_seq"])
            self.con.execute(f"CREATE UNIQUE INDEX {table}__pk ON {table} ({pk})")
        self.con.register("_append_df", df)
        cols = ", ".join(df.columns)
        self.con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _append_df")
        self.con.unregister("_append_df")

    def _tables(self) -> set[str]:
        return {r[0] for r in self.con.execute("SHOW TABLES").fetchall()}

    def has_table(self, table: str) -> bool:
        return table in self._tables()

    def _next_ingest_seq(self, table: str) -> int:
        """Next ``_ingest_seq`` from the PERSISTED max (+1), or 0 for a not-yet-created table.

        Persistence-derived so the counter survives a close/reopen — never an in-memory reset.
        """
        if table not in self._tables():
            return 0
        row = self.con.execute(f"SELECT COALESCE(MAX(_ingest_seq), -1) + 1 FROM {table}").fetchone()
        return int(row[0]) if row is not None else 0

    def watermark(self, table: str, as_of) -> int:
        """Visible watermark: ``max(_ingest_seq)`` among rows with ``knowledge_date <= as_of`` (-1 if
        none). The exact feature-cache invalidation term (s80 D1) — a backfill (reconstructed
        ``kd <= as_of``, fresh ``_ingest_seq``) BUMPS it; a future-kd append (``kd > as_of``) is
        excluded, so it does NOT (no over-invalidation)."""
        if table not in self._tables():
            return -1
        row = self.con.execute(
            f"SELECT COALESCE(MAX(_ingest_seq), -1) FROM {table} WHERE knowledge_date <= ?",
            [pd.Timestamp(as_of)],
        ).fetchone()
        return int(row[0]) if row is not None else -1

    def pk_columns(self, table: str) -> list[str]:
        """The primary-key columns (includes knowledge_date AND _ingest_seq — item 1/B2)."""
        return [*ENTITY_KEYS[table], "event_date", "knowledge_date", "_ingest_seq"]

    # ---- read (as-of resolves on knowledge time) -------------------------
    def as_of(self, table: str, as_of) -> pd.DataFrame:
        """Latest ``knowledge_date <= as_of`` per ``(entity_keys, event_date)`` (tie-break _ingest_seq).

        Canonical output order (``entity_keys, event_date``) so equality checks are order-stable.
        """
        part = ", ".join([*ENTITY_KEYS[table], "event_date"])
        return self.con.execute(
            f"""
            SELECT * EXCLUDE (_rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY {part} ORDER BY knowledge_date DESC, _ingest_seq DESC) AS _rn
                FROM {table} WHERE knowledge_date <= ?
            ) WHERE _rn = 1
            ORDER BY {part}
            """,
            [pd.Timestamp(as_of)],
        ).df()

    def as_of_join(
        self,
        left: pd.DataFrame,
        table: str,
        *,
        on: str | list[str],
        as_of,
    ) -> pd.DataFrame:
        """Join ``left`` to the as-of snapshot of ``table`` on ``on`` — each left row sees only facts
        with ``knowledge_date <= as_of`` (the join is also leak-free, not just single-table as_of)."""
        right = self.as_of(table, as_of)
        keys = [on] if isinstance(on, str) else list(on)
        return left.merge(right, on=keys, how="left", suffixes=("", "_r"))

    # ---- adjusted prices (B5: only corp_actions with kd <= as_of) --------
    def adjusted_prices(self, as_of, *, price_col: str = "close") -> pd.DataFrame:
        """Split-adjusted prices as of ``as_of``.

        Back-adjustment uses ONLY ``corp_actions`` (splits) with ``knowledge_date <= as_of`` (B5): a
        naive all-splits view would retro-adjust prices visible BEFORE the split was detectable
        (detection-based kd >= ex), leaking the future split into past prices.
        """
        prices = self.as_of("prices", as_of)
        if prices.empty:
            prices["adj_factor"] = []
            prices[f"adj_{price_col}"] = []
            return prices
        if "corp_actions" in self._tables():
            ca = self.as_of("corp_actions", as_of)
            ca = ca[ca["type"] == "split"] if "type" in ca.columns else ca.iloc[0:0]
        else:
            ca = pd.DataFrame(columns=["quantlake_id", "event_date", "raw_factor"])
        prices = prices.sort_values(["quantlake_id", "event_date"]).reset_index(drop=True)
        prices["adj_factor"] = 1.0
        for qid, splits in ca.groupby("quantlake_id"):
            mask = prices["quantlake_id"] == qid
            if not mask.any():
                continue
            factor = pd.Series(1.0, index=prices.index[mask])
            # Standard back-adjustment: a split with ex-date D divides all prices STRICTLY BEFORE D.
            for ex_date, raw in zip(splits["event_date"], splits["raw_factor"]):
                before = prices.loc[mask, "event_date"] < pd.Timestamp(ex_date)
                factor.loc[before.index[before]] /= float(raw)
            prices.loc[mask, "adj_factor"] = factor.to_numpy()
        prices[f"adj_{price_col}"] = prices[price_col] * prices["adj_factor"]
        return prices


# Module-level convenience wrappers (the public surface re-exported from quantlake).
def as_of(store: BitemporalStore, table: str, as_of_ts) -> pd.DataFrame:  # noqa: A001 - public name
    return store.as_of(table, as_of_ts)


def as_of_join(
    store: BitemporalStore, left: pd.DataFrame, table: str, *, on: str | list[str], as_of_ts
) -> pd.DataFrame:
    return store.as_of_join(left, table, on=on, as_of=as_of_ts)


__all__ = [
    "ENTITY_KEYS",
    "TEMPORAL_POLICY",
    "BitemporalStore",
    "as_of",
    "as_of_join",
]
