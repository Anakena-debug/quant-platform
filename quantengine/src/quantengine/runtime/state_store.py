"""DuckDB persistence for backtest runs.

Every ``ReplayRunner.run(..., store=DuckDBStore(path))`` call appends a
single "run" — the final ``PortfolioState`` summary plus all
``Ledger`` events — into a DuckDB file. Multiple runs live in the same
file, distinguished by ``run_id`` (a UUID). Results are queryable in
standard SQL, directly from a notebook or the ``duckdb`` CLI.

Schema
------
::

    runs(run_id PK, started_at, initial_cash, final_cash, realized_pnl,
         total_commission, skipped_steps, n_events, journal_digest, metadata)
    orders(run_id, seq, timestamp, order_id, ticker, side, quantity,
           order_type, limit_price, parent_signal_ts, metadata,
           stop_price, trail_amount, trail_percent, limit_offset)
    fills (run_id, seq, timestamp, fill_id, order_id, ticker,
           signed_quantity, price, commission, metadata)
    positions(run_id, ticker, quantity, avg_cost)
    lifecycle_events(run_id, seq, timestamp, kind, payload_json)

    PK(orders)           = (run_id, seq)
    PK(fills)            = (run_id, seq)
    PK(positions)        = (run_id, ticker)
    PK(lifecycle_events) = (run_id, seq)

``lifecycle_events`` holds all non-submit / non-fill ledger events —
``ORDER_ACKED``, ``ORDER_WORKING``, ``ORDER_CANCELLED``,
``ORDER_REJECTED``, ``CASH_ADJ`` — with the payload serialized as JSON.
The hash-chain digest in ``runs.journal_digest`` is computed over the
*entire* ledger (all event kinds), so tamper evidence spans every row.

Design notes
------------
- Lazy ``import duckdb``: the module itself imports without the optional
  dependency. Only ``DuckDBStore(path)`` instantiation will raise
  ``ImportError`` — with a message pointing at
  ``pip install 'quantengine[persistence]'``.
- UUIDs stored as VARCHAR(36) for cross-version portability.
- Timestamps stored as VARCHAR (ISO-8601, matching ``MarketSnapshot``).
- Metadata stored as JSON strings (VARCHAR). Use
  ``json_extract(metadata, '$.key')`` in SQL for filtering.
- Bulk inserts via ``con.register("df", df)`` → fast columnar path.
- Single-writer per file. Backtests are offline; don't share a file
  across concurrent processes.
- Crash-during-run recovery is NOT handled here — ``save_run`` is called
  once, after the loop completes. For incremental checkpointing, see
  ``Phase 2.5`` note in ARCHITECTURE.md.

Usage
-----
::

    from quantengine.runtime.state_store import DuckDBStore

    with DuckDBStore("backtests/runs.duckdb") as store:
        final = replay.run(clock, "prices.parquet", store=store,
                           initial_cash=1e6,
                           metadata={"strategy_version": "v3.2.1"})

    # Query back:
    df = store.list_runs()
    run = store.load_run(run_id)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

import pandas as pd

from quantengine.audit.journal import chain_digest
from quantengine.contracts.orders import Fill, Order
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Protocol: what ReplayRunner expects of a store
# ---------------------------------------------------------------------------
@runtime_checkable
class RunStore(Protocol):
    """Structural contract for any backtest persistence backend.

    ``ReplayRunner`` only calls ``save_run``; everything else (list, load,
    delete, custom analytics) is store-specific.
    """

    def save_run(
        self,
        state: PortfolioState,
        ledger: Ledger,
        *,
        run_id: UUID | None = ...,
        initial_cash: float | None = ...,
        skipped_steps: int = ...,
        metadata: dict[str, Any] | None = ...,
    ) -> UUID: ...


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           VARCHAR PRIMARY KEY,
    started_at       TIMESTAMP,
    initial_cash     DOUBLE,
    final_cash       DOUBLE,
    realized_pnl     DOUBLE,
    total_commission DOUBLE,
    skipped_steps    INTEGER,
    n_events         INTEGER,
    journal_digest   VARCHAR,
    metadata         VARCHAR
);

CREATE TABLE IF NOT EXISTS orders (
    run_id            VARCHAR,
    seq               BIGINT,
    timestamp         VARCHAR,
    order_id          VARCHAR,
    ticker            VARCHAR,
    side              VARCHAR,
    quantity          INTEGER,
    order_type        VARCHAR,
    limit_price       DOUBLE,
    parent_signal_ts  VARCHAR,
    metadata          VARCHAR,
    stop_price        DOUBLE,
    trail_amount      DOUBLE,
    trail_percent     DOUBLE,
    limit_offset      DOUBLE,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS fills (
    run_id           VARCHAR,
    seq              BIGINT,
    timestamp        VARCHAR,
    fill_id          VARCHAR,
    order_id         VARCHAR,
    ticker           VARCHAR,
    signed_quantity  INTEGER,
    price            DOUBLE,
    commission       DOUBLE,
    metadata         VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS positions (
    run_id     VARCHAR,
    ticker     VARCHAR,
    quantity   INTEGER,
    avg_cost   DOUBLE,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    run_id       VARCHAR,
    seq          BIGINT,
    timestamp    VARCHAR,
    kind         VARCHAR,
    payload_json VARCHAR,
    PRIMARY KEY (run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_orders_run_ticker     ON orders(run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_fills_run_ticker      ON fills(run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_lifecycle_run_kind    ON lifecycle_events(run_id, kind);
"""

# Idempotent migration step: DuckDB files that existed before a column was
# introduced get it backfilled as NULL. Safe to re-run on every connect.
# s73: the stop/trail trigger columns are ALTER-appended here (DuckDB appends new
# columns last) AND listed last in the CREATE TABLE + _ORDERS_COLS, so fresh and
# migrated DBs share one column order — required by the positional
# `INSERT ... SELECT *` in save_run.
_MIGRATE = """
ALTER TABLE runs ADD COLUMN IF NOT EXISTS journal_digest VARCHAR;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS stop_price DOUBLE;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS trail_amount DOUBLE;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS trail_percent DOUBLE;
ALTER TABLE orders ADD COLUMN IF NOT EXISTS limit_offset DOUBLE;
"""


# Column order must match DDL insertion order when we do SELECT * FROM registered df.
_ORDERS_COLS = [
    "run_id",
    "seq",
    "timestamp",
    "order_id",
    "ticker",
    "side",
    "quantity",
    "order_type",
    "limit_price",
    "parent_signal_ts",
    "metadata",
    # s73: appended last to match the DDL + ALTER column order (positional INSERT).
    "stop_price",
    "trail_amount",
    "trail_percent",
    "limit_offset",
]
_FILLS_COLS = [
    "run_id",
    "seq",
    "timestamp",
    "fill_id",
    "order_id",
    "ticker",
    "signed_quantity",
    "price",
    "commission",
    "metadata",
]
_POSITIONS_COLS = ["run_id", "ticker", "quantity", "avg_cost"]
_LIFECYCLE_COLS = ["run_id", "seq", "timestamp", "kind", "payload_json"]

# Event kinds that map onto dedicated tables; everything else lands in
# ``lifecycle_events``.
_DEDICATED_KINDS = frozenset({"ORDER_SUBMITTED", "ORDER_FILLED"})


# ---------------------------------------------------------------------------
# DuckDBStore
# ---------------------------------------------------------------------------
@dataclass
class DuckDBStore:
    """Append-only DuckDB backend for backtest runs.

    Construction lazily imports ``duckdb``. Use as a context manager or
    call ``close()`` explicitly.
    """

    path: str | Path
    read_only: bool = False

    def __post_init__(self) -> None:
        try:
            import duckdb  # noqa: F401  (checked at construction)
        except ImportError as e:  # pragma: no cover — exercised only when dep missing
            raise ImportError(
                "DuckDBStore requires the `duckdb` package. Install via "
                "`pip install 'quantengine[persistence]'` or `pip install duckdb`."
            ) from e
        self._duckdb = duckdb
        self.path = Path(self.path)
        self._con: Any = None

    # ---- connection lifecycle ------------------------------------------
    def connect(self) -> Any:
        if self._con is None:
            self._con = self._duckdb.connect(str(self.path), read_only=self.read_only)
            if not self.read_only:
                self._init_schema()
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "DuckDBStore":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._con.execute(_DDL)
        # Best-effort migration for files pre-dating the journal_digest
        # column. ``ADD COLUMN IF NOT EXISTS`` is a no-op on fresh DBs
        # and a one-shot ALTER on legacy ones.
        try:
            self._con.execute(_MIGRATE)
        except Exception:  # pragma: no cover — older duckdb without IF NOT EXISTS
            # Fall through: if ``ALTER TABLE`` is unsupported, writes will
            # fail loudly below and the user can upgrade duckdb.
            pass

    # ---- write API -----------------------------------------------------
    def save_run(
        self,
        state: PortfolioState,
        ledger: Ledger,
        *,
        run_id: UUID | None = None,
        initial_cash: float | None = None,
        skipped_steps: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Persist one backtest run. Returns the run_id used.

        The hash-chain digest over every ledger event is computed once
        (``chain_digest``) and stored in ``runs.journal_digest``, giving
        callers a single scalar to compare against a re-run for tamper
        evidence.
        """
        con = self.connect()
        rid = run_id or uuid4()
        rid_str = str(rid)

        orders_df, fills_df, lifecycle_df = self._events_to_dataframes(ledger, rid_str)
        positions_df = self._positions_to_dataframe(state, rid_str)
        digest = chain_digest(ledger.events()).digest

        con.execute("BEGIN TRANSACTION;")
        try:
            con.execute(
                """
                INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rid_str,
                    pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None),
                    float(initial_cash) if initial_cash is not None else None,
                    float(state.cash),
                    float(state.realized_pnl),
                    float(state.total_commission),
                    int(skipped_steps),
                    len(ledger),
                    digest,
                    json.dumps(metadata or {}, default=str),
                ],
            )
            self._bulk_insert(con, "orders", orders_df, _ORDERS_COLS)
            self._bulk_insert(con, "fills", fills_df, _FILLS_COLS)
            self._bulk_insert(con, "positions", positions_df, _POSITIONS_COLS)
            self._bulk_insert(con, "lifecycle_events", lifecycle_df, _LIFECYCLE_COLS)
            con.execute("COMMIT;")
        except Exception:
            con.execute("ROLLBACK;")
            raise

        return rid

    @staticmethod
    def _bulk_insert(con: Any, table: str, df: pd.DataFrame, columns: list[str]) -> None:
        if df.empty:
            return
        # Ensure column order matches DDL order for SELECT *.
        df = df[columns]
        view_name = f"__qe_{table}_df"
        con.register(view_name, df)
        try:
            con.execute(f"INSERT INTO {table} SELECT * FROM {view_name}")
        finally:
            con.unregister(view_name)

    @staticmethod
    def _events_to_dataframes(
        ledger: Ledger, run_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the ledger into three dataframes:

        - ``orders`` (ORDER_SUBMITTED with Order payload)
        - ``fills``  (ORDER_FILLED with Fill payload)
        - ``lifecycle_events`` (everything else: ACKED / WORKING / CANCELLED /
          REJECTED / CASH_ADJ; payload JSON-serialized)
        """
        orders: list[dict[str, Any]] = []
        fills: list[dict[str, Any]] = []
        lifecycle: list[dict[str, Any]] = []
        for e in ledger.events():
            if e.kind == "ORDER_SUBMITTED" and isinstance(e.payload, Order):
                o = e.payload
                orders.append(
                    {
                        "run_id": run_id,
                        "seq": e.seq,
                        "timestamp": e.timestamp,
                        "order_id": str(o.order_id),
                        "ticker": o.ticker,
                        "side": o.side.value,
                        "quantity": int(o.quantity),
                        "order_type": o.order_type.value,
                        "limit_price": (
                            float(o.limit_price) if o.limit_price is not None else None
                        ),
                        "parent_signal_ts": o.parent_signal_ts,
                        "metadata": json.dumps(o.metadata, default=str),
                        # s73: stop/trail trigger fields (None for non-stop orders).
                        "stop_price": (float(o.stop_price) if o.stop_price is not None else None),
                        "trail_amount": (
                            float(o.trail_amount) if o.trail_amount is not None else None
                        ),
                        "trail_percent": (
                            float(o.trail_percent) if o.trail_percent is not None else None
                        ),
                        "limit_offset": (
                            float(o.limit_offset) if o.limit_offset is not None else None
                        ),
                    }
                )
            elif e.kind == "ORDER_FILLED" and isinstance(e.payload, Fill):
                f = e.payload
                fills.append(
                    {
                        "run_id": run_id,
                        "seq": e.seq,
                        "timestamp": e.timestamp,
                        "fill_id": str(f.fill_id),
                        "order_id": str(f.order_id),
                        "ticker": f.ticker,
                        "signed_quantity": int(f.signed_quantity),
                        "price": float(f.price),
                        "commission": float(f.commission),
                        "metadata": json.dumps(f.metadata, default=str),
                    }
                )
            else:
                # Lifecycle / cash-adjustment events. Serialize the payload as
                # canonical JSON so queries like
                #   SELECT json_extract(payload_json, '$.reason') ...
                # work without further munging.
                payload = e.payload
                if isinstance(payload, Order):
                    payload_json = json.dumps(
                        {
                            "order_id": str(payload.order_id),
                            "ticker": payload.ticker,
                            "side": payload.side.value,
                            "quantity": int(payload.quantity),
                            "order_type": payload.order_type.value,
                        },
                        default=str,
                    )
                elif isinstance(payload, Fill):
                    payload_json = json.dumps(
                        {
                            "fill_id": str(payload.fill_id),
                            "order_id": str(payload.order_id),
                            "ticker": payload.ticker,
                            "signed_quantity": int(payload.signed_quantity),
                            "price": float(payload.price),
                            "commission": float(payload.commission),
                        },
                        default=str,
                    )
                else:
                    payload_json = json.dumps(payload, default=str, sort_keys=True)
                lifecycle.append(
                    {
                        "run_id": run_id,
                        "seq": e.seq,
                        "timestamp": e.timestamp,
                        "kind": str(e.kind),
                        "payload_json": payload_json,
                    }
                )
        return (
            pd.DataFrame(orders, columns=_ORDERS_COLS)
            if orders
            else pd.DataFrame(columns=_ORDERS_COLS),
            pd.DataFrame(fills, columns=_FILLS_COLS)
            if fills
            else pd.DataFrame(columns=_FILLS_COLS),
            pd.DataFrame(lifecycle, columns=_LIFECYCLE_COLS)
            if lifecycle
            else pd.DataFrame(columns=_LIFECYCLE_COLS),
        )

    @staticmethod
    def _positions_to_dataframe(state: PortfolioState, run_id: str) -> pd.DataFrame:
        rows = [
            {
                "run_id": run_id,
                "ticker": p.ticker,
                "quantity": int(p.quantity),
                "avg_cost": float(p.avg_cost),
            }
            for p in state.positions.values()
        ]
        return (
            pd.DataFrame(rows, columns=_POSITIONS_COLS)
            if rows
            else pd.DataFrame(columns=_POSITIONS_COLS)
        )

    # ---- read API ------------------------------------------------------
    def list_runs(self) -> pd.DataFrame:
        """Return all runs, newest first."""
        return self.connect().execute("SELECT * FROM runs ORDER BY started_at DESC").df()

    def load_run(self, run_id: UUID | str) -> dict[str, pd.DataFrame]:
        """Return {'run', 'orders', 'fills', 'positions', 'lifecycle'} frames."""
        rid = str(run_id)
        con = self.connect()
        return {
            "run": con.execute("SELECT * FROM runs WHERE run_id = ?", [rid]).df(),
            "orders": con.execute("SELECT * FROM orders WHERE run_id = ? ORDER BY seq", [rid]).df(),
            "fills": con.execute("SELECT * FROM fills WHERE run_id = ? ORDER BY seq", [rid]).df(),
            "positions": con.execute(
                "SELECT * FROM positions WHERE run_id = ? ORDER BY ticker", [rid]
            ).df(),
            "lifecycle": con.execute(
                "SELECT * FROM lifecycle_events WHERE run_id = ? ORDER BY seq", [rid]
            ).df(),
        }

    def delete_run(self, run_id: UUID | str) -> None:
        """Remove a run and all its rows. Transactional."""
        rid = str(run_id)
        con = self.connect()
        con.execute("BEGIN TRANSACTION;")
        try:
            for table in ("positions", "fills", "orders", "lifecycle_events", "runs"):
                con.execute(f"DELETE FROM {table} WHERE run_id = ?", [rid])
            con.execute("COMMIT;")
        except Exception:
            con.execute("ROLLBACK;")
            raise


__all__ = ["DuckDBStore", "RunStore"]
