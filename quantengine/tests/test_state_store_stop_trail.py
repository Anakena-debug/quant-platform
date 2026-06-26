"""s73 — DuckDBStore persists STOP/TRAIL trigger fields and migrates legacy DBs.

The DuckDB ``orders`` table previously stored only ``limit_price``; a saved
STOP/STOP_LIMIT/TRAIL/TRAIL_LIMIT order dropped its trigger fields. s73 appends
``stop_price`` / ``trail_amount`` / ``trail_percent`` / ``limit_offset`` LAST in
the DDL + ``_ORDERS_COLS`` and ALTER-adds them in ``_MIGRATE`` (DuckDB appends
new columns last), so fresh and migrated DBs share one column order — the
invariant the positional ``INSERT ... SELECT *`` in ``save_run`` depends on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantengine.contracts.orders import Order, OrderType
from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState
from quantengine.runtime.state_store import DuckDBStore

pytest.importorskip("duckdb")


def _ledger_with_stop_and_trail() -> Ledger:
    led = Ledger()
    led.append(
        "t",
        "ORDER_SUBMITTED",
        Order.new("AAPL", -100, OrderType.STOP, stop_price=95.0, timestamp="t"),
    )
    led.append(
        "t",
        "ORDER_SUBMITTED",
        Order.new("MSFT", -50, OrderType.TRAIL, trail_percent=3.0, timestamp="t"),
    )
    led.append(
        "t",
        "ORDER_SUBMITTED",
        Order.new(
            "NVDA", 10, OrderType.TRAIL_LIMIT, trail_amount=1.5, limit_offset=0.25, timestamp="t"
        ),
    )
    return led


def test_save_load_preserves_stop_trail_columns(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "runs.duckdb")
    rid = store.save_run(PortfolioState.empty(1_000_000.0), _ledger_with_stop_and_trail())
    orders = store.load_run(rid)["orders"].set_index("ticker")
    store.close()

    assert orders.loc["AAPL", "stop_price"] == 95.0
    assert orders.loc["MSFT", "trail_percent"] == 3.0
    assert orders.loc["NVDA", "trail_amount"] == 1.5
    assert orders.loc["NVDA", "limit_offset"] == 0.25
    # Trail row carries no fixed stop; STOP row carries no trail distance — NULL -> NaN.
    assert pd.isna(orders.loc["MSFT", "stop_price"])
    assert pd.isna(orders.loc["AAPL", "trail_percent"])


def test_migrate_adds_columns_to_legacy_orders_table(tmp_path: Path) -> None:
    """A pre-s73 ``orders`` table (no stop/trail columns) must be ALTER-migrated on
    connect AND remain writable: the positional INSERT has to align against the
    ALTER-appended columns, then load_run reads the values back."""
    import duckdb

    path = tmp_path / "legacy.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE orders ("
        "run_id VARCHAR, seq BIGINT, timestamp VARCHAR, order_id VARCHAR, ticker VARCHAR, "
        "side VARCHAR, quantity INTEGER, order_type VARCHAR, limit_price DOUBLE, "
        "parent_signal_ts VARCHAR, metadata VARCHAR, PRIMARY KEY (run_id, seq))"
    )
    con.close()

    store = DuckDBStore(path)
    # connect()/save_run trigger _MIGRATE (ADD COLUMN IF NOT EXISTS) then a positional insert.
    rid = store.save_run(PortfolioState.empty(1_000_000.0), _ledger_with_stop_and_trail())
    orders = store.load_run(rid)["orders"].set_index("ticker")
    con = store.connect()
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'orders'"
        ).fetchall()
    }
    store.close()

    assert {"stop_price", "trail_amount", "trail_percent", "limit_offset"} <= cols
    # and the positional INSERT landed the values in the right (ALTER-appended) columns
    assert orders.loc["AAPL", "stop_price"] == 95.0
    assert orders.loc["MSFT", "trail_percent"] == 3.0
    assert orders.loc["NVDA", "limit_offset"] == 0.25
