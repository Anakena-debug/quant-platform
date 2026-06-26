"""Item 10a (prices half) — the survfree OHLCV is reproducible THROUGH quantlake.store, byte-identical.

A real parity oracle (the bars-module lesson): stated tolerance = OHLCV byte-identical after float64
normalization, no statistical slack. Skips when the local (gitignored) panel is absent.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quantlake.store.bitemporal import BitemporalStore

PANEL = (
    Path(__file__).resolve().parents[2]
    / "alpha_R"
    / "outputs"
    / "survfree"
    / "databento_xs_panel.parquet"
)


@pytest.mark.skipif(
    not PANEL.exists(), reason="survfree panel not present (gitignored, local-only)"
)
def test_survfree_ohlcv_round_trips_byte_identical():
    con = duckdb.connect()
    tickers = [
        r[0] for r in con.execute(f"SELECT DISTINCT ticker FROM '{PANEL}' LIMIT 5").fetchall()
    ]
    src = con.execute(
        f"SELECT ticker, date, open, high, low, close, volume FROM '{PANEL}' "
        f"WHERE ticker IN ({','.join('?' * len(tickers))}) ORDER BY ticker, date",
        tickers,
    ).df()
    assert not src.empty

    # Map ticker -> synthetic quantlake_id; load into the store (knowledge_date = session date).
    ids = {t: i + 1 for i, t in enumerate(tickers)}
    load = src.assign(
        quantlake_id=src["ticker"].map(ids),
        event_date=pd.to_datetime(src["date"]),
        knowledge_date=pd.to_datetime(src["date"]),
    ).drop(columns=["ticker", "date"])

    s = BitemporalStore()
    s.append("prices", load)
    got = s.as_of("prices", load["event_date"].max())

    # Compare OHLCV per (quantlake_id, event_date) — byte-identical after float64 normalization.
    cols = ["open", "high", "low", "close", "volume"]
    left = load.sort_values(["quantlake_id", "event_date"]).reset_index(drop=True)
    right = got.sort_values(["quantlake_id", "event_date"]).reset_index(drop=True)
    assert len(left) == len(right)
    for c in cols:
        pd.testing.assert_series_equal(
            left[c].astype("float64"), right[c].astype("float64"), check_names=False
        )
