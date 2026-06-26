"""s81 — LSEG ingest: melt spot-check (REQ5) + map->bitemporal-rows round-trip."""

from __future__ import annotations

import pandas as pd

from quantlake.ingest.lseg import build_ric_mapping, map_instruments, melt_daily_panel, to_lake_rows
from quantlake.store.bitemporal import BitemporalStore
from quantlake.universe.security_master import SecurityMaster

KD = "2008-01-01"


def _wide_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame(
        {
            "Date": dates,
            "('A.N', 'Price Close')": [10.0, 11.0, 12.0],
            "('A.N', 'Total Return')": [0.0, 0.1, 0.09],
            "('A.N', 'Volume')": [100, 110, 120],
            "('A.N', 'Company Market Cap')": [1e9, 1.1e9, 1.2e9],
            "('AAPL.OQ', 'Price Close')": [50.0, 51.0, 52.0],
            "('AAPL.OQ', 'Total Return')": [0.0, 0.02, 0.0196],
            "('AAPL.OQ', 'Volume')": [200, 210, 220],
            "('AAPL.OQ', 'Company Market Cap')": [2e12, 2.1e12, 2.2e12],
        }
    )


def test_melt_spot_check_cells_against_wide():
    wide = _wide_panel()
    long = melt_daily_panel(wide)
    assert set(long["ric"]) == {"A.N", "AAPL.OQ"}
    assert len(long) == 2 * 3  # 2 rics x 3 dates

    # spot-check sampled cells against the wide original (no trust-me reshape)
    def cell(ric, label, row):
        return wide[f"('{ric}', '{label}')"].iloc[row]

    a = long[(long["ric"] == "AAPL.OQ")].sort_values("event_date").reset_index(drop=True)
    assert a["price_close"].iloc[2] == cell("AAPL.OQ", "Price Close", 2) == 52.0
    assert a["market_cap"].iloc[0] == cell("AAPL.OQ", "Company Market Cap", 0) == 2e12
    n = long[(long["ric"] == "A.N")].sort_values("event_date").reset_index(drop=True)
    assert n["volume"].iloc[1] == cell("A.N", "Volume", 1) == 110


def test_ingest_fundamentals_to_bitemporal_rows():
    s = BitemporalStore()
    sm = SecurityMaster(s)
    m = build_ric_mapping(["AAPL.OQ"], sm, KD)
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ"],
            "Period End Date": ["2024-03-31"],
            "Report Date": ["2024-05-02"],  # released after period end (kd > event)
            "Revenue": [123.0],
        }
    )
    mapped, quarantine, cov = map_instruments(raw, m)
    assert quarantine.empty and cov == 1.0
    rows = to_lake_rows(mapped, "lseg_fundamentals", ["Revenue"])
    s.append("lseg_fundamentals", rows)
    # not visible before the report date; visible after (kd binding works)
    assert s.as_of("lseg_fundamentals", "2024-04-15").empty
    got = s.as_of("lseg_fundamentals", "2024-06-01")
    assert len(got) == 1 and float(got.iloc[0]["Revenue"]) == 123.0
