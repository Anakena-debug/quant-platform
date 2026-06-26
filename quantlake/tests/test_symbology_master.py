"""s82 — symbology bridge + unified master: the FIXTURE SENTINEL (heal mechanism, always runs).

One quantlake_id per entity, resolvable via ric AND ticker AND databento_iid; LSEG rows and Databento
prices land under the SAME id, so the PEAD-style join (announcement -> forward return) — impossible
under the s81 forked id-spaces — works through the healed master.
"""

from __future__ import annotations

import pandas as pd

from quantlake.ingest.lseg import build_ric_mapping, map_instruments, to_lake_rows
from quantlake.ingest.symbology import (
    build_unified_master,
    map_databento_prices,
    parse_symbology_json,
)
from quantlake.store.bitemporal import BitemporalStore
from quantlake.universe.security_master import SecurityMaster

PAYLOAD = {
    "result": {
        "12345": [{"d0": "2023-03-28", "d1": "2026-06-06", "s": "AAPL"}],
        "67890": [
            {"d0": "2023-03-28", "d1": "2024-06-01", "s": "OLDCO"},
            {"d0": "2024-06-01", "d1": "2026-06-06", "s": "NEWCO"},  # rename, same instrument
        ],
        "11111": [{"d0": "2023-03-28", "d1": "2026-06-06", "s": None}],  # no bridge info
    },
    "not_found": ["99999"],
}
RICS = ["AAPL.OQ", "GONE.N"]  # GONE: no Databento entity -> fresh mint, counted


def _master():
    sm = SecurityMaster(BitemporalStore())
    um = build_unified_master(sm, parse_symbology_json(PAYLOAD), RICS)
    return sm, um


def test_parse_symbology_intervals():
    df = parse_symbology_json(PAYLOAD)
    assert len(df) == 3  # AAPL + 2 OLDCO/NEWCO intervals; the None symbol is skipped
    assert set(df["ticker"]) == {"AAPL", "OLDCO", "NEWCO"}


def test_one_id_three_identifier_types():
    sm, um = _master()
    qid = um.ric_to_id["AAPL.OQ"]
    assert um.iid_to_id[12345] == qid  # ONE entity across both vendors
    assert sm.resolve("AAPL.OQ", "ric", "2024-06-01") == qid
    assert sm.resolve("AAPL", "ticker", "2024-06-01") == qid
    assert sm.resolve("12345", "databento_iid", "2024-06-01") == qid
    # rename keeps one id; resolution is date-ranged
    r = um.iid_to_id[67890]
    assert sm.resolve("OLDCO", "ticker", "2024-01-15") == r
    assert sm.resolve("NEWCO", "ticker", "2025-01-15") == r
    # stats: GONE.N minted fresh and COUNTED, never silent
    assert um.lseg_only_rics == ["GONE.N"] and um.n_rics_matched == 1
    assert um.ticker_overlap_anomalies == 0


def test_pead_join_works_through_healed_master():
    sm, um = _master()
    s = sm.store
    # LSEG earnings row via the UNCHANGED s81 ingest path (UnifiedMaster is RicMapping-compatible)
    raw = pd.DataFrame(
        {
            "Instrument": ["AAPL.OQ"],
            "Period End Date": ["2024-03-31"],
            "Report Date": ["2024-05-02"],
            "SUE": [2.5],
        }
    )
    mapped, quar, cov = map_instruments(raw, um)
    assert quar.empty and cov == 1.0
    s.append("lseg_earnings_surprise", to_lake_rows(mapped, "lseg_earnings_surprise", ["SUE"]))
    # Databento prices keyed by instrument_id land under the SAME quantlake_id
    panel = pd.DataFrame(
        {
            "ticker": [12345, 12345, 99999],  # 99999 unmapped -> quarantined
            "date": ["2024-05-01", "2024-05-03", "2024-05-03"],
            "close": [100.0, 104.0, 1.0],
        }
    )
    rows, pq, pcov = map_databento_prices(panel, um.iid_to_id)
    assert len(pq) == 1 and abs(pcov - 2 / 3) < 1e-9  # quarantined, not dropped
    s.append("prices", rows)
    # THE JOIN THE FORK MADE IMPOSSIBLE: announcement -> forward return, one id-space
    ev = s.as_of("lseg_earnings_surprise", "2024-06-01").iloc[0]
    px = s.as_of("prices", "2024-06-01")
    fwd = px[(px["quantlake_id"] == ev["quantlake_id"]) & (px["event_date"] > ev["event_date"])]
    assert len(fwd) == 1 and float(fwd.iloc[0]["close"]) == 104.0


def test_s81_path_is_the_fork_zero_shared_ids():
    # The forked construction, demonstrated: standalone RIC mint shares NO ids with the iid space.
    sm = SecurityMaster(BitemporalStore())
    um = build_unified_master(sm, parse_symbology_json(PAYLOAD), [])  # databento entities only
    forked = build_ric_mapping(RICS, sm, "2008-01-01")  # s81 standalone mint, same master
    assert set(forked.ric_to_id.values()).isdisjoint(set(um.iid_to_id.values()))
