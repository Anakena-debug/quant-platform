"""s81 REQ3 — RIC->quantlake_id: venue-suffix parsed (not discarded); unmapped -> quarantine, no silent drop."""

from __future__ import annotations

import pandas as pd

from quantlake.ingest.lseg import VENUE_BY_SUFFIX, build_ric_mapping, map_instruments, parse_ric
from quantlake.store.bitemporal import BitemporalStore
from quantlake.universe.security_master import SecurityMaster

KD = "2008-01-01"


def test_parse_ric_keeps_venue():
    assert parse_ric("AAPL.OQ") == ("AAPL", "OQ")
    assert parse_ric("A.N") == ("A", "N")
    assert parse_ric("NODOT") == ("NODOT", "")
    assert VENUE_BY_SUFFIX["OQ"] == "NASDAQ" and VENUE_BY_SUFFIX["N"] == "NYSE"


def test_build_mapping_registers_ric_and_ticker_root():
    sm = SecurityMaster(BitemporalStore())
    m = build_ric_mapping(["AAPL.OQ", "A.N"], sm, KD)
    assert set(m.ric_to_id) == {"AAPL.OQ", "A.N"} and not m.quarantine
    # full RIC (venue retained) AND ticker root both resolve to the same quantlake_id
    qid = m.ric_to_id["AAPL.OQ"]
    assert sm.resolve("AAPL.OQ", "ric", "2008-06-01") == qid
    assert sm.resolve("AAPL", "ticker", "2008-06-01") == qid


def test_map_instruments_quarantines_unmapped_no_silent_drop():
    sm = SecurityMaster(BitemporalStore())
    m = build_ric_mapping(["AAPL.OQ", "A.N"], sm, KD)
    df = pd.DataFrame({"Instrument": ["AAPL.OQ", "A.N", "DELISTED.OQ"], "Revenue": [1.0, 2.0, 3.0]})
    mapped, quarantine, coverage = map_instruments(df, m)
    assert len(mapped) == 2 and len(quarantine) == 1  # DELISTED.OQ not in universe -> quarantined
    assert len(mapped) + len(quarantine) == len(df)  # NO silent drop
    assert list(quarantine["Instrument"]) == ["DELISTED.OQ"]
    assert abs(coverage - 2 / 3) < 1e-9
    assert "quantlake_id" in mapped.columns
