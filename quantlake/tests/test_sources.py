"""Item 9 — sourcing/bias inventory; the three gaps are flagged, not silently absent."""

from __future__ import annotations

from quantlake.store.bitemporal import ENTITY_KEYS
from quantlake.store.sources import KD_BASES, SOURCES, UNIVERSE_BASES, unsourced_columns


def test_every_table_has_a_source_entry():
    for table in ENTITY_KEYS:
        assert table in SOURCES, f"{table} has no SOURCES entry"
        assert SOURCES[table].source and SOURCES[table].biases


def test_every_table_declares_kd_basis_and_universe_basis():
    # s81 REQ1 (schema-first): both are present and in-enum for EVERY table, before any rows.
    for table in ENTITY_KEYS:
        s = SOURCES[table]
        assert s.kd_basis in KD_BASES, f"{table}: kd_basis {s.kd_basis!r}"
        assert s.universe_basis in UNIVERSE_BASES, f"{table}: universe_basis {s.universe_basis!r}"


def test_every_lseg_table_is_current_constituents():
    # The survivorship flag is machine-readable: every LSEG overlay is current-SPX backfilled.
    lseg = [t for t in SOURCES if t.startswith("lseg_")]
    assert lseg, "expected LSEG tables registered"
    for t in lseg:
        assert SOURCES[t].universe_basis == "current_constituents", t
    # the Databento price base stays survivorship-free
    assert SOURCES["prices"].universe_basis == "survfree"


def test_delisting_return_is_flagged_unsourced_with_bias_direction():
    s = SOURCES["instrument_status"]
    assert "delisting_return" in s.nullable_with_flag  # present-but-null-with-flag, not absent
    # the acid test: bias DIRECTION + magnitude are stated, not hand-waved
    biases = s.biases.lower()
    assert "inflate" in biases and ("distress" in biases or "bankrupt" in biases)


def test_index_membership_and_identifiers_flagged():
    assert "index_name" in unsourced_columns("universe_membership")  # no index history in v0
    assert set(unsourced_columns("security_master")) == {"cusip", "figi"}
