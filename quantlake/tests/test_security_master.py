"""Item 2 — permanent internal id; resolve incl databento_iid; ticker reuse; merger; no ticker join."""

from __future__ import annotations

from quantlake.store.bitemporal import ENTITY_KEYS, BitemporalStore
from quantlake.universe.security_master import SecurityMaster


def _sm() -> SecurityMaster:
    return SecurityMaster(BitemporalStore())


def test_quantlake_id_is_internal_sequence_not_vendor_id():
    sm = _sm()
    a, b = sm.new_id(), sm.new_id()
    assert (a, b) == (1, 2)  # monotonic internal sequence
    sm.add_identifier(a, "999999", "databento_iid", "2024-01-01", "2024-01-01")
    # the quantlake_id is NOT the vendor id
    assert sm.resolve("999999", "databento_iid", "2024-02-01") == a != 999999


def test_resolve_round_trip_ticker_and_databento_iid():
    sm = _sm()
    qid = sm.new_id()
    sm.add_identifier(qid, "AAPL", "ticker", "2024-01-01", "2024-01-01")
    sm.add_identifier(qid, "12345", "databento_iid", "2024-01-01", "2024-01-01")
    assert sm.resolve("AAPL", "ticker", "2024-06-01") == qid
    assert sm.resolve("12345", "databento_iid", "2024-06-01") == qid
    assert sm.resolve("MSFT", "ticker", "2024-06-01") is None


def test_ticker_reuse_non_overlapping_intervals():
    sm = _sm()
    old, new = sm.new_id(), sm.new_id()
    sm.add_identifier(old, "ABC", "ticker", "2010-01-01", "2010-01-01", valid_to="2015-01-01")
    sm.add_identifier(new, "ABC", "ticker", "2016-01-01", "2016-01-01")  # symbol reassigned
    assert sm.resolve("ABC", "ticker", "2012-06-01") == old
    assert sm.resolve("ABC", "ticker", "2020-06-01") == new
    assert sm.resolve("ABC", "ticker", "2015-06-01") is None  # gap between intervals


def test_merger_closes_target_and_links_successor():
    sm = _sm()
    target, acquirer = sm.new_id(), sm.new_id()
    sm.add_identifier(target, "TGT", "ticker", "2018-01-01", "2018-01-01")
    sm.merge(
        target_id=target, acquirer_id=acquirer, effective="2021-01-01", knowledge_date="2021-01-01"
    )
    # target's TGT interval is closed at the merger effective date
    assert sm.resolve("TGT", "ticker", "2020-06-01") == target
    assert sm.resolve("TGT", "ticker", "2022-06-01") is None
    sm_rows = sm.store.as_of("security_master", "2021-06-01")
    closed = sm_rows[(sm_rows["quantlake_id"] == target) & sm_rows["valid_to"].notna()]
    assert len(closed) == 1 and int(closed.iloc[0]["successor_id"]) == acquirer


def test_no_lake_table_joins_on_ticker():
    # Every non-security_master table keys on quantlake_id; none uses ticker/symbol as an entity key.
    for table, keys in ENTITY_KEYS.items():
        if table == "security_master":
            continue
        assert "quantlake_id" in keys, f"{table} must key on quantlake_id"
        assert not ({"ticker", "symbol"} & set(keys)), f"{table} keys on a ticker/symbol: {keys}"
