"""Item 1 + 8 — bitemporal schema invariants."""

from __future__ import annotations

import pandas as pd

from quantlake.store.bitemporal import ENTITY_KEYS, TEMPORAL_POLICY, BitemporalStore
from quantlake.universe.security_master import SecurityMaster


def test_pk_includes_knowledge_date_and_ingest_seq():
    s = BitemporalStore()
    for table in ENTITY_KEYS:
        pk = s.pk_columns(table)
        assert "knowledge_date" in pk and "_ingest_seq" in pk, (
            f"{table} PK missing kd/_ingest_seq: {pk}"
        )
        assert pk[-1] == "_ingest_seq"  # B2: tie-break column is in the key


def test_temporal_policy_covers_every_table_with_reconstruction():
    for table in ENTITY_KEYS:
        pol = TEMPORAL_POLICY[table]
        assert {"event_date", "knowledge_date", "reconstruction"} <= set(pol), table
        assert pol["reconstruction"]  # non-empty: how kd is stamped for backfills


def test_schema_version_and_ingest_seq_stamped_not_caller_set():
    s = BitemporalStore()
    s.append(
        "prices",
        pd.DataFrame(
            [
                {
                    "quantlake_id": 1,
                    "event_date": "2024-01-02",
                    "knowledge_date": "2024-01-02",
                    "close": 1.0,
                }
            ]
        ),
    )
    raw = s.con.execute("SELECT * FROM prices").df()
    assert "_schema_version" in raw.columns and "_ingest_seq" in raw.columns
    assert raw["_schema_version"].iloc[0] == 1
    # caller may not set the stamp columns
    bad = pd.DataFrame(
        [
            {
                "quantlake_id": 1,
                "event_date": "2024-01-03",
                "knowledge_date": "2024-01-03",
                "close": 1.0,
                "_ingest_seq": 99,
            }
        ]
    )
    try:
        s.append("prices", bad)
    except ValueError as e:
        assert "_ingest_seq" in str(e)
    else:
        raise AssertionError("expected ValueError on caller-set _ingest_seq")


def test_item8_no_equity_hardcode():
    # Identity accepts a non-equity asset_class without error (schema is asset-class-agnostic).
    sm = SecurityMaster(BitemporalStore())
    sm.add_identifier(
        sm.new_id(), "ESZ4", "ticker", "2024-01-01", "2024-01-01", asset_class="future"
    )
    row = sm.store.as_of("security_master", "2024-02-01")
    assert row["asset_class"].iloc[0] == "future"
