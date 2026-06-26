"""s79 commit 1 — _ingest_seq survives close/reopen.

The failure single-process Hypothesis can't see: an in-memory counter resets across sessions, so a
file-backed close->reopen->append collides (verified [0,0] pre-fix). _ingest_seq is now derived from the
persisted MAX, so it is globally monotonic across sessions — both for uniqueness and for the same-kd
tie-break.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from quantlake.store.bitemporal import BitemporalStore


def _row(kd: str, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"quantlake_id": 1, "event_date": "2024-01-02", "knowledge_date": kd, "close": close}]
    )


def test_ingest_seq_strictly_increasing_across_reopen(tmp_path: Path):
    path = str(tmp_path / "lake.duckdb")
    s1 = BitemporalStore(con=duckdb.connect(path))
    s1.append("prices", _row("2024-01-02", 10.0))
    s1.append("prices", _row("2024-01-03", 10.5))
    s1.con.close()

    s2 = BitemporalStore(con=duckdb.connect(path))  # reopen: new Store, no in-memory counter
    s2.append("prices", _row("2024-01-04", 11.0))
    s2.append("prices", _row("2024-01-05", 11.5))
    seqs = (
        s2.con.execute("SELECT _ingest_seq FROM prices ORDER BY _ingest_seq")
        .df()["_ingest_seq"]
        .tolist()
    )
    s2.con.close()

    assert seqs == [0, 1, 2, 3]  # continued across the reopen, not reset
    assert len(set(seqs)) == len(seqs)  # globally unique (no reset collision)


def test_same_kd_restatement_tiebreak_survives_reopen(tmp_path: Path):
    """A same-(qid,event,kd) restatement appended AFTER reopen must win the tie-break (higher
    _ingest_seq) — and not collide on the PK. Pre-fix this got _ingest_seq=0 => unique-index violation."""
    path = str(tmp_path / "lake.duckdb")
    s1 = BitemporalStore(con=duckdb.connect(path))
    s1.append("prices", _row("2024-01-02", 10.0))  # (1, 2024-01-02, 2024-01-02, seq=0)
    s1.con.close()

    s2 = BitemporalStore(con=duckdb.connect(path))
    s2.append(
        "prices", _row("2024-01-02", 99.0)
    )  # same (qid,event,kd) restatement -> seq=1, must win
    got = s2.as_of("prices", "2024-01-03")
    s2.con.close()

    assert len(got) == 1 and float(got.iloc[0]["close"]) == 99.0  # latest _ingest_seq wins
