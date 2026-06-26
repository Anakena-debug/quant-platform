"""s82 — REAL-DATA identity sentinel (skip-gated until the operator's symbology pull lands).

The seal criterion, verbatim: AAPL.OQ and Databento-AAPL share ONE quantlake_id; close series overlap
>= 100 sessions with rel-diff < tol. Aggregate coverage stats can look perfect while the vendors can't
see each other — the single end-to-end company is the test the fork is healed. Then the s81-deferred
cross-vendor parity rate runs for real (conventions verified in s81: BOTH closes are split-adjusted)
and is written as a data-quality artifact (gate on review, not a threshold).

Arms automatically once ``alpha_R/outputs/survfree/symbology.json`` exists (operator command:
``cd alpha_R && DATABENTO_API_KEY=<key> uv run python scripts/73_pull_databento_symbology.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantlake.ingest.parity import cross_vendor_parity
from quantlake.ingest.symbology import (
    build_unified_master,
    map_databento_prices,
    parse_symbology_json,
)
from quantlake.store.bitemporal import BitemporalStore
from quantlake.universe.security_master import SecurityMaster

_REPO = Path(__file__).resolve().parents[2]
SYMBOLOGY = _REPO / "alpha_R" / "outputs" / "survfree" / "symbology.json"
DB_PANEL = _REPO / "alpha_R" / "outputs" / "survfree" / "databento_xs_panel.parquet"
LSEG_PANEL = Path.home() / "dev" / "M4" / "lseg" / "daily_panel.parquet"
UNIVERSE = Path.home() / "dev" / "M4" / "lseg" / "universe_rics.txt"
PARITY_ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "parity_lseg_databento.json"

_armed = SYMBOLOGY.exists() and DB_PANEL.exists() and LSEG_PANEL.exists() and UNIVERSE.exists()
pytestmark = pytest.mark.skipif(
    not _armed, reason="awaiting operator symbology pull (scripts/73) + local panels"
)


@pytest.fixture(scope="module")
def master():
    payload = json.loads(SYMBOLOGY.read_text())
    symb = parse_symbology_json(payload)
    rics = [r.strip() for r in UNIVERSE.read_text().splitlines() if r.strip()]
    sm = SecurityMaster(BitemporalStore())
    um = build_unified_master(sm, symb, rics)
    return sm, um, symb


def _lseg_close(ric: str) -> pd.Series:
    col = f"('{ric}', 'Price Close')"
    wide = pd.read_parquet(LSEG_PANEL, columns=["Date", col])
    s = wide.set_index(pd.to_datetime(wide["Date"]).dt.normalize())[col].dropna()
    return s.astype(float)


def test_aapl_one_id_and_close_overlap(master):
    sm, um, symb = master
    qid = um.ric_to_id["AAPL.OQ"]
    aapl_iids = symb[symb["ticker"] == "AAPL"]["instrument_id"].tolist()
    assert any(um.iid_to_id[i] == qid for i in aapl_iids), "fork NOT healed: AAPL split across ids"

    lseg = _lseg_close("AAPL.OQ")
    db = pd.read_parquet(DB_PANEL, columns=["ticker", "date", "close"])
    db = db[db["ticker"].isin(aapl_iids)]
    dbs = db.set_index(pd.to_datetime(db["date"]).dt.normalize())["close"].astype(float)
    j = pd.concat([lseg.rename("lseg"), dbs.rename("db")], axis=1).dropna()
    assert len(j) >= 100, f"overlap {len(j)} < 100 sessions"
    rel = (j["lseg"] / j["db"] - 1.0).abs()
    assert float((rel <= 0.01).mean()) >= 0.95, (
        f"rel-diff: only {(rel <= 0.01).mean():.1%} within 1%"
    )


def test_full_universe_parity_rate_artifact(master):
    sm, um, _symb = master
    from quantlake.ingest.lseg import melt_daily_panel
    from quantlake.ingest.raw_zone import ingest_to_raw_zone

    ingest_to_raw_zone([SYMBOLOGY])  # the bridge is an input: content-address it like any source
    wide = pd.read_parquet(LSEG_PANEL)
    long = melt_daily_panel(wide)
    long = long.dropna(subset=["price_close"])
    long["quantlake_id"] = long["ric"].map(um.ric_to_id)
    long["date"] = pd.to_datetime(long["event_date"]).dt.normalize()
    lseg_px = long.dropna(subset=["quantlake_id"])[["quantlake_id", "date", "price_close"]]

    db = pd.read_parquet(DB_PANEL, columns=["ticker", "date", "close"])
    rows, quar, cov = map_databento_prices(db, um.iid_to_id)
    rows["date"] = pd.to_datetime(rows["event_date"]).dt.normalize()
    keep = set(int(q) for q in lseg_px["quantlake_id"].unique())
    db_px = rows[rows["quantlake_id"].isin(keep)][["quantlake_id", "date", "close"]]

    res = cross_vendor_parity(
        lseg_px,
        db_px,
        on=["quantlake_id", "date"],
        left_col="price_close",
        right_col="close",
        conventions_match=True,  # verified s81: BOTH split-adjusted (NVDA smooth through 10:1)
        tol=0.01,
        top=20,
    )
    assert res.n_compared >= 10_000, f"parity overlap too thin: {res.n_compared}"
    PARITY_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    PARITY_ARTIFACT.write_text(
        json.dumps(
            {
                "n_compared": res.n_compared,
                "disagreement_rate_at_1pct": res.disagreement_rate,
                "databento_price_coverage_via_master": cov,
                "worst": res.worst.assign(date=res.worst["date"].astype(str)).to_dict("records"),
            },
            indent=2,
            default=str,
        )
    )
    # gate on REVIEW, not a threshold: the artifact is the deliverable
    assert PARITY_ARTIFACT.exists()
