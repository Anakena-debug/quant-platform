"""Databento symbology bridge + unified identity build (s82) — heals the s81 fork.

s81 minted fresh quantlake_ids for the LSEG universe while Databento prices stayed keyed by vendor
instrument_ids that were never registered in the master: two disjoint id-spaces, so the PEAD/revisions
joins (LSEG signal -> Databento forward return) were impossible. The bridge is the
``symbology.resolve`` payload (instrument_id -> raw_symbol, date-ranged); the unified master
RESOLVES-THEN-MINTS: one quantlake_id per entity, with ``databento_iid``, date-ranged ``ticker``, and
``ric`` identifier rows all pointing at it. Only RICs with no Databento entity mint fresh (counted,
reported — never silent).

Reconciliation is BY REBUILD, not aliasing: the lake is a derived artifact of raw zone + code;
rebuilds are legal until external consumers exist (design doc §s82). The s81 ingest path
(``map_instruments``/``to_lake_rows``) is reused unchanged with the unified mapping — the rebuild IS
running the corrected flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantlake.ingest.lseg import parse_ric
from quantlake.universe.security_master import SecurityMaster

_SYMB_COLS = ["instrument_id", "ticker", "valid_from", "valid_to"]


def parse_symbology_json(payload: dict[str, object]) -> pd.DataFrame:
    """``symbology.resolve`` payload (stype_in=instrument_id, stype_out=raw_symbol) -> tidy intervals
    ``(instrument_id, ticker, valid_from, valid_to)``.

    Databento intervals are ``[d0, d1)`` — d1 exclusive, matching the master's ``valid_to > as_of``
    resolution semantics. Null/'None' symbols are skipped (they carry no bridge information).
    """
    result = payload.get("result")
    if not isinstance(result, dict):
        result = {}
    rows: list[dict[str, object]] = []
    for iid, intervals in result.items():
        for iv in intervals or []:
            sym = iv.get("s")
            if not sym or sym == "None":
                continue
            rows.append(
                {
                    "instrument_id": int(iid),
                    "ticker": str(sym),
                    "valid_from": pd.Timestamp(iv["d0"]),
                    "valid_to": pd.Timestamp(iv["d1"]),
                }
            )
    return pd.DataFrame(rows, columns=_SYMB_COLS)


@dataclass
class UnifiedMaster:
    """The healed identity layer + its build statistics (duck-compatible with ``RicMapping``:
    ``map_instruments(df, master)`` works unchanged via ``ric_to_id``)."""

    sm: SecurityMaster
    iid_to_id: dict[int, int]
    ric_to_id: dict[str, int]
    n_instruments: int = 0
    n_ticker_intervals: int = 0
    n_rics_matched: int = 0
    lseg_only_rics: list[str] = field(default_factory=list)  # no Databento entity -> fresh mint
    malformed_rics: list[str] = field(default_factory=list)
    ticker_overlap_anomalies: int = 0  # same-ticker overlapping intervals (should be 0)


def build_unified_master(
    sm: SecurityMaster,
    symbology: pd.DataFrame,
    rics: list[str],
    *,
    lseg_valid_from="2008-01-01",
) -> UnifiedMaster:
    """One quantlake_id per entity (RESOLVE-THEN-MINT) from the symbology intervals + LSEG universe.

    Databento instruments mint first (``databento_iid`` + a date-ranged ``ticker`` row per interval;
    kd = reconstructed availability = the interval's ``valid_from``). RICs then join onto the SAME ids
    via the ticker root over the Databento coverage window; a RIC whose root matches no Databento
    entity mints fresh and is counted in ``lseg_only_rics`` (reported, never silent). Identifier rows
    are bulk-appended in one batch.
    """
    rows: list[dict[str, object]] = []

    def _idrow(qid: int, symbol: str, stype: str, valid_from, valid_to=None) -> dict[str, object]:
        vf = pd.Timestamp(valid_from)
        return {
            "quantlake_id": int(qid),
            "asset_class": "equity",
            "symbol": symbol,
            "symbol_type": stype,
            "valid_from": vf,
            "valid_to": pd.Timestamp(valid_to) if valid_to is not None else pd.NaT,
            "successor_id": float("nan"),
            "event_date": vf,
            "knowledge_date": vf,  # reconstructed availability (TEMPORAL_POLICY security_master)
        }

    # --- Databento entities first -----------------------------------------
    iid_to_id: dict[int, int] = {}
    # ticker -> [(valid_from, valid_to, qid)] for the in-memory RIC join (same interval semantics
    # as SecurityMaster.resolve, without 503 window queries).
    ticker_index: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
    n_ticker_intervals = 0
    if len(symbology):
        for iid, g in symbology.sort_values(["instrument_id", "valid_from"]).groupby(
            "instrument_id"
        ):
            qid = sm.new_id()
            iid_to_id[int(iid)] = qid  # pyright: ignore[reportArgumentType]  # groupby key stub
            rows.append(
                _idrow(
                    qid,
                    str(int(iid)),  # pyright: ignore[reportArgumentType]  # groupby key stub
                    "databento_iid",
                    g["valid_from"].min(),
                    g["valid_to"].max(),
                )
            )
            for _, r in g.iterrows():
                rows.append(_idrow(qid, str(r["ticker"]), "ticker", r["valid_from"], r["valid_to"]))
                ticker_index.setdefault(str(r["ticker"]), []).append(
                    (pd.Timestamp(r["valid_from"]), pd.Timestamp(r["valid_to"]), qid)  # pyright: ignore[reportArgumentType]
                )
                n_ticker_intervals += 1

    # same-ticker overlapping intervals are an anomaly (symbol reuse must be non-overlapping)
    overlaps = 0
    for ivs in ticker_index.values():
        ivs.sort(key=lambda t: t[0])
        for (f1, t1, _), (f2, _t2, _) in zip(ivs, ivs[1:]):
            if f2 < t1:
                overlaps += 1

    # --- RICs join onto the SAME entities (resolve-then-mint) -------------
    ric_to_id: dict[str, int] = {}
    lseg_only: list[str] = []
    malformed: list[str] = []
    n_matched = 0
    for ric in rics:
        root, _venue = parse_ric(ric)
        if not root:
            malformed.append(ric)
            continue
        qid: int | None = None
        for vf, vt, cand in reversed(ticker_index.get(root, [])):  # latest interval wins
            if vf < vt:
                qid = cand
                break
        if qid is None:
            qid = sm.new_id()
            rows.append(_idrow(qid, root, "ticker", lseg_valid_from))
            lseg_only.append(ric)
        else:
            n_matched += 1
        rows.append(_idrow(qid, ric, "ric", lseg_valid_from))
        ric_to_id[ric] = qid

    if rows:
        sm.store.append("security_master", pd.DataFrame(rows))
    return UnifiedMaster(
        sm=sm,
        iid_to_id=iid_to_id,
        ric_to_id=ric_to_id,
        n_instruments=len(iid_to_id),
        n_ticker_intervals=n_ticker_intervals,
        n_rics_matched=n_matched,
        lseg_only_rics=lseg_only,
        malformed_rics=malformed,
        ticker_overlap_anomalies=overlaps,
    )


def map_databento_prices(
    panel: pd.DataFrame,
    iid_to_id: dict[int, int],
    *,
    id_col: str = "ticker",
    date_col: str = "date",
    value_cols: tuple[str, ...] = ("close",),
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Databento panel rows (instrument_id-keyed; the survfree panel calls that column ``ticker``) ->
    bitemporal ``prices`` rows under MASTER ids. Unmapped instrument_ids go to a quarantine frame
    (counts reported, never silently dropped)."""
    qid = panel[id_col].astype("int64").map(iid_to_id)  # pyright: ignore[reportArgumentType]
    mapped = panel[qid.notna()].copy()
    quarantine = panel[qid.isna()].copy()
    coverage = len(mapped) / len(panel) if len(panel) else 1.0
    dates = pd.to_datetime(mapped[date_col]).to_numpy()
    rows = pd.DataFrame(
        {
            "quantlake_id": qid[qid.notna()].astype("int64").to_numpy(),  # pyright: ignore[reportAttributeAccessIssue]
            "event_date": dates,
            "knowledge_date": dates,  # session-close availability (TEMPORAL_POLICY prices)
        }
    )
    for c in value_cols:
        rows[c] = mapped[c].to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    return rows, quarantine, coverage  # pyright: ignore[reportReturnType]


__all__ = ["UnifiedMaster", "build_unified_master", "map_databento_prices", "parse_symbology_json"]
