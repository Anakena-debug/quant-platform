"""LSEG ingest transforms (s81) — RIC mapping + quarantine (REQ3), wide->long melt (REQ5).

Pure, testable transforms. The RIC->quantlake_id mapping mints ids for the LSEG universe and registers
each in the security master (full RIC + ticker root + venue retained); rows whose Instrument is not in the
mapping go to a QUARANTINE frame (counts reported, NO silent drops). The daily-panel melt is deterministic
and spot-check-tested against the wide original.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import pandas as pd

from quantlake.universe.security_master import SecurityMaster

# Venue suffixes — parsed and retained, never discarded (REQ3).
VENUE_BY_SUFFIX = {"N": "NYSE", "OQ": "NASDAQ", "O": "NASDAQ", "A": "NYSE_AMEX", "K": "NYSE_ARCA"}

# LSEG daily_panel field label -> quantlake column.
PANEL_FIELDS = {
    "Price Close": "price_close",
    "Total Return": "total_return",
    "Volume": "volume",
    "Company Market Cap": "market_cap",
}


def parse_ric(ric: str) -> tuple[str, str]:
    """Split ``ROOT.VENUE`` -> ``(root, venue_suffix)``; ``('AAPL.OQ') -> ('AAPL', 'OQ')``. Venue is
    returned (not discarded); a RIC without a dot returns an empty venue."""
    if "." in ric:
        root, suffix = ric.rsplit(".", 1)
        return root, suffix
    return ric, ""


@dataclass
class RicMapping:
    ric_to_id: dict[str, int]
    quarantine: list[str] = field(default_factory=list)  # malformed / empty-root RICs

    def coverage(self, total: int) -> float:
        return len(self.ric_to_id) / total if total else 1.0


def build_ric_mapping(rics: list[str], sm: SecurityMaster, knowledge_date) -> RicMapping:
    """Mint a quantlake_id per RIC and register it in the security master: the full RIC
    (``symbol_type='ric'``, venue retained in the symbol) + the ticker root (``symbol_type='ticker'``,
    for the cross-vendor join). Empty-root RICs are quarantined.

    s81 path — STANDALONE mint: it creates an LSEG-only id-space (the s82 identity fork). Production
    identity builds go through ``quantlake.ingest.symbology.build_unified_master`` (resolve-then-mint
    onto the Databento entities); this stays for tests/fixtures."""
    ric_to_id: dict[str, int] = {}
    quarantine: list[str] = []
    for ric in rics:
        root, _venue = parse_ric(ric)
        if not root:
            quarantine.append(ric)
            continue
        qid = sm.new_id()
        sm.add_identifier(qid, ric, "ric", knowledge_date, knowledge_date)
        if root != ric:
            sm.add_identifier(qid, root, "ticker", knowledge_date, knowledge_date)
        ric_to_id[ric] = qid
    return RicMapping(ric_to_id=ric_to_id, quarantine=quarantine)


def map_instruments(
    df: pd.DataFrame, mapping: RicMapping, *, instrument_col: str = "Instrument"
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Attach ``quantlake_id`` to LSEG rows via the RIC mapping. Returns ``(mapped, quarantine,
    coverage)`` — rows whose Instrument is not in the mapping are QUARANTINED, never dropped silently."""
    qid = df[instrument_col].map(mapping.ric_to_id)  # pyright: ignore[reportArgumentType]  # Series.map(dict)
    mapped = df[qid.notna()].copy()
    mapped["quantlake_id"] = qid[qid.notna()].astype("int64")
    quarantine = df[qid.isna()].copy()
    coverage = len(mapped) / len(df) if len(df) else 1.0
    return mapped, quarantine, coverage  # pyright: ignore[reportReturnType]  # pandas stub df[mask]


def melt_daily_panel(wide: pd.DataFrame, *, date_col: str = "Date") -> pd.DataFrame:
    """Wide LSEG daily_panel (columns are stringified ``('RIC','Field')`` tuples) -> long
    ``(ric, event_date, price_close, total_return, volume, market_cap)``. Deterministic."""
    dates = wide[date_col].to_numpy()
    per_ric: dict[str, dict[str, object]] = {}
    for col in wide.columns:
        if col == date_col:
            continue
        ric, label = ast.literal_eval(col)  # "('A.N', 'Price Close')" -> ('A.N', 'Price Close')
        if label not in PANEL_FIELDS:
            continue
        per_ric.setdefault(ric, {})[PANEL_FIELDS[label]] = wide[col].to_numpy()
    frames: list[pd.DataFrame] = []
    for ric, cols in sorted(per_ric.items()):
        d = pd.DataFrame({"ric": ric, "event_date": dates})
        for qcol in PANEL_FIELDS.values():
            d[qcol] = cols.get(qcol)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# event_date / knowledge_date column bindings per LSEG table (s81 plan).
TABLE_BINDINGS = {
    "lseg_fundamentals": {"event_date": "Period End Date", "knowledge_date": "Report Date"},
    "lseg_earnings_surprise": {"event_date": "Report Date", "knowledge_date": "Report Date"},
    "lseg_ibes_consensus": {"event_date": "Snapshot Date", "knowledge_date": "Snapshot Date"},
}


def to_lake_rows(mapped: pd.DataFrame, table: str, value_cols: list[str]) -> pd.DataFrame:
    """Map LSEG-named rows (already carrying ``quantlake_id``) to the bitemporal schema for ``table``,
    binding ``event_date``/``knowledge_date`` per :data:`TABLE_BINDINGS` and keeping ``value_cols``."""
    b = TABLE_BINDINGS[table]
    out = pd.DataFrame(
        {
            "quantlake_id": mapped["quantlake_id"].to_numpy(),
            "event_date": pd.to_datetime(mapped[b["event_date"]]).to_numpy(),
            "knowledge_date": pd.to_datetime(mapped[b["knowledge_date"]]).to_numpy(),
        }
    )
    for c in value_cols:
        out[c] = mapped[c].to_numpy()
    return out


__all__ = [
    "PANEL_FIELDS",
    "TABLE_BINDINGS",
    "VENUE_BY_SUFFIX",
    "RicMapping",
    "build_ric_mapping",
    "map_instruments",
    "melt_daily_panel",
    "parse_ric",
    "to_lake_rows",
]
