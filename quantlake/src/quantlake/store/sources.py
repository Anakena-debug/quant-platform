"""Data sourcing + bias inventory (item 9; extended s81 with kd_basis + universe_basis).

Names the v0 source per table, its biases, which columns are NULL-with-a-flag (never silently absent),
and — schema-first (s81 REQ1) — the **knowledge-date basis** and **universe basis** so survivorship and
PIT-fidelity are machine-readable, not folklore. The two-tier policy: discovery on the LSEG 17y panel,
validation on the Databento survfree overlap; no allocator-facing number cites the LSEG universe alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# kd_basis: is the knowledge_date a genuine availability (exact), a reconstructed proxy, or unknown?
KD_BASES = frozenset({"exact", "reconstructed", "unknown"})
# universe_basis: is the cross-section survivorship-free, current-constituents-backfilled, or unknown?
UNIVERSE_BASES = frozenset({"survfree", "current_constituents", "unknown"})


@dataclass(frozen=True)
class Source:
    table: str
    source: str
    biases: str
    nullable_with_flag: tuple[str, ...]  # columns present but unsourced (NULL + a _source_flag)
    kd_basis: str  # one of KD_BASES
    universe_basis: str  # one of UNIVERSE_BASES

    def __post_init__(self) -> None:
        if self.kd_basis not in KD_BASES:
            raise ValueError(f"{self.table}: kd_basis {self.kd_basis!r} not in {sorted(KD_BASES)}")
        if self.universe_basis not in UNIVERSE_BASES:
            raise ValueError(
                f"{self.table}: universe_basis {self.universe_basis!r} not in {sorted(UNIVERSE_BASES)}"
            )


SOURCES: dict[str, Source] = {
    "prices": Source(
        table="prices",
        source="Databento EQUS.MINI ohlcv-1d (survfree, delisting-inclusive, RAW) + databento_xs_panel.parquet",
        biases="RAW (splits detected, not authoritative); PRICE-RETURN ONLY (no dividends -> total-return gap, disclosed); ~3.2y",
        nullable_with_flag=(),
        kd_basis="reconstructed",  # session-close availability, backfilled
        universe_basis="survfree",  # Databento ALL_SYMBOLS, delisting-inclusive
    ),
    "security_master": Source(
        table="security_master",
        source="Databento instrument_id + raw_symbol",
        biases="CUSIP/FIGI unsourced",
        nullable_with_flag=("cusip", "figi"),
        kd_basis="reconstructed",
        universe_basis="survfree",
    ),
    "universe_membership": Source(
        table="universe_membership",
        source="UNSOURCED index history -> LIQUIDITY-DEFINED (top-N median dollar-vol, PIT trailing window)",
        biases="no index-membership source; a liquidity proxy stands in (PIT-constructed, no full-sample stat)",
        nullable_with_flag=("index_name",),
        kd_basis="reconstructed",
        universe_basis="survfree",
    ),
    "instrument_status": Source(
        table="instrument_status",
        source="Databento (delisted instruments simply stop having bars)",
        biases=(
            "delisting RETURN UNSOURCED (ohlcv-1d has no terminal return). BIAS: omission INFLATES "
            "backtest performance (missing returns are predominantly large-negative); SEVERE for "
            "distress/bankruptcy delistings (terminal ~ -100%). v0 cannot repair this by schema alone; "
            "CRSP/authoritative delisting returns are the s-later fix."
        ),
        nullable_with_flag=("delisting_return",),
        kd_basis="reconstructed",
        universe_basis="survfree",
    ),
    "corp_actions": Source(
        table="corp_actions",
        source="detected from RAW prices (quantdata heuristic, 0.22% residual)",
        biases="heuristic, not authoritative; dividends absent (price-return)",
        nullable_with_flag=(),
        kd_basis="reconstructed",
        universe_basis="survfree",
    ),
    "features": Source(
        table="features",
        source="DERIVED (s80) — materialized from a feature spec's dep tables; lineage = spec.deps + fn hash",
        biases="no independent source; INHERITS the biases of its dep tables",
        nullable_with_flag=(),
        kd_basis="exact",  # kd = as_of (definitionally the date it was computable)
        universe_basis="unknown",  # inherits its deps' universe
    ),
    # --- s81 LSEG ingest (every LSEG table is current-SPX backfilled -> current_constituents) ---
    "lseg_fundamentals": Source(
        table="lseg_fundamentals",
        source="LSEG fundamentals_q_pit_2008_2025 (Revenue/EBITDA/Total Debt/CapEx; 7-col clean table)",
        biases="Report Date is a release PROXY, not legal filing; NI/Assets/Equity absent (re-pull backlog)",
        nullable_with_flag=(),
        kd_basis="reconstructed",  # Report Date = release proxy
        universe_basis="current_constituents",  # 0#.SPX backfilled -> survivorship
    ),
    "lseg_earnings_surprise": Source(
        table="lseg_earnings_surprise",
        source="LSEG earnings_surprise_pit (SUE = as-of(Snapshot<=Report) Actual-vs-Consensus)",
        biases="Report Date is a vendor release proxy (despite intraday precision); current-SPX universe",
        nullable_with_flag=(),
        kd_basis="reconstructed",
        universe_basis="current_constituents",
    ),
    "lseg_ibes_consensus": Source(
        table="lseg_ibes_consensus",
        source="LSEG ibes_consensus_snapshots (monthly forward-FQ1 EPS mean/std/count)",
        biases="current-SPX universe (survivorship); supersedes the old synthetic-date ibes_consensus",
        nullable_with_flag=(),
        kd_basis="exact",  # Snapshot/Calc Date = the date the consensus existed
        universe_basis="current_constituents",
    ),
    "lseg_daily_panel": Source(
        table="lseg_daily_panel",
        source="LSEG daily_panel (Price Close + Total Return + Company Market Cap; 2008-2025)",
        biases="current-SPX universe (survivorship; penalizes IPOs/additions/renames); Tier-2 cross-vendor check",
        nullable_with_flag=(),
        kd_basis="reconstructed",  # session availability
        universe_basis="current_constituents",
    ),
}


def unsourced_columns(table: str) -> tuple[str, ...]:
    """Columns that are present-but-NULL-with-a-flag for ``table`` (never silently absent)."""
    return SOURCES[table].nullable_with_flag


__all__ = ["KD_BASES", "SOURCES", "UNIVERSE_BASES", "Source", "unsourced_columns"]
