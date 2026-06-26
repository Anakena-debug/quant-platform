"""Security master (item 2) — permanent INTERNAL identity, never join on ticker.

``quantlake_id`` is an internal synthetic surrogate (a monotonic sequence), NOT seeded from any vendor
id (B3: vendor ids are dataset-scoped with unverified recycling). Vendor ids — including Databento
``instrument_id`` — are just mapping rows (``symbol_type='databento_iid'``). Every other lake table keys
on ``quantlake_id``. Symbol->id mappings are non-overlapping validity intervals per ``(symbol,
symbol_type)`` (ticker reuse safe); a merger closes the target's intervals and records ``successor_id``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from quantlake.store.bitemporal import BitemporalStore

SYMBOL_TYPES = ("ticker", "cusip", "figi", "databento_iid", "ric")


@dataclass
class SecurityMaster:
    store: BitemporalStore
    _ids: "itertools.count[int]" = field(default_factory=lambda: itertools.count(1))

    def new_id(self) -> int:
        """Mint a fresh INTERNAL quantlake_id (a sequence — never a vendor id)."""
        return next(self._ids)

    def add_identifier(
        self,
        quantlake_id: int,
        symbol: str,
        symbol_type: str,
        valid_from,
        knowledge_date,
        *,
        valid_to=None,
        asset_class: str = "equity",
        successor_id: int | None = None,
    ) -> None:
        if symbol_type not in SYMBOL_TYPES:
            raise ValueError(f"symbol_type must be one of {SYMBOL_TYPES}; got {symbol_type!r}")
        self.store.append(
            "security_master",
            pd.DataFrame(
                [
                    {
                        "quantlake_id": int(quantlake_id),
                        "asset_class": asset_class,
                        "symbol": symbol,
                        "symbol_type": symbol_type,
                        "valid_from": pd.Timestamp(valid_from),
                        "valid_to": pd.Timestamp(valid_to) if valid_to is not None else pd.NaT,
                        "successor_id": float(successor_id)
                        if successor_id is not None
                        else float("nan"),
                        "event_date": pd.Timestamp(valid_from),
                        "knowledge_date": pd.Timestamp(knowledge_date),
                    }
                ]
            ),
        )

    def resolve(self, symbol: str, symbol_type: str, as_of) -> int | None:
        """The quantlake_id that ``(symbol, symbol_type)`` maps to AT ``as_of`` (interval-covering,
        knowledge-resolved). Returns None if no interval covers ``as_of``."""
        as_of = pd.Timestamp(as_of)
        sm = self.store.as_of("security_master", as_of)
        if sm.empty:
            return None
        m = (
            (sm["symbol"] == symbol)
            & (sm["symbol_type"] == symbol_type)
            & (sm["valid_from"] <= as_of)
        )
        m &= sm["valid_to"].isna() | (sm["valid_to"] > as_of)
        hits = sm[m]
        return int(hits.iloc[0]["quantlake_id"]) if len(hits) else None

    def merge(self, target_id: int, acquirer_id: int, effective, knowledge_date) -> None:
        """Record a merger: the acquirer's quantlake_id SURVIVES; the target's open identifier
        intervals are closed at ``effective`` and linked via ``successor_id`` (target history retained)."""
        effective = pd.Timestamp(effective)
        sm = self.store.as_of("security_master", knowledge_date)
        open_rows = sm[(sm["quantlake_id"] == target_id) & sm["valid_to"].isna()]
        for _, r in open_rows.iterrows():
            self.add_identifier(
                target_id,
                str(r["symbol"]),
                str(r["symbol_type"]),
                r["valid_from"],  # add_identifier wraps in pd.Timestamp
                knowledge_date,
                valid_to=effective,
                asset_class=str(r["asset_class"]),
                successor_id=int(acquirer_id),
            )


__all__ = ["SYMBOL_TYPES", "SecurityMaster"]
