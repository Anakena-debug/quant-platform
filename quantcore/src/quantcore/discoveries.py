"""quantcore.discoveries — a persistent ledger of validated factor findings.

The :mod:`quantcore.catalog` is a static library of *computable primitives* (each
:class:`~quantcore.catalog.FactorSpec` resolves to a callable). A factor that survives the
:mod:`quantcore.factory` gates is a different thing: a *research finding* about some data —
a column name plus the statistics that earned it, with no module to import. This module is
its home: an append-/update-able ledger that records each survivor with its provenance and
screen parameters, persisted as JSON so a desk accumulates validated edges across runs
instead of re-screening from scratch.

    from quantcore.discoveries import DiscoveryLedger
    ledger = DiscoveryLedger.load("discoveries.json")   # empty if the file is absent
    ledger.register_survivors(verdicts, source="us_equities_2024.parquet",
                              screen_params={"method": "spearman", "fdr": 0.10})
    ledger.save("discoveries.json")

Each :class:`Discovery` carries an ``in_catalog`` flag cross-linking back to the catalog when
a primitive of the same name exists — the one bridge between the static library and the
research record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from quantcore import catalog
from quantcore.provenance import RunManifest

if TYPE_CHECKING:
    from quantcore.factory import FactoryVerdict


def _utcnow() -> str:
    """Current UTC timestamp, ISO-8601 to the second (the default discovery stamp)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _in_catalog(name: str) -> bool:
    """True iff the static catalog has a resolvable primitive of this name."""
    try:
        catalog.get(name)
    except KeyError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Discovery:
    """One validated factor finding: identity, provenance, and the stats that earned it."""

    name: str
    category: str
    source: str  # provenance: the dataset / file the factor was validated on
    discovered_utc: str  # ISO-8601 timestamp of when it was recorded
    n_days: int
    mean_ic: float
    ic_t_stat: float
    ic_q_value: float
    ann_sharpe: float
    deflated_sharpe: float
    in_catalog: bool  # does quantcore.catalog hold a resolvable primitive by this name?
    run_id: str | None = None  # provenance: the RunManifest that reproduces this finding
    screen_params: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DiscoveryLedger:
    """A name-keyed, newest-wins registry of :class:`Discovery` records (JSON-persistable)."""

    def __init__(self, discoveries: Iterable[Discovery] = ()) -> None:
        self._by_name: dict[str, Discovery] = {d.name: d for d in discoveries}
        self._manifests: dict[str, RunManifest] = {}

    @property
    def records(self) -> list[Discovery]:
        """All discoveries, ranked by Deflated Sharpe descending."""
        return sorted(self._by_name.values(), key=lambda d: -d.deflated_sharpe)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def add(self, discovery: Discovery) -> None:
        """Insert (or replace, by name) a discovery — the latest record for a name wins."""
        self._by_name[discovery.name] = discovery

    def get(self, name: str) -> Discovery:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no discovery named {name!r} in the ledger") from None

    @property
    def manifests(self) -> list[RunManifest]:
        """All recorded provenance manifests."""
        return list(self._manifests.values())

    def get_manifest(self, run_id: str) -> RunManifest:
        """Fetch the provenance manifest for a ``run_id`` (raises if absent)."""
        try:
            return self._manifests[run_id]
        except KeyError:
            raise KeyError(f"no manifest with run_id {run_id!r} in the ledger") from None

    def register_survivors(
        self,
        verdicts: Iterable[FactoryVerdict],
        *,
        source: str,
        category: str = "discovered",
        screen_params: dict[str, object] | None = None,
        manifest: RunManifest | None = None,
        now: str | None = None,
    ) -> list[Discovery]:
        """Record every *passed* verdict as a :class:`Discovery`; return the ones added.

        Non-survivors are skipped — only factors that cleared both factory gates are written.
        When a ``manifest`` is given, its ``run_id`` is stamped on each survivor and the manifest
        is stored in the ledger (so the finding can later be reproduced via
        :func:`quantcore.provenance.verify_manifest`). ``now`` defaults to the current UTC
        timestamp (override for deterministic records).
        """
        stamp = now or _utcnow()
        params = dict(screen_params or {})
        run_id = manifest.run_id if manifest is not None else None
        if manifest is not None:
            self._manifests[manifest.run_id] = manifest
        added: list[Discovery] = []
        for v in verdicts:
            if not v.passed:
                continue
            discovery = Discovery(
                name=v.name,
                category=category,
                source=source,
                discovered_utc=stamp,
                n_days=v.n_days,
                mean_ic=v.mean_ic,
                ic_t_stat=v.ic_t_stat,
                ic_q_value=v.ic_q_value,
                ann_sharpe=v.ann_sharpe,
                deflated_sharpe=v.deflated_sharpe,
                in_catalog=_in_catalog(v.name),
                run_id=run_id,
                screen_params=params,
            )
            self.add(discovery)
            added.append(discovery)
        return added

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the ledger (ranked discoveries + provenance manifests) to JSON."""
        return json.dumps(
            {
                "schema_version": 2,
                "discoveries": [d.to_dict() for d in self.records],
                "manifests": {rid: m.to_dict() for rid, m in self._manifests.items()},
            },
            indent=indent,
        )

    @classmethod
    def from_json(cls, text: str) -> DiscoveryLedger:
        """Rebuild a ledger from :meth:`to_json` output (tolerant of the legacy flat-list form)."""
        data = json.loads(text)
        if isinstance(data, list):  # schema 1: a bare array of discoveries, no manifests
            return cls(Discovery(**row) for row in data)
        ledger = cls(Discovery(**row) for row in data.get("discoveries", []))
        for rid, mdict in data.get("manifests", {}).items():
            ledger._manifests[rid] = RunManifest.from_dict(mdict)
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> DiscoveryLedger:
        """Load a ledger from a JSON file; an absent file yields an empty ledger."""
        p = Path(path)
        return cls.from_json(p.read_text(encoding="utf-8")) if p.exists() else cls()

    def save(self, path: str | Path) -> None:
        """Persist the ledger to a JSON file (creating parent directories as needed)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


__all__ = ["Discovery", "DiscoveryLedger"]
