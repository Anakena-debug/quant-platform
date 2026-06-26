"""Feature-store contract + registry (item 6; materialization in ``features/store.py``, s80).

A feature's compute fn takes lake REFERENCES (``LakeRef`` — a table + an as-of handle resolved through
the bitemporal store), NEVER raw DataFrames, so lineage is fully captured by ``(deps, as_of)`` + the
store's contents. **fn identity is ENFORCED, not trusted:** registering ``(name, version)`` with a
different fn content hash raises — a real change demands a version bump. The hash is therefore kept for
lineage but is NOT in the cache key (s80 D1). The cache key is ``(name, version, as_of, per-dep
watermark)``; see ``features/store.py``.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from quantlake.store.bitemporal import BitemporalStore


def fn_content_hash(fn: object) -> str:
    """Content hash of a feature fn (source if available, else repr) — for lineage + the registry's
    version-bump guard. NOT part of the cache key."""
    try:
        src = inspect.getsource(fn)  # pyright: ignore[reportArgumentType]  # accepts any callable
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()


@dataclass(frozen=True)
class LakeRef:
    """A controlled, as-of-resolved reference to a lake table — the ONLY permitted feature input.

    ``resolve()`` reads through the bitemporal store at ``as_of`` (leak-free); a feature never receives
    a raw DataFrame, so its lineage is fully captured by ``(table, as_of)`` + the store's contents.
    """

    store: BitemporalStore
    table: str
    as_of: pd.Timestamp

    def resolve(self) -> pd.DataFrame:
        return self.store.as_of(self.table, self.as_of)


@runtime_checkable
class FeatureFn(Protocol):
    """Compute signature: lake references in, a ``(quantlake_id, event_date, value)`` frame out."""

    def __call__(self, *refs: LakeRef) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FeatureSpec:
    """A registered feature definition. ``fn_hash`` is lineage + the version-bump guard, NOT a key term."""

    name: str
    version: int
    fn: FeatureFn
    deps: tuple[str, ...]  # lake table names this feature reads (lineage + watermark domain)
    fn_hash: str

    def cache_key(
        self, as_of: pd.Timestamp, watermarks: tuple[tuple[str, int], ...]
    ) -> tuple[str, int, pd.Timestamp, tuple[tuple[str, int], ...]]:
        """s80 D1 cache key. ``watermarks`` = per-dep ``max(_ingest_seq) where kd <= as_of``.

        NOT ``(name, version, knowledge_date)`` (kd is a row attribute, not a query identity) and NOT
        ``(name, version, as_of)`` (a backfill with reconstructed ``kd <= as_of`` legitimately changes
        ``as_of(T)``; the watermark catches it, a future-kd append does not bump it). fn identity is
        enforced by the registry, so the fn hash is not in the key.
        """
        return (self.name, self.version, pd.Timestamp(as_of), watermarks)  # pyright: ignore[reportReturnType]


@dataclass
class FeatureRegistry:
    """Feature registration. Enforces ``(name, version) -> single fn`` (version bumps are mandatory)."""

    _specs: dict[tuple[str, int], FeatureSpec] = field(default_factory=dict)

    def register(
        self, name: str, version: int, fn: FeatureFn, deps: tuple[str, ...]
    ) -> FeatureSpec:
        if not all(isinstance(d, str) for d in deps):
            raise TypeError(
                "deps must be lake table NAMES (str), not DataFrames — lineage requires refs"
            )
        h = fn_content_hash(fn)
        existing = self._specs.get((name, version))
        if existing is not None and existing.fn_hash != h:
            raise ValueError(
                f"feature {name!r} v{version} already registered with a different fn — bump the version "
                "(fn identity is enforced, not trusted; s80 D1)"
            )
        spec = FeatureSpec(name=name, version=version, fn=fn, deps=tuple(deps), fn_hash=h)
        self._specs[(name, version)] = spec
        return spec

    def get(self, name: str, version: int) -> FeatureSpec:
        return self._specs[(name, version)]


__all__ = ["FeatureFn", "FeatureRegistry", "FeatureSpec", "LakeRef", "fn_content_hash"]
