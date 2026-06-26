"""Feature materialization (s80) — materialized features ARE bitemporal rows (D2).

``materialize(store, spec, as_of)``: cache-checks via the D1 key ``(name, version, as_of, per-dep
watermark)``; on a miss, computes ``fn(*LakeRefs)`` and APPENDS the result to the bitemporal ``features``
table (``kd = as_of``; a fresh ``_ingest_seq`` supersedes a prior materialization at the same ``as_of``
after a backfill). The feature layer therefore INHERITS the store's no-leak invariant — reading a feature
is just ``store.as_of("features", as_of)``, and no feature data lives outside the bitemporal schema.
"""

from __future__ import annotations

import pandas as pd

from quantlake.features.contract import FeatureSpec, LakeRef
from quantlake.store.bitemporal import BitemporalStore

_FN_OUT_COLS = ("quantlake_id", "event_date", "value")


def watermarks(
    store: BitemporalStore, spec: FeatureSpec, as_of: pd.Timestamp
) -> tuple[tuple[str, int], ...]:
    """Per-dep visible watermark at ``as_of`` (the D1 invalidation term)."""
    return tuple((dep, store.watermark(dep, as_of)) for dep in spec.deps)


def _sig(wms: tuple[tuple[str, int], ...]) -> str:
    return ";".join(f"{dep}:{wm}" for dep, wm in wms)


def materialize(store: BitemporalStore, spec: FeatureSpec, as_of) -> pd.DataFrame:
    """Return feature ``spec`` at ``as_of`` — from cache if the per-dep watermark is unchanged, else
    compute over the deps' as-of snapshots and append as bitemporal feature rows."""
    as_of = pd.Timestamp(as_of)
    sig = _sig(watermarks(store, spec, as_of))  # pyright: ignore[reportArgumentType]  # pandas Timestamp stub

    # D1 cache check: a prior materialization of (name, version) visible at as_of whose stored watermark
    # signature equals the current one is still valid (a backfill would have bumped it).
    cached = store.as_of("features", as_of) if store.has_table("features") else pd.DataFrame()
    if not cached.empty:
        mine = cached[
            (cached["feature_name"] == spec.name) & (cached["feature_version"] == spec.version)
        ]
        if not mine.empty and bool((mine["_watermark_sig"] == sig).all()):
            return mine.reset_index(drop=True)  # pyright: ignore[reportReturnType]  # pandas stub

    # Miss: compute over lake refs (controlled inputs) and append as bitemporal rows (D2).
    out = spec.fn(*[LakeRef(store, dep, as_of) for dep in spec.deps])  # pyright: ignore[reportArgumentType]
    missing = set(_FN_OUT_COLS) - set(out.columns)
    if missing:
        raise ValueError(
            f"feature {spec.name!r} fn must return columns {sorted(_FN_OUT_COLS)}; missing {sorted(missing)}"
        )
    rows = out.loc[:, list(_FN_OUT_COLS)].copy()
    rows["feature_name"] = spec.name
    rows["feature_version"] = spec.version
    rows["_watermark_sig"] = sig
    rows["knowledge_date"] = as_of
    store.append("features", rows)

    fresh = store.as_of("features", as_of)
    return fresh[  # pyright: ignore[reportReturnType]  # pandas stub
        (fresh["feature_name"] == spec.name) & (fresh["feature_version"] == spec.version)
    ].reset_index(drop=True)


__all__ = ["materialize", "watermarks"]
