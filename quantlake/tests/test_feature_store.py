"""s80 D1 — feature cache: a backfill (kd<=as_of) invalidates; a future-kd append does NOT; the
registry enforces fn identity (a real fn change demands a version bump)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlake.features.contract import FeatureRegistry
from quantlake.features.store import materialize
from quantlake.store.bitemporal import BitemporalStore

T = "2024-03-01"


def _lastclose(ref):  # trivial feature: value = close per (quantlake_id, event_date)
    df = ref.resolve()
    return df[["quantlake_id", "event_date"]].assign(value=df["close"].to_numpy())


def _px(store: BitemporalStore, qid: int, ed: str, kd: str, close: float) -> None:
    store.append(
        "prices",
        pd.DataFrame(
            [{"quantlake_id": qid, "event_date": ed, "knowledge_date": kd, "close": close}]
        ),
    )


def _nfeat(store: BitemporalStore) -> int:
    return int(store.con.execute("SELECT count(*) FROM features").fetchone()[0])


def _spec():
    r = FeatureRegistry()
    return r.register("lastclose", 1, _lastclose, ("prices",))


def test_cache_hit_when_nothing_changed():
    s = BitemporalStore()
    _px(s, 1, "2024-01-02", "2024-01-02", 10.0)
    spec = _spec()
    a = materialize(s, spec, T)
    n = _nfeat(s)
    b = materialize(s, spec, T)  # nothing changed -> HIT, no append
    assert _nfeat(s) == n
    assert a.equals(b)


def test_backfill_kd_le_asof_invalidates():
    s = BitemporalStore()
    _px(s, 1, "2024-01-02", "2024-01-02", 10.0)
    spec = _spec()
    materialize(s, spec, T)
    n = _nfeat(s)
    _px(
        s, 1, "2024-02-01", "2024-02-01", 11.0
    )  # backfill: reconstructed kd <= as_of bumps watermark
    out = materialize(s, spec, T)  # MISS -> recompute
    assert _nfeat(s) > n
    assert set(pd.to_datetime(out["event_date"])) == {
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-02-01"),
    }


def test_future_kd_append_does_not_invalidate():
    s = BitemporalStore()
    _px(s, 1, "2024-01-02", "2024-01-02", 10.0)
    spec = _spec()
    materialize(s, spec, T)
    n = _nfeat(s)
    _px(
        s, 1, "2024-01-02", "2024-09-01", 99.0
    )  # future kd > as_of -> invisible at T -> watermark stable
    materialize(s, spec, T)
    assert _nfeat(s) == n  # HIT, no recompute


def test_registry_enforces_fn_identity_version_bump():
    r = FeatureRegistry()
    r.register("f", 1, _lastclose, ("prices",))
    r.register("f", 1, _lastclose, ("prices",))  # same fn -> idempotent, no raise
    with pytest.raises(ValueError, match="bump the version"):
        r.register(
            "f", 1, lambda ref: ref.resolve().head(0), ("prices",)
        )  # different fn, same version
