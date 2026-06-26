"""quantcore.discoveries — the validated-factor ledger (registration, persistence, catalog link)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantcore.discoveries import Discovery, DiscoveryLedger
from quantcore.factory import FactoryVerdict
from quantcore.provenance import CodeVersion, build_manifest

_STAMP = "2026-06-17T00:00:00+00:00"
_CODE = CodeVersion(commit="c" * 40, dirty=False, quantcore_version="0.1.0")


def _manifest():
    df = pd.DataFrame(
        {"date": [1, 2, 3], "asset": ["A", "B", "C"], "forward_return": [0.1, 0.2, 0.3]}
    )
    return build_manifest(df, params={"fdr": 0.1}, now=_STAMP, code=_CODE, source="t.csv")


def _verdict(name: str, *, passed: bool, dsr: float = 0.97, ic: float = 0.05) -> FactoryVerdict:
    return FactoryVerdict(
        name=name,
        n_days=120,
        mean_ic=ic,
        ic_t_stat=3.1,
        ic_q_value=0.01,
        ic_significant=True,
        ann_sharpe=1.2,
        deflated_sharpe=dsr,
        passed=passed,
        reason="passed" if passed else "deflated Sharpe below threshold",
        rank=1,
    )


def test_register_survivors_skips_non_passers():
    ledger = DiscoveryLedger()
    added = ledger.register_survivors(
        [_verdict("vpin", passed=True), _verdict("noise", passed=False)],
        source="panel.parquet",
        screen_params={"fdr": 0.10},
        now=_STAMP,
    )
    assert [d.name for d in added] == ["vpin"]
    assert len(ledger) == 1 and "vpin" in ledger and "noise" not in ledger
    d = ledger.get("vpin")
    assert d.discovered_utc == _STAMP and d.source == "panel.parquet"
    assert d.screen_params == {"fdr": 0.10}


def test_in_catalog_flag_cross_links_to_catalog():
    ledger = DiscoveryLedger()
    ledger.register_survivors(
        [_verdict("vpin", passed=True), _verdict("made_up_factor_xyz", passed=True)],
        source="s",
        now=_STAMP,
    )
    assert ledger.get("vpin").in_catalog is True  # a real catalog primitive
    assert ledger.get("made_up_factor_xyz").in_catalog is False


def test_newest_record_wins_by_name():
    ledger = DiscoveryLedger()
    ledger.register_survivors([_verdict("mom", passed=True, dsr=0.96)], source="old", now=_STAMP)
    ledger.register_survivors([_verdict("mom", passed=True, dsr=0.99)], source="new", now=_STAMP)
    assert len(ledger) == 1
    assert ledger.get("mom").deflated_sharpe == 0.99 and ledger.get("mom").source == "new"


def test_records_ranked_by_deflated_sharpe():
    ledger = DiscoveryLedger()
    ledger.register_survivors(
        [_verdict("a", passed=True, dsr=0.90), _verdict("b", passed=True, dsr=0.99)],
        source="s",
        now=_STAMP,
    )
    assert [d.name for d in ledger.records] == ["b", "a"]


def test_json_round_trip():
    ledger = DiscoveryLedger()
    ledger.register_survivors([_verdict("vpin", passed=True)], source="s", now=_STAMP)
    back = DiscoveryLedger.from_json(ledger.to_json())
    assert back.get("vpin").to_dict() == ledger.get("vpin").to_dict()


def test_load_missing_file_is_empty(tmp_path):
    ledger = DiscoveryLedger.load(tmp_path / "does_not_exist.json")
    assert len(ledger) == 0


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "discoveries.json"  # parents created on save
    ledger = DiscoveryLedger()
    ledger.register_survivors([_verdict("vpin", passed=True)], source="s", now=_STAMP)
    ledger.save(path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["discoveries"][0]["name"] == "vpin" and payload["discoveries"][0]["in_catalog"]
    reloaded = DiscoveryLedger.load(path)
    assert "vpin" in reloaded and len(reloaded) == 1


def test_register_with_manifest_stamps_run_id_and_stores_manifest():
    m = _manifest()
    ledger = DiscoveryLedger()
    added = ledger.register_survivors(
        [_verdict("vpin", passed=True)], source="t.csv", manifest=m, now=_STAMP
    )
    assert added[0].run_id == m.run_id
    assert ledger.get("vpin").run_id == m.run_id
    assert ledger.get_manifest(m.run_id) == m
    assert [mm.run_id for mm in ledger.manifests] == [m.run_id]


def test_register_without_manifest_leaves_run_id_none():
    ledger = DiscoveryLedger()
    ledger.register_survivors([_verdict("vpin", passed=True)], source="s", now=_STAMP)
    assert ledger.get("vpin").run_id is None and ledger.manifests == []


def test_v2_json_round_trip_includes_manifests():
    m = _manifest()
    ledger = DiscoveryLedger()
    ledger.register_survivors(
        [_verdict("vpin", passed=True)], source="t.csv", manifest=m, now=_STAMP
    )
    back = DiscoveryLedger.from_json(ledger.to_json())
    assert back.get("vpin").run_id == m.run_id
    assert back.get_manifest(m.run_id) == m


def test_get_unknown_manifest_raises():
    with pytest.raises(KeyError, match="no manifest with run_id"):
        DiscoveryLedger().get_manifest("deadbeef")


def test_from_json_tolerates_legacy_flat_list():
    # schema 1: a bare JSON array of discovery dicts (pre-provenance ledgers), no run_id key.
    legacy = json.dumps(
        [
            {
                "name": "mom",
                "category": "discovered",
                "source": "old.csv",
                "discovered_utc": _STAMP,
                "n_days": 120,
                "mean_ic": 0.05,
                "ic_t_stat": 3.0,
                "ic_q_value": 0.01,
                "ann_sharpe": 1.2,
                "deflated_sharpe": 0.97,
                "in_catalog": False,
                "screen_params": {},
            }
        ]
    )
    ledger = DiscoveryLedger.from_json(legacy)
    assert "mom" in ledger and ledger.get("mom").run_id is None and ledger.manifests == []


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="no discovery named"):
        DiscoveryLedger().get("nope")


def test_discovery_to_dict_has_expected_keys():
    d = Discovery(
        name="x",
        category="discovered",
        source="s",
        discovered_utc=_STAMP,
        n_days=10,
        mean_ic=0.1,
        ic_t_stat=2.0,
        ic_q_value=0.02,
        ann_sharpe=1.0,
        deflated_sharpe=0.95,
        in_catalog=False,
    )
    assert {"name", "source", "discovered_utc", "deflated_sharpe", "in_catalog"} <= set(d.to_dict())
