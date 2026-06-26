"""S43 — FrozenFlowRegimeArtifact write/read round-trip + validation guards.

Pins the disk contract that lets alpha_R (writer) hand a frozen dual-horizon
model package to quantengine (reader): byte-faithful round-trip of the two
estimators + the manifest scalars, and the validation that makes read() REFUSE a
corrupt / drifted / version-mismatched package rather than return a silently
wrong estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from quantcore.models import (
    N_FEATURES,
    RAW_FLOW_COLS,
    FrozenFlowRegimeArtifact,
    HorizonSpec,
    IncompatibleArtifactError,
)

_CREATED = "2026-05-30T00:00:00Z"


def _feature_order() -> list[str]:
    cols: list[str] = []
    for raw in RAW_FLOW_COLS:
        cols += [raw, f"{raw}_sum5", f"{raw}_sum10", f"{raw}_sum20", f"{raw}_z20"]
    assert len(cols) == N_FEATURES
    return cols


def _toy_gbc(seed: int) -> GradientBoostingClassifier:
    """A tiny 3-class GBC fit on 15 features (so n_features_in_ == 15)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((120, N_FEATURES))
    # 3-class target correlated with feature 0 so all of {-1,0,1} appear.
    y = np.sign(x[:, 0] + 0.3 * rng.standard_normal(120)).astype(int)
    # guarantee all three classes present
    y[:3] = [-1, 0, 1]
    m = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=seed)
    m.fit(x, y)
    return m


def _horizon(name: str, h: int, lo: float, hi: float, classes: list[int]) -> HorizonSpec:
    return HorizonSpec(
        name=name,
        horizon=h,
        model_file=f"model_h{h}.joblib",
        spread_lo_bps=lo,
        spread_hi_bps=hi,
        entry_th=0.0010,
        exit_th=0.0005,
        flip_th=0.0015,
        mu={-1: -0.001, 0: 0.0, 1: 0.0012},
        classes=classes,
    )


def _artifact(models: dict[str, GradientBoostingClassifier]) -> FrozenFlowRegimeArtifact:
    classes_t = [int(c) for c in models["tight"].classes_]
    classes_m = [int(c) for c in models["moderate"].classes_]
    return FrozenFlowRegimeArtifact(
        ticker="TXN",
        feature_order=_feature_order(),
        raw_flow_cols=RAW_FLOW_COLS,
        horizons=[
            _horizon("tight", 100, 0.0, 5.0, classes_t),
            _horizon("moderate", 50, 10.0, 25.0, classes_m),
        ],
        regime_priority=["tight", "moderate"],
        cost_bps=5.0,
        label_config={"cusum_threshold": 0.01, "pt_sl": [0.75, 0.75]},
        bar_config={"kind": "dollar", "threshold": 500_000},
        train_window={"n_bars": 120},
        provenance={"sklearn_version": __import__("sklearn").__version__, "created_utc": _CREATED},
    )


def _write(tmp: Path) -> tuple[Path, dict[str, GradientBoostingClassifier]]:
    models = {"tight": _toy_gbc(1), "moderate": _toy_gbc(2)}
    art = _artifact(models)
    d = art.write(tmp / "art", models)
    return d, models


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_reloads_equal_and_predicts(tmp_path: Path) -> None:
    d, models = _write(tmp_path)
    art2, models2 = FrozenFlowRegimeArtifact.read(d)

    assert art2.ticker == "TXN"
    assert art2.feature_order == _feature_order()
    assert art2.raw_flow_cols == RAW_FLOW_COLS
    assert [h.name for h in art2.horizons] == ["tight", "moderate"]
    assert art2.horizons[0].mu == {-1: -0.001, 0: 0.0, 1: 0.0012}

    # Reloaded estimators reproduce predict_proba byte-for-byte (cross-process
    # fidelity — the property s44's live path depends on).
    x = np.random.default_rng(9).standard_normal((4, N_FEATURES))
    for name in ("tight", "moderate"):
        np.testing.assert_array_equal(models[name].predict_proba(x), models2[name].predict_proba(x))


def test_read_manifest_is_cheap_json(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    man = FrozenFlowRegimeArtifact.read_manifest(d)
    assert man["artifact_kind"] == "frozen_flow_regime_strategy"
    assert man["schema_version"] == 1
    assert "model_sha256" in man["provenance"]
    assert set(man["provenance"]["model_sha256"]) == {"model_h100.joblib", "model_h50.joblib"}


# ---------------------------------------------------------------------------
# Validation guards — read() must REFUSE
# ---------------------------------------------------------------------------


def test_sha_mismatch_raises(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    # Corrupt one payload after the manifest hash was computed.
    (d / "model_h50.joblib").write_bytes(b"corrupted-not-a-model")
    with pytest.raises(IncompatibleArtifactError, match="sha256 mismatch"):
        FrozenFlowRegimeArtifact.read(d)


def test_sklearn_version_mismatch_raises_then_warns(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    man["provenance"]["sklearn_version"] = "0.0.0-not-installed"
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(IncompatibleArtifactError, match="sklearn"):
        FrozenFlowRegimeArtifact.read(d)
    # Override downgrades to a warning and still loads.
    with pytest.warns(RuntimeWarning, match="sklearn"):
        art, models = FrozenFlowRegimeArtifact.read(d, allow_version_mismatch=True)
    assert set(models) == {"tight", "moderate"}


def test_feature_order_length_mismatch_raises(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    man["feature_order"] = man["feature_order"][:10]  # wrong length
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(IncompatibleArtifactError, match="feature_order"):
        FrozenFlowRegimeArtifact.read(d)


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    man["schema_version"] = 99
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(IncompatibleArtifactError, match="schema_version"):
        FrozenFlowRegimeArtifact.read(d)


def test_classes_mismatch_raises(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    man["horizons"][0]["classes"] = [0, 1]  # estimator has [-1,0,1]
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(IncompatibleArtifactError, match="classes_"):
        FrozenFlowRegimeArtifact.read(d)


def test_missing_payload_raises(tmp_path: Path) -> None:
    d, _ = _write(tmp_path)
    (d / "model_h100.joblib").unlink()
    with pytest.raises(IncompatibleArtifactError, match="payload missing"):
        FrozenFlowRegimeArtifact.read(d)


def test_missing_models_on_write_raises(tmp_path: Path) -> None:
    models = {"tight": _toy_gbc(1)}  # 'moderate' absent
    art = _artifact({"tight": models["tight"], "moderate": _toy_gbc(2)})
    with pytest.raises(ValueError, match="models missing"):
        art.write(tmp_path / "art", models)
