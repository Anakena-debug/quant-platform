"""Smoke tests for quantengine.data.signal.SignalArtifact.

We use the ``json`` format here — the ``parquet`` format's behavior is
identical at the dataframe layer, and this sandbox lacks a parquet
engine. Production code will use ``fmt='parquet'`` and the round-trip
invariant is the same.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from quantengine.contracts.signal import build_alpha_signal
from quantengine.data.signal import SCHEMA_VERSION, SignalArtifact


def _sig(with_kelly: bool = True):
    return build_alpha_signal(
        tickers=("AAPL", "MSFT", "NVDA", "SPY"),
        expected_return=[0.02, 0.01, 0.03, 0.005],
        lower=[0.005, 0.002, 0.01, 0.001],
        upper=[0.04, 0.02, 0.05, 0.01],
        alpha=0.10,
        kelly_weights=[0.25, 0.20, 0.30, 0.15] if with_kelly else None,
        timestamp="2026-04-17T16:00:00Z",
        metadata={"ignored_on_write": True},
    )


def test_round_trip_json_preserves_floats():
    sig = _sig()
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        art.write(sig, run_id="r-001", model_sha="abc1234")
        back = art.read()
    assert back.tickers == sig.tickers
    assert np.allclose(back.expected_return, sig.expected_return, atol=0, rtol=0)
    assert np.allclose(back.lower, sig.lower, atol=0, rtol=0)
    assert np.allclose(back.upper, sig.upper, atol=0, rtol=0)
    assert back.kelly_weights is not None
    assert np.allclose(back.kelly_weights, sig.kelly_weights, atol=0, rtol=0)
    assert back.alpha == sig.alpha


def test_round_trip_without_kelly():
    sig = _sig(with_kelly=False)
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        art.write(sig, run_id="r-002", model_sha="def5678")
        back = art.read()
    assert back.kelly_weights is None


def test_manifest_contains_provenance():
    sig = _sig()
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        art.write(sig, run_id="r-42", model_sha="cafebabe")
        manifest = json.loads((art.path / "manifest.json").read_text())
    assert manifest["run_id"] == "r-42"
    assert manifest["model_sha"] == "cafebabe"
    assert manifest["alpha"] == 0.10
    assert manifest["as_of"] == "2026-04-17T16:00:00Z"
    assert manifest["n"] == 4
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["format"] == "json"
    assert manifest["has_kelly"] is True


def test_read_manifest_independently():
    sig = _sig()
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        art.write(sig, run_id="r-x", model_sha="11112222")
        m = art.read_manifest()
    assert m["run_id"] == "r-x"


def test_missing_manifest_raises():
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "nonexistent", fmt="json")
        raised = False
        try:
            art.read_manifest()
        except FileNotFoundError:
            raised = True
        assert raised


def test_extra_cannot_override_core_keys():
    sig = _sig()
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        raised = False
        try:
            art.write(sig, run_id="r", model_sha="s", extra={"alpha": 0.99})
        except ValueError:
            raised = True
        assert raised


def test_roundtrip_metadata_has_run_and_model():
    sig = _sig()
    with tempfile.TemporaryDirectory() as td:
        art = SignalArtifact(path=Path(td) / "sig", fmt="json")
        art.write(sig, run_id="r-meta", model_sha="deadbeef")
        back = art.read()
    assert back.metadata.get("run_id") == "r-meta"
    assert back.metadata.get("model_sha") == "deadbeef"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_round_trip_json_preserves_floats,
        test_round_trip_without_kelly,
        test_manifest_contains_provenance,
        test_read_manifest_independently,
        test_missing_manifest_raises,
        test_extra_cannot_override_core_keys,
        test_roundtrip_metadata_has_run_and_model,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\ndata.signal: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
