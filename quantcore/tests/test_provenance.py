"""quantcore.provenance — content-addressed lineage (fingerprint, manifest, verification)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcore.provenance import (
    CodeVersion,
    RunManifest,
    build_manifest,
    code_version,
    frame_fingerprint,
    panels_fingerprint,
    verify_manifest,
)

_NOW = "2026-06-17T00:00:00+00:00"
_CLEAN = CodeVersion(commit="a" * 40, dirty=False, quantcore_version="0.1.0")
_DIRTY = CodeVersion(commit="a" * 40, dirty=True, quantcore_version="0.1.0")
_OTHER_COMMIT = CodeVersion(commit="b" * 40, dirty=False, quantcore_version="0.1.0")


def _frame(seed: int = 0, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"date": np.arange(n), "asset": ["A"] * n, "forward_return": rng.standard_normal(n)}
    )


def test_frame_fingerprint_is_deterministic():
    assert frame_fingerprint(_frame(1)) == frame_fingerprint(_frame(1))


def test_frame_fingerprint_changes_with_values():
    assert frame_fingerprint(_frame(1)) != frame_fingerprint(_frame(2))


def test_frame_fingerprint_is_order_insensitive():
    f = _frame(3)
    shuffled = f.sample(frac=1.0, random_state=7)[list(reversed(f.columns))]
    assert frame_fingerprint(f) == frame_fingerprint(shuffled)


def test_panels_fingerprint_key_order_insensitive():
    a, b = _frame(4), _frame(5)
    assert panels_fingerprint({"x": a, "y": b}) == panels_fingerprint({"y": b, "x": a})


def test_code_version_reports_quantcore_version():
    cv = code_version()
    assert cv.quantcore_version == "0.1.0"
    # In this repo git is present, so commit is a 40-hex SHA; tolerate None (installed wheel).
    assert cv.commit is None or (
        len(cv.commit) == 40 and all(c in "0123456789abcdef" for c in cv.commit)
    )


def test_run_id_is_timestamp_independent():
    f = _frame(6)
    m1 = build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_CLEAN)
    m2 = build_manifest(f, params={"fdr": 0.1}, now="2099-01-01T00:00:00+00:00", code=_CLEAN)
    assert m1.run_id == m2.run_id  # same data+code+params -> same id, regardless of when
    assert m1.created_utc != m2.created_utc


def test_run_id_changes_with_params_data_and_commit():
    f = _frame(6)
    base = build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_CLEAN).run_id
    assert build_manifest(f, params={"fdr": 0.2}, now=_NOW, code=_CLEAN).run_id != base
    assert build_manifest(_frame(7), params={"fdr": 0.1}, now=_NOW, code=_CLEAN).run_id != base
    assert build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_OTHER_COMMIT).run_id != base


def test_manifest_records_shape_and_source():
    f = _frame(8, n=25)
    m = build_manifest(f, params={}, source="panel.parquet", now=_NOW, code=_CLEAN)
    assert m.n_rows == 25 and m.source == "panel.parquet"
    assert m.data_fingerprint == frame_fingerprint(f)


def test_manifest_dict_round_trip():
    m = build_manifest(_frame(9), params={"fdr": 0.1}, now=_NOW, code=_CLEAN)
    back = RunManifest.from_dict(m.to_dict())
    assert back == m
    assert isinstance(back.code, CodeVersion) and back.code.commit == "a" * 40


def test_verify_matches_on_same_data_and_code():
    f = _frame(10)
    m = build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_CLEAN)
    check = verify_manifest(m, f, code=_CLEAN)
    assert check.data_matches and check.code_matches and check.run_id_matches
    assert check.reproducible


def test_verify_detects_data_drift():
    m = build_manifest(_frame(11), params={"fdr": 0.1}, now=_NOW, code=_CLEAN)
    check = verify_manifest(m, _frame(12), code=_CLEAN)
    assert not check.data_matches and not check.reproducible
    assert "data fingerprint differs" in check.detail


def test_verify_detects_code_drift():
    f = _frame(13)
    m = build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_CLEAN)
    check = verify_manifest(m, f, code=_OTHER_COMMIT)
    assert not check.code_matches and not check.reproducible
    assert "code commit differs" in check.detail


def test_verify_refuses_to_certify_dirty_tree():
    f = _frame(14)
    m = build_manifest(f, params={"fdr": 0.1}, now=_NOW, code=_DIRTY)
    check = verify_manifest(m, f, code=_DIRTY)
    # Data and id still match, but a dirty tree is not reproducible.
    assert check.data_matches and check.run_id_matches and not check.reproducible
    assert "dirty" in check.detail
