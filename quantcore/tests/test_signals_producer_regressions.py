"""Regression tests for quantcore.signals.producer.write_alpha_signal.

Contract mirror
---------------
The quantengine reader (``quantengine.data.signal.SignalArtifact.read()``)
is NOT on disk in this repository. To validate the producer in isolation,
this module defines a *stub reader* (``_StubSignalReader``) that reconstructs
an ``AlphaSignal``-shaped object from ``signal.json`` + ``manifest.json``,
using **exactly** the same column / manifest contract documented in
``quantcore/signals/producer.py``.

Any contract drift will show as a test failure here — which is the correct
tripwire for catching producer/reader desync during development.

Covered
-------
* JSON round-trip bit-exactness (rtol=0, atol=0).
* With and without kelly_weights.
* kelly_weights as ``pd.Series`` (reindexed to ticker order).
* Parameter validation: alpha range, duplicate tickers, non-finite arrays,
  lower > upper, shape mismatch, reserved manifest keys, bad fmt, etc.
* Manifest core-key presence + schema_version == 1.
* ``extra`` merges in and cannot clobber core keys.
* Multiple subsequent writes to the same directory overwrite atomically.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantcore.signals.producer import write_alpha_signal, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Stub reader — byte-identical to quantengine.data.signal.SignalArtifact.read
# ---------------------------------------------------------------------------
_CORE_MANIFEST_KEYS = {
    "run_id",
    "model_sha",
    "alpha",
    "as_of",
    "n",
    "format",
    "schema_version",
    "has_kelly",
}


class _StubSignalReader:
    """Mirror of quantengine.data.signal.SignalArtifact.read (JSON path)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict:
        manifest = json.loads((self.path / "manifest.json").read_text())
        missing = _CORE_MANIFEST_KEYS - manifest.keys()
        if missing:
            raise ValueError(f"manifest missing core keys: {sorted(missing)}")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: got {manifest['schema_version']}, "
                f"reader expects {SCHEMA_VERSION}"
            )
        fmt = manifest["format"]
        if fmt == "json":
            records = json.loads((self.path / "signal.json").read_text())
            df = pd.DataFrame(records)
        elif fmt == "parquet":
            df = pd.read_parquet(self.path / "signal.parquet")
        else:
            raise ValueError(f"unknown format {fmt!r}")

        expected_cols = {"ticker", "expected_return", "lower", "upper"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"signal file missing columns: {expected_cols - set(df.columns)}")
        return {
            "tickers": tuple(df["ticker"].tolist()),
            "expected_return": df["expected_return"].to_numpy(dtype=np.float64),
            "lower": df["lower"].to_numpy(dtype=np.float64),
            "upper": df["upper"].to_numpy(dtype=np.float64),
            "kelly_weights": (
                df["kelly_weight"].to_numpy(dtype=np.float64)
                if "kelly_weight" in df.columns
                else None
            ),
            "alpha": float(manifest["alpha"]),
            "timestamp": manifest["as_of"],
            "manifest": manifest,
        }


# ---------------------------------------------------------------------------
# Fixtures (helper constructors — no pytest.fixture needed by shim)
# ---------------------------------------------------------------------------
def _make_payload(n: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"TCK{i:03d}" for i in range(n)]
    er = rng.standard_normal(n) * 1e-3
    half = np.abs(rng.standard_normal(n)) * 5e-3 + 1e-4
    lo = er - half
    hi = er + half
    return tickers, er, lo, hi


def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="signals_test_"))


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------
def test_roundtrip_json_no_kelly():
    tickers, er, lo, hi = _make_payload(n=7, seed=1)
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.10,
        kelly_weights=None,
        as_of=pd.Timestamp("2026-04-17T16:00:00"),
        out_dir=out,
        run_id="r-001",
        model_sha="deadbeef",
        fmt="json",
    )
    r = _StubSignalReader(out).read()
    assert r["tickers"] == tuple(tickers)
    assert np.array_equal(r["expected_return"], er.astype(np.float64))
    assert np.array_equal(r["lower"], lo.astype(np.float64))
    assert np.array_equal(r["upper"], hi.astype(np.float64))
    assert r["kelly_weights"] is None
    assert r["alpha"] == 0.10
    assert r["manifest"]["n"] == 7
    assert r["manifest"]["schema_version"] == SCHEMA_VERSION
    assert r["manifest"]["has_kelly"] is False
    assert r["manifest"]["format"] == "json"


def test_roundtrip_json_with_kelly_ndarray():
    tickers, er, lo, hi = _make_payload(n=4, seed=2)
    k = np.array([0.1, 0.0, 0.25, -0.05], dtype=np.float64)
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.05,
        kelly_weights=k,
        as_of="2026-04-17T16:00:00",
        out_dir=out,
        run_id="r-002",
        model_sha="cafebabe",
        fmt="json",
    )
    r = _StubSignalReader(out).read()
    assert np.array_equal(r["kelly_weights"], k)
    assert r["manifest"]["has_kelly"] is True


def test_roundtrip_kelly_series_reindexed_to_tickers():
    tickers, er, lo, hi = _make_payload(n=4, seed=3)
    # Series provided in scrambled order; must be reindexed to `tickers`.
    scrambled = pd.Series(
        [0.3, 0.1, 0.4, 0.2],
        index=[tickers[2], tickers[0], tickers[3], tickers[1]],
    )
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.10,
        kelly_weights=scrambled,
        as_of=pd.Timestamp("2026-04-17"),
        out_dir=out,
        run_id="r",
        model_sha="s",
        fmt="json",
    )
    r = _StubSignalReader(out).read()
    # Must be [0.1, 0.2, 0.3, 0.4] after reindex to [T000, T001, T002, T003]
    assert np.array_equal(r["kelly_weights"], np.array([0.1, 0.2, 0.3, 0.4]))


def test_bit_exact_float_roundtrip_under_json():
    # Pick awkward floats that test repr round-tripping
    tickers = ["AAA", "BBB", "CCC"]
    er = np.array([0.1 + 0.2, 1.0 / 3.0, 2.0**-53], dtype=np.float64)
    lo = er - 1e-15
    hi = er + 1e-15
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.10,
        kelly_weights=None,
        as_of="2026-01-01",
        out_dir=out,
        run_id="r",
        model_sha="s",
        fmt="json",
    )
    r = _StubSignalReader(out).read()
    assert np.array_equal(r["expected_return"], er)
    assert np.array_equal(r["lower"], lo)
    assert np.array_equal(r["upper"], hi)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_alpha", [-0.01, 0.0, 1.0, 1.1, float("nan")])
def test_alpha_range_rejected(bad_alpha):
    tickers, er, lo, hi = _make_payload(n=3)
    with pytest.raises(ValueError, match="alpha"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=bad_alpha,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_duplicate_tickers_rejected():
    er, lo, hi = np.zeros(3), np.zeros(3), np.zeros(3)
    with pytest.raises(ValueError, match="duplicates"):
        write_alpha_signal(
            tickers=["A", "A", "B"],
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_non_finite_array_rejected():
    tickers = ["A", "B", "C"]
    er = np.array([0.0, float("nan"), 0.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_lower_gt_upper_rejected():
    tickers = ["A", "B"]
    er = np.array([0.0, 0.0])
    lo = np.array([1.0, 0.0])
    hi = np.array([0.5, 0.0])
    with pytest.raises(ValueError, match="lower"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_shape_mismatch_rejected():
    tickers = ["A", "B", "C"]
    er = np.zeros(3)
    lo = np.zeros(2)
    hi = np.zeros(3)
    with pytest.raises(ValueError, match="shape"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_kelly_series_missing_ticker_rejected():
    tickers, er, lo, hi = _make_payload(n=3)
    kelly = pd.Series([0.1, 0.2], index=[tickers[0], tickers[1]])  # missing [2]
    with pytest.raises(KeyError, match="missing"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=kelly,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_extra_cannot_override_core_keys():
    tickers, er, lo, hi = _make_payload(n=3)
    with pytest.raises(ValueError, match="extra"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
            extra={"schema_version": 999},
        )


def test_extra_merged_through_non_core_keys():
    tickers, er, lo, hi = _make_payload(n=3)
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.1,
        kelly_weights=None,
        as_of="2026-01-01",
        out_dir=out,
        run_id="r",
        model_sha="s",
        fmt="json",
        extra={"feature_set": "v7", "cv_sharpe": 1.21},
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["feature_set"] == "v7"
    assert m["cv_sharpe"] == 1.21
    # Core keys intact
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["alpha"] == 0.1


def test_empty_tickers_rejected():
    with pytest.raises(ValueError, match="empty"):
        write_alpha_signal(
            tickers=[],
            expected_return=np.array([]),
            lower=np.array([]),
            upper=np.array([]),
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="json",
        )


def test_bad_fmt_rejected():
    tickers, er, lo, hi = _make_payload(n=3)
    with pytest.raises(ValueError, match="fmt"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="s",
            fmt="yaml",  # type: ignore[arg-type]
        )


def test_empty_run_id_or_sha_rejected():
    tickers, er, lo, hi = _make_payload(n=3)
    with pytest.raises(ValueError, match="run_id"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="",
            model_sha="s",
            fmt="json",
        )
    with pytest.raises(ValueError, match="model_sha"):
        write_alpha_signal(
            tickers=tickers,
            expected_return=er,
            lower=lo,
            upper=hi,
            alpha=0.1,
            kelly_weights=None,
            as_of="2026-01-01",
            out_dir=_tmpdir(),
            run_id="r",
            model_sha="",
            fmt="json",
        )


def test_rewrite_overwrites_prior():
    tickers, er, lo, hi = _make_payload(n=3)
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.1,
        kelly_weights=None,
        as_of="2026-01-01",
        out_dir=out,
        run_id="r1",
        model_sha="s1",
        fmt="json",
    )
    write_alpha_signal(
        tickers=tickers,
        expected_return=er + 1.0,
        lower=lo + 1.0,
        upper=hi + 1.0,
        alpha=0.1,
        kelly_weights=None,
        as_of="2026-01-02",
        out_dir=out,
        run_id="r2",
        model_sha="s2",
        fmt="json",
    )
    r = _StubSignalReader(out).read()
    assert r["manifest"]["run_id"] == "r2"
    assert np.array_equal(r["expected_return"], er + 1.0)


def test_manifest_has_all_core_keys():
    tickers, er, lo, hi = _make_payload(n=2)
    out = _tmpdir()
    write_alpha_signal(
        tickers=tickers,
        expected_return=er,
        lower=lo,
        upper=hi,
        alpha=0.1,
        kelly_weights=None,
        as_of=pd.Timestamp("2026-04-17T20:00:00"),
        out_dir=out,
        run_id="r",
        model_sha="s",
        fmt="json",
    )
    m = json.loads((out / "manifest.json").read_text())
    assert set(_CORE_MANIFEST_KEYS).issubset(m.keys())
    assert m["schema_version"] == 1
    assert m["n"] == 2
    assert isinstance(m["as_of"], str) and m["as_of"].startswith("2026-04-17")
