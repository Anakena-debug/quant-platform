"""S28 PR2 — research-run schema + manifest tests (AC2.1–AC2.7).

Validates ``quantstrat/tests/research/_research_manifest.py``. Runs under
plain ``uv run --directory quantstrat pytest`` — no ``--extra research``
flag because PR2 schemas must remain backend-agnostic (AC2.7).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from research._research_manifest import (
    DatasetManifest,
    MetricsSchema,
    RunIdentity,
    TagsSchema,
    build_dataset_manifest,
    build_run_identity,
    canonical_json,
    sha256_json,
)

MODULE_FILE: Path = Path(__file__).resolve().parent / "_research_manifest.py"


# ─── Expected schema key-sets (AC2.1) ───────────────────────────────────

EXPECTED_RUN_IDENTITY_KEYS = {
    "git_sha",
    "branch",
    "sprint",
    "source_sprint",
    "experiment_family",
    "run_family",
}
EXPECTED_DATASET_MANIFEST_KEYS = {
    "data_source",
    "data_frequency",
    "universe_name",
    "parsed_ticker_count",
    "available_ticker_count",
    "start_date",
    "end_date",
    "as_of",
    "row_count",
    "missing_rate_summary",
    "duplicate_count",
    "schema_hash",
    "dataset_manifest_hash",
    "timezone",
    "corporate_action_policy",
}
EXPECTED_METRICS_KEYS = {
    "n_tickers",
    "n_train",
    "tradeable_count",
    "tradeable_fraction",
    "expected_return_mean",
    "expected_return_std",
    "interval_half_width_median",
    "interval_half_width_p90",
    "max_abs_expected_return",
}
EXPECTED_TAGS_KEYS = {"git_sha", "sprint", "source_sprint", "run_family"}


# ─── Panel + builder-kwargs fixtures ────────────────────────────────────


def _panel_a() -> pd.DataFrame:
    """3-ticker × 2-date panel — distinct shape from panel_b."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "session_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "price": [100.0, 200.0, 300.0, 101.0, 201.0, 301.0],
        }
    )


def _panel_b() -> pd.DataFrame:
    """2-ticker × 3-date panel — same row_count as panel_a, different shape."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "session_date": pd.to_datetime(
                ["2024-01-02"] * 2 + ["2024-01-03"] * 2 + ["2024-01-04"] * 2
            ),
            "price": [100.0, 200.0, 101.0, 201.0, 102.0, 202.0],
        }
    )


def _kwargs_a() -> dict[str, Any]:
    return {
        "panel": _panel_a(),
        "universe_name": "UNI_A",
        "data_source": "quantdata.MarketData",
        "data_frequency": "daily",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "as_of": "2024-01-03",
        "parsed_tickers": ("AAA", "BBB", "CCC"),
        "timezone": None,
        "corporate_action_policy": "yfinance_adjusted_close",
    }


def _kwargs_b() -> dict[str, Any]:
    return {
        "panel": _panel_b(),
        "universe_name": "UNI_B",
        "data_source": "quantdata.MarketData",
        "data_frequency": "daily",
        "start_date": "2024-01-02",
        "end_date": "2024-01-04",
        "as_of": "2024-01-04",
        "parsed_tickers": ("AAA", "BBB"),
        "timezone": "UTC",
        "corporate_action_policy": None,
    }


# ─── AC2.1 — schema key-sets present ────────────────────────────────────


def test_run_identity_schema_keys() -> None:
    """AC2.1 — RunIdentity declares the documented keys."""
    assert set(RunIdentity.__annotations__.keys()) == EXPECTED_RUN_IDENTITY_KEYS


def test_dataset_manifest_schema_keys() -> None:
    """AC2.1 — DatasetManifest declares the documented keys."""
    assert set(DatasetManifest.__annotations__.keys()) == EXPECTED_DATASET_MANIFEST_KEYS


def test_metrics_schema_keys() -> None:
    """AC2.1 — MetricsSchema declares the AC3.metrics keys (signal-side only)."""
    assert set(MetricsSchema.__annotations__.keys()) == EXPECTED_METRICS_KEYS


def test_tags_schema_keys() -> None:
    """AC2.1 — TagsSchema declares the AC3.tags keys."""
    assert set(TagsSchema.__annotations__.keys()) == EXPECTED_TAGS_KEYS


# ─── AC2.2 — builders return well-formed instances ──────────────────────


def test_build_run_identity_returns_well_formed() -> None:
    """AC2.2 — build_run_identity captures git_sha + branch and passthroughs."""
    ri = build_run_identity(
        sprint="s28",
        source_sprint="s27",
        experiment_family="afml",
        run_family="smoke",
    )
    assert set(ri.keys()) == EXPECTED_RUN_IDENTITY_KEYS
    assert isinstance(ri["git_sha"], str) and len(ri["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in ri["git_sha"])
    assert isinstance(ri["branch"], str) and ri["branch"]
    assert ri["sprint"] == "s28"
    assert ri["source_sprint"] == "s27"
    assert ri["experiment_family"] == "afml"
    assert ri["run_family"] == "smoke"


def test_build_run_identity_accepts_none_source_sprint() -> None:
    """AC2.2 — source_sprint may be None for sprints with no upstream."""
    ri = build_run_identity(
        sprint="s28",
        source_sprint=None,
        experiment_family="afml",
        run_family="smoke",
    )
    assert ri["source_sprint"] is None


def test_build_dataset_manifest_returns_well_formed() -> None:
    """AC2.2 — build_dataset_manifest returns a full DatasetManifest."""
    manifest = build_dataset_manifest(**_kwargs_a())
    assert set(manifest.keys()) == EXPECTED_DATASET_MANIFEST_KEYS
    assert manifest["data_source"] == "quantdata.MarketData"
    assert manifest["data_frequency"] == "daily"
    assert manifest["universe_name"] == "UNI_A"
    assert manifest["parsed_ticker_count"] == 3
    assert manifest["available_ticker_count"] == 3
    assert manifest["row_count"] == 6
    assert manifest["duplicate_count"] == 0
    assert isinstance(manifest["missing_rate_summary"], dict)
    assert all(0.0 <= v <= 1.0 for v in manifest["missing_rate_summary"].values())
    assert isinstance(manifest["schema_hash"], str)
    assert len(manifest["schema_hash"]) == 64
    assert isinstance(manifest["dataset_manifest_hash"], str)
    assert len(manifest["dataset_manifest_hash"]) == 64
    assert manifest["timezone"] is None
    assert manifest["corporate_action_policy"] == "yfinance_adjusted_close"


# ─── AC2.3 — JSON round-trip byte-equal ─────────────────────────────────


@pytest.mark.parametrize(
    "kwargs_factory",
    [_kwargs_a, _kwargs_b],
    ids=["panel_a", "panel_b"],
)
def test_dataset_manifest_json_round_trip(kwargs_factory) -> None:
    """AC2.3 — canonical JSON survives json.loads → canonical JSON byte-equal."""
    manifest = build_dataset_manifest(**kwargs_factory())
    payload = canonical_json(manifest)
    roundtrip = canonical_json(json.loads(payload))
    assert payload == roundtrip


def test_run_identity_json_round_trip() -> None:
    """AC2.3 — RunIdentity (with source_sprint=None) round-trips byte-equal."""
    ri = build_run_identity(
        sprint="s28",
        source_sprint=None,
        experiment_family="afml",
        run_family="smoke",
    )
    payload = canonical_json(ri)
    roundtrip = canonical_json(json.loads(payload))
    assert payload == roundtrip


# ─── AC2.4 — stable hashes + distinct hashes across shapes ──────────────


@pytest.mark.parametrize(
    "kwargs_factory",
    [_kwargs_a, _kwargs_b],
    ids=["panel_a", "panel_b"],
)
def test_dataset_manifest_hashes_stable(kwargs_factory) -> None:
    """AC2.4 — two consecutive builder calls produce equal hashes."""
    m1 = build_dataset_manifest(**kwargs_factory())
    m2 = build_dataset_manifest(**kwargs_factory())
    assert m1["schema_hash"] == m2["schema_hash"]
    assert m1["dataset_manifest_hash"] == m2["dataset_manifest_hash"]


def test_dataset_manifest_hash_distinguishes_shapes() -> None:
    """AC2.4 — distinct manifest shapes produce distinct hashes (no collision)."""
    m_a = build_dataset_manifest(**_kwargs_a())
    m_b = build_dataset_manifest(**_kwargs_b())
    # schema_hashes can coincide (both panels share the same columns/dtypes)
    # but the full-manifest hash must always differ on distinct shapes.
    assert m_a["dataset_manifest_hash"] != m_b["dataset_manifest_hash"]


def test_canonical_json_and_sha256_helpers_are_deterministic() -> None:
    """sanity — canonical_json + sha256_json are deterministic and reactive."""
    obj1 = {"b": 2, "a": 1, "nested": {"y": 0, "x": [3, 2, 1]}}
    obj2 = {"a": 1, "b": 2, "nested": {"x": [3, 2, 1], "y": 0}}
    assert canonical_json(obj1) == canonical_json(obj2)
    assert sha256_json(obj1) == sha256_json(obj2)
    assert sha256_json(obj1) != sha256_json({"a": 1})


# ─── AC2.5 — small serialisation + no raw-data leakage ──────────────────

SENTINEL_PRICE: float = 1234567.890123


def test_dataset_manifest_under_8kib_and_no_raw_leakage() -> None:
    """AC2.5 — serialised manifest < 8 KiB and contains no sentinel raw values."""
    sentinel_prices = [
        SENTINEL_PRICE,
        SENTINEL_PRICE + 1.0,
        SENTINEL_PRICE + 2.0,
        SENTINEL_PRICE + 3.0,
    ]
    sentinel_panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "session_date": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
            "price": sentinel_prices,
        }
    )
    manifest = build_dataset_manifest(
        panel=sentinel_panel,
        universe_name="UNI_SENT",
        data_source="quantdata.MarketData",
        data_frequency="daily",
        start_date="2024-01-02",
        end_date="2024-01-03",
        as_of="2024-01-03",
        parsed_tickers=("AAA", "BBB"),
    )
    payload = canonical_json(manifest)
    payload_bytes = payload.encode("utf-8")
    assert len(payload_bytes) < 8 * 1024, (
        f"Serialised manifest is {len(payload_bytes)} bytes "
        f"(>= 8 KiB cap); raw row data may have leaked."
    )
    for raw_value in sentinel_prices:
        assert str(raw_value) not in payload, (
            f"Raw price {raw_value!r} leaked into serialised manifest."
        )


# ─── AC2.6 — malformed-input rejection ──────────────────────────────────


def test_empty_parsed_tickers_raises() -> None:
    kwargs = _kwargs_a()
    kwargs["parsed_tickers"] = ()
    with pytest.raises(ValueError, match="parsed_tickers"):
        build_dataset_manifest(**kwargs)


def test_end_date_before_start_date_raises() -> None:
    kwargs = _kwargs_a()
    kwargs["start_date"] = "2024-01-10"
    kwargs["end_date"] = "2024-01-02"
    with pytest.raises(ValueError, match="end_date"):
        build_dataset_manifest(**kwargs)


def test_as_of_outside_window_raises() -> None:
    kwargs = _kwargs_a()
    kwargs["as_of"] = "2025-01-01"
    with pytest.raises(ValueError, match="as_of"):
        build_dataset_manifest(**kwargs)


def test_empty_panel_yields_non_finite_missing_rate_raises() -> None:
    empty_panel = pd.DataFrame(
        {
            "ticker": pd.Series([], dtype=str),
            "session_date": pd.Series([], dtype="datetime64[ns]"),
            "price": pd.Series([], dtype=float),
        }
    )
    kwargs = _kwargs_a()
    kwargs["panel"] = empty_panel
    with pytest.raises(ValueError, match="non-finite"):
        build_dataset_manifest(**kwargs)


def test_empty_universe_name_raises() -> None:
    kwargs = _kwargs_a()
    kwargs["universe_name"] = ""
    with pytest.raises(ValueError, match="universe_name"):
        build_dataset_manifest(**kwargs)


def test_panel_missing_ticker_column_raises() -> None:
    """Builder fails loud when the panel has no ``ticker`` column."""
    no_ticker_panel = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "price": [100.0, 101.0],
        }
    )
    kwargs = _kwargs_a()
    kwargs["panel"] = no_ticker_panel
    with pytest.raises(ValueError, match="ticker"):
        build_dataset_manifest(**kwargs)


def test_build_run_identity_rejects_empty_sprint() -> None:
    with pytest.raises(ValueError, match="sprint"):
        build_run_identity(
            sprint="",
            source_sprint=None,
            experiment_family="afml",
            run_family="smoke",
        )


def test_build_run_identity_rejects_empty_run_family() -> None:
    with pytest.raises(ValueError, match="run_family"):
        build_run_identity(
            sprint="s28",
            source_sprint=None,
            experiment_family="afml",
            run_family="",
        )


# ─── AC2.7 — structural: no mlflow import ───────────────────────────────


def test_module_does_not_import_mlflow() -> None:
    """AC2.7 — ast walk confirms `_research_manifest.py` never imports mlflow."""
    tree = ast.parse(MODULE_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("mlflow"), (
                    f"_research_manifest.py must not `import {alias.name}` (AC2.7)."
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("mlflow"), (
                f"_research_manifest.py must not `from {module} import ...` (AC2.7)."
            )
