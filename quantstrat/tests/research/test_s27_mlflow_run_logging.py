"""S28 PR3 — log the S27 realistic AFML run to local MLflow (AC3.1–AC3.7).

Logs the S27 realistic-data producer output as the first canonical S28
research run. The run lives under pytest's ``tmp_path`` via PR1's
``with_local_tracking`` helper, carries PR2's run-identity and dataset
manifest schemas, and never leaks execution-side or raw-panel data.

This module deliberately does NOT import any of ``PaperBroker``,
``PortfolioState``, ``RebalanceEngine``, ``run_daily_cycle``,
``IBKRBroker``, ``IBKRConnection``, or ``ib_async`` — PR3 is signal-side
only (AC3.5 / ARCHITECTURE.md invariant 5).

Runs under ``uv run --directory quantstrat --extra research pytest`` (PR3 imports the
PR1 helper, which imports ``mlflow``).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest

from research._mlflow_tracking import log_research_run, with_local_tracking
from research._realistic_panel import (
    END_DATE as PANEL_END_DATE,
    START_DATE as PANEL_START_DATE,
    _load_dj30_tickers,
)
from research._research_manifest import (
    build_dataset_manifest,
    build_run_identity,
)
from research.test_realistic_afml_signal_producer import (
    AS_OF as PRODUCER_AS_OF,
    CALIBRATION_FRACTION,
    CONFORMAL_ALPHA,
    RIDGE_ALPHA,
    SEED,
    run_realistic_afml,
)

# ─── Path resolution (AC1.2 / AC3.1) ────────────────────────────────────

TEST_FILE: Path = Path(__file__).resolve()
QUANTSTRAT_DIR: Path = TEST_FILE.parents[2]
REPO_ROOT: Path = TEST_FILE.parents[3]

# ─── AC3.4 — forbidden artifact extensions ──────────────────────────────

FORBIDDEN_ARTIFACT_EXTENSIONS: tuple[str, ...] = (
    ".parquet",
    ".duckdb",
    ".pkl",
    ".npy",
    ".npz",
)

# ─── AC3.5 — forbidden execution-side import symbols / modules ─────────

FORBIDDEN_IMPORT_SYMBOLS: tuple[str, ...] = (
    "PaperBroker",
    "PortfolioState",
    "RebalanceEngine",
    "run_daily_cycle",
    "IBKRBroker",
    "IBKRConnection",
    "ib_async",
    "quantengine.execution",
    "quantengine.portfolio",
    "quantengine.runtime",
)

# ─── AC3.artifacts — canonical artifact filenames ───────────────────────

ARTIFACT_FILENAMES: tuple[str, ...] = (
    "config.json",
    "dataset_manifest.json",
    "metrics.json",
    "signal_summary.json",
)

# Per-artifact-file size cap (plan §6 reproducibility contract).
ARTIFACT_TOTAL_BUDGET_BYTES: int = 1024 * 1024  # 1 MiB total per AC1.5 / AC3.4


# ─── Helpers ────────────────────────────────────────────────────────────


def _assert_no_repo_mlruns() -> None:
    """AC1.2 / AC3.1 — fail loud if mlruns/ exists at repo root or quantstrat."""
    for label, path in (
        ("repo-root", REPO_ROOT / "mlruns"),
        ("quantstrat", QUANTSTRAT_DIR / "mlruns"),
    ):
        if path.exists():
            pytest.fail(
                f"Forbidden {label} mlruns/ at {path}; "
                f"operator must remove it (S28 §3 AC1.2 / AC3.1)."
            )


def _build_params() -> dict[str, Any]:
    """AC3.params — exactly these 10 MLflow params, all type-stable."""
    return {
        "universe": "DJ30",
        "start_date": PANEL_START_DATE,
        "end_date": PANEL_END_DATE,
        "as_of": PRODUCER_AS_OF.strftime("%Y-%m-%d"),
        "features": "mom5,z20,vol20",
        "label": "1d_forward_log_return",
        "model": f"Ridge(alpha={RIDGE_ALPHA})",
        "conformal_alpha": CONFORMAL_ALPHA,
        "calibration_fraction": CALIBRATION_FRACTION,
        "seed": SEED,
    }


def _build_metrics(out: dict[str, Any]) -> dict[str, float]:
    """AC3.metrics — signal-side only. NO PnL / NAV / cash / fills / orders."""
    tickers = list(out["tickers"])
    expected_return = np.asarray(out["expected_return"], dtype=np.float64)
    lower = np.asarray(out["lower"], dtype=np.float64)
    upper = np.asarray(out["upper"], dtype=np.float64)
    n_tickers = int(len(tickers))
    n_train = int(out["n_train"])
    tradeable_mask = (lower > 0.0) | (upper < 0.0)
    tradeable_count = int(tradeable_mask.sum())
    half_width = (upper - lower) / 2.0
    return {
        "n_tickers": float(n_tickers),
        "n_train": float(n_train),
        "tradeable_count": float(tradeable_count),
        "tradeable_fraction": float(tradeable_count / n_tickers),
        "expected_return_mean": float(expected_return.mean()),
        "expected_return_std": float(expected_return.std(ddof=0)),
        "interval_half_width_median": float(np.median(half_width)),
        "interval_half_width_p90": float(np.quantile(half_width, 0.90)),
        "max_abs_expected_return": float(np.abs(expected_return).max()),
    }


def _thin_panel_for_manifest(parsed_tickers: list[str]) -> pd.DataFrame:
    """Schema-faithful thin view of the DJ30 panel.

    PR3 logs metadata only; we do not re-load the real ~22 000-row panel
    (plan §5 PR3 step 6: "does NOT re-load the panel, to keep PR3 cheap;
    the manifest is metadata-only"). The thin view carries the real
    ticker set and the same schema (column names + dtypes) as
    ``_realistic_panel.load_dj30_panel()`` so ``schema_hash`` is stable
    and meaningful across reruns.
    """
    n = len(parsed_tickers)
    panel = pd.DataFrame(
        {
            "ticker": list(parsed_tickers),
            "session_date": pd.to_datetime([PANEL_START_DATE] * n),
            "price": [0.0] * n,
        }
    )
    panel["ticker"] = panel["ticker"].astype("object")
    panel["session_date"] = panel["session_date"].astype("datetime64[ns]")
    panel["price"] = panel["price"].astype("float64")
    return panel


def _build_dataset_manifest_dict() -> dict[str, Any]:
    """AC3.artifacts — dataset_manifest.json contents (metadata only)."""
    parsed_tickers = sorted(_load_dj30_tickers())
    panel_view = _thin_panel_for_manifest(parsed_tickers)
    manifest = build_dataset_manifest(
        panel=panel_view,
        universe_name="DJ30",
        data_source="quantdata.MarketData",
        data_frequency="daily",
        start_date=PANEL_START_DATE,
        end_date=PANEL_END_DATE,
        as_of=PRODUCER_AS_OF.strftime("%Y-%m-%d"),
        parsed_tickers=parsed_tickers,
        timezone=None,
        corporate_action_policy="yfinance_adjusted_close",
    )
    return dict(manifest)


def _build_signal_summary(out: dict[str, Any]) -> list[dict[str, Any]]:
    """AC3.artifacts — per-ticker rows sorted by ticker, full float repr precision."""
    tickers = list(out["tickers"])
    expected_return = [float(x) for x in out["expected_return"]]
    lower = [float(x) for x in out["lower"]]
    upper = [float(x) for x in out["upper"]]
    kelly_weights = [float(x) for x in out["kelly_weights"]]
    sort_idx = sorted(range(len(tickers)), key=lambda i: tickers[i])
    rows: list[dict[str, Any]] = []
    for i in sort_idx:
        rows.append(
            {
                "ticker": tickers[i],
                "expected_return": expected_return[i],
                "lower": lower[i],
                "upper": upper[i],
                "kelly_weight": kelly_weights[i],
                "tradeable": bool(lower[i] > 0.0 or upper[i] < 0.0),
            }
        )
    return rows


def _write_json(path: Path, obj: Any) -> None:
    """Deterministic JSON write: sorted keys, 2-space indent, trailing newline."""
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def _prepare_artifacts(
    *,
    artifact_dir: Path,
    params: dict[str, Any],
    run_identity: dict[str, Any],
    dataset_manifest: dict[str, Any],
    metrics: dict[str, float],
    signal_summary: list[dict[str, Any]],
) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path = artifact_dir / "config.json"
    manifest_path = artifact_dir / "dataset_manifest.json"
    metrics_path = artifact_dir / "metrics.json"
    summary_path = artifact_dir / "signal_summary.json"
    _write_json(config_path, {"params": params, "run_identity": run_identity})
    _write_json(manifest_path, dataset_manifest)
    _write_json(metrics_path, metrics)
    _write_json(summary_path, signal_summary)
    return {
        "config": config_path,
        "dataset_manifest": manifest_path,
        "metrics": metrics_path,
        "signal_summary": summary_path,
    }


def _file_hashes(directory: Path) -> dict[str, str]:
    """SHA-256 hex digest of every file directly under ``directory``."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def _downloaded_files(run_id: str) -> dict[str, Path]:
    root = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
    return {p.name: p for p in root.iterdir() if p.is_file()}


# ─── AC3.5 — structural negative on execution-side imports ──────────────


def test_module_has_no_execution_side_imports() -> None:
    """AC3.5 — ast walk confirms no execution / IBKR / ib_async imports."""
    tree = ast.parse(TEST_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in FORBIDDEN_IMPORT_SYMBOLS:
                    assert forbidden not in alias.name, (
                        f"AC3.5: forbidden import {alias.name!r} contains {forbidden!r}."
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for forbidden in FORBIDDEN_IMPORT_SYMBOLS:
                assert forbidden not in module, (
                    f"AC3.5: forbidden import from {module!r} contains {forbidden!r}."
                )
                for alias in node.names:
                    assert forbidden not in alias.name, (
                        f"AC3.5: forbidden import name {alias.name!r} contains {forbidden!r}."
                    )


# ─── Main test: AC3.1 / AC3.params / AC3.metrics / AC3.tags /
#                AC3.artifacts / AC3.4 / AC3.6 ─────────────────────────


def test_s27_run_logged_to_mlflow(tmp_path: Path) -> None:
    """Log the S27 producer output as the first canonical S28 research run.

    Covers AC3.1 (run under tmp_path), AC3.params, AC3.metrics, AC3.tags,
    AC3.artifacts (4 JSON files), AC3.4 (no raw / no forbidden extension /
    < 1 MiB), and AC3.6 (byte-equal artifacts across two distinct runs).
    """
    _assert_no_repo_mlruns()

    expected_params = _build_params()
    expected_dataset_manifest = _build_dataset_manifest_dict()

    out_a = run_realistic_afml()
    expected_metrics = _build_metrics(out_a)
    expected_signal_summary = _build_signal_summary(out_a)

    artifact_dir_a = tmp_path / "artifacts_a"
    artifact_dir_b = tmp_path / "artifacts_b"

    expected_tracking_uri = f"file://{tmp_path}/mlruns"

    with with_local_tracking(tmp_path):
        # AC3.1 — tracking URI is the tmp_path file URI.
        assert mlflow.get_tracking_uri() == expected_tracking_uri

        run_identity = build_run_identity(
            sprint="s28",
            source_sprint="s27",
            experiment_family="realistic_afml",
            run_family="realistic_daily_afml",
        )
        tags = {
            "git_sha": run_identity["git_sha"],
            "sprint": "s28",
            "source_sprint": "s27",
            "run_family": "realistic_daily_afml",
        }

        # ─── First run ────────────────────────────────────────────────
        artifacts_a = _prepare_artifacts(
            artifact_dir=artifact_dir_a,
            params=expected_params,
            run_identity=run_identity,
            dataset_manifest=expected_dataset_manifest,
            metrics=expected_metrics,
            signal_summary=expected_signal_summary,
        )
        run_id_a = log_research_run(
            params=expected_params,
            metrics=expected_metrics,
            tags=tags,
            artifacts=artifacts_a,
            run_name="s28-pr3-realistic-a",
        )
        assert isinstance(run_id_a, str) and run_id_a

        # AC3.params — MLflow stringifies params on round-trip.
        run = mlflow.get_run(run_id_a)
        for key, value in expected_params.items():
            assert run.data.params[key] == str(value), (
                f"AC3.params: {key} read back {run.data.params[key]!r}, expected {str(value)!r}."
            )

        # AC3.metrics — floats round-trip exactly under the file backend.
        for key, value in expected_metrics.items():
            assert run.data.metrics[key] == value, (
                f"AC3.metrics: {key} read back {run.data.metrics[key]!r}, expected {value!r}."
            )

        # AC3.tags — user-set tags survive round-trip alongside MLflow
        # internals.
        for key, value in tags.items():
            assert run.data.tags[key] == value, (
                f"AC3.tags: {key} read back {run.data.tags[key]!r}, expected {value!r}."
            )

        # AC3.artifacts — exactly our 4 JSON files at the run root.
        downloaded_a = _downloaded_files(run_id_a)
        for name in ARTIFACT_FILENAMES:
            assert name in downloaded_a, (
                f"AC3.artifacts: missing required artifact {name!r}; "
                f"downloaded={sorted(downloaded_a)}."
            )

        # AC3.4 — no forbidden extensions, total < 1 MiB.
        for path in downloaded_a.values():
            assert path.suffix not in FORBIDDEN_ARTIFACT_EXTENSIONS, (
                f"AC3.4: forbidden artifact extension {path.suffix!r} on {path.name!r}."
            )
        total_bytes_a = sum(p.stat().st_size for p in downloaded_a.values())
        assert total_bytes_a < ARTIFACT_TOTAL_BUDGET_BYTES, (
            f"AC3.4: total artifact size {total_bytes_a} bytes >= 1 MiB cap."
        )

        # AC3.4 (defensive) — every JSON artifact deserialises cleanly.
        for name in ARTIFACT_FILENAMES:
            json.loads(downloaded_a[name].read_text())

        # ─── Second run for AC3.6 ─────────────────────────────────────
        # Rebuild every input independently to prove generation is
        # byte-deterministic, then log under a distinct run_id.
        params_b = _build_params()
        manifest_b = _build_dataset_manifest_dict()
        run_identity_b = build_run_identity(
            sprint="s28",
            source_sprint="s27",
            experiment_family="realistic_afml",
            run_family="realistic_daily_afml",
        )
        out_b = run_realistic_afml()
        metrics_b = _build_metrics(out_b)
        signal_summary_b = _build_signal_summary(out_b)
        tags_b = {
            "git_sha": run_identity_b["git_sha"],
            "sprint": "s28",
            "source_sprint": "s27",
            "run_family": "realistic_daily_afml",
        }
        artifacts_b = _prepare_artifacts(
            artifact_dir=artifact_dir_b,
            params=params_b,
            run_identity=run_identity_b,
            dataset_manifest=manifest_b,
            metrics=metrics_b,
            signal_summary=signal_summary_b,
        )
        run_id_b = log_research_run(
            params=params_b,
            metrics=metrics_b,
            tags=tags_b,
            artifacts=artifacts_b,
            run_name="s28-pr3-realistic-b",
        )
        assert run_id_b != run_id_a, "AC3.6: distinct invocations must yield distinct run_ids."

        # AC3.6 — local artifact files byte-equal across the two builds.
        hashes_a = _file_hashes(artifact_dir_a)
        hashes_b = _file_hashes(artifact_dir_b)
        for name in ARTIFACT_FILENAMES:
            assert hashes_a[name] == hashes_b[name], (
                f"AC3.6: local artifact {name} byte-mismatch across runs "
                f"({hashes_a[name]} vs {hashes_b[name]})."
            )

        # AC3.6 — downloaded artifacts byte-equal between runs and to source.
        downloaded_b = _downloaded_files(run_id_b)
        for name in ARTIFACT_FILENAMES:
            blob_a = downloaded_a[name].read_bytes()
            blob_b = downloaded_b[name].read_bytes()
            assert blob_a == blob_b, (
                f"AC3.6: downloaded artifact {name} byte-mismatch "
                f"between {run_id_a[:8]}.. and {run_id_b[:8]}.."
            )
            source_blob = (artifact_dir_a / name).read_bytes()
            assert blob_a == source_blob, f"AC3.6: downloaded {name} differs from local source."

    _assert_no_repo_mlruns()
