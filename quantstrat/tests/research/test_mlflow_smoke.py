"""S28 PR1 — local MLflow smoke test (AC1.1–AC1.9).

Validates ``quantstrat/tests/research/_mlflow_tracking.py``. Runs under
``uv run --directory quantstrat --extra research pytest`` so the optional
``research`` extra (MLflow) is installed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import mlflow
import pytest

from research._mlflow_tracking import (
    ARTIFACT_BUDGET_BYTES,
    log_research_run,
    with_local_tracking,
)

# Path resolution per AC1.2 — derive from the test file, not os.getcwd().
TEST_FILE: Path = Path(__file__).resolve()
QUANTSTRAT_DIR: Path = TEST_FILE.parents[2]
REPO_ROOT: Path = TEST_FILE.parents[3]

# AC1.6 — env vars that would re-route MLflow off the local file store.
REMOTE_TRACKING_ENV_VARS: tuple[str, ...] = (
    "MLFLOW_REGISTRY_URI",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def _assert_no_repo_mlruns() -> None:
    """AC1.2 — fail loud if mlruns/ exists at repo root or under quantstrat."""
    for label, path in (
        ("repo-root", REPO_ROOT / "mlruns"),
        ("quantstrat", QUANTSTRAT_DIR / "mlruns"),
    ):
        if path.exists():
            pytest.fail(
                f"Forbidden {label} mlruns/ directory exists at {path}; "
                f"operator must remove it (see S28 §3 AC1.2)."
            )


def _make_summary_artifact(tmp_path: Path) -> Path:
    """Deterministic toy JSON artifact for the happy-path / determinism tests."""
    payload = {"alpha": 0.2, "seed": 0, "toy_metric": 1.0}
    artifact_path = tmp_path / "summary.json"
    artifact_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return artifact_path


def test_helper_api_surface() -> None:
    """AC1.3 — both helpers expose the documented signature."""
    ctx_sig = inspect.signature(with_local_tracking)
    assert list(ctx_sig.parameters) == ["tmp_path"]

    log_sig = inspect.signature(log_research_run)
    assert list(log_sig.parameters) == [
        "params",
        "metrics",
        "tags",
        "artifacts",
        "run_name",
    ]
    for name in ("params", "metrics", "tags", "artifacts", "run_name"):
        assert log_sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_local_tracking_uri_set_and_restored(tmp_path: Path) -> None:
    """AC1.1 — context manager sets the file URI on enter, restores on exit.

    Also re-asserts AC1.2 before and after the context body.
    """
    _assert_no_repo_mlruns()
    prior = mlflow.get_tracking_uri()
    expected = f"file://{tmp_path}/mlruns"

    with with_local_tracking(tmp_path):
        assert mlflow.get_tracking_uri() == expected

    assert mlflow.get_tracking_uri() == prior
    _assert_no_repo_mlruns()


def test_no_remote_env_vars_during_run(tmp_path: Path) -> None:
    """AC1.6 — cloud / remote-tracking env vars must be unset or empty."""
    _assert_no_repo_mlruns()
    expected = f"file://{tmp_path}/mlruns"
    with with_local_tracking(tmp_path):
        for name in REMOTE_TRACKING_ENV_VARS:
            assert os.environ.get(name, "") == "", (
                f"{name} must be unset/empty during the smoke run (see S28 §3 AC1.6)."
            )
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        assert env_uri in ("", expected), (
            f"MLFLOW_TRACKING_URI={env_uri!r} would override the helper; "
            f"must be unset or equal to {expected!r}."
        )
        assert mlflow.get_tracking_uri() == expected
    _assert_no_repo_mlruns()


def test_gitignore_contains_mlruns_in_both_files() -> None:
    """AC1.7 — `mlruns/` on its own line in repo-root and quantstrat .gitignore."""
    for label, path in (
        ("repo-root", REPO_ROOT / ".gitignore"),
        ("quantstrat", QUANTSTRAT_DIR / ".gitignore"),
    ):
        assert path.exists(), f"{label} .gitignore missing at {path}"
        lines = {line.strip() for line in path.read_text().splitlines()}
        assert "mlruns/" in lines, (
            f"{label} .gitignore at {path} must contain 'mlruns/' on its "
            f"own line (see S28 §3 AC1.7)."
        )


def test_research_extra_declares_mlflow() -> None:
    """AC1.8 — quantstrat/pyproject.toml's `research` extra includes mlflow."""
    pyproject = (QUANTSTRAT_DIR / "pyproject.toml").read_text()
    assert "research" in pyproject
    assert "mlflow" in pyproject


def test_happy_path_logs_and_reads_back(tmp_path: Path) -> None:
    """AC1.4 — params/metrics/tags/artifact round-trip via mlflow.get_run."""
    _assert_no_repo_mlruns()
    artifact = _make_summary_artifact(tmp_path)
    assert artifact.stat().st_size < 1024

    params = {"alpha": 0.2, "seed": 0}
    metrics = {"toy_metric": 1.0}
    tags = {"sprint": "s28", "run_family": "smoke"}

    with with_local_tracking(tmp_path):
        run_id = log_research_run(
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts={"summary": artifact},
            run_name="s28-pr1-smoke",
        )
        assert isinstance(run_id, str) and len(run_id) > 0

        run = mlflow.get_run(run_id)
        # MLflow stringifies params and tags; metrics remain float.
        assert run.data.params == {"alpha": "0.2", "seed": "0"}
        assert run.data.metrics == {"toy_metric": 1.0}
        for key, value in tags.items():
            assert run.data.tags[key] == value

        local_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
        downloaded = local_dir / "summary.json"
        assert downloaded.exists()
        assert downloaded.stat().st_size < 1024
        assert downloaded.read_bytes() == artifact.read_bytes()

    _assert_no_repo_mlruns()


@pytest.mark.parametrize(
    ("size_bytes", "expect_raise"),
    [
        (1024, False),
        (ARTIFACT_BUDGET_BYTES, True),  # at cap — strict < required
        (ARTIFACT_BUDGET_BYTES + 1, True),  # over cap
    ],
)
def test_artifact_budget_enforced(
    tmp_path: Path,
    size_bytes: int,
    expect_raise: bool,
) -> None:
    """AC1.5 — over-cap (or at-cap) artifact bundles raise ValueError pre-log."""
    _assert_no_repo_mlruns()
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\0" * size_bytes)

    with with_local_tracking(tmp_path):
        if expect_raise:
            with pytest.raises(ValueError, match="Artifact budget exceeded"):
                log_research_run(
                    params={},
                    metrics={},
                    tags={},
                    artifacts={"blob": blob},
                )
        else:
            run_id = log_research_run(
                params={},
                metrics={},
                tags={},
                artifacts={"blob": blob},
            )
            assert isinstance(run_id, str)

    _assert_no_repo_mlruns()


def test_determinism_across_two_runs(tmp_path: Path) -> None:
    """AC1.9 — two identical invocations: distinct run_ids, equal contents."""
    _assert_no_repo_mlruns()
    artifact = _make_summary_artifact(tmp_path)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()

    params = {"alpha": 0.2, "seed": 0}
    metrics = {"toy_metric": 1.0}
    tags = {"sprint": "s28", "run_family": "smoke"}

    with with_local_tracking(tmp_path):
        rid1 = log_research_run(
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts={"summary": artifact},
        )
        rid2 = log_research_run(
            params=params,
            metrics=metrics,
            tags=tags,
            artifacts={"summary": artifact},
        )
        assert rid1 != rid2

        for rid in (rid1, rid2):
            run = mlflow.get_run(rid)
            assert run.data.params == {"alpha": "0.2", "seed": "0"}
            assert run.data.metrics == {"toy_metric": 1.0}
            for key, value in tags.items():
                assert run.data.tags[key] == value
            local_dir = Path(mlflow.artifacts.download_artifacts(run_id=rid))
            blob = (local_dir / "summary.json").read_bytes()
            assert hashlib.sha256(blob).hexdigest() == artifact_hash

    _assert_no_repo_mlruns()
