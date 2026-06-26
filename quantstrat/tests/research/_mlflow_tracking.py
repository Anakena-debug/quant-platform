"""S28 PR1 — local-only MLflow tracking helper (test-local).

Lives under ``quantstrat/tests/research/`` because S28 holds all MLflow code
test-local for now; promotion to ``quantstrat/src/quantstrat/research/`` is
deferred to a post-S28 sprint.

Two helpers:

* ``with_local_tracking(tmp_path)`` — context manager that points MLflow at
  ``file://{tmp_path}/mlruns`` on enter and restores the prior tracking URI
  on exit. Tests pass pytest's ``tmp_path`` fixture; the loose ``mlruns``
  subdirectory is cleaned up by pytest when the fixture tears down.
  Repo-level ``mlruns/`` is forbidden — ``.gitignore`` and
  ``quantstrat/.gitignore`` carry an explicit entry as a second layer of
  defense (AC1.2 / AC1.7).

* ``log_research_run(...)`` — opens an MLflow run, logs params/metrics/tags,
  then logs each prepared artifact file. Enforces a strict per-run total
  artifact-byte budget of < 1 MiB *before* any MLflow call lands (AC1.5),
  so an over-cap call never leaves a partial / orphan run in the store.
  Callers prepare artifact files on disk; the helper does not write artifact
  bytes itself (separation of concerns: PR4 owns artifact body generation).

The leading-underscore filename signals "test helper, not pytest collection
target," following ``_toy_afml.py`` (S26) and ``_realistic_panel.py`` (S27).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from pathlib import Path

import mlflow

# AC1.5 — total artifact bytes across all files must be strictly < 1 MiB.
ARTIFACT_BUDGET_BYTES: int = 1024 * 1024

EXPERIMENT_NAME: str = "quantstrat-research"


@contextlib.contextmanager
def with_local_tracking(tmp_path: Path) -> Iterator[None]:
    """Point MLflow at ``file://{tmp_path}/mlruns`` for the context body.

    Restores the prior tracking URI on exit. Never deletes
    ``tmp_path/mlruns`` — pytest's ``tmp_path`` fixture handles cleanup.
    """
    prior_uri = mlflow.get_tracking_uri()
    local_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(local_uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        yield
    finally:
        mlflow.set_tracking_uri(prior_uri)


def log_research_run(
    *,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifacts: Mapping[str, Path],
    run_name: str | None = None,
) -> str:
    """Open an MLflow run, log {params, metrics, tags, artifacts}, return run_id.

    Raises ``ValueError`` before any ``mlflow.*`` call if the total
    artifact-byte budget is exceeded, so an over-cap call never leaves a
    partial run in the tracking store.
    """
    total_bytes = sum(path.stat().st_size for path in artifacts.values())
    if total_bytes >= ARTIFACT_BUDGET_BYTES:
        raise ValueError(
            f"Artifact budget exceeded: total {total_bytes} bytes "
            f">= cap {ARTIFACT_BUDGET_BYTES} bytes (< 1 MiB required)."
        )

    with mlflow.start_run(run_name=run_name) as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        for _, path in artifacts.items():
            mlflow.log_artifact(str(path))
        return run.info.run_id
