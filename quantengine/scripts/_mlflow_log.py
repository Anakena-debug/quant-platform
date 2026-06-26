"""MLflow logging utilities for quantengine research scripts."""

from __future__ import annotations

import contextlib
import math
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
MLRUNS_DIR: Path = REPO_ROOT / "results" / "mlruns"


@contextlib.contextmanager
def local_tracking(mlruns_dir: Path = MLRUNS_DIR) -> Iterator[None]:
    """Point MLflow at a local file store for the duration of the block."""
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    prior = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    try:
        yield
    finally:
        mlflow.set_tracking_uri(prior)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()[:12]
        )
    except Exception:
        return "unknown"


def _is_finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError, OverflowError):
        return False


def _clean_metrics(raw: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in raw.items() if _is_finite(v)}


def _clean_params(raw: dict[str, Any]) -> dict[str, str]:
    return {k: str(v)[:500] for k, v in raw.items()}


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> None:
    """Log a complete MLflow run (params + metrics + artifacts)."""
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.set_tag("git_sha", _git_sha())
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(_clean_params(params))
        if metrics:
            clean = _clean_metrics(metrics)
            if clean:
                mlflow.log_metrics(clean)
        if artifacts:
            for a in artifacts:
                if a.exists():
                    mlflow.log_artifact(str(a))
