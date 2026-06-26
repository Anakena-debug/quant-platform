"""S28 PR2 — research-run metadata schemas + deterministic builders.

Lives under ``quantstrat/tests/research/`` (test-local; promotion deferred).

The schemas are backend-agnostic: this module **does not** import
``mlflow`` (AC2.7). That decouples run-identity / dataset-manifest
semantics from the tracking sink, so the schemas survive a future
migration off MLflow without churn.

Two builders:

* ``build_run_identity(*, sprint, source_sprint, experiment_family,
  run_family)`` — captures ``git_sha`` and ``branch`` via ``subprocess.run``
  and fails loud on any git error. No ``"unknown"`` fallback — see plan
  §7 failure mode 6 (silent fallback corrupts run identity).
* ``build_dataset_manifest(*, panel, ...)`` — derives counts, duplicates,
  missing rates, and two stable SHA-256 hashes from the panel's
  *metadata* (column names + dtypes + ticker set + row count +
  date window). The panel's row bytes are **never** hashed (AC2.4 /
  AC2.5) and never stored.

Hashes are SHA-256 hex digests of canonical-form JSON
(``sort_keys=True, separators=(",", ":"), ensure_ascii=True``). The
``dataset_manifest_hash`` field is computed over the manifest dict
*excluding itself*, then injected — that avoids a self-referential
cycle while keeping every other field hash-load-bearing.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from typing import TypedDict

import pandas as pd

__all__ = [
    "RunIdentity",
    "DatasetManifest",
    "MetricsSchema",
    "TagsSchema",
    "canonical_json",
    "sha256_json",
    "build_run_identity",
    "build_dataset_manifest",
]


# ─── Schemas (AC2.1) ────────────────────────────────────────────────────


class RunIdentity(TypedDict):
    """Per-run identity (AC2.1 + AC3.tags).

    ``source_sprint`` is included so PR3 / PR4 can persist the full
    identity (e.g. into ``config.json`` per AC3.artifacts) without
    plumbing a second argument through the tracking layer.
    """

    git_sha: str
    branch: str
    sprint: str
    source_sprint: str | None
    experiment_family: str
    run_family: str


class DatasetManifest(TypedDict):
    """Dataset metadata for a research run (AC2.1)."""

    data_source: str
    data_frequency: str
    universe_name: str
    parsed_ticker_count: int
    available_ticker_count: int
    start_date: str
    end_date: str
    as_of: str
    row_count: int
    missing_rate_summary: Mapping[str, float]
    duplicate_count: int
    schema_hash: str
    dataset_manifest_hash: str
    timezone: str | None
    corporate_action_policy: str | None


class MetricsSchema(TypedDict):
    """Canonical signal-side metric keys (AC3.metrics).

    All values are ``float``-typed. PnL is forbidden — see
    ``quantengine/ARCHITECTURE.md`` invariant 5 and §forbidden_actions.
    """

    n_tickers: float
    n_train: float
    tradeable_count: float
    tradeable_fraction: float
    expected_return_mean: float
    expected_return_std: float
    interval_half_width_median: float
    interval_half_width_p90: float
    max_abs_expected_return: float


class TagsSchema(TypedDict):
    """Canonical MLflow tag keys (AC3.tags)."""

    git_sha: str
    sprint: str
    source_sprint: str | None
    run_family: str


# ─── Hashing helpers ────────────────────────────────────────────────────


def canonical_json(obj: object) -> str:
    """Canonical-form JSON: sorted keys, no whitespace, ASCII-escaped."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(obj: object) -> str:
    """SHA-256 hex digest of ``canonical_json(obj).encode("utf-8")``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ─── Builders (AC2.2) ───────────────────────────────────────────────────


def _git(cmd: list[str]) -> str:
    """Run a git subcommand and return stripped stdout. Fail loud on error."""
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def build_run_identity(
    *,
    sprint: str,
    source_sprint: str | None,
    experiment_family: str,
    run_family: str,
) -> RunIdentity:
    """Capture ``git_sha`` + ``branch`` and seal them into a ``RunIdentity``.

    Raises ``subprocess.CalledProcessError`` / ``FileNotFoundError`` if git
    is unreachable; raises ``ValueError`` if any caller-supplied string is
    empty. The helper never falls back to a placeholder — silent defaults
    corrupt run identity (plan §7 failure mode 6).
    """
    for name, value in (
        ("sprint", sprint),
        ("experiment_family", experiment_family),
        ("run_family", run_family),
    ):
        if not value:
            raise ValueError(f"{name} must be a non-empty string; got {value!r}.")

    git_sha = _git(["git", "rev-parse", "HEAD"])
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return RunIdentity(
        git_sha=git_sha,
        branch=branch,
        sprint=sprint,
        source_sprint=source_sprint,
        experiment_family=experiment_family,
        run_family=run_family,
    )


def _validate_dates(*, start_date: str, end_date: str, as_of: str) -> None:
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date!r}) < start_date ({start_date!r}); window inverted.")
    if not (start_date <= as_of <= end_date):
        raise ValueError(
            f"as_of ({as_of!r}) outside window [start_date={start_date!r}, end_date={end_date!r}]."
        )


def build_dataset_manifest(
    *,
    panel: pd.DataFrame,
    universe_name: str,
    data_source: str,
    data_frequency: str,
    start_date: str,
    end_date: str,
    as_of: str,
    parsed_tickers: Sequence[str],
    timezone: str | None = None,
    corporate_action_policy: str | None = None,
) -> DatasetManifest:
    """Derive counts / duplicates / hashes from panel metadata.

    Raises ``ValueError`` on malformed input (empty ticker list, inverted
    date window, ``as_of`` outside the window, non-finite missing rate,
    negative count). The hashes are SHA-256 of canonical JSON over metadata
    only — the panel's row bytes are never read into the hash payload.
    """
    if not parsed_tickers:
        raise ValueError("parsed_tickers must be a non-empty sequence.")
    if not universe_name:
        raise ValueError("universe_name must be a non-empty string.")
    if not data_source:
        raise ValueError("data_source must be a non-empty string.")
    if not data_frequency:
        raise ValueError("data_frequency must be a non-empty string.")
    _validate_dates(start_date=start_date, end_date=end_date, as_of=as_of)

    parsed_ticker_count = len(parsed_tickers)
    row_count = int(len(panel))

    if "ticker" not in panel.columns:
        raise ValueError(
            f"panel is missing required 'ticker' column; got columns {list(panel.columns)!r}."
        )
    available_ticker_count = int(panel["ticker"].astype(str).nunique())

    missing_rate_summary: dict[str, float] = {}
    for col in panel.columns:
        rate = float(panel[col].isna().mean()) if row_count > 0 else math.nan
        if not math.isfinite(rate):
            raise ValueError(
                f"missing_rate_summary[{col!r}] is non-finite ({rate}); "
                f"likely caused by an empty panel (row_count={row_count})."
            )
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"missing_rate_summary[{col!r}] outside [0, 1]: got {rate}.")
        missing_rate_summary[str(col)] = rate

    duplicate_count = int(panel.duplicated().sum())

    for name, count in (
        ("parsed_ticker_count", parsed_ticker_count),
        ("available_ticker_count", available_ticker_count),
        ("row_count", row_count),
        ("duplicate_count", duplicate_count),
    ):
        if count < 0:
            raise ValueError(f"{name} is negative ({count}); should be impossible.")

    schema_payload: list[list[str]] = [[str(col), str(panel[col].dtype)] for col in panel.columns]
    schema_hash = sha256_json(schema_payload)

    partial: dict[str, object] = {
        "data_source": data_source,
        "data_frequency": data_frequency,
        "universe_name": universe_name,
        "parsed_ticker_count": parsed_ticker_count,
        "available_ticker_count": available_ticker_count,
        "start_date": start_date,
        "end_date": end_date,
        "as_of": as_of,
        "row_count": row_count,
        "missing_rate_summary": missing_rate_summary,
        "duplicate_count": duplicate_count,
        "schema_hash": schema_hash,
        "timezone": timezone,
        "corporate_action_policy": corporate_action_policy,
    }
    dataset_manifest_hash = sha256_json(partial)

    return DatasetManifest(
        data_source=data_source,
        data_frequency=data_frequency,
        universe_name=universe_name,
        parsed_ticker_count=parsed_ticker_count,
        available_ticker_count=available_ticker_count,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
        row_count=row_count,
        missing_rate_summary=missing_rate_summary,
        duplicate_count=duplicate_count,
        schema_hash=schema_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        timezone=timezone,
        corporate_action_policy=corporate_action_policy,
    )
