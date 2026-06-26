"""DGP-specific recordings artifact (S12 P11.3).

Non-asserting test that writes a single-file Markdown artifact with
per-(fixture, config, seed) observables that are NOT cross-fixture
invariant by S12's Shape B-restricted findings:

  - ``cv_mean_selected`` / ``cv_mean_all`` — DGP-specific
  - ``abstain_oos`` — DGP-specific
  - ``primary_oos_overall_acc`` / ``primary_oos_acc_on_takes``
  - ``precision_lift`` — sign-stable cross-fixture at canonical
                         (Pin 4 in test_fixture_invariants.py),
                         BUT sign-flips at low-SNR (V2-B5 anomaly)
  - ``n_events``, ``selected`` (selected feature set)
  - ``runtime_seconds`` — wall-time per pipeline run, for
                          environmental-drift triage

Cross-fixture pins live in ``test_fixture_invariants.py``. This file
is the recording side of the contract — no values asserted, just
written for retrospective comparison.

Cache scope: this module owns its own cache rather than sharing
``test_fixture_invariants.py``'s ``runs_cache``. The ~30s of
pipeline-run reuse is not worth cross-file fixture entanglement
(see S12 sprint plan, P11.3 cache-strategy decision).

Output: ``quantcore/tests/fixtures/calibration_recordings/latest.md``.
The directory is git-tracked (with ``.gitignore``); the artifact
itself is overwritten each run and not version-controlled. For a
historical snapshot, ``cp latest.md somewhere-dated.md`` before
the next pytest invocation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fixture_calibration import FIXTURE_REGISTRY, REGISTRY_SEEDS


# Columns rendered per row. Order matters — this is the column
# order in every (fixture, config) section's table.
RECORDED_COLUMNS: tuple[str, ...] = (
    "cv_mean_selected",
    "cv_mean_all",
    "primary_oos_overall_acc",
    "primary_oos_acc_on_takes",
    "precision_lift",
    "abstain_oos",
    "n_events",
    "runtime_seconds",
    "selected_features",
)

# Subset of runner-output keys we keep in cache. Trims the heavy
# DataFrame / array fields the runners produce; the recordings
# artifact is scalar-only plus the selected-feature list.
_CACHE_KEEP_KEYS: tuple[str, ...] = (
    "cv_mean_selected",
    "cv_mean_all",
    "primary_oos_overall_acc",
    "primary_oos_acc_on_takes",
    "precision_lift",
    "abstain_oos",
    "n_events",
    "selected",
)


# -----------------------------------------------------------------------------
# Module-scoped cache: 5 seeds × 2 fixtures × 2 configs = 20 runs.
# Build cost ~5 min wall time at worst (pinned in plan watch-items).
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recordings_cache() -> dict[tuple[str, str, int], dict[str, Any]]:
    """Build per-(fixture, config, seed) slim recordings cache.

    Configs: ``canonical`` uses each fixture's
    ``canonical_drift_coef`` (default runner behavior);
    ``low_snr`` uses ``low_snr_drift_coef`` from registry metadata.

    Each run is timed with ``time.perf_counter()`` — wall time only,
    not CPU time, not deterministic across machines. Used for
    environmental-drift triage in the recordings artifact, not as
    a pinned property.
    """
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    for fixture_name, meta in FIXTURE_REGISTRY.items():
        configs = (
            ("canonical", meta.canonical_drift_coef),
            ("low_snr", meta.low_snr_drift_coef),
        )
        for config_name, drift_coef in configs:
            for seed in REGISTRY_SEEDS:
                t0 = time.perf_counter()
                run = meta.runner(seed=seed, drift_coef=drift_coef)
                runtime_seconds = time.perf_counter() - t0
                slim = {key: run[key] for key in _CACHE_KEEP_KEYS}
                slim["runtime_seconds"] = runtime_seconds
                cache[(fixture_name, config_name, seed)] = slim
    return cache


# -----------------------------------------------------------------------------
# Markdown rendering helpers.
# -----------------------------------------------------------------------------


def _git_sha() -> str:
    """Short git SHA of HEAD; falls back to ``unknown`` if git missing
    or repo absent (e.g., a CI runner without the .git directory)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _fixture_module_hash() -> str:
    """SHA-256 prefix of both fixture source files concatenated.

    Concrete handle answering 'do these recordings come from the
    same fixture code as the file I'm currently reading?' — useful
    when a stale ``latest.md`` is found on disk after a fixture
    edit. Hashes the source bytes, not bytecode.
    """
    here = Path(__file__).parent / "fixtures"
    blobs = (
        (here / "tml_composition_spike.py").read_bytes()
        + b"\n--SEPARATOR--\n"
        + (here / "tml_composition_spike_v2.py").read_bytes()
    )
    return hashlib.sha256(blobs).hexdigest()[:16]


def _format_cell(key: str, value: Any) -> str:
    """Per-key cell formatter. Floats get signed 4-decimal; ints get
    plain integer; selected-features get sorted comma-join; runtime
    gets 2-decimal seconds."""
    if key == "selected_features":
        if not value:
            return "(empty)"
        return ", ".join(sorted(value))
    if key == "n_events":
        return f"{int(value)}"
    if key == "runtime_seconds":
        return f"{float(value):.2f}"
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return str(value)


def _render_recordings_md(
    cache: dict[tuple[str, str, int], dict[str, Any]],
) -> str:
    """Render the cache as Markdown.

    Layout: header + one ``## fixture / config`` section per
    (fixture, config) pair, each with a table whose rows are seeds
    in ``REGISTRY_SEEDS`` order and columns are ``RECORDED_COLUMNS``.
    """
    lines: list[str] = []
    lines.append("# S12 fixture-calibration recordings — `latest.md`")
    lines.append("")
    lines.append(f"- Generated (UTC): {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"- Git SHA (HEAD): `{_git_sha()}`")
    lines.append(
        f"- Fixture-source sha256[:16] (spike + spike_v2 concatenated): `{_fixture_module_hash()}`"
    )
    lines.append("")
    lines.append(
        "Per-(fixture, config, seed) DGP-specific observables. "
        "**Non-asserting** — these values are NOT pinned cross-fixture "
        "(see `test_fixture_invariants.py` for the four cross-fixture "
        "invariants and S12 sprint plan for Shape B-restricted "
        "rationale). This file is overwritten every test run and is "
        "not version-controlled (see `.gitignore` in this directory). "
        "Copy out before re-running for a historical snapshot."
    )
    lines.append("")
    lines.append(
        "Configs: `canonical` uses each fixture's "
        "`canonical_drift_coef` (default runner behavior); `low_snr` "
        "uses `low_snr_drift_coef` from registry metadata "
        "(V1: 0.002 = drift/3; V2-B5: ≈ 0.00195 = drift/3, preserves "
        "V1's SNR ratio)."
    )
    lines.append("")

    for fixture_name in FIXTURE_REGISTRY:
        for config_name in ("canonical", "low_snr"):
            lines.append(f"## `{fixture_name}` / `{config_name}`")
            lines.append("")
            header = ["seed", *RECORDED_COLUMNS]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for seed in REGISTRY_SEEDS:
                run = cache[(fixture_name, config_name, seed)]
                cells = [str(seed)]
                for key in RECORDED_COLUMNS:
                    if key == "selected_features":
                        cells.append(_format_cell(key, run.get("selected", [])))
                    else:
                        cells.append(_format_cell(key, run.get(key)))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# The single non-asserting test.
# -----------------------------------------------------------------------------


def test_record_dgp_specific_observables(
    recordings_cache: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    """Write the recordings artifact. Asserts only that the file
    exists and has non-zero size after writing — the contents are
    intentionally not pinned (DGP-specific by Shape B-restricted
    findings). Pinning happens in
    ``test_fixture_invariants.py``."""
    artifact_dir = Path(__file__).parent / "fixtures" / "calibration_recordings"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "latest.md"
    md = _render_recordings_md(recordings_cache)
    artifact_path.write_text(md)
    assert artifact_path.exists(), f"recordings artifact not written: {artifact_path}"
    assert artifact_path.stat().st_size > 0, f"recordings artifact empty: {artifact_path}"
