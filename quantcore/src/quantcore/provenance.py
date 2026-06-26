"""quantcore.provenance — content-addressed lineage for reproducible research.

A finding is only worth as much as your ability to reproduce it. This module captures the
three things that fully determine a screen/factory result and binds them into one stable,
content-addressed identifier:

1. **data**   — a deterministic fingerprint of the input panel (:func:`frame_fingerprint`):
   same bytes in, same hash out; insensitive to incidental row/column ordering.
2. **code**   — the git commit of the quantcore repo plus a dirty flag, and the quantcore
   version (:func:`code_version`).
3. **params** — the screen parameters (method, HAC lags, FDR, DSR threshold, ...).

:func:`build_manifest` hashes ``(data, commit, quantcore_version, params)`` into a 16-hex
``run_id`` — note the *timestamp is deliberately excluded*, so the same data + code + params
reproduce the same id no matter when they run. :func:`verify_manifest` re-derives all three
against a candidate dataset and reports whether the finding is reproducible (and refuses to
certify a dirty working tree, whose changes aren't captured by any commit).

    from quantcore.provenance import build_manifest, verify_manifest
    m = build_manifest(panel_df, params={"method": "spearman", "fdr": 0.10}, source="us_eq.parquet")
    m.run_id                              # -> stable content id, e.g. "3f9a1c0e7b2d4a55"
    verify_manifest(m, panel_df).reproducible   # -> True iff data+code+id match and tree is clean
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import quantcore


def _utcnow() -> str:
    """Current UTC timestamp, ISO-8601 to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a DataFrame's content (index, columns, dtypes, values).

    The frame is canonicalized by sorting both axes, so incidental row/column ordering does
    not change identity, while values, labels, and dtypes do. Built on
    :func:`pandas.util.hash_pandas_object` (a fixed, process-stable hashing scheme), so the
    same data fingerprints identically across runs and machines on a given pandas version.
    """
    canon = frame.sort_index(axis=0).sort_index(axis=1)
    # pandas-stubs mistypes pandas.util; the runtime call is correct (a fixed hashing scheme).
    hashed = pd.util.hash_pandas_object(canon, index=True)  # pyright: ignore[reportAttributeAccessIssue]
    row_hashes = hashed.to_numpy()
    h = hashlib.sha256()
    h.update(b"cols:" + "|".join(map(str, canon.columns)).encode())
    h.update(b"|dtypes:" + "|".join(str(d) for d in canon.dtypes).encode())
    h.update(b"|shape:" + repr(canon.shape).encode())
    h.update(b"|rows:")
    h.update(row_hashes.tobytes())
    return h.hexdigest()


def panels_fingerprint(panels: Mapping[str, pd.DataFrame]) -> str:
    """Fingerprint a name-keyed family of panels (order-independent in the keys)."""
    h = hashlib.sha256()
    for name in sorted(panels):
        h.update(name.encode() + b"=" + frame_fingerprint(panels[name]).encode() + b";")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class CodeVersion:
    """Identity of the code that produced a result."""

    commit: str | None  # git HEAD SHA, or None outside a repo
    dirty: bool  # uncommitted changes present -> not reproducible from the commit alone
    quantcore_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def code_version() -> CodeVersion:
    """Capture the current quantcore code identity: git HEAD, dirty flag, and package version.

    Git is queried relative to the quantcore source directory (not the process cwd), so the
    commit refers to the *code's* repo. The dirty flag is **scoped to the importable source
    package** (``git status --porcelain -- src/quantcore``): only uncommitted changes to the
    code that computes results count. Data caches, the lockfile (which ``uv run`` rewrites on
    every invocation), pyproject, tests, and sibling monorepo packages do not mark quantcore
    dirty. Degrades gracefully to ``commit=None`` when git or a repo is unavailable (e.g. an
    installed wheel).
    """
    code_dir = Path(quantcore.__file__).resolve().parent  # .../quantcore/src/quantcore
    commit: str | None = None
    dirty = False
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=code_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(code_dir)],
            cwd=code_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (subprocess.SubprocessError, OSError):
        commit = None
    return CodeVersion(commit=commit, dirty=dirty, quantcore_version=quantcore.__version__)


def _run_id(data_fingerprint: str, code: CodeVersion, params: Mapping[str, object]) -> str:
    """Content-address (data, commit, quantcore version, params) -> stable 16-hex id.

    Deliberately excludes the timestamp and the dirty flag: identity is what determines the
    result, not when it was recorded.
    """
    canon = json.dumps(
        {
            "data": data_fingerprint,
            "commit": code.commit,
            "quantcore": code.quantcore_version,
            "params": params,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The reproducibility record for one screen/factory run."""

    run_id: str
    created_utc: str
    data_fingerprint: str
    code: CodeVersion
    params: dict[str, object] = field(default_factory=dict)
    n_rows: int | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunManifest:
        fields: dict[str, Any] = dict(data)
        code = fields.get("code")
        if isinstance(code, Mapping):
            code_fields: dict[str, Any] = dict(code)
            fields["code"] = CodeVersion(**code_fields)
        return cls(**fields)


def build_manifest(
    data: pd.DataFrame,
    *,
    params: Mapping[str, object],
    source: str | None = None,
    now: str | None = None,
    code: CodeVersion | None = None,
) -> RunManifest:
    """Fingerprint ``data`` + capture code + bind params into a content-addressed manifest.

    ``now`` / ``code`` are injectable for deterministic tests; both default to live capture.
    The ``run_id`` is independent of ``now``, so re-recording the same run is idempotent.
    """
    fingerprint = frame_fingerprint(data)
    cv = code if code is not None else code_version()
    run_id = _run_id(fingerprint, cv, params)
    return RunManifest(
        run_id=run_id,
        created_utc=now or _utcnow(),
        data_fingerprint=fingerprint,
        code=cv,
        params=dict(params),
        n_rows=int(len(data)),
        source=source,
    )


@dataclass(frozen=True, slots=True)
class ManifestCheck:
    """The verdict of re-deriving a manifest against a candidate dataset."""

    run_id_matches: bool
    data_matches: bool
    code_matches: bool
    reproducible: bool  # data + code + id all match AND neither tree is dirty
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_manifest(
    manifest: RunManifest,
    data: pd.DataFrame,
    *,
    code: CodeVersion | None = None,
) -> ManifestCheck:
    """Re-derive ``manifest`` against ``data`` and the current code; report reproducibility.

    A manifest is ``reproducible`` only when the data fingerprint, the code commit, and the
    recomputed ``run_id`` all match *and* neither the recorded nor the current working tree is
    dirty — uncommitted changes are not captured by any commit, so identity cannot be certified.
    """
    cv = code if code is not None else code_version()
    fingerprint = frame_fingerprint(data)
    data_ok = fingerprint == manifest.data_fingerprint
    code_ok = cv.commit is not None and cv.commit == manifest.code.commit
    id_ok = _run_id(fingerprint, cv, manifest.params) == manifest.run_id
    clean = not cv.dirty and not manifest.code.dirty
    reproducible = data_ok and code_ok and id_ok and clean

    if reproducible:
        detail = "reproducible: data, code, and run_id all match a clean tree"
    else:
        parts: list[str] = []
        if not data_ok:
            parts.append("data fingerprint differs")
        if not code_ok:
            parts.append(f"code commit differs (manifest={manifest.code.commit}, now={cv.commit})")
        if not id_ok:
            parts.append("recomputed run_id differs")
        if not clean:
            parts.append("working tree is dirty (uncommitted changes not captured)")
        detail = "; ".join(parts) or "mismatch"

    return ManifestCheck(
        run_id_matches=id_ok,
        data_matches=data_ok,
        code_matches=code_ok,
        reproducible=reproducible,
        detail=detail,
    )


__all__ = [
    "CodeVersion",
    "ManifestCheck",
    "RunManifest",
    "build_manifest",
    "code_version",
    "frame_fingerprint",
    "panels_fingerprint",
    "verify_manifest",
]
