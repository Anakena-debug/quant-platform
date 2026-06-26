"""Raw zone (s81 REQ2) — content-addressed copies BEFORE parsing.

Every source file is copied to ``<raw_root>/<sha256>/<name>`` with a manifest entry (original path,
sha256, bytes, mtime) before any parser touches it. Parsers read the raw-zone copy, never the live
source path, so provenance is fixed at ingest time and a later edit upstream can't silently change what
was ingested. Default root is ``quantlake/data/raw`` (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
_READ_CHUNK = 1 << 20


@dataclass(frozen=True)
class RawEntry:
    name: str
    original_path: str
    sha256: str
    bytes: int
    mtime: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_to_raw_zone(
    paths: list[Path | str], *, raw_root: Path | str = DEFAULT_RAW_ROOT
) -> list[RawEntry]:
    """Copy each source file into the content-addressed raw zone; write/refresh ``raw_manifest.json``.

    Idempotent: a file whose content already lives under its sha is not re-copied.
    """
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    entries: list[RawEntry] = []
    for p in paths:
        src = Path(p)
        if not src.exists():
            raise FileNotFoundError(f"raw-zone source missing: {src}")
        sha = _sha256(src)
        dest = root / sha / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        st = src.stat()
        entries.append(
            RawEntry(
                name=src.name,
                original_path=str(src),
                sha256=sha,
                bytes=st.st_size,
                mtime=st.st_mtime,
            )
        )
    (root / "raw_manifest.json").write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return entries


def raw_path(entry: RawEntry, *, raw_root: Path | str = DEFAULT_RAW_ROOT) -> Path:
    """The content-addressed location of a raw-zone file (what parsers read)."""
    return Path(raw_root) / entry.sha256 / entry.name


__all__ = ["DEFAULT_RAW_ROOT", "RawEntry", "ingest_to_raw_zone", "raw_path"]
