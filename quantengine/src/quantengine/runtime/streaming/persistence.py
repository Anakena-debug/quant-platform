"""JsonlJournal — atomic single-writer JSONL persistence (S37 D3 / D4).

OrderTracker state mutations and any other journal-backed component
append records here. The S35 SafeBroker has its own
`_write_and_fsync` + `JournalRecord` path for broker events (locked
by S37 forbidden_actions); this module is parallel persistence for
the things S35 didn't cover, primarily the OrderTracker transitions
described in D4. RecoveryCoordinator (PR3) reads both journals.

Design per D3:

  Single-writer-per-file enforcement.
      Multiple writers to the same path produce interleaved lines
      under tick rate. ``JsonlJournal.__init__`` acquires an
      exclusive lock by creating a sibling ``<path>.lock`` file via
      ``O_CREAT | O_EXCL``. A second open on the same path raises
      ``JsonlLockError``. The lock is released at ``close()`` (or
      ``__exit__``).

  Selective fsync (NOT per-write).
      Per-write ``os.fsync`` costs 50µs-10ms depending on disk
      class; at 100 events/s sustained that's 5-100% wallclock in
      fsync alone. The policy:

        ``terminal=True``  → write + flush + os.fsync(fd)
        ``terminal=False`` → write + flush only (kernel buffer; OS
                              eventually persists)

      The caller marks terminal-state records (``Fill``, ``Cancel``,
      ``Reject``) — these cannot be reconstructed from a broker
      query on restart, so they must hit disk before the call
      returns. Intermediate states (``Pending``, ``Submitted``,
      ``PartiallyFilled``) are reconstructable via broker query
      during recovery (D5 step 3a) and don't need per-record fsync.

  Trailing fsync on close.
      The final flush + os.fsync runs unconditionally at close()
      (and __exit__), so any intermediate records buffered since
      the last terminal fsync land on disk before exit. Graceful
      shutdown (and signal-caught shutdown that calls close())
      preserves the journal tail.

The format is JSON-lines (one record = one line of UTF-8 JSON). No
schema is enforced here — callers pass dicts; the journal is
schema-agnostic so multiple S37 components (OrderTracker, future
reconciler-event-log, etc.) can share the type. RecoveryCoordinator
in PR3 owns the schema interpretation.

What this module does NOT do:

  - Schema validation. JSON-encodable dicts only; callers ensure
    type discipline.
  - Concurrent writers. The lock guarantees one writer per file;
    multi-writer is out of scope (the S37 workload is one engine
    one journal).
  - Rotation. Operator-driven per D3 ("Daily rotation, operator-
    driven").
  - Replay. RecoveryCoordinator does this in PR3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from typing import IO


class JsonlJournalError(IOError):
    """Base class for JsonlJournal failures."""


class JsonlLockError(JsonlJournalError):
    """Raised when the journal's sibling ``.lock`` file already exists.

    Mostly indicates either (a) another process holds the journal,
    or (b) a previous process crashed without releasing the lock.
    Operators resolve case (b) by deleting the orphan ``.lock`` file
    after confirming no live writer.
    """


class JsonlClosedError(JsonlJournalError):
    """Raised when append/fsync/close are called on a closed journal."""


_LOCK_SUFFIX: Final[str] = ".lock"


class JsonlJournal:
    """Atomic single-writer JSONL persistence with selective fsync.

    Use as a context manager (preferred — guarantees lock release on
    exception) or explicitly via ``close()``::

        with JsonlJournal(path) as j:
            j.append({"type": "Pending", "order_id": ...})
            j.append({"type": "Fill", ...}, terminal=True)

    The lock file path is ``<journal_path>.lock`` and contains the
    writer PID for forensic purposes.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_name(self._path.name + _LOCK_SUFFIX)
        self._lock_fd: int | None = None
        self._file: IO[str] | None = None
        self._open()

    def _open(self) -> None:
        # Ensure the parent directory exists; callers shouldn't have
        # to mkdir before constructing the journal.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_EXCL is the atomic guard against a concurrent open.
            self._lock_fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as e:
            raise JsonlLockError(
                f"journal lock file already exists: {self._lock_path}. "
                "Another writer may hold the journal, or a previous "
                "writer crashed without releasing the lock. Resolve "
                "by confirming no live writer + deleting the orphan "
                ".lock file."
            ) from e
        try:
            os.write(self._lock_fd, f"{os.getpid()}\n".encode("ascii"))
        except OSError:
            # If the write fails, release the lock so we don't leak it.
            os.close(self._lock_fd)
            self._lock_fd = None
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
            raise
        # Open the data file in append-text mode. UTF-8 for portability.
        self._file = self._path.open("a", encoding="utf-8")

    def append(self, record: dict[str, Any], *, terminal: bool = False) -> None:
        """Serialize ``record`` to JSON and append one line.

        ``terminal=True`` forces an ``os.fsync`` after the write —
        use for Fill/Cancel/Reject records that the recovery path
        cannot reconstruct from a broker query. Default is False
        (write+flush only; kernel buffers persist eventually).
        """
        if self._file is None:
            raise JsonlClosedError("append() on a closed journal")
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        self._file.write(line)
        self._file.flush()
        if terminal:
            os.fsync(self._file.fileno())

    def fsync(self) -> None:
        """Explicit fsync. Useful for periodic flushes outside the
        terminal-state policy (e.g., snapshot boundaries)."""
        if self._file is None:
            raise JsonlClosedError("fsync() on a closed journal")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        """Trailing fsync + release lock.

        Idempotent: a second close() on an already-closed journal
        is a no-op. The trailing fsync runs unconditionally so any
        intermediate-state records buffered since the last terminal
        fsync land on disk.
        """
        if self._file is None:
            return
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        finally:
            self._file.close()
            self._file = None
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass

    @property
    def path(self) -> Path:
        """The journal data file path."""
        return self._path

    @property
    def lock_path(self) -> Path:
        """The journal's sibling lock file path."""
        return self._lock_path

    def __enter__(self) -> JsonlJournal:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()


__all__ = [
    "JsonlClosedError",
    "JsonlJournal",
    "JsonlJournalError",
    "JsonlLockError",
]
