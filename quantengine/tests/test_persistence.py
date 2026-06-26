"""Tests for JsonlJournal (S37 PR1).

Coverage:

- ``TestBasicRoundTrip``: append + read-back via plain file IO; FIFO
  ordering preserved across multi-record writes.
- ``TestSelectiveFsync``: mock os.fsync; verify it fires only for
  ``terminal=True`` writes (per D3 policy) and unconditionally on
  ``close()`` (trailing fsync).
- ``TestLockContention``: second open on the same path raises
  ``JsonlLockError`` (single-writer guarantee via O_EXCL on .lock).
- ``TestLifecycle``: append after close raises; close is idempotent;
  context manager exit closes even on exception; lock file removed
  on close.
- ``TestParentDirAutoCreate``: __init__ creates the parent directory
  if absent (operator convenience).
- ``TestLockFilePid``: the .lock file contains the writer PID for
  forensic recovery from orphan locks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from quantengine.runtime.streaming.persistence import (
    JsonlClosedError,
    JsonlJournal,
    JsonlLockError,
)


class TestBasicRoundTrip:
    def test_single_record_round_trip(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with JsonlJournal(journal_path) as j:
            j.append({"type": "Pending", "order_id": "abc", "seq": 1})
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"type": "Pending", "order_id": "abc", "seq": 1}

    def test_multi_record_fifo_ordering(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        records = [{"seq": i, "type": "Submitted"} for i in range(50)]
        with JsonlJournal(journal_path) as j:
            for r in records:
                j.append(r)
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        decoded = [json.loads(line) for line in lines]
        assert decoded == records

    def test_unicode_preserved(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with JsonlJournal(journal_path) as j:
            j.append({"ticker": "BRK.B", "note": "café — non-ASCII"})
        decoded = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[0])
        assert decoded["note"] == "café — non-ASCII"


class TestSelectiveFsync:
    def test_non_terminal_does_not_call_os_fsync(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with patch("os.fsync") as mock_fsync:
            with JsonlJournal(journal_path) as j:
                j.append({"type": "Submitted", "order_id": "x"}, terminal=False)
            # Append did not fsync; only close() fsyncs at exit.
            # So total fsync calls = 1 (the trailing one).
            assert mock_fsync.call_count == 1

    def test_terminal_calls_os_fsync(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with patch("os.fsync") as mock_fsync:
            with JsonlJournal(journal_path) as j:
                j.append({"type": "Fill", "order_id": "x"}, terminal=True)
            # terminal=True fsync + trailing fsync = 2 calls.
            assert mock_fsync.call_count == 2

    def test_mixed_terminal_and_intermediate(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with patch("os.fsync") as mock_fsync:
            with JsonlJournal(journal_path) as j:
                j.append({"type": "Pending"}, terminal=False)
                j.append({"type": "Submitted"}, terminal=False)
                j.append({"type": "Fill"}, terminal=True)
                j.append({"type": "PartiallyFilled"}, terminal=False)
                j.append({"type": "Cancel"}, terminal=True)
            # Two terminal writes + one trailing close fsync = 3.
            assert mock_fsync.call_count == 3

    def test_explicit_fsync_call(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with patch("os.fsync") as mock_fsync:
            with JsonlJournal(journal_path) as j:
                j.append({"type": "Submitted"}, terminal=False)
                j.fsync()
            # Explicit fsync + trailing close fsync = 2.
            assert mock_fsync.call_count == 2

    def test_close_is_unconditional_fsync(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        with patch("os.fsync") as mock_fsync:
            j = JsonlJournal(journal_path)
            j.close()
            # Even with zero appends, close fsyncs the empty file.
            assert mock_fsync.call_count == 1


class TestLockContention:
    def test_second_open_raises_lock_error(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        first = JsonlJournal(journal_path)
        try:
            with pytest.raises(JsonlLockError, match="lock file already exists"):
                JsonlJournal(journal_path)
        finally:
            first.close()

    def test_after_close_lock_released_and_reopen_succeeds(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        first = JsonlJournal(journal_path)
        first.close()
        # Second open on the same path now succeeds (lock released).
        second = JsonlJournal(journal_path)
        try:
            second.append({"ok": True})
        finally:
            second.close()


class TestLifecycle:
    def test_append_after_close_raises(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        j = JsonlJournal(journal_path)
        j.close()
        with pytest.raises(JsonlClosedError, match="closed journal"):
            j.append({"after": "close"})

    def test_fsync_after_close_raises(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        j = JsonlJournal(journal_path)
        j.close()
        with pytest.raises(JsonlClosedError):
            j.fsync()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        j = JsonlJournal(journal_path)
        j.close()
        # Second close is a no-op (no raise).
        j.close()

    def test_lock_file_removed_on_close(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        j = JsonlJournal(journal_path)
        lock_path = j.lock_path
        assert lock_path.exists()
        j.close()
        assert not lock_path.exists()

    def test_context_manager_releases_lock_on_exception(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        lock_path = journal_path.with_name(journal_path.name + ".lock")
        with pytest.raises(RuntimeError, match="strategy fail"):
            with JsonlJournal(journal_path) as j:
                j.append({"before": "raise"})
                raise RuntimeError("strategy fail")
        # Lock released even though an exception propagated.
        assert not lock_path.exists()
        # And the record we wrote before the raise is durable.
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0]) == {"before": "raise"}


class TestParentDirAutoCreate:
    def test_nested_path_creates_parents(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "nested" / "more" / "journal.jsonl"
        assert not journal_path.parent.exists()
        with JsonlJournal(journal_path) as j:
            j.append({"created": "ok"})
        assert journal_path.exists()


class TestLockFilePid:
    def test_lock_file_contains_pid(self, tmp_path: Path) -> None:
        journal_path = tmp_path / "journal.jsonl"
        j = JsonlJournal(journal_path)
        try:
            lock_contents = j.lock_path.read_text(encoding="ascii").strip()
            assert lock_contents == str(os.getpid())
        finally:
            j.close()
