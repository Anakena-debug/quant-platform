"""Tests for quantengine.runtime.streaming.cli.

Pins:

- ``--help`` exits 0 (AC11; the in-process check; the
  ``shell:cd quantengine && uv run python -m ...`` AC fires that same
  invocation from the AC pipeline).
- ``build_parser()`` constructs without error and exposes the three
  documented subcommands (start, replay, status — D13).
- ``cmd_replay`` round-trips a JSON-line journal: writes, then
  replays, and stdout contains every record.

quantcore-independence: no quantcore imports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from quantengine.runtime.streaming.cli import (
    build_parser,
    cmd_replay,
    cmd_status,
    main,
)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------
def test_build_parser_returns_argparse_parser() -> None:
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_three_documented_subcommands_present() -> None:
    """D13: start / replay / status. The parser must expose all three."""
    parser = build_parser()
    # argparse stores subparsers via the action's choices map.
    sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(sub_actions) == 1
    choices = set(sub_actions[0].choices.keys())
    assert choices == {"start", "replay", "status"}, choices


# ---------------------------------------------------------------------------
# --help exits 0 (AC11 in-process)
# ---------------------------------------------------------------------------
def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "start" in out
    assert "replay" in out
    assert "status" in out


def test_no_subcommand_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    """argparse with required=True forces SystemExit on missing subcommand."""
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# replay subcommand — handcrafted JSONL round-trip
# ---------------------------------------------------------------------------
def test_replay_prints_each_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = tmp_path / "replay.jsonl"
    records = [
        {"ts_ns": 100, "event_type": "submit", "order_id": "o1"},
        {"ts_ns": 200, "event_type": "fill", "fill_id": "f1", "order_id": "o1"},
    ]
    journal.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    args = argparse.Namespace(journal=str(journal))
    rc = cmd_replay(args)
    captured = capsys.readouterr()

    assert rc == 0
    # Each record present in stdout.
    assert "submit" in captured.out
    assert "fill" in captured.out


def test_replay_missing_file_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(journal=str(tmp_path / "does_not_exist.jsonl"))
    rc = cmd_replay(args)
    assert rc == 1


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------
def test_status_no_pidfile_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(pidfile=None)
    rc = cmd_status(args)
    assert rc == 1


def test_status_with_pidfile_prints_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pf = tmp_path / "engine.pid"
    pf.write_text("12345\n")
    args = argparse.Namespace(pidfile=str(pf))
    rc = cmd_status(args)
    captured = capsys.readouterr()
    assert rc == 0
    assert "12345" in captured.out
