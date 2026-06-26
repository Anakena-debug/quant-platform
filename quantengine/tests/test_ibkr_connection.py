"""Tests for IBKRConnection + assert_paper_account + derive_client_id.

All tests use mocked ``ib_async.IB`` instances (no real network
calls). The end-to-end smoke test against a real paper account lives
in PR5's ``test_ibkr_paper_smoke.py`` and is gated behind a marker +
env-var.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from quantengine.execution.ibkr import connection as connection_mod
from quantengine.execution.ibkr.config import IBKRConfig
from quantengine.execution.ibkr.connection import (
    IBKRCircuitOpen,
    IBKRConnection,
    assert_paper_account,
    derive_client_id,
)


# ---- derive_client_id ------------------------------------------------


def test_derive_client_id_uses_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """``derive_client_id`` pins to ``base + (os.getpid() % 900)``."""
    # PID 42 → 100 + 42 % 900 = 142.
    monkeypatch.setattr("os.getpid", lambda: 42)
    assert derive_client_id(base=100) == 142
    # PID 1042 → 100 + 1042 % 900 = 100 + 142 = 242.
    monkeypatch.setattr("os.getpid", lambda: 1042)
    assert derive_client_id(base=100) == 242
    # Boundary: PID at the high end of the 900-slot range.
    monkeypatch.setattr("os.getpid", lambda: 899)
    assert derive_client_id(base=100) == 999
    # Wrap: PID 900 → 100 + 0 = 100 (back to base).
    monkeypatch.setattr("os.getpid", lambda: 900)
    assert derive_client_id(base=100) == 100


# ---- IBKRConnection lifecycle ---------------------------------------


def _make_config(client_id: int = 42) -> IBKRConfig:
    return IBKRConfig(host="127.0.0.1", port=7497, client_id=client_id, account="DU123")


def _patch_ib_factory(monkeypatch: pytest.MonkeyPatch, mock_ib: MagicMock) -> None:
    """Patch the ``_new_ib`` factory in the connection module so that
    constructing ``IBKRConnection`` returns ``mock_ib`` as the
    underlying IB instance.
    """
    monkeypatch.setattr(connection_mod, "_new_ib", lambda: mock_ib)


def test_ibkr_connection_connect_disconnect_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``connect(cfg)`` calls ``ib.connect`` once with cfg fields.

    ``is_connected()`` reflects state; ``disconnect()`` routes through.
    """
    mock_ib = MagicMock()
    _patch_ib_factory(monkeypatch, mock_ib)

    conn = IBKRConnection()
    assert conn.is_connected() is False  # not yet connected
    cfg = _make_config()
    conn.connect(cfg)
    # Underlying ib.connect called once with cfg fields.
    mock_ib.connect.assert_called_once()
    kwargs = mock_ib.connect.call_args.kwargs
    assert kwargs["host"] == cfg.host
    assert kwargs["port"] == cfg.port
    assert kwargs["clientId"] == cfg.client_id
    assert kwargs["account"] == cfg.account
    assert conn.is_connected() is True
    # Disconnect routes through to the underlying IB.
    conn.disconnect()
    mock_ib.disconnect.assert_called_once()
    assert conn.is_connected() is False
    # Idempotent: second disconnect is a no-op.
    conn.disconnect()
    assert mock_ib.disconnect.call_count == 1


def test_ibkr_connection_reconnect_on_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First two connects raise ConnectionError; third attempt succeeds.

    Exercises the exponential-backoff retry path. ``time.sleep`` is
    monkeypatched to a no-op to keep the test fast.
    """
    call_count = {"n": 0}

    def flaky_connect(**_kwargs: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise ConnectionError("simulated transient failure")
        # Third attempt succeeds.

    mock_ib = MagicMock()
    mock_ib.connect.side_effect = flaky_connect
    _patch_ib_factory(monkeypatch, mock_ib)
    monkeypatch.setattr("time.sleep", lambda _: None)

    conn = IBKRConnection()
    conn.reconnect(_make_config())
    # Eventually succeeded on the third attempt.
    assert conn.is_connected() is True
    assert call_count["n"] == 3


def test_ibkr_connection_circuit_breaker_after_n_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three consecutive ``connect()`` failures open the circuit.

    The fourth call hits the circuit-open guard and raises
    ``IBKRCircuitOpen`` immediately without invoking ``ib_async``.
    """

    def always_fail(**_kwargs: Any) -> None:
        raise ConnectionError("simulated persistent failure")

    mock_ib = MagicMock()
    mock_ib.connect.side_effect = always_fail
    _patch_ib_factory(monkeypatch, mock_ib)
    monkeypatch.setattr("time.sleep", lambda _: None)

    conn = IBKRConnection()
    cfg = _make_config()
    # First three calls fail; the third opens the circuit.
    for _ in range(3):
        with pytest.raises(ConnectionError):
            conn.connect(cfg)
    pre_ib_call_count = mock_ib.connect.call_count
    # Fourth call: circuit-open path; ib.connect is NOT invoked again.
    with pytest.raises(IBKRCircuitOpen, match="circuit-breaker is open"):
        conn.connect(cfg)
    assert mock_ib.connect.call_count == pre_ib_call_count


# ---- assert_paper_account -------------------------------------------


def test_ibkr_connection_assert_paper_account_passes_when_account_in_managed_set() -> None:
    """``managedAccounts()`` returns the expected account → no raise."""
    mock_ib = MagicMock()
    mock_ib.managedAccounts.return_value = ["DU123"]
    assert_paper_account(mock_ib, "DU123")  # no raise expected


def test_ibkr_connection_assert_paper_account_fails_when_account_not_in_managed_set() -> None:
    """``managedAccounts()`` lacks the expected → ``RuntimeError``.

    Two failure shapes covered:

    1. Different paper account (session bound to wrong DU... ID).
    2. Live account (session is a live TWS instance returning
       U-prefixed IDs — caught here by the cross-check even though
       the configured port was paper).
    """
    # Case 1: different DU account.
    mock_ib_diff_paper = MagicMock()
    mock_ib_diff_paper.managedAccounts.return_value = ["DU456"]
    with pytest.raises(RuntimeError, match=r"\['DU456'\]"):
        assert_paper_account(mock_ib_diff_paper, "DU123")

    # Case 2: live account (paper port wired to live TWS instance).
    mock_ib_live = MagicMock()
    mock_ib_live.managedAccounts.return_value = ["U7654321"]
    with pytest.raises(RuntimeError, match=r"\['U7654321'\]"):
        assert_paper_account(mock_ib_live, "DU123")
