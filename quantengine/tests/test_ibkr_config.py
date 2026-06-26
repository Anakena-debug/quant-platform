"""Tests for IBKRConfig + TimeoutPolicy + paper-prefix regex + port whitelist."""

from __future__ import annotations

import pytest

from quantengine.execution.ibkr.config import (
    PAPER_PORTS,
    IBKRConfig,
    TimeoutPolicy,
)


# ---- TimeoutPolicy ---------------------------------------------------


def test_timeout_policy_defaults() -> None:
    """TimeoutPolicy default values match the plan-pinned policy."""
    p = TimeoutPolicy()
    assert p.per_order_seconds == 60.0
    assert p.batch_ceiling_seconds == 300.0
    # Override-only-one path: defaults still apply where unspecified.
    p2 = TimeoutPolicy(per_order_seconds=10.0)
    assert p2.per_order_seconds == 10.0
    assert p2.batch_ceiling_seconds == 300.0


# ---- Port whitelist (layer 3 of paper-account gate) ------------------


def test_ibkr_config_accepts_paper_ports_only() -> None:
    """Both paper ports (TWS=7497, Gateway=4002) construct successfully."""
    cfg_tws = IBKRConfig(host="127.0.0.1", port=7497, client_id=42, account="DU123")
    assert cfg_tws.port == 7497
    cfg_gw = IBKRConfig(host="127.0.0.1", port=4002, client_id=42, account="DU123")
    assert cfg_gw.port == 4002
    # Both ports are members of PAPER_PORTS.
    assert cfg_tws.port in PAPER_PORTS
    assert cfg_gw.port in PAPER_PORTS
    assert PAPER_PORTS == frozenset({7497, 4002})


def test_ibkr_config_rejects_live_port() -> None:
    """Live ports (TWS live=7496, Gateway live=4001) raise ValueError.

    Tests the layer-3 defence-in-depth gate. The error message must
    contain the literal substring "paper-only" to identify the
    failure mode unambiguously.
    """
    for live_port in (7496, 4001):
        with pytest.raises(ValueError, match="paper-only"):
            IBKRConfig(
                host="127.0.0.1",
                port=live_port,
                client_id=42,
                account="DU123",
            )


# ---- Paper-prefix regex (layer 1 of paper-account gate) --------------


def test_ibkr_config_accepts_du_duh_df_prefixes() -> None:
    """All three paper-account prefix variants pass layer-1 regex.

    Verified against DUM268500 during pre-PR1 REPL verification on
    2026-05-07: the ``DUM`` prefix is matched by ``^D[UFH]`` because
    the regex constrains positions 0-1 only (D + U/F/H), with the
    tail unconstrained.
    """
    for account in ("DU123", "DUH456", "DF789", "DUM268500"):
        cfg = IBKRConfig(host="127.0.0.1", port=7497, client_id=42, account=account)
        assert cfg.account == account


def test_ibkr_config_rejects_non_paper_prefix() -> None:
    """Live-account prefixes (U..., X..., DD...) raise ValueError.

    The error message contains the literal substring "paper-prefix"
    to identify the failure mode unambiguously.
    """
    for bad_account in ("U123", "X123", "DD123", ""):
        with pytest.raises(ValueError, match="paper-prefix"):
            IBKRConfig(
                host="127.0.0.1",
                port=7497,
                client_id=42,
                account=bad_account,
            )


# ---- from_env --------------------------------------------------------


def test_ibkr_config_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IBKRConfig.from_env`` reads the four documented env vars."""
    monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
    monkeypatch.setenv("IBKR_PORT", "7497")
    monkeypatch.setenv("IBKR_CLIENT_ID", "42")
    monkeypatch.setenv("IBKR_ACCOUNT", "DU123")
    cfg = IBKRConfig.from_env()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 7497
    assert cfg.client_id == 42
    assert cfg.account == "DU123"
    # Default factory: TimeoutPolicy with default values.
    assert cfg.timeouts.per_order_seconds == 60.0
    assert cfg.timeouts.batch_ceiling_seconds == 300.0

    # Missing variable -> KeyError naming the missing variable.
    monkeypatch.delenv("IBKR_ACCOUNT", raising=False)
    with pytest.raises(KeyError, match="IBKR_ACCOUNT"):
        IBKRConfig.from_env()
