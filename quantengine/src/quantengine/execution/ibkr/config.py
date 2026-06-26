"""IBKR configuration: paper-only ports, paper-prefix accounts, env loader.

Layer 1 of the two-layer paper-account gate is enforced here at config-
construction time:

1. ``port`` must be one of ``PAPER_PORTS = {7497, 4002}`` — TWS paper
   and IB Gateway paper. The live equivalents are explicitly absent
   and any attempt to use them raises ``ValueError``.
2. ``account`` must match the regex ``^D[UFH]`` — paper accounts begin
   with ``DU`` (standard), ``DUH`` (advisor sub-accounts), or ``DF``
   (some firmware variants). The ``DUM`` prefix observed during PR1
   verification (paper account DUM268500) matches because the regex
   constrains positions 0-1 only; the tail is unconstrained.

Layer 2 (the authoritative cross-check) lives in
``quantengine.execution.ibkr.connection.assert_paper_account``.

Why two layers? The regex catches the common case fast at construction
time. The runtime cross-check via ``ib.managedAccounts()`` is the
broker-session contract — it confirms (a) the configured account
string is real AND (b) the connection is bound to the TWS instance
that owns it. The ``accountSummary()['AccountType']`` field was
originally proposed but verified empirically on 2026-05-07 to NOT
distinguish paper from live: a paper account returned ``'INDIVIDUAL'``,
proving AccountType reflects account *structure* (individual / joint /
IRA / trust), not *environment* (paper / live).

No credentials live in this module. TWS authentication is handled at
the TWS side via API Settings → Trusted IPs (typically restricted to
``127.0.0.1``). The four operational environment variables
``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_CLIENT_ID``, ``IBKR_ACCOUNT`` are
not secrets — they're connection coordinates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Paper-account ID prefix regex. Widened from ``^DU`` per S22 review:
# DUH covers advisor sub-accounts; DF covers some firmware variants.
# Verified against DUM-prefixed accounts during PR1 REPL verification
# (2026-05-07). The regex is a heuristic; layer 2 (managedAccounts
# cross-check at connect time) is the authoritative gate.
_PAPER_PREFIX = re.compile(r"^D[UFH]")

# Paper-only TWS / Gateway ports. The live equivalents are
# intentionally absent — passing one to ``IBKRConfig`` raises
# ``ValueError`` at construction. Defends against the
# paper-port-wired-to-live-TWS misconfiguration without relying on
# the managedAccounts cross-check (which catches it at connect time
# as a defence-in-depth layer).
PAPER_PORTS: frozenset[int] = frozenset({7497, 4002})


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    """Two-tier timeout for ``IBKRBroker.submit_orders`` (PR2).

    ``per_order_seconds`` is the per-order cancel deadline. Default
    60.0 is roughly 30× the expected p99 paper-fill latency for a
    liquid US equity (paper fills typically land in under 2s). Catches
    Gateway hiccups without masking real failures.

    ``batch_ceiling_seconds`` is the batch-level ceiling across all
    orders submitted in one ``submit_orders`` call. Default 300.0
    (5 minutes) suits an end-of-session rebalance with up to ~50
    orders; raise it for larger batches.

    Both fields are configurable so backtests / CI can override.
    """

    per_order_seconds: float = 60.0
    batch_ceiling_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class IBKRConfig:
    """All configuration needed to open one IBKR paper session.

    Construction-time validation enforces layers 1 + 3 of the paper-
    account gate (regex prefix on account; port whitelist).
    Layer 2 (``managedAccounts()`` cross-check) runs after
    ``connect()`` via ``connection.assert_paper_account``.
    """

    host: str
    port: int
    client_id: int
    account: str
    connect_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 60.0
    read_only: bool = False
    timeouts: TimeoutPolicy = field(default_factory=TimeoutPolicy)

    def __post_init__(self) -> None:
        if self.port not in PAPER_PORTS:
            raise ValueError(
                f"port {self.port} is not in PAPER_PORTS "
                f"{sorted(PAPER_PORTS)}; this connector accepts "
                "paper-only ports (TWS=7497, Gateway=4002). Live ports "
                "are refused by design."
            )
        if not _PAPER_PREFIX.match(self.account):
            raise ValueError(
                f"account {self.account!r} does not match the "
                f"paper-prefix regex {_PAPER_PREFIX.pattern!r}. "
                "Expected DU, DUH, or DF prefix. If your paper account "
                "uses a different prefix, widen the regex in config.py."
            )

    @classmethod
    def from_env(cls) -> IBKRConfig:
        """Build an ``IBKRConfig`` from environment variables.

        Reads ``IBKR_HOST`` (str), ``IBKR_PORT`` (int),
        ``IBKR_CLIENT_ID`` (int), ``IBKR_ACCOUNT`` (str). Missing
        variables raise ``KeyError`` naming the missing variable.
        Operators may set ``IBKR_CLIENT_ID`` to a literal integer or
        delegate to ``connection.derive_client_id()`` at construction.
        """
        try:
            host = os.environ["IBKR_HOST"]
            port = int(os.environ["IBKR_PORT"])
            client_id = int(os.environ["IBKR_CLIENT_ID"])
            account = os.environ["IBKR_ACCOUNT"]
        except KeyError as e:
            raise KeyError(
                f"missing IBKR environment variable: {e.args[0]!r}. "
                "Expected: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, "
                "IBKR_ACCOUNT"
            ) from e
        return cls(host=host, port=port, client_id=client_id, account=account)


__all__ = ["IBKRConfig", "PAPER_PORTS", "TimeoutPolicy"]
