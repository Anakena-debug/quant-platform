"""IBKR connection wrapper: ``ib_async.IB()`` lifecycle, reconnect with
backoff, circuit-breaker, paper-account assertion, PID-derived
clientId.

Layer 2 of the two-layer paper-account gate lives here:
``assert_paper_account(ib, expected_account)`` calls
``ib.managedAccounts()`` and raises if ``expected_account`` is not in
the returned list. The TWS session's account binding is the
authoritative paper-vs-live signal — paper TWS instances return
DU-prefixed IDs in ``managedAccounts()``; live TWS instances return
U-prefixed IDs.

The structural-classification field from ``accountSummary()`` was
originally proposed as the layer-2 source of truth but verified
empirically on 2026-05-07 to NOT distinguish paper from live (paper
account DUM268500 was classified as ``'INDIVIDUAL'``). That field
reflects account *structure* — individual / joint / IRA / trust /
corporation — not *environment* (paper / live).

Mid-cycle disconnect is hard-fail by design: the ``IBKRBroker``
(PR2) raises if the socket drops mid-cycle. The daily cycle is
restartable manually. Phase 4 will add ``OrderTracker`` persistence
and replay-on-reconnect; out of scope for S22.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING

from quantengine.execution.ibkr.config import IBKRConfig

if TYPE_CHECKING:
    from ib_async import IB

# Reconnect backoff schedule (seconds). Exponential up to ~4s.
_BACKOFF_DELAYS_SECONDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
# Circuit breaker: opens after this many consecutive connect() failures.
_CIRCUIT_BREAKER_THRESHOLD: int = 3
# Cooldown after the breaker opens (seconds). 5 minutes.
_CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 300.0


class IBKRCircuitOpen(RuntimeError):
    """Raised when ``IBKRConnection.connect()`` is called while the
    circuit-breaker is open.

    The breaker opens after ``_CIRCUIT_BREAKER_THRESHOLD`` consecutive
    ``connect()`` failures and refuses retries for
    ``_CIRCUIT_BREAKER_COOLDOWN_SECONDS``. This prevents tight retry
    loops from masking persistent connectivity issues.
    """


def derive_client_id(base: int = 100) -> int:
    """Derive a collision-free clientId from the current process's PID.

    ``ib_async`` requires a unique clientId per concurrent socket. The
    most common collision is a debug REPL bumping a running daily
    cycle when both default to clientId=1. PID-derivation gives each
    process a distinct slot in ``[base, base+900)`` without manual
    coordination. Two PIDs collide only if they land on the same
    residue mod 900, which is rare in practice.

    The default ``base=100`` reserves clientId=0 for TWS itself and
    leaves 1-99 available for hand-set tooling.
    """
    return base + (os.getpid() % 900)


def assert_paper_account(ib: IB, expected_account: str) -> None:
    """Layer 2 of the paper-account gate.

    Cross-checks ``expected_account in ib.managedAccounts()``. A paper
    TWS instance returns DU-prefixed IDs; a live TWS instance returns
    U-prefixed IDs. If the expected account is not in the returned
    list, the connection is bound to a different account or to a live
    TWS instance — raises ``RuntimeError`` whose message lists the
    actual managed accounts for forensic clarity.

    The structural-classification field from ``accountSummary()`` is
    intentionally NOT used. Verified empirically on 2026-05-07 (paper
    account DUM268500 was classified as 'INDIVIDUAL') to NOT
    distinguish paper from live — that field reflects account
    *structure*, not *environment*.
    """
    managed = list(ib.managedAccounts())
    if expected_account not in managed:
        raise RuntimeError(
            f"account {expected_account!r} not in managedAccounts() "
            f"= {managed!r}. The TWS session is bound to a different "
            "account; refusing to proceed. Common cause: paper port "
            "wired to a live TWS instance, in which case "
            "managedAccounts() returns U-prefixed (live) IDs."
        )


@dataclass
class IBKRConnection:
    """Lifecycle-managed wrapper over ``ib_async.IB()``.

    Construction is cheap and side-effect-free; networking happens
    only inside ``connect()``. The wrapper provides:

    - explicit ``connect(config)`` / ``disconnect()`` calls;
    - ``with`` context-manager support that disconnects on exit;
    - ``reconnect(config)`` with exponential backoff
      (``_BACKOFF_DELAYS_SECONDS``);
    - a circuit-breaker: 3 consecutive ``connect()`` failures open
      the breaker for 5 minutes; further calls during that window
      raise ``IBKRCircuitOpen`` immediately without invoking
      ``ib_async``.

    The wrapper does NOT replay orders on reconnect — that's
    ``OrderTracker`` / ``Ledger`` territory. The connection is a
    pure transport. Mid-cycle disconnect propagates to the caller as
    a hard-fail (Phase 3 limitation; Phase 4 adds replay).
    """

    # Underlying ib_async.IB instance, lazily constructed via factory
    # so tests can monkeypatch ``IB`` in this module's namespace.
    _ib: IB = field(default_factory=lambda: _new_ib(), init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _circuit_opened_at: float | None = field(default=None, init=False, repr=False)

    @property
    def ib(self) -> IB:
        return self._ib

    def is_connected(self) -> bool:
        return self._connected

    def connect(self, config: IBKRConfig) -> None:
        """Open the underlying socket. Raises ``IBKRCircuitOpen`` if the
        breaker is open; raises the underlying ``Exception`` if the
        connection attempt fails.

        Failure increments ``_consecutive_failures``; success resets
        it. When ``_consecutive_failures`` reaches
        ``_CIRCUIT_BREAKER_THRESHOLD``, the breaker opens.
        """
        self._check_circuit_breaker()
        try:
            self._ib.connect(
                host=config.host,
                port=config.port,
                clientId=config.client_id,
                timeout=config.connect_timeout_seconds,
                readonly=config.read_only,
                account=config.account,
            )
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_opened_at = time.monotonic()
            raise
        self._connected = True
        self._consecutive_failures = 0

    def disconnect(self) -> None:
        """Close the underlying socket. Idempotent."""
        if self._connected:
            self._ib.disconnect()
            self._connected = False

    def reconnect(self, config: IBKRConfig) -> None:
        """Disconnect (if connected) and retry ``connect`` with
        exponential backoff. Propagates ``IBKRCircuitOpen``
        immediately if the breaker opens mid-retry.
        """
        self.disconnect()
        last_error: BaseException | None = None
        for delay in _BACKOFF_DELAYS_SECONDS:
            try:
                self.connect(config)
                return
            except IBKRCircuitOpen:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(delay)
        raise RuntimeError(
            f"reconnect exhausted {len(_BACKOFF_DELAYS_SECONDS)} retries "
            f"({_BACKOFF_DELAYS_SECONDS}); last error: {last_error!r}"
        ) from last_error

    def __enter__(self) -> IBKRConnection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    def _check_circuit_breaker(self) -> None:
        if self._circuit_opened_at is None:
            return
        elapsed = time.monotonic() - self._circuit_opened_at
        if elapsed < _CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            remaining = _CIRCUIT_BREAKER_COOLDOWN_SECONDS - elapsed
            raise IBKRCircuitOpen(
                f"circuit-breaker is open after "
                f"{self._consecutive_failures} consecutive connection "
                f"failures; refusing retries for {remaining:.1f}s more"
            )
        # Cooldown elapsed — reset state and allow retries.
        self._circuit_opened_at = None
        self._consecutive_failures = 0


def _new_ib() -> IB:
    """Construct a fresh ``ib_async.IB()``. Tests monkeypatch
    ``IB`` in this module's namespace to inject mocks; this lambda-
    style factory respects that patching.
    """
    from ib_async import IB as _IB  # noqa: N814 — match the upstream symbol's casing

    return _IB()


__all__ = [
    "IBKRCircuitOpen",
    "IBKRConnection",
    "assert_paper_account",
    "derive_client_id",
]
