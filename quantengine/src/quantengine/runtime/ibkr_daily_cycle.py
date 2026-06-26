"""IBKR-specific daily cycle wrapper.

Wraps ``run_daily_cycle`` with three IBKR-specific layers:

1. **RTH gate** — aborts if ``as_of`` is outside regular trading
   hours or on a NYSE holiday. Uses the existing ``TradingCalendar``
   (which wraps ``exchange_calendars`` when installed; falls back to
   a 2015-2035 hand-rolled engine that handles DST + observed
   holidays + early closes correctly).
2. **Connection scope** — ``with connection: ...`` ensures that the
   IBKR socket is disconnected deterministically on exit, even on
   exception. The caller is responsible for ``connection.connect()``
   before entering the cycle; the function asserts
   ``is_connected()`` and raises if not.
3. **Layer-2 paper-account assertion at entry** — calls
   ``assert_paper_account(connection.ib, account)`` once per cycle,
   before any orders go out. This is the runtime cross-check via
   ``ib.managedAccounts()`` (the AccountType field was found
   unreliable on 2026-05-07).

Plus a ``snapshot_provider`` closure that wraps the IBKR-specific
``pull_broker_snapshot(connection.ib, account, as_of=...)`` and
feeds it to ``run_daily_cycle`` for pre-/post-trade reconcile.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

import pandas as pd

from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import SnapshotSource
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.ibkr.connection import (
    IBKRConnection,
    assert_paper_account,
)
from quantengine.execution.ibkr.positions import pull_broker_snapshot
from quantengine.execution.order_state import OrderTracker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate
from quantengine.runtime.calendar import TradingCalendar
from quantengine.runtime.daily_cycle import PaperCycleResult, run_daily_cycle
from quantengine.runtime.reconcile import BrokerSnapshot


def run_ibkr_paper_cycle(
    as_of: pd.Timestamp,
    *,
    snapshot_source: SnapshotSource,
    signal_artifact: SignalArtifact,
    state: PortfolioState,
    constraints: RebalanceConstraints,
    gate: RiskGate,
    broker: AbstractBroker,
    tracker: OrderTracker,
    connection: IBKRConnection,
    account: str,
    calendar: TradingCalendar | None = None,
    store: Any | None = None,
    run_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> PaperCycleResult:
    """Execute one session-close rebalance against an IBKR paper account.

    Args:
        as_of: session-close timestamp. May be tz-aware or tz-naive;
            tz-aware values are converted to ``America/New_York``
            wall-clock for RTH comparison.
        connection: an ``IBKRConnection`` that has been ``connect()``-ed
            before this call. The function asserts ``is_connected()``
            and raises if not.
        account: IBKR account ID. Used for ``assert_paper_account``
            (layer-2 cross-check via ``managedAccounts()``) and for
            ``pull_broker_snapshot`` queries against
            ``ib.accountValues(account)`` and ``ib.portfolio(account)``.
        calendar: ``TradingCalendar`` instance. Defaults to a fresh
            ``TradingCalendar()`` (which wraps ``exchange_calendars``
            when installed).
        Other args mirror ``run_daily_cycle``.

    Sequence:
        1. RTH gate: raise ``RuntimeError`` if ``as_of`` is outside
           a regular trading session or outside RTH within a session.
        2. Assert ``connection.is_connected()`` and enter the
           ``with connection`` context (disconnects on exit).
        3. Layer-2 paper-account assertion: raise ``RuntimeError`` if
           the configured ``account`` is not in
           ``ib.managedAccounts()``.
        4. Build a ``snapshot_provider`` closure and delegate to
           ``run_daily_cycle`` with pre-/post-trade reconcile enabled.

    Mid-cycle disconnect is hard-fail by design (S22 Phase 3
    limitation). Any exception inside the ``with connection`` block
    propagates after the disconnect cleanup; callers restart manually.
    """
    cal = calendar or TradingCalendar()
    as_of_ts = cast(pd.Timestamp, pd.Timestamp(as_of))
    # Convert tz-aware to US/Eastern wall-clock for RTH comparison.
    if as_of_ts.tz is not None:
        as_of_et = as_of_ts.tz_convert("America/New_York").tz_localize(None)
    else:
        as_of_et = as_of_ts

    # --- 1. RTH gate -----------------------------------------------------
    if not cal.is_session(as_of_et.date()):
        raise RuntimeError(
            f"{as_of_et.date()} is not a NYSE trading session (weekend or observed holiday). Cycle aborted."
        )
    open_ = cal.session_open(as_of_et.date())
    close_ = cal.session_close(as_of_et.date())
    if not (open_ <= as_of_et <= close_):
        raise RuntimeError(f"as_of {as_of_et} is outside RTH [{open_}, {close_}]. Cycle aborted.")

    # --- 2. Connection check --------------------------------------------
    if not connection.is_connected():
        raise RuntimeError(
            "IBKRConnection must be connected before run_ibkr_paper_cycle(). Call connection.connect(config) first."
        )

    # --- 3 + 4. Connection scope, paper assertion, delegate -------------
    with connection:
        # Layer-2 paper-account assertion at entry.
        assert_paper_account(connection.ib, account)

        # Snapshot closure for pre-/post-trade reconcile.
        snapshot_as_of = as_of_ts.isoformat()

        def snapshot_provider() -> BrokerSnapshot:
            return pull_broker_snapshot(connection.ib, account, as_of=snapshot_as_of)

        return run_daily_cycle(
            as_of_ts,
            snapshot_source=snapshot_source,
            signal_artifact=signal_artifact,
            state=state,
            constraints=constraints,
            gate=gate,
            broker=broker,
            tracker=tracker,
            pull_broker_snapshot=snapshot_provider,
            store=store,
            run_id=run_id,
            metadata=metadata,
        )


__all__ = ["run_ibkr_paper_cycle"]
