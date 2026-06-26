"""One-shot daily trading cycle (broker-agnostic).

This is the thin orchestration layer that links the data plane
(quantdata snapshot + quantcore signal artifact) to the execution
plane (rebalance → gate → broker → ledger → persistence).

It is deliberately **not** a long-running process. A scheduler (cron,
systemd timer, the ``schedule`` Skill, etc.) invokes
``run_daily_cycle`` once per session close. State is carried between
cycles via the ``DuckDBStore`` (``previous_state`` is loaded from the
prior run's row).

S22 PR4.0 renamed the entrypoint from ``run_paper_cycle`` to
``run_daily_cycle`` and added an optional ``pull_broker_snapshot``
parameter for pre-/post-trade reconciliation against a broker's
ground-truth book. ``run_paper_cycle`` is kept as a one-line
back-compat wrapper that invokes ``run_daily_cycle`` with
``pull_broker_snapshot=None`` (the no-reconcile path matching
``PaperBroker``'s no-network semantics). Existing callers that pass
``broker=PaperBroker(...)`` continue to work unchanged.

Contract
--------
Given:
    - ``as_of``:           session-close timestamp (tz-naive US/Eastern wall clock).
    - ``snapshot_source``: produces ``MarketSnapshot`` PIT-safe.
    - ``signal_artifact``: on-disk handoff from quantcore.
    - ``state``:           opening ``PortfolioState`` (previous close).
    - ``constraints``:     rebalance knobs (max gross leverage, lot, etc.).
    - ``gate``:            pre-trade risk gate.
    - ``broker``:          PaperBroker, IBKRBroker, or any AbstractBroker.
    - ``tracker``:         lifecycle FSM wrapping the ledger.
    - ``pull_broker_snapshot`` (optional): zero-arg callable returning a
                            ``BrokerSnapshot``. When provided,
                            ``assert_reconciled`` runs before order
                            construction (pre-trade) and after fills
                            are applied (post-trade). When ``None``,
                            no reconcile happens (matching the
                            pre-S22 ``run_paper_cycle`` semantics).
    - ``store`` (optional): DuckDBStore for run persistence.

Produce:
    - ``PaperCycleResult`` bundling (final state, fills, rejections,
       journal digest, run_id).

Each call produces **one** row in ``runs`` in the store (if supplied),
plus the full ledger hash-chain in ``runs.journal_digest``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID, uuid4

import pandas as pd

from quantengine.audit.journal import chain_digest
from quantengine.contracts.orders import Fill
from quantengine.data.signal import SignalArtifact
from quantengine.data.snapshot import SnapshotSource
from quantengine.execution.broker import AbstractBroker
from quantengine.execution.order_state import OrderTracker
from quantengine.portfolio.constraints import RebalanceConstraints
from quantengine.portfolio.rebalance import RebalanceEngine
from quantengine.portfolio.state import PortfolioState
from quantengine.risk.gate import RiskGate, RiskRejection
from quantengine.runtime.reconcile import (
    BrokerSnapshot,
    ReconciliationError,
    assert_reconciled,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    """Outcome of one ``run_daily_cycle`` call.

    Attributes
    ----------
    run_id          : UUID assigned to this cycle.
    as_of           : session-close timestamp the cycle ran at.
    initial_state   : snapshot of ``PortfolioState`` at cycle entry.
    final_state     : new ``PortfolioState`` after all fills applied.
    orders_built    : orders produced by ``RebalanceEngine``.
    orders_accepted : orders that survived the ``RiskGate``.
    rejections      : ``list[RiskRejection]`` for any rejected orders.
    fills           : broker fills for the accepted orders.
    journal_digest  : hex digest of the ledger's hash-chain for this run.
    """

    run_id: UUID
    as_of: pd.Timestamp
    initial_state: PortfolioState
    final_state: PortfolioState
    orders_built: tuple
    orders_accepted: tuple
    rejections: tuple[RiskRejection, ...]
    fills: tuple[Fill, ...]
    journal_digest: str

    @property
    def n_orders(self) -> int:
        return len(self.orders_built)

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    def summary(self) -> str:
        return (
            f"[{self.as_of.isoformat()}] run_id={str(self.run_id)[:8]}… "
            f"built={self.n_orders} accepted={len(self.orders_accepted)} "
            f"rejected={len(self.rejections)} fills={self.n_fills} "
            f"cash={self.final_state.cash:,.2f} "
            f"digest={self.journal_digest[:12]}…"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_daily_cycle(
    as_of: pd.Timestamp,
    *,
    snapshot_source: SnapshotSource,
    signal_artifact: SignalArtifact,
    state: PortfolioState,
    constraints: RebalanceConstraints,
    gate: RiskGate,
    broker: AbstractBroker,
    tracker: OrderTracker,
    pull_broker_snapshot: Callable[[], BrokerSnapshot] | None = None,
    store: Any | None = None,
    run_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> PaperCycleResult:
    """Execute one session-close rebalance end to end (broker-agnostic).

    Sequence
    --------
    0. (Optional) Pre-trade reconcile: if ``pull_broker_snapshot`` is
       provided, fetch a ``BrokerSnapshot`` and call
       ``assert_reconciled(state, snapshot, ledger=tracker.ledger)``.
       Raises ``ReconciliationError`` on drift before any orders are
       constructed — fail loud, not silent.
    1. Read ``AlphaSignal`` from ``signal_artifact``.
    2. Resolve ``MarketSnapshot`` via ``snapshot_source.load(as_of, sig.tickers)``.
    3. Build orders via ``RebalanceEngine.rebalance(sig, state, market)``.
    4. Validate with ``RiskGate``:
         - rejected orders are submitted-then-rejected in the tracker so the
           lifecycle FSM records both edges (``PENDING→SUBMITTED→REJECTED``);
         - accepted orders go through ``tracker.submit`` only.
    5. Submit accepted orders to the broker; apply resulting fills through
       ``state.apply`` and record them in the tracker.
    6. (Optional) Post-trade reconcile: if ``pull_broker_snapshot`` is
       provided, fetch a fresh snapshot and call
       ``assert_reconciled(new_state, snapshot, ledger=tracker.ledger)``.
       Raises ``ReconciliationError`` if final state diverges from the
       broker's ground-truth book.
    7. Persist the run (if ``store`` is provided) and return a
       ``PaperCycleResult``.

    The state returned in ``PaperCycleResult.final_state`` is the
    opening book for tomorrow's cycle.
    """
    as_of = pd.Timestamp(as_of)
    rid = run_id or uuid4()
    initial_state = state

    # --- 0. Pre-trade reconcile (optional) -------------------------------
    if pull_broker_snapshot is not None:
        pre_snapshot = pull_broker_snapshot()
        try:
            assert_reconciled(state, pre_snapshot, ledger=tracker.ledger)
        except ReconciliationError as e:
            # Fail-closed must also fail-LOUD (s89, F24 third lesson): a refusal
            # leaves a permanent run row so absence-of-trades is forever
            # distinguishable from absence-of-attempts. The ledger already carries
            # the RECONCILE drift event appended by assert_reconciled; the row
            # persists the UNCHANGED opening state, so it is state-neutral for any
            # consumer reconstructing a book from the latest run.
            if store is not None:
                failed_metadata: dict[str, Any] = {
                    "as_of": as_of.isoformat(),
                    "status": "FAILED_PRE_TRADE_RECONCILE",
                    "reason": str(e),
                }
                if metadata:
                    failed_metadata.update(metadata)
                    failed_metadata["status"] = "FAILED_PRE_TRADE_RECONCILE"
                store.save_run(
                    state,
                    tracker.ledger,
                    run_id=rid,
                    initial_cash=float(initial_state.cash),
                    metadata=failed_metadata,
                )
            raise

    # --- 1. Load signal ---------------------------------------------------
    sig = signal_artifact.read()

    # --- 2. Load market snapshot (PIT-safe by construction) --------------
    market = snapshot_source.load(as_of, sig.tickers)

    # --- 3. Build orders --------------------------------------------------
    engine = RebalanceEngine(constraints=constraints)
    orders = engine.rebalance(sig, state, market)
    orders_tuple = tuple(orders)

    # --- 4. Risk gate -----------------------------------------------------
    accepted, rejections = gate.validate(orders, state, market)
    rejected_set = {r.order.order_id for r in rejections}

    # Record rejected orders into the tracker before submitting accepted
    # ones. This preserves FSM transitions for the audit log even though
    # no broker call occurs for the rejected set.
    for rj in rejections:
        tracker.submit(rj.order, market.timestamp)
        tracker.reject(rj.order.order_id, market.timestamp, reason=rj.reason)

    # --- 5. Submit the survivors + apply fills ---------------------------
    for o in accepted:
        # Defensive: accepted list should have no overlap with rejected ids,
        # but guard against future gate implementations that don't prune.
        if o.order_id in rejected_set:
            continue
        tracker.submit(o, market.timestamp)

    fills = broker.submit_orders(list(accepted), market)
    new_state = state
    for f in fills:
        tracker.on_fill(f)
        new_state = new_state.apply(f)

    # --- 6. Post-trade reconcile (optional) ------------------------------
    if pull_broker_snapshot is not None:
        post_snapshot = pull_broker_snapshot()
        assert_reconciled(new_state, post_snapshot, ledger=tracker.ledger)

    # --- 7. Persist + summarize ------------------------------------------
    digest = chain_digest(tracker.ledger.events()).digest
    if store is not None:
        persist_metadata = {
            "as_of": as_of.isoformat(),
            "status": "OK",  # s89: rows without this key (pre-s89 history) read as OK
            "n_orders_built": len(orders_tuple),
            "n_accepted": len(accepted),
            "n_rejected": len(rejections),
            "n_fills": len(fills),
        }
        if metadata:
            persist_metadata.update(metadata)
        store.save_run(
            new_state,
            tracker.ledger,
            run_id=rid,
            initial_cash=float(initial_state.cash),
            metadata=persist_metadata,
        )

    return PaperCycleResult(
        run_id=rid,
        as_of=as_of,
        initial_state=initial_state,
        final_state=new_state,
        orders_built=orders_tuple,
        orders_accepted=tuple(accepted),
        rejections=tuple(rejections),
        fills=tuple(fills),
        journal_digest=digest,
    )


def run_paper_cycle(
    as_of: pd.Timestamp,
    *,
    snapshot_source: SnapshotSource,
    signal_artifact: SignalArtifact,
    state: PortfolioState,
    constraints: RebalanceConstraints,
    gate: RiskGate,
    broker: AbstractBroker,
    tracker: OrderTracker,
    store: Any | None = None,
    run_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> PaperCycleResult:
    """Back-compat wrapper for ``run_daily_cycle`` (S22 PR4.0).

    Pre-S22 callers invoked ``run_paper_cycle`` with ``broker=PaperBroker(...)``
    (or any other ``AbstractBroker``) and got a no-reconcile cycle. This
    wrapper preserves that semantic by delegating to ``run_daily_cycle``
    with ``pull_broker_snapshot=None``. New code that needs broker-side
    reconcile should call ``run_daily_cycle`` directly with a non-None
    provider — see ``runtime.ibkr_daily_cycle.run_ibkr_paper_cycle`` for
    the IBKR-specific construction pattern.
    """
    return run_daily_cycle(
        as_of,
        snapshot_source=snapshot_source,
        signal_artifact=signal_artifact,
        state=state,
        constraints=constraints,
        gate=gate,
        broker=broker,
        tracker=tracker,
        pull_broker_snapshot=None,
        store=store,
        run_id=run_id,
        metadata=metadata,
    )


__all__ = ["PaperCycleResult", "run_daily_cycle", "run_paper_cycle"]
