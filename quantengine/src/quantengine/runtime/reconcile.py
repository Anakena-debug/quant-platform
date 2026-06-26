"""Broker-vs-internal state reconciliation.

Why this exists
---------------
Every quant desk eventually experiences the same bug class: at some
point, the broker's true position differs from the engine's internal
``PortfolioState``. Causes include:

- A fill that arrived after the engine's event loop exited.
- A corporate action the broker applied but we didn't (splits,
  dividends, symbol changes).
- A manual cancel or position unwind outside the engine.
- A restart that replayed a truncated ledger.
- A session-gap where the IBKR gateway dropped the order-event stream.

If this drift goes undetected, every subsequent decision is based on a
fictitious book. The only safe pattern is: at every session boundary
(market open, market close, on restart), pull the broker's ground-truth
snapshot and compare.

This module defines the *data contracts* and the *comparison logic*.
Wiring up the broker-snapshot source is broker-specific and will land
in the IBKR adapter (Phase 3); for now, tests and the paper broker can
supply a ``BrokerSnapshot`` by hand.

Design principles
-----------------
- **Fail loud, not silent**: reconciliation produces a
  ``ReconcileReport``; callers decide whether to halt, auto-correct,
  or log. The opinionated helper ``assert_reconciled`` raises
  ``ReconciliationError`` if *any* tolerance is exceeded.
- **Integer shares, dollar cash**: equity positions compared by
  integer-share equality; cash by absolute tolerance (default $1).
- **Whitelist / ignore list**: tickers we intentionally hold off-book
  (e.g., a manual hedge) can be excluded via ``ignore_tickers``.
- **No mutation**: the reconciler never rewrites internal state. If
  auto-correction is desired, the caller applies a
  ``CASH_ADJ`` / ``RECONCILE`` event to the ledger and derives a
  corrected state explicitly.

Math
----
For each ticker in ``internal ∪ broker``:

.. math::

    \\Delta q_i = q^{\\text{broker}}_i - q^{\\text{internal}}_i

    \\Delta \\text{cost}_i = c^{\\text{broker}}_i - c^{\\text{internal}}_i
    \\quad \\text{(only reported when } \\Delta q_i = 0\\text{)}

A ticker is a *drift* iff :math:`\\Delta q_i \\neq 0` or
:math:`|\\Delta \\text{cost}_i| > \\epsilon_{\\text{cost}}`. Cash drift is
:math:`|\\text{cash}^{\\text{broker}} - \\text{cash}^{\\text{internal}}| > \\epsilon_{\\$}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ReconciliationError(RuntimeError):
    """Raised when ``assert_reconciled`` sees any drift above tolerance.

    Subclass of ``RuntimeError`` — reconciliation failure is an
    operational invariant violation, not a user input error.
    """


# ---------------------------------------------------------------------------
# Input: broker snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """A single line of the broker's ground-truth position book."""

    ticker: str
    quantity: int  # signed; short = negative
    avg_cost: float  # broker-reported cost basis (may be NaN if unknown)


@dataclass(frozen=True, slots=True)
class BrokerSnapshot:
    """Ground-truth account state pulled from the broker.

    ``as_of`` is an ISO-8601 timestamp so reconciliation can be
    replayed from the ledger if needed.
    """

    as_of: str
    cash: float
    positions: tuple[BrokerPosition, ...]

    def quantities(self) -> dict[str, int]:
        return {p.ticker: int(p.quantity) for p in self.positions}

    def avg_costs(self) -> dict[str, float]:
        return {p.ticker: float(p.avg_cost) for p in self.positions}


# ---------------------------------------------------------------------------
# Output: drift report
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionDrift:
    """Per-ticker discrepancy row.

    ``kind`` is one of ``QUANTITY`` | ``COST`` | ``UNKNOWN_TO_BROKER``
    | ``UNKNOWN_TO_INTERNAL``.
    """

    ticker: str
    kind: str
    internal_qty: int
    broker_qty: int
    internal_avg_cost: float
    broker_avg_cost: float


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Summary of drift. ``ok == True`` iff no drifts exceed tolerance."""

    as_of: str
    cash_internal: float
    cash_broker: float
    cash_drift_usd: float
    position_drifts: tuple[PositionDrift, ...]
    ignored_tickers: tuple[str, ...] = field(default_factory=tuple)
    tol_cash_usd: float = 1.0
    tol_cost_usd: float = 0.01

    @property
    def ok(self) -> bool:
        return abs(self.cash_drift_usd) <= self.tol_cash_usd and len(self.position_drifts) == 0

    def summary(self) -> str:
        if self.ok:
            return f"[reconcile {self.as_of}] OK — no drift."
        pieces = [
            f"[reconcile {self.as_of}] DRIFT DETECTED",
            f"  cash: internal={self.cash_internal:,.2f} "
            f"broker={self.cash_broker:,.2f} "
            f"Δ={self.cash_drift_usd:+,.2f}",
        ]
        for d in self.position_drifts:
            pieces.append(
                f"  {d.ticker}: kind={d.kind} "
                f"internal_qty={d.internal_qty} broker_qty={d.broker_qty} "
                f"internal_cost={d.internal_avg_cost:.4f} "
                f"broker_cost={d.broker_avg_cost:.4f}"
            )
        return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------
def reconcile(
    internal: PortfolioState,
    broker: BrokerSnapshot,
    *,
    tol_cash_usd: float = 1.0,
    tol_cost_usd: float = 0.01,
    ignore_tickers: Iterable[str] = (),
) -> ReconcileReport:
    """Compare an internal ``PortfolioState`` against a broker snapshot.

    Tolerances
    ----------
    - ``tol_cash_usd``: absolute tolerance on cash ($).
    - ``tol_cost_usd``: absolute tolerance on avg-cost drift per share
      (only checked when quantity matches; otherwise avg-cost comparison
      is meaningless).

    The report lists every ticker whose drift exceeds these tolerances.
    Integer-share equality is exact (not toleranced) — you can't own
    half a share in US equities.
    """
    ignored = frozenset(ignore_tickers)

    int_qty = {t: p.quantity for t, p in internal.positions.items() if t not in ignored}
    int_cost = {t: p.avg_cost for t, p in internal.positions.items() if t not in ignored}
    brk_qty = {t: q for t, q in broker.quantities().items() if t not in ignored}
    brk_cost = {t: c for t, c in broker.avg_costs().items() if t not in ignored}

    all_tickers = set(int_qty) | set(brk_qty)
    drifts: list[PositionDrift] = []

    for t in sorted(all_tickers):
        iq = int_qty.get(t, 0)
        bq = brk_qty.get(t, 0)
        ic = int_cost.get(t, 0.0)
        bc = brk_cost.get(t, 0.0)

        if t not in int_qty:
            drifts.append(
                PositionDrift(
                    ticker=t,
                    kind="UNKNOWN_TO_INTERNAL",
                    internal_qty=0,
                    broker_qty=int(bq),
                    internal_avg_cost=0.0,
                    broker_avg_cost=float(bc),
                )
            )
            continue
        if t not in brk_qty:
            drifts.append(
                PositionDrift(
                    ticker=t,
                    kind="UNKNOWN_TO_BROKER",
                    internal_qty=int(iq),
                    broker_qty=0,
                    internal_avg_cost=float(ic),
                    broker_avg_cost=0.0,
                )
            )
            continue
        if iq != bq:
            drifts.append(
                PositionDrift(
                    ticker=t,
                    kind="QUANTITY",
                    internal_qty=int(iq),
                    broker_qty=int(bq),
                    internal_avg_cost=float(ic),
                    broker_avg_cost=float(bc),
                )
            )
            continue
        # Quantities match; compare avg cost (ignore NaN from broker).
        if bc == bc and abs(ic - bc) > tol_cost_usd:  # NaN-safe
            drifts.append(
                PositionDrift(
                    ticker=t,
                    kind="COST",
                    internal_qty=int(iq),
                    broker_qty=int(bq),
                    internal_avg_cost=float(ic),
                    broker_avg_cost=float(bc),
                )
            )

    cash_drift = float(broker.cash - internal.cash)

    return ReconcileReport(
        as_of=broker.as_of,
        cash_internal=float(internal.cash),
        cash_broker=float(broker.cash),
        cash_drift_usd=cash_drift,
        position_drifts=tuple(drifts),
        ignored_tickers=tuple(sorted(ignored)),
        tol_cash_usd=float(tol_cash_usd),
        tol_cost_usd=float(tol_cost_usd),
    )


def assert_reconciled(
    internal: PortfolioState,
    broker: BrokerSnapshot,
    ledger: Ledger | None = None,
    **kwargs,
) -> ReconcileReport:
    """Run reconcile; raise ReconciliationError if any drift exceeds tol.

    Regardless of success or failure, appends a ``RECONCILE`` event to
    the ledger (if supplied) so the audit chain captures every
    reconciliation attempt — not just the anomalies.
    """
    report = reconcile(internal, broker, **kwargs)
    if ledger is not None:
        ledger.append(
            broker.as_of,
            "RECONCILE",
            {
                "ok": bool(report.ok),
                "cash_drift_usd": float(report.cash_drift_usd),
                "n_drifts": len(report.position_drifts),
                "drifts": [
                    {
                        "ticker": d.ticker,
                        "kind": d.kind,
                        "internal_qty": d.internal_qty,
                        "broker_qty": d.broker_qty,
                    }
                    for d in report.position_drifts
                ],
            },
        )
    if not report.ok:
        raise ReconciliationError(report.summary())
    return report


__all__ = [
    "BrokerPosition",
    "BrokerSnapshot",
    "PositionDrift",
    "ReconcileReport",
    "ReconciliationError",
    "assert_reconciled",
    "reconcile",
]
