"""Corporate-actions reducer.

Why this exists
---------------
A multi-year replay of US cash equities that ignores corporate actions
silently produces the wrong book. The two highest-frequency offenders
are:

- **Forward splits** (e.g., NVDA 10:1 in 2024): position share count
  multiplies, average cost divides, notional and PnL unchanged.
- **Cash dividends**: cash credit equal to ``qty × per_share`` on
  ex-date (short positions pay out).

Less common but still material for a proper replay:

- **Reverse splits** (ratio < 1).
- **Spin-offs**: one position becomes two; cost basis allocated by
  distribution ratio.
- **Cash mergers / buyouts**: position retired for a fixed cash amount.
- **Stock-for-stock mergers**: ticker A → ticker B at a known ratio.

Phase-2 scope of this module: **StockSplit** + **CashDividend**. Spin-offs
and mergers are out of scope for the first pass (they require a
richer schema — ``target_ticker``, distribution fractions, etc.) and
will land in Phase 3 alongside the IBKR adapter.

Mathematical identities enforced
--------------------------------
For a position :math:`(q, c)` (signed shares, avg cost) and split ratio
:math:`r > 0`:

.. math::

    q' = q \\cdot r, \\quad c' = c / r, \\quad q' \\cdot c' = q \\cdot c

PnL impact is exactly zero by construction.

For a cash dividend of :math:`d` per share on position :math:`q`:

.. math::

    \\Delta \\text{cash} = q \\cdot d

(short positions pay, longs receive — same sign convention as the share
count). Dividend income is deliberately NOT rolled into
``realized_pnl`` so downstream tax/attribution bookkeeping can keep
capital gains and income apart.

Integration
-----------
``CorpActionHandler.apply`` is the single entrypoint: it mutates nothing,
returns a new ``PortfolioState``, and appends a structured
``CORP_ACTION`` event to a ``Ledger``. The hash-chain journal
(``quantengine.audit.journal``) automatically includes this event in
the chain — any tampering is detectable.

Use at session boundaries:

.. code-block:: python

    for action in corp_action_feed.actions_for(session_date):
        state = handler.apply(state, action, ledger, ts=session_date_iso)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from quantengine.portfolio.ledger import Ledger
from quantengine.portfolio.state import PortfolioState


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StockSplit:
    """Forward (r>1) or reverse (0<r<1) stock split."""

    ticker: str
    ratio: float  # new / old; 2.0 = 2-for-1 forward split
    ex_date: str  # ISO-8601 date string

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError(f"StockSplit.ratio must be > 0, got {self.ratio}")


@dataclass(frozen=True, slots=True)
class CashDividend:
    """Ordinary cash dividend on ex-date."""

    ticker: str
    per_share: float  # USD per share; >= 0
    ex_date: str

    def __post_init__(self) -> None:
        if self.per_share < 0:
            raise ValueError(f"CashDividend.per_share must be >= 0, got {self.per_share}")


CorpAction = Union[StockSplit, CashDividend]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
@dataclass
class CorpActionHandler:
    """Pure reducer: (state, action, ledger) → new state + ledger event.

    The handler is deliberately stateless. Callers drive it with actions
    pulled from an external feed (FMP's ``stock_split_calendar`` /
    ``stock_dividend_calendar``, IBKR's corporate-actions stream, etc.).
    """

    def apply(
        self,
        state: PortfolioState,
        action: CorpAction,
        ledger: Ledger,
        *,
        timestamp: str,
    ) -> PortfolioState:
        """Apply a single corporate action. Idempotent *per action object*
        only if the caller does not pass the same action twice — the handler
        writes a new ledger event on every call."""
        if isinstance(action, StockSplit):
            new_state = state.apply_split(action.ticker, action.ratio)
            old_qty = state.quantity_of(action.ticker)
            new_qty = new_state.quantity_of(action.ticker)
            ledger.append(
                timestamp,
                "CORP_ACTION",
                {
                    "type": "StockSplit",
                    "ticker": action.ticker,
                    "ratio": float(action.ratio),
                    "ex_date": action.ex_date,
                    "old_qty": int(old_qty),
                    "new_qty": int(new_qty),
                },
            )
            return new_state

        if isinstance(action, CashDividend):
            new_state = state.apply_cash_dividend(action.ticker, action.per_share)
            qty = state.quantity_of(action.ticker)
            cash_delta = qty * action.per_share
            ledger.append(
                timestamp,
                "CORP_ACTION",
                {
                    "type": "CashDividend",
                    "ticker": action.ticker,
                    "per_share": float(action.per_share),
                    "ex_date": action.ex_date,
                    "qty": int(qty),
                    "cash_delta": float(cash_delta),
                },
            )
            return new_state

        raise TypeError(
            f"Unsupported corporate action: {type(action).__name__}. "
            "Phase-2 supports StockSplit and CashDividend only."
        )


__all__ = [
    "CashDividend",
    "CorpAction",
    "CorpActionHandler",
    "StockSplit",
]
