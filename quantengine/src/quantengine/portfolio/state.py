"""PortfolioState — the execution book of record.

Single reducer: `apply(fill)`. Replay, paper, and live all use this function,
so identical inputs produce bit-identical state. No floats are truncated; we
use full float64 accounting and rely on integer share counts for positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from quantengine.contracts.orders import Fill


@dataclass(frozen=True, slots=True)
class Position:
    ticker: str
    quantity: int  # signed; negative for short
    avg_cost: float  # cost basis per share (weighted); undefined if qty == 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (price - self.avg_cost) * self.quantity


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Immutable book snapshot. Mutate by calling `apply(fill)` → new state."""

    cash: float
    positions: Mapping[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_commission: float = 0.0

    # ---------- metrics ---------------------------------------------------
    def gross_exposure(self, prices: Mapping[str, float]) -> float:
        return sum(abs(p.quantity) * prices[p.ticker] for p in self.positions.values())

    def net_exposure(self, prices: Mapping[str, float]) -> float:
        return sum(p.quantity * prices[p.ticker] for p in self.positions.values())

    def nav(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.net_exposure(prices)

    def quantity_of(self, ticker: str) -> int:
        pos = self.positions.get(ticker)
        return pos.quantity if pos is not None else 0

    # ---------- reducer ---------------------------------------------------
    def apply(self, fill: Fill) -> "PortfolioState":
        """Return a new state after applying a fill. Pure, no side effects."""
        new_cash = self.cash + fill.cash_delta
        new_total_comm = self.total_commission + fill.commission

        old_pos = self.positions.get(fill.ticker, Position(fill.ticker, 0, 0.0))
        new_qty = old_pos.quantity + fill.signed_quantity

        realized_delta = 0.0
        if old_pos.quantity == 0 or (old_pos.quantity > 0) == (fill.signed_quantity > 0):
            # Opening or increasing same-sign: weighted-average cost.
            if new_qty == 0:
                new_avg = 0.0
            else:
                total_cost = old_pos.avg_cost * old_pos.quantity + fill.price * fill.signed_quantity
                new_avg = total_cost / new_qty
        else:
            # Reducing or flipping: realize PnL on the offset portion.
            closing_qty = min(abs(old_pos.quantity), abs(fill.signed_quantity))
            sign = 1 if old_pos.quantity > 0 else -1
            realized_delta = (fill.price - old_pos.avg_cost) * closing_qty * sign
            if abs(fill.signed_quantity) <= abs(old_pos.quantity):
                # Partial close — keep old avg cost
                new_avg = old_pos.avg_cost if new_qty != 0 else 0.0
            else:
                # Flipped through zero — new avg cost is fill price
                new_avg = fill.price if new_qty != 0 else 0.0

        new_positions = dict(self.positions)
        if new_qty == 0:
            new_positions.pop(fill.ticker, None)
        else:
            new_positions[fill.ticker] = Position(fill.ticker, new_qty, new_avg)

        return PortfolioState(
            cash=new_cash,
            positions=new_positions,
            realized_pnl=self.realized_pnl + realized_delta,
            total_commission=new_total_comm,
        )

    # ---------- corporate-action reducer ---------------------------------
    def apply_split(self, ticker: str, ratio: float) -> "PortfolioState":
        """Apply a stock split of the given ratio (``new / old``).

        A 2-for-1 split is ``ratio == 2.0``: share count doubles, average
        cost halves, cash is untouched. Reverse splits use ``ratio < 1``
        (e.g., 1-for-10 → ``ratio = 0.1``).

        Mathematical identity: notional = qty × avg_cost is invariant
        across the split, so there is no PnL impact.

        Returns self unchanged if the ticker is not held or qty is 0.
        """
        if ratio <= 0:
            raise ValueError(f"Split ratio must be > 0, got {ratio}")
        pos = self.positions.get(ticker)
        if pos is None or pos.quantity == 0:
            return self
        # Integer-share assumption: a split that produces a non-integer
        # share count (unusual on the NYSE — usually handled by cash in
        # lieu) is a hard error. Caller must decide: round, or emit a
        # supplemental cash adjustment.
        new_qty_float = pos.quantity * ratio
        if not float(new_qty_float).is_integer():
            raise ValueError(
                f"Split ratio {ratio} on position qty={pos.quantity} "
                f"produces non-integer share count {new_qty_float}. "
                "Handle cash-in-lieu separately before calling apply_split."
            )
        new_qty = int(new_qty_float)
        new_avg = pos.avg_cost / ratio  # notional preserved
        new_positions = dict(self.positions)
        new_positions[ticker] = Position(ticker, new_qty, new_avg)
        return PortfolioState(
            cash=self.cash,
            positions=new_positions,
            realized_pnl=self.realized_pnl,
            total_commission=self.total_commission,
        )

    def apply_cash_dividend(self, ticker: str, per_share: float) -> "PortfolioState":
        """Apply a cash dividend of ``per_share`` USD on the given position.

        Dividend cash is added to ``cash``. We deliberately do NOT move
        this into ``realized_pnl`` — dividends are income, not capital
        gains. Downstream tax/PnL reports should keep the two buckets
        separate; call ``nav(prices)`` to see total book value either
        way.

        If the position is short, the dividend is *paid out* (cash
        decreases by ``|qty| * per_share``) — standard short-sale rule.

        Returns self unchanged if the ticker is not held.
        """
        if per_share < 0:
            # Special dividends can in theory be negative (rare); we
            # disallow here to catch sign-flip bugs. If you need one,
            # pass a positive magnitude and encode the direction
            # upstream.
            raise ValueError(f"per_share must be >= 0, got {per_share}")
        pos = self.positions.get(ticker)
        if pos is None or pos.quantity == 0 or per_share == 0:
            return self
        delta = pos.quantity * per_share  # signed: short qty → cash out
        return PortfolioState(
            cash=self.cash + delta,
            positions=self.positions,
            realized_pnl=self.realized_pnl,
            total_commission=self.total_commission,
        )

    # ---------- factory ---------------------------------------------------
    @classmethod
    def empty(cls, initial_cash: float) -> "PortfolioState":
        return cls(cash=initial_cash, positions={})

    def with_cash(self, cash: float) -> "PortfolioState":
        return replace(self, cash=cash)
