"""Cost models — slippage and commission.

Keep these dead simple in Phase 1. Calibrated cost models (Almgren-Chriss,
Grinold-Kahn implementation shortfall, etc.) belong in `quantcore.cost` as
research artifacts; quantengine only applies a frozen callable.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from quantengine.contracts.orders import Order


class CostRealismWarning(UserWarning):
    """Emitted when a backtest is priced with optimistically low execution costs.

    A distinct category so callers can silence or escalate it (``warnings.filterwarnings``)
    independently of other warnings.
    """


class CostModel(ABC):
    @abstractmethod
    def fill_price(self, order: Order, reference_price: float) -> float:
        """Effective per-share fill price including slippage."""

    @abstractmethod
    def commission(self, order: Order, fill_price: float) -> float:
        """Absolute per-order commission in dollars, always >= 0."""


@dataclass(frozen=True, slots=True)
class LinearCostModel(CostModel):
    """Constant per-share slippage + per-share commission with a minimum.

    slippage_bps applied as a signed half-spread against the mid:
        buy_price  = ref * (1 + slippage_bps/1e4)
        sell_price = ref * (1 - slippage_bps/1e4)

    commission = max(commission_per_share * qty, commission_min).
    """

    slippage_bps: float = 1.0  # 1 bp default
    commission_per_share: float = 0.005  # IBKR tiered-lite default
    commission_min: float = 1.00  # $1 minimum per order

    def fill_price(self, order: Order, reference_price: float) -> float:
        bump = reference_price * self.slippage_bps / 1e4
        return reference_price + bump if order.side.value == "BUY" else reference_price - bump

    def commission(self, order: Order, fill_price: float) -> float:
        return max(self.commission_per_share * order.quantity, self.commission_min)

    @classmethod
    def from_lab_surface(
        cls,
        surface_path: str | Path,
        *,
        discipline: str = "at_open",
        commission_per_share: float = 0.005,
        commission_min: float = 1.00,
    ) -> "LinearCostModel":
        """Calibrate slippage to the MEASURED standing cost surface (s86/s91).

        ``surface_path`` is the lab's ``cost_model.json``; ``slippage_bps`` becomes the
        surface's one-way TOTAL (half-spread + modeled impact at book size) for the named
        execution ``discipline``:

        - ``"at_open"``           -> ``at_open_total_one_way_median`` (the live book's
                                     current discipline; D-A5 keeps it at the open)
        - ``"intraday_typical"``  -> ``intraday_typical_total_one_way_median``

        This replaces the uncalibrated 1bp default everywhere a backtest or paper broker
        should price like the measured market (the s91 audit's cost-surface-disconnect
        findings). Commission stays the explicit broker fee — the surface's one-way total
        is price-impact only, so there is no double count.
        """
        surface = json.loads(Path(surface_path).read_text())["standing_cost_for_gates_bps"]
        key = {
            "at_open": "at_open_total_one_way_median",
            "intraday_typical": "intraday_typical_total_one_way_median",
        }.get(discipline)
        if key is None:
            raise ValueError(f"unknown execution discipline {discipline!r}")
        return cls(
            slippage_bps=float(surface[key]),
            commission_per_share=commission_per_share,
            commission_min=commission_min,
        )

    @classmethod
    def realistic(
        cls,
        *,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        commission_min: float = 1.00,
    ) -> "LinearCostModel":
        """A conservative, lab-free cost preset for honest backtests.

        ``slippage_bps`` defaults to 5 bp one-way — a deliberately conservative HEURISTIC for
        liquid US large-cap equities (half-spread + light impact), chosen to sit safely above the
        optimistic 1 bp default rather than being measured. For a real deployment verdict prefer
        :meth:`from_lab_surface` (the measured s86/s91 standing-cost surface); ``realistic`` is the
        honest fallback when that surface isn't on hand.
        """
        return cls(
            slippage_bps=slippage_bps,
            commission_per_share=commission_per_share,
            commission_min=commission_min,
        )

    def assumptions(self) -> dict[str, float]:
        """The cost parameters as a dict — for surfacing in a backtest report / handoff record."""
        return {
            "slippage_bps": self.slippage_bps,
            "commission_per_share": self.commission_per_share,
            "commission_min": self.commission_min,
        }

    def is_optimistic(self, *, slippage_floor_bps: float = 2.0) -> bool:
        """True when one-way slippage is implausibly low for real US-equity execution.

        Below ``slippage_floor_bps`` (default 2 bp) the model under-charges for the half-spread
        and impact a live fill actually pays, so a backtest using it overstates net performance.
        The bare 1 bp default trips this; :meth:`realistic` and a typical :meth:`from_lab_surface`
        calibration do not.
        """
        return self.slippage_bps < slippage_floor_bps
