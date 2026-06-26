"""Execution-side constraints for RebalanceEngine.

All constraints here are *operational*, not research. Kelly scaling and
leverage limits that reflect the model's *preferred* risk live in quantcore;
what lives here is: integer shares, cash buffer, turnover caps, minimum trade
sizes, short-sale permissions, and the no-trade-mask policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NoTradePolicy(str, Enum):
    """What to do when tradeable_i = 0 but an existing position h_i != 0."""

    HOLD = "HOLD"  # default: leave the position untouched
    FLATTEN = "FLATTEN"  # close it on this rebalance
    DECAY = "DECAY"  # reduce by `decay_fraction` per rebalance


@dataclass(frozen=True, slots=True)
class RebalanceConstraints:
    """All the knobs RebalanceEngine respects.

    Attributes
    ----------
    cash_buffer           : fraction of NAV to retain as cash (β in docs).
    max_gross_leverage    : cap on Σ|w_i|. quantcore should also respect this;
                            we enforce here as a last-line defense.
    max_turnover          : cap on Σ|Δnotional|/NAV per rebalance.
    min_trade_notional    : any trade below this $ is zeroed out.
    allow_short           : if False, negative target weights are clipped to 0.
    lot_size              : whole-share = 1 for US equities. Fractional = 0.
    no_trade_policy       : see `NoTradePolicy`.
    decay_fraction        : used only when `no_trade_policy == DECAY`.
    max_position_weight   : per-name cap |w_i| <= this. 0.0 disables.
    """

    cash_buffer: float = 0.02
    max_gross_leverage: float = 1.0
    max_turnover: float = 1.0  # 100% of NAV by default
    min_trade_notional: float = 100.0  # $100 floor; brokers often need more
    allow_short: bool = False
    lot_size: int = 1
    no_trade_policy: NoTradePolicy = NoTradePolicy.HOLD
    decay_fraction: float = 0.25
    max_position_weight: float = 0.0  # 0 = no per-name cap

    def __post_init__(self) -> None:
        if not (0.0 <= self.cash_buffer < 1.0):
            raise ValueError(f"cash_buffer must be in [0,1), got {self.cash_buffer}")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be > 0")
        if self.max_turnover <= 0:
            raise ValueError("max_turnover must be > 0")
        if self.min_trade_notional < 0:
            raise ValueError("min_trade_notional must be >= 0")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1 (fractional shares not supported)")
        if not (0.0 < self.decay_fraction <= 1.0):
            raise ValueError("decay_fraction must be in (0,1]")
        if self.max_position_weight < 0:
            raise ValueError("max_position_weight must be >= 0")
