"""Typed ``Bar`` shape for the streaming layer (S34 §3.AC1).

Pure frozen+slotted data. No methods, no validation. Per S33 §5.D1,
ABCs are for machines with behaviour; data classes are shape only.

``BarKind`` encodes the threshold *unit* (S34 §5.D6); sampling
algorithm (plain / imbalance / runs) is encoded by the builder class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from quantcore.data.events import BaseEvent


class BarKind(IntEnum):
    """Threshold unit. Sampling algorithm is encoded by the builder class."""

    TICK = 1
    VOLUME = 2
    DOLLAR = 3


@dataclass(frozen=True, slots=True)
class Bar(BaseEvent):
    ts_open: int
    kind: BarKind
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    tick_count: int
    dollar_volume: float
    spread_mean: float = float("nan")
    spread_last: float = float("nan")
    spread_std: float = float("nan")
    imbalance_mean: float = float("nan")
    imbalance_last: float = float("nan")
    imbalance_std: float = float("nan")
    microprice_dev_mean: float = float("nan")
    microprice_dev_last: float = float("nan")
    microprice_dev_std: float = float("nan")
    signed_volume_sum: float = float("nan")
    signed_dollar_sum: float = float("nan")
    signed_tick_imbalance: float = float("nan")
