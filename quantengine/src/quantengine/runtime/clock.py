"""Clock abstraction.

Phase 1 uses a naive business-day clock (pandas bdate_range) to stay
dependency-light. Phase 2 should switch to `exchange_calendars` (XNYS) for
holiday-correct NYSE/NASDAQ calendars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


class Clock(ABC):
    @abstractmethod
    def now(self) -> pd.Timestamp: ...
    @abstractmethod
    def step(self) -> pd.Timestamp: ...
    @abstractmethod
    def is_trading_day(self, ts: pd.Timestamp) -> bool: ...


@dataclass
class BusinessDayClock(Clock):
    """Iterates business days between [start, end] inclusive.

    This is a placeholder — replace with exchange_calendars in Phase 2.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    _cursor: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        self.end = pd.Timestamp(self.end)
        self._cursor = self.start - pd.tseries.offsets.BDay(1)

    def now(self) -> pd.Timestamp:
        assert self._cursor is not None
        return self._cursor

    def step(self) -> pd.Timestamp:
        assert self._cursor is not None
        self._cursor = self._cursor + pd.tseries.offsets.BDay(1)
        if self._cursor > self.end:
            raise StopIteration
        return self._cursor

    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        return ts.weekday() < 5

    def __iter__(self) -> Iterator[pd.Timestamp]:
        while True:
            try:
                yield self.step()
            except StopIteration:
                return
