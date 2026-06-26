"""NYSE / NASDAQ trading calendar.

Two backends
------------
1. **Preferred**: delegates to ``exchange_calendars.get_calendar("XNYS")``
   when the package is installed. This is the canonical, long-tail
   accurate calendar (back to 1885 and forward as holidays are
   announced).

2. **Fallback**: a self-contained hand-rolled rule engine covering
   2015-2035. Sufficient for replay/paper-trading on contemporary
   data. Covers fixed-date, nth-weekday-of-month, and Good Friday
   (computed via the Anonymous Gregorian Easter algorithm). Also
   encodes early-close rules (Black Friday 1pm close, Christmas Eve
   1pm close, Independence Day Eve 1pm close).

The fallback is *correct* for the covered range but not comprehensive:
unscheduled market closures (e.g., 9/11, Hurricane Sandy, presidential
funerals) are NOT in the fallback. If your replay spans any such day,
install ``exchange_calendars`` or extend ``_SPECIAL_CLOSURES``.

Semantics
---------
Every method returns tz-naive ``pd.Timestamp``s in US/Eastern wall-clock
interpretation (i.e., 9:30 AM means 9:30 AM ET, always — regardless of
whether DST is in effect). This matches how US-equity trading sessions
are specified operationally.

See also
--------
- ``quantengine.runtime.clock.Clock`` : base clock interface.
- ``quantengine.backtest.replay.HistoricalClock`` : replay-time driver.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

import pandas as pd

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised only when the optional dep is present
    import exchange_calendars as _xcals  # type: ignore

    _HAVE_XCALS = True
except ImportError:
    _xcals = None
    _HAVE_XCALS = False


# ---------------------------------------------------------------------------
# Session times (US Eastern, wall-clock)
# ---------------------------------------------------------------------------
REGULAR_OPEN = _dt.time(9, 30)
REGULAR_CLOSE = _dt.time(16, 0)
EARLY_CLOSE = _dt.time(13, 0)


# ---------------------------------------------------------------------------
# Holiday rules (fallback backend)
# ---------------------------------------------------------------------------
def _easter_sunday(year: int) -> _dt.date:
    """Anonymous Gregorian Easter algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def _good_friday(year: int) -> _dt.date:
    return _easter_sunday(year) - _dt.timedelta(days=2)


def _nth_weekday(year: int, month: int, n: int, weekday: int) -> _dt.date:
    """n-th occurrence of ``weekday`` (Mon=0 … Sun=6) in ``month``."""
    first = _dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + _dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    # Try day 31 down to 25 and pick the latest matching weekday.
    for day in range(31, 24, -1):
        try:
            d = _dt.date(year, month, day)
        except ValueError:
            continue
        if d.weekday() == weekday:
            return d
    raise RuntimeError("unreachable")


def _observed(d: _dt.date) -> _dt.date:
    """Weekend-to-weekday observed-holiday rule for fixed-date holidays.

    Saturday → preceding Friday; Sunday → following Monday. This is the
    NYSE convention.
    """
    if d.weekday() == 5:  # Saturday
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + _dt.timedelta(days=1)
    return d


# Hard-coded unscheduled closures (extend as needed).
_SPECIAL_CLOSURES: frozenset[_dt.date] = frozenset(
    {
        _dt.date(2001, 9, 11),
        _dt.date(2001, 9, 12),
        _dt.date(2001, 9, 13),
        _dt.date(2001, 9, 14),
        _dt.date(2012, 10, 29),
        _dt.date(2012, 10, 30),  # Hurricane Sandy
        _dt.date(2018, 12, 5),  # George H.W. Bush funeral
        _dt.date(2025, 1, 9),  # Jimmy Carter national day of mourning
    }
)


@lru_cache(maxsize=64)
def _holidays(year: int) -> frozenset[_dt.date]:
    """All NYSE full-day closures in ``year`` (scheduled + special)."""
    out = set()
    # Fixed-date (observed).
    out.add(_observed(_dt.date(year, 1, 1)))  # New Year
    out.add(_observed(_dt.date(year, 7, 4)))  # Independence
    out.add(_observed(_dt.date(year, 12, 25)))  # Christmas
    if year >= 2022:  # Juneteenth (federal 2021; NYSE observed from 2022)
        out.add(_observed(_dt.date(year, 6, 19)))
    # Nth-weekday rules.
    out.add(_nth_weekday(year, 1, 3, 0))  # MLK Day (3rd Mon Jan)
    out.add(_nth_weekday(year, 2, 3, 0))  # Presidents Day (3rd Mon Feb)
    out.add(_nth_weekday(year, 9, 1, 0))  # Labor Day (1st Mon Sep)
    out.add(_nth_weekday(year, 11, 4, 3))  # Thanksgiving (4th Thu Nov)
    out.add(_last_weekday(year, 5, 0))  # Memorial Day (last Mon May)
    # Good Friday.
    out.add(_good_friday(year))
    # Unscheduled closures.
    out |= {d for d in _SPECIAL_CLOSURES if d.year == year}
    return frozenset(out)


@lru_cache(maxsize=64)
def _early_closes(year: int) -> frozenset[_dt.date]:
    """Days with 1:00pm close (ET). Day-after-Thanksgiving, Christmas
    Eve (if a weekday and market open), Independence Day Eve (same)."""
    out: set[_dt.date] = set()
    thx = _nth_weekday(year, 11, 4, 3)
    out.add(thx + _dt.timedelta(days=1))  # Black Friday
    # July 3 early close rule: if July 4 falls on a Tue/Wed/Thu/Fri, July 3
    # is an early close day. If July 4 is Sat/Sun/Mon, no early close.
    jul4 = _dt.date(year, 7, 4)
    if jul4.weekday() in (1, 2, 3, 4):  # Tue-Fri
        out.add(_dt.date(year, 7, 3))
    # Christmas Eve: if Dec 24 is a weekday, it's an early close day
    # (NYSE practice; sometimes skipped when adjacent weekend creates a
    # full holiday). We adopt the conservative rule: Dec 24 is early
    # close iff it's Mon-Fri and not in the holiday set.
    dec24 = _dt.date(year, 12, 24)
    if dec24.weekday() < 5 and dec24 not in _holidays(year):
        out.add(dec24)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------
@dataclass
class TradingCalendar:
    """XNYS calendar with a clean, minimal API.

    Internally uses ``exchange_calendars`` if available, else the
    hand-rolled fallback.
    """

    _use_xcals: bool = field(default=_HAVE_XCALS, init=False)
    _xcal: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._use_xcals:
            self._xcal = _xcals.get_calendar("XNYS")

    # ---- predicates ----------------------------------------------------
    def is_session(self, ts: pd.Timestamp | _dt.date) -> bool:
        """True iff the calendar day is a trading session (not holiday/weekend).

        Early-close days ARE sessions (they're partial).
        """
        d = self._as_date(ts)
        if self._use_xcals:
            return bool(self._xcal.is_session(pd.Timestamp(d)))
        if d.weekday() >= 5:
            return False
        return d not in _holidays(d.year)

    def is_early_close(self, ts: pd.Timestamp | _dt.date) -> bool:
        d = self._as_date(ts)
        if self._use_xcals:
            # exchange_calendars exposes early_closes as a DatetimeIndex.
            return pd.Timestamp(d) in self._xcal.early_closes
        return d in _early_closes(d.year)

    # ---- session times -------------------------------------------------
    def session_open(self, ts: pd.Timestamp | _dt.date) -> pd.Timestamp:
        """Regular open = 09:30 ET on a session day."""
        d = self._as_date(ts)
        if not self.is_session(d):
            raise ValueError(f"{d} is not a trading session.")
        return pd.Timestamp.combine(d, REGULAR_OPEN)

    def session_close(self, ts: pd.Timestamp | _dt.date) -> pd.Timestamp:
        """Close = 16:00 on a regular session, 13:00 on an early-close day."""
        d = self._as_date(ts)
        if not self.is_session(d):
            raise ValueError(f"{d} is not a trading session.")
        close_time = EARLY_CLOSE if self.is_early_close(d) else REGULAR_CLOSE
        return pd.Timestamp.combine(d, close_time)

    # ---- navigation ----------------------------------------------------
    def sessions_in_range(
        self,
        start: pd.Timestamp | _dt.date,
        end: pd.Timestamp | _dt.date,
    ) -> pd.DatetimeIndex:
        """All trading sessions in [start, end] inclusive."""
        s = pd.Timestamp(self._as_date(start))
        e = pd.Timestamp(self._as_date(end))
        if self._use_xcals:
            return self._xcal.sessions_in_range(s, e)
        # Hand-rolled: iterate weekdays, filter holidays.
        days = pd.bdate_range(s, e)
        holidays_all: set[_dt.date] = set()
        for y in range(s.year, e.year + 1):
            holidays_all |= set(_holidays(y))
        return pd.DatetimeIndex([d for d in days if d.date() not in holidays_all])

    def next_session(self, ts: pd.Timestamp | _dt.date) -> pd.Timestamp:
        """First session strictly after ``ts``'s date."""
        d = self._as_date(ts) + _dt.timedelta(days=1)
        while not self.is_session(d):
            d += _dt.timedelta(days=1)
        return pd.Timestamp(d)

    def previous_session(self, ts: pd.Timestamp | _dt.date) -> pd.Timestamp:
        d = self._as_date(ts) - _dt.timedelta(days=1)
        while not self.is_session(d):
            d -= _dt.timedelta(days=1)
        return pd.Timestamp(d)

    # ---- MOC/LOO helpers ----------------------------------------------
    def moc_cutoff(self, ts: pd.Timestamp | _dt.date) -> pd.Timestamp:
        """Market-on-close order cutoff: 10 minutes before the close.

        NYSE MOC cutoff is 15:50 on regular days, 12:50 on early-close
        days. Orders must be received before this time to participate
        in the closing auction.
        """
        close = self.session_close(ts)
        return close - pd.Timedelta(minutes=10)

    # ---- internals -----------------------------------------------------
    @staticmethod
    def _as_date(ts: pd.Timestamp | _dt.date) -> _dt.date:
        if isinstance(ts, _dt.datetime):
            return ts.date()
        if isinstance(ts, _dt.date):
            return ts
        return pd.Timestamp(ts).date()


# ---------------------------------------------------------------------------
# XNYS-aware Clock implementation
# ---------------------------------------------------------------------------
from quantengine.runtime.clock import Clock  # noqa: E402  (late import avoids cycle)


@dataclass
class XNYSClock(Clock):
    """Session-aware clock: yields ``session_close`` per trading day.

    This matches the most common rebalance cadence (end-of-day decision,
    MOC fill). For an open-only or intraday rhythm, subclass and
    override ``step``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    calendar: TradingCalendar = field(default_factory=TradingCalendar)
    _sessions: pd.DatetimeIndex | None = field(default=None, init=False, repr=False)
    _i: int = field(default=-1, init=False, repr=False)
    _current: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        self.end = pd.Timestamp(self.end)
        self._sessions = self.calendar.sessions_in_range(self.start, self.end)

    def now(self) -> pd.Timestamp:
        if self._current is None:
            raise RuntimeError("Clock has not started. Call step() first.")
        return self._current

    def step(self) -> pd.Timestamp:
        self._i += 1
        if self._i >= len(self._sessions):
            raise StopIteration
        session_day = self._sessions[self._i]
        self._current = self.calendar.session_close(session_day)
        return self._current

    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        return self.calendar.is_session(ts)

    def __iter__(self) -> Iterator[pd.Timestamp]:
        while True:
            try:
                yield self.step()
            except StopIteration:
                return

    def __len__(self) -> int:
        return len(self._sessions) if self._sessions is not None else 0


__all__ = [
    "EARLY_CLOSE",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "TradingCalendar",
    "XNYSClock",
]
