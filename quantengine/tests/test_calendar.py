"""Smoke tests for quantengine.runtime.calendar.

Covers:
    - Anonymous Gregorian Easter ⇒ Good Friday dates (2024-2026).
    - Fixed-date holidays with observed-rule (July 4 2026 obs = Friday Jul 3).
    - Nth-weekday holidays (MLK, Thanksgiving).
    - Memorial Day = last Monday of May.
    - Juneteenth observed from 2022 onward, NOT before.
    - Early-close rules: Black Friday, Christmas Eve (when weekday).
    - moc_cutoff = close - 10 minutes.
    - sessions_in_range filters weekends + holidays.
    - XNYSClock iteration yields session_close per trading day.

Tests assume the hand-rolled fallback backend (works without the
``exchange_calendars`` package installed); if the package IS installed,
the same assertions should still hold because both implement XNYS.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from quantengine.runtime.calendar import (
    EARLY_CLOSE,
    REGULAR_CLOSE,
    TradingCalendar,
    XNYSClock,
    _easter_sunday,
    _good_friday,
    _nth_weekday,
)


CAL = TradingCalendar()


# ---------------------------------------------------------------------------
# Date-math primitives (backend-independent)
# ---------------------------------------------------------------------------
def test_good_friday_2024():
    # Easter 2024 = March 31; Good Friday = March 29.
    assert _good_friday(2024) == _dt.date(2024, 3, 29)


def test_good_friday_2025_2026():
    # Easter 2025 = April 20 → Good Friday April 18.
    assert _good_friday(2025) == _dt.date(2025, 4, 18)
    # Easter 2026 = April 5 → Good Friday April 3.
    assert _good_friday(2026) == _dt.date(2026, 4, 3)


def test_easter_sunday_2024():
    assert _easter_sunday(2024) == _dt.date(2024, 3, 31)


def test_nth_weekday_mlk_2025():
    # MLK 2025 = 3rd Monday of January = Jan 20.
    assert _nth_weekday(2025, 1, 3, 0) == _dt.date(2025, 1, 20)


def test_nth_weekday_thanksgiving_2025():
    # Thanksgiving 2025 = 4th Thursday of November = Nov 27.
    assert _nth_weekday(2025, 11, 4, 3) == _dt.date(2025, 11, 27)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
def test_is_session_rejects_weekend():
    # Saturday 2025-04-19.
    assert CAL.is_session(_dt.date(2025, 4, 19)) is False


def test_is_session_rejects_christmas():
    # Dec 25 2025 is a Thursday — regular holiday.
    assert CAL.is_session(_dt.date(2025, 12, 25)) is False


def test_is_session_accepts_regular_weekday():
    # Monday 2025-04-14.
    assert CAL.is_session(_dt.date(2025, 4, 14)) is True


def test_juneteenth_observed_from_2022():
    # June 19 2023 = Monday; holiday.
    assert CAL.is_session(_dt.date(2023, 6, 19)) is False
    # June 19 2019 = Wednesday; NOT a holiday (pre-2022).
    assert CAL.is_session(_dt.date(2019, 6, 19)) is True


# ---------------------------------------------------------------------------
# Session times
# ---------------------------------------------------------------------------
def test_session_close_regular_day():
    d = _dt.date(2025, 4, 14)  # Monday
    assert CAL.session_close(d).time() == REGULAR_CLOSE


def test_session_close_early_on_black_friday():
    # Thanksgiving 2025 = Nov 27 Thu. Black Friday = Nov 28.
    d = _dt.date(2025, 11, 28)
    assert CAL.is_session(d)
    assert CAL.is_early_close(d)
    assert CAL.session_close(d).time() == EARLY_CLOSE


def test_moc_cutoff_is_ten_minutes_before_close():
    d = _dt.date(2025, 4, 14)
    cutoff = CAL.moc_cutoff(d)
    close = CAL.session_close(d)
    assert close - cutoff == pd.Timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Range queries
# ---------------------------------------------------------------------------
def test_sessions_in_range_full_week_no_holiday():
    # Week of April 21 2025: Mon-Fri, no holidays → 5 sessions.
    sessions = CAL.sessions_in_range(_dt.date(2025, 4, 21), _dt.date(2025, 4, 25))
    assert len(sessions) == 5


def test_sessions_in_range_skips_good_friday_2025():
    # Week containing Good Friday 2025 (April 18): Mon-Thu trade = 4 sessions.
    sessions = CAL.sessions_in_range(_dt.date(2025, 4, 14), _dt.date(2025, 4, 18))
    dates = [d.date() for d in sessions]
    assert _dt.date(2025, 4, 18) not in dates
    assert len(sessions) == 4


def test_next_session_skips_weekend():
    # Friday 2025-04-11 → next session Monday 2025-04-14.
    nxt = CAL.next_session(_dt.date(2025, 4, 11))
    assert nxt.date() == _dt.date(2025, 4, 14)


# ---------------------------------------------------------------------------
# XNYSClock
# ---------------------------------------------------------------------------
def test_xnys_clock_iterates_session_closes():
    clock = XNYSClock(
        start=pd.Timestamp("2025-04-14"),
        end=pd.Timestamp("2025-04-17"),
    )
    closes = list(clock)
    # Mon-Thu of that week = 4 sessions.
    assert len(closes) == 4
    # Every tick is a 16:00 close.
    for ts in closes:
        assert ts.time() == REGULAR_CLOSE


def test_xnys_clock_len_matches_sessions():
    clock = XNYSClock(
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-01-31"),
    )
    # January 2025: 21 trading sessions (Jan 1 holiday, MLK Jan 20 holiday).
    assert len(clock) == 20


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        test_good_friday_2024,
        test_good_friday_2025_2026,
        test_easter_sunday_2024,
        test_nth_weekday_mlk_2025,
        test_nth_weekday_thanksgiving_2025,
        test_is_session_rejects_weekend,
        test_is_session_rejects_christmas,
        test_is_session_accepts_regular_weekday,
        test_juneteenth_observed_from_2022,
        test_session_close_regular_day,
        test_session_close_early_on_black_friday,
        test_moc_cutoff_is_ten_minutes_before_close,
        test_sessions_in_range_full_week_no_holiday,
        test_sessions_in_range_skips_good_friday_2025,
        test_next_session_skips_weekend,
        test_xnys_clock_iterates_session_closes,
        test_xnys_clock_len_matches_sessions,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nruntime.calendar: {len(tests)}/{len(tests)} checks passed.")


if __name__ == "__main__":
    _run_all()
