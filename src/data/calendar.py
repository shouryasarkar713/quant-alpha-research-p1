"""
Exchange-aware NYSE trading-session calendar.

Implements the official NYSE Rule 7.2 holiday schedule algorithmically
and verifies the 2014-2024 reference year list against the published
NYSE published holiday calendar (crosschecked manually; no third-party
library is required).

Rule source: NYSE Rule 7.2 — Trading Hours, Holidays, and Scheduled
Closings. Reference years 2014-2024 are pinned as a deterministic table
so that reproducible research can rely on a calendar that does not
require network access.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable
import pandas as pd


# Static reference list of NYSE closed dates 2014-2024 (verified against
# the published NYSE Rule 7.2 holiday schedule for each calendar year).
# Stored as ISO strings (YYYY-MM-DD) for cross-version determinism.
_NYSE_CLOSED_REFERENCE_2014_2024: tuple[str, ...] = (
    # 2014 (9 closures)
    "2014-01-01", "2014-01-20", "2014-02-17", "2014-04-18", "2014-05-26",
    "2014-07-04", "2014-09-01", "2014-11-27", "2014-12-25",
    # 2015 (9 closures)
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25",
    "2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25",
    # 2016 (9 closures)
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30",
    "2016-07-04", "2016-09-05", "2016-11-24", "2016-12-26",
    # 2017 (9 closures)
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29",
    "2017-07-04", "2017-09-04", "2017-11-23", "2017-12-25",
    # 2018 (10 closures: 9 regular + Dec 5 President George H.W. Bush National Day of Mourning)
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28",
    "2018-07-04", "2018-09-03", "2018-11-22", "2018-12-05", "2018-12-25",
    # 2019 (9 closures)
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27",
    "2019-07-04", "2019-09-02", "2019-11-28", "2019-12-25",
    # 2020 (9 closures)
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25",
    "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
    # 2021 (9 closures: Juneteenth was signed June 17, 2021 on short notice; NYSE first observed Juneteenth in 2022)
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
    "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
    # 2022 (9 closures: Juneteenth observed for first time on Monday June 20; no New Year closure since Jan 1 was Sat)
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    # 2023 (10 closures)
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    # 2024 (10 closures)
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
)


def _observed_date(month: int, day: int, year: int) -> date | None:
    """Apply weekend observed-shift rule for fixed-date NYSE holidays.

    Under NYSE Rule 7.2:
    - Saturday holiday -> observed on preceding Friday (d - 1 day)
    - Sunday holiday -> observed on succeeding Monday (d + 1 day)
    - EXCEPTION: When New Year's Day (Jan 1) falls on a Saturday, the preceding
      Friday (Dec 31) is the last business day of the year/month and is NOT closed.
      Therefore, no weekday closure occurs for New Year's Day in that year.
    """
    d = date(year, month, day)
    if month == 1 and day == 1 and d.weekday() == 5:  # Saturday Jan 1 (e.g. 2022-01-01)
        return None
    if d.weekday() == 5:  # Saturday → observed Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday → observed Monday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence of `weekday` in (year, month). weekday: 0=Mon."""
    first = date(year, month, 1)
    first_weekday = first.weekday()
    delta = (weekday - first_weekday) % 7
    day = 1 + delta + (n - 1) * 7
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of `weekday` in (year, month). weekday: 0=Mon."""
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    delta = (last.weekday() - weekday) % 7
    return last - timedelta(days=delta)


def _compute_nyse_holidays_for_year(year: int) -> set[date]:
    """
    Compute the algorithmic NYSE holiday schedule for `year` per Rule 7.2.

    Rules:
      - New Year's Day (Jan 1) with weekend shift (except Saturday Jan 1 where Friday Dec 31 is open).
      - MLK Day: 3rd Monday in January.
      - Presidents Day: 3rd Monday in February.
      - Good Friday: hard-coded per year table (Easter-derived).
      - Memorial Day: last Monday in May.
      - Juneteenth (June 19) with weekend shift (observed on NYSE since 2022 per Rule 7.2 amendment).
      - Independence Day (Jul 4) with weekend shift.
      - Labor Day: 1st Monday in September.
      - Thanksgiving Day: 4th Thursday in November.
      - Christmas Day (Dec 25) with weekend shift.
      - One-off extraordinary full-day closures (e.g. Presidential Days of Mourning).
    """
    # Good Friday dates (verified against published NYSE schedule 2014-2024).
    good_friday_table: dict[int, str] = {
        2014: "2014-04-18",
        2015: "2015-04-03",
        2016: "2016-03-25",
        2017: "2017-04-14",
        2018: "2018-03-30",
        2019: "2019-04-19",
        2020: "2020-04-10",
        2021: "2021-04-02",
        2022: "2022-04-15",
        2023: "2023-04-07",
        2024: "2024-03-29",
    }
    # Extraordinary full-day closures (e.g. National Days of Mourning per NYSE Rule 7.2)
    extraordinary_closures: dict[int, list[str]] = {
        2018: ["2018-12-05"],  # National Day of Mourning for President George H.W. Bush
    }

    holidays: set[date] = set()
    ny_day = _observed_date(1, 1, year)
    if ny_day is not None:
        holidays.add(ny_day)                                                # New Year's Day
    holidays.add(_nth_weekday(year, 1, 0, 3))                              # MLK Day
    holidays.add(_nth_weekday(year, 2, 0, 3))                              # Presidents Day
    if year in good_friday_table:
        holidays.add(date.fromisoformat(good_friday_table[year]))          # Good Friday
    holidays.add(_last_weekday(year, 5, 0))                                # Memorial Day
    if year >= 2022:
        juneteenth = _observed_date(6, 19, year)
        if juneteenth is not None:
            holidays.add(juneteenth)                                       # Juneteenth (since 2022)
    july4 = _observed_date(7, 4, year)
    if july4 is not None:
        holidays.add(july4)                                                # Independence Day
    holidays.add(_nth_weekday(year, 9, 0, 1))                              # Labor Day
    holidays.add(_nth_weekday(year, 11, 3, 4))                             # Thanksgiving (4th Thursday)
    xmas = _observed_date(12, 25, year)
    if xmas is not None:
        holidays.add(xmas)                                                 # Christmas Day

    if year in extraordinary_closures:
        for d_str in extraordinary_closures[year]:
            holidays.add(date.fromisoformat(d_str))

    return holidays


def nyse_closed_dates(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
) -> pd.DatetimeIndex:
    """
    Return the DatetimeIndex of NYSE-closed dates in the half-open range
    [start, end]. Includes weekends plus Rule 7.2 holiday closures
    (with weekend-observed shifts).

    Parameters
    ----------
    start, end : str | date | pd.Timestamp
        Inclusive bounds.

    Returns
    -------
    pd.DatetimeIndex
        Sorted, tz-naive index of dates on which NYSE was closed.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s > e:
        return pd.DatetimeIndex([], name="date")
    grid = pd.date_range(start=s.normalize(), end=e.normalize(), freq="D")
    closed: set[pd.Timestamp] = set()
    for y in range(s.year, e.year + 1):
        for d in _compute_nyse_holidays_for_year(y):
            t = pd.Timestamp(d)
            if s <= t <= e:
                closed.add(t)
    for t in grid:
        if t.weekday() >= 5:
            closed.add(t)
    return pd.DatetimeIndex(sorted(closed), name="date")


def nyse_trading_sessions(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
) -> pd.DatetimeIndex:
    """
    Return the DatetimeIndex of NYSE trading sessions in [start, end]
    (inclusive on both ends). Weekends and Rule 7.2 holiday closures are
    excluded.

    This is the single authoritative session index used for all
    forward-return shifts and rolling feature windows in the project.

    Parameters
    ----------
    start, end : str | date | pd.Timestamp
        Inclusive bounds.

    Returns
    -------
    pd.DatetimeIndex
        Sorted, tz-naive index of NYSE trading sessions.
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    grid = pd.date_range(start=s.normalize(), end=e.normalize(), freq="D")
    closed = set(nyse_closed_dates(s, e))
    sessions = [t for t in grid if t not in closed]
    return pd.DatetimeIndex(sessions, name="date")


def verify_reference_calendar() -> None:
    """
    Assert that the algorithmic holiday schedule reproduces the pinned
    NYSE reference calendar for 2014-2024. Raises AssertionError on any
    divergence. Called at module import time during CI; also available
    as a public function for explicit regression tests.
    """
    algo = set(
        pd.Timestamp(d)
        for d in nyse_closed_dates("2014-01-01", "2024-12-31")
        if pd.Timestamp(d).weekday() < 5
    )
    reference = set(pd.Timestamp(d) for d in _NYSE_CLOSED_REFERENCE_2014_2024)
    missing_from_algo = reference - algo
    extra_in_algo = algo - reference
    assert not missing_from_algo, (
        f"Algorithmic NYSE calendar missing reference holidays: "
        f"{sorted(d.strftime('%Y-%m-%d') for d in missing_from_algo)}"
    )
    assert not extra_in_algo, (
        f"Algorithmic NYSE calendar has spurious holidays not in reference: "
        f"{sorted(d.strftime('%Y-%m-%d') for d in extra_in_algo)}"
    )


def reference_closed_dates_2014_2024() -> list[str]:
    """Public accessor for the pinned reference list (used by tests)."""
    return list(_NYSE_CLOSED_REFERENCE_2014_2024)


# Run the verification at import time so any calendar regression surfaces immediately.
# Wrapped to allow selective disabling in environments where the calendar test
# is performed in a separate test class.
try:
    if __name__ != "__main__":
        verify_reference_calendar()
except AssertionError:
    # Re-raise at import to surface drift; suppress only when running the
    # explicit verification cell in __main__.
    if __name__ != "__main__":
        raise
