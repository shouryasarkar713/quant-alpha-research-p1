"""Unit tests for the exchange-aware NYSE calendar and trading session generation."""

from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from src.data.calendar import (
    _observed_date,
    nyse_closed_dates,
    nyse_trading_sessions,
    reference_closed_dates_2014_2024,
    verify_reference_calendar,
)


def test_saturday_fixed_date_holiday_observation():
    """Verify fixed-date holidays falling on Saturday are observed on Friday."""
    # July 4, 2015 (Saturday) -> Observed on Friday, July 3, 2015
    obs_july4_2015 = _observed_date(7, 4, 2015)
    assert obs_july4_2015 == date(2015, 7, 3)

    # July 4, 2020 (Saturday) -> Observed on Friday, July 3, 2020
    obs_july4_2020 = _observed_date(7, 4, 2020)
    assert obs_july4_2020 == date(2020, 7, 3)

    # Christmas Dec 25, 2021 (Saturday) -> Observed on Friday, Dec 24, 2021
    obs_xmas_2021 = _observed_date(12, 25, 2021)
    assert obs_xmas_2021 == date(2021, 12, 24)


def test_sunday_fixed_date_holiday_observation():
    """Verify fixed-date holidays falling on Sunday are observed on Monday."""
    # July 4, 2021 (Sunday) -> Observed on Monday, July 5, 2021
    obs_july4_2021 = _observed_date(7, 4, 2021)
    assert obs_july4_2021 == date(2021, 7, 5)

    # Christmas Dec 25, 2016 (Sunday) -> Observed on Monday, Dec 26, 2016
    obs_xmas_2016 = _observed_date(12, 25, 2016)
    assert obs_xmas_2016 == date(2016, 12, 26)

    # Christmas Dec 25, 2022 (Sunday) -> Observed on Monday, Dec 26, 2022
    obs_xmas_2022 = _observed_date(12, 25, 2022)
    assert obs_xmas_2022 == date(2022, 12, 26)

    # Juneteenth June 19, 2022 (Sunday) -> Observed on Monday, June 20, 2022
    obs_june19_2022 = _observed_date(6, 19, 2022)
    assert obs_june19_2022 == date(2022, 6, 20)


def test_jan_1_2022_edge_case():
    """
    Verify New Year's Day 2022 Saturday edge case.
    2022-01-01 was Saturday.
    Per NYSE Rule 7.2 exception: 2021-12-31 is the last business day of the year
    and remains OPEN. No holiday closure is created for Jan 1, 2022.
    """
    obs_ny_2022 = _observed_date(1, 1, 2022)
    assert obs_ny_2022 is None

    # Check that 2021-12-31 is an open trading session
    sessions_dec2021 = nyse_trading_sessions("2021-12-20", "2022-01-05")
    assert pd.Timestamp("2021-12-31") in sessions_dec2021

    # Check that 2021-12-31 is not in closed dates
    closed_dates = nyse_closed_dates("2021-12-20", "2022-01-05")
    assert pd.Timestamp("2021-12-31") not in closed_dates


def test_jan_1_2023_edge_case():
    """Verify New Year's Day 2023 (Sunday) is observed on Monday 2023-01-02."""
    obs_ny_2023 = _observed_date(1, 1, 2023)
    assert obs_ny_2023 == date(2023, 1, 2)

    sessions_jan2023 = nyse_trading_sessions("2023-01-01", "2023-01-10")
    assert pd.Timestamp("2023-01-02") not in sessions_jan2023
    assert pd.Timestamp("2023-01-03") in sessions_jan2023

    closed_dates = nyse_closed_dates("2023-01-01", "2023-01-10")
    assert pd.Timestamp("2023-01-02") in closed_dates


def test_jan_1_2017_edge_case():
    """Verify New Year's Day 2017 (Sunday) is observed on Monday 2017-01-02."""
    obs_ny_2017 = _observed_date(1, 1, 2017)
    assert obs_ny_2017 == date(2017, 1, 2)

    sessions_jan2017 = nyse_trading_sessions("2017-01-01", "2017-01-10")
    assert pd.Timestamp("2017-01-02") not in sessions_jan2017
    assert pd.Timestamp("2017-01-03") in sessions_jan2017

    closed_dates = nyse_closed_dates("2017-01-01", "2017-01-10")
    assert pd.Timestamp("2017-01-02") in closed_dates


def test_george_hw_bush_mourning_day_2018_12_05():
    """
    Verify full-day extraordinary closure on Wednesday, December 5, 2018
    for the National Day of Mourning for President George H.W. Bush.
    """
    closed_2018 = nyse_closed_dates("2018-01-01", "2018-12-31")
    assert pd.Timestamp("2018-12-05") in closed_2018

    sessions_2018 = nyse_trading_sessions("2018-01-01", "2018-12-31")
    assert pd.Timestamp("2018-12-05") not in sessions_2018


def test_juneteenth_transition_2021_vs_2022():
    """
    Verify Juneteenth holiday transition:
    - 2021: Signed into law June 17, 2021 on short notice; NYSE was OPEN on Friday June 18, 2021.
    - 2022: First year observed by NYSE; closed on Monday June 20, 2022 (June 19 fell on Sunday).
    """
    # 2021-06-18 was an open trading session
    sessions_2021 = nyse_trading_sessions("2021-06-14", "2021-06-25")
    assert pd.Timestamp("2021-06-18") in sessions_2021
    closed_2021 = nyse_closed_dates("2021-06-14", "2021-06-25")
    assert pd.Timestamp("2021-06-18") not in closed_2021

    # 2022-06-20 was closed
    sessions_2022 = nyse_trading_sessions("2022-06-13", "2022-06-24")
    assert pd.Timestamp("2022-06-20") not in sessions_2022
    closed_2022 = nyse_closed_dates("2022-06-13", "2022-06-24")
    assert pd.Timestamp("2022-06-20") in closed_2022


def test_christmas_saturday_sunday_boundaries():
    """Verify Christmas Day Saturday and Sunday observation boundaries."""
    # 2021: Dec 25 Saturday -> Dec 24 Friday closed
    closed_2021 = nyse_closed_dates("2021-12-20", "2021-12-31")
    assert pd.Timestamp("2021-12-24") in closed_2021
    sessions_2021 = nyse_trading_sessions("2021-12-20", "2021-12-31")
    assert pd.Timestamp("2021-12-24") not in sessions_2021

    # 2016: Dec 25 Sunday -> Dec 26 Monday closed
    closed_2016 = nyse_closed_dates("2016-12-20", "2016-12-31")
    assert pd.Timestamp("2016-12-26") in closed_2016
    sessions_2016 = nyse_trading_sessions("2016-12-20", "2016-12-31")
    assert pd.Timestamp("2016-12-26") not in sessions_2016

    # 2022: Dec 25 Sunday -> Dec 26 Monday closed
    closed_2022 = nyse_closed_dates("2022-12-20", "2022-12-31")
    assert pd.Timestamp("2022-12-26") in closed_2022
    sessions_2022 = nyse_trading_sessions("2022-12-20", "2022-12-31")
    assert pd.Timestamp("2022-12-26") not in sessions_2022


def test_july_4_saturday_sunday_boundaries():
    """Verify Independence Day Saturday and Sunday observation boundaries."""
    # 2015: July 4 Saturday -> July 3 Friday closed
    closed_2015 = nyse_closed_dates("2015-07-01", "2015-07-10")
    assert pd.Timestamp("2015-07-03") in closed_2015
    sessions_2015 = nyse_trading_sessions("2015-07-01", "2015-07-10")
    assert pd.Timestamp("2015-07-03") not in sessions_2015

    # 2020: July 4 Saturday -> July 3 Friday closed
    closed_2020 = nyse_closed_dates("2020-07-01", "2020-07-10")
    assert pd.Timestamp("2020-07-03") in closed_2020
    sessions_2020 = nyse_trading_sessions("2020-07-01", "2020-07-10")
    assert pd.Timestamp("2020-07-03") not in sessions_2020

    # 2021: July 4 Sunday -> July 5 Monday closed
    closed_2021 = nyse_closed_dates("2021-07-01", "2021-07-10")
    assert pd.Timestamp("2021-07-05") in closed_2021
    sessions_2021 = nyse_trading_sessions("2021-07-01", "2021-07-10")
    assert pd.Timestamp("2021-07-05") not in sessions_2021


def test_exact_parity_with_pinned_reference_list():
    """Verify exact parity between algorithmic calendar and pinned reference closed dates."""
    verify_reference_calendar()

    algo_closed_weekdays = [
        d.strftime("%Y-%m-%d")
        for d in nyse_closed_dates("2014-01-01", "2024-12-31")
        if d.weekday() < 5
    ]
    reference = reference_closed_dates_2014_2024()
    assert algo_closed_weekdays == reference
    assert len(algo_closed_weekdays) == 102


def test_annual_trading_session_counts():
    """Verify exact official annual trading session counts for each year 2014-2024."""
    expected_annual_counts = {
        2014: 252,
        2015: 252,
        2016: 252,
        2017: 251,
        2018: 251,  # 261 weekdays - (9 regular holidays + Dec 5 Bush mourning) = 251
        2019: 252,
        2020: 253,
        2021: 252,  # Jun 18 open, Dec 31 open
        2022: 251,  # Jun 20 closed, no New Year closure
        2023: 250,  # Jan 2 + Jun 19 closed
        2024: 252,  # Jan 1 + Jun 19 closed
    }

    for y, exp_count in expected_annual_counts.items():
        sess_y = nyse_trading_sessions(f"{y}-01-01", f"{y}-12-31")
        assert len(sess_y) == exp_count, f"Year {y} expected {exp_count} sessions, got {len(sess_y)}"


def test_total_trading_sessions_count_2014_2024():
    """Verify that 2014-01-01 through 2024-12-31 has exactly 2,768 trading sessions."""
    sessions = nyse_trading_sessions("2014-01-01", "2024-12-31")
    assert len(sessions) == 2768
    assert sessions[0] == pd.Timestamp("2014-01-02")  # 2014-01-01 was New Year's Day closure
    assert sessions[-1] == pd.Timestamp("2024-12-31")


def test_no_weekend_dates_in_trading_sessions():
    """Verify that nyse_trading_sessions() contains zero Saturday or Sunday dates."""
    sessions = nyse_trading_sessions("2014-01-01", "2024-12-31")
    assert (sessions.dayofweek >= 5).sum() == 0
    assert all(s.weekday() < 5 for s in sessions)
