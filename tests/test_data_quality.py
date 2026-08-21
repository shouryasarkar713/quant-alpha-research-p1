"""Tests for data quality, universe integrity, missing-data handling, split volume adjustment, and anomalies."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import adjust_volume_by_split_factors, clean_ohlcv, validate_ohlcv
from src.data.loader import generate_synthetic_ohlcv
from src.data.universe import load_universe, load_universe_metadata


def test_universe_loads_correct_documented_start_of_sample():
    """
    Verify historical 2014 S&P 100 universe contains documented 100 constituents
    including historical constituents that later merged/delisted.
    """
    tickers = load_universe("sp100_20140101")
    assert len(tickers) == 100

    # Must contain historical constituents active in Jan 2014
    expected_historical = ["AAPL", "MSFT", "XOM", "GE", "EMC", "MON", "RTN", "TWX", "UTX", "WAG"]
    for t in expected_historical:
        assert t in tickers, f"Expected historical 2014 constituent {t} not found in universe."

    # Must NOT contain later additions / IPOs from subsequent years
    unwanted_future = ["TSLA", "PYPL", "ABNB", "UBER", "COIN", "SNOW"]
    for t in unwanted_future:
        assert t not in tickers, f"Future addition {t} leaked into 2014 start-of-sample universe."

    # Verify metadata exists and documents source
    meta = load_universe_metadata("sp100_20140101")
    assert meta.get("as_of_date") == "2014-01-01"
    assert "Wikipedia" in meta.get("source", "")
    assert meta.get("source_revision_id") == 589981231


def test_missing_bars_identifiable_as_nan():
    """
    Ensure genuinely missing bars for a stock on an active market trading day
    are preserved as NaN and never forward-filled.
    """
    dates = pd.date_range("2020-01-06", periods=4, freq="B")
    records = [
        # AAPL has missing bar on day 1
        {"date": dates[0], "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "adj_close": 100.5, "volume": 1000},
        {"date": dates[1], "ticker": "AAPL", "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan, "adj_close": np.nan, "volume": np.nan},
        {"date": dates[2], "ticker": "AAPL", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "adj_close": 102.5, "volume": 1200},
        {"date": dates[3], "ticker": "AAPL", "open": 103.0, "high": 104.0, "low": 102.0, "close": 103.5, "adj_close": 103.5, "volume": 1100},
        # MSFT trades all 4 days (proving exchange is open on day 1)
        {"date": dates[0], "ticker": "MSFT", "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "adj_close": 200.5, "volume": 2000},
        {"date": dates[1], "ticker": "MSFT", "open": 201.0, "high": 202.0, "low": 200.0, "close": 201.5, "adj_close": 201.5, "volume": 2100},
        {"date": dates[2], "ticker": "MSFT", "open": 202.0, "high": 203.0, "low": 201.0, "close": 202.5, "adj_close": 202.5, "volume": 2200},
        {"date": dates[3], "ticker": "MSFT", "open": 203.0, "high": 204.0, "low": 202.0, "close": 203.5, "adj_close": 203.5, "volume": 2300},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    cleaned, report = clean_ohlcv(df)

    assert (dates[1], "AAPL") in cleaned.index
    assert pd.isna(cleaned.loc[(dates[1], "AAPL"), "adj_close"])
    assert pd.isna(cleaned.loc[(dates[1], "AAPL"), "close"])


def test_security_termination_documented():
    """
    Verify that when a ticker ends mid-sample, its last valid date is recorded
    in the DataQualityReport and no subsequent rows are fabricated.
    """
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    records = []
    # AAPL trades all 5 days
    for d in dates:
        records.append({"date": d, "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000})
    # EMC terminates after day 2
    for d in dates[:2]:
        records.append({"date": d, "ticker": "EMC", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "adj_close": 50.0, "volume": 500})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    cleaned, report = clean_ohlcv(df)

    assert "EMC" in report.terminated_tickers
    assert report.terminated_tickers["EMC"] == dates[1].strftime("%Y-%m-%d")
    assert (dates[2], "EMC") not in cleaned.index


def test_validation_distinguishes_structural_errors_from_large_moves():
    """
    Verify validate_ohlcv flags negative prices and low > high as structural errors,
    while legitimate large returns (>50%) are flagged as diagnostic observations without
    being treated as fatal or deleted.
    """
    dates = pd.date_range("2020-01-06", periods=4, freq="B")
    records = [
        # Bar 0: Baseline
        {"date": dates[0], "ticker": "AAPL", "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "adj_close": 100.0, "volume": 1000},
        # Bar 1: Legitimate large news jump (+60% return) - must NOT be a structural error
        {"date": dates[1], "ticker": "AAPL", "open": 150.0, "high": 165.0, "low": 145.0, "close": 160.0, "adj_close": 160.0, "volume": 5000},
        # Bar 2: Structural error (low > high)
        {"date": dates[2], "ticker": "AAPL", "open": 150.0, "high": 140.0, "low": 160.0, "close": 150.0, "adj_close": 150.0, "volume": 2000},
        # Bar 3: Structural error (negative price)
        {"date": dates[3], "ticker": "AAPL", "open": 100.0, "high": 105.0, "low": -10.0, "close": 100.0, "adj_close": 100.0, "volume": 1000},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    report = validate_ohlcv(df)

    assert len(report.large_price_movements) == 1
    assert "Large daily return (+60.00%)" in report.large_price_movements[0]["issue"]
    assert len(report.structural_errors) == 4
    assert not report.is_clean


def test_dividend_does_not_alter_volume_while_split_does():
    """
    Verify that cash dividends (which change adj_close relative to close)
    do NOT alter volume_split_adjusted, whereas a stock split adjusts historical volume.
    """
    dates = pd.date_range("2020-01-06", periods=3, freq="B")
    # Case A: Cash dividend of $5.00 on day 2.
    # Day 0 & 1 raw close = 100, Day 2 raw close = 95.
    # adj_close for Day 0 & 1 is adjusted down to 95.0, but no split occurred.
    dividend_df = pd.DataFrame([
        {"date": dates[0], "ticker": "XYZ", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 95.0, "volume": 1_000_000, "split_factor": 1.0},
        {"date": dates[1], "ticker": "XYZ", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 95.0, "volume": 1_000_000, "split_factor": 1.0},
        {"date": dates[2], "ticker": "XYZ", "open": 95.0, "high": 96.0, "low": 94.0, "close": 95.0, "adj_close": 95.0, "volume": 1_000_000, "split_factor": 1.0},
    ]).set_index(["date", "ticker"])

    res_div = adjust_volume_by_split_factors(dividend_df)
    # Dividend MUST NOT change volume_split_adjusted
    assert np.allclose(res_div["volume_split_adjusted"].values, [1_000_000.0, 1_000_000.0, 1_000_000.0])

    # Case B: 2-for-1 Stock Split on day 2.
    # 1 old share -> 2 new shares. split_factor on day 2 = 2.0.
    # Historical volume on day 0 and day 1 should be adjusted by 2.0 (scaled to today's shares).
    split_df = pd.DataFrame([
        {"date": dates[0], "ticker": "ABC", "open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0, "adj_close": 100.0, "volume": 500_000, "split_factor": 1.0},
        {"date": dates[1], "ticker": "ABC", "open": 200.0, "high": 202.0, "low": 198.0, "close": 200.0, "adj_close": 100.0, "volume": 500_000, "split_factor": 1.0},
        {"date": dates[2], "ticker": "ABC", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1_000_000, "split_factor": 2.0},
    ]).set_index(["date", "ticker"])

    res_split = adjust_volume_by_split_factors(split_df)
    # Historical days scaled to 1,000,000 shares
    assert np.allclose(res_split["volume_split_adjusted"].values, [1_000_000.0, 1_000_000.0, 1_000_000.0])


def test_weekends_removed_in_clean_ohlcv():
    """
    Verify weekend observations are filtered out entirely.
    """
    dates = pd.date_range("2020-01-10", "2020-01-13", freq="D")
    records = []
    for d in dates:
        records.append({"date": d, "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    cleaned, _ = clean_ohlcv(df)

    cleaned_dates = cleaned.index.get_level_values("date")
    assert all(d.dayofweek < 5 for d in cleaned_dates)
    assert len(cleaned_dates) == 2  # Friday and Monday only


def test_synthetic_data_generator_validity():
    """
    Verify generate_synthetic_ohlcv creates clean, schema-conforming multi-asset dataset.
    """
    df = generate_synthetic_ohlcv(
        tickers=["AAPL", "MSFT", "EMC"],
        start_date="2020-01-01",
        end_date="2020-03-31",
        seed=42,
    )
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["date", "ticker"]
    for col in ["open", "high", "low", "close", "adj_close", "volume", "volume_split_adjusted"]:
        assert col in df.columns

    report = validate_ohlcv(df)
    assert report.is_clean


def test_research_mode_rejects_failed_real_downloads(monkeypatch):
    """
    Verify research mode (download_ohlcv) raises DataDownloadError when download fails,
    and NEVER silently generates synthetic fallback data.
    """
    from src.data.loader import DataDownloadError, download_ohlcv

    # Requesting non-existent / invalid tickers with 0.0 failure tolerance
    with pytest.raises(DataDownloadError):
        download_ohlcv(
            tickers=["NON_EXISTENT_TICKER_XYZ_123"],
            start_date="2020-01-01",
            end_date="2020-01-31",
            max_failure_rate=0.0,
        )


def test_no_holiday_bars_in_cleaned_dataset():
    """
    Verify that clean_ohlcv strictly excludes exchange holidays (such as 2018-12-05)
    and produces zero holiday bars.
    """
    from src.data.calendar import nyse_closed_dates

    # Simulate raw data that mistakenly contains a bar on 2018-12-05 (Bush Day of Mourning)
    records = [
        {"date": pd.Timestamp("2018-12-04"), "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000},
        {"date": pd.Timestamp("2018-12-05"), "ticker": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "adj_close": 100.0, "volume": 1000},
        {"date": pd.Timestamp("2018-12-06"), "ticker": "AAPL", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.0, "adj_close": 102.0, "volume": 1000},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    cleaned, report = clean_ohlcv(df)

    # 2018-12-05 must be filtered out
    assert pd.Timestamp("2018-12-05") not in cleaned.index.get_level_values("date")
    assert len(cleaned) == 2


def test_research_dates_are_authoritative_nyse_sessions():
    """
    Verify the persisted cleaned_ohlcv.parquet strictly matches the 2,768 authoritative
    NYSE trading sessions with 0 holiday bars and 0 weekend bars.
    """
    from src.data.calendar import nyse_trading_sessions

    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    sessions = nyse_trading_sessions("2014-01-01", "2024-12-31")
    dates = df_cleaned.index.get_level_values("date").unique()

    assert len(dates) == 2768
    assert dates.isin(sessions).all()
    assert sessions.isin(dates).all()
    assert pd.Timestamp("2018-12-05") not in dates
    assert (dates.dayofweek >= 5).sum() == 0


def test_100_securities_accounted_for_in_research_dataset():
    """
    Verify all 100 constituents of the frozen 2014 S&P 100 universe are accounted for
    (93 genuine historical constituents with data and 7 documented unavailable/legacy delistings).
    """
    universe_tickers = load_universe("sp100_20140101")
    assert len(universe_tickers) == 100

    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    cleaned_tickers = df_cleaned.index.get_level_values("ticker").unique().tolist()
    assert len(cleaned_tickers) == 93

    # Verify all 93 cleaned tickers are genuine members of the 100-ticker universe
    for t in cleaned_tickers:
        assert t in universe_tickers, f"Unknown ticker {t} in cleaned dataset."

    # Verify all 7 unavailable legacy delistings are documented and absent
    missing = set(universe_tickers) - set(cleaned_tickers)
    assert missing == {"APC", "DOW", "EMC", "FOXA", "MON", "RTN", "WAG"}

    # Verify exact panel dimensions: 92 * 2768 + 1 * 1121 = 255,777 rows
    assert len(df_cleaned) == 255777


def test_recycled_and_spinoff_tickers_rejected_as_non_frozen_identities():
    """
    Verify that recycled ticker symbols and post-2014 spin-offs are excluded from the research panel:
    - EMC: 2023 SPAC (Emerging Markets Horizon Corp)
    - DOW: 2019 spin-off Dow Inc.
    - FOXA: 2019 spin-off Fox Corporation
    """
    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    tickers = set(df_cleaned.index.get_level_values("ticker").unique())

    assert "EMC" not in tickers
    assert "DOW" not in tickers
    assert "FOXA" not in tickers


def test_frozen_identity_mappings_preserved_in_panel():
    """
    Verify that research identities are preserved under their frozen 2014 ticker symbols:
    - BK remains BK (not BNY)
    - UTX remains UTX (not RTX)
    - FB remains FB (not META)
    - BRK.B remains BRK.B (not BRK-B)
    """
    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    tickers = set(df_cleaned.index.get_level_values("ticker").unique())

    assert "BK" in tickers
    assert "BNY" not in tickers
    assert "UTX" in tickers
    assert "RTX" not in tickers
    assert "FB" in tickers
    assert "META" not in tickers
    assert "BRK.B" in tickers
    assert "BRK-B" not in tickers


def test_active_security_count_trajectory():
    """
    Verify the active security count trajectory is strictly non-increasing:
    - 2014-01-02 to 2018-06-14: exactly 93 active constituents in grid
    - 2018-06-15 to 2024-12-31: exactly 92 active constituents in grid
    """
    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    grid_counts = df_cleaned.groupby(level="date").size()

    assert (grid_counts.loc[:"2018-06-14"] == 93).all()
    assert (grid_counts.loc["2018-06-15":] == 92).all()


def test_twx_termination_date_and_zero_volume_exclusion():
    """
    Regression test for TWX termination:
    1. 2018-06-14 was the final genuine active trading day (merger with AT&T closed).
    2. 2018-06-15 was a zero-volume administrative settlement record and MUST be excluded.
    3. The cleaned dataset and DataQualityReport must record TWX termination as 2018-06-14.
    """
    cleaned_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not cleaned_path.exists():
        pytest.skip("data/processed/cleaned_ohlcv.parquet not found.")

    df_cleaned = pd.read_parquet(cleaned_path)
    twx_dates = df_cleaned.xs("TWX", level="ticker").dropna(subset=["adj_close"]).index

    assert twx_dates.max() == pd.Timestamp("2018-06-14")
    assert pd.Timestamp("2018-06-15") not in df_cleaned.xs("TWX", level="ticker").index

    # Also test generic clean_ohlcv zero-volume trailing administrative record handling
    dates = pd.date_range("2020-01-06", periods=4, freq="B")
    records = [
        # Active day 0 & 1
        {"date": dates[0], "ticker": "TEST_DELIST", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "adj_close": 50.0, "volume": 1000},
        {"date": dates[1], "ticker": "TEST_DELIST", "open": 52.0, "high": 53.0, "low": 51.0, "close": 52.0, "adj_close": 52.0, "volume": 5000},
        # Administrative zero-volume tick on day 2 (suspended/delisted)
        {"date": dates[2], "ticker": "TEST_DELIST", "open": 52.0, "high": 52.0, "low": 52.0, "close": 52.0, "adj_close": 52.0, "volume": 0},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    cleaned, report = clean_ohlcv(df)

    assert report.terminated_tickers["TEST_DELIST"] == dates[1].strftime("%Y-%m-%d")
    assert (dates[2], "TEST_DELIST") not in cleaned.index


