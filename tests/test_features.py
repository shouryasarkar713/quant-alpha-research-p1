"""Unit tests for feature engineering: technical indicators, volatilities, volumes, and cross-sectional ranks."""

import numpy as np
import pandas as pd
import pytest

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.engine import compute_features
from src.features.technical import (
    classify_regime,
    realized_volatility,
    rolling_mean,
    rolling_std,
    zscore_price,
)
from src.features.volume import relative_volume, volume_sma, volume_zscore


def test_rolling_mean():
    """
    Prices: [10, 12, 14, 16, 18]
    rolling_mean(window=3): [NaN, NaN, 12.0, 14.0, 16.0]
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = [10.0, 12.0, 14.0, 16.0, 18.0]
    df = pd.DataFrame([{"date": dates[i], "ticker": "AAPL", "adj_close": prices[i]} for i in range(5)]).set_index(["date", "ticker"])

    sma = rolling_mean(df, window=3)
    vals = sma["sma_3"].values
    assert np.isnan(vals[0])
    assert np.isnan(vals[1])
    assert np.isclose(vals[2], 12.0, atol=1e-8)
    assert np.isclose(vals[3], 14.0, atol=1e-8)
    assert np.isclose(vals[4], 16.0, atol=1e-8)


def test_zscore_price_known_values():
    """
    Hand-calculated z-score using rolling_std of PRICES (not returns).
    Verify denominator is in price units ($).

    Prices: [100, 102, 98, 104, 100], window=3
    Day 4 (prices: 98, 104, 100):
      SMA_3 = 100.666667
      std_3 = 3.055050
      zscore = (100 - 100.666667) / 3.055050 ≈ -0.218218
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = [100.0, 102.0, 98.0, 104.0, 100.0]
    df = pd.DataFrame([{"date": dates[i], "ticker": "AAPL", "adj_close": prices[i]} for i in range(5)]).set_index(["date", "ticker"])

    z = zscore_price(df, window=3)
    val = z.loc[(dates[4], "AAPL"), "zscore_price_3"]
    expected_z = (100.0 - (98.0 + 104.0 + 100.0) / 3.0) / np.std([98.0, 104.0, 100.0], ddof=1)
    assert np.isclose(val, expected_z, atol=1e-6)
    assert np.isclose(val, -0.218218, atol=1e-4)


def test_rolling_std_price_vs_return_different():
    """
    Verify rolling_std(column='adj_close') produces price dispersion (USD)
    while rolling_std(column='ret_1d') produces return volatility (dimensionless).
    They are fundamentally different quantities and must not be confused.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    records = [
        {"date": dates[0], "ticker": "AAPL", "adj_close": 100.0, "ret_1d": np.nan},
        {"date": dates[1], "ticker": "AAPL", "adj_close": 102.0, "ret_1d": 0.02},
        {"date": dates[2], "ticker": "AAPL", "adj_close": 98.0, "ret_1d": -0.0392157},
        {"date": dates[3], "ticker": "AAPL", "adj_close": 104.0, "ret_1d": 0.0612245},
        {"date": dates[4], "ticker": "AAPL", "adj_close": 100.0, "ret_1d": -0.0384615},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])

    p_std = rolling_std(df, window=3, column="adj_close")
    r_std = rolling_std(df, window=3, column="ret_1d")

    val_p = p_std.loc[(dates[4], "AAPL"), "rolling_std_price_3"]
    val_r = r_std.loc[(dates[4], "AAPL"), "rolling_std_ret_3"]

    assert np.isclose(val_p, 3.055050, atol=1e-4)  # in USD
    assert np.isclose(val_r, 0.057773, atol=1e-4)  # dimensionless
    assert abs(val_p - val_r) > 1.0  # Confirms distinct magnitude and dimension


def test_realized_vol_annualization():
    """
    If daily return standard deviation is 0.01 (1%),
    annualized realized volatility is 0.01 * sqrt(252) ≈ 0.158745 (15.87%).
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    # Returns with std = 0.01
    ret_series = [np.nan, 0.01, -0.01, 0.01]
    df = pd.DataFrame([{"date": dates[i], "ticker": "AAPL", "ret_1d": ret_series[i]} for i in range(4)]).set_index(["date", "ticker"])

    vol = realized_volatility(df, window=3, annualize=True, column="ret_1d")
    val = vol.loc[(dates[3], "AAPL"), "realized_vol_3"]

    sample_std = np.std([0.01, -0.01, 0.01], ddof=1)
    expected_vol = sample_std * np.sqrt(252.0)
    assert np.isclose(val, expected_vol, atol=1e-8)


def test_relative_volume():
    """
    Volume: [100, 100, 100, 200]
    With window=3: relative_volume[3] = 200 / 100 = 2.0
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    vols = [100.0, 100.0, 100.0, 200.0]
    df = pd.DataFrame([{"date": dates[i], "ticker": "AAPL", "volume_split_adjusted": vols[i]} for i in range(4)]).set_index(["date", "ticker"])

    rvol = relative_volume(df, window=3)
    assert np.isclose(rvol.loc[(dates[3], "AAPL"), "relative_volume"], 2.0, atol=1e-8)


def test_cross_sectional_rank_range():
    """
    Verify cross-sectional percentile ranking is in [0, 1] and preserves rank order across tickers.
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {"date": dates[0], "ticker": "A", "val": 10.0},
        {"date": dates[0], "ticker": "B", "val": 30.0},
        {"date": dates[0], "ticker": "C", "val": 20.0},
        {"date": dates[1], "ticker": "A", "val": 50.0},
        {"date": dates[1], "ticker": "B", "val": 10.0},
        {"date": dates[1], "ticker": "C", "val": 20.0},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    ranks = cross_sectional_rank(df, column="val")

    # On date 0: A=10 (lowest), C=20 (mid), B=30 (highest)
    assert ranks.loc[(dates[0], "A"), "rank_val"] < ranks.loc[(dates[0], "C"), "rank_val"]
    assert ranks.loc[(dates[0], "C"), "rank_val"] < ranks.loc[(dates[0], "B"), "rank_val"]
    assert (ranks["rank_val"].dropna() >= 0.0).all()
    assert (ranks["rank_val"].dropna() <= 1.0).all()


def test_feature_nan_on_missing_bar():
    """
    If adj_close is NaN on day t (missing bar), all features derived from that
    price should be NaN on day t. No imputation.
    """
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    records = []
    for i, d in enumerate(dates):
        # Missing bar on day 15
        p = np.nan if i == 15 else 100.0 + i
        records.append({"date": d, "ticker": "AAPL", "adj_close": p, "volume_split_adjusted": 1000.0})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    z = zscore_price(df, window=20)
    assert np.isnan(z.loc[(dates[15], "AAPL"), "zscore_price_20"])


def test_compute_features_orchestrator(sample_toy_prices):
    """
    Verify compute_features generates complete Section 10.3 feature schema without errors.
    """
    features = compute_features(sample_toy_prices, lag=0, include_forward_targets=True)
    expected_cols = [
        "ret_1d", "log_ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_252d",
        "ret_252d_skip21d", "sma_20", "sma_60", "rolling_std_ret_20",
        "rolling_std_ret_60", "rolling_std_price_20", "zscore_price_20",
        "realized_vol_20", "realized_vol_60", "vol_ratio", "volume_sma_20",
        "relative_volume", "volume_zscore_20", "rank_ret_20d",
        "rank_ret_252d_skip21d", "rank_vol_20", "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"
    ]
    for col in expected_cols:
        assert col in features.columns, f"Feature column '{col}' missing from engine output."
    assert isinstance(features.index, pd.MultiIndex)
