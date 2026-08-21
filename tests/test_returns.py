"""Unit tests for return calculations: simple, log, multi-horizon, skip, and forward returns."""

import numpy as np
import pandas as pd
import pytest

from src.features.returns import (
    forward_return,
    log_return,
    simple_return,
    skip_return,
)


def test_simple_return_basic():
    """
    Prices: [100, 110, 105]
    Expected 1-day returns: [NaN, 0.10, -0.0454545...]
    """
    dates = pd.date_range("2020-01-06", periods=3, freq="B")
    df = pd.DataFrame([
        {"date": dates[0], "ticker": "AAPL", "adj_close": 100.0},
        {"date": dates[1], "ticker": "AAPL", "adj_close": 110.0},
        {"date": dates[2], "ticker": "AAPL", "adj_close": 105.0},
    ]).set_index(["date", "ticker"])

    ret = simple_return(df, period=1)
    values = ret["ret_1d"].values

    assert np.isnan(values[0])
    assert np.isclose(values[1], 0.10, atol=1e-8)
    assert np.isclose(values[2], (105.0 - 110.0) / 110.0, atol=1e-8)


def test_log_return_basic():
    """
    Prices: [100, 110, 105]
    Expected 1-day log returns: [NaN, ln(1.1), ln(105/110)]
    """
    dates = pd.date_range("2020-01-06", periods=3, freq="B")
    df = pd.DataFrame([
        {"date": dates[0], "ticker": "AAPL", "adj_close": 100.0},
        {"date": dates[1], "ticker": "AAPL", "adj_close": 110.0},
        {"date": dates[2], "ticker": "AAPL", "adj_close": 105.0},
    ]).set_index(["date", "ticker"])

    ret = log_return(df, period=1)
    values = ret["log_ret_1d"].values

    assert np.isnan(values[0])
    assert np.isclose(values[1], np.log(1.10), atol=1e-8)
    assert np.isclose(values[2], np.log(105.0 / 110.0), atol=1e-8)


def test_skip_return():
    """
    Known 10-day price sequence.
    skip_return(total=5, skip=2) should use prices: (P_{t-2} - P_{t-5}) / P_{t-5}.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    records = [{"date": dates[i], "ticker": "AAPL", "adj_close": prices[i]} for i in range(10)]
    df = pd.DataFrame(records).set_index(["date", "ticker"])

    ret = skip_return(df, total_period=5, skip_period=2)
    # For day index 6 (prices[6] = 22.0):
    # P_{t-2} = prices[4] = 18.0
    # P_{t-5} = prices[1] = 12.0
    # Expected return = (18 - 12) / 12 = 0.50
    val = ret.loc[(dates[6], "AAPL"), "ret_5d_skip2d"]
    assert np.isclose(val, 0.50, atol=1e-8)


def test_skip_return_12_1_boundary():
    """
    Deterministic boundary test for Jegadeesh-Titman 12-1 momentum:
    Let day t be index 252 (total 253 days, from day 0 to day 252).
    P[t-252] (day 0)   = 100.0
    P[t-21]  (day 231) = 120.0
    P[t]     (day 252) = 200.0 (huge price spike in most recent month)

    The 12-1 momentum return at day t MUST be (120 - 100) / 100 = 20.0% (0.20),
    and must NOT be (200 - 100) / 100 = 100.0% (1.00).
    """
    dates = pd.date_range("2020-01-01", periods=253, freq="B")
    prices = np.full(253, 100.0)
    prices[0] = 100.0      # day t - 252
    prices[231] = 120.0    # day t - 21
    prices[252] = 200.0    # day t

    df = pd.DataFrame([{"date": dates[i], "ticker": "AAPL", "adj_close": prices[i]} for i in range(253)]).set_index(["date", "ticker"])

    ret = skip_return(df, total_period=252, skip_period=21)
    val_t = ret.loc[(dates[252], "AAPL"), "ret_252d_skip21d"]

    # Must be exactly +20%, completely ignoring the recent spike to 200.0
    assert np.isclose(val_t, 0.20, atol=1e-8)
    assert not np.isclose(val_t, 1.00, atol=1e-8)


def test_return_nan_propagation():
    """If a price is NaN (missing bar), the return should be NaN (not 0 or interpolated)."""
    dates = pd.date_range("2020-01-06", periods=4, freq="B")
    records = [
        {"date": dates[0], "ticker": "AAPL", "adj_close": 100.0},
        {"date": dates[1], "ticker": "AAPL", "adj_close": np.nan},
        {"date": dates[2], "ticker": "AAPL", "adj_close": 105.0},
        {"date": dates[3], "ticker": "AAPL", "adj_close": 110.0},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])

    ret = simple_return(df, period=1)
    assert np.isnan(ret.loc[(dates[0], "AAPL"), "ret_1d"])
    assert np.isnan(ret.loc[(dates[1], "AAPL"), "ret_1d"])
    assert np.isnan(ret.loc[(dates[2], "AAPL"), "ret_1d"])
    assert np.isclose(ret.loc[(dates[3], "AAPL"), "ret_1d"], 5.0 / 105.0, atol=1e-8)


def test_forward_return_alignment():
    """
    Verify forward return for horizon h:
    fwd_ret[t] = adj_close[t+h] / adj_close[t] - 1
    Hand-calculated for a 5-day sequence.
    """
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    prices = [100.0, 102.0, 104.0, 106.0, 108.0]
    records = [{"date": dates[i], "ticker": "AAPL", "adj_close": prices[i]} for i in range(5)]
    df = pd.DataFrame(records).set_index(["date", "ticker"])

    fwd1 = forward_return(df, horizon=1)
    assert np.isclose(fwd1.loc[(dates[0], "AAPL"), "fwd_ret_1d"], 0.02, atol=1e-8)
    assert np.isnan(fwd1.loc[(dates[4], "AAPL"), "fwd_ret_1d"])

    fwd2 = forward_return(df, horizon=2)
    assert np.isclose(fwd2.loc[(dates[0], "AAPL"), "fwd_ret_2d"], 0.04, atol=1e-8)
    assert np.isnan(fwd2.loc[(dates[3], "AAPL"), "fwd_ret_2d"])
