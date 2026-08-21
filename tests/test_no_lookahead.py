"""Dedicated unit tests auditing look-ahead bias and data leakage across features and signals."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_ohlcv
from src.features.cross_sectional import cross_sectional_rank
from src.features.engine import compute_features
from src.signals.volume_signal import AbnormalVolumeSignal


def test_features_use_only_past_data():
    """
    Feature Timing Audit:
    Given all features calculated at date t:
    Corrupting/modifying all market data after date t (t+1, t+2, ...)
    MUST NOT change any feature value at date t.
    """
    df = generate_synthetic_ohlcv(
        tickers=["AAPL", "MSFT", "JPM", "XOM"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        seed=42,
    )
    features_orig = compute_features(df, lag=0, include_forward_targets=False)

    dates = df.index.get_level_values("date").unique()
    t_eval = dates[len(dates) // 2]

    df_corrupted = df.copy()
    future_mask = df_corrupted.index.get_level_values("date") > t_eval
    df_corrupted.loc[future_mask, "adj_close"] = 999999.0
    df_corrupted.loc[future_mask, "close"] = 999999.0
    df_corrupted.loc[future_mask, "open"] = 999999.0
    df_corrupted.loc[future_mask, "high"] = 999999.0
    df_corrupted.loc[future_mask, "low"] = 999999.0
    df_corrupted.loc[future_mask, "volume"] = 999999999
    df_corrupted.loc[future_mask, "volume_split_adjusted"] = 999999999.0

    features_corrupted = compute_features(df_corrupted, lag=0, include_forward_targets=False)

    orig_at_t = features_orig.xs(t_eval, level="date")
    corrupt_at_t = features_corrupted.xs(t_eval, level="date")

    for col in orig_at_t.columns:
        s_orig = orig_at_t[col].dropna()
        s_corr = corrupt_at_t[col].dropna()
        assert len(s_orig) == len(s_corr), f"Mismatch in valid tickers on day t for {col}"
        np.testing.assert_allclose(
            s_orig.values,
            s_corr.values,
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"Look-ahead detected! Feature '{col}' at date t changed after modifying future data!",
        )


def test_abnormal_volume_timing_and_no_lookahead():
    """
    Abnormal volume signal timing audit:
    signal[t] = rank_cs(sign(r_1d[t]) * log(relative_volume[t])) * 2 - 1.
    Both r_1d[t] and relative_volume[t] use information available at close[t].
    Modifying any price or volume on t+1, t+2, ... MUST NOT change abnormal_volume[t].
    """
    df = generate_synthetic_ohlcv(
        tickers=["AAPL", "MSFT", "JPM", "XOM"],
        start_date="2020-01-01",
        end_date="2020-06-30",
        seed=42,
    )
    features_orig = compute_features(df, lag=0, include_forward_targets=False)
    sig_orig = AbnormalVolumeSignal().compute(features_orig)

    dates = df.index.get_level_values("date").unique()
    t_eval = dates[len(dates) // 2]

    # Corrupt all observations strictly after t_eval
    df_corrupt = df.copy()
    future_mask = df_corrupt.index.get_level_values("date") > t_eval
    df_corrupt.loc[future_mask, "adj_close"] = 88888.0
    df_corrupt.loc[future_mask, "volume_split_adjusted"] = 88888888.0

    features_corrupt = compute_features(df_corrupt, lag=0, include_forward_targets=False)
    sig_corrupt = AbnormalVolumeSignal().compute(features_corrupt)

    val_orig = sig_orig.xs(t_eval, level="date").dropna()
    val_corr = sig_corrupt.xs(t_eval, level="date").dropna()

    assert len(val_orig) == len(val_corr)
    np.testing.assert_allclose(
        val_orig.values,
        val_corr.values,
        atol=1e-8,
        err_msg="Abnormal volume signal at day t leaked future data after day t!",
    )


def test_no_future_in_cross_sectional_rank():
    """
    Cross-sectional rank on day t must only use data from day t.
    Modifying or dropping future observations on day t+1 must not alter ranks on day t.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    records = [
        {"date": dates[0], "ticker": "A", "val": 10.0},
        {"date": dates[0], "ticker": "B", "val": 20.0},
        {"date": dates[0], "ticker": "C", "val": 30.0},
        {"date": dates[1], "ticker": "A", "val": 15.0},
        {"date": dates[1], "ticker": "B", "val": 25.0},
        {"date": dates[1], "ticker": "C", "val": 35.0},
    ]
    df1 = pd.DataFrame(records).set_index(["date", "ticker"])
    ranks1 = cross_sectional_rank(df1, column="val")

    df2 = df1.copy()
    df2.loc[(dates[1], "A"), "val"] = 99999.0
    ranks2 = cross_sectional_rank(df2, column="val")

    r0_1 = ranks1.xs(dates[0], level="date")["rank_val"].values
    r0_2 = ranks2.xs(dates[0], level="date")["rank_val"].values
    np.testing.assert_allclose(r0_1, r0_2, atol=1e-8)
