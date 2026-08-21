"""Unit and statistical property tests for IC, HAC standard errors, block bootstrap, within-date permutation, and stationarity."""

import numpy as np
import pandas as pd
import pytest

from src.statistics.hypothesis_tests import (
    bootstrap_mean_ci,
    hac_standard_error,
    permutation_test_ic,
    quintile_spread_analysis,
)
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.statistics.stationarity import adf_test, kpss_test


def test_ic_perfect_positive_signal():
    """Rank identical signal and forward returns -> Spearman IC must equal +1.0 exactly."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    records = []
    for d in dates:
        for val, ticker in [(1.0, "A"), (2.0, "B"), (3.0, "C"), (4.0, "D"), (5.0, "E")]:
            records.append({"date": d, "ticker": ticker, "signal": val, "fwd_ret": val * 0.01})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    ic = compute_ic(df["signal"], df["fwd_ret"], method="spearman")

    assert len(ic) == 5
    np.testing.assert_allclose(ic.values, 1.0, atol=1e-8)


def test_ic_perfect_inverted_signal():
    """Inverted signal -> Spearman IC must equal -1.0 exactly."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    records = []
    for d in dates:
        for val, ticker in [(1.0, "A"), (2.0, "B"), (3.0, "C"), (4.0, "D"), (5.0, "E")]:
            records.append({"date": d, "ticker": ticker, "signal": val, "fwd_ret": -val * 0.01})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    ic = compute_ic(df["signal"], df["fwd_ret"], method="spearman")

    assert len(ic) == 5
    np.testing.assert_allclose(ic.values, -1.0, atol=1e-8)


def test_ic_summary_pipeline_operates_on_daily_series():
    """
    Verification Check 2:
    The inference pipeline is:
    cross-sectional IC for each date t -> daily 1D IC time series -> mean IC -> HAC standard error -> t-stat/p-value.
    It operates on the 1D daily IC series, NOT by pooling cross-sectional stock-level observations across dates.
    """
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    ic_values = pd.Series(
        [0.05, 0.08, -0.02, 0.10, 0.04, 0.06, 0.03, -0.01, 0.07, 0.05,
         0.04, 0.02, 0.09, 0.06, 0.05, -0.03, 0.08, 0.04, 0.06, 0.05],
        index=dates,
        name="daily_ic",
    )
    res = ic_summary(ic_values, forward_horizon=5)

    assert isinstance(res, dict)
    assert res["n_days"] == 20
    assert np.isclose(res["mean_ic"], ic_values.mean(), atol=1e-6)
    assert not np.isnan(res["ic_hac_t_stat"])
    assert not np.isnan(res["ic_hac_p_value"])
    # Hit rate = fraction of days with positive IC
    assert np.isclose(res["ic_hit_rate"], (ic_values > 0).mean(), atol=1e-6)


def test_hac_standard_error_positively_autocorrelated():
    """
    Mandatory Rule 2:
    Do NOT test requiring HAC to always exceed naive standard error in arbitrary series.
    Instead, test numerical correctness generally, and use a specifically constructed
    positively autocorrelated toy series (AR(1) with rho = 0.8) where positive autocorrelation
    mathematically increases uncertainty, verifying se_hac > se_naive.
    """
    # 1. General numerical properties on white noise
    rng = np.random.default_rng(42)
    wn = rng.normal(0, 1, size=200)
    se_hac_wn = hac_standard_error(wn, max_lag=5)
    se_naive_wn = np.std(wn, ddof=1) / np.sqrt(200)
    assert se_hac_wn > 0
    assert np.isfinite(se_hac_wn)
    assert abs(se_hac_wn - se_naive_wn) < 0.03

    # 2. Specifically constructed positively autocorrelated series: x_t = 0.8 * x_{t-1} + e_t
    ar1 = np.zeros(500)
    for t in range(1, 500):
        ar1[t] = 0.8 * ar1[t - 1] + rng.normal(0, 1)

    se_naive_ar = np.std(ar1, ddof=1) / np.sqrt(500)
    se_hac_ar = hac_standard_error(ar1, max_lag=10)

    # Positive persistence must increase HAC uncertainty over naive i.i.d. assumption
    assert se_hac_ar > se_naive_ar, f"Expected se_hac ({se_hac_ar}) > se_naive ({se_naive_ar}) for AR(1) rho=0.8"


def test_permutation_within_date_invariance_and_finite_sample_p_value():
    """
    Mandatory Rule 3 & Verification 5:
    Within-date cross-sectional permutation test:
    - Shuffles signal values only among securities on the same date
    - Keeps forward returns unchanged
    - Never moves observations between dates
    - Finite-sample-safe p-value: p = (1 + extreme_count) / (B + 1) > 0
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    records = []
    for d in dates:
        for i, ticker in enumerate(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"]):
            ret_val = float(i) * 0.01
            records.append({
                "date": d,
                "ticker": ticker,
                "signal": float(i),
                "fwd_ret": ret_val,
            })

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    obs_mean, p_val, perm_dist = permutation_test_ic(
        df["signal"],
        df["fwd_ret"],
        n_permutations=200,
        seed=42,
    )

    assert np.isclose(obs_mean, 1.0, atol=1e-8)
    assert abs(perm_dist.mean()) < 0.15
    # Finite-sample safe p-value is strictly > 0: exactly 1 / (200 + 1) if 0 permuted stats exceed 1.0
    assert p_val > 0.0
    assert p_val <= 0.05
    assert len(perm_dist) == 200


def test_stationary_bootstrap_mean_ci_coverage():
    """
    Verification Check 3:
    Stationary block bootstrap (Politis & Romano 1994) with geometric block lengths
    captures the true population mean inside the 95% confidence interval.
    """
    rng = np.random.default_rng(123)
    true_mean = 0.05
    data = true_mean + rng.normal(0, 0.02, size=300)

    obs_mean, ci_low, ci_high = bootstrap_mean_ci(data, n_bootstrap=1000, confidence=0.95, block_size=5, seed=42)

    assert ci_low < obs_mean < ci_high
    assert ci_low <= true_mean <= ci_high


def test_quintile_spread_daily_cross_sectional_formation():
    """
    Verification Check 6:
    Quintiles are formed separately on each date using ONLY that date's valid signal values,
    and HAC inference is performed on the resulting daily spread time series.
    """
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    records = []
    rng = np.random.default_rng(99)

    for d in dates:
        for i in range(10):
            ticker = f"TICKER_{i}"
            sig_val = float(i)
            ret_val = 0.005 * sig_val + rng.normal(0, 0.002)
            records.append({"date": d, "ticker": ticker, "signal": sig_val, "fwd_ret": ret_val})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    res = quintile_spread_analysis(df["signal"], df["fwd_ret"], n_quantiles=5)

    q_rets = res["quintile_returns"]
    assert q_rets["Q5"] > q_rets["Q1"]
    assert res["spread_mean"] > 0
    assert res["spread_t_stat"] > 2.0
    assert len(res["daily_spreads"]) == 50


def test_stationarity_diagnostics():
    """
    Verification Check 4:
    ADF and KPSS tests execute as diagnostic tools on stationary series.
    """
    rng = np.random.default_rng(42)
    stationary_series = pd.Series(rng.normal(0, 1, size=250))

    adf_res = adf_test(stationary_series)
    kpss_res = kpss_test(stationary_series)

    assert adf_res["is_stationary_5pct"] is True
    assert kpss_res["is_stationary_5pct"] is True
