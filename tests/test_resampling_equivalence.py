"""Equivalence and numerical correctness tests: Reference vs Optimized Resampling."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.statistics.hypothesis_tests import bootstrap_mean_ci, permutation_test_ic


def _slow_reference_permutation_test_ic(
    signal: pd.DataFrame | pd.Series,
    forward_returns: pd.DataFrame | pd.Series,
    n_permutations: int = 200,
    seed: int = 42,
    min_obs: int = 5,
) -> tuple[float, float, pd.Series]:
    """Slow unvectorized reference implementation for equivalence verification."""
    sig_s = signal.iloc[:, 0] if isinstance(signal, pd.DataFrame) else signal.copy()
    fwd_s = forward_returns.iloc[:, 0] if isinstance(forward_returns, pd.DataFrame) else forward_returns.copy()

    aligned = pd.concat([sig_s.rename("signal"), fwd_s.rename("fwd_ret")], axis=1).dropna()
    dates_grouped = []
    for d, grp in aligned.groupby(level="date"):
        if len(grp) >= min_obs:
            dates_grouped.append((grp["signal"].values.copy(), grp["fwd_ret"].values.copy()))

    obs_ics = []
    for s_vals, r_vals in dates_grouped:
        r_corr = stats.spearmanr(s_vals, r_vals).statistic
        if not np.isnan(r_corr):
            obs_ics.append(r_corr)
    obs_mean = float(np.mean(obs_ics))

    rng = np.random.default_rng(seed)
    perm_means = np.zeros(n_permutations)
    for p in range(n_permutations):
        daily_ics = []
        for s_vals, r_vals in dates_grouped:
            shuffled_s = rng.permutation(s_vals)
            p_corr = stats.spearmanr(shuffled_s, r_vals).statistic
            if not np.isnan(p_corr):
                daily_ics.append(p_corr)
        perm_means[p] = np.mean(daily_ics) if daily_ics else 0.0

    extreme_count = np.sum(np.abs(perm_means) >= np.abs(obs_mean) - 1e-12)
    p_val = float((1.0 + extreme_count) / (float(n_permutations) + 1.0))
    return (obs_mean, p_val, pd.Series(perm_means))


def _slow_reference_bootstrap_ci(
    series: pd.Series | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Slow unvectorized reference bootstrap for equivalence verification."""
    x = series.dropna().values if isinstance(series, pd.Series) else np.asarray(series)
    T = len(x)
    observed_mean = float(np.mean(x))

    p_geom = 1.0 / float(block_size)
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        resampled_indices = np.zeros(T, dtype=int)
        current_idx = rng.integers(0, T)
        resampled_indices[0] = current_idx

        new_block_flags = rng.random(size=T - 1) < p_geom
        for t in range(1, T):
            if new_block_flags[t - 1]:
                current_idx = rng.integers(0, T)
            else:
                current_idx = (current_idx + 1) % T
            resampled_indices[t] = current_idx

        boot_means[b] = np.mean(x[resampled_indices])

    alpha_tail = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha_tail * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha_tail) * 100.0))
    return (observed_mean, ci_lower, ci_upper)


def test_permutation_test_ic_numerical_and_statistical_equivalence():
    """
    Verify optimized permutation_test_ic matches reference implementation:
    1. Observed mean IC matches to 1e-12.
    2. Null distribution mean and variance match within statistical tolerance.
    3. p-values are strictly finite-sample safe (p in (0, 1]).
    """
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    tickers = [f"TICKER_{i}" for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    sig = pd.Series(np.random.randn(len(idx)), index=idx)
    # Target return with weak correlation to signal
    ret = pd.Series(0.15 * sig.values + np.random.randn(len(idx)), index=idx)

    obs_ref, p_ref, dist_ref = _slow_reference_permutation_test_ic(sig, ret, n_permutations=300, seed=42)
    obs_opt, p_opt, dist_opt = permutation_test_ic(sig, ret, n_permutations=300, seed=42)

    # Observed mean IC must match to machine precision
    assert np.isclose(obs_ref, obs_opt, atol=1e-12), f"Observed mean IC mismatch: {obs_ref} vs {obs_opt}"

    # Permuted mean IC distributions: both must have null mean ~ 0 and matching standard deviations
    assert np.abs(dist_opt.mean()) < 0.05
    assert np.isclose(dist_ref.std(), dist_opt.std(), rtol=0.15)
    # p-values must be consistent within Monte Carlo error
    assert np.abs(p_ref - p_opt) < 0.08
    assert 0.0 < p_opt <= 1.0


def test_stationary_bootstrap_ci_equivalence():
    """
    Verify optimized bootstrap_mean_ci matches reference Politis-Romano implementation:
    1. Observed sample mean matches to 1e-12.
    2. 95% CI bounds match within Monte Carlo tolerance.
    """
    np.random.seed(456)
    T = 200
    # Autoregressive AR(1) series to test stationary block bootstrap
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + np.random.randn()

    obs_ref, lo_ref, hi_ref = _slow_reference_bootstrap_ci(x, n_bootstrap=2000, block_size=10, seed=42)
    obs_opt, lo_opt, hi_opt = bootstrap_mean_ci(x, n_bootstrap=2000, block_size=10, seed=42)

    assert np.isclose(obs_ref, obs_opt, atol=1e-12)
    assert np.isclose(lo_ref, lo_opt, atol=0.08)
    assert np.isclose(hi_ref, hi_opt, atol=0.08)
    assert lo_opt < obs_opt < hi_opt
