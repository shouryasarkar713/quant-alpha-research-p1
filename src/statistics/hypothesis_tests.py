"""Non-parametric and HAC hypothesis testing: Newey-West, stationary block bootstrap, within-date permutation, and quintile spreads."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats


def hac_standard_error(
    series: pd.Series | np.ndarray,
    max_lag: int | None = None,
) -> float:
    """
    Compute Heteroskedasticity and Autocorrelation Consistent (HAC / Newey-West 1987)
    standard error of the sample mean:
        var_HAC(x_bar) = (1 / T) * [ gamma_0 + 2 * sum_{l=1}^L (1 - l / (L+1)) * gamma_l ]
        se_HAC(x_bar)  = sqrt(var_HAC(x_bar))

    HAC Lag Convention:
    For an h-day forward return target, overlapping periods induce an MA(h-1) autocorrelation
    structure in the daily IC series. The Bartlett kernel truncation lag L is chosen as:
        L = max(1, h)
    For arbitrary time series where h is unspecified, Newey & West (1994) plug-in rule is used:
        L = max(1, int(4 * (T / 100)**(2/9)))

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series data (specifically the 1D daily IC series {IC_t} or daily strategy returns).
    max_lag : int | None
        Maximum autocorrelation lag L.

    Returns
    -------
    float
        HAC standard error of the mean.
    """
    if isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = np.asarray(series)
        x = x[~np.isnan(x)]

    T = len(x)
    if T <= 1:
        return np.nan

    if max_lag is None:
        max_lag = max(1, int(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    max_lag = min(max_lag, T - 1)

    # Demeaned series
    x_demeaned = x - np.mean(x)

    # Sample autocovariance gamma_0
    gamma_0 = np.dot(x_demeaned, x_demeaned) / T
    omega = gamma_0

    # Sample autocovariances gamma_l with Bartlett weights w_l = 1 - l / (L + 1)
    for l in range(1, max_lag + 1):
        weight = 1.0 - (l / (max_lag + 1.0))
        gamma_l = np.dot(x_demeaned[l:], x_demeaned[:-l]) / T
        omega += 2.0 * weight * gamma_l

    # Ensure non-negative spectral density estimate
    omega = max(omega, 1e-14)
    var_mean = omega / T
    return float(np.sqrt(var_mean))


def hac_t_test(
    series: pd.Series | np.ndarray,
    max_lag: int | None = None,
) -> tuple[float, float]:
    """
    Perform HAC / Newey-West t-test of H0: mean == 0 vs H1: mean != 0.

    Returns
    -------
    tuple[float, float]
        (t_stat, p_value)
    """
    if isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = np.asarray(series)
        x = x[~np.isnan(x)]

    T = len(x)
    if T <= 1:
        return (0.0, 1.0)

    mean_val = float(np.mean(x))
    se = hac_standard_error(x, max_lag=max_lag)
    if se <= 1e-12 or np.isnan(se):
        return (0.0, 1.0)

    t_stat = mean_val / se
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(1, T - 1))))
    return (float(t_stat), float(p_val))


def bootstrap_mean_ci(
    series: pd.Series | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int = 42,
    batch_size: int = 5000,
    progress_callback: Any = None,
) -> tuple[float, float, float]:
    """
    Stationary Block Bootstrap (Politis & Romano, 1994) confidence interval for the sample mean (Optimized Vectorized).

    Methodology:
    1. Stationary bootstrap with random geometric block lengths to ensure the resampled
       series is strictly stationary (unlike fixed-block bootstrap which breaks stationarity
       at block boundaries).
    2. Geometric distribution parameter: p = 1.0 / mean_block_size.
    3. Circular wrapping is used so all starting positions [0, T-1] have equal probability.
    4. Mean block size rule (pre-specified):
       - If block_size is provided: mean_block_size = block_size.
       - Otherwise: mean_block_size = max(2, int(T**(1/3))).
       - For an h-day horizon: set block_size = max(2, h).

    Parameters
    ----------
    series : pd.Series | np.ndarray
        1D daily time series observations (e.g. daily IC series).
    n_bootstrap : int
        Number of bootstrap replications (default 10,000).
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    block_size : int | None
        Expected mean block length (geometric distribution mean).
    seed : int
        Random seed for deterministic reproducibility.
    batch_size : int
        Batch size for vectorization (default 5,000).
    progress_callback : callable | None
        Optional callback func(completed, total) for progress reporting.

    Returns
    -------
    tuple[float, float, float]
        (observed_mean, ci_lower, ci_upper)
    """
    if isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = np.asarray(series)
        x = x[~np.isnan(x)]

    T = len(x)
    if T == 0:
        return (np.nan, np.nan, np.nan)

    observed_mean = float(np.mean(x))
    if T < 5:
        return (observed_mean, observed_mean, observed_mean)

    if block_size is None:
        mean_block = max(2.0, float(T ** (1.0 / 3.0)))
    else:
        mean_block = max(2.0, float(block_size))

    # Probability of ending a block in geometric distribution
    p_geom = 1.0 / mean_block
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap, dtype=float)

    for start_b in range(0, n_bootstrap, batch_size):
        end_b = min(start_b + batch_size, n_bootstrap)
        cur_batch = end_b - start_b

        # Simulate stationary bootstrap index matrix of shape (cur_batch, T)
        idx_matrix = np.empty((cur_batch, T), dtype=np.int32)
        idx_matrix[:, 0] = rng.integers(0, T, size=cur_batch)

        new_block_flags = rng.random(size=(cur_batch, T - 1)) < p_geom
        random_starts = rng.integers(0, T, size=(cur_batch, T - 1))

        for t in range(1, T):
            idx_matrix[:, t] = np.where(
                new_block_flags[:, t - 1],
                random_starts[:, t - 1],
                (idx_matrix[:, t - 1] + 1) % T,
            )

        resampled_vals = x[idx_matrix]
        boot_means[start_b:end_b] = np.mean(resampled_vals, axis=1)

        if progress_callback is not None:
            progress_callback(end_b, n_bootstrap)

    alpha_tail = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha_tail * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha_tail) * 100.0))

    return (observed_mean, ci_lower, ci_upper)


def permutation_test_ic(
    signal: pd.DataFrame | pd.Series,
    forward_returns: pd.DataFrame | pd.Series,
    n_permutations: int = 1_000,
    seed: int = 42,
    min_obs: int = 5,
    batch_size: int = 500,
    progress_callback: Any = None,
) -> tuple[float, float, pd.Series]:
    """
    Within-date cross-sectional permutation test for mean IC (Optimized Vectorized).

    Methodology (Specification Section 14.4 & User Mandatory Rule 3):
    1. For each date t:
       - Keep forward returns strictly fixed for that date.
       - Randomly permute the signal values ONLY among securities present on that date.
       - Never move observations between dates.
       - Preserve the daily cross-section size and set of signal values within each date.
    2. Compute daily cross-sectional Spearman IC using exact rank-correlation formulation:
       For date t with demeaned signal ranks x_t and demeaned return ranks y_t:
           IC_{t, b} = (x_{t, pi_b}^T y_t) / (||x_t||_2 * ||y_t||_2)
       where denominator D_t = ||x_t||_2 * ||y_t||_2 is strictly invariant to any permutation pi_b.
    3. Finite-Sample Safe Empirical Two-Sided p-value (Davison & Hinkley 1997 / Phipson & Smyth 2010):
           p = (1 + sum_{b=1}^B I(|perm_mean_ic_b| >= |obs_mean_ic| - 1e-12)) / (B + 1)
       Ensures p > 0 and prevents exact zero probabilities under finite resampling.

    Parameters
    ----------
    signal : pd.DataFrame | pd.Series
        Signal values with MultiIndex (date, ticker).
    forward_returns : pd.DataFrame | pd.Series
        Forward return values with MultiIndex (date, ticker).
    n_permutations : int
        Number of permutations (default 1,000 for standard research runs, 10,000 for final).
    seed : int
        Random seed for deterministic reproducibility.
    min_obs : int
        Minimum valid tickers on a date to evaluate IC.
    batch_size : int
        Permutation batch size (default 500).
    progress_callback : callable | None
        Optional callback func(completed, total) for progress reporting.

    Returns
    -------
    tuple[float, float, pd.Series]
        (observed_mean_ic, p_value, permutation_distribution)
    """
    if isinstance(signal, pd.DataFrame):
        sig_s = signal.iloc[:, 0]
    else:
        sig_s = signal.copy()

    if isinstance(forward_returns, pd.DataFrame):
        fwd_s = forward_returns.iloc[:, 0]
    else:
        fwd_s = forward_returns.copy()

    aligned = pd.concat([sig_s.rename("signal"), fwd_s.rename("fwd_ret")], axis=1).dropna()
    if aligned.empty:
        return (np.nan, np.nan, pd.Series(dtype=float))

    # Pre-rank and compute demeaned ranks & weight vectors per date
    daily_data = []
    obs_ics = []

    for d, grp in aligned.groupby(level="date"):
        if len(grp) < min_obs:
            continue
        s_vals = grp["signal"].values
        r_vals = grp["fwd_ret"].values
        n_k = len(s_vals)

        # Check for zero variance
        if np.all(s_vals == s_vals[0]) or np.all(r_vals == r_vals[0]):
            continue

        # Fractional ranks (handling ties identical to scipy.stats.rankdata)
        rank_s = stats.rankdata(s_vals).astype(float)
        rank_r = stats.rankdata(r_vals).astype(float)

        x_d = rank_s - np.mean(rank_s)
        y_d = rank_r - np.mean(rank_r)

        norm_x = np.sqrt(np.sum(x_d ** 2))
        norm_y = np.sqrt(np.sum(y_d ** 2))

        if norm_x < 1e-12 or norm_y < 1e-12:
            continue

        # Observed IC for this date
        obs_ic = float(np.sum(x_d * y_d) / (norm_x * norm_y))
        obs_ics.append(obs_ic)

        # Precomputed weight vector w_t = y_d / (norm_x * norm_y)
        w_t = y_d / (norm_x * norm_y)
        daily_data.append((x_d, w_t, n_k))

    if not daily_data:
        return (np.nan, np.nan, pd.Series(dtype=float))

    observed_mean_ic = float(np.mean(obs_ics))
    n_dates = len(daily_data)

    rng = np.random.default_rng(seed)
    perm_means = np.zeros(n_permutations, dtype=float)

    # Process permutations in memory-safe batches
    for start_b in range(0, n_permutations, batch_size):
        end_b = min(start_b + batch_size, n_permutations)
        cur_batch = end_b - start_b

        # sum of ICs across dates for this batch of permutations
        batch_sum_ic = np.zeros(cur_batch, dtype=float)

        for x_d, w_t, n_k in daily_data:
            # Generate cur_batch random permutations of length n_k
            random_keys = rng.random(size=(cur_batch, n_k))
            perm_indices = np.argsort(random_keys, axis=1)  # shape (cur_batch, n_k)

            # Permuted signal ranks: x_d[perm_indices] dot w_t -> (cur_batch,)
            batch_ic_t = np.dot(x_d[perm_indices], w_t)
            batch_sum_ic += batch_ic_t

        perm_means[start_b:end_b] = batch_sum_ic / float(n_dates)

        if progress_callback is not None:
            progress_callback(end_b, n_permutations)

    extreme_count = int(np.sum(np.abs(perm_means) >= np.abs(observed_mean_ic) - 1e-12))
    p_val = float((1.0 + extreme_count) / (float(n_permutations) + 1.0))
    perm_series = pd.Series(perm_means, name="permutation_mean_ics")

    return (observed_mean_ic, p_val, perm_series)


def quintile_spread_analysis(
    signal: pd.DataFrame | pd.Series,
    forward_returns: pd.DataFrame | pd.Series,
    n_quantiles: int = 5,
    min_obs: int = 5,
) -> dict:
    """
    Perform cross-sectional quintile spread analysis:
    1. On each date t, sort stocks by signal into n_quantiles (Q1 to Q5) using ONLY date t's data.
    2. Compute equal-weight forward return for each quintile on date t.
    3. Daily long-short spread: spread_t = Q5_t - Q1_t.
    4. Compute HAC standard error and t-statistic on the resulting daily spread time series {spread_t}.

    No full-sample ranking or cross-date pooling is used.

    Parameters
    ----------
    signal : pd.DataFrame | pd.Series
        Signal values with MultiIndex (date, ticker).
    forward_returns : pd.DataFrame | pd.Series
        Forward returns with MultiIndex (date, ticker).
    n_quantiles : int
        Number of quantiles (default 5 for quintiles).

    Returns
    -------
    dict
        {
            'quintile_returns': dict[str, float], # 'Q1' to 'Q5' mean return
            'spread_mean': float,                  # Mean Q5 - Q1 return
            'spread_std': float,                   # Std of daily Q5 - Q1 spread
            'spread_t_stat': float,                # HAC t-stat of spread
            'spread_p_value': float,
            'daily_spreads': pd.Series             # Time series of Q5 - Q1
        }
    """
    if isinstance(signal, pd.DataFrame):
        sig_s = signal.iloc[:, 0]
    else:
        sig_s = signal.copy()

    if isinstance(forward_returns, pd.DataFrame):
        fwd_s = forward_returns.iloc[:, 0]
    else:
        fwd_s = forward_returns.copy()

    aligned = pd.concat([sig_s.rename("signal"), fwd_s.rename("fwd_ret")], axis=1).dropna()
    if aligned.empty:
        return {}

    q_labels = [f"Q{i+1}" for i in range(n_quantiles)]
    daily_q_returns = {q: [] for q in q_labels}
    daily_spread_list = []
    dates_list = []

    for d, grp in aligned.groupby(level="date"):
        if len(grp) < min_obs:
            continue

        try:
            # Assign quantile bins strictly cross-sectionally on day t
            ranks = grp["signal"].rank(method="first")
            q_bins = pd.qcut(ranks, q=n_quantiles, labels=q_labels)
            grp_with_q = grp.assign(quantile=q_bins)
            mean_by_q = grp_with_q.groupby("quantile", observed=False)["fwd_ret"].mean()

            for q in q_labels:
                daily_q_returns[q].append(mean_by_q.get(q, np.nan))

            spread = mean_by_q.get(f"Q{n_quantiles}", np.nan) - mean_by_q.get("Q1", np.nan)
            daily_spread_list.append(spread)
            dates_list.append(d)
        except Exception:
            continue

    daily_spread_series = pd.Series(daily_spread_list, index=pd.to_datetime(dates_list), name="quintile_spread")
    valid_spreads = daily_spread_series.dropna()

    mean_spread = float(valid_spreads.mean()) if not valid_spreads.empty else np.nan
    std_spread = float(valid_spreads.std(ddof=1)) if len(valid_spreads) > 1 else np.nan

    se_hac = hac_standard_error(valid_spreads, max_lag=5) if len(valid_spreads) > 5 else (std_spread / np.sqrt(max(1, len(valid_spreads))))
    t_hac = mean_spread / se_hac if (se_hac and se_hac > 0) else np.nan
    p_hac = 2.0 * (1.0 - stats.norm.cdf(abs(t_hac))) if not np.isnan(t_hac) else np.nan

    quintile_means = {q: float(np.nanmean(daily_q_returns[q])) for q in q_labels}

    return {
        "quintile_returns": quintile_means,
        "spread_mean": mean_spread,
        "spread_std": std_spread,
        "spread_t_stat": t_hac,
        "spread_p_value": p_hac,
        "daily_spreads": daily_spread_series,
    }
