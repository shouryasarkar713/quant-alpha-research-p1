"""Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado (2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


def compute_probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """
    Compute the Probabilistic Sharpe Ratio (PSR) following Bailey & Lopez de Prado (2012):
        PSR(SR*) = Phi( ((SR - SR*) * sqrt(T - 1)) / sqrt(1 - gamma_3 * SR + ((gamma_4 - 1)/4) * SR^2) )

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Daily strategy return series.
    benchmark_sharpe : float
        Benchmark annualized Sharpe ratio (default 0.0).
    annualization_factor : float
        Trading periods per year (default 252).

    Returns
    -------
    float
        PSR value in [0, 1].
    """
    clean_r = np.asarray(returns)[~np.isnan(returns)]
    t = len(clean_r)
    if t < 5:
        return 0.5

    mean_r = float(np.mean(clean_r))
    std_r = float(np.std(clean_r, ddof=1))
    if std_r < 1e-12:
        return 0.5

    # Daily Sharpe and benchmark daily Sharpe
    sr_daily = mean_r / std_r
    sr_bench_daily = benchmark_sharpe / np.sqrt(annualization_factor)

    # Skewness (gamma_3) and Fisher Kurtosis (gamma_4 - 3, so raw kurtosis gamma_4 = excess + 3)
    sk = float(skew(clean_r))
    excess_kurt = float(kurtosis(clean_r, fisher=True))
    raw_kurt = excess_kurt + 3.0  # Pearson kurtosis

    variance_term = 1.0 - sk * sr_daily + ((raw_kurt - 1.0) / 4.0) * (sr_daily ** 2)
    if variance_term <= 0:
        return 0.5

    z_stat = (sr_daily - sr_bench_daily) * np.sqrt(t - 1.0) / np.sqrt(variance_term)
    return float(norm.cdf(z_stat))


def compute_deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    num_trials: int = 10,
    variance_of_trials: float = 0.5,
    annualization_factor: float = 252.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR) following Bailey & Lopez de Prado (2014):
    Adjusts the estimated Sharpe ratio for:
    1. Non-normality (skewness and fat tails).
    2. Selection bias / multi-trial overfitting from testing N alternative signals/parameters.

    Expected Maximum Sharpe among N independent trials:
        SR* = sqrt(V) * ( (1 - gamma_euler) * Z^-1(1 - 1/N) + gamma_euler * Z^-1(1 - 1/(N * e)) )

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        Daily strategy return series.
    num_trials : int
        Number of configurations/trials tested (N >= 1).
    variance_of_trials : float
        Variance of annualized Sharpe ratios across tested trials (default 0.5).
    annualization_factor : float
        Annualization multiplier (default 252).

    Returns
    -------
    float
        DSR value in [0, 1].
    """
    clean_r = np.asarray(returns)[~np.isnan(returns)]
    t = len(clean_r)
    if t < 5 or num_trials < 1:
        return 0.5

    euler_gamma = 0.57721566490153286

    if num_trials == 1:
        expected_max_sr = 0.0
    else:
        # Extreme value expectation for max of N standard normal variables
        z_1 = norm.ppf(1.0 - 1.0 / num_trials)
        z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        expected_max_sr = np.sqrt(variance_of_trials) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)

    return compute_probabilistic_sharpe_ratio(
        returns=clean_r,
        benchmark_sharpe=expected_max_sr,
        annualization_factor=annualization_factor,
    )
