"""Information Coefficient (IC) calculations and summary inference metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(
    signal: pd.DataFrame | pd.Series,
    forward_returns: pd.DataFrame | pd.Series,
    method: str = "spearman",
    min_obs: int = 5,
) -> pd.Series:
    """
    Compute daily cross-sectional Information Coefficient (IC).

    Definition:
        IC_t = Correlation(signal_{i,t}, forward_return_{i,t}) across all valid tickers i on date t.

    Parameters
    ----------
    signal : pd.DataFrame | pd.Series
        MultiIndex (date, ticker) with signal values.
    forward_returns : pd.DataFrame | pd.Series
        MultiIndex (date, ticker) with forward returns over horizon [t+1, t+h].
    method : str
        'spearman' (rank correlation) or 'pearson' (linear correlation).
    min_obs : int
        Minimum number of valid (signal, return) pairs required on date t (default 5).
        Days with fewer valid pairs are set to NaN.

    Returns
    -------
    pd.Series
        Series indexed by date containing the cross-sectional IC for each trading day.
    """
    if isinstance(signal, pd.DataFrame):
        sig_s = signal.iloc[:, 0]
    else:
        sig_s = signal.copy()

    if isinstance(forward_returns, pd.DataFrame):
        fwd_s = forward_returns.iloc[:, 0]
    else:
        fwd_s = forward_returns.copy()

    # Align indices explicitly
    aligned = pd.concat([sig_s.rename("signal"), fwd_s.rename("fwd_ret")], axis=1).dropna()

    if aligned.empty:
        return pd.Series(dtype=float)

    # Group by date level
    grouped = aligned.groupby(level="date")

    def _calc_daily_corr(group: pd.DataFrame) -> float:
        if len(group) < min_obs:
            return np.nan
        s_vals = group["signal"].values
        r_vals = group["fwd_ret"].values
        # Check for zero variance
        if np.all(s_vals == s_vals[0]) or np.all(r_vals == r_vals[0]):
            return 0.0
        if method == "spearman":
            res = stats.spearmanr(s_vals, r_vals)
            return float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        elif method == "pearson":
            res = stats.pearsonr(s_vals, r_vals)
            return float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    ic_series = grouped.apply(_calc_daily_corr)
    ic_series.name = f"{method}_ic"
    return ic_series


def ic_summary(
    ic_series: pd.Series,
    forward_horizon: int = 1,
) -> dict[str, float]:
    """
    Compute comprehensive IC summary statistics and inferential diagnostics:
    - Mean IC: arithmetic mean across dates
    - Std IC: sample standard deviation
    - ICIR (Information Ratio): Mean IC / Std IC
    - IC Naive t-statistic: Diagnostic only (assumes i.i.d.)
    - IC HAC t-statistic: Primary inference (Newey-West autocorrelation consistent)
    - IC Hit Rate: fraction of days with IC > 0
    - Rolling 60-day mean IC

    Parameters
    ----------
    ic_series : pd.Series
        Series of daily cross-sectional ICs indexed by date.
    forward_horizon : int
        Forward return horizon in trading days (used for HAC lag parameter).

    Returns
    -------
    dict[str, float]
    """
    from src.statistics.hypothesis_tests import hac_standard_error

    valid_ic = ic_series.dropna()
    n = len(valid_ic)
    if n == 0:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_naive_t_stat": np.nan,
            "ic_naive_p_value": np.nan,
            "ic_hac_t_stat": np.nan,
            "ic_hac_p_value": np.nan,
            "ic_hit_rate": np.nan,
            "n_days": 0,
        }

    mean_ic = float(valid_ic.mean())
    std_ic = float(valid_ic.std(ddof=1)) if n > 1 else np.nan
    ic_ir = mean_ic / std_ic if (std_ic and std_ic > 0) else np.nan

    # Naive t-statistic (diagnostic only)
    se_naive = std_ic / np.sqrt(n) if n > 1 else np.nan
    t_naive = mean_ic / se_naive if (se_naive and se_naive > 0) else np.nan
    p_naive = 2.0 * (1.0 - stats.t.cdf(abs(t_naive), df=n - 1)) if not np.isnan(t_naive) else np.nan

    # HAC / Newey-West standard error with lag = forward_horizon
    hac_lag = max(1, forward_horizon)
    se_hac = hac_standard_error(valid_ic, max_lag=hac_lag)
    t_hac = mean_ic / se_hac if (se_hac and se_hac > 0) else np.nan
    p_hac = 2.0 * (1.0 - stats.norm.cdf(abs(t_hac))) if not np.isnan(t_hac) else np.nan

    # Hit rate
    hit_rate = float((valid_ic > 0).mean())

    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "ic_naive_t_stat": t_naive,
        "ic_naive_p_value": p_naive,
        "ic_hac_t_stat": t_hac,
        "ic_hac_p_value": p_hac,
        "ic_hit_rate": hit_rate,
        "n_days": n,
    }
