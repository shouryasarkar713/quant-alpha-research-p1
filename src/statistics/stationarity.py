"""Stationarity diagnostics: Augmented Dickey-Fuller (ADF) and KPSS tests.

IMPORTANT METHODOLOGICAL NOTE:
ADF and KPSS tests are diagnostic tools only. Neither test definitively 'proves' stationarity,
and their results are NOT used to selectively modify the inference methodology after observing results.
The primary inferential framework remains the pre-specified HAC (Newey-West) standard error on the
daily IC time series and the non-parametric stationary block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series | np.ndarray) -> dict[str, float | bool]:
    """
    Augmented Dickey-Fuller (ADF) unit-root test for stationarity (diagnostic).

    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternative (H1): The series is stationary.

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series data (e.g. daily IC series, return series).

    Returns
    -------
    dict
        {
            'test_stat': float,
            'p_value': float,
            'lags_used': int,
            'n_obs': int,
            'critical_values': dict[str, float],
            'is_stationary_5pct': bool  # True if p_value < 0.05 (rejects unit root)
        }
    """
    if isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = np.asarray(series)
        x = x[~np.isnan(x)]

    if len(x) < 10:
        return {
            "test_stat": np.nan,
            "p_value": np.nan,
            "lags_used": 0,
            "n_obs": len(x),
            "critical_values": {},
            "is_stationary_5pct": False,
        }

    res = adfuller(x, autolag="AIC")
    return {
        "test_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "n_obs": int(res[3]),
        "critical_values": {k: float(v) for k, v in res[4].items()},
        "is_stationary_5pct": bool(res[1] < 0.05),
    }


def kpss_test(series: pd.Series | np.ndarray, regression: str = "c") -> dict[str, float | bool]:
    """
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity test (diagnostic).

    Null Hypothesis (H0): The series is level/trend stationary.
    Alternative (H1): The series has a unit root (is non-stationary).

    Parameters
    ----------
    series : pd.Series | np.ndarray
        Time series data.
    regression : str
        'c' for level stationary, 'ct' for trend stationary.

    Returns
    -------
    dict
        {
            'test_stat': float,
            'p_value': float,
            'lags_used': int,
            'critical_values': dict[str, float],
            'is_stationary_5pct': bool  # True if p_value >= 0.05 (fails to reject stationarity)
        }
    """
    if isinstance(series, pd.Series):
        x = series.dropna().values
    else:
        x = np.asarray(series)
        x = x[~np.isnan(x)]

    if len(x) < 10:
        return {
            "test_stat": np.nan,
            "p_value": np.nan,
            "lags_used": 0,
            "critical_values": {},
            "is_stationary_5pct": False,
        }

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kpss(x, regression=regression, nlags="auto")

    return {
        "test_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "critical_values": {k: float(v) for k, v in res[3].items()},
        "is_stationary_5pct": bool(res[1] >= 0.05),
    }
