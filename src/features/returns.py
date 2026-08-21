"""Return calculations: simple returns, log returns, multi-horizon, skip returns, and forward targets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _extract_price_series(prices: pd.DataFrame | pd.Series, column: str = "adj_close") -> pd.Series:
    """Extract a single price series from DataFrame or Series with MultiIndex (date, ticker)."""
    if isinstance(prices, pd.DataFrame):
        if column in prices.columns:
            s = prices[column]
        elif len(prices.columns) == 1:
            s = prices.iloc[:, 0]
        else:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
    else:
        s = prices.copy()

    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError("Price series must have MultiIndex (date, ticker).")
    return s


def simple_return(prices: pd.DataFrame | pd.Series, period: int = 1, column: str = "adj_close") -> pd.DataFrame:
    """
    Compute simple return: (P_t - P_{t-period}) / P_{t-period} per ticker.

    Parameters
    ----------
    prices : pd.DataFrame | pd.Series
        MultiIndex (date, ticker) with adjusted close prices.
    period : int
        Number of trading days.
    column : str
        Price column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with column f'ret_{period}d'.
    """
    s = _extract_price_series(prices, column=column)
    wide = s.unstack(level="ticker")
    shifted = wide.shift(period)
    ret_wide = (wide - shifted) / shifted
    ret_series = ret_wide.stack(future_stack=True) if hasattr(ret_wide, "stack") else ret_wide.stack(dropna=False)
    ret_series = ret_series.reindex(s.index)
    col_name = f"ret_{period}d" if period != 1 else "ret_1d"
    return ret_series.to_frame(name=col_name)


def log_return(prices: pd.DataFrame | pd.Series, period: int = 1, column: str = "adj_close") -> pd.DataFrame:
    """
    Compute log return: ln(P_t / P_{t-period}) per ticker.

    Parameters
    ----------
    prices : pd.DataFrame | pd.Series
        MultiIndex (date, ticker) with adjusted close prices.
    period : int
        Number of trading days.
    column : str
        Price column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with column f'log_ret_{period}d'.
    """
    s = _extract_price_series(prices, column=column)
    wide = s.unstack(level="ticker")
    shifted = wide.shift(period)
    log_ret_wide = np.log(wide / shifted)
    log_ret_series = log_ret_wide.stack(future_stack=True) if hasattr(log_ret_wide, "stack") else log_ret_wide.stack(dropna=False)
    log_ret_series = log_ret_series.reindex(s.index)
    col_name = f"log_ret_{period}d" if period != 1 else "log_ret_1d"
    return log_ret_series.to_frame(name=col_name)


def skip_return(
    prices: pd.DataFrame | pd.Series,
    total_period: int = 252,
    skip_period: int = 21,
    column: str = "adj_close",
) -> pd.DataFrame:
    """
    Compute return from t-total_period to t-skip_period:
    (P_{t-skip_period} - P_{t-total_period}) / P_{t-total_period}

    Classic Jegadeesh-Titman momentum (12-1):
    skip_return(252, 21) = 12-month return skipping last month.
    """
    if total_period <= skip_period:
        raise ValueError(f"total_period ({total_period}) must be strictly greater than skip_period ({skip_period}).")

    s = _extract_price_series(prices, column=column)
    wide = s.unstack(level="ticker")
    p_skip = wide.shift(skip_period)
    p_total = wide.shift(total_period)
    ret_wide = (p_skip - p_total) / p_total
    ret_series = ret_wide.stack(future_stack=True) if hasattr(ret_wide, "stack") else ret_wide.stack(dropna=False)
    ret_series = ret_series.reindex(s.index)
    col_name = f"ret_{total_period}d_skip{skip_period}d"
    return ret_series.to_frame(name=col_name)


def forward_return(
    prices: pd.DataFrame | pd.Series,
    horizon: int = 1,
    column: str = "adj_close",
) -> pd.DataFrame:
    """
    Compute forward return: P_{t+horizon} / P_t - 1 per ticker.
    Used for IC computation and signal evaluation.
    """
    s = _extract_price_series(prices, column=column)
    wide = s.unstack(level="ticker")
    p_future = wide.shift(-horizon)
    fwd_wide = (p_future - wide) / wide
    fwd_series = fwd_wide.stack(future_stack=True) if hasattr(fwd_wide, "stack") else fwd_wide.stack(dropna=False)
    fwd_series = fwd_series.reindex(s.index)
    col_name = f"fwd_ret_{horizon}d" if horizon != 1 else "fwd_ret_1d"
    return fwd_series.to_frame(name=col_name)
