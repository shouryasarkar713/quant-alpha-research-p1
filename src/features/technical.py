"""Technical and statistical feature calculations: rolling moving averages, volatilities, and z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _extract_series(df_or_series: pd.DataFrame | pd.Series, column: str | None = None) -> pd.Series:
    """Extract a single Series from DataFrame or Series with MultiIndex (date, ticker)."""
    if isinstance(df_or_series, pd.DataFrame):
        if column is not None and column in df_or_series.columns:
            s = df_or_series[column]
        elif len(df_or_series.columns) == 1:
            s = df_or_series.iloc[:, 0]
        else:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
    else:
        s = df_or_series.copy()

    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError("Input series must have MultiIndex (date, ticker).")
    return s


def rolling_mean(
    series: pd.DataFrame | pd.Series,
    window: int,
    column: str = "adj_close",
) -> pd.DataFrame:
    """
    Compute rolling simple moving average per ticker.

    Parameters
    ----------
    series : pd.DataFrame | pd.Series
        MultiIndex (date, ticker).
    window : int
        Rolling window in trading days.
    column : str
        Target column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with column f'sma_{window}' (or f'{column}_sma_{window}').
    """
    s = _extract_series(series, column=column)
    wide = s.unstack(level="ticker")
    sma_wide = wide.rolling(window=window, min_periods=window).mean()
    sma_series = sma_wide.stack(future_stack=True) if hasattr(sma_wide, "stack") else sma_wide.stack(dropna=False)
    sma_series = sma_series.reindex(s.index)
    col_name = f"sma_{window}" if column == "adj_close" else f"{column}_sma_{window}"
    return sma_series.to_frame(name=col_name)


def rolling_std(
    series: pd.DataFrame | pd.Series,
    window: int,
    column: str = "ret_1d",
) -> pd.DataFrame:
    """
    Compute rolling sample standard deviation per ticker.

    IMPORTANT DISTINCTION:
    - column="ret_1d" -> std of daily returns (dimensionless, used for realized volatility)
    - column="adj_close" -> std of prices (in USD, used for price z-score denominator)

    Parameters
    ----------
    series : pd.DataFrame | pd.Series
        MultiIndex (date, ticker).
    window : int
        Rolling window in trading days.
    column : str
        Target column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with column f'rolling_std_{column}_{window}'.
    """
    s = _extract_series(series, column=column)
    wide = s.unstack(level="ticker")
    rstd_wide = wide.rolling(window=window, min_periods=window).std(ddof=1)
    rstd_series = rstd_wide.stack(future_stack=True) if hasattr(rstd_wide, "stack") else rstd_wide.stack(dropna=False)
    rstd_series = rstd_series.reindex(s.index)
    
    if column == "ret_1d":
        col_name = f"rolling_std_ret_{window}"
    elif column == "adj_close":
        col_name = f"rolling_std_price_{window}"
    else:
        col_name = f"rolling_std_{column}_{window}"

    return rstd_series.to_frame(name=col_name)


def zscore_price(
    prices: pd.DataFrame | pd.Series,
    window: int = 20,
    column: str = "adj_close",
) -> pd.DataFrame:
    """
    Price-based standardized z-score:
        z_{i,t} = (P_{i,t} - SMA_window(P_i)_t) / rolling_std_price_window(P_i)_t

    Dimensional consistency:
    - Numerator: Price deviation in USD ($)
    - Denominator: Price standard deviation in USD ($)
    - Result: Dimensionless z-score
    """
    s = _extract_series(prices, column=column)
    wide = s.unstack(level="ticker")
    sma_wide = wide.rolling(window=window, min_periods=window).mean()
    p_std_wide = wide.rolling(window=window, min_periods=window).std(ddof=1)

    z_wide = (wide - sma_wide) / p_std_wide.replace(0, np.nan)
    z_series = z_wide.stack(future_stack=True) if hasattr(z_wide, "stack") else z_wide.stack(dropna=False)
    z_series = z_series.reindex(s.index)
    return z_series.to_frame(name=f"zscore_price_{window}")


def realized_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 20,
    annualize: bool = True,
    column: str = "ret_1d",
) -> pd.DataFrame:
    """
    Compute realized volatility from daily returns per ticker:
        sigma_{i,t} = std(r_{i,t}, ..., r_{i,t-window+1}) [* sqrt(252)]

    Realized volatility is strictly based on the standard deviation of returns,
    never prices.
    """
    s = _extract_series(returns, column=column)
    wide = s.unstack(level="ticker")
    rstd_wide = wide.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol_wide = rstd_wide * np.sqrt(252.0)
    else:
        vol_wide = rstd_wide
    vol_series = vol_wide.stack(future_stack=True) if hasattr(vol_wide, "stack") else vol_wide.stack(dropna=False)
    vol_series = vol_series.reindex(s.index)
    return vol_series.to_frame(name=f"realized_vol_{window}")


def classify_regime(
    benchmark_returns: pd.Series | pd.DataFrame,
    vol_lookback: int = 20,
    trend_lookback: int = 60,
    vol_percentile_window: int = 252,
) -> pd.DataFrame:
    """
    Classify market environment into observable, lagged volatility and trend regimes.
    """
    if isinstance(benchmark_returns, pd.DataFrame):
        r = benchmark_returns.iloc[:, 0].copy()
    else:
        r = benchmark_returns.copy()

    vol_20 = r.rolling(window=vol_lookback, min_periods=vol_lookback).std(ddof=1) * np.sqrt(252.0)

    q25 = vol_20.rolling(window=vol_percentile_window, min_periods=min(60, vol_percentile_window)).quantile(0.25)
    q75 = vol_20.rolling(window=vol_percentile_window, min_periods=min(60, vol_percentile_window)).quantile(0.75)

    vol_regime = pd.Series("medium", index=r.index)
    vol_regime[vol_20 < q25] = "low"
    vol_regime[vol_20 > q75] = "high"
    vol_regime[vol_20.isna()] = np.nan

    cum_ret_60 = (1.0 + r).rolling(window=trend_lookback, min_periods=trend_lookback).apply(np.prod, raw=True) - 1.0
    trend_regime = pd.Series("non_trending", index=r.index)
    trend_regime[cum_ret_60 > 0] = "trending"
    trend_regime[cum_ret_60.isna()] = np.nan

    composite = vol_regime.astype(str) + "_" + trend_regime.astype(str)
    composite[vol_regime.isna() | trend_regime.isna()] = np.nan

    return pd.DataFrame({
        "vol_regime": vol_regime,
        "trend_regime": trend_regime,
        "regime_composite": composite,
    }, index=r.index)
