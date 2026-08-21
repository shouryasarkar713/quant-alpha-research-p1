"""Observable market regime definitions: volatility regimes and trend regimes (Specification Section 23)."""

from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd

from src.evaluation.metrics import PerformanceMetrics, compute_performance_metrics


def compute_market_volatility_regime(
    market_returns: pd.Series,
    window: int = 20,
    low_pct: float = 0.40,
    high_pct: float = 0.60,
) -> pd.Series:
    """
    Define market volatility regimes based on rolling 20-day market return volatility:
    - 'LOW_VOL': rolling vol <= 40th percentile
    - 'NORMAL_VOL': 40th percentile < rolling vol <= 60th percentile
    - 'HIGH_VOL': rolling vol > 60th percentile

    Parameters
    ----------
    market_returns : pd.Series
        Equal-weighted or benchmark market return series.
    window : int
        Rolling lookback window.
    low_pct : float
        Lower quantile threshold.
    high_pct : float
        Upper quantile threshold.

    Returns
    -------
    pd.Series
        Categorical regime label per date.
    """
    rolling_vol = market_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252.0)
    low_thresh = rolling_vol.quantile(low_pct)
    high_thresh = rolling_vol.quantile(high_pct)

    regimes = pd.Series("NORMAL_VOL", index=market_returns.index)
    regimes[rolling_vol <= low_thresh] = "LOW_VOL"
    regimes[rolling_vol > high_thresh] = "HIGH_VOL"
    regimes[rolling_vol.isna()] = np.nan
    return regimes


def compute_market_trend_regime(
    market_prices: pd.Series,
    fast_window: int = 50,
    slow_window: int = 200,
) -> pd.Series:
    """
    Define market trend regime based on moving average crossover:
    - 'BULL_TREND': Fast MA (50d) > Slow MA (200d)
    - 'BEAR_TREND': Fast MA (50d) <= Slow MA (200d)

    Parameters
    ----------
    market_prices : pd.Series
        Benchmark or average market price level series.
    fast_window : int
        Fast moving average window (default 50).
    slow_window : int
        Slow moving average window (default 200).

    Returns
    -------
    pd.Series
        Categorical regime label per date ('BULL_TREND' / 'BEAR_TREND').
    """
    fast_ma = market_prices.rolling(fast_window, min_periods=fast_window // 2).mean()
    slow_ma = market_prices.rolling(slow_window, min_periods=slow_window // 2).mean()

    regimes = pd.Series("BULL_TREND", index=market_prices.index)
    regimes[fast_ma <= slow_ma] = "BEAR_TREND"
    regimes[fast_ma.isna() | slow_ma.isna()] = np.nan
    return regimes


def evaluate_regime_performance(
    strategy_returns: pd.Series,
    regime_series: pd.Series,
) -> dict[str, PerformanceMetrics]:
    """
    Evaluate conditional strategy performance sliced across distinct market regimes.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy return series.
    regime_series : pd.Series
        Daily categorical regime classification.

    Returns
    -------
    dict[str, PerformanceMetrics]
        Metrics per active regime.
    """
    aligned = pd.concat([strategy_returns.rename("return"), regime_series.rename("regime")], axis=1).dropna()
    results = {}

    for regime_name, group in aligned.groupby("regime"):
        rets = group["return"]
        if len(rets) >= 5:
            results[str(regime_name)] = compute_performance_metrics(rets)

    return results
