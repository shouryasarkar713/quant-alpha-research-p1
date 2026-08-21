"""Volume-based feature calculations: moving averages, relative volume, and volume z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _extract_volume_series(volume: pd.DataFrame | pd.Series, column: str = "volume_split_adjusted") -> pd.Series:
    """Extract volume series from DataFrame or Series with MultiIndex (date, ticker)."""
    if isinstance(volume, pd.DataFrame):
        if column in volume.columns:
            s = volume[column]
        elif "volume" in volume.columns:
            s = volume["volume"]
        elif len(volume.columns) == 1:
            s = volume.iloc[:, 0]
        else:
            raise KeyError(f"Column '{column}' not found in volume DataFrame.")
    else:
        s = volume.copy()

    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError("Volume series must have MultiIndex (date, ticker).")
    return s


def volume_sma(
    volume: pd.DataFrame | pd.Series,
    window: int = 20,
    column: str = "volume_split_adjusted",
) -> pd.DataFrame:
    """
    Compute rolling moving average of split-adjusted trading volume per ticker.
    """
    s = _extract_volume_series(volume, column=column)
    wide = s.unstack(level="ticker")
    v_sma_wide = wide.rolling(window=window, min_periods=window).mean()
    v_sma_series = v_sma_wide.stack(future_stack=True) if hasattr(v_sma_wide, "stack") else v_sma_wide.stack(dropna=False)
    v_sma_series = v_sma_series.reindex(s.index)
    return v_sma_series.to_frame(name=f"volume_sma_{window}")


def relative_volume(
    volume: pd.DataFrame | pd.Series,
    window: int = 20,
    column: str = "volume_split_adjusted",
) -> pd.DataFrame:
    """
    Compute relative volume: V^{adj}_t / Baseline_SMA_window(V^{adj})_t per ticker.
    Compares today's volume to the trailing prior window baseline average (t-1 to t-window).
    """
    s = _extract_volume_series(volume, column=column)
    wide = s.unstack(level="ticker")
    # Baseline average of the prior window days
    v_baseline = wide.shift(1).rolling(window=window, min_periods=window).mean()
    rvol_wide = wide / v_baseline.replace(0, np.nan)
    rvol_series = rvol_wide.stack(future_stack=True) if hasattr(rvol_wide, "stack") else rvol_wide.stack(dropna=False)
    rvol_series = rvol_series.reindex(s.index)
    return rvol_series.to_frame(name="relative_volume")


def volume_zscore(
    volume: pd.DataFrame | pd.Series,
    window: int = 20,
    column: str = "volume_split_adjusted",
) -> pd.DataFrame:
    """
    Standardized volume anomaly z-score:
        (V^{adj}_t - volume_baseline_mean) / volume_baseline_std per ticker.
    """
    s = _extract_volume_series(volume, column=column)
    wide = s.unstack(level="ticker")
    v_baseline_mean = wide.shift(1).rolling(window=window, min_periods=window).mean()
    v_baseline_std = wide.shift(1).rolling(window=window, min_periods=window).std(ddof=1)
    z_wide = (wide - v_baseline_mean) / v_baseline_std.replace(0, np.nan)
    z_series = z_wide.stack(future_stack=True) if hasattr(z_wide, "stack") else z_wide.stack(dropna=False)
    z_series = z_series.reindex(s.index)
    return z_series.to_frame(name=f"volume_zscore_{window}")
