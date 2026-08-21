"""Cross-sectional normalization features: percentile ranking and cross-sectional z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _extract_series(df_or_series: pd.DataFrame | pd.Series, column: str | None = None) -> pd.Series:
    """Extract Series from DataFrame or Series with MultiIndex (date, ticker)."""
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


def cross_sectional_rank(
    series: pd.DataFrame | pd.Series,
    column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional percentile rank (0.0 to 1.0) across tickers for each date:
        percentile_{i,t} = (rank(x_{i,t}) - 1) / (N_valid - 1)

    Spans exactly [0.0, 1.0] for N_valid >= 2.
    Uses method='average' for tie-breaking.
    Days with fewer than 2 valid tickers emit NaN.
    """
    s = _extract_series(series, column=column)
    col_name = s.name or "feature"
    out_name = output_column or (f"rank_{col_name}" if not str(col_name).startswith("rank_") else str(col_name))

    wide = s.unstack(level="ticker")

    def _rank_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        n_valid = len(valid)
        if n_valid < 2:
            return pd.Series(np.nan, index=row.index)
        ranks = row.rank(method="average")
        # Scale to [0.0, 1.0]
        return (ranks - 1.0) / (n_valid - 1.0)

    ranked_wide = wide.apply(_rank_row, axis=1)
    ranked_series = ranked_wide.stack(future_stack=True) if hasattr(ranked_wide, "stack") else ranked_wide.stack(dropna=False)
    ranked_series = ranked_series.reindex(s.index)
    return ranked_series.to_frame(name=out_name)


def cross_sectional_zscore(
    series: pd.DataFrame | pd.Series,
    column: str | None = None,
    output_column: str | None = None,
) -> pd.DataFrame:
    """
    Compute cross-sectional z-score across tickers for each date:
        z_{i,t} = (x_{i,t} - mean(x)_t) / std(x)_t
    """
    s = _extract_series(series, column=column)
    col_name = s.name or "feature"
    out_name = output_column or f"cs_zscore_{col_name}"

    wide = s.unstack(level="ticker")

    def _zscore_row(row: pd.Series) -> pd.Series:
        valid_cnt = row.notna().sum()
        if valid_cnt < 2:
            return pd.Series(np.nan, index=row.index)
        std_val = row.std(ddof=1)
        if std_val == 0 or np.isnan(std_val):
            return pd.Series(0.0, index=row.index)
        return (row - row.mean()) / std_val

    z_wide = wide.apply(_zscore_row, axis=1)
    z_series = z_wide.stack(future_stack=True) if hasattr(z_wide, "stack") else z_wide.stack(dropna=False)
    z_series = z_series.reindex(s.index)
    return z_series.to_frame(name=out_name)
