"""Short-term price mean-reversion signal family."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.technical import zscore_price
from src.signals.base import BaseSignal


class MeanReversionSignal(BaseSignal):
    """
    Short-Term Price Mean Reversion:
    Stocks that have fallen significantly below their recent moving average in price
    standard deviation units (negative price z-score) tend to bounce back upward.

    Primary Configuration:
    - lookback_days: 20
    - threshold: None (continuous signal)
    - forward_horizon: 5 days
    """

    @property
    def name(self) -> str:
        return "mean_reversion_zscore"

    @property
    def description(self) -> str:
        return (
            "Short-term price mean reversion: negates the 20-day price z-score "
            "and cross-sectionally ranks into [-1, 1] (oversold -> positive signal)."
        )

    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        params = params or {}
        lookback = params.get("lookback", params.get("lookback_days", 20))
        threshold = params.get("threshold", params.get("z_threshold", None))
        ranking_method = params.get("ranking_method", "percentile")

        col_name = f"zscore_price_{lookback}"
        if col_name in features.columns:
            z = features[col_name]
        elif "adj_close" in features.columns:
            z = zscore_price(features, window=lookback, column="adj_close").iloc[:, 0]
        else:
            raise KeyError(f"Neither '{col_name}' nor 'adj_close' found in features DataFrame.")

        # Negate z-score: extreme negative price deviation -> positive signal (long bias)
        neg_z = -z

        # Apply threshold filter if specified
        if threshold is not None and threshold > 0:
            active_mask = z.abs() >= threshold
            neg_z = neg_z.where(active_mask, np.nan)

        if ranking_method == "percentile":
            ranks = cross_sectional_rank(neg_z).iloc[:, 0]
            signal = 2.0 * ranks - 1.0
        elif ranking_method == "zscore":
            cs_z = cross_sectional_zscore(neg_z).iloc[:, 0]
            signal = np.clip(cs_z / 3.0, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown ranking_method: {ranking_method}")

        signal.name = "mean_reversion_signal"
        return signal
