"""Cross-sectional momentum signal family (Jegadeesh & Titman, 1993)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.returns import skip_return
from src.signals.base import BaseSignal


class MomentumSignal(BaseSignal):
    """
    Cross-Sectional Momentum (12-1 Month):
    Past winners over the last 12 months (skipping the most recent month to avoid
    short-term reversals) tend to continue outperforming past losers.

    Primary Configuration:
    - lookback_days: 252 (~12 months)
    - skip_days: 21 (~1 month)
    - ranking_method: 'percentile'
    - forward_horizon: 20 days
    """

    @property
    def name(self) -> str:
        return "momentum_12_1"

    @property
    def description(self) -> str:
        return (
            "Cross-sectional 12-1 momentum: ranks stocks by past 252-day return "
            "skipping the most recent 21 days, mapped to [-1, 1]."
        )

    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        params = params or {}
        lookback = params.get("lookback", params.get("lookback_days", 252))
        skip = params.get("skip", params.get("skip_days", 21))
        ranking_method = params.get("ranking_method", "percentile")

        # Feature retrieval or dynamic calculation
        col_name = f"ret_{lookback}d_skip{skip}d" if skip > 0 else f"ret_{lookback}d"
        if col_name in features.columns:
            raw_ret = features[col_name]
        elif "adj_close" in features.columns:
            if skip > 0:
                raw_ret = skip_return(features, total_period=lookback, skip_period=skip, column="adj_close").iloc[:, 0]
            else:
                from src.features.returns import simple_return
                raw_ret = simple_return(features, period=lookback, column="adj_close").iloc[:, 0]
        else:
            raise KeyError(f"Neither '{col_name}' nor 'adj_close' found in features DataFrame.")

        # Cross-sectional normalization to [-1, 1]
        if ranking_method == "percentile":
            ranks = cross_sectional_rank(raw_ret).iloc[:, 0]
            signal = 2.0 * ranks - 1.0
        elif ranking_method == "zscore":
            z = cross_sectional_zscore(raw_ret).iloc[:, 0]
            # Clip z-score to [-3, 3] and map to [-1, 1]
            signal = np.clip(z / 3.0, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown ranking_method: {ranking_method}")

        signal.name = "momentum_signal"
        return signal
