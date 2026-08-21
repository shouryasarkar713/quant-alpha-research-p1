"""Low-volatility anomaly signal family."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.technical import realized_volatility
from src.signals.base import BaseSignal


class VolatilitySignal(BaseSignal):
    """
    Low Volatility Signal:
    Stocks with lower recent realized return volatility historically delivered
    higher risk-adjusted returns (the low-volatility anomaly).

    Primary Configuration:
    - lookback_days: 60
    - forward_horizon: 20 days
    - ranking_method: 'percentile'
    """

    @property
    def name(self) -> str:
        return "low_vol"

    @property
    def description(self) -> str:
        return (
            "Low-volatility anomaly signal: ranks stocks inversely by 60-day realized "
            "return volatility, mapped to [-1, 1] (lowest volatility -> +1.0)."
        )

    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        params = params or {}
        lookback = params.get("lookback", params.get("lookback_days", 60))
        ranking_method = params.get("ranking_method", "percentile")

        col_name = f"realized_vol_{lookback}"
        if col_name in features.columns:
            vol = features[col_name]
        elif "ret_1d" in features.columns:
            vol = realized_volatility(features, window=lookback, annualize=True, column="ret_1d").iloc[:, 0]
        else:
            raise KeyError(f"Neither '{col_name}' nor 'ret_1d' found in features DataFrame.")

        # Negate volatility: low vol -> positive signal
        neg_vol = -vol

        if ranking_method == "percentile":
            ranks = cross_sectional_rank(neg_vol).iloc[:, 0]
            signal = 2.0 * ranks - 1.0
        elif ranking_method == "zscore":
            cs_z = cross_sectional_zscore(neg_vol).iloc[:, 0]
            signal = np.clip(cs_z / 3.0, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown ranking_method: {ranking_method}")

        signal.name = "volatility_signal"
        return signal
