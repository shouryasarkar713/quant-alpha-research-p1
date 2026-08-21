"""Pre-specified equal-weight combined statistical baseline signal."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank
from src.signals.base import BaseSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.momentum import MomentumSignal
from src.signals.volatility import VolatilitySignal
from src.signals.volume_signal import AbnormalVolumeSignal


class CombinedSignal(BaseSignal):
    """
    Pre-Specified Combined Statistical Signal (Baseline for ML Comparison):

    Order of operations as specified in v1.2:
    1. Compute the four individual signals:
       - Momentum (12-1)
       - Short-term mean reversion (20d price z-score)
       - Low volatility (60d realized vol)
       - Abnormal volume (20d rvol * return direction)
       (each cross-sectionally ranked in [-1, 1])
    2. Compute equal-weight average:
       raw_combined = (momentum + mean_reversion + volatility + volume) / 4.0
       Note: If ANY component signal is NaN for a stock on date t, raw_combined is NaN.
       We do NOT renormalize the remaining signals.
    3. Re-rank cross-sectionally across available securities on date t:
       combined_signal = 2 * rank_cs(raw_combined) - 1

    This combined signal is pre-specified BEFORE inspecting any results to avoid
    winner-selection bias.
    """

    def __init__(self) -> None:
        self.mom = MomentumSignal()
        self.mr = MeanReversionSignal()
        self.vol = VolatilitySignal()
        self.vol_sig = AbnormalVolumeSignal()

    @property
    def name(self) -> str:
        return "combined_signal"

    @property
    def description(self) -> str:
        return (
            "Pre-specified equal-weight combination of momentum (12-1), short-term mean reversion (20d), "
            "low volatility (60d), and abnormal volume (20d), cross-sectionally re-ranked to [-1, 1]."
        )

    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        params = params or {}
        weights = params.get("weights", {
            "momentum_12_1": 0.25,
            "mean_reversion_zscore": 0.25,
            "low_vol": 0.25,
            "abnormal_volume": 0.25,
        })

        s_mom = self.mom.compute(features)
        s_mr = self.mr.compute(features)
        s_vol = self.vol.compute(features)
        s_vol_sig = self.vol_sig.compute(features)

        # Strict NaN propagation: If ANY component is NaN, the combination is NaN
        raw_combined = (
            weights.get("momentum_12_1", 0.25) * s_mom
            + weights.get("mean_reversion_zscore", 0.25) * s_mr
            + weights.get("low_vol", 0.25) * s_vol
            + weights.get("abnormal_volume", 0.25) * s_vol_sig
        )

        # Re-rank cross-sectionally to map to [-1, 1]
        ranks = cross_sectional_rank(raw_combined).iloc[:, 0]
        signal = 2.0 * ranks - 1.0
        signal.name = "combined_signal"
        return signal
