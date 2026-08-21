"""Abnormal volume with price direction signal family."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.volume import relative_volume
from src.signals.base import BaseSignal


class AbnormalVolumeSignal(BaseSignal):
    """
    Abnormal Volume + Return Direction:
    High trading volume confirms price move direction, predicting short-term continuation.
    Positive return + high volume -> positive continuation signal.
    Negative return + high volume -> negative continuation signal.

    Primary Configuration:
    - volume_lookback: 20
    - volume_threshold: None (continuous)
    - direction_horizon: 1
    - forward_horizon: 5 days
    """

    @property
    def name(self) -> str:
        return "abnormal_volume"

    @property
    def description(self) -> str:
        return (
            "Abnormal volume continuation signal: sign(ret_1d) * log(relative_volume), "
            "cross-sectionally ranked to [-1, 1]."
        )

    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        params = params or {}
        lookback = params.get("volume_lookback", params.get("lookback", 20))
        threshold = params.get("volume_threshold", None)
        direction_horizon = params.get("direction_horizon", 1)
        ranking_method = params.get("ranking_method", "percentile")

        # 1. Relative volume
        if "relative_volume" in features.columns and lookback == 20:
            rvol = features["relative_volume"]
        elif "volume_split_adjusted" in features.columns or "volume" in features.columns:
            rvol = relative_volume(features, window=lookback).iloc[:, 0]
        else:
            raise KeyError("Neither 'relative_volume' nor volume columns found in features.")

        # 2. Return direction
        ret_col = f"ret_{direction_horizon}d" if direction_horizon != 1 else "ret_1d"
        if ret_col in features.columns:
            ret = features[ret_col]
        else:
            raise KeyError(f"Return column '{ret_col}' not found in features.")

        # Compute raw indicator: sign(return) * ln(rvol)
        # Add small epsilon to rvol to prevent log(0)
        safe_rvol = np.maximum(rvol, 1e-4)
        direction = np.sign(ret)
        raw_indicator = direction * np.log(safe_rvol)

        # Apply optional volume anomaly threshold
        if threshold is not None and threshold > 1.0:
            high_vol_mask = rvol >= threshold
            raw_indicator = raw_indicator.where(high_vol_mask, 0.0)

        if ranking_method == "percentile":
            ranks = cross_sectional_rank(raw_indicator).iloc[:, 0]
            signal = 2.0 * ranks - 1.0
        elif ranking_method == "zscore":
            cs_z = cross_sectional_zscore(raw_indicator).iloc[:, 0]
            signal = np.clip(cs_z / 3.0, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown ranking_method: {ranking_method}")

        signal.name = "volume_signal"
        return signal
