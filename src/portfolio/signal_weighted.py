"""Signal-weighted long-short portfolio constructor."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.constraints import enforce_portfolio_constraints


class SignalWeightedLongShort(BasePortfolioConstructor):
    """
    Continuous Signal-Weighted Long-Short Portfolio:
    - Cross-sectionally de-means signals on each date: s_tilde_i = s_i - mean(s)
    - Normalizes by sum of absolute values to achieve exactly 1.0 gross exposure:
          w_hat_i = s_tilde_i / sum(|s_tilde|)
    - If sum(|s_tilde|) < tolerance (e.g. all signals identical), returns all-zero weights (cash).
    - If fewer than min_eligible securities (default 5) are valid on date t, returns all-zero weights.
    - Missing value policy: NaN signals are excluded from allocation.
    - Gross exposure = 1.0, Net exposure = 0.0
    - Applies hard constraint Euclidean projection (max position <= 0.10, net <= 0.20, gross <= 1.0)
    """

    @property
    def name(self) -> str:
        return "signal_weighted"

    def construct_weights(
        self,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        min_eligible = params.get("min_eligible", 5)
        max_gross = params.get("max_gross", 1.0)
        max_net = params.get("max_net", 0.20)
        max_position = params.get("max_position", 0.10)

        if isinstance(signals, pd.DataFrame):
            sig_s = signals.iloc[:, 0].copy()
        else:
            sig_s = signals.copy()

        wide = sig_s.unstack(level="ticker")

        def _allocate_date(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            n = len(valid)
            weights = pd.Series(np.nan, index=row.index)
            weights[valid.index] = 0.0

            if n < min_eligible:
                return weights

            # 1. Demean signals
            demeaned = valid - valid.mean()

            # 2. Normalize by L1 norm
            l1_norm = np.sum(np.abs(demeaned))
            if l1_norm < 1e-12 or np.isnan(l1_norm):
                # All signals identical -> zero allocation
                return weights

            raw_w = demeaned / l1_norm
            weights[valid.index] = raw_w
            return weights

        raw_weights_wide = wide.apply(_allocate_date, axis=1)
        raw_weights_series = raw_weights_wide.stack(future_stack=True) if hasattr(raw_weights_wide, "stack") else raw_weights_wide.stack(dropna=False)
        raw_weights_series = raw_weights_series.reindex(sig_s.index)

        # Enforce hard constraints
        constrained = enforce_portfolio_constraints(
            raw_weights_series,
            max_gross=max_gross,
            max_net=max_net,
            max_position=max_position,
        )
        return constrained.to_frame(name="target_weight")
