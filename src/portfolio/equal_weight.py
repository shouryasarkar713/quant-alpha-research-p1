"""Equal-weight top/bottom quantile long-short portfolio constructor."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.constraints import enforce_portfolio_constraints


class EqualWeightLongShort(BasePortfolioConstructor):
    """
    Equal-Weight Long-Short Portfolio:
    - Long top quantile (e.g. top 20% quintile) with weight +1 / (2 * N_long)
    - Short bottom quantile (e.g. bottom 20% quintile) with weight -1 / (2 * N_short)
    - Intermediate securities: 0.0 weight
    - Missing value policy: NaN signals are excluded from allocation.
    - If fewer than min_eligible securities (default 5) are valid on date t, returns all-zero weights.
    - Gross exposure = 1.0, Net exposure = 0.0
    """

    @property
    def name(self) -> str:
        return "equal_weight_long_short"

    def construct_weights(
        self,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        quantile_pct = params.get("quantile_pct", 0.20)
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

            # Number of long and short assets
            k = max(1, int(np.floor(n * quantile_pct)))
            # If n >= 10, expand k to at least 5 so individual position does not breach 0.10
            if n >= 10:
                k = max(k, int(np.ceil(0.5 / max_position)))
            k = min(k, n // 2)

            if k == 0:
                return weights

            # Stable sorting for ties at quantile boundaries
            sorted_valid = valid.sort_values(ascending=False, kind="stable")
            long_tickers = sorted_valid.index[:k]
            short_tickers = sorted_valid.index[-k:]

            w_long = 0.5 / k
            w_short = -0.5 / k

            weights[long_tickers] = w_long
            weights[short_tickers] = w_short
            return weights

        raw_weights_wide = wide.apply(_allocate_date, axis=1)
        raw_weights_series = raw_weights_wide.stack(future_stack=True) if hasattr(raw_weights_wide, "stack") else raw_weights_wide.stack(dropna=False)
        raw_weights_series = raw_weights_series.reindex(sig_s.index)

        constrained = enforce_portfolio_constraints(
            raw_weights_series,
            max_gross=max_gross,
            max_net=max_net,
            max_position=max_position,
        )
        return constrained.to_frame(name="target_weight")
