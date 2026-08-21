"""Inverse-volatility signal-weighted portfolio constructor."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.constraints import enforce_portfolio_constraints


class InverseVolatilitySignalWeighted(BasePortfolioConstructor):
    """
    Inverse-Volatility Signal Weighting (Specification Section 16.3 & Mandatory Rule 4):
    - Individual asset weight is proportional to demeaned signal divided by effective volatility:
          w_tilde_i = s_tilde_i / max(volatility_i, sigma_min)
      where sigma_min = 0.05 (pre-specified volatility floor to prevent unstable large weights).
    - Securities with NaN signals or NaN/negative volatility are strictly excluded from allocation.
    - Zero or near-zero volatility is floored at sigma_min.
    - If fewer than min_eligible securities (default 5) have valid signal and volatility on date t,
      returns all-zero target weights (cash).
    - Demeaned to ensure dollar neutrality (net exposure = 0.0).
    - Normalized by sum of absolute values to achieve strictly <= 1.0 gross exposure:
          w_hat_i = w_tilde_i / sum(|w_tilde|)
    - NO portfolio-level volatility targeting or leverage scaling.
    - Applies hard constraint Euclidean projection (max position <= 0.10, net <= 0.20, gross <= 1.0).
    """

    @property
    def name(self) -> str:
        return "inverse_volatility_signal_weighted"

    def construct_weights(
        self,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        vol_lookback = params.get("vol_lookback", 60)
        sigma_min = params.get("sigma_min", 0.05)  # Volatility floor
        min_eligible = params.get("min_eligible", 5)  # Minimum active securities
        max_gross = params.get("max_gross", 1.0)
        max_net = params.get("max_net", 0.20)
        max_position = params.get("max_position", 0.10)

        if isinstance(signals, pd.DataFrame):
            sig_s = signals.iloc[:, 0].copy()
        else:
            sig_s = signals.copy()

        # Retrieve realized volatility series from features
        vol_col = f"realized_vol_{vol_lookback}"
        if features is not None and vol_col in features.columns:
            vol_s = features[vol_col]
        elif features is not None and "realized_vol_20" in features.columns:
            vol_s = features["realized_vol_20"]
        elif features is not None and "ret_1d" in features.columns:
            from src.features.technical import realized_volatility
            vol_s = realized_volatility(features, window=vol_lookback).iloc[:, 0]
        else:
            vol_s = pd.Series(1.0, index=sig_s.index)

        # Missing value policy: only securities with BOTH valid signal and non-null, non-negative vol
        aligned = pd.concat([sig_s.rename("signal"), vol_s.rename("vol")], axis=1)
        wide_sig = aligned["signal"].unstack(level="ticker")
        wide_vol = aligned["vol"].unstack(level="ticker")

        dates = wide_sig.index
        weights_dict = {}

        for d in dates:
            s_row = wide_sig.loc[d]
            v_row = wide_vol.loc[d]

            # Filter for valid, non-null, non-negative volatility
            valid_mask = s_row.notna() & v_row.notna() & (v_row >= 0)
            s_valid = s_row[valid_mask]
            v_valid = v_row[valid_mask]

            n = len(s_valid)
            if n < min_eligible:
                row_res = pd.Series(np.nan, index=wide_sig.columns)
                row_res[s_valid.index] = 0.0
                weights_dict[d] = row_res
                continue

            # 1. Demean signals
            demeaned_sig = s_valid - s_valid.mean()

            # 2. Scale by inverse volatility with floor (zero/near-zero floored to sigma_min)
            sigma_eff = np.maximum(v_valid.values, sigma_min)
            raw_inv_vol = demeaned_sig.values / sigma_eff

            # 3. Check for zero variance / zero denominator
            l1_norm = np.sum(np.abs(raw_inv_vol))
            if l1_norm < 1e-12 or np.isnan(l1_norm):
                row_res = pd.Series(np.nan, index=wide_sig.columns)
                row_res[s_valid.index] = 0.0
                weights_dict[d] = row_res
                continue

            # 4. Demean to ensure exact dollar neutrality
            demeaned_w = raw_inv_vol - np.mean(raw_inv_vol)
            l1_norm_demeaned = np.sum(np.abs(demeaned_w))

            if l1_norm_demeaned < 1e-12 or np.isnan(l1_norm_demeaned):
                row_res = pd.Series(np.nan, index=wide_sig.columns)
                row_res[s_valid.index] = 0.0
                weights_dict[d] = row_res
                continue

            scaled_w = (demeaned_w / l1_norm_demeaned) * max_gross

            row_res = pd.Series(np.nan, index=wide_sig.columns)
            row_res[s_valid.index] = scaled_w
            weights_dict[d] = row_res

        raw_weights_df = pd.DataFrame(weights_dict).T
        raw_weights_series = raw_weights_df.stack(future_stack=True) if hasattr(raw_weights_df, "stack") else raw_weights_df.stack(dropna=False)
        raw_weights_series = raw_weights_series.reindex(sig_s.index)

        constrained = enforce_portfolio_constraints(
            raw_weights_series,
            max_gross=max_gross,
            max_net=max_net,
            max_position=max_position,
        )
        return constrained.to_frame(name="target_weight")
