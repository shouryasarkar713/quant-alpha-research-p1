"""Machine learning training and IC comparative inference pipeline (Specification Section 24)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ml.models import BaseMLModel, get_ml_model
from src.statistics.hypothesis_tests import hac_t_test
from src.statistics.information_coefficient import compute_ic


@dataclass
class MLComparisonResult:
    """Statistical comparison of an ML model vs baseline statistical combination."""

    model_name: str
    target_horizon_days: int
    ml_mean_ic: float
    ml_ic_std: float
    ml_ic_ir: float
    ml_hac_t_stat: float
    ml_hac_p_value: float
    baseline_mean_ic: float
    baseline_ic_ir: float
    ic_diff_mean: float
    ic_diff_hac_t_stat: float
    ic_diff_hac_p_value: float  # Test H0: E[IC_ML - IC_base] = 0
    ml_daily_ic_series: pd.Series
    baseline_daily_ic_series: pd.Series
    ic_diff_series: pd.Series


class MLAlphaPipeline:
    """
    ML Alpha Comparison Pipeline (Specification Section 24):
    - Target: Common 20-day forward return (fwd_ret_20d).
    - Standardizes features in-sample (zero look-ahead).
    - Produces cross-sectionally ranked [-1, 1] signals out-of-sample.
    - Evaluates daily Spearman IC series.
    - Runs Newey-West HAC test on daily IC difference series (Delta IC_t = IC_t^ML - IC_t^Base).
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        target_col: str = "fwd_ret_20d",
        target_horizon: int = 20,
    ) -> None:
        self.feature_columns = feature_columns or [
            "ret_12_1_mom",
            "zscore_price_20",
            "realized_vol_20",
            "volume_relative_20",
        ]
        self.target_col = target_col
        self.target_horizon = target_horizon

    def prepare_dataset(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract valid feature matrix X and target y."""
        available_cols = [c for c in self.feature_columns if c in features_df.columns]
        if not available_cols:
            raise ValueError(f"None of requested feature columns {self.feature_columns} found in dataframe.")

        if self.target_col not in features_df.columns:
            from src.features.returns import forward_return
            price_col = "adj_close" if "adj_close" in features_df.columns else "close"
            fwd_s = forward_return(features_df[[price_col]], horizon=self.target_horizon).iloc[:, 0]
            features_df = features_df.copy()
            features_df[self.target_col] = fwd_s

        valid_mask = features_df[available_cols].notna().all(axis=1) & features_df[self.target_col].notna()
        df_valid = features_df[valid_mask]
        X = df_valid[available_cols]
        y = df_valid[self.target_col]
        return X, y

    def train_and_predict(
        self,
        model_name: str,
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        model_kwargs: dict[str, Any] | None = None,
    ) -> pd.Series:
        """
        Train ML model on features_train and generate cross-sectionally ranked signals on features_test.

        Returns
        -------
        pd.Series
            MultiIndex (date, ticker) with [-1, 1] bounded alpha signals.
        """
        model_kwargs = model_kwargs or {}
        X_train, y_train = self.prepare_dataset(features_train)
        X_test, _ = self.prepare_dataset(features_test)

        available_cols = X_train.columns.tolist()

        # In-sample standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test[available_cols])

        # Fit model
        model = get_ml_model(model_name, **model_kwargs)
        model.fit(X_train_scaled, y_train.values)

        # Raw continuous predictions
        raw_preds = model.predict(X_test_scaled)
        preds_series = pd.Series(raw_preds, index=X_test.index, name="ml_pred")

        # Cross-sectional ranking to [-1, 1] per date
        def _rank_date(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            n = len(valid)
            if n < 2:
                return pd.Series(0.0, index=row.index)
            ranks = valid.rank(ascending=True, method="average")
            norm_ranks = (ranks - 1.0) / (n - 1.0)
            return (norm_ranks - 0.5) * 2.0

        wide_preds = preds_series.unstack(level="ticker")
        ranked_wide = wide_preds.apply(_rank_date, axis=1)
        ranked_series = ranked_wide.stack(future_stack=True) if hasattr(ranked_wide, "stack") else ranked_wide.stack(dropna=False)
        ranked_series = ranked_series.reindex(features_test.index).fillna(0.0)
        ranked_series.name = f"{model_name}_signal"
        return ranked_series

    def compare_with_baseline(
        self,
        ml_signals: pd.Series,
        baseline_signals: pd.Series,
        forward_returns: pd.Series,
        model_name: str = "XGBoost",
    ) -> MLComparisonResult:
        """
        Perform rigorous statistical comparison between ML model and statistical baseline:
        1. Daily Spearman IC series for ML model and baseline.
        2. Daily IC difference series: Delta IC_t = IC_t^ML - IC_t^Base.
        3. Newey-West HAC inference on Delta IC_t with lag L = max(1, horizon).
        """
        # Daily IC series
        ml_ic = compute_ic(ml_signals, forward_returns, method="spearman").dropna()
        base_ic = compute_ic(baseline_signals, forward_returns, method="spearman").dropna()

        # Align on common trading dates
        aligned = pd.concat([ml_ic.rename("ml"), base_ic.rename("base")], axis=1).dropna()
        aligned["diff"] = aligned["ml"] - aligned["base"]

        # HAC tests with lag L = max(1, horizon)
        hac_lag = max(1, self.target_horizon)

        ml_mean = float(aligned["ml"].mean())
        ml_std = float(aligned["ml"].std(ddof=1))
        ml_ir = ml_mean / ml_std if ml_std > 1e-12 else 0.0
        ml_t, ml_p = hac_t_test(aligned["ml"], max_lag=hac_lag)

        base_mean = float(aligned["base"].mean())
        base_std = float(aligned["base"].std(ddof=1))
        base_ir = base_mean / base_std if base_std > 1e-12 else 0.0

        diff_mean = float(aligned["diff"].mean())
        diff_t, diff_p = hac_t_test(aligned["diff"], max_lag=hac_lag)

        return MLComparisonResult(
            model_name=model_name,
            target_horizon_days=self.target_horizon,
            ml_mean_ic=ml_mean,
            ml_ic_std=ml_std,
            ml_ic_ir=ml_ir,
            ml_hac_t_stat=ml_t,
            ml_hac_p_value=ml_p,
            baseline_mean_ic=base_mean,
            baseline_ic_ir=base_ir,
            ic_diff_mean=diff_mean,
            ic_diff_hac_t_stat=diff_t,
            ic_diff_hac_p_value=diff_p,
            ml_daily_ic_series=aligned["ml"],
            baseline_daily_ic_series=aligned["base"],
            ic_diff_series=aligned["diff"],
        )
