"""Unit tests for machine learning forecasting models and comparative IC HAC testing."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_ohlcv
from src.features.engine import compute_features
from src.features.returns import forward_return
from src.ml import (
    LassoModel,
    MLAlphaPipeline,
    OLSModel,
    RidgeModel,
    XGBoostModel,
    get_ml_model,
)
from src.signals import CombinedSignal


def test_ml_models_fit_and_predict():
    """Verify OLS, Ridge, Lasso, and XGBoost fit and predict correctly."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    beta = np.array([0.5, -0.3, 0.2, 0.0])
    y = X @ beta + rng.normal(scale=0.1, size=100)

    for name in ["ols", "ridge", "lasso", "xgboost"]:
        model = get_ml_model(name)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert np.isfinite(preds).all()


def test_ml_pipeline_signal_generation_and_comparison():
    """Verify MLAlphaPipeline trains, ranks signals to [-1, 1], and compares against baseline."""
    # Generate multi-year dataset to ensure sufficient warmup for 12-1 momentum
    df_full = generate_synthetic_ohlcv(
        tickers=[f"STK_{i}" for i in range(12)],
        start_date="2016-01-01",
        end_date="2020-12-31",
        seed=42,
    )
    feat_full = compute_features(df_full, lag=0, include_forward_targets=True)

    # Slice train and test from continuously warmed-up feature panel
    dates = feat_full.index.get_level_values("date")
    feat_train = feat_full[(dates >= "2017-01-01") & (dates <= "2019-12-31")]
    feat_test = feat_full[dates >= "2020-01-01"]

    pipeline = MLAlphaPipeline(
        feature_columns=["ret_12_1_mom", "zscore_price_20", "realized_vol_20", "volume_relative_20"],
        target_col="fwd_ret_20d",
        target_horizon=20,
    )

    # Train XGBoost and generate out-of-sample signals
    ml_signals = pipeline.train_and_predict("xgboost", feat_train, feat_test)

    # Bounded in [-1, 1]
    assert ml_signals.min() >= -1.000001
    assert ml_signals.max() <= 1.000001

    # Statistical baseline combined signal
    base_signals = CombinedSignal().compute(feat_test)
    fwd_20d = feat_test["fwd_ret_20d"]

    # Comparative evaluation
    comp_res = pipeline.compare_with_baseline(
        ml_signals=ml_signals,
        baseline_signals=base_signals,
        forward_returns=fwd_20d,
        model_name="XGBoost",
    )

    assert comp_res.model_name == "XGBoost"
    assert comp_res.target_horizon_days == 20
    assert np.isfinite(comp_res.ic_diff_mean)
    assert np.isfinite(comp_res.ic_diff_hac_t_stat)
    assert 0.0 <= comp_res.ic_diff_hac_p_value <= 1.0
    assert len(comp_res.ic_diff_series) > 50
