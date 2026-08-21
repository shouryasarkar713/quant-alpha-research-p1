"""Unit tests for walk-forward expanding window validation, PSR, and DSR."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_ohlcv
from src.features.engine import compute_features
from src.signals import MeanReversionSignal
from src.validation import (
    DEFAULT_WALK_FORWARD_WINDOWS,
    WalkForwardReport,
    WalkForwardValidator,
    WalkForwardWindow,
    compute_deflated_sharpe_ratio,
    compute_probabilistic_sharpe_ratio,
)


def test_walk_forward_window_definitions():
    """Verify standard 7 expanding windows and single 2024 final holdout isolation."""
    windows = DEFAULT_WALK_FORWARD_WINDOWS
    assert len(windows) == 7

    for i in range(6):
        assert not windows[i].is_final_holdout
        assert int(windows[i].test_start[:4]) == 2018 + i

    # Window 7 must be isolated final holdout for 2024
    assert windows[6].is_final_holdout
    assert windows[6].test_start == "2024-01-01"
    assert windows[6].test_end == "2024-12-31"

    # Verify expanding training windows
    for i in range(1, 7):
        assert windows[i].train_end > windows[i - 1].train_end


def test_probabilistic_sharpe_ratio():
    """Verify PSR responds logically to strong positive vs negative returns."""
    # Strong positive strategy: Sharpe ~ 2.0
    rng = np.random.default_rng(42)
    pos_rets = rng.normal(0.001, 0.008, 500)
    psr_pos = compute_probabilistic_sharpe_ratio(pos_rets, benchmark_sharpe=0.0)
    assert psr_pos > 0.95

    # Negative strategy
    neg_rets = rng.normal(-0.001, 0.008, 500)
    psr_neg = compute_probabilistic_sharpe_ratio(neg_rets, benchmark_sharpe=0.0)
    assert psr_neg < 0.05


def test_deflated_sharpe_ratio_trial_penalty():
    """Verify DSR decreases monotonically as number of tested trials increases."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0006, 0.010, 500)

    dsr_1 = compute_deflated_sharpe_ratio(rets, num_trials=1)
    dsr_10 = compute_deflated_sharpe_ratio(rets, num_trials=10)
    dsr_100 = compute_deflated_sharpe_ratio(rets, num_trials=100)

    # Multi-trial search penalty deflates estimated significance
    assert dsr_1 >= dsr_10 >= dsr_100


def test_walk_forward_execution_on_synthetic_data():
    """Verify WalkForwardValidator runs end-to-end across configured windows."""
    df = generate_synthetic_ohlcv(
        tickers=[f"STK_{i}" for i in range(10)],
        start_date="2018-01-01",
        end_date="2021-12-31",
        seed=42,
    )
    features = compute_features(df, lag=0, include_forward_targets=False)
    signal = MeanReversionSignal().compute(features)

    # Custom 2-window test configuration
    custom_windows = [
        WalkForwardWindow(1, "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31", False),
        WalkForwardWindow(2, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", True),
    ]

    validator = WalkForwardValidator(windows=custom_windows)
    report = validator.run_validation(
        prices_df=df,
        signals_df_or_series=signal,
        features_df=features,
        evaluate_final_holdout=True,
    )

    assert isinstance(report, WalkForwardReport)
    assert len(report.window_results) == 2
    assert report.dev_oos_combined_metrics.total_trading_days > 200
    assert report.final_holdout_metrics is not None
    assert report.final_holdout_metrics.total_trading_days > 200
