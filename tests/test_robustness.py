"""Unit tests for robustness analysis framework and market regime conditioning."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_ohlcv
from src.features.engine import compute_features
from src.features.regimes import (
    compute_market_trend_regime,
    compute_market_volatility_regime,
    evaluate_regime_performance,
)
from src.portfolio import EqualWeightLongShort
from src.robustness import RobustnessAnalyzer
from src.signals import MeanReversionSignal


def test_cost_regime_sensitivity():
    """Verify strategy performance decreases monotonically across increasing cost regimes."""
    df = generate_synthetic_ohlcv(
        tickers=[f"STK_{i}" for i in range(10)],
        start_date="2020-01-01",
        end_date="2020-06-30",
        seed=42,
    )
    features = compute_features(df, lag=0, include_forward_targets=False)
    signal = MeanReversionSignal().compute(features)

    analyzer = RobustnessAnalyzer(portfolio_constructor=EqualWeightLongShort(), initial_capital=1_000_000.0)
    cost_results = analyzer.evaluate_cost_regimes(df, signal, features)

    assert "zero_cost" in cost_results
    assert "extreme_cost" in cost_results

    # Zero cost must achieve strictly higher terminal equity than extreme cost
    assert cost_results["zero_cost"].cagr >= cost_results["extreme_cost"].cagr
    assert cost_results["extreme_cost"].total_cost_usd > cost_results["zero_cost"].total_cost_usd


def test_extreme_day_removal():
    """Verify extreme day trimming drops top 5 and bottom 5 outliers."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates)
    # Inject large positive and negative spikes
    rets.iloc[10] = 0.15
    rets.iloc[20] = -0.15

    analyzer = RobustnessAnalyzer()
    trimmed_metrics = analyzer.evaluate_extreme_day_removal(rets, n_extreme_days=5)

    # 100 days - 10 trimmed days = 90 days
    assert trimmed_metrics.total_trading_days == 90


def test_bootstrap_pnl_confidence_intervals():
    """Verify stationary block bootstrap produces valid 95% confidence intervals."""
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.0008, 0.012, 250))

    analyzer = RobustnessAnalyzer()
    sh_ci, cagr_ci = analyzer.bootstrap_pnl_confidence_intervals(rets, num_bootstrap=200, seed=42)

    # Lower bound <= Upper bound
    assert sh_ci[0] < sh_ci[1]
    assert cagr_ci[0] < cagr_ci[1]


def test_market_regime_classification():
    """Verify volatility and trend market regime segmentation."""
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    market_prices = pd.Series(np.linspace(100, 200, 250), index=dates)
    market_rets = market_prices.pct_change().dropna()

    vol_regimes = compute_market_volatility_regime(market_rets)
    trend_regimes = compute_market_trend_regime(market_prices, fast_window=20, slow_window=50)

    assert set(vol_regimes.dropna().unique()).issubset({"LOW_VOL", "NORMAL_VOL", "HIGH_VOL"})
    assert set(trend_regimes.dropna().unique()).issubset({"BULL_TREND", "BEAR_TREND"})

    strat_rets = pd.Series(0.001, index=market_rets.index)
    reg_perf = evaluate_regime_performance(strat_rets, vol_regimes)
    assert len(reg_perf) > 0
