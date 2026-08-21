"""Unit tests for performance metrics, Sharpe/Sortino with Rf=0, drawdowns, and tearsheet."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    PerformanceMetrics,
    compute_drawdowns,
    compute_performance_metrics,
)
from src.evaluation.tearsheet import generate_tearsheet


def test_max_drawdown_and_duration():
    """Verify exact maximum drawdown and duration on deterministic equity curve."""
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    equity = pd.Series([100.0, 120.0, 90.0, 80.0, 110.0, 130.0], index=dates)

    dd_series, max_dd, max_dur = compute_drawdowns(equity)

    # Peak is 120, trough is 80 -> Max Drawdown is (80 - 120)/120 = -40/120 = -0.333333
    assert np.isclose(max_dd, -40.0 / 120.0, atol=1e-5)
    # Trough at 80 has drawdown -33.33%
    assert np.isclose(dd_series.loc[dates[3]], -0.333333, atol=1e-5)
    # 0 drawdown at peak 120 and new peak 130
    assert dd_series.loc[dates[1]] == 0.0
    assert dd_series.loc[dates[5]] == 0.0


def test_sharpe_and_sortino_rf_zero():
    """
    Verify Sharpe and Sortino ratios with strictly Rf = 0.0:
    Sharpe = (mean(r) / std(r)) * sqrt(252)
    Sortino = (mean(r) / downside_std(r)) * sqrt(252)
    """
    rets = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.005])
    metrics = compute_performance_metrics(rets, risk_free_rate=0.0)

    mean_r = rets.mean()
    std_r = rets.std(ddof=1)
    expected_sharpe = (mean_r / std_r) * np.sqrt(252.0)

    neg_rets = rets[rets < 0]
    downside_dev = np.sqrt(np.mean(neg_rets ** 2)) * np.sqrt(252.0)
    expected_sortino = (mean_r * 252.0) / downside_dev

    assert np.isclose(metrics.sharpe_ratio, expected_sharpe, atol=1e-4)
    assert np.isclose(metrics.sortino_ratio, expected_sortino, atol=1e-4)


def test_hit_rate_and_profit_factor():
    """Verify daily hit rate and profit factor calculations."""
    # 3 positive days (+0.01, +0.02, +0.03), 1 negative day (-0.02) -> total 4 days
    rets = pd.Series([0.01, 0.02, -0.02, 0.03])
    metrics = compute_performance_metrics(rets)

    assert metrics.daily_hit_rate == 0.75
    # Profit factor: sum(gains) / sum(losses) = (0.01 + 0.02 + 0.03) / 0.02 = 0.06 / 0.02 = 3.0
    assert np.isclose(metrics.profit_factor, 3.0, atol=1e-4)


def test_generate_tearsheet():
    """Verify tearsheet contains all required markdown metrics."""
    rets = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    metrics = compute_performance_metrics(rets, total_cost_usd=150.0)
    ts = generate_tearsheet(metrics, strategy_name="Test Alpha Strategy")

    assert "Performance & Risk Tearsheet: Test Alpha Strategy" in ts
    assert "CAGR" in ts
    assert "Sharpe Ratio ($R_f=0$)" in ts
    assert "Sortino Ratio ($R_f=0$)" in ts
    assert "Maximum Drawdown" in ts
    assert "$150.00" in ts
