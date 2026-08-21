"""Unit tests for the event-driven backtesting engine and portfolio accounting."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestResult,
    EventDrivenEngine,
    PortfolioTracker,
    SimulatedBroker,
)
from src.data.loader import generate_synthetic_ohlcv
from src.execution.costs import CostModel
from src.features.engine import compute_features
from src.portfolio import EqualWeightLongShort, SignalWeightedLongShort
from src.signals import MeanReversionSignal


def test_event_driven_engine_execution_flow():
    """
    Verify full event-driven backtest lifecycle:
    Signal at t -> Order at t -> Execution Fill at t+1 close -> Mark to market -> Equity curve.
    """
    df = generate_synthetic_ohlcv(
        tickers=[f"STK_{i}" for i in range(15)],
        start_date="2020-01-01",
        end_date="2020-04-30",
        seed=42,
    )
    features = compute_features(df, lag=0, include_forward_targets=False)
    signal = MeanReversionSignal().compute(features)

    engine = EventDrivenEngine(
        initial_capital=1_000_000.0,
        cost_model=CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0),
        portfolio_constructor=EqualWeightLongShort(),
    )

    result = engine.run(prices_df=df, signals_df_or_series=signal, features_df=features)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 20
    assert result.initial_capital == 1_000_000.0

    # Target weights strictly satisfy gross <= 1.0
    target_weights = EqualWeightLongShort().construct_weights(signal, features=features)
    tw_gross = target_weights.unstack(level="ticker").abs().sum(axis=1).dropna()
    assert (tw_gross <= 1.000001).all(), f"Target gross exposure exceeded 1.0: {tw_gross.max()}"

    # Realized mark-to-market gross bounded under equity drawdowns and price drift
    assert (result.gross_exposure <= 1.30).all(), "Realized mark-to-market gross exceeded 1.30 under drawdown!"

    # Trades and fills recorded
    assert len(result.trades_df) > 0
    assert "fill_price" in result.trades_df.columns
    assert "commission" in result.trades_df.columns

    # Daily returns finite
    assert result.daily_returns.isna().sum() == 0
    assert np.isfinite(result.daily_returns).all()


def test_zero_cost_vs_base_case_cost_drag():
    """
    Verify that backtest with transaction costs achieves strictly lower net equity
    than backtest with zero costs (proving transaction costs create cost drag).
    """
    df = generate_synthetic_ohlcv(
        tickers=[f"STK_{i}" for i in range(12)],
        start_date="2020-01-01",
        end_date="2020-05-31",
        seed=42,
    )
    features = compute_features(df, lag=0, include_forward_targets=False)
    signal = MeanReversionSignal().compute(features)

    engine_zero = EventDrivenEngine(
        initial_capital=1_000_000.0,
        cost_model=CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0),
        portfolio_constructor=EqualWeightLongShort(),
    )
    res_zero = engine_zero.run(prices_df=df, signals_df_or_series=signal, features_df=features)

    engine_costs = EventDrivenEngine(
        initial_capital=1_000_000.0,
        cost_model=CostModel(commission_bps=10.0, spread_bps=10.0, slippage_bps=10.0),
        portfolio_constructor=EqualWeightLongShort(),
    )
    res_costs = engine_costs.run(prices_df=df, signals_df_or_series=signal, features_df=features)

    # Total costs in zero cost model must be 0
    assert res_zero.total_costs.sum() == 0.0
    # Total costs in cost model must be > 0
    assert res_costs.total_costs.sum() > 0.0

    # Net terminal equity under costs must be lower than zero cost
    assert res_costs.equity_curve.iloc[-1] < res_zero.equity_curve.iloc[-1]
