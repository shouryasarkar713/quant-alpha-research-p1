"""CLI script to run individual signal/portfolio backtests with custom parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.backtest.engine import EventDrivenEngine
from src.config.schema import load_config
from src.evaluation.metrics import compute_performance_metrics
from src.evaluation.tearsheet import generate_tearsheet
from src.execution.costs import CostModel, get_cost_model
from src.features.engine import compute_features
from src.portfolio import get_portfolio_constructor
from src.signals import get_signal


def run_custom_backtest(
    config_path: str | None = None,
    signal_name: str = "combined_signal",
    portfolio_name: str = "equal_weight_long_short",
    cost_regime: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 10_000_000.0,
    data_path: str = "data/processed/cleaned_ohlcv.parquet",
) -> None:
    """Run an isolated backtest configuration and print tearsheet."""
    if config_path:
        print(f"Loading experiment configuration from {config_path}...")
        cfg = load_config(config_path)
        cost_model = CostModel.from_config(cfg.costs)
        initial_capital = cfg.backtest.initial_cash
        if not signal_name or signal_name == "combined_signal":
            signal_name = cfg.signal.name
    elif cost_regime:
        cost_model = get_cost_model(cost_regime)
    else:
        cost_model = CostModel()

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    dates = df.index.get_level_values("date")
    if start_date:
        df = df[dates >= start_date]
    if end_date:
        dates = df.index.get_level_values("date")
        df = df[dates <= end_date]

    print("Computing features...")
    features_df = compute_features(df, lag=0, include_forward_targets=False)

    print(f"Generating signal '{signal_name}'...")
    sig_obj = get_signal(signal_name)
    signals = sig_obj.compute(features_df)

    print(f"Initializing portfolio '{portfolio_name}' with cost model ({cost_model.commission_bps} comm, {cost_model.spread_bps} spr, {cost_model.slippage_bps} slip)...")
    constructor = get_portfolio_constructor(portfolio_name)

    engine = EventDrivenEngine(
        initial_capital=initial_capital,
        cost_model=cost_model,
        portfolio_constructor=constructor,
    )

    print("Executing event-driven simulation...")
    result = engine.run(df, signals, features_df)

    metrics = compute_performance_metrics(
        daily_returns=result.daily_returns,
        equity_curve=result.equity_curve,
        turnover_series=result.turnover,
        total_cost_usd=float(result.total_costs.sum()),
    )

    print("\n" + generate_tearsheet(metrics, strategy_name=f"{signal_name} ({portfolio_name})"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single quantitative strategy backtest.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML experiment config.")
    parser.add_argument("--signal", type=str, default="combined_signal")
    parser.add_argument("--portfolio", type=str, default="equal_weight_long_short")
    parser.add_argument("--cost-regime", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=10_000_000.0)
    parser.add_argument("--data-path", type=str, default="data/processed/cleaned_ohlcv.parquet")
    args = parser.parse_args()

    run_custom_backtest(
        config_path=args.config,
        signal_name=args.signal,
        portfolio_name=args.portfolio,
        cost_regime=args.cost_regime,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        data_path=args.data_path,
    )
