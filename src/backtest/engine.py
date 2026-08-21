"""Event-driven backtesting execution engine coordinating market, signal, order, and fill events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from src.backtest.broker import SimulatedBroker
from src.backtest.events import FillEvent, OrderEvent
from src.backtest.portfolio import DailyPortfolioSnapshot, PortfolioTracker
from src.execution.costs import CostModel, get_cost_model
from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.equal_weight import EqualWeightLongShort
from src.signals.base import BaseSignal


@dataclass
class BacktestResult:
    """Complete structured output from an event-driven backtest simulation."""

    initial_capital: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    cash_ratio: pd.Series
    turnover: pd.Series
    total_costs: pd.Series
    commissions: pd.Series
    slippage_costs: pd.Series
    spread_costs: pd.Series
    positions_history: pd.DataFrame
    weights_history: pd.DataFrame
    trades_df: pd.DataFrame
    snapshots: list[DailyPortfolioSnapshot]


class EventDrivenEngine:
    """
    Event-Driven Backtesting Engine (Specification Section 17, 18, 19):

    Execution Lifecycle:
    1. Signal Generation (t close):
       - Observable features and signals computed at close t.
       - Portfolio constructor maps signals to target weights w_{i,t}.
       - PortfolioTracker compares target allocation with current holdings to emit OrderEvents.
    2. Execution (t+1 close):
       - SimulatedBroker fills OrderEvents at next-day close price t+1 using single-pass cost model.
       - FillEvents update cash and share positions.
       - Portfolio marked to market at close t+1.
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        cost_model: CostModel | None = None,
        portfolio_constructor: BasePortfolioConstructor | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.portfolio_constructor = portfolio_constructor or EqualWeightLongShort()

    def run(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
        rebalance_frequency: int = 1,
    ) -> BacktestResult:
        """
        Execute full event-driven simulation across date history.

        Parameters
        ----------
        prices_df : pd.DataFrame
            MultiIndex (date, ticker) with 'close' / 'adj_close' and 'volume_split_adjusted'.
        signals_df_or_series : pd.DataFrame | pd.Series
            MultiIndex (date, ticker) with trading signal values.
        features_df : pd.DataFrame | None
            Pre-computed feature panel.
        rebalance_frequency : int
            Rebalance interval in trading days (default 1 for daily).

        Returns
        -------
        BacktestResult
        """
        price_col = "adj_close" if "adj_close" in prices_df.columns else "close"
        prices_wide = prices_df[price_col].unstack(level="ticker")

        # Causal ADV for diagnostic / liquidity logging (trailing 20-day historical window)
        if "volume_sma_20" in prices_df.columns:
            adv_wide = prices_df["volume_sma_20"].unstack(level="ticker")
        elif "volume_split_adjusted" in prices_df.columns:
            adv_wide = (
                prices_df["volume_split_adjusted"]
                .unstack(level="ticker")
                .rolling(20, min_periods=1)
                .mean()
            )
        elif "volume" in prices_df.columns:
            adv_wide = (
                prices_df["volume"]
                .unstack(level="ticker")
                .rolling(20, min_periods=1)
                .mean()
            )
        else:
            adv_wide = None

        if isinstance(signals_df_or_series, pd.DataFrame):
            sig_s = signals_df_or_series.iloc[:, 0]
        else:
            sig_s = signals_df_or_series.copy()

        # Construct target weights for all dates
        weights_df = self.portfolio_constructor.construct_weights(sig_s, features=features_df)
        weights_wide = weights_df.iloc[:, 0].unstack(level="ticker")

        dates = prices_wide.index.unique().sort_values()
        n_dates = len(dates)

        tracker = PortfolioTracker(initial_capital=self.initial_capital)
        broker = SimulatedBroker(cost_model=self.cost_model)

        pending_orders: list[OrderEvent] = []
        all_fills: list[FillEvent] = []

        for i in range(n_dates):
            current_date = dates[i]
            current_prices = prices_wide.loc[current_date]
            current_adv = adv_wide.loc[current_date] if adv_wide is not None else None

            # 1. Execute pending orders from previous day t-1 at today's close t
            todays_fills: list[FillEvent] = []
            if pending_orders:
                for order in pending_orders:
                    p = current_prices.get(order.ticker, np.nan)
                    adv = current_adv.get(order.ticker, None) if current_adv is not None else None
                    fill = broker.execute_order(
                        order=order,
                        execution_timestamp=current_date,
                        execution_price=p,
                        adv_20d=adv,
                    )
                    if fill is not None:
                        todays_fills.append(fill)
                        all_fills.append(fill)
                pending_orders = []

            # 2. Mark portfolio to market at today's close
            snapshot = tracker.mark_to_market(
                date=current_date,
                prices=current_prices,
                fills=todays_fills,
            )

            # 3. If not last day, generate new rebalance orders at today's close for execution at t+1
            if i < n_dates - 1 and (i % rebalance_frequency == 0):
                if current_date in weights_wide.index:
                    target_w = weights_wide.loc[current_date].dropna()
                    pending_orders = tracker.generate_rebalance_orders(
                        target_weights=target_w,
                        current_prices=current_prices,
                        timestamp=current_date,
                    )

        # Assemble structured BacktestResult
        snapshots = tracker.history
        dates_idx = [s.date for s in snapshots]

        equity_curve = pd.Series([s.equity for s in snapshots], index=dates_idx, name="equity")
        daily_returns = pd.Series([s.daily_return for s in snapshots], index=dates_idx, name="daily_return")
        gross_exp = pd.Series([s.gross_exposure for s in snapshots], index=dates_idx, name="gross_exposure")
        net_exp = pd.Series([s.net_exposure for s in snapshots], index=dates_idx, name="net_exposure")
        cash_ratio = pd.Series([s.cash_ratio for s in snapshots], index=dates_idx, name="cash_ratio")
        turnover = pd.Series([s.turnover for s in snapshots], index=dates_idx, name="turnover")
        total_costs = pd.Series([s.total_cost for s in snapshots], index=dates_idx, name="total_costs")
        commissions = pd.Series([s.commissions for s in snapshots], index=dates_idx, name="commissions")
        slippage_costs = pd.Series([s.slippage_cost for s in snapshots], index=dates_idx, name="slippage_costs")
        spread_costs = pd.Series([s.spread_cost for s in snapshots], index=dates_idx, name="spread_costs")

        positions_df = pd.DataFrame([s.positions for s in snapshots], index=dates_idx).fillna(0.0)
        weights_hist_df = pd.DataFrame([s.weights for s in snapshots], index=dates_idx).fillna(0.0)

        trades_records = [
            {
                "timestamp": f.timestamp,
                "ticker": f.ticker,
                "direction": f.direction,
                "quantity": f.quantity,
                "fill_price": f.fill_price,
                "reference_price": f.reference_price,
                "commission": f.commission,
                "spread_cost": f.spread_cost,
                "slippage_cost": f.slippage_cost,
                "total_cost": f.total_cost,
            }
            for f in all_fills
        ]
        trades_df = pd.DataFrame(trades_records)

        return BacktestResult(
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            cash_ratio=cash_ratio,
            turnover=turnover,
            total_costs=total_costs,
            commissions=commissions,
            slippage_costs=slippage_costs,
            spread_costs=spread_costs,
            positions_history=positions_df,
            weights_history=weights_hist_df,
            trades_df=trades_df,
            snapshots=snapshots,
        )
