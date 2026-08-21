"""Portfolio state tracking, cash accounting, and order generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.backtest.events import FillEvent, OrderEvent


@dataclass
class DailyPortfolioSnapshot:
    """End-of-day portfolio accounting snapshot."""

    date: pd.Timestamp
    equity: float
    cash: float
    holdings_value: float
    gross_exposure: float
    net_exposure: float
    cash_ratio: float
    daily_return: float
    turnover: float
    commissions: float
    slippage_cost: float
    spread_cost: float
    total_cost: float
    positions: dict[str, float]
    weights: dict[str, float]


class PortfolioTracker:
    """
    Portfolio Ledger and State Tracker (Specification Section 17 & 18):
    - Maintains exact cash balance and share holdings.
    - Single-pass transaction accounting: fill price absorbs spread/slippage, commission is cash fee.
    - Generates target rebalancing orders at close t based on target weights w_{i,t}.
    - Updates holdings and marks to market at close t+1.
    """

    def __init__(self, initial_capital: float = 10_000_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # ticker -> share count
        self.history: list[DailyPortfolioSnapshot] = []
        self._prev_equity = initial_capital

    @property
    def current_equity(self) -> float:
        """Current total portfolio value (cash + marked-to-market positions)."""
        return self._prev_equity

    def generate_rebalance_orders(
        self,
        target_weights: pd.Series | dict[str, float],
        current_prices: pd.Series | dict[str, float],
        timestamp: pd.Timestamp,
    ) -> list[OrderEvent]:
        """
        Generate OrderEvents on day t to transition from current portfolio to target_weights.

        Parameters
        ----------
        target_weights : pd.Series | dict[str, float]
            Target weights w_{i,t} for active securities on date t.
        current_prices : pd.Series | dict[str, float]
            Closing prices on date t.
        timestamp : pd.Timestamp
            Signal timestamp t.

        Returns
        -------
        list[OrderEvent]
        """
        orders: list[OrderEvent] = []
        equity = self._prev_equity

        # Combine all universe tickers (both target and currently held)
        all_tickers = set(target_weights.keys()) if isinstance(target_weights, dict) else set(target_weights.index)
        all_tickers.update(self.positions.keys())

        for ticker in all_tickers:
            t_weight = float(target_weights.get(ticker, 0.0)) if not pd.isna(target_weights.get(ticker, 0.0)) else 0.0
            price = float(current_prices.get(ticker, np.nan))

            if pd.isna(price) or price <= 0:
                continue

            current_shares = self.positions.get(ticker, 0.0)
            target_dollar = t_weight * equity
            target_shares = target_dollar / price

            delta_shares = target_shares - current_shares

            # Filter tiny dust orders (< $10 notional)
            if abs(delta_shares * price) < 10.0:
                continue

            direction = "BUY" if delta_shares > 0 else "SELL"
            qty = abs(delta_shares)

            orders.append(
                OrderEvent(
                    timestamp=timestamp,
                    ticker=ticker,
                    direction=direction,
                    quantity=qty,
                    target_weight=t_weight,
                    reference_price=price,
                )
            )

        return orders

    def update_from_fills(self, fills: list[FillEvent]) -> tuple[float, float, float, float]:
        """
        Apply execution fills to cash and positions in single-pass ledger.

        Returns
        -------
        tuple[float, float, float, float]
            (commissions, slippage_cost, spread_cost, total_cost)
        """
        commissions = 0.0
        slippage_cost = 0.0
        spread_cost = 0.0

        for fill in fills:
            notional = fill.fill_price * fill.quantity

            if fill.direction == "BUY":
                # Cash outflows: fill cost + commission
                self.cash -= (notional + fill.commission)
                self.positions[fill.ticker] = self.positions.get(fill.ticker, 0.0) + fill.quantity
            else:
                # Cash inflows: fill proceeds - commission
                self.cash += (notional - fill.commission)
                self.positions[fill.ticker] = self.positions.get(fill.ticker, 0.0) - fill.quantity

            commissions += fill.commission
            slippage_cost += fill.slippage_cost
            spread_cost += fill.spread_cost

            # Clean zero/dust holdings
            if abs(self.positions.get(fill.ticker, 0.0)) < 1e-6:
                self.positions.pop(fill.ticker, None)

        total_cost = commissions + slippage_cost + spread_cost
        return (commissions, slippage_cost, spread_cost, total_cost)

    def mark_to_market(
        self,
        date: pd.Timestamp,
        prices: pd.Series | dict[str, float],
        fills: list[FillEvent] | None = None,
    ) -> DailyPortfolioSnapshot:
        """
        Mark portfolio holdings to market at day t+1 close and record daily snapshot.

        Parameters
        ----------
        date : pd.Timestamp
            Market close timestamp t+1.
        prices : pd.Series | dict[str, float]
            Closing prices on day t+1.
        fills : list[FillEvent] | None
            Fills executed on day t+1.

        Returns
        -------
        DailyPortfolioSnapshot
        """
        fills = fills or []
        comm, slip, sprd, tot_cost = self.update_from_fills(fills)

        holdings_val = 0.0
        long_val = 0.0
        short_val = 0.0
        weights: dict[str, float] = {}

        # Mark positions to market
        for ticker, shares in list(self.positions.items()):
            p = float(prices.get(ticker, np.nan))
            if pd.isna(p) or p <= 0:
                # If delisted/liquidated, treat position as zero value or cash
                continue
            pos_val = shares * p
            holdings_val += pos_val
            if pos_val > 0:
                long_val += pos_val
            else:
                short_val += abs(pos_val)

        equity = self.cash + holdings_val

        # Weights and exposures
        if equity > 0:
            for ticker, shares in self.positions.items():
                p = float(prices.get(ticker, 0.0))
                weights[ticker] = (shares * p) / equity
            gross_exp = (long_val + short_val) / equity
            net_exp = (long_val - short_val) / equity
            cash_ratio = self.cash / equity
        else:
            gross_exp = 0.0
            net_exp = 0.0
            cash_ratio = 1.0

        daily_ret = (equity - self._prev_equity) / self._prev_equity if self._prev_equity > 0 else 0.0

        # Turnover = total traded notional / average equity
        traded_notional = sum(f.fill_price * f.quantity for f in fills)
        turnover = traded_notional / equity if equity > 0 else 0.0

        snapshot = DailyPortfolioSnapshot(
            date=date,
            equity=equity,
            cash=self.cash,
            holdings_value=holdings_val,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            cash_ratio=cash_ratio,
            daily_return=daily_ret,
            turnover=turnover,
            commissions=comm,
            slippage_cost=slip,
            spread_cost=sprd,
            total_cost=tot_cost,
            positions=self.positions.copy(),
            weights=weights,
        )

        self.history.append(snapshot)
        self._prev_equity = equity
        return snapshot
