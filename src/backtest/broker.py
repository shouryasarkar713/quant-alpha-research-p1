"""Simulated broker executing orders with transaction costs at next-day close."""

from __future__ import annotations

import pandas as pd

from src.backtest.events import FillEvent, OrderEvent
from src.execution.costs import CostModel


class SimulatedBroker:
    """
    Simulated Execution Broker (Specification Section 17 & 18):
    - Receives OrderEvents created at close t.
    - Executes orders at synthetic next-day close (t+1).
    - Incorporates single-pass transaction costs (spread, slippage, commission).
    - Validates price existence (cancels order if asset is halted/missing/delisted on t+1).
    """

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    def execute_order(
        self,
        order: OrderEvent,
        execution_timestamp: pd.Timestamp,
        execution_price: float,
        adv_20d: float | None = None,
    ) -> FillEvent | None:
        """
        Execute an order at the execution timestamp and price.

        Parameters
        ----------
        order : OrderEvent
            Order to execute.
        execution_timestamp : pd.Timestamp
            Timestamp t+1 of execution.
        execution_price : float
            Close price on day t+1.
        adv_20d : float
            20-day ADV for slippage modeling.

        Returns
        -------
        FillEvent | None
            Returns FillEvent if successfully executed, or None if price is invalid.
        """
        if pd.isna(execution_price) or execution_price <= 0:
            # Cannot execute on missing/delisted bar
            return None

        if order.quantity <= 0:
            return None

        fill_info = self.cost_model.calculate_fill(
            ticker=order.ticker,
            direction=order.direction,
            quantity=order.quantity,
            reference_price=execution_price,
            adv_20d=adv_20d,
        )

        return FillEvent(
            timestamp=execution_timestamp,
            ticker=order.ticker,
            direction=order.direction,
            quantity=fill_info.quantity,
            fill_price=fill_info.fill_price,
            reference_price=fill_info.reference_price,
            commission=fill_info.commission,
            spread_cost=fill_info.spread_cost_usd,
            slippage_cost=fill_info.slippage_cost_usd,
            total_cost=fill_info.total_cost_usd,
        )
