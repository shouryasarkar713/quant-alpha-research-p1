"""Execution and single-pass transaction cost model: bid-ask spread, market impact slippage, and commissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class ExecutionFill:
    """Record of an executed order fill."""

    ticker: str
    direction: Literal["BUY", "SELL"]
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    spread_cost_usd: float
    slippage_cost_usd: float
    total_cost_usd: float


class CostModel:
    """
    Single-Pass Transaction Cost Accounting Model (Specification Section 18).

    Methodology (Specification Section 18.1):
    1. Effective Execution Fill Price:
       - execution_adjustment = (spread_bps + slippage_bps) / 10,000
       - For BUY:
             fill_price = reference_price * (1 + execution_adjustment)
       - For SELL:
             fill_price = reference_price * (1 - execution_adjustment)
         (Spread and slippage are absorbed directly into the fill price in a SINGLE PASS).
    2. Commissions:
       - Handled as a separate cash fee:
             commission = max(min_commission, |reference_price * quantity| * commission_bps / 10,000)

    IMPORTANT: fill_price already includes spread and slippage.
    PortfolioTracker uses fill_price for position and cash accounting and deducts commission separately.
    Spread and slippage are NOT counted a second time.
    """

    def __init__(
        self,
        commission_bps: float = 5.0,
        spread_bps: float = 5.0,
        slippage_bps: float = 5.0,
        min_commission: float = 0.0,
        config: Any | None = None,
    ) -> None:
        if config is not None:
            self.commission_bps = float(getattr(config, "commission_bps", 5.0))
            self.spread_bps = float(getattr(config, "spread_bps", 5.0))
            self.slippage_bps = float(getattr(config, "slippage_bps", 5.0))
            self.min_commission = float(getattr(config, "min_commission", 0.0))
        else:
            self.commission_bps = float(commission_bps)
            self.spread_bps = float(spread_bps)
            self.slippage_bps = float(slippage_bps)
            self.min_commission = float(min_commission)

    @classmethod
    def from_config(cls, config: Any) -> CostModel:
        """Instantiate CostModel directly from a CostConfig object or dict."""
        if isinstance(config, dict):
            return cls(
                commission_bps=config.get("commission_bps", 5.0),
                spread_bps=config.get("spread_bps", 5.0),
                slippage_bps=config.get("slippage_bps", 5.0),
                min_commission=config.get("min_commission", 0.0),
            )
        return cls(config=config)

    def calculate_fill(
        self,
        ticker: str,
        direction: Literal["BUY", "SELL"],
        quantity: float,
        reference_price: float,
        adv_20d: float | None = None,
    ) -> ExecutionFill:
        """
        Calculate single-pass fill price, transaction costs, and commissions.

        Parameters
        ----------
        ticker : str
            Security symbol.
        direction : Literal['BUY', 'SELL']
            Order direction.
        quantity : float
            Number of shares (must be > 0).
        reference_price : float
            Unadjusted market close price at execution timestamp t+1.
        adv_20d : float | None
            Optional 20-day ADV (reserved for logging / diagnostics).

        Returns
        -------
        ExecutionFill
        """
        if quantity <= 0 or reference_price <= 0:
            return ExecutionFill(
                ticker=ticker,
                direction=direction,
                quantity=0.0,
                reference_price=reference_price,
                fill_price=reference_price,
                commission=0.0,
                spread_cost_usd=0.0,
                slippage_cost_usd=0.0,
                total_cost_usd=0.0,
            )

        # 1. Single-pass execution adjustment (spread + slippage)
        execution_adjustment = (self.spread_bps + self.slippage_bps) / 10_000.0

        # 2. Fill price calculation
        if direction == "BUY":
            fill_price = reference_price * (1.0 + execution_adjustment)
        else:
            fill_price = reference_price * (1.0 - execution_adjustment)
            fill_price = max(fill_price, 0.01)  # Floor at 1 cent

        # 3. USD Cost breakdowns
        trade_value = reference_price * quantity
        spread_cost_usd = reference_price * (self.spread_bps / 10_000.0) * quantity
        slippage_cost_usd = reference_price * (self.slippage_bps / 10_000.0) * quantity

        # 4. Commission (USD)
        commission = max(self.min_commission, trade_value * (self.commission_bps / 10_000.0))
        total_cost_usd = spread_cost_usd + slippage_cost_usd + commission

        return ExecutionFill(
            ticker=ticker,
            direction=direction,
            quantity=quantity,
            reference_price=reference_price,
            fill_price=fill_price,
            commission=commission,
            spread_cost_usd=spread_cost_usd,
            slippage_cost_usd=slippage_cost_usd,
            total_cost_usd=total_cost_usd,
        )


# Predefined Cost Regimes (Specification Section 18.2)
COST_REGIMES = {
    "zero": CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0, min_commission=0.0),
    "zero_cost": CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0, min_commission=0.0),
    "low": CostModel(commission_bps=2.0, spread_bps=3.0, slippage_bps=2.0, min_commission=0.0),
    "low_cost": CostModel(commission_bps=2.0, spread_bps=3.0, slippage_bps=2.0, min_commission=0.0),
    "medium": CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=0.0),
    "base_case": CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=0.0),
    "baseline": CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=0.0),
    "high": CostModel(commission_bps=10.0, spread_bps=10.0, slippage_bps=10.0, min_commission=0.0),
    "high_cost": CostModel(commission_bps=10.0, spread_bps=10.0, slippage_bps=10.0, min_commission=0.0),
    "very_high": CostModel(commission_bps=15.0, spread_bps=15.0, slippage_bps=20.0, min_commission=0.0),
    "extreme_cost": CostModel(commission_bps=15.0, spread_bps=15.0, slippage_bps=20.0, min_commission=0.0),
}


def get_cost_model(regime: str = "medium") -> CostModel:
    """Factory to retrieve a cost model by regime name."""
    key = regime.strip().lower()
    if key not in COST_REGIMES:
        raise ValueError(f"Unknown cost regime '{regime}'. Available: {sorted(list(COST_REGIMES.keys()))}")
    return COST_REGIMES[key]
