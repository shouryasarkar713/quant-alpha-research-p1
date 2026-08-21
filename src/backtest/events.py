"""Event classes for the event-driven backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import pandas as pd


@dataclass(frozen=True)
class Event:
    """Base event class."""

    timestamp: pd.Timestamp


@dataclass(frozen=True)
class MarketEvent(Event):
    """Market price update event."""

    ticker: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


@dataclass(frozen=True)
class SignalEvent(Event):
    """Trading signal generated at market close t."""

    ticker: str
    signal_value: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OrderEvent(Event):
    """Order generated at close t to rebalance portfolio at close t+1."""

    ticker: str
    direction: Literal["BUY", "SELL"]
    quantity: float
    target_weight: float
    reference_price: float


@dataclass(frozen=True)
class FillEvent(Event):
    """Execution fill event executed at market close t+1."""

    ticker: str
    direction: Literal["BUY", "SELL"]
    quantity: float
    fill_price: float
    reference_price: float
    commission: float
    spread_cost: float
    slippage_cost: float
    total_cost: float
