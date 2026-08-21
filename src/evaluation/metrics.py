"""Risk and performance evaluation metrics: Sharpe, Sortino, CAGR, Drawdowns, Calmar, Hit Rate, Turnover, and Cost Drag."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    """Standard 14-metric performance and risk evaluation summary (Specification Section 20)."""

    cagr: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float  # With Rf = 0.0 strictly
    sortino_ratio: float  # With Rf = 0.0 strictly
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    daily_hit_rate: float
    profit_factor: float
    average_daily_turnover: float
    annualized_turnover: float
    total_cost_usd: float
    cost_drag_bps: float
    total_trading_days: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


def compute_drawdowns(equity_curve: pd.Series) -> tuple[pd.Series, float, int]:
    """
    Compute drawdown series, maximum drawdown, and maximum drawdown duration in trading days.

    Parameters
    ----------
    equity_curve : pd.Series
        Time series of total portfolio equity.

    Returns
    -------
    tuple[pd.Series, float, int]
        (drawdown_series, max_drawdown, max_drawdown_duration_days)
    """
    if len(equity_curve) == 0:
        return pd.Series(dtype=float), 0.0, 0

    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Duration calculation
    is_zero = drawdowns == 0
    duration_series = (~is_zero).astype(int).groupby(is_zero.cumsum()).cumsum()
    max_duration = int(duration_series.max()) if len(duration_series) > 0 else 0

    return drawdowns, max_dd, max_duration


def compute_performance_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series | None = None,
    turnover_series: pd.Series | None = None,
    total_cost_usd: float = 0.0,
    gross_daily_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,  # Specification Section 20: strictly Rf = 0.0
) -> PerformanceMetrics:
    """
    Compute comprehensive 14 performance and risk metrics according to Specification Section 20.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio return time series r_t.
    equity_curve : pd.Series | None
        Daily equity curve E_t. If None, constructed by compounding daily_returns.
    turnover_series : pd.Series | None
        Daily turnover time series.
    total_cost_usd : float
        Cumulative dollar transaction costs.
    gross_daily_returns : pd.Series | None
        Uncosted (gross) daily returns for calculating cost drag.
    risk_free_rate : float
        Risk-free rate (strictly 0.0 as mandated by specification).

    Returns
    -------
    PerformanceMetrics
    """
    clean_rets = daily_returns.dropna()
    t = len(clean_rets)

    if t < 2:
        return PerformanceMetrics(
            cagr=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio=0.0,
            daily_hit_rate=0.0,
            profit_factor=0.0,
            average_daily_turnover=0.0,
            annualized_turnover=0.0,
            total_cost_usd=total_cost_usd,
            cost_drag_bps=0.0,
            total_trading_days=t,
        )

    # 1. Equity curve & Drawdowns
    if equity_curve is None:
        eq_curve = (1.0 + clean_rets).cumprod()
    else:
        eq_curve = equity_curve.dropna()

    drawdown_series, max_dd, max_duration = compute_drawdowns(eq_curve)

    # 2. Annualized Return & CAGR
    total_compound_return = (eq_curve.iloc[-1] / eq_curve.iloc[0]) - 1.0 if eq_curve.iloc[0] > 0 else 0.0
    cagr = ((1.0 + total_compound_return) ** (252.0 / t)) - 1.0 if (1.0 + total_compound_return) > 0 else -1.0
    ann_return = float(clean_rets.mean() * 252.0)

    # 3. Volatility & Sharpe Ratio (with Rf = 0.0)
    daily_std = float(clean_rets.std(ddof=1))
    ann_vol = daily_std * np.sqrt(252.0)
    sharpe = (clean_rets.mean() - risk_free_rate) / daily_std * np.sqrt(252.0) if daily_std > 1e-12 else 0.0

    # 4. Downside Deviation & Sortino Ratio (with Rf = 0.0)
    negative_returns = clean_rets[clean_rets < risk_free_rate]
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252.0) if len(negative_returns) > 0 else 1e-6
    sortino = (clean_rets.mean() - risk_free_rate) * 252.0 / downside_dev if downside_dev > 1e-12 else 0.0

    # 5. Calmar Ratio
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    # 6. Daily Hit Rate & Profit Factor
    positive_days = (clean_rets > 0).sum()
    hit_rate = float(positive_days / t) if t > 0 else 0.0

    sum_gains = clean_rets[clean_rets > 0].sum()
    sum_losses = clean_rets[clean_rets < 0].abs().sum()
    profit_factor = float(sum_gains / sum_losses) if sum_losses > 1e-12 else (100.0 if sum_gains > 0 else 0.0)

    # 7. Turnover
    avg_turnover = float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) > 0 else 0.0
    ann_turnover = avg_turnover * 252.0

    # 8. Cost Drag (bps)
    if gross_daily_returns is not None and len(gross_daily_returns) > 0:
        gross_eq = (1.0 + gross_daily_returns.dropna()).cumprod()
        gross_cagr = ((gross_eq.iloc[-1] / gross_eq.iloc[0]) ** (252.0 / len(gross_daily_returns))) - 1.0
        cost_drag_bps = (gross_cagr - cagr) * 10_000.0
    else:
        cost_drag_bps = 0.0

    return PerformanceMetrics(
        cagr=float(cagr),
        annualized_return=float(ann_return),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        max_drawdown_duration_days=int(max_duration),
        calmar_ratio=float(calmar),
        daily_hit_rate=float(hit_rate),
        profit_factor=float(profit_factor),
        average_daily_turnover=float(avg_turnover),
        annualized_turnover=float(ann_turnover),
        total_cost_usd=float(total_cost_usd),
        cost_drag_bps=float(cost_drag_bps),
        total_trading_days=int(t),
    )
