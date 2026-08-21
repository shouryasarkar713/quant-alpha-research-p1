"""Performance tearsheet generation and formatted summary tables."""

from __future__ import annotations

import pandas as pd
from src.evaluation.metrics import PerformanceMetrics


def generate_tearsheet(
    metrics: PerformanceMetrics,
    strategy_name: str = "Quantitative Strategy",
    benchmark_metrics: PerformanceMetrics | None = None,
) -> str:
    """
    Generate a formatted Markdown tearsheet summarizing performance metrics.

    Parameters
    ----------
    metrics : PerformanceMetrics
        Primary strategy performance metrics.
    strategy_name : str
        Name of strategy.
    benchmark_metrics : PerformanceMetrics | None
        Optional benchmark metrics for side-by-side comparison.

    Returns
    -------
    str
        Markdown tearsheet text.
    """
    bm = benchmark_metrics
    lines = [
        f"## Performance & Risk Tearsheet: {strategy_name}",
        "",
        "| Metric | Strategy Value | Benchmark Value |",
        "| :--- | :--- | :--- |",
        f"| **CAGR (Annualized Compound Return)** | {metrics.cagr:.2%} | {f'{bm.cagr:.2%}' if bm else 'N/A'} |",
        f"| **Annualized Volatility** | {metrics.annualized_volatility:.2%} | {f'{bm.annualized_volatility:.2%}' if bm else 'N/A'} |",
        f"| **Sharpe Ratio ($R_f=0$)** | {metrics.sharpe_ratio:.3f} | {f'{bm.sharpe_ratio:.3f}' if bm else 'N/A'} |",
        f"| **Sortino Ratio ($R_f=0$)** | {metrics.sortino_ratio:.3f} | {f'{bm.sortino_ratio:.3f}' if bm else 'N/A'} |",
        f"| **Maximum Drawdown** | {metrics.max_drawdown:.2%} | {f'{bm.max_drawdown:.2%}' if bm else 'N/A'} |",
        f"| **Max Drawdown Duration** | {metrics.max_drawdown_duration_days} days | {f'{bm.max_drawdown_duration_days} days' if bm else 'N/A'} |",
        f"| **Calmar Ratio** | {metrics.calmar_ratio:.3f} | {f'{bm.calmar_ratio:.3f}' if bm else 'N/A'} |",
        f"| **Daily Hit Rate (Win Rate)** | {metrics.daily_hit_rate:.2%} | {f'{bm.daily_hit_rate:.2%}' if bm else 'N/A'} |",
        f"| **Profit Factor** | {metrics.profit_factor:.3f} | {f'{bm.profit_factor:.3f}' if bm else 'N/A'} |",
        f"| **Daily Turnover (Mean)** | {metrics.average_daily_turnover:.2%} | {f'{bm.average_daily_turnover:.2%}' if bm else 'N/A'} |",
        f"| **Annualized Turnover** | {metrics.annualized_turnover:.1f}x | {f'{bm.annualized_turnover:.1f}x' if bm else 'N/A'} |",
        f"| **Total Transaction Costs (USD)** | ${metrics.total_cost_usd:,.2f} | N/A |",
        f"| **Cost Drag** | {metrics.cost_drag_bps:.1f} bps | N/A |",
        f"| **Total Trading Days** | {metrics.total_trading_days} | {f'{bm.total_trading_days}' if bm else 'N/A'} |",
        "",
    ]
    return "\n".join(lines)
