"""Comprehensive robustness analysis suite: parameter sensitivity, sub-periods, asset-drop stability, cost regimes, extreme day removal, and bootstrap PnL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, EventDrivenEngine
from src.evaluation.metrics import PerformanceMetrics, compute_performance_metrics
from src.execution.costs import COST_REGIMES, CostModel, get_cost_model
from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.equal_weight import EqualWeightLongShort
from src.signals.base import BaseSignal
from src.statistics.hypothesis_tests import bootstrap_mean_ci


@dataclass
class RobustnessReport:
    """Consolidated results of all robustness checks (Specification Section 22)."""

    cost_regime_results: dict[str, PerformanceMetrics]
    subperiod_results: dict[str, PerformanceMetrics]
    asset_drop_results: list[PerformanceMetrics]
    extreme_days_removed_metrics: PerformanceMetrics
    bootstrap_sharpe_ci: tuple[float, float]
    bootstrap_cagr_ci: tuple[float, float]


class RobustnessAnalyzer:
    """
    Robustness Analysis Framework (Specification Section 22):
    1. Transaction Cost Sensitivity (5 regimes).
    2. Sub-period split (Pre-2020 vs Post-2020).
    3. Asset-Drop Stability (random 10% universe exclusion jackknife).
    4. Extreme Day Removal (dropping top 5 and bottom 5 PnL days).
    5. Stationary Bootstrap of PnL for 95% confidence intervals.
    """

    def __init__(
        self,
        portfolio_constructor: BasePortfolioConstructor | None = None,
        initial_capital: float = 10_000_000.0,
    ) -> None:
        self.portfolio_constructor = portfolio_constructor or EqualWeightLongShort()
        self.initial_capital = initial_capital

    def evaluate_cost_regimes(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
    ) -> dict[str, PerformanceMetrics]:
        """Evaluate strategy across all 5 standard cost regimes."""
        results = {}
        for name, cm in COST_REGIMES.items():
            engine = EventDrivenEngine(
                initial_capital=self.initial_capital,
                cost_model=cm,
                portfolio_constructor=self.portfolio_constructor,
            )
            res = engine.run(prices_df, signals_df_or_series, features_df)
            metrics = compute_performance_metrics(
                daily_returns=res.daily_returns,
                equity_curve=res.equity_curve,
                turnover_series=res.turnover,
                total_cost_usd=float(res.total_costs.sum()),
            )
            results[name] = metrics
        return results

    def evaluate_subperiods(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
        split_date: str = "2020-01-01",
    ) -> dict[str, PerformanceMetrics]:
        """Evaluate strategy on pre-split and post-split historical periods."""
        engine = EventDrivenEngine(
            initial_capital=self.initial_capital,
            cost_model=get_cost_model("base_case"),
            portfolio_constructor=self.portfolio_constructor,
        )

        dates = prices_df.index.get_level_values("date")
        pre_mask = dates < split_date
        post_mask = dates >= split_date

        res_pre = engine.run(
            prices_df[pre_mask],
            signals_df_or_series[pre_mask],
            features_df[pre_mask] if features_df is not None else None,
        )
        res_post = engine.run(
            prices_df[post_mask],
            signals_df_or_series[post_mask],
            features_df[post_mask] if features_df is not None else None,
        )

        return {
            f"pre_{split_date[:4]}": compute_performance_metrics(res_pre.daily_returns, res_pre.equity_curve),
            f"post_{split_date[:4]}": compute_performance_metrics(res_post.daily_returns, res_post.equity_curve),
        }

    def evaluate_asset_drop_stability(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
        drop_pct: float = 0.10,
        iterations: int = 5,
        seed: int = 42,
    ) -> list[PerformanceMetrics]:
        """Jackknife / random 10% asset exclusion to verify stability."""
        rng = np.random.default_rng(seed)
        all_tickers = prices_df.index.get_level_values("ticker").unique().to_numpy()
        n_drop = max(1, int(len(all_tickers) * drop_pct))

        engine = EventDrivenEngine(
            initial_capital=self.initial_capital,
            cost_model=get_cost_model("base_case"),
            portfolio_constructor=self.portfolio_constructor,
        )

        results = []
        for _ in range(iterations):
            dropped = rng.choice(all_tickers, size=n_drop, replace=False)
            keep_tickers = set(all_tickers) - set(dropped)

            mask = prices_df.index.get_level_values("ticker").isin(keep_tickers)
            p_sub = prices_df[mask]
            s_sub = signals_df_or_series[mask]
            f_sub = features_df[mask] if features_df is not None else None

            res = engine.run(p_sub, s_sub, f_sub)
            results.append(compute_performance_metrics(res.daily_returns, res.equity_curve))

        return results

    def evaluate_extreme_day_removal(
        self,
        daily_returns: pd.Series,
        n_extreme_days: int = 5,
    ) -> PerformanceMetrics:
        """Drop top N best and bottom N worst PnL trading days to evaluate tail reliance."""
        clean_rets = daily_returns.dropna().copy()
        if len(clean_rets) <= 2 * n_extreme_days:
            return compute_performance_metrics(clean_rets)

        top_days = clean_rets.nlargest(n_extreme_days).index
        bottom_days = clean_rets.nsmallest(n_extreme_days).index
        drop_indices = top_days.union(bottom_days)

        trimmed_rets = clean_rets.drop(drop_indices)
        return compute_performance_metrics(trimmed_rets)

    def bootstrap_pnl_confidence_intervals(
        self,
        daily_returns: pd.Series,
        num_bootstrap: int = 1000,
        expected_block_size: int = 10,
        seed: int = 42,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute stationary block bootstrap 95% confidence intervals for Sharpe ratio and CAGR.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            ((sharpe_lower, sharpe_upper), (cagr_lower, cagr_upper))
        """
        clean_rets = daily_returns.dropna().to_numpy()
        t = len(clean_rets)
        if t < 10:
            return ((0.0, 0.0), (0.0, 0.0))

        # Stationary bootstrap resampling
        p_geom = 1.0 / max(2.0, float(expected_block_size))
        rng = np.random.default_rng(seed)

        boot_sharpes = []
        boot_cagrs = []

        for _ in range(num_bootstrap):
            resampled_idx = np.zeros(t, dtype=int)
            curr = rng.integers(0, t)
            for j in range(t):
                if j > 0:
                    if rng.random() < p_geom:
                        curr = rng.integers(0, t)
                    else:
                        curr = (curr + 1) % t
                resampled_idx[j] = curr

            sample = clean_rets[resampled_idx]
            s_std = np.std(sample, ddof=1)
            sh = (np.mean(sample) / s_std) * np.sqrt(252.0) if s_std > 1e-12 else 0.0
            cum_ret = np.prod(1.0 + sample) - 1.0
            cagr = ((1.0 + cum_ret) ** (252.0 / t)) - 1.0 if (1.0 + cum_ret) > 0 else -1.0
            boot_sharpes.append(sh)
            boot_cagrs.append(cagr)

        sharpe_ci = (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5)))
        cagr_ci = (float(np.percentile(boot_cagrs, 2.5)), float(np.percentile(boot_cagrs, 97.5)))

        return sharpe_ci, cagr_ci

    def run_full_robustness_suite(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
        base_backtest_result: BacktestResult | None = None,
    ) -> RobustnessReport:
        """Run all robustness analyses and assemble consolidated report."""
        if base_backtest_result is None:
            engine = EventDrivenEngine(
                initial_capital=self.initial_capital,
                cost_model=get_cost_model("base_case"),
                portfolio_constructor=self.portfolio_constructor,
            )
            base_backtest_result = engine.run(prices_df, signals_df_or_series, features_df)

        cost_res = self.evaluate_cost_regimes(prices_df, signals_df_or_series, features_df)
        sub_res = self.evaluate_subperiods(prices_df, signals_df_or_series, features_df)
        asset_res = self.evaluate_asset_drop_stability(prices_df, signals_df_or_series, features_df)
        extreme_res = self.evaluate_extreme_day_removal(base_backtest_result.daily_returns)
        sh_ci, cagr_ci = self.bootstrap_pnl_confidence_intervals(base_backtest_result.daily_returns)

        return RobustnessReport(
            cost_regime_results=cost_res,
            subperiod_results=sub_res,
            asset_drop_results=asset_res,
            extreme_days_removed_metrics=extreme_res,
            bootstrap_sharpe_ci=sh_ci,
            bootstrap_cagr_ci=cagr_ci,
        )
