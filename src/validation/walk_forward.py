"""Expanding-window walk-forward validation framework: 7 expanding windows and untouched 2024 final holdout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import pandas as pd

from src.backtest.engine import BacktestResult, EventDrivenEngine
from src.evaluation.metrics import PerformanceMetrics, compute_performance_metrics
from src.execution.costs import CostModel, get_cost_model
from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.equal_weight import EqualWeightLongShort
from src.signals.base import BaseSignal
from src.validation.dsr import compute_deflated_sharpe_ratio, compute_probabilistic_sharpe_ratio


@dataclass(frozen=True)
class WalkForwardWindow:
    """Expanding-window split definition."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_final_holdout: bool


# 7 Pre-specified Expanding Windows (Specification Section 21.1)
DEFAULT_WALK_FORWARD_WINDOWS = [
    WalkForwardWindow(1, "2014-01-01", "2017-12-31", "2018-01-01", "2018-12-31", False),
    WalkForwardWindow(2, "2014-01-01", "2018-12-31", "2019-01-01", "2019-12-31", False),
    WalkForwardWindow(3, "2014-01-01", "2019-12-31", "2020-01-01", "2020-12-31", False),
    WalkForwardWindow(4, "2014-01-01", "2020-12-31", "2021-01-01", "2021-12-31", False),
    WalkForwardWindow(5, "2014-01-01", "2021-12-31", "2022-01-01", "2022-12-31", False),
    WalkForwardWindow(6, "2014-01-01", "2022-12-31", "2023-01-01", "2023-12-31", False),
    WalkForwardWindow(7, "2014-01-01", "2023-12-31", "2024-01-01", "2024-12-31", True),  # Final holdout
]


@dataclass
class WindowEvaluationResult:
    """Evaluation summary for a single walk-forward window."""

    window: WalkForwardWindow
    is_metrics: PerformanceMetrics
    oos_metrics: PerformanceMetrics
    oos_psr: float
    oos_dsr: float | None
    oos_equity: pd.Series
    oos_returns: pd.Series


@dataclass
class WalkForwardReport:
    """Full Walk-Forward Validation Report."""

    window_results: list[WindowEvaluationResult]
    dev_oos_combined_metrics: PerformanceMetrics  # Windows 1-6 concatenated
    final_holdout_metrics: PerformanceMetrics | None  # Window 7 evaluated once
    dev_oos_returns: pd.Series
    final_holdout_returns: pd.Series | None


class WalkForwardValidator:
    """
    Expanding-Window Walk-Forward Validator (Specification Section 21):
    - Windows 1-6 (2018-2023): Development OOS for tuning and evaluation.
    - Window 7 (2024): Single-use final holdout evaluated strictly once on finalized strategy.
    """

    def __init__(
        self,
        windows: list[WalkForwardWindow] | None = None,
        cost_model: CostModel | None = None,
        portfolio_constructor: BasePortfolioConstructor | None = None,
    ) -> None:
        self.windows = windows or DEFAULT_WALK_FORWARD_WINDOWS
        self.cost_model = cost_model or CostModel()
        self.portfolio_constructor = portfolio_constructor or EqualWeightLongShort()

    def run_validation(
        self,
        prices_df: pd.DataFrame,
        signals_df_or_series: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame | None = None,
        evaluate_final_holdout: bool = True,
        num_trials: int | None = None,
    ) -> WalkForwardReport:
        """
        Execute expanding-window walk-forward validation across all configured windows.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Full historical price panel.
        signals_df_or_series : pd.DataFrame | pd.Series
            Signal panel.
        features_df : pd.DataFrame | None
            Features panel.
        evaluate_final_holdout : bool
            Whether to evaluate Window 7 final holdout (set False during development).
        num_trials : int
            Number of tested trials for DSR calculation.

        Returns
        -------
        WalkForwardReport
        """
        engine = EventDrivenEngine(
            initial_capital=10_000_000.0,
            cost_model=self.cost_model,
            portfolio_constructor=self.portfolio_constructor,
        )

        window_results: list[WindowEvaluationResult] = []
        dev_oos_returns_list: list[pd.Series] = []
        final_holdout_res: WindowEvaluationResult | None = None

        if isinstance(signals_df_or_series, pd.DataFrame):
            sig_s = signals_df_or_series.iloc[:, 0]
        else:
            sig_s = signals_df_or_series.copy()

        for win in self.windows:
            if win.is_final_holdout and not evaluate_final_holdout:
                continue

            train_start_ts = pd.to_datetime(win.train_start)
            train_end_ts = pd.to_datetime(win.train_end)
            test_start_ts = pd.to_datetime(win.test_start)
            test_end_ts = pd.to_datetime(win.test_end)

            p_train = prices_df.loc[
                (prices_df.index.get_level_values("date") >= train_start_ts)
                & (prices_df.index.get_level_values("date") <= train_end_ts)
            ]
            s_train = sig_s.loc[
                (sig_s.index.get_level_values("date") >= train_start_ts)
                & (sig_s.index.get_level_values("date") <= train_end_ts)
            ]
            f_train = (
                features_df.loc[
                    (features_df.index.get_level_values("date") >= train_start_ts)
                    & (features_df.index.get_level_values("date") <= train_end_ts)
                ]
                if features_df is not None
                else None
            )

            p_test = prices_df.loc[
                (prices_df.index.get_level_values("date") >= test_start_ts)
                & (prices_df.index.get_level_values("date") <= test_end_ts)
            ]
            s_test = sig_s.loc[
                (sig_s.index.get_level_values("date") >= test_start_ts)
                & (sig_s.index.get_level_values("date") <= test_end_ts)
            ]
            f_test = (
                features_df.loc[
                    (features_df.index.get_level_values("date") >= test_start_ts)
                    & (features_df.index.get_level_values("date") <= test_end_ts)
                ]
                if features_df is not None
                else None
            )

            if len(p_test) == 0:
                continue

            # 1. In-sample backtest
            res_is = engine.run(p_train, s_train, f_train)
            is_metrics = compute_performance_metrics(
                daily_returns=res_is.daily_returns,
                equity_curve=res_is.equity_curve,
                turnover_series=res_is.turnover,
                total_cost_usd=float(res_is.total_costs.sum()),
            )

            # 2. Out-of-sample backtest
            res_oos = engine.run(p_test, s_test, f_test)
            oos_metrics = compute_performance_metrics(
                daily_returns=res_oos.daily_returns,
                equity_curve=res_oos.equity_curve,
                turnover_series=res_oos.turnover,
                total_cost_usd=float(res_oos.total_costs.sum()),
            )

            # 3. Statistical testing (PSR as primary auxiliary diagnostic; DSR only if explicit num_trials given)
            oos_psr = compute_probabilistic_sharpe_ratio(res_oos.daily_returns, benchmark_sharpe=0.0)
            oos_dsr = (
                compute_deflated_sharpe_ratio(res_oos.daily_returns, num_trials=num_trials)
                if num_trials is not None
                else None
            )

            win_res = WindowEvaluationResult(
                window=win,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                oos_psr=oos_psr,
                oos_dsr=oos_dsr,
                oos_equity=res_oos.equity_curve,
                oos_returns=res_oos.daily_returns,
            )
            window_results.append(win_res)

            if win.is_final_holdout:
                final_holdout_res = win_res
            else:
                dev_oos_returns_list.append(res_oos.daily_returns)

        # Concatenate development OOS returns (Windows 1-6)
        if dev_oos_returns_list:
            combined_dev_rets = pd.concat(dev_oos_returns_list).sort_index()
            dev_metrics = compute_performance_metrics(combined_dev_rets)
        else:
            combined_dev_rets = pd.Series(dtype=float)
            dev_metrics = compute_performance_metrics(combined_dev_rets)

        final_holdout_metrics = final_holdout_res.oos_metrics if final_holdout_res else None
        final_holdout_returns = final_holdout_res.oos_returns if final_holdout_res else None

        return WalkForwardReport(
            window_results=window_results,
            dev_oos_combined_metrics=dev_metrics,
            final_holdout_metrics=final_holdout_metrics,
            dev_oos_returns=combined_dev_rets,
            final_holdout_returns=final_holdout_returns,
        )
