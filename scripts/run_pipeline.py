"""Comprehensive end-to-end quantitative alpha research pipeline runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.backtest.engine import EventDrivenEngine
from src.data.cleaning import clean_ohlcv_data, validate_data_quality
from src.data.loader import load_sp100_universe, load_universe_ohlcv
from src.evaluation.metrics import compute_performance_metrics
from src.evaluation.tearsheet import generate_tearsheet
from src.execution.costs import get_cost_model
from src.features.engine import compute_features
from src.features.regimes import compute_market_trend_regime, compute_market_volatility_regime
from src.ml.pipeline import MLAlphaPipeline
from src.portfolio import (
    EqualWeightLongShort,
    InverseVolatilitySignalWeighted,
    SignalWeightedLongShort,
    get_portfolio_constructor,
)
from src.robustness import RobustnessAnalyzer
from src.signals import (
    AbnormalVolumeSignal,
    CombinedSignal,
    MeanReversionSignal,
    MomentumSignal,
    VolatilitySignal,
)
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.statistics.multiple_testing import (
    ConfirmatoryHypothesisRegistry,
    adjust_confirmatory_family,
    run_random_signal_null_simulation,
)
from src.validation.walk_forward import WalkForwardValidator


def run_full_pipeline(
    data_path: str = "data/processed/cleaned_ohlcv.parquet",
    output_dir: str = "reports",
) -> None:
    """Execute the full end-to-end quantitative research framework."""
    print("=" * 80)
    print("STATISTICAL ALPHA RESEARCH & EVENT-DRIVEN BACKTESTING FRAMEWORK")
    print("Authoritative Implementation per Specification v1.2")
    print("=" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion & Quality Validation
    print("\n[Step 1/8] Loading and Validating Historical Data...")
    if Path(data_path).exists():
        df_clean = pd.read_parquet(data_path)
    else:
        print("Processed data not found on disk, running data cleaning...")
        raw_df = load_universe_ohlcv(use_cached=True)
        df_clean, report = clean_ohlcv_data(raw_df)

    print(f"Dataset active rows: {len(df_clean):,} across {df_clean.index.get_level_values('ticker').nunique()} securities.")

    # 2. Feature Computation
    print("\n[Step 2/8] Computing Time-Series and Cross-Sectional Features...")
    features_df = compute_features(df_clean, lag=0, include_forward_targets=True)
    print(f"Features generated: {features_df.shape[1]} columns.")

    # 3. Core Confirmatory Signals
    print("\n[Step 3/8] Generating Confirmatory Signals (H1-H4) & Equal-Weighted Combination...")
    h1_mom = MomentumSignal().compute(features_df)
    h2_mr = MeanReversionSignal().compute(features_df)
    h3_vol = VolatilitySignal().compute(features_df)
    h4_volm = AbnormalVolumeSignal().compute(features_df)
    comb_sig = CombinedSignal().compute(features_df)

    # 4. Statistical Hypothesis Testing & Multiple-Testing Correction
    print("\n[Step 4/8] Running Daily Spearman IC Inference & Multiple Testing Corrections...")
    registry = ConfirmatoryHypothesisRegistry()
    print(f"Confirmatory Hypothesis Registry Hash (SHA-256): {registry.registry_hash()}")

    # Evaluate H1-H4 dynamically from frozen registry definitions
    h1_eval = registry.evaluate_hypothesis_ic("H1_MOMENTUM", h1_mom, features_df, method="spearman")
    h2_eval = registry.evaluate_hypothesis_ic("H2_MEAN_REVERSION", h2_mr, features_df, method="spearman")
    h3_eval = registry.evaluate_hypothesis_ic("H3_LOW_VOLATILITY", h3_vol, features_df, method="spearman")
    h4_eval = registry.evaluate_hypothesis_ic("H4_ABNORMAL_VOLUME", h4_volm, features_df, method="spearman")

    # Combined baseline on standard 20d horizon
    comb_ic_series = compute_ic(comb_sig, features_df["fwd_ret_20d"], method="spearman")
    comb_eval = ic_summary(comb_ic_series, forward_horizon=20)

    print(f"H1 Momentum (horizon {h1_eval['forward_horizon']}d, target {h1_eval['target_column']}): Mean IC={h1_eval['mean_ic']:+.4f}, HAC t={h1_eval['ic_hac_t_stat']:+.3f}, HAC p={h1_eval['ic_hac_p_value']:.6f}")
    print(f"H2 Mean Reversion (horizon {h2_eval['forward_horizon']}d, target {h2_eval['target_column']}): Mean IC={h2_eval['mean_ic']:+.4f}, HAC t={h2_eval['ic_hac_t_stat']:+.3f}, HAC p={h2_eval['ic_hac_p_value']:.6f}")
    print(f"H3 Low Volatility (horizon {h3_eval['forward_horizon']}d, target {h3_eval['target_column']}): Mean IC={h3_eval['mean_ic']:+.4f}, HAC t={h3_eval['ic_hac_t_stat']:+.3f}, HAC p={h3_eval['ic_hac_p_value']:.6f}")
    print(f"H4 Abnormal Volume (horizon {h4_eval['forward_horizon']}d, target {h4_eval['target_column']}): Mean IC={h4_eval['mean_ic']:+.4f}, HAC t={h4_eval['ic_hac_t_stat']:+.3f}, HAC p={h4_eval['ic_hac_p_value']:.6f}")
    print(f"Combined Baseline (horizon 20d, target fwd_ret_20d): Mean IC={comb_eval['mean_ic']:+.4f}, HAC t={comb_eval['ic_hac_t_stat']:+.3f}, HAC p={comb_eval['ic_hac_p_value']:.6f}")

    # Confirmatory Family Multiple Testing Adjustment (m=4)
    hac_p_values = {
        "H1_MOMENTUM": h1_eval["ic_hac_p_value"],
        "H2_MEAN_REVERSION": h2_eval["ic_hac_p_value"],
        "H3_LOW_VOLATILITY": h3_eval["ic_hac_p_value"],
        "H4_ABNORMAL_VOLUME": h4_eval["ic_hac_p_value"],
    }
    adj_df = adjust_confirmatory_family(hac_p_values, alpha_fwer=0.05, alpha_fdr=0.10)
    print("\nConfirmatory Multiple-Testing Results (m=4 Primary Hypotheses):")
    print(adj_df[["raw_p_value", "bonferroni_p_value", "significant_bonferroni", "bh_fdr_q_value", "significant_bh_fdr"]].to_string())

    # 5. Event-Driven Backtesting
    print("\n[Step 5/8] Running Event-Driven Backtest Simulation (Base Cost Regime)...")
    engine = EventDrivenEngine(
        initial_capital=10_000_000.0,
        cost_model=get_cost_model("base_case"),
        portfolio_constructor=EqualWeightLongShort(),
    )
    bt_result = engine.run(df_clean, comb_sig, features_df)
    perf_metrics = compute_performance_metrics(
        daily_returns=bt_result.daily_returns,
        equity_curve=bt_result.equity_curve,
        turnover_series=bt_result.turnover,
        total_cost_usd=float(bt_result.total_costs.sum()),
    )
    print("\n" + generate_tearsheet(perf_metrics, strategy_name="Combined Alpha Strategy (Equal-Weight)"))

    # 6. Expanding-Window Walk-Forward Validation
    print("\n[Step 6/8] Executing 7 Expanding-Window Walk-Forward Validation...")
    wf_validator = WalkForwardValidator()
    wf_report = wf_validator.run_validation(
        prices_df=df_clean,
        signals_df_or_series=comb_sig,
        features_df=features_df,
        evaluate_final_holdout=True,
    )
    print("\nWalk-Forward OOS Performance Summary:")
    for res in wf_report.window_results:
        w = res.window
        tag = "[FINAL HOLDOUT]" if w.is_final_holdout else "[DEV OOS]"
        print(
            f"Window {w.window_id} {tag} ({w.test_start[:4]}): "
            f"OOS Sharpe={res.oos_metrics.sharpe_ratio:.2f}, CAGR={res.oos_metrics.cagr:.2%}, "
            f"MaxDD={res.oos_metrics.max_drawdown:.2%}, PSR={res.oos_psr:.2%}, DSR={res.oos_dsr:.2%}"
        )

    # 7. Robustness & Market Regime Conditioning
    print("\n[Step 7/8] Executing Robustness Suite & Observable Market Regimes...")
    analyzer = RobustnessAnalyzer(portfolio_constructor=EqualWeightLongShort())
    rob_report = analyzer.run_full_robustness_suite(df_clean, comb_sig, features_df, bt_result)

    print("\nCost Regime Sensitivity (CAGR / Sharpe / Costs):")
    for regime_name, met in rob_report.cost_regime_results.items():
        print(f"  {regime_name:12s} -> CAGR: {met.cagr:6.2%}, Sharpe: {met.sharpe_ratio:5.2f}, Total Cost: ${met.total_cost_usd:,.0f}")

    print(f"\nBootstrap 95% CI Sharpe: [{rob_report.bootstrap_sharpe_ci[0]:.2f}, {rob_report.bootstrap_sharpe_ci[1]:.2f}]")
    print(f"Bootstrap 95% CI CAGR:   [{rob_report.bootstrap_cagr_ci[0]:.2%}, {rob_report.bootstrap_cagr_ci[1]:.2%}]")

    # 8. Machine Learning Benchmark Comparison
    print("\n[Step 8/8] Benchmarking Machine Learning Models (OLS, Ridge, Lasso, XGBoost)...")
    ml_pipe = MLAlphaPipeline(target_horizon=20)
    dates_idx = features_df.index.get_level_values("date")
    feat_train = features_df[dates_idx < "2020-01-01"]
    feat_test = features_df[dates_idx >= "2020-01-01"]

    for model_name in ["OLS", "Ridge", "Lasso", "XGBoost"]:
        ml_sig = ml_pipe.train_and_predict(model_name.lower(), feat_train, feat_test)
        comp_res = ml_pipe.compare_with_baseline(
            ml_signals=ml_sig,
            baseline_signals=comb_sig[dates_idx >= "2020-01-01"],
            forward_returns=features_df.loc[dates_idx >= "2020-01-01", "fwd_ret_20d"],
            model_name=model_name,
        )
        print(
            f"  {model_name:8s} -> Mean IC: {comp_res.ml_mean_ic:.4f}, IC IR: {comp_res.ml_ic_ir:.2f}, "
            f"Delta IC (ML - Base): {comp_res.ic_diff_mean:+.4f} (HAC t={comp_res.ic_diff_hac_t_stat:.2f}, p={comp_res.ic_diff_hac_p_value:.4f})"
        )

    print("\n" + "=" * 80)
    print("QUANTITATIVE RESEARCH PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete quantitative alpha research pipeline.")
    parser.add_argument("--data-path", type=str, default="data/processed/cleaned_ohlcv.parquet")
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    run_full_pipeline(data_path=args.data_path, output_dir=args.output_dir)
