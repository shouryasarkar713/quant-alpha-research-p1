"""Phase 2E execution script: Machine Learning Benchmark & Robustness Analysis."""

import json
import datetime
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.backtest.engine import EventDrivenEngine
from src.config.schema import load_config
from src.evaluation.metrics import compute_performance_metrics
from src.execution.costs import CostModel, get_cost_model, COST_REGIMES
from src.features.regimes import (
    compute_market_trend_regime,
    compute_market_volatility_regime,
    evaluate_regime_performance,
)
from src.ml import (
    MLAlphaPipeline,
    OLSModel,
    RidgeModel,
    LassoModel,
    XGBoostModel,
)
from src.portfolio import (
    EqualWeightLongShort,
    SignalWeightedLongShort,
    InverseVolatilitySignalWeighted,
)
from src.robustness import RobustnessAnalyzer
from src.statistics.hypothesis_tests import hac_t_test
from src.statistics.information_coefficient import compute_ic
from src.statistics.multiple_testing import ConfirmatoryHypothesisRegistry
from src.validation.walk_forward import DEFAULT_WALK_FORWARD_WINDOWS

# Load Data and Cached Features/Signals
data_path = Path("data/processed/cleaned_ohlcv.parquet")
feat_path = Path("data/cache/features_c3d67525d09fc052.parquet")
sig_path = Path("data/cache/signals_c3d67525d09fc052.parquet")

with open(data_path, "rb") as f:
    data_hash = hashlib.sha256(f.read()).hexdigest()[:16]
with open(feat_path, "rb") as f:
    feat_hash = hashlib.sha256(f.read()).hexdigest()[:16]
with open(sig_path, "rb") as f:
    sig_hash = hashlib.sha256(f.read()).hexdigest()[:16]

reg = ConfirmatoryHypothesisRegistry()
reg_hash = reg.registry_hash()

cfg = load_config("configs/default.yaml")
cfg_hash = cfg.compute_hash()

print("=== DATA & CACHE IDENTIFIERS ===")
print("Data Hash:", data_hash)
print("Feature Hash:", feat_hash)
print("Signal Hash:", sig_hash)
print("Registry Hash:", reg_hash)
print("Config Hash:", cfg_hash)

df_clean = pd.read_parquet(data_path)
features_df = pd.read_parquet(feat_path)
signals_df = pd.read_parquet(sig_path)
comb_sig = signals_df["comb_sig"]
target_20d = features_df["fwd_ret_20d"]

feature_cols = ["ret_12_1_mom", "zscore_price_20", "realized_vol_20", "volume_relative_20"]

# =========================================================================
# PART 1: MACHINE LEARNING BENCHMARK (OLS, RIDGE, LASSO, XGBOOST)
# =========================================================================
print("\n" + "=" * 80)
print("PART 1: MACHINE LEARNING BENCHMARK (ON COMMON 20-DAY TARGET)")
print("=" * 80)

ml_pipeline = MLAlphaPipeline(
    feature_columns=feature_cols,
    target_col="fwd_ret_20d",
    target_horizon=20,
)

ml_models = ["ols", "ridge", "lasso", "xgboost"]
ml_results = {}
ml_window_results = {m: [] for m in ml_models}

for model_name in ml_models:
    print(f"\n--- Training & Evaluating {model_name.upper()} Across 7 Expanding Windows ---")
    oos_preds_list = []
    
    for w in DEFAULT_WALK_FORWARD_WINDOWS:
        dates = features_df.index.get_level_values("date")
        train_mask = (dates >= w.train_start) & (dates <= w.train_end)
        test_mask = (dates >= w.test_start) & (dates <= w.test_end)
        
        feat_train = features_df[train_mask]
        feat_test = features_df[test_mask]
        
        pred_w = ml_pipeline.train_and_predict(model_name, feat_train, feat_test)
        oos_preds_list.append(pred_w)
        
        # Window-specific IC
        tgt_w = target_20d[test_mask]
        comb_w = comb_sig[test_mask]
        ic_ml_w = compute_ic(pred_w, tgt_w, method="spearman").dropna()
        ic_base_w = compute_ic(comb_w, tgt_w, method="spearman").dropna()
        
        aligned_w = pd.concat([ic_ml_w.rename("ml"), ic_base_w.rename("base")], axis=1).dropna()
        aligned_w["diff"] = aligned_w["ml"] - aligned_w["base"]
        
        m_ic = float(aligned_w["ml"].mean())
        s_ic = float(aligned_w["ml"].std(ddof=1))
        ir_ic = m_ic / s_ic if s_ic > 1e-12 else 0.0
        t_ml, p_ml = hac_t_test(aligned_w["ml"], max_lag=20)
        
        d_mean = float(aligned_w["diff"].mean())
        d_t, d_p = hac_t_test(aligned_w["diff"], max_lag=20)
        
        tag = "[FINAL HOLDOUT]" if w.is_final_holdout else "[DEV OOS]"
        win_info = {
            "window_id": w.window_id,
            "train_period": f"{w.train_start} to {w.train_end}",
            "test_period": f"{w.test_start} to {w.test_end}",
            "is_final_holdout": w.is_final_holdout,
            "tag": tag,
            "ml_mean_ic": m_ic,
            "ml_ic_std": s_ic,
            "ml_ic_ir": ir_ic,
            "ml_hac_t_stat": t_ml,
            "ml_hac_p_value": p_ml,
            "baseline_mean_ic": float(aligned_w["base"].mean()),
            "ic_diff_mean": d_mean,
            "ic_diff_hac_t_stat": d_t,
            "ic_diff_hac_p_value": d_p,
        }
        ml_window_results[model_name].append(win_info)
        print(f"  Window {w.window_id} {tag} ({w.test_start[:4]}): ML IC={m_ic:+.4f} (HAC t={t_ml:+.2f}, p={p_ml:.4f}) | Delta IC={d_mean:+.4f} (HAC t={d_t:+.2f}, p={d_p:.4f})")

    # Full Out-of-Sample evaluation (Windows 1-7 concatenated: 2018-2024)
    full_ml_signals = pd.concat(oos_preds_list)
    comp_full = ml_pipeline.compare_with_baseline(
        ml_signals=full_ml_signals,
        baseline_signals=comb_sig.loc[full_ml_signals.index],
        forward_returns=target_20d.loc[full_ml_signals.index],
        model_name=model_name.upper(),
    )
    
    # Dev OOS evaluation (Windows 1-6 concatenated: 2018-2023)
    dev_dates = pd.to_datetime("2023-12-31")
    dev_mask = full_ml_signals.index.get_level_values("date") <= dev_dates
    ml_dev = full_ml_signals[dev_mask]
    comp_dev = ml_pipeline.compare_with_baseline(
        ml_signals=ml_dev,
        baseline_signals=comb_sig.loc[ml_dev.index],
        forward_returns=target_20d.loc[ml_dev.index],
        model_name=model_name.upper(),
    )
    
    # Final Holdout evaluation (Window 7: 2024)
    holdout_mask = full_ml_signals.index.get_level_values("date") > dev_dates
    ml_holdout = full_ml_signals[holdout_mask]
    comp_holdout = ml_pipeline.compare_with_baseline(
        ml_signals=ml_holdout,
        baseline_signals=comb_sig.loc[ml_holdout.index],
        forward_returns=target_20d.loc[ml_holdout.index],
        model_name=model_name.upper(),
    )
    
    ml_results[model_name.upper()] = {
        "full_sample": {
            "ml_mean_ic": comp_full.ml_mean_ic,
            "ml_ic_std": comp_full.ml_ic_std,
            "ml_ic_ir": comp_full.ml_ic_ir,
            "ml_hac_t_stat": comp_full.ml_hac_t_stat,
            "ml_hac_p_value": comp_full.ml_hac_p_value,
            "baseline_mean_ic": comp_full.baseline_mean_ic,
            "baseline_ic_ir": comp_full.baseline_ic_ir,
            "ic_diff_mean": comp_full.ic_diff_mean,
            "ic_diff_hac_t_stat": comp_full.ic_diff_hac_t_stat,
            "ic_diff_hac_p_value": comp_full.ic_diff_hac_p_value,
            "is_significant_improvement": bool(comp_full.ic_diff_hac_p_value < 0.05 and comp_full.ic_diff_hac_t_stat > 0),
        },
        "dev_oos_2018_2023": {
            "ml_mean_ic": comp_dev.ml_mean_ic,
            "ml_ic_std": comp_dev.ml_ic_std,
            "ml_ic_ir": comp_dev.ml_ic_ir,
            "ml_hac_t_stat": comp_dev.ml_hac_t_stat,
            "ml_hac_p_value": comp_dev.ml_hac_p_value,
            "baseline_mean_ic": comp_dev.baseline_mean_ic,
            "ic_diff_mean": comp_dev.ic_diff_mean,
            "ic_diff_hac_t_stat": comp_dev.ic_diff_hac_t_stat,
            "ic_diff_hac_p_value": comp_dev.ic_diff_hac_p_value,
        },
        "holdout_2024": {
            "ml_mean_ic": comp_holdout.ml_mean_ic,
            "ml_ic_std": comp_holdout.ml_ic_std,
            "ml_ic_ir": comp_holdout.ml_ic_ir,
            "ml_hac_t_stat": comp_holdout.ml_hac_t_stat,
            "ml_hac_p_value": comp_holdout.ml_hac_p_value,
            "baseline_mean_ic": comp_holdout.baseline_mean_ic,
            "ic_diff_mean": comp_holdout.ic_diff_mean,
            "ic_diff_hac_t_stat": comp_holdout.ic_diff_hac_t_stat,
            "ic_diff_hac_p_value": comp_holdout.ic_diff_hac_p_value,
        },
        "windows": ml_window_results[model_name],
    }

print("\n=== ML FULL OOS (2018-2024) SUMMARY TABLE ===")
print(f"{'Model':10s} | {'ML Mean IC':10s} | {'ML IC Std':10s} | {'ML ICIR':8s} | {'ML HAC t (p)':18s} | {'Base Mean IC':12s} | {'Delta IC':10s} | {'Delta HAC t (p)':18s} | {'Sig Outperf?'}")
print("-" * 115)
for m_name, res_dict in ml_results.items():
    f_res = res_dict["full_sample"]
    sig_str = "YES" if f_res["is_significant_improvement"] else "NO"
    print(f"{m_name:10s} | {f_res['ml_mean_ic']:+10.4f} | {f_res['ml_ic_std']:10.4f} | {f_res['ml_ic_ir']:+8.3f} | {f_res['ml_hac_t_stat']:+6.2f} (p={f_res['ml_hac_p_value']:.4f}) | {f_res['baseline_mean_ic']:+12.4f} | {f_res['ic_diff_mean']:+10.4f} | {f_res['ic_diff_hac_t_stat']:+6.2f} (p={f_res['ic_diff_hac_p_value']:.4f}) | {sig_str}")

# =========================================================================
# PART 2: ROBUSTNESS ANALYSIS
# =========================================================================
print("\n" + "=" * 80)
print("PART 2: ROBUSTNESS ANALYSIS SUITE")
print("=" * 80)

# Base strategy execution (EqualWeightLongShort on Combined Signal, Base-Case Cost)
engine_base = EventDrivenEngine(
    initial_capital=10_000_000.0,
    cost_model=CostModel.from_config(cfg.costs),
    portfolio_constructor=EqualWeightLongShort(),
)
base_res = engine_base.run(df_clean, comb_sig, features_df)
base_rets = base_res.daily_returns

analyzer = RobustnessAnalyzer(
    portfolio_constructor=EqualWeightLongShort(),
    initial_capital=10_000_000.0,
)

# 1. Cost Regimes
cost_sensitivity = {}
for r_name in ["zero", "low", "medium", "high", "very_high"]:
    cm = get_cost_model(r_name)
    eng = EventDrivenEngine(initial_capital=10_000_000.0, cost_model=cm, portfolio_constructor=EqualWeightLongShort())
    r_res = eng.run(df_clean, comb_sig, features_df)
    r_pm = compute_performance_metrics(r_res.daily_returns, r_res.equity_curve, r_res.turnover, float(r_res.total_costs.sum()))
    cost_sensitivity[r_name] = r_pm.to_dict()
    cost_sensitivity[r_name]["terminal_equity"] = float(r_res.equity_curve.iloc[-1])

# 2. Pre/Post-2020 Subperiod Split
subperiods = analyzer.evaluate_subperiods(df_clean, comb_sig, features_df, split_date="2020-01-01")
subperiod_dict = {k: v.to_dict() for k, v in subperiods.items()}

# 3. Asset Jackknife (10% exclusion, 5 iterations)
asset_jackknife = analyzer.evaluate_asset_drop_stability(df_clean, comb_sig, features_df, drop_pct=0.10, iterations=5, seed=42)
jackknife_dict = [pm.to_dict() for pm in asset_jackknife]

# 4. Extreme Day Removal (Top 5 & Bottom 5 trimmed)
extreme_day_metrics = analyzer.evaluate_extreme_day_removal(base_rets, n_extreme_days=5)

# 5. Stationary Block Bootstrap PnL 95% CIs
sharpe_ci, cagr_ci = analyzer.bootstrap_pnl_confidence_intervals(base_rets, num_bootstrap=1000, expected_block_size=10, seed=42)

# 6. Market Regime Conditioning (Volatility & Trend)
# Market benchmark returns = cross-sectional mean daily return
daily_mkt_ret = df_clean["close"].unstack(level="ticker").pct_change().mean(axis=1).dropna()
daily_mkt_price = (1.0 + daily_mkt_ret).cumprod() * 100.0

vol_regimes = compute_market_volatility_regime(daily_mkt_ret, window=20, low_pct=0.40, high_pct=0.60)
trend_regimes = compute_market_trend_regime(daily_mkt_price, fast_window=50, slow_window=200)

vol_perf = evaluate_regime_performance(base_rets, vol_regimes)
trend_perf = evaluate_regime_performance(base_rets, trend_regimes)

regime_dict = {
    "volatility_regimes": {k: v.to_dict() for k, v in vol_perf.items()},
    "trend_regimes": {k: v.to_dict() for k, v in trend_perf.items()},
}

print("\n--- 1. Pre/Post-2020 Subperiod Results ---")
for k, v in subperiod_dict.items():
    print(f"  {k:10s}: CAGR={v['cagr']:+.2%} | Vol={v['annualized_volatility']:.2%} | Sharpe={v['sharpe_ratio']:+.3f} | MaxDD={v['max_drawdown']:.2%}")

print("\n--- 2. Asset Jackknife (10% Random Drop, 5 Iterations) ---")
for i, pm_d in enumerate(jackknife_dict):
    print(f"  Iteration {i+1}: CAGR={pm_d['cagr']:+.2%} | Vol={pm_d['annualized_volatility']:.2%} | Sharpe={pm_d['sharpe_ratio']:+.3f} | MaxDD={pm_d['max_drawdown']:.2%}")

print("\n--- 3. Extreme-Day Removal (Top 5 & Bottom 5 Trimmed) ---")
print(f"  Trimmed (N={extreme_day_metrics.total_trading_days}): CAGR={extreme_day_metrics.cagr:+.2%} | Vol={extreme_day_metrics.annualized_volatility:.2%} | Sharpe={extreme_day_metrics.sharpe_ratio:+.3f} | MaxDD={extreme_day_metrics.max_drawdown:.2%}")

print("\n--- 4. Stationary Bootstrap PnL 95% Confidence Intervals ---")
print(f"  Sharpe 95% CI: [{sharpe_ci[0]:+.3f}, {sharpe_ci[1]:+.3f}]")
print(f"  CAGR 95% CI:   [{cagr_ci[0]:+.2%}, {cagr_ci[1]:+.2%}]")

print("\n--- 5. Market Regime Slicing ---")
print("  Volatility Regimes:")
for k, v in regime_dict["volatility_regimes"].items():
    print(f"    {k:12s} (N={v['total_trading_days']}d): CAGR={v['cagr']:+.2%} | Vol={v['annualized_volatility']:.2%} | Sharpe={v['sharpe_ratio']:+.3f} | Win Rate={v['daily_hit_rate']:.1%}")
print("  Trend Regimes:")
for k, v in regime_dict["trend_regimes"].items():
    print(f"    {k:12s} (N={v['total_trading_days']}d): CAGR={v['cagr']:+.2%} | Vol={v['annualized_volatility']:.2%} | Sharpe={v['sharpe_ratio']:+.3f} | Win Rate={v['daily_hit_rate']:.1%}")

# =========================================================================
# PART 3: PERSIST REPRODUCIBILITY CHECKPOINTS
# =========================================================================
ckpt_dir = Path("results/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# 1. ML Benchmark Checkpoint
ml_ckpt_path = ckpt_dir / f"ml_benchmark_{data_hash}.json"
with open(ml_ckpt_path, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoint_version": "1.2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_hash": data_hash,
        "feature_hash": feat_hash,
        "signal_hash": sig_hash,
        "config_hash": cfg_hash,
        "hypothesis_registry_hash": reg_hash,
        "feature_columns": feature_cols,
        "target_column": "fwd_ret_20d",
        "target_horizon": 20,
        "random_seed": 42,
        "models_evaluated": ml_models,
        "results": ml_results,
    }, f, indent=2)

# 2. Robustness Checkpoint
robustness_ckpt_path = ckpt_dir / f"robustness_analysis_{data_hash}.json"
with open(robustness_ckpt_path, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoint_version": "1.2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_hash": data_hash,
        "feature_hash": feat_hash,
        "signal_hash": sig_hash,
        "config_hash": cfg_hash,
        "hypothesis_registry_hash": reg_hash,
        "base_strategy": "EqualWeightLongShort on Combined Baseline (Medium Cost)",
        "cost_sensitivity": cost_sensitivity,
        "subperiod_analysis": subperiod_dict,
        "asset_jackknife": jackknife_dict,
        "extreme_day_removal": extreme_day_metrics.to_dict(),
        "bootstrap_pnl_95_ci": {
            "sharpe_ci": list(sharpe_ci),
            "cagr_ci": list(cagr_ci),
        },
        "regime_conditioning": regime_dict,
    }, f, indent=2)

print(f"\nSaved Phase 2E Checkpoints:\n  {ml_ckpt_path}\n  {robustness_ckpt_path}")
