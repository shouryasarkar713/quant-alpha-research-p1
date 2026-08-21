"""Script to recompute all empirical results on real data with optimized resampling, caching, checkpointing, and ML benchmark."""

import hashlib
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest.engine import EventDrivenEngine
from src.config.schema import load_config
from src.evaluation.metrics import compute_performance_metrics
from src.execution.costs import CostModel, get_cost_model
from src.features.engine import compute_features
from src.ml import (
    LassoModel,
    MLAlphaPipeline,
    OLSModel,
    RidgeModel,
    XGBoostModel,
)
from src.portfolio import (
    EqualWeightLongShort,
    InverseVolatilitySignalWeighted,
    SignalWeightedLongShort,
)
from src.signals import (
    AbnormalVolumeSignal,
    CombinedSignal,
    MeanReversionSignal,
    MomentumSignal,
    VolatilitySignal,
)
from src.statistics.hypothesis_tests import (
    bootstrap_mean_ci,
    permutation_test_ic,
    quintile_spread_analysis,
)
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.statistics.multiple_testing import (
    ConfirmatoryHypothesisRegistry,
    adjust_confirmatory_family,
)
from src.validation.walk_forward import DEFAULT_WALK_FORWARD_WINDOWS, WalkForwardValidator


def log_progress(msg: str):
    """Emit immediate unbuffered progress message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stdout.write(f"[{ts}] {msg}\n")
    sys.stdout.flush()


def compute_file_hash(filepath: Path) -> str:
    """Compute sha256 hash of a file for cache invalidation."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def create_progress_reporter(stage_name: str, interval: float = 0.5):
    """Create a throttled progress reporter callback."""
    last_reported = [0.0]
    t_start = [time.time()]

    def callback(completed: int, total: int):
        now = time.time()
        pct = (completed / total) * 100.0
        elapsed = now - t_start[0]
        eta = (elapsed / completed) * (total - completed) if completed > 0 else 0.0

        if completed == total or (now - last_reported[0] >= interval):
            last_reported[0] = now
            if completed == total:
                log_progress(f"{stage_name}: {completed}/{total} (100.0%) complete in {elapsed:.2f}s")
            else:
                log_progress(f"{stage_name}: {completed}/{total} ({pct:.1f}%), ETA {eta:.1f}s")

    return callback


def main():
    tracemalloc.start()
    t_pipeline_start = time.perf_counter()

    log_progress("=" * 80)
    log_progress("STARTING EMPIRICAL ALPHA RESEARCH RECOMPUTATION PIPELINE (2014-2024)")
    log_progress("=" * 80)

    cache_dir = Path("data/cache")
    ckpt_dir = Path("results/checkpoints")
    reports_dir = Path("results/reports")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/processed/cleaned_ohlcv.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    data_hash = compute_file_hash(data_path)
    registry = ConfirmatoryHypothesisRegistry()
    reg_hash = registry.registry_hash()
    log_progress(f"Data Hash: {data_hash} | Confirmatory Registry Hash: {reg_hash}")

    # 1. Feature Panel Ingestion & Caching
    feat_cache_file = cache_dir / f"features_{data_hash}.parquet"
    if feat_cache_file.exists():
        log_progress(f"Loading cached feature panel ({feat_cache_file})...")
        features_df = pd.read_parquet(feat_cache_file)
        df_clean = pd.read_parquet(data_path)
    else:
        log_progress("Ingesting cleaned OHLCV data and generating full features panel...")
        df_clean = pd.read_parquet(data_path)
        features_df = compute_features(df_clean, lag=0, include_forward_targets=True)
        features_df.to_parquet(feat_cache_file)
        log_progress(f"Features panel saved to cache ({len(features_df):,} rows)")

    # 2. Primary Signals Panel Caching
    sig_cache_file = cache_dir / f"signals_{data_hash}.parquet"
    if sig_cache_file.exists():
        log_progress(f"Loading cached signals panel ({sig_cache_file})...")
        signals_df = pd.read_parquet(sig_cache_file)
        h1_mom = signals_df["h1_mom"]
        h2_mr = signals_df["h2_mr"]
        h3_vol = signals_df["h3_vol"]
        h4_volm = signals_df["h4_volm"]
        comb_sig = signals_df["comb_sig"]
    else:
        log_progress("Computing primary signals (H1-H4) and Combined Baseline...")
        h1_mom = MomentumSignal().compute(features_df)
        h2_mr = MeanReversionSignal().compute(features_df)
        h3_vol = VolatilitySignal().compute(features_df)
        h4_volm = AbnormalVolumeSignal().compute(features_df)
        comb_sig = CombinedSignal().compute(features_df)

        signals_df = pd.DataFrame({
            "h1_mom": h1_mom,
            "h2_mr": h2_mr,
            "h3_vol": h3_vol,
            "h4_volm": h4_volm,
            "comb_sig": comb_sig,
        })
        signals_df.to_parquet(sig_cache_file)
        log_progress("Signals panel saved to cache")

    # =========================================================================
    # STAGE 1: CONFIRMATORY IC & RESAMPLING INFERENCE
    # =========================================================================
    log_progress("\n" + "=" * 80)
    log_progress("STAGE 1: CONFIRMATORY IC & RESAMPLING INFERENCE (H1:20d, H2:5d, H3:20d, H4:5d, Comb:20d)")
    log_progress("=" * 80)

    res_ckpt_file = ckpt_dir / f"confirmatory_results_{data_hash}_{reg_hash}.json"
    if res_ckpt_file.exists():
        log_progress(f"Loading validated Stage 1 results from checkpoint: {res_ckpt_file}")
        with open(res_ckpt_file, "r") as f:
            stage1_ckpt = json.load(f)
        conf_results = stage1_ckpt["confirmatory_results"]
    else:
        # Dynamic IC evaluation through registry
        h1_eval = registry.evaluate_hypothesis_ic("H1_MOMENTUM", h1_mom, features_df)
        h2_eval = registry.evaluate_hypothesis_ic("H2_MEAN_REVERSION", h2_mr, features_df)
        h3_eval = registry.evaluate_hypothesis_ic("H3_LOW_VOLATILITY", h3_vol, features_df)
        h4_eval = registry.evaluate_hypothesis_ic("H4_ABNORMAL_VOLUME", h4_volm, features_df)

        comb_ic = compute_ic(comb_sig, features_df["fwd_ret_20d"])
        comb_eval = ic_summary(comb_ic, forward_horizon=20)
        comb_eval["ic_series"] = comb_ic

        # Old 1-day exploratory evaluations (for audit comparison only)
        h2_1d_ic = compute_ic(h2_mr, features_df["fwd_ret_1d"])
        h2_1d_eval = ic_summary(h2_1d_ic, forward_horizon=1)
        h4_1d_ic = compute_ic(h4_volm, features_df["fwd_ret_1d"])
        h4_1d_eval = ic_summary(h4_1d_ic, forward_horizon=1)

        # Within-date Permutations (B=1,000)
        log_progress("Running within-date cross-sectional permutation tests (B=1,000)...")
        _, p_perm_h1, _ = permutation_test_ic(
            h1_mom, features_df["fwd_ret_20d"], n_permutations=1000, seed=42, batch_size=500,
            progress_callback=create_progress_reporter("H1 Momentum Permutation (B=1000)")
        )
        _, p_perm_h2, _ = permutation_test_ic(
            h2_mr, features_df["fwd_ret_5d"], n_permutations=1000, seed=42, batch_size=500,
            progress_callback=create_progress_reporter("H2 Mean Reversion Permutation (B=1000)")
        )
        _, p_perm_h3, _ = permutation_test_ic(
            h3_vol, features_df["fwd_ret_20d"], n_permutations=1000, seed=42, batch_size=500,
            progress_callback=create_progress_reporter("H3 Low Volatility Permutation (B=1000)")
        )
        _, p_perm_h4, _ = permutation_test_ic(
            h4_volm, features_df["fwd_ret_5d"], n_permutations=1000, seed=42, batch_size=500,
            progress_callback=create_progress_reporter("H4 Abnormal Volume Permutation (B=1000)")
        )
        _, p_perm_comb, _ = permutation_test_ic(
            comb_sig, features_df["fwd_ret_20d"], n_permutations=1000, seed=42, batch_size=500,
            progress_callback=create_progress_reporter("Combined Baseline Permutation (B=1000)")
        )

        # Stationary Block Bootstrap 95% CI (B=10,000)
        log_progress("Running stationary block bootstrap 95% CIs (B=10,000)...")
        _, ci_l_h1, ci_u_h1 = bootstrap_mean_ci(
            h1_eval["ic_series"].dropna().values, block_size=20, n_bootstrap=10_000, seed=42, batch_size=2500,
            progress_callback=create_progress_reporter("H1 Bootstrap 95% CI (B=10000)")
        )
        _, ci_l_h2, ci_u_h2 = bootstrap_mean_ci(
            h2_eval["ic_series"].dropna().values, block_size=5, n_bootstrap=10_000, seed=42, batch_size=2500,
            progress_callback=create_progress_reporter("H2 Bootstrap 95% CI (B=10000)")
        )
        _, ci_l_h3, ci_u_h3 = bootstrap_mean_ci(
            h3_eval["ic_series"].dropna().values, block_size=20, n_bootstrap=10_000, seed=42, batch_size=2500,
            progress_callback=create_progress_reporter("H3 Bootstrap 95% CI (B=10000)")
        )
        _, ci_l_h4, ci_u_h4 = bootstrap_mean_ci(
            h4_eval["ic_series"].dropna().values, block_size=5, n_bootstrap=10_000, seed=42, batch_size=2500,
            progress_callback=create_progress_reporter("H4 Bootstrap 95% CI (B=10000)")
        )
        _, ci_l_comb, ci_u_comb = bootstrap_mean_ci(
            comb_eval["ic_series"].dropna().values, block_size=20, n_bootstrap=10_000, seed=42, batch_size=2500,
            progress_callback=create_progress_reporter("Combined Bootstrap 95% CI (B=10000)")
        )

        # Quintile Spread Analysis
        log_progress("Evaluating cross-sectional quintile spreads (Q5 - Q1)...")
        q_h1 = quintile_spread_analysis(h1_mom, features_df["fwd_ret_20d"])
        q_h2 = quintile_spread_analysis(h2_mr, features_df["fwd_ret_5d"])
        q_h3 = quintile_spread_analysis(h3_vol, features_df["fwd_ret_20d"])
        q_h4 = quintile_spread_analysis(h4_volm, features_df["fwd_ret_5d"])
        q_comb = quintile_spread_analysis(comb_sig, features_df["fwd_ret_20d"])

        # Multiple-Testing Adjustments (m=4 Confirmatory Family)
        hac_p_dict = {
            "H1_MOMENTUM": h1_eval["ic_hac_p_value"],
            "H2_MEAN_REVERSION": h2_eval["ic_hac_p_value"],
            "H3_LOW_VOLATILITY": h3_eval["ic_hac_p_value"],
            "H4_ABNORMAL_VOLUME": h4_eval["ic_hac_p_value"],
        }
        adj_df = adjust_confirmatory_family(hac_p_dict, alpha=0.05)

        summary_rows = [
            ("H1: Momentum", 20, h1_eval, p_perm_h1, (ci_l_h1, ci_u_h1), q_h1, float(adj_df.loc["H1_MOMENTUM", "bonferroni_p_value"]), float(adj_df.loc["H1_MOMENTUM", "bh_fdr_q_value"])),
            ("H2: Mean Reversion", 5, h2_eval, p_perm_h2, (ci_l_h2, ci_u_h2), q_h2, float(adj_df.loc["H2_MEAN_REVERSION", "bonferroni_p_value"]), float(adj_df.loc["H2_MEAN_REVERSION", "bh_fdr_q_value"])),
            ("H3: Low Volatility", 20, h3_eval, p_perm_h3, (ci_l_h3, ci_u_h3), q_h3, float(adj_df.loc["H3_LOW_VOLATILITY", "bonferroni_p_value"]), float(adj_df.loc["H3_LOW_VOLATILITY", "bh_fdr_q_value"])),
            ("H4: Abnormal Volume", 5, h4_eval, p_perm_h4, (ci_l_h4, ci_u_h4), q_h4, float(adj_df.loc["H4_ABNORMAL_VOLUME", "bonferroni_p_value"]), float(adj_df.loc["H4_ABNORMAL_VOLUME", "bh_fdr_q_value"])),
            ("Combined Baseline", 20, comb_eval, p_perm_comb, (ci_l_comb, ci_u_comb), q_comb, np.nan, np.nan),
        ]

        conf_results = {
            row[0]: {
                "horizon": row[1],
                "mean_ic": row[2]["mean_ic"],
                "ic_std": row[2]["ic_std"],
                "ic_ir": row[2]["ic_ir"],
                "ic_naive_t_stat": row[2]["ic_naive_t_stat"],
                "ic_hac_t_stat": row[2]["ic_hac_t_stat"],
                "ic_hac_p_value": row[2]["ic_hac_p_value"],
                "perm_p_value": row[3],
                "boot_95_ci": list(row[4]),
                "spread_mean": row[5].get("spread_mean", np.nan),
                "spread_hac_t": row[5].get("spread_t_stat", np.nan),
                "bonferroni_p": row[6],
                "bh_fdr_q": row[7],
            }
            for row in summary_rows
        }

        ckpt_payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_hash": data_hash,
                "registry_hash": reg_hash,
                "n_permutations": 1000,
                "n_bootstrap": 10000,
                "seed": 42,
            },
            "confirmatory_results": conf_results,
            "old_1d_exploratory": {
                "H2_MEAN_REVERSION_1D": {
                    "horizon": 1,
                    "mean_ic": h2_1d_eval["mean_ic"],
                    "ic_std": h2_1d_eval["ic_std"],
                    "ic_ir": h2_1d_eval["ic_ir"],
                    "ic_hac_t_stat": h2_1d_eval["ic_hac_t_stat"],
                    "ic_hac_p_value": h2_1d_eval["ic_hac_p_value"],
                },
                "H4_ABNORMAL_VOLUME_1D": {
                    "horizon": 1,
                    "mean_ic": h4_1d_eval["mean_ic"],
                    "ic_std": h4_1d_eval["ic_std"],
                    "ic_ir": h4_1d_eval["ic_ir"],
                    "ic_hac_t_stat": h4_1d_eval["ic_hac_t_stat"],
                    "ic_hac_p_value": h4_1d_eval["ic_hac_p_value"],
                },
            }
        }
        with open(res_ckpt_file, "w") as f:
            json.dump(ckpt_payload, f, indent=2)
        log_progress(f"Confirmatory results checkpoint saved to {res_ckpt_file}")

    # Print summary table
    log_progress("\n" + "=" * 80)
    log_progress("CONFIRMATORY EMPIRICAL RESULTS SUMMARY TABLE (REAL 2014-2024 DATA)")
    log_progress("=" * 80)
    for name, r in conf_results.items():
        spread_mean = r.get("spread_mean", np.nan)
        spread_t = r.get("spread_hac_t", np.nan)
        p_bonf = r.get("bonferroni_p", np.nan)
        q_fdr = r.get("bh_fdr_q", np.nan)
        bonf_str = f"{p_bonf:.6f}" if not np.isnan(p_bonf) else "N/A"
        fdr_str = f"{q_fdr:.6f}" if not np.isnan(q_fdr) else "N/A"
        log_progress(
            f"{name:20s} | h={r['horizon']:2d}d | Mean IC={r['mean_ic']:+.4f} | Std={r['ic_std']:.4f} | "
            f"IR={r['ic_ir']:.3f} | Naive t={r['ic_naive_t_stat']:+.2f} | HAC t={r['ic_hac_t_stat']:+.2f} | "
            f"HAC p={r['ic_hac_p_value']:.6f} | Perm p={r['perm_p_value']:.4f} | Boot 95% CI=[{r['boot_95_ci'][0]:+.4f}, {r['boot_95_ci'][1]:+.4f}] | "
            f"Bonf p={bonf_str} | FDR q={fdr_str} | Spread={spread_mean:+.4f} (HAC t={spread_t:+.2f})"
        )

    # =========================================================================
    # STAGE 2: EVENT-DRIVEN PORTFOLIO BACKTESTS (2014-2024)
    # =========================================================================
    log_progress("\n" + "=" * 80)
    log_progress("STAGE 2: EVENT-DRIVEN PORTFOLIO BACKTESTS (3 MODELS, 2014-2024)")
    log_progress("=" * 80)
    stage2_ckpt_file = ckpt_dir / f"portfolio_results_{data_hash}.json"
    if stage2_ckpt_file.exists():
        log_progress(f"Loading validated Stage 2 results from checkpoint: {stage2_ckpt_file}")
        with open(stage2_ckpt_file, "r") as f:
            portfolio_results = json.load(f)
    else:
        prod_cfg = load_config("configs/default.yaml")
        cost_model = CostModel.from_config(prod_cfg.costs)
        portfolio_results = {}
        for constructor, name in [
            (EqualWeightLongShort(), "Combined Strategy (Equal-Weight)"),
            (SignalWeightedLongShort(), "Signal-Weighted Long-Short"),
            (InverseVolatilitySignalWeighted(), "Inverse-Volatility Long-Short"),
        ]:
            log_progress(f"Running backtest for {name}...")
            engine = EventDrivenEngine(
                initial_capital=prod_cfg.backtest.initial_cash,
                cost_model=cost_model,
                portfolio_constructor=constructor,
            )
            res = engine.run(df_clean, comb_sig, features_df)
            pm = compute_performance_metrics(
                daily_returns=res.daily_returns,
                equity_curve=res.equity_curve,
                turnover_series=res.turnover,
                total_cost_usd=float(res.total_costs.sum()),
            )
            portfolio_results[name] = pm.to_dict()
            log_progress(
                f"{name} Results: CAGR={pm.cagr:+.2%} | Vol={pm.annualized_volatility:.2%} | "
                f"Sharpe={pm.sharpe_ratio:.3f} | Sortino={pm.sortino_ratio:.3f} | MaxDD={pm.max_drawdown:.2%} "
                f"({pm.max_drawdown_duration_days}d) | Hit Rate={pm.daily_hit_rate:.1%} | Profit Factor={pm.profit_factor:.3f} | "
                f"Ann Turnover={pm.annualized_turnover:.1f}x | Total Costs=${pm.total_cost_usd:,.0f} ({pm.cost_drag_bps:.0f} bps drag)"
            )
        with open(stage2_ckpt_file, "w") as f:
            json.dump(portfolio_results, f, indent=2)
        log_progress(f"Stage 2 portfolio results saved to {stage2_ckpt_file}")

    # =========================================================================
    # STAGE 3: WALK-FORWARD VALIDATION (7 EXPANDING WINDOWS)
    # =========================================================================
    log_progress("\n" + "=" * 80)
    log_progress("STAGE 3: WALK-FORWARD VALIDATION (7 EXPANDING WINDOWS: 6 DEV OOS + 1 2024 HOLDOUT)")
    log_progress("=" * 80)
    stage3_ckpt_file = ckpt_dir / f"walk_forward_results_{data_hash}.json"
    if stage3_ckpt_file.exists():
        log_progress(f"Loading validated Stage 3 results from checkpoint: {stage3_ckpt_file}")
        with open(stage3_ckpt_file, "r") as f:
            wf_dict = json.load(f)
    else:
        prod_cfg = load_config("configs/default.yaml")
        cost_model = CostModel.from_config(prod_cfg.costs)
        wf_validator = WalkForwardValidator(cost_model=cost_model)
        wf_report = wf_validator.run_validation(
            prices_df=df_clean,
            signals_df_or_series=comb_sig,
            features_df=features_df,
            evaluate_final_holdout=True,
        )
        wf_dict = {
            "windows": [],
            "dev_oos_combined": wf_report.dev_oos_combined_metrics.to_dict(),
        }
        for res in wf_report.window_results:
            w = res.window
            tag = "[FINAL HOLDOUT]" if w.is_final_holdout else "[DEV OOS]"
            wf_dict["windows"].append({
                "window_id": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "is_final_holdout": w.is_final_holdout,
                "oos_metrics": res.oos_metrics.to_dict(),
                "oos_psr": res.oos_psr,
                "oos_dsr": res.oos_dsr,
            })
            log_progress(
                f"Window {w.window_id} {tag} ({w.test_start[:4]}): "
                f"OOS Sharpe={res.oos_metrics.sharpe_ratio:.3f} | CAGR={res.oos_metrics.cagr:+.2%} | "
                f"Vol={res.oos_metrics.annualized_volatility:.2%} | MaxDD={res.oos_metrics.max_drawdown:.2%} | "
                f"PSR={res.oos_psr:.1%} | DSR={res.oos_dsr:.1%}"
            )
        log_progress(
            f"Aggregated Dev OOS (2018-2023): CAGR={wf_report.dev_oos_combined_metrics.cagr:+.2%} | "
            f"Sharpe={wf_report.dev_oos_combined_metrics.sharpe_ratio:.3f} | "
            f"MaxDD={wf_report.dev_oos_combined_metrics.max_drawdown:.2%}"
        )
        with open(stage3_ckpt_file, "w") as f:
            json.dump(wf_dict, f, indent=2)
        log_progress(f"Stage 3 walk-forward results saved to {stage3_ckpt_file}")

    # =========================================================================
    # STAGE 4: MACHINE LEARNING BENCHMARK (OLS, RIDGE, LASSO, XGBOOST)
    # =========================================================================
    log_progress("\n" + "=" * 80)
    log_progress("STAGE 4: MACHINE LEARNING BENCHMARK (OLS, RIDGE, LASSO, XGBOOST ON 20-DAY TARGET)")
    log_progress("=" * 80)
    ml_pipe = MLAlphaPipeline(
        feature_columns=["ret_12_1_mom", "zscore_price_20", "realized_vol_20", "volume_relative_20"],
        target_col="fwd_ret_20d",
        target_horizon=20,
    )
    ml_models = ["ols", "ridge", "lasso", "xgboost"]
    ml_benchmark_results = {}

    for model_name in ml_models:
        log_progress(f"Training and evaluating {model_name.upper()} across 7 expanding windows...")
        oos_signals_list = []
        for w in DEFAULT_WALK_FORWARD_WINDOWS:
            dates = features_df.index.get_level_values("date")
            feat_train = features_df[(dates >= w.train_start) & (dates <= w.train_end)]
            feat_test = features_df[(dates >= w.test_start) & (dates <= w.test_end)]
            sig_w = ml_pipe.train_and_predict(model_name, feat_train, feat_test)
            oos_signals_list.append(sig_w)

        full_ml_oos_signals = pd.concat(oos_signals_list)
        comp_res = ml_pipe.compare_with_baseline(
            ml_signals=full_ml_oos_signals,
            baseline_signals=comb_sig,
            forward_returns=features_df["fwd_ret_20d"],
            model_name=model_name.upper(),
        )
        ml_benchmark_results[model_name.upper()] = {
            "ml_mean_ic": comp_res.ml_mean_ic,
            "ml_ic_std": comp_res.ml_ic_std,
            "ml_ic_ir": comp_res.ml_ic_ir,
            "ml_hac_t_stat": comp_res.ml_hac_t_stat,
            "ml_hac_p_value": comp_res.ml_hac_p_value,
            "baseline_mean_ic": comp_res.baseline_mean_ic,
            "ic_diff_mean": comp_res.ic_diff_mean,
            "ic_diff_hac_t_stat": comp_res.ic_diff_hac_t_stat,
            "ic_diff_hac_p_value": comp_res.ic_diff_hac_p_value,
            "is_significant_improvement": bool(comp_res.ic_diff_hac_p_value < 0.05 and comp_res.ic_diff_hac_t_stat > 0),
        }
        log_progress(
            f"{comp_res.model_name:10s} | OOS Mean IC={comp_res.ml_mean_ic:+.4f} | IC Std={comp_res.ml_ic_std:.4f} | "
            f"IC IR={comp_res.ml_ic_ir:.3f} | HAC t={comp_res.ml_hac_t_stat:+.2f} (p={comp_res.ml_hac_p_value:.4f}) | "
            f"Delta IC={comp_res.ic_diff_mean:+.4f} (HAC t={comp_res.ic_diff_hac_t_stat:+.2f}, p={comp_res.ic_diff_hac_p_value:.4f})"
        )

    t_pipeline_end = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Final Complete Checkpoint
    complete_payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "data_hash": data_hash,
            "registry_hash": reg_hash,
            "runtime_seconds": t_pipeline_end - t_pipeline_start,
            "peak_memory_mb": peak_memory / (1024 * 1024),
        },
        "confirmatory_results": conf_results,
        "portfolio_results": portfolio_results,
        "walk_forward_results": wf_dict,
        "ml_benchmark_results": ml_benchmark_results,
    }
    complete_ckpt_file = ckpt_dir / "empirical_results_complete.json"
    with open(complete_ckpt_file, "w") as f:
        json.dump(complete_payload, f, indent=2)
    log_progress(f"Full empirical results saved to {complete_ckpt_file}")

    log_progress("\n" + "=" * 80)
    log_progress(f"FULL EMPIRICAL RECOMPUTATION COMPLETE IN {t_pipeline_end - t_pipeline_start:.2f} SECONDS")
    log_progress(f"Peak Process Memory: {peak_memory / (1024 * 1024):.2f} MB")
    log_progress("=" * 80)


if __name__ == "__main__":
    main()
