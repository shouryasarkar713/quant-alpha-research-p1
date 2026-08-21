"""Phase 2D-2 computation script for portfolio backtesting and walk-forward validation."""

import json
import datetime
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.backtest.engine import EventDrivenEngine, BacktestResult
from src.config.schema import load_config
from src.evaluation.metrics import compute_performance_metrics
from src.execution.costs import CostModel, get_cost_model, COST_REGIMES
from src.portfolio import (
    EqualWeightLongShort,
    SignalWeightedLongShort,
    InverseVolatilitySignalWeighted,
)
from src.statistics.multiple_testing import ConfirmatoryHypothesisRegistry
from src.validation.walk_forward import DEFAULT_WALK_FORWARD_WINDOWS, WalkForwardValidator
from src.validation.dsr import compute_probabilistic_sharpe_ratio

# Hashes and Artifacts
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

print("=== DATA IDENTIFIERS ===")
print("Data Hash:", data_hash)
print("Feature Hash:", feat_hash)
print("Signal Hash:", sig_hash)
print("Registry Hash:", reg_hash)
print("Config Hash:", cfg_hash)

df_clean = pd.read_parquet(data_path)
features_df = pd.read_parquet(feat_path)
signals_df = pd.read_parquet(sig_path)
comb_sig = signals_df["comb_sig"]

# =========================================================================
# 1. FULL-SAMPLE PORTFOLIO BACKTESTS ACROSS CONSTRUCTORS & COST REGIMES
# =========================================================================
constructors = [
    ("EqualWeightLongShort", EqualWeightLongShort()),
    ("SignalWeightedLongShort", SignalWeightedLongShort()),
    ("InverseVolatilitySignalWeighted", InverseVolatilitySignalWeighted()),
]

regimes = ["zero", "low", "medium", "high", "very_high"]

portfolio_results = {}
sensitivity_results = {}

for name, constructor in constructors:
    portfolio_results[name] = {}
    sensitivity_results[name] = {}
    for r in regimes:
        cm = get_cost_model(r)
        engine = EventDrivenEngine(
            initial_capital=10_000_000.0,
            cost_model=cm,
            portfolio_constructor=constructor,
        )
        res = engine.run(df_clean, comb_sig, features_df)
        pm = compute_performance_metrics(
            daily_returns=res.daily_returns,
            equity_curve=res.equity_curve,
            turnover_series=res.turnover,
            total_cost_usd=float(res.total_costs.sum()),
        )
        metrics_dict = pm.to_dict()
        metrics_dict["terminal_equity"] = float(res.equity_curve.iloc[-1])
        metrics_dict["gross_exposure_max"] = float(res.gross_exposure.max())
        metrics_dict["gross_exposure_mean"] = float(res.gross_exposure.mean())
        metrics_dict["net_exposure_max"] = float(res.net_exposure.abs().max())
        metrics_dict["total_trades"] = len(res.trades_df)
        
        sensitivity_results[name][r] = metrics_dict
        if r in ["zero", "medium"]:
            portfolio_results[name][r] = metrics_dict

# Print Table for Zero and Medium
print("\n=== 1. FULL SAMPLE PORTFOLIO BACKTEST RESULTS ===")
for name in portfolio_results:
    m_zero = portfolio_results[name]["zero"]
    m_base = portfolio_results[name]["medium"]
    cost_drag_cagr = m_zero["cagr"] - m_base["cagr"]
    cost_drag_sharpe = m_zero["sharpe_ratio"] - m_base["sharpe_ratio"]
    print(f"\n--- {name} ---")
    print(f"  [Zero-Cost]   CAGR: {m_zero['cagr']:+.2%} | Vol: {m_zero['annualized_volatility']:.2%} | Sharpe: {m_zero['sharpe_ratio']:+.3f} | Sortino: {m_zero['sortino_ratio']:+.3f} | MaxDD: {m_zero['max_drawdown']:.2%} | TermEq: ${m_zero['terminal_equity']:,.0f}")
    print(f"  [Base-Case]   CAGR: {m_base['cagr']:+.2%} | Vol: {m_base['annualized_volatility']:.2%} | Sharpe: {m_base['sharpe_ratio']:+.3f} | Sortino: {m_base['sortino_ratio']:+.3f} | MaxDD: {m_base['max_drawdown']:.2%} | TermEq: ${m_base['terminal_equity']:,.0f}")
    print(f"  Costs/Trades: Turnover={m_base['annualized_turnover']:.1f}x | TotalCosts=${m_base['total_cost_usd']:,.0f} | Drag={m_base['cost_drag_bps']:.0f} bps | TotalTrades={m_base['total_trades']:,}")
    print(f"  Drag (CAGR diff): {cost_drag_cagr:+.2%} | Drag (Sharpe diff): {cost_drag_sharpe:+.3f}")

# =========================================================================
# 2. WALK-FORWARD VALIDATION (7 EXPANDING WINDOWS)
# =========================================================================
print("\n=== 2. WALK-FORWARD EXPANDING WINDOWS VALIDATION ===")
wf_validator = WalkForwardValidator(
    windows=DEFAULT_WALK_FORWARD_WINDOWS,
    cost_model=CostModel.from_config(cfg.costs),
    portfolio_constructor=EqualWeightLongShort(),
)

wf_report = wf_validator.run_validation(
    prices_df=df_clean,
    signals_df_or_series=comb_sig,
    features_df=features_df,
    evaluate_final_holdout=True,
    num_trials=None,  # DSR omitted per Option A
)

wf_summary = []
for res in wf_report.window_results:
    w = res.window
    m = res.oos_metrics
    tag = "[FINAL HOLDOUT]" if w.is_final_holdout else "[DEV OOS]"
    row = {
        "window_id": w.window_id,
        "train_period": f"{w.train_start} to {w.train_end}",
        "test_period": f"{w.test_start} to {w.test_end}",
        "is_final_holdout": w.is_final_holdout,
        "tag": tag,
        "cagr": m.cagr,
        "volatility": m.annualized_volatility,
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "max_dd": m.max_drawdown,
        "psr": res.oos_psr,
        "turnover": m.annualized_turnover,
        "total_cost": m.total_cost_usd,
        "terminal_equity": float(res.oos_equity.iloc[-1]),
    }
    wf_summary.append(row)
    print(f"  Window {w.window_id} {tag} ({w.test_start[:4]}): CAGR={m.cagr:+.2%} | Vol={m.annualized_volatility:.2%} | Sharpe={m.sharpe_ratio:+.3f} | MaxDD={m.max_drawdown:.2%} | PSR={res.oos_psr:.1%} | Costs=${m.total_cost_usd:,.0f}")

dev_comb = wf_report.dev_oos_combined_metrics
print(f"\n  Aggregate Dev OOS (2018-2023): CAGR={dev_comb.cagr:+.2%} | Vol={dev_comb.annualized_volatility:.2%} | Sharpe={dev_comb.sharpe_ratio:+.3f} | MaxDD={dev_comb.max_drawdown:.2%}")

# =========================================================================
# 3. BACKTEST INTEGRITY AUDIT CHECKS
# =========================================================================
print("\n=== 3. BACKTEST INTEGRITY AUDIT ===")
base_engine = EventDrivenEngine(
    initial_capital=10_000_000.0,
    cost_model=CostModel.from_config(cfg.costs),
    portfolio_constructor=EqualWeightLongShort(),
)
base_res = base_engine.run(df_clean, comb_sig, features_df)

# Check 1: Order Timing (Signal at t -> execution at t+1)
trades = base_res.trades_df
min_data_date = df_clean.index.get_level_values("date").min()
first_trade_date = trades["timestamp"].min()
print(f"  Check 1 (Timing): First trade date: {first_trade_date.date()} (strictly > min date {min_data_date.date()}) -> {first_trade_date > min_data_date}")

# Check 2: TWX termination check (TWX terminates 2018-06-14, zero trades after 2018-06-15)
twx_trades_after = trades[(trades["ticker"] == "TWX") & (trades["timestamp"] > pd.Timestamp("2018-06-15"))]
print(f"  Check 2 (TWX Delisting): TWX trades after 2018-06-15 count: {len(twx_trades_after)}")

# Check 3: Unavailable tickers check (APC, DOW, EMC, FOXA, MON, RTN, WAG)
unavail_in_trades = set(["APC", "DOW", "EMC", "FOXA", "MON", "RTN", "WAG"]).intersection(set(trades["ticker"].unique()))
print(f"  Check 3 (Unavailable Tickers): Unavailable tickers in trades: {unavail_in_trades}")

# Check 4: Exposure constraints
print(f"  Check 4 (Max Gross Exposure): {base_res.gross_exposure.max():.4f} (<= 1.0 target, MTM drift <= 1.30)")
print(f"  Check 4 (Max Net Exposure): {base_res.net_exposure.abs().max():.4f} (<= 0.20)")

# =========================================================================
# 4. SAVE CHECKPOINTS
# =========================================================================
ckpt_dir = Path("results/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

# 1. Full Sample Portfolio Checkpoint
portfolio_ckpt_path = ckpt_dir / f"portfolio_backtests_{data_hash}.json"
with open(portfolio_ckpt_path, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoint_version": "1.2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_hash": data_hash,
        "feature_hash": feat_hash,
        "signal_hash": sig_hash,
        "config_hash": cfg_hash,
        "hypothesis_registry_hash": reg_hash,
        "portfolio_results": portfolio_results,
        "sensitivity_results": sensitivity_results,
    }, f, indent=2)

# 2. Walk-Forward Checkpoint
wf_ckpt_path = ckpt_dir / f"walk_forward_{data_hash}.json"
with open(wf_ckpt_path, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoint_version": "1.2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_hash": data_hash,
        "feature_hash": feat_hash,
        "signal_hash": sig_hash,
        "config_hash": cfg_hash,
        "hypothesis_registry_hash": reg_hash,
        "windows": wf_summary,
        "dev_oos_combined_metrics": dev_comb.to_dict(),
    }, f, indent=2)

# 3. 2024 Final Holdout Checkpoint
holdout_ckpt_path = ckpt_dir / f"final_holdout_2024_{data_hash}.json"
holdout_data = [w for w in wf_summary if w["is_final_holdout"]][0]
with open(holdout_ckpt_path, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoint_version": "1.2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_hash": data_hash,
        "feature_hash": feat_hash,
        "signal_hash": sig_hash,
        "config_hash": cfg_hash,
        "hypothesis_registry_hash": reg_hash,
        "holdout_window": holdout_data,
    }, f, indent=2)

print(f"\nSaved checkpoints:\n  {portfolio_ckpt_path}\n  {wf_ckpt_path}\n  {holdout_ckpt_path}")
