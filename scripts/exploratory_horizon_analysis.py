"""
Exploratory Signal Half-Life & Horizon Profile Analysis.

DISCLAIMER:
This is an EXPLORATORY analysis designed to investigate signal decay dynamics.
It is explicitly NOT part of the pre-specified confirmatory hypothesis family.
It does NOT alter the frozen hypothesis registry, primary confirmatory horizons,
or empirical research conclusions.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.features.returns import forward_return

DATA_PATH = Path("data/processed/cleaned_ohlcv.parquet")
SIGNALS_PATH = Path("data/cache/signals_c3d67525d09fc052.parquet")


def run_exploratory_horizon_analysis() -> pd.DataFrame:
    """Evaluate daily cross-sectional Spearman IC across 1d, 5d, 10d, 20d, 40d, 60d."""
    if not DATA_PATH.exists() or not SIGNALS_PATH.exists():
        raise FileNotFoundError("Processed market data or signal cache missing.")

    df = pd.read_parquet(DATA_PATH)
    signals = pd.read_parquet(SIGNALS_PATH)

    horizons = [1, 5, 10, 20, 40, 60]
    signal_cols = [
        ("h1_mom", "H1: Price Momentum"),
        ("h2_mr", "H2: Mean Reversion"),
        ("h3_vol", "H3: Low Volatility"),
        ("h4_volm", "H4: Abnormal Volume"),
    ]

    records = []

    for h in horizons:
        fwd_ret = forward_return(df["adj_close"], horizon=h)
        for col_name, signal_label in signal_cols:
            ic_series = compute_ic(signals[col_name], fwd_ret)
            summary = ic_summary(ic_series, forward_horizon=h)
            records.append({
                "signal_id": col_name,
                "signal_name": signal_label,
                "horizon_days": h,
                "mean_ic": summary["mean_ic"],
                "ic_std": summary["ic_std"],
                "icir": summary["ic_ir"],
                "hac_t_stat": summary["ic_hac_t_stat"],
                "hac_p_value": summary["ic_hac_p_value"],
                "hit_rate": summary["ic_hit_rate"],
                "n_days": summary["n_days"],
            })

    res_df = pd.DataFrame(records)
    return res_df


if __name__ == "__main__":
    print("================================================================================")
    print("EXPLORATORY SIGNAL HORIZON PROFILE (1d to 60d)")
    print("Note: Exploratory analysis only — not part of confirmatory testing family.")
    print("================================================================================")
    df = run_exploratory_horizon_analysis()
    
    # Pivot table for display
    pivot = df.pivot(index="signal_name", columns="horizon_days", values="mean_ic")
    print("\nMean Rank IC by Forward Return Horizon:")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nHAC p-values by Forward Return Horizon:")
    pivot_p = df.pivot(index="signal_name", columns="horizon_days", values="hac_p_value")
    print(pivot_p.to_string(float_format=lambda x: f"{x:.4f}"))
