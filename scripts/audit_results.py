import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.features.engine import compute_features
from src.signals import (
    MomentumSignal,
    MeanReversionSignal,
    VolatilitySignal,
    AbnormalVolumeSignal,
    CombinedSignal,
)
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.statistics.multiple_testing import (
    ConfirmatoryHypothesisRegistry,
    adjust_confirmatory_family,
)
from src.statistics.hypothesis_tests import permutation_test_ic


def main():
    print("Loading data...")
    df = pd.read_parquet("data/processed/cleaned_ohlcv.parquet")
    print("Computing features...")
    feat = compute_features(df, lag=0, include_forward_targets=True)

    print("Computing signals...")
    h1 = MomentumSignal().compute(feat)
    h2 = MeanReversionSignal().compute(feat)
    h3 = VolatilitySignal().compute(feat)
    h4 = AbnormalVolumeSignal().compute(feat)
    comb = CombinedSignal().compute(feat)

    # Compute daily IC series
    ic_h1 = compute_ic(h1, feat["fwd_ret_20d"])
    ic_h2_5d = compute_ic(h2, feat["fwd_ret_5d"])
    ic_h2_1d = compute_ic(h2, feat["fwd_ret_1d"])
    ic_h3 = compute_ic(h3, feat["fwd_ret_20d"])
    ic_h4_5d = compute_ic(h4, feat["fwd_ret_5d"])
    ic_h4_1d = compute_ic(h4, feat["fwd_ret_1d"])
    ic_comb = compute_ic(comb, feat["fwd_ret_20d"])

    # Summaries with horizon-matched HAC lag
    s_h1 = ic_summary(ic_h1, forward_horizon=20)
    s_h2_5d = ic_summary(ic_h2_5d, forward_horizon=5)
    s_h2_1d = ic_summary(ic_h2_1d, forward_horizon=1)
    s_h3 = ic_summary(ic_h3, forward_horizon=20)
    s_h4_5d = ic_summary(ic_h4_5d, forward_horizon=5)
    s_h4_1d = ic_summary(ic_h4_1d, forward_horizon=1)
    s_comb = ic_summary(ic_comb, forward_horizon=20)

    # Permutation tests
    print("Running permutation tests...")
    perm_h1 = permutation_test_ic(h1, feat["fwd_ret_20d"], n_permutations=200, seed=42)
    perm_h2_5d = permutation_test_ic(h2, feat["fwd_ret_5d"], n_permutations=200, seed=42)
    perm_h3 = permutation_test_ic(h3, feat["fwd_ret_20d"], n_permutations=200, seed=42)
    perm_h4_5d = permutation_test_ic(h4, feat["fwd_ret_5d"], n_permutations=200, seed=42)

    print("\n" + "=" * 80)
    print("EXACT IC METRICS ON FROZEN REGISTRY (H1:20d, H2:5d, H3:20d, H4:5d)")
    print("=" * 80)
    print(f"H1 (20d): Mean IC={s_h1['mean_ic']:+.4f}, Std={s_h1['ic_std']:.4f}, IR={s_h1['ic_ir']:.3f}, HAC t={s_h1['ic_hac_t_stat']:+.3f}, HAC p={s_h1['ic_hac_p_value']:.6f}, Perm p={perm_h1:.4f}")
    print(f"H2 (5d):  Mean IC={s_h2_5d['mean_ic']:+.4f}, Std={s_h2_5d['ic_std']:.4f}, IR={s_h2_5d['ic_ir']:.3f}, HAC t={s_h2_5d['ic_hac_t_stat']:+.3f}, HAC p={s_h2_5d['ic_hac_p_value']:.6f}, Perm p={perm_h2_5d:.4f}")
    print(f"H3 (20d): Mean IC={s_h3['mean_ic']:+.4f}, Std={s_h3['ic_std']:.4f}, IR={s_h3['ic_ir']:.3f}, HAC t={s_h3['ic_hac_t_stat']:+.3f}, HAC p={s_h3['ic_hac_p_value']:.6f}, Perm p={perm_h3:.4f}")
    print(f"H4 (5d):  Mean IC={s_h4_5d['mean_ic']:+.4f}, Std={s_h4_5d['ic_std']:.4f}, IR={s_h4_5d['ic_ir']:.3f}, HAC t={s_h4_5d['ic_hac_t_stat']:+.3f}, HAC p={s_h4_5d['ic_hac_p_value']:.6f}, Perm p={perm_h4_5d:.4f}")
    print(f"Comb (20d): Mean IC={s_comb['mean_ic']:+.4f}, Std={s_comb['ic_std']:.4f}, IR={s_comb['ic_ir']:.3f}, HAC t={s_comb['ic_hac_t_stat']:+.3f}, HAC p={s_comb['ic_hac_p_value']:.6f}")

    print("\n" + "=" * 80)
    print("EXPLORATORY 1-DAY HORIZON COMPARISON FOR H2 & H4")
    print("=" * 80)
    print(f"H2 (1d):  Mean IC={s_h2_1d['mean_ic']:+.4f}, Std={s_h2_1d['ic_std']:.4f}, IR={s_h2_1d['ic_ir']:.3f}, HAC t={s_h2_1d['ic_hac_t_stat']:+.3f}, HAC p={s_h2_1d['ic_hac_p_value']:.6f}")
    print(f"H4 (1d):  Mean IC={s_h4_1d['mean_ic']:+.4f}, Std={s_h4_1d['ic_std']:.4f}, IR={s_h4_1d['ic_ir']:.3f}, HAC t={s_h4_1d['ic_hac_t_stat']:+.3f}, HAC p={s_h4_1d['ic_hac_p_value']:.6f}")

    print("\n" + "=" * 80)
    print("CONFIRMATORY MULTIPLE TESTING ADJUSTMENTS (m=4, Frozen Horizons)")
    print("=" * 80)
    p_dict = {
        "H1_MOMENTUM": s_h1["ic_hac_p_value"],
        "H2_MEAN_REVERSION": s_h2_5d["ic_hac_p_value"],
        "H3_LOW_VOLATILITY": s_h3["ic_hac_p_value"],
        "H4_ABNORMAL_VOLUME": s_h4_5d["ic_hac_p_value"],
    }
    adj = adjust_confirmatory_family(p_dict, alpha_fwer=0.05, alpha_fdr=0.10)
    print(adj[["raw_p", "bonferroni_p", "bonferroni_reject", "bh_fdr_p", "bh_reject"]].to_string())

    reg = ConfirmatoryHypothesisRegistry()
    print(f"\nRegistry Hash: {reg.registry_hash()}")


if __name__ == "__main__":
    main()
