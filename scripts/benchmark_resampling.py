"""Full-Scale Performance benchmark: Real-scale dimensions (2,768 dates x 100 stocks, B=1000 perm, B=10000 boot)."""

import sys
import time
import tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.statistics.hypothesis_tests import (
    bootstrap_mean_ci,
    permutation_test_ic,
)


def main():
    print("=" * 80)
    print("FULL-SCALE RESAMPLING RUNTIME & PEAK MEMORY AUDIT")
    print("=" * 80)

    # Exact full dataset dimensions
    n_dates = 2768
    n_stocks = 100
    dates = pd.date_range("2014-01-01", periods=n_dates, freq="B")
    tickers = [f"STK_{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    np.random.seed(42)
    sig = pd.Series(np.random.randn(len(idx)), index=idx)
    ret = pd.Series(0.04 * sig.values + np.random.randn(len(idx)), index=idx)
    daily_ic_mock = np.random.randn(n_dates) * 0.08 + 0.025

    print(f"Full-scale test panel: {n_dates:,} trading dates x {n_stocks} securities ({len(idx):,} total observations)")
    print(f"Permutations: B = 1,000 | Bootstrap Replications: B = 10,000 | Seed = 42\n")

    # 1. Full-Scale Permutation Test (B=1000)
    print("--- 1. Full-Scale Permutation Test (B=1,000 on 2,768 dates x 100 assets) ---")
    tracemalloc.start()
    t0 = time.perf_counter()
    obs_ic, p_perm, dist = permutation_test_ic(sig, ret, n_permutations=1000, seed=42, batch_size=500)
    t_perm = time.perf_counter() - t0
    current_mem, peak_mem_perm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Runtime: {t_perm:.3f} seconds")
    print(f"Peak Memory: {peak_mem_perm / (1024 * 1024):.2f} MB")
    print(f"Observed Mean IC: {obs_ic:+.4f} | Permutation p-value: {p_perm:.4f}\n")

    # 2. Full-Scale Stationary Block Bootstrap (B=10,000)
    print("--- 2. Full-Scale Stationary Block Bootstrap (B=10,000 on 2,768 dates) ---")
    tracemalloc.start()
    t0 = time.perf_counter()
    mean_boot, ci_lo, ci_hi = bootstrap_mean_ci(
        daily_ic_mock, n_bootstrap=10_000, block_size=20, seed=42, batch_size=2500
    )
    t_boot = time.perf_counter() - t0
    current_mem, peak_mem_boot = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Runtime: {t_boot:.3f} seconds")
    print(f"Peak Memory: {peak_mem_boot / (1024 * 1024):.2f} MB")
    print(f"Sample Mean: {mean_boot:+.4f} | 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]\n")

    print("=" * 80)
    print("FULL-SCALE SUMMARY")
    print("=" * 80)
    print(f"Single Signal Permutation (B=1000):  {t_perm:.2f}s | Peak Mem: {peak_mem_perm / (1024*1024):.2f} MB")
    print(f"Single Signal Bootstrap (B=10000):   {t_boot:.2f}s | Peak Mem: {peak_mem_boot / (1024*1024):.2f} MB")
    print(f"Total 5 Signals Permutations:        {5 * t_perm:.2f}s")
    print(f"Total 5 Signals Bootstraps:          {5 * t_boot:.2f}s")


if __name__ == "__main__":
    main()
