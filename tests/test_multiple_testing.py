"""Unit and property tests for multiple testing adjustments, frozen confirmatory registry, and null simulation."""

import dataclasses
import numpy as np
import pandas as pd
import pytest

from src.statistics.multiple_testing import (
    ConfirmatoryHypothesisRegistry,
    HypothesisSpec,
    adjust_confirmatory_family,
    benjamini_hochberg_fdr,
    bonferroni_correction,
    run_random_signal_null_simulation,
)


def test_bonferroni_known_values():
    """
    Given p = [0.01, 0.04, 0.20, 0.50] with m = 4:
    p_adj = [0.04, 0.16, 0.80, 1.00]
    At alpha = 0.05: only H_1 (0.04) is significant.
    """
    p_raw = [0.01, 0.04, 0.20, 0.50]
    res = bonferroni_correction(p_raw, alpha=0.05)

    expected_adj = [0.04, 0.16, 0.80, 1.00]
    np.testing.assert_allclose(res["bonferroni_p_value"].values, expected_adj, atol=1e-8)
    assert res["significant_fwer"].tolist() == [True, False, False, False]


def test_bh_fdr_known_values_and_monotonicity():
    """
    Benjamini-Hochberg FDR calculation with known inputs and monotonicity check:
    Raw p: [0.01, 0.03, 0.035, 0.20] with m = 4.
    Ranks: 1, 2, 3, 4
    Raw q: [4/1 * 0.01, 4/2 * 0.03, 4/3 * 0.035, 4/4 * 0.20]
         = [0.04, 0.06, 0.046667, 0.20]
    Enforcing monotonicity backwards:
    q_3 = min(0.046667, 0.20) = 0.046667
    q_2 = min(0.06, 0.046667) = 0.046667  <- adjusted down by monotonicity!
    q_1 = min(0.04, 0.046667) = 0.04
    """
    p_raw = [0.01, 0.03, 0.035, 0.20]
    res = benjamini_hochberg_fdr(p_raw, alpha=0.05)

    expected_q = [0.04, 0.04666667, 0.04666667, 0.20]
    np.testing.assert_allclose(res["fdr_q_value"].values, expected_q, atol=1e-6)
    assert res["significant_fdr"].tolist() == [True, True, True, False]


def test_multiple_testing_edge_cases():
    """Verify behavior on extreme edge cases: p=0, p=1, duplicate p-values."""
    res_extremes = bonferroni_correction([0.0, 1.0], alpha=0.05)
    assert res_extremes.loc["H_1", "bonferroni_p_value"] == 0.0
    assert res_extremes.loc["H_2", "bonferroni_p_value"] == 1.0

    res_bh_extremes = benjamini_hochberg_fdr([0.0, 1.0], alpha=0.05)
    assert res_bh_extremes.loc["H_1", "fdr_q_value"] == 0.0
    assert res_bh_extremes.loc["H_2", "fdr_q_value"] == 1.0

    res_dup = benjamini_hochberg_fdr([0.02, 0.02, 0.10], alpha=0.05)
    assert np.isclose(res_dup.loc["H_1", "fdr_q_value"], res_dup.loc["H_2", "fdr_q_value"])


def test_frozen_confirmatory_hypothesis_registry():
    """
    Verify ConfirmatoryHypothesisRegistry is immutable, frozen with exactly 4 hypotheses,
    designates HAC IC p-value as the single primary test, and computes deterministic hash.
    """
    reg = ConfirmatoryHypothesisRegistry()
    assert reg.family_size == 4

    expected_ids = ["H1_MOMENTUM", "H2_MEAN_REVERSION", "H3_LOW_VOLATILITY", "H4_ABNORMAL_VOLUME"]
    actual_ids = [h.id for h in reg.hypotheses]
    assert actual_ids == expected_ids

    for h in reg.hypotheses:
        assert h.primary_test == "hac_ic_t_test", "Primary confirmatory test must strictly be HAC IC t-test."
        assert h.is_confirmatory is True

    # Auditability hash
    h_hash = reg.registry_hash()
    assert isinstance(h_hash, str)
    assert len(h_hash) == 64  # SHA-256 hex string

    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        reg.hypotheses = ()  # type: ignore

    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        reg.hypotheses[0].forward_horizon = 10  # type: ignore


def test_adjust_confirmatory_family_pipeline():
    """
    Verify simultaneous adjustment strictly on the 4 confirmatory primary HAC IC p-values.
    Rejects input with size != 4.
    """
    raw_p_dict = {
        "H1_MOMENTUM": 0.005,
        "H2_MEAN_REVERSION": 0.020,
        "H3_LOW_VOLATILITY": 0.080,
        "H4_ABNORMAL_VOLUME": 0.350,
    }
    df_adj = adjust_confirmatory_family(raw_p_dict, alpha=0.05)

    assert len(df_adj) == 4
    # H1: raw 0.005 -> Bonf 0.020 (<0.05 True), BH 0.020 (<0.05 True)
    assert bool(df_adj.loc["H1_MOMENTUM", "significant_bonferroni"]) is True
    assert bool(df_adj.loc["H1_MOMENTUM", "significant_bh_fdr"]) is True

    # H2: raw 0.020 -> Bonf 0.080 (>0.05 False), BH (4/2 * 0.02 = 0.040 < 0.05 True)
    assert bool(df_adj.loc["H2_MEAN_REVERSION", "significant_bonferroni"]) is False
    assert bool(df_adj.loc["H2_MEAN_REVERSION", "significant_bh_fdr"]) is True

    # Rejects non-4-hypothesis dictionary
    with pytest.raises(ValueError, match="must contain exactly 4"):
        adjust_confirmatory_family({"H1": 0.01, "H2": 0.02}, alpha=0.05)


def test_random_signal_null_simulation_reproducibility():
    """
    Verify 100-random-signal null simulation executes deterministically,
    records E[V] = M * alpha = 5.0, and controls false discoveries.
    """
    res1 = run_random_signal_null_simulation(n_signals=100, n_days=300, n_assets=30, alpha=0.05, seed=42)
    res2 = run_random_signal_null_simulation(n_signals=100, n_days=300, n_assets=30, alpha=0.05, seed=42)

    assert res1["expected_false_positives"] == 5.0
    assert res1["raw_false_positives"] == res2["raw_false_positives"]
    assert res1["bonferroni_discoveries"] == res2["bonferroni_discoveries"]
    assert res1["bh_fdr_discoveries"] == res2["bh_fdr_discoveries"]
    np.testing.assert_allclose(res1["raw_p_values"], res2["raw_p_values"], atol=1e-12)

    assert res1["bonferroni_discoveries"] <= res1["raw_false_positives"]
    assert res1["bh_fdr_discoveries"] <= res1["raw_false_positives"]


def test_confirmatory_registry_horizon_consistency():
    """
    Test exact frozen confirmatory horizons:
    1. H1 Momentum == 20d
    2. H2 Mean Reversion == 5d
    3. H3 Low Volatility == 20d
    4. H4 Abnormal Volume == 5d
    """
    reg = ConfirmatoryHypothesisRegistry()

    h1 = reg.get_hypothesis("H1_MOMENTUM")
    h2 = reg.get_hypothesis("H2_MEAN_REVERSION")
    h3 = reg.get_hypothesis("H3_LOW_VOLATILITY")
    h4 = reg.get_hypothesis("H4_ABNORMAL_VOLUME")

    assert h1.forward_horizon == 20, f"H1 horizon must be 20d, found {h1.forward_horizon}"
    assert h2.forward_horizon == 5, f"H2 horizon must be 5d, found {h2.forward_horizon}"
    assert h3.forward_horizon == 20, f"H3 horizon must be 20d, found {h3.forward_horizon}"
    assert h4.forward_horizon == 5, f"H4 horizon must be 5d, found {h4.forward_horizon}"

    # Target column derivation
    assert reg.get_target_column("H1_MOMENTUM") == "fwd_ret_20d"
    assert reg.get_target_column("H2_MEAN_REVERSION") == "fwd_ret_5d"
    assert reg.get_target_column("H3_LOW_VOLATILITY") == "fwd_ret_20d"
    assert reg.get_target_column("H4_ABNORMAL_VOLUME") == "fwd_ret_5d"


def test_registry_pipeline_target_column_missing_loud_failure():
    """
    Verify the pipeline target selection fails loudly if a required forward return target column is missing.
    """
    reg = ConfirmatoryHypothesisRegistry()
    available_cols = ["open", "high", "low", "close", "fwd_ret_1d", "fwd_ret_20d"]

    # H1 needs fwd_ret_20d -> exists
    assert reg.get_target_column("H1_MOMENTUM", available_columns=available_cols) == "fwd_ret_20d"

    # H2 needs fwd_ret_5d -> missing -> must raise ValueError
    with pytest.raises(ValueError, match="Target column 'fwd_ret_5d' required for hypothesis 'H2_MEAN_REVERSION'"):
        reg.get_target_column("H2_MEAN_REVERSION", available_columns=available_cols)

    # H4 needs fwd_ret_5d -> missing -> must raise ValueError
    with pytest.raises(ValueError, match="Target column 'fwd_ret_5d' required for hypothesis 'H4_ABNORMAL_VOLUME'"):
        reg.get_target_column("H4_ABNORMAL_VOLUME", available_columns=available_cols)


def test_evaluate_hypothesis_ic_dynamic_horizon():
    """
    Verify evaluate_hypothesis_ic extracts the exact target column derived from the hypothesis specification.
    """
    reg = ConfirmatoryHypothesisRegistry()

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    np.random.seed(42)
    sig_series = pd.Series(np.random.randn(len(idx)), index=idx)
    feat_df = pd.DataFrame(
        {
            "fwd_ret_5d": np.random.randn(len(idx)) * 0.02,
            "fwd_ret_20d": np.random.randn(len(idx)) * 0.04,
        },
        index=idx,
    )

    res_h1 = reg.evaluate_hypothesis_ic("H1_MOMENTUM", sig_series, feat_df)
    assert res_h1["target_column"] == "fwd_ret_20d"
    assert res_h1["forward_horizon"] == 20
    assert "mean_ic" in res_h1
    assert "ic_hac_t_stat" in res_h1

    res_h2 = reg.evaluate_hypothesis_ic("H2_MEAN_REVERSION", sig_series, feat_df)
    assert res_h2["target_column"] == "fwd_ret_5d"
    assert res_h2["forward_horizon"] == 5

