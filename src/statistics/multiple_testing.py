"""Multiple hypothesis testing corrections and frozen confirmatory registry.

Distinction between error rate controls:
1. Family-Wise Error Rate (FWER) via Bonferroni:
   Controls the probability of making AT LEAST ONE Type I error (false positive) across
   the predefined four-hypothesis confirmatory family:
       P(V >= 1) <= alpha,  where p_adj = min(m * p, 1.0), with m = 4
   Highly conservative; ideal for confirmatory alpha validation.

2. False Discovery Rate (FDR) via Benjamini-Hochberg (1995):
   Controls the expected proportion of false discoveries among all rejected hypotheses:
       FDR = E[V / max(R, 1)] <= alpha
   Less conservative than Bonferroni while guarding against data mining.

PRIMARY INFERENCE CONVENTION:
The confirmatory family consists of exactly FOUR primary hypotheses (m=4), where the single
primary test statistic for each hypothesis is the HAC/Newey-West Information Coefficient p-value:
    H1: Momentum        -> HAC IC p-value
    H2: Mean Reversion  -> HAC IC p-value
    H3: Low Volatility  -> HAC IC p-value
    H4: Abnormal Volume -> HAC IC p-value

The within-date cross-sectional permutation test is designated as a secondary non-parametric
robustness check and is NOT an additional member of the confirmatory multiple-testing family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class HypothesisSpec:
    """Immutable specification for a single statistical hypothesis."""

    id: str
    signal_name: str
    description: str
    primary_params: dict[str, Any]
    forward_horizon: int
    predicted_direction: str  # 'positive' or 'negative'
    primary_test: str = "hac_ic_t_test"  # Single primary confirmatory test
    is_confirmatory: bool = True
    secondary_tests: tuple[str, ...] = ("within_date_permutation_test", "stationary_bootstrap_ci")


@dataclass(frozen=True)
class ConfirmatoryHypothesisRegistry:
    """
    Frozen registry containing strictly the four pre-specified primary hypotheses (m=4).

    Per Section 15 of project specification v1.2, only 4 primary hypotheses are confirmatory:
    - H1: Momentum (12-1) -> HAC IC p-value
    - H2: Short-term price mean reversion (20-day z-score) -> HAC IC p-value
    - H3: Low volatility anomaly (60-day realized vol) -> HAC IC p-value
    - H4: Abnormal volume continuation (20-day rvol + 1-day direction) -> HAC IC p-value

    This registry is frozen and immutable to prevent post-hoc modification or data dredging.
    """

    hypotheses: tuple[HypothesisSpec, ...] = field(
        default_factory=lambda: (
            HypothesisSpec(
                id="H1_MOMENTUM",
                signal_name="momentum_12_1",
                description="Past 12-month return skipping last month positively predicts 20-day forward return",
                primary_params={"lookback_days": 252, "skip_days": 21, "ranking_method": "percentile"},
                forward_horizon=20,
                predicted_direction="positive",
                primary_test="hac_ic_t_test",
                is_confirmatory=True,
            ),
            HypothesisSpec(
                id="H2_MEAN_REVERSION",
                signal_name="mean_reversion_zscore",
                description="Oversold 20-day price z-score positively predicts 5-day forward return (reversion)",
                primary_params={"lookback_days": 20, "threshold": None, "ranking_method": "percentile"},
                forward_horizon=5,
                predicted_direction="positive",
                primary_test="hac_ic_t_test",
                is_confirmatory=True,
            ),
            HypothesisSpec(
                id="H3_LOW_VOLATILITY",
                signal_name="low_vol",
                description="Low 60-day realized return volatility positively predicts 20-day forward risk-adjusted return",
                primary_params={"lookback_days": 60, "ranking_method": "percentile"},
                forward_horizon=20,
                predicted_direction="positive",
                primary_test="hac_ic_t_test",
                is_confirmatory=True,
            ),
            HypothesisSpec(
                id="H4_ABNORMAL_VOLUME",
                signal_name="abnormal_volume",
                description="Abnormal volume with price direction positively predicts 5-day continuation return",
                primary_params={"volume_lookback": 20, "direction_horizon": 1, "ranking_method": "percentile"},
                forward_horizon=5,
                predicted_direction="positive",
                primary_test="hac_ic_t_test",
                is_confirmatory=True,
            ),
        )
    )

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisSpec:
        """Lookup hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id or h.signal_name == hypothesis_id:
                return h
        raise KeyError(f"Hypothesis '{hypothesis_id}' not found in confirmatory registry.")

    @property
    def family_size(self) -> int:
        """Number of confirmatory hypotheses in the family (strictly m=4)."""
        return len(self.hypotheses)

    def get_target_column(self, hypothesis_id: str, available_columns: Sequence[str] | None = None) -> str:
        """
        Derive the required forward target column name for a given hypothesis.

        Guarantees that the evaluated target horizon strictly matches the frozen hypothesis definition.
        Fails loudly if available_columns is provided and the target column does not exist.
        """
        spec = self.get_hypothesis(hypothesis_id)
        target_col = f"fwd_ret_{spec.forward_horizon}d"
        if available_columns is not None and target_col not in available_columns:
            raise ValueError(
                f"Target column '{target_col}' required for hypothesis '{spec.id}' (horizon {spec.forward_horizon}d) "
                f"is missing from available features: {list(available_columns)}"
            )
        return target_col

    def evaluate_hypothesis_ic(
        self,
        hypothesis_id: str,
        signal: pd.DataFrame | pd.Series,
        features_df: pd.DataFrame,
        method: str = "spearman",
        min_obs: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate a confirmatory hypothesis IC against its frozen forward-return target horizon.

        Derives the required target column directly from the frozen registry, extracts the target series,
        and computes daily IC series and IC summary diagnostics with horizon-matched HAC lag.
        """
        from src.statistics.information_coefficient import compute_ic, ic_summary

        spec = self.get_hypothesis(hypothesis_id)
        target_col = self.get_target_column(hypothesis_id, available_columns=features_df.columns)
        target_series = features_df[target_col]

        ic_series = compute_ic(signal, target_series, method=method, min_obs=min_obs)
        summary = ic_summary(ic_series, forward_horizon=spec.forward_horizon)
        summary["hypothesis_id"] = spec.id
        summary["signal_name"] = spec.signal_name
        summary["forward_horizon"] = spec.forward_horizon
        summary["target_column"] = target_col
        summary["ic_series"] = ic_series
        return summary

    def registry_hash(self) -> str:
        """Deterministic SHA-256 hash of the frozen confirmatory hypothesis registry for auditability."""
        payload = [
            {
                "id": h.id,
                "signal_name": h.signal_name,
                "params": h.primary_params,
                "horizon": h.forward_horizon,
                "primary_test": h.primary_test,
            }
            for h in self.hypotheses
        ]
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def bonferroni_correction(
    p_values: Sequence[float] | pd.Series | dict[str, float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Bonferroni Family-Wise Error Rate (FWER) correction:
        p_adj = min(m * p, 1.0)
        significant = p_adj < alpha

    Parameters
    ----------
    p_values : Sequence[float] | pd.Series | dict[str, float]
        Raw p-values from statistical tests.
    alpha : float
        Target significance level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Table with columns: ['raw_p_value', 'bonferroni_p_value', 'significant_fwer']
    """
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        p_arr = np.array([p_values[k] for k in keys], dtype=float)
    elif isinstance(p_values, pd.Series):
        keys = list(p_values.index)
        p_arr = p_values.values.astype(float)
    else:
        p_arr = np.asarray(p_values, dtype=float)
        keys = [f"H_{i+1}" for i in range(len(p_arr))]

    m = len(p_arr)
    if m == 0:
        return pd.DataFrame(columns=["raw_p_value", "bonferroni_p_value", "significant_fwer"])

    # Compute p_adj = min(m * p, 1.0)
    p_adj = np.minimum(m * p_arr, 1.0)
    p_adj = np.maximum(p_adj, 0.0)

    sig = p_adj < alpha

    return pd.DataFrame(
        {
            "raw_p_value": p_arr,
            "bonferroni_p_value": p_adj,
            "significant_fwer": sig,
        },
        index=keys,
    )


def benjamini_hochberg_fdr(
    p_values: Sequence[float] | pd.Series | dict[str, float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg (1995) False Discovery Rate (FDR) correction:
        FDR = E[V / max(R, 1)] <= alpha

    Procedure:
    1. Sort p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m).
    2. Compute raw q-values: q_(i) = (m / i) * p_(i).
    3. Enforce monotonicity backwards: q_(i) = min(q_(i), q_(i+1)).
    4. Cap at 1.0: q_(i) = min(q_(i), 1.0).
    5. Re-sort back to original input order.

    Parameters
    ----------
    p_values : Sequence[float] | pd.Series | dict[str, float]
        Raw p-values from statistical tests.
    alpha : float
        Target FDR level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Table with columns: ['raw_p_value', 'fdr_q_value', 'rank', 'significant_fdr']
    """
    if isinstance(p_values, dict):
        keys = list(p_values.keys())
        p_arr = np.array([p_values[k] for k in keys], dtype=float)
    elif isinstance(p_values, pd.Series):
        keys = list(p_values.index)
        p_arr = p_values.values.astype(float)
    else:
        p_arr = np.asarray(p_values, dtype=float)
        keys = [f"H_{i+1}" for i in range(len(p_arr))]

    m = len(p_arr)
    if m == 0:
        return pd.DataFrame(columns=["raw_p_value", "fdr_q_value", "rank", "significant_fdr"])

    # Sort indices
    sorted_order = np.argsort(p_arr)
    sorted_p = p_arr[sorted_order]

    # Rank 1 to m
    ranks = np.arange(1, m + 1)

    # Raw q-values
    q_vals = (m / ranks) * sorted_p

    # Enforce right-to-left monotonicity: q_(i) = min(q_(i), q_(i+1))
    for i in range(m - 2, -1, -1):
        q_vals[i] = min(q_vals[i], q_vals[i + 1])

    # Cap at 1.0 and floor at 0.0
    q_vals = np.clip(q_vals, 0.0, 1.0)

    # Re-order to original positions
    orig_q = np.empty(m, dtype=float)
    orig_q[sorted_order] = q_vals

    orig_ranks = np.empty(m, dtype=int)
    orig_ranks[sorted_order] = ranks

    sig = orig_q < alpha

    return pd.DataFrame(
        {
            "raw_p_value": p_arr,
            "fdr_q_value": orig_q,
            "rank": orig_ranks,
            "significant_fdr": sig,
        },
        index=keys,
    )


def adjust_confirmatory_family(
    test_results: dict[str, float],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Simultaneously apply Bonferroni and Benjamini-Hochberg FDR adjustments to the
    four confirmatory primary HAC IC p-values.

    Parameters
    ----------
    test_results : dict[str, float]
        Dictionary mapping hypothesis ID (e.g. 'H1_MOMENTUM') to its single primary HAC p-value.
        Must contain exactly 4 hypotheses corresponding to the confirmatory family.
    alpha : float
        Target significance level (default 0.05).

    Returns
    -------
    pd.DataFrame
        Comprehensive adjustment summary table.
    """
    if len(test_results) != 4:
        raise ValueError(
            f"Confirmatory multiple-testing family must contain exactly 4 primary hypotheses. Found {len(test_results)}."
        )

    bonf = bonferroni_correction(test_results, alpha=alpha)
    bh = benjamini_hochberg_fdr(test_results, alpha=alpha)

    summary = pd.DataFrame(
        {
            "raw_p_value": bonf["raw_p_value"],
            "bonferroni_p_value": bonf["bonferroni_p_value"],
            "significant_bonferroni": bonf["significant_fwer"],
            "bh_fdr_q_value": bh["fdr_q_value"],
            "significant_bh_fdr": bh["significant_fdr"],
        },
        index=bonf.index,
    )
    return summary


def _fast_spearman_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fast vectorized daily Spearman rank correlation between X and Y matrices (n_days, n_assets)."""
    rx = np.argsort(np.argsort(X, axis=1), axis=1).astype(float)
    ry = np.argsort(np.argsort(Y, axis=1), axis=1).astype(float)
    rx_d = rx - rx.mean(axis=1, keepdims=True)
    ry_d = ry - ry.mean(axis=1, keepdims=True)
    cov = (rx_d * ry_d).sum(axis=1)
    std_x = np.sqrt((rx_d ** 2).sum(axis=1))
    std_y = np.sqrt((ry_d ** 2).sum(axis=1))
    denom = std_x * std_y
    denom[denom == 0] = 1e-12
    return cov / denom


def run_random_signal_null_simulation(
    n_signals: int = 100,
    n_days: int = 500,
    n_assets: int = 50,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """
    100-Random-Signal Null Demonstration (Specification Section 15.3 & 28.2).

    Theoretical Property:
    Under the ideal independent null setup where all M=100 signals are pure noise:
        E[V] = M * alpha = 100 * 0.05 = 5.0 false discoveries on average,
        with binomial standard deviation sigma = sqrt(100 * 0.05 * 0.95) = 2.18.
    The exact number of false discoveries in any single simulation run is stochastic.
    Bonferroni and Benjamini-Hochberg FDR adjustments tightly control false discoveries.

    Parameters
    ----------
    n_signals : int
        Number of independent null signals (default 100).
    n_days : int
        Number of simulated trading days (default 500).
    n_assets : int
        Number of assets per cross-section (default 50).
    alpha : float
        Significance threshold (default 0.05).
    seed : int
        Random seed for deterministic reproducibility.

    Returns
    -------
    dict[str, Any]
        Results containing raw p-values, adjustment table, and rejection counts.
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(0, 0.015, size=(n_days, n_assets))

    raw_p_values = np.zeros(n_signals)
    mean_ics = np.zeros(n_signals)

    for k in range(n_signals):
        signal = rng.normal(0, 1, size=(n_days, n_assets))
        daily_ics = _fast_spearman_matrix(signal, returns)

        mean_ic = float(np.mean(daily_ics))
        mean_ics[k] = mean_ic
        std_ic = float(np.std(daily_ics, ddof=1))
        se_ic = std_ic / np.sqrt(n_days) if std_ic > 0 else 1.0
        t_stat = mean_ic / se_ic
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        raw_p_values[k] = p_val

    bonf = bonferroni_correction(raw_p_values, alpha=alpha)
    bh = benjamini_hochberg_fdr(raw_p_values, alpha=alpha)

    raw_false_positives = int((raw_p_values < alpha).sum())
    bonf_discoveries = int(bonf["significant_fwer"].sum())
    bh_discoveries = int(bh["significant_fdr"].sum())

    return {
        "n_signals": n_signals,
        "alpha": alpha,
        "seed": seed,
        "raw_p_values": raw_p_values,
        "mean_ics": mean_ics,
        "expected_false_positives": float(n_signals * alpha),
        "raw_false_positives": raw_false_positives,
        "bonferroni_discoveries": bonf_discoveries,
        "bh_fdr_discoveries": bh_discoveries,
        "adjustment_table": pd.DataFrame({
            "raw_p_value": raw_p_values,
            "bonferroni_p_value": bonf["bonferroni_p_value"].values,
            "bh_fdr_q_value": bh["fdr_q_value"].values,
            "significant_raw": raw_p_values < alpha,
            "significant_bonf": bonf["significant_fwer"].values,
            "significant_bh": bh["significant_fdr"].values,
        }),
    }
