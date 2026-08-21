"""Statistical testing module: Information Coefficient, HAC inference, Bootstrap, Permutation, Stationarity, and Multiple Testing."""

from src.statistics.hypothesis_tests import (
    bootstrap_mean_ci,
    hac_standard_error,
    hac_t_test,
    permutation_test_ic,
    quintile_spread_analysis,
)
from src.statistics.information_coefficient import compute_ic, ic_summary
from src.statistics.multiple_testing import (
    ConfirmatoryHypothesisRegistry,
    HypothesisSpec,
    adjust_confirmatory_family,
    benjamini_hochberg_fdr,
    bonferroni_correction,
    run_random_signal_null_simulation,
)
from src.statistics.stationarity import adf_test, kpss_test

__all__ = [
    "compute_ic",
    "ic_summary",
    "hac_standard_error",
    "hac_t_test",
    "bootstrap_mean_ci",
    "permutation_test_ic",
    "quintile_spread_analysis",
    "adf_test",
    "kpss_test",
    "ConfirmatoryHypothesisRegistry",
    "HypothesisSpec",
    "bonferroni_correction",
    "benjamini_hochberg_fdr",
    "adjust_confirmatory_family",
    "run_random_signal_null_simulation",
]
