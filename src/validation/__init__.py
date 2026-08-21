"""Validation and out-of-sample evaluation package."""

from src.validation.dsr import (
    compute_deflated_sharpe_ratio,
    compute_probabilistic_sharpe_ratio,
)
from src.validation.walk_forward import (
    DEFAULT_WALK_FORWARD_WINDOWS,
    WalkForwardReport,
    WalkForwardValidator,
    WalkForwardWindow,
    WindowEvaluationResult,
)

__all__ = [
    "compute_probabilistic_sharpe_ratio",
    "compute_deflated_sharpe_ratio",
    "WalkForwardWindow",
    "DEFAULT_WALK_FORWARD_WINDOWS",
    "WindowEvaluationResult",
    "WalkForwardReport",
    "WalkForwardValidator",
]
