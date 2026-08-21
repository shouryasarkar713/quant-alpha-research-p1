"""Feature engineering library: returns, technical indicators, volume features, cross-sectional rankings."""

from src.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from src.features.engine import compute_features
from src.features.returns import (
    forward_return,
    log_return,
    simple_return,
    skip_return,
)
from src.features.technical import (
    classify_regime,
    realized_volatility,
    rolling_mean,
    rolling_std,
    zscore_price,
)
from src.features.volume import (
    relative_volume,
    volume_sma,
    volume_zscore,
)

__all__ = [
    "simple_return",
    "log_return",
    "skip_return",
    "forward_return",
    "rolling_mean",
    "rolling_std",
    "zscore_price",
    "realized_volatility",
    "classify_regime",
    "volume_sma",
    "relative_volume",
    "volume_zscore",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "compute_features",
]
