"""Feature engine orchestrator: computes full research feature panel from cleaned market data."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.features.cross_sectional import cross_sectional_rank
from src.features.returns import (
    forward_return,
    log_return,
    simple_return,
    skip_return,
)
from src.features.technical import (
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

logger = logging.getLogger(__name__)


def compute_features(
    df: pd.DataFrame,
    lag: int = 0,
    include_forward_targets: bool = True,
) -> pd.DataFrame:
    """
    Compute full standardized feature set from cleaned market data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned market data with MultiIndex (date, ticker) and columns:
        [open, high, low, close, adj_close, volume, volume_split_adjusted]
    lag : int
        Lag periods to shift feature values forward (default 0).
        When lag > 0, feature at date t uses data from t - lag.
    include_forward_targets : bool
        If True, attaches forward return targets (fwd_ret_1d, fwd_ret_5d, fwd_ret_20d).

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex (date, ticker) containing all features
        conforming to Section 10.3 of the specification.
    """
    if df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Input market data must have MultiIndex (date, ticker).")

    logger.info("Computing features for %d rows...", len(df))
    features = pd.DataFrame(index=df.index)

    # 1. Returns
    features["ret_1d"] = simple_return(df, period=1, column="adj_close")["ret_1d"]
    features["log_ret_1d"] = log_return(df, period=1, column="adj_close")["log_ret_1d"]
    features["ret_5d"] = simple_return(df, period=5, column="adj_close")["ret_5d"]
    features["ret_20d"] = simple_return(df, period=20, column="adj_close")["ret_20d"]
    features["ret_60d"] = simple_return(df, period=60, column="adj_close")["ret_60d"]
    features["ret_252d"] = simple_return(df, period=252, column="adj_close")["ret_252d"]
    features["ret_252d_skip21d"] = skip_return(df, total_period=252, skip_period=21, column="adj_close")["ret_252d_skip21d"]

    # 2. Moving averages
    features["sma_20"] = rolling_mean(df, window=20, column="adj_close")["sma_20"]
    features["sma_60"] = rolling_mean(df, window=60, column="adj_close")["sma_60"]

    # 3. Rolling standard deviations (Strictly distinguishing returns vs prices)
    features["rolling_std_ret_20"] = rolling_std(features, window=20, column="ret_1d")["rolling_std_ret_20"]
    features["rolling_std_ret_60"] = rolling_std(features, window=60, column="ret_1d")["rolling_std_ret_60"]
    features["rolling_std_price_20"] = rolling_std(df, window=20, column="adj_close")["rolling_std_price_20"]

    # 4. Price z-score
    features["zscore_price_20"] = zscore_price(df, window=20, column="adj_close")["zscore_price_20"]

    # 5. Realized volatility
    features["realized_vol_20"] = realized_volatility(features, window=20, annualize=True, column="ret_1d")["realized_vol_20"]
    features["realized_vol_60"] = realized_volatility(features, window=60, annualize=True, column="ret_1d")["realized_vol_60"]
    features["vol_ratio"] = features["realized_vol_20"] / features["realized_vol_60"].replace(0, np.nan)

    # 6. Volume features
    vol_col = "volume_split_adjusted" if "volume_split_adjusted" in df.columns else "volume"
    features["volume_sma_20"] = volume_sma(df, window=20, column=vol_col)["volume_sma_20"]
    features["relative_volume"] = relative_volume(df, window=20, column=vol_col)["relative_volume"]
    features["volume_zscore_20"] = volume_zscore(df, window=20, column=vol_col)["volume_zscore_20"]

    # 7. Cross-sectional rankings
    features["rank_ret_20d"] = cross_sectional_rank(features, column="ret_20d", output_column="rank_ret_20d")["rank_ret_20d"]
    features["rank_ret_252d_skip21d"] = cross_sectional_rank(
        features, column="ret_252d_skip21d", output_column="rank_ret_252d_skip21d"
    )["rank_ret_252d_skip21d"]
    features["rank_vol_20"] = cross_sectional_rank(
        features, column="realized_vol_20", output_column="rank_vol_20"
    )["rank_vol_20"]

    # 8. Apply lag discipline if lag > 0
    if lag > 0:
        grouped = features.groupby(level="ticker", group_keys=False)
        features = grouped.shift(lag)

    # 9. Forward return targets (never lagged)
    if include_forward_targets:
        features["fwd_ret_1d"] = forward_return(df, horizon=1, column="adj_close")["fwd_ret_1d"]
        features["fwd_ret_5d"] = forward_return(df, horizon=5, column="adj_close")["fwd_ret_5d"]
        features["fwd_ret_20d"] = forward_return(df, horizon=20, column="adj_close")["fwd_ret_20d"]

    return features.sort_index()
