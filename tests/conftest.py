"""Shared pytest fixtures and synthetic test datasets."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config.schema import ExperimentConfig, load_config


@pytest.fixture
def sample_toy_prices() -> pd.DataFrame:
    """
    3 tickers across 5 trading days with known hand-calculable prices.
    """
    dates = pd.date_range("2020-01-06", periods=5, freq="B")
    data = []
    
    # AAPL: steadily rising
    aapl_prices = [100.0, 102.0, 105.0, 107.0, 110.0]
    # MSFT: mean reverting
    msft_prices = [200.0, 205.0, 195.0, 198.0, 202.0]
    # GE: volatile / downward
    ge_prices = [50.0, 48.0, 47.0, 46.0, 45.0]

    for d_idx, d in enumerate(dates):
        data.append({
            "date": d, "ticker": "AAPL",
            "open": aapl_prices[d_idx] - 0.5, "high": aapl_prices[d_idx] + 1.0,
            "low": aapl_prices[d_idx] - 1.0, "close": aapl_prices[d_idx],
            "adj_close": aapl_prices[d_idx], "volume": 1000000,
            "volume_split_adjusted": 1000000.0,
        })
        data.append({
            "date": d, "ticker": "MSFT",
            "open": msft_prices[d_idx] - 0.5, "high": msft_prices[d_idx] + 1.0,
            "low": msft_prices[d_idx] - 1.0, "close": msft_prices[d_idx],
            "adj_close": msft_prices[d_idx], "volume": 2000000,
            "volume_split_adjusted": 2000000.0,
        })
        data.append({
            "date": d, "ticker": "GE",
            "open": ge_prices[d_idx] - 0.5, "high": ge_prices[d_idx] + 1.0,
            "low": ge_prices[d_idx] - 1.0, "close": ge_prices[d_idx],
            "adj_close": ge_prices[d_idx], "volume": 3000000,
            "volume_split_adjusted": 3000000.0,
        })

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def default_experiment_config() -> ExperimentConfig:
    """Load default experiment configuration from configs/default.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    return load_config(config_path)
