"""Configuration schema and validation for experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Data loading and cleaning parameters."""
    universe: str = "sp100_20140101"
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    data_source: str = "yfinance"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    universe_dir: str = "data/universe"


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    name: str = "momentum_12_1"
    params: dict[str, Any] = field(default_factory=lambda: {"lookback": 252, "skip": 21})
    forward_horizon: int = 20
    hypothesis_type: str = "confirmatory"  # 'confirmatory' or 'exploratory'
    ranking_method: str = "percentile"  # 'percentile' or 'zscore'


@dataclass
class CostConfig:
    """Transaction cost model configuration (single-pass accounting)."""
    commission_bps: float = 5.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    min_commission: float = 0.0


@dataclass
class PortfolioConfig:
    """Portfolio construction and exposure constraints."""
    method: str = "vol_scaled"  # 'equal_weight', 'signal_weight', 'vol_scaled'
    gross_exposure: float = 1.0  # Must be <= 1.0 (no leverage)
    max_position_weight: float = 0.10  # 10% maximum per stock
    max_net_exposure: float = 0.20  # 20% maximum net exposure
    long_quantile: int = 5
    short_quantile: int = 1
    n_quantiles: int = 5


@dataclass
class BacktestConfig:
    """Event-driven backtester execution parameters."""
    initial_cash: float = 1_000_000.0
    execution_price: str = "next_adj_close"  # synthetic next-day adjusted close


@dataclass
class WalkForwardConfig:
    """Walk-forward cross-validation configuration."""
    min_train_years: int = 3
    val_years: int = 1
    test_years: int = 1
    step_years: int = 1
    method: str = "expanding"  # 'expanding' or 'rolling'
    final_holdout: bool = True  # Window 7 (2024) reserved as untouched holdout


@dataclass
class ExperimentConfig:
    """Complete configuration specification for an experiment."""
    experiment_id: str = "default_experiment"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    output_dir: str = "results/experiments/default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of configuration."""
        raw_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load YAML configuration file into a validated ExperimentConfig dataclass.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**data.get("data", {}))
    signal_cfg = SignalConfig(**data.get("signal", {}))
    costs_cfg = CostConfig(**data.get("costs", {}))
    portfolio_cfg = PortfolioConfig(**data.get("portfolio", {}))
    backtest_cfg = BacktestConfig(**data.get("backtest", {}))
    walk_forward_cfg = WalkForwardConfig(**data.get("walk_forward", {}))

    # Validate portfolio gross exposure hard constraint
    if portfolio_cfg.gross_exposure > 1.0:
        raise ValueError(
            f"Gross exposure ({portfolio_cfg.gross_exposure}) exceeds 1.0 (no-leverage rule violated)."
        )

    return ExperimentConfig(
        experiment_id=data.get("experiment_id", path.stem),
        random_seed=data.get("random_seed", 42),
        data=data_cfg,
        signal=signal_cfg,
        costs=costs_cfg,
        portfolio=portfolio_cfg,
        backtest=backtest_cfg,
        walk_forward=walk_forward_cfg,
        output_dir=data.get("output_dir", f"results/experiments/{path.stem}"),
    )


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """Save an ExperimentConfig to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def compute_config_hash(config_path_or_obj: str | Path | ExperimentConfig) -> str:
    """Compute SHA-256 hash from file path or ExperimentConfig instance."""
    if isinstance(config_path_or_obj, ExperimentConfig):
        return config_path_or_obj.compute_hash()
    cfg = load_config(config_path_or_obj)
    return cfg.compute_hash()
