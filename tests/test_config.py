"""Unit tests for configuration subsystem and schema validation."""

import pytest
from pathlib import Path
from src.config.schema import (
    BacktestConfig,
    CostConfig,
    DataConfig,
    ExperimentConfig,
    PortfolioConfig,
    SignalConfig,
    WalkForwardConfig,
    compute_config_hash,
    load_config,
    save_config,
)


def test_default_config_instantiation():
    """Verify default dataclasses initialize with expected values."""
    cfg = ExperimentConfig()
    assert cfg.random_seed == 42
    assert cfg.data.universe == "sp100_20140101"
    assert cfg.signal.name == "momentum_12_1"
    assert cfg.costs.commission_bps == 5.0
    assert cfg.portfolio.gross_exposure == 1.0
    assert cfg.portfolio.max_position_weight == 0.10
    assert cfg.portfolio.max_net_exposure == 0.20
    assert cfg.backtest.initial_cash == 1_000_000.0
    assert cfg.walk_forward.final_holdout is True


def test_gross_exposure_leverage_rejection(tmp_path):
    """Verify hard error when gross exposure exceeds 1.0."""
    invalid_yaml = tmp_path / "invalid_leverage.yaml"
    invalid_yaml.write_text(
        """
experiment_id: "leverage_test"
portfolio:
  gross_exposure: 1.5
"""
    )
    with pytest.raises(ValueError, match="Gross exposure .* exceeds 1.0"):
        load_config(invalid_yaml)


def test_config_hashing_deterministic():
    """Verify SHA-256 configuration hash is stable and deterministic."""
    cfg1 = ExperimentConfig(experiment_id="exp_1")
    cfg2 = ExperimentConfig(experiment_id="exp_1")
    cfg3 = ExperimentConfig(experiment_id="exp_2")

    hash1 = cfg1.compute_hash()
    hash2 = cfg2.compute_hash()
    hash3 = cfg3.compute_hash()

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_all_yaml_configs_load_validly():
    """Verify all YAML experiment configs in configs/ load correctly."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))
    assert len(yaml_files) >= 7, f"Expected at least 7 config files, found {len(yaml_files)}"

    for yf in yaml_files:
        cfg = load_config(yf)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.portfolio.gross_exposure <= 1.0
        assert cfg.data.universe == "sp100_20140101"


def test_config_save_and_reload(tmp_path):
    """Verify round-trip serialization of ExperimentConfig."""
    cfg = ExperimentConfig(experiment_id="roundtrip_test", random_seed=123)
    target = tmp_path / "roundtrip.yaml"
    save_config(cfg, target)

    reloaded = load_config(target)
    assert reloaded.experiment_id == "roundtrip_test"
    assert reloaded.random_seed == 123
    assert reloaded.compute_hash() == cfg.compute_hash()


def test_production_cost_binding_smoke_test(tmp_path):
    """
    Deterministic smoke test proving changing YAML cost values changes
    actual fill economics in the production backtest path.
    """
    import numpy as np
    import pandas as pd
    from src.backtest.engine import EventDrivenEngine
    from src.execution.costs import CostModel
    from src.portfolio import EqualWeightLongShort

    # 1. Create two YAML configs: Config A (zero costs) and Config B (30 bps friction)
    yaml_a = tmp_path / "config_a_zero.yaml"
    yaml_a.write_text(
        """
experiment_id: "smoke_zero"
costs:
  commission_bps: 0.0
  spread_bps: 0.0
  slippage_bps: 0.0
  min_commission: 0.0
portfolio:
  gross_exposure: 1.0
backtest:
  initial_cash: 100000.0
"""
    )

    yaml_b = tmp_path / "config_b_friction.yaml"
    yaml_b.write_text(
        """
experiment_id: "smoke_friction"
costs:
  commission_bps: 10.0
  spread_bps: 10.0
  slippage_bps: 10.0
  min_commission: 0.0
portfolio:
  gross_exposure: 1.0
backtest:
  initial_cash: 100000.0
"""
    )

    cfg_a = load_config(yaml_a)
    cfg_b = load_config(yaml_b)

    # 2. Production instantiation path
    cm_a = CostModel.from_config(cfg_a.costs)
    cm_b = CostModel.from_config(cfg_b.costs)

    assert cm_a.commission_bps == 0.0 and cm_a.spread_bps == 0.0 and cm_a.slippage_bps == 0.0
    assert cm_b.commission_bps == 10.0 and cm_b.spread_bps == 10.0 and cm_b.slippage_bps == 10.0

    engine_a = EventDrivenEngine(
        initial_capital=cfg_a.backtest.initial_cash,
        cost_model=cm_a,
        portfolio_constructor=EqualWeightLongShort(),
    )
    engine_b = EventDrivenEngine(
        initial_capital=cfg_b.backtest.initial_cash,
        cost_model=cm_b,
        portfolio_constructor=EqualWeightLongShort(),
    )

    # 3. Create a tiny 2-day deterministic price & signal panel (10 stocks)
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    tickers = [f"STK_{i}" for i in range(10)]
    tuples = [(d, tk) for d in dates for tk in tickers]
    idx = pd.MultiIndex.from_tuples(tuples, names=["date", "ticker"])
    prices_df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume_split_adjusted": 1_000_000.0,
        },
        index=idx,
    )
    # Signal: top 5 positive, bottom 5 negative
    sig_vals = {}
    for d in dates:
        for i, tk in enumerate(tickers):
            sig_vals[(d, tk)] = float(i)  # 0..4 bottom (short), 5..9 top (long)
    signals = pd.Series(sig_vals, index=idx)

    res_a = engine_a.run(prices_df, signals)
    res_b = engine_b.run(prices_df, signals)

    # 4. Verify trades and fill prices
    assert len(res_a.trades_df) == 10
    assert len(res_b.trades_df) == 10

    # For Config A (zero cost): total_cost == 0.0
    assert np.isclose(res_a.total_costs.sum(), 0.0, atol=1e-6)

    # For Config B (30 bps friction):
    # Total traded notional on $100,000 gross portfolio = $100,000
    # Total costs = $100,000 * 30 / 10000 = $300.0
    assert np.isclose(res_b.total_costs.sum(), 300.0, atol=1e-4)

    # Net terminal equity under Config B is strictly lower than Config A by exactly the $300.00 friction
    assert np.isclose(res_a.equity_curve.iloc[-1] - res_b.equity_curve.iloc[-1], 300.0, atol=1e-4)
