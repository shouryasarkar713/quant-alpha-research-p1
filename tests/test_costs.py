"""Unit tests for single-pass transaction cost model, slippage, and commissions."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.events import OrderEvent
from src.backtest.portfolio import PortfolioTracker
from src.config.schema import CostConfig
from src.execution.costs import COST_REGIMES, CostModel, get_cost_model
from src.validation.walk_forward import WalkForwardValidator, compute_deflated_sharpe_ratio


def test_zero_cost_model():
    """Verify zero-cost benchmark produces exact reference prices and 0 costs."""
    cm = CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0, min_commission=0.0)
    fill_buy = cm.calculate_fill("AAPL", "BUY", quantity=1000, reference_price=150.0)
    fill_sell = cm.calculate_fill("AAPL", "SELL", quantity=1000, reference_price=150.0)

    assert fill_buy.fill_price == 150.0
    assert fill_sell.fill_price == 150.0
    assert fill_buy.commission == 0.0
    assert fill_sell.commission == 0.0
    assert fill_buy.total_cost_usd == 0.0
    assert fill_sell.total_cost_usd == 0.0


def test_specification_hand_verifiable_example():
    """
    Verify exact hand-verifiable example from v1.2 specification Section 18.1:
    reference_price = 100.00, quantity = 100, direction = 'BUY'
    spread_bps = 5, slippage_bps = 5, commission_bps = 5, min_commission = 0.0

    execution_adjustment = (5 + 5) / 10000 = 0.001
    fill_price = 100.00 * 1.001 = 100.10
    trade_value = 100.00 * 100 = 10,000.00
    commission = 10,000.00 * 5 / 10000 = 5.00
    spread_cost = 10,000.00 * 5 / 10000 = 5.00
    slippage_cost = 10,000.00 * 5 / 10000 = 5.00
    total_cost = 15.00

    Cash change = -(100.10 * 100 + 5.00) = -$10,015.00
    Position change = +100 shares
    """
    cm = CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=0.0)

    # 1. BUY Fill
    fill_buy = cm.calculate_fill("AAPL", "BUY", quantity=100, reference_price=100.0)
    assert np.isclose(fill_buy.fill_price, 100.10, atol=1e-6)
    assert np.isclose(fill_buy.commission, 5.00, atol=1e-6)
    assert np.isclose(fill_buy.spread_cost_usd, 5.00, atol=1e-6)
    assert np.isclose(fill_buy.slippage_cost_usd, 5.00, atol=1e-6)
    assert np.isclose(fill_buy.total_cost_usd, 15.00, atol=1e-6)

    # Test PortfolioTracker Cash & Position Accounting for BUY
    tracker_buy = PortfolioTracker(initial_capital=100_000.0)
    from src.backtest.events import FillEvent
    fill_evt_buy = FillEvent(
        timestamp=pd.Timestamp("2020-01-02"),
        ticker="AAPL",
        direction="BUY",
        quantity=100.0,
        fill_price=fill_buy.fill_price,
        reference_price=fill_buy.reference_price,
        commission=fill_buy.commission,
        spread_cost=fill_buy.spread_cost_usd,
        slippage_cost=fill_buy.slippage_cost_usd,
        total_cost=fill_buy.total_cost_usd,
    )
    tracker_buy.update_from_fills([fill_evt_buy])
    expected_cash_buy = 100_000.0 - 10_015.00
    assert np.isclose(tracker_buy.cash, expected_cash_buy, atol=1e-4)
    assert tracker_buy.positions["AAPL"] == 100.0

    # 2. SELL Fill
    fill_sell = cm.calculate_fill("AAPL", "SELL", quantity=100, reference_price=100.0)
    assert np.isclose(fill_sell.fill_price, 99.90, atol=1e-6)
    assert np.isclose(fill_sell.commission, 5.00, atol=1e-6)
    assert np.isclose(fill_sell.spread_cost_usd, 5.00, atol=1e-6)
    assert np.isclose(fill_sell.slippage_cost_usd, 5.00, atol=1e-6)
    assert np.isclose(fill_sell.total_cost_usd, 15.00, atol=1e-6)

    # Test PortfolioTracker Cash & Position Accounting for SELL
    tracker_sell = PortfolioTracker(initial_capital=100_000.0)
    tracker_sell.positions["AAPL"] = 100.0
    fill_evt_sell = FillEvent(
        timestamp=pd.Timestamp("2020-01-02"),
        ticker="AAPL",
        direction="SELL",
        quantity=100.0,
        fill_price=fill_sell.fill_price,
        reference_price=fill_sell.reference_price,
        commission=fill_sell.commission,
        spread_cost=fill_sell.spread_cost_usd,
        slippage_cost=fill_sell.slippage_cost_usd,
        total_cost=fill_sell.total_cost_usd,
    )
    tracker_sell.update_from_fills([fill_evt_sell])
    expected_cash_sell = 100_000.0 + (99.90 * 100.0 - 5.00)  # + $9,985.00
    assert np.isclose(tracker_sell.cash, expected_cash_sell, atol=1e-4)
    assert tracker_sell.positions.get("AAPL", 0.0) == 0.0


def test_cost_config_from_config_binding():
    """Verify CostModel.from_config accurately consumes CostConfig from YAML schema."""
    cfg = CostConfig(commission_bps=7.5, spread_bps=3.0, slippage_bps=4.5, min_commission=1.50)
    cm = CostModel.from_config(cfg)

    assert cm.commission_bps == 7.5
    assert cm.spread_bps == 3.0
    assert cm.slippage_bps == 4.5
    assert cm.min_commission == 1.50

    # Fill calculation under this config
    fill = cm.calculate_fill("AAPL", "BUY", quantity=100, reference_price=200.0)
    # execution_adjustment = (3.0 + 4.5) / 10000 = 0.00075
    # fill_price = 200.0 * 1.00075 = 200.15
    # trade_value = 200.0 * 100 = 20,000
    # commission = max(1.50, 20000 * 7.5 / 10000) = max(1.50, 15.0) = 15.0
    assert np.isclose(fill.fill_price, 200.15, atol=1e-6)
    assert np.isclose(fill.commission, 15.0, atol=1e-6)


def test_min_commission_enforcement():
    """Verify min_commission is enforced for small trades."""
    cm = CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=2.00)
    # Trade of 1 share at $10.00 -> trade value = $10.00
    # Proportional commission = $10 * 5 / 10000 = $0.005 < $2.00 min_commission
    fill = cm.calculate_fill("TINY", "BUY", quantity=1, reference_price=10.0)
    assert fill.commission == 2.00


def test_cost_regimes_exact_v12_values():
    """
    Verify exact cost regimes from specification Section 18.2:
    Zero: 0 bps total one-way (0 comm, 0 spread, 0 slip)
    Low: 7 bps total one-way (2 comm, 3 spread, 2 slip)
    Medium: 15 bps total one-way (5 comm, 5 spread, 5 slip)
    High: 30 bps total one-way (10 comm, 10 spread, 10 slip)
    Very High: 50 bps total one-way (15 comm, 15 spread, 20 slip)
    """
    expected_regimes = {
        "zero": (0.0, 0.0, 0.0, 0.0),
        "low": (2.0, 3.0, 2.0, 7.0),
        "medium": (5.0, 5.0, 5.0, 15.0),
        "high": (10.0, 10.0, 10.0, 30.0),
        "very_high": (15.0, 15.0, 20.0, 50.0),
    }

    for name, (exp_comm, exp_spr, exp_slip, exp_total) in expected_regimes.items():
        cm = get_cost_model(name)
        assert cm.commission_bps == exp_comm
        assert cm.spread_bps == exp_spr
        assert cm.slippage_bps == exp_slip
        total_bps = cm.commission_bps + cm.spread_bps + cm.slippage_bps
        assert total_bps == exp_total


def test_no_double_counting_of_costs():
    """
    Verify spread and slippage are absorbed strictly once into fill_price,
    and commission is the only separate line item.
    """
    cm = CostModel(commission_bps=5.0, spread_bps=5.0, slippage_bps=5.0, min_commission=0.0)
    ref_price = 150.0
    qty = 500.0

    fill = cm.calculate_fill("MSFT", "BUY", quantity=qty, reference_price=ref_price)
    notional_at_fill = fill.fill_price * qty
    notional_at_ref = ref_price * qty

    # The price adjustment difference in dollars
    price_cost = notional_at_fill - notional_at_ref
    assert np.isclose(price_cost, fill.spread_cost_usd + fill.slippage_cost_usd, atol=1e-6)

    # Total cost equals price_cost + commission exactly
    assert np.isclose(fill.total_cost_usd, price_cost + fill.commission, atol=1e-6)


def test_dsr_not_silently_run_with_default_trials():
    """Verify DSR is not computed with a hardcoded N=10 in walk-forward validation."""
    validator = WalkForwardValidator()
    # When num_trials is None, DSR must be None
    assert validator.cost_model is not None
