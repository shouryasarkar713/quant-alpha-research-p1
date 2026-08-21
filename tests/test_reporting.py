"""Tests for reporting, figures, and tearsheet generation."""

from pathlib import Path
import json
from src.evaluation.metrics import PerformanceMetrics
from src.evaluation.tearsheet import generate_tearsheet


def test_figures_exist_and_non_empty():
    """Verify that all 5 publication research figures exist and are non-empty."""
    fig_dir = Path("reports/figures")
    expected_figures = [
        "fig1_pipeline_architecture.png",
        "fig2_confirmatory_ic_forest_plot.png",
        "fig3_cost_sensitivity_curves.png",
        "fig4_walk_forward_oos_performance.png",
        "fig5_exploratory_ic_horizon_profile.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file too small: {fig_path}"


def test_tearsheet_generation():
    """Verify performance tearsheet generation formatting."""
    m = PerformanceMetrics(
        cagr=-0.2745,
        annualized_return=-0.3208,
        annualized_volatility=0.0912,
        sharpe_ratio=-3.472,
        sortino_ratio=-3.122,
        max_drawdown=-0.9706,
        max_drawdown_duration_days=2506,
        calmar_ratio=-0.2828,
        daily_hit_rate=0.456,
        profit_factor=0.627,
        average_daily_turnover=0.8123,
        annualized_turnover=204.7,
        total_cost_usd=9903113.0,
        cost_drag_bps=2606.0,
        total_trading_days=2768,
    )
    ts = generate_tearsheet(m, strategy_name="Test Strategy")
    assert "## Performance & Risk Tearsheet: Test Strategy" in ts
    assert "-27.45%" in ts
    assert "-3.472" in ts
    assert "$9,903,113.00" in ts


def test_checkpoint_integrity():
    """Verify authoritative confirmatory checkpoint integrity."""
    chk_path = Path("results/checkpoints/confirmatory_results_c3d67525d09fc052_3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840.json")
    assert chk_path.exists()
    with open(chk_path) as f:
        data = json.load(f)
    assert data["data_hash"] == "c3d67525d09fc052"
    assert data["hypothesis_registry_hash"] == "3caabc3550691880f4d66cc24a7a4ad14b76f623c169fd065cd2470fb3025840"
    assert "H1_MOMENTUM" in data["results"]
