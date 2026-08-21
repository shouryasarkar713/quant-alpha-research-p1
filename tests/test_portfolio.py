"""Unit tests for portfolio construction methodologies, volatility floor, universe edge cases, and hard constraints."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    EqualWeightLongShort,
    InverseVolatilitySignalWeighted,
    SignalWeightedLongShort,
    enforce_portfolio_constraints,
    get_portfolio_constructor,
)


def test_equal_weight_long_short_constraints():
    """
    Verify EqualWeightLongShort creates valid long-short portfolio:
    Gross exposure <= 1.0, Net exposure == 0.0, max position <= 0.10.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = [f"TICKER_{i}" for i in range(20)]
    records = []
    for d in dates:
        for i, t in enumerate(tickers):
            sig = -1.0 + 2.0 * (i / 19.0)
            records.append({"date": d, "ticker": t, "signal": sig})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constructor = EqualWeightLongShort()
    weights_df = constructor.construct_weights(df["signal"])

    for d in dates:
        w_d = weights_df.xs(d, level="date")["target_weight"].dropna()
        gross = np.sum(np.abs(w_d))
        net = np.sum(w_d)
        max_pos = np.max(np.abs(w_d))

        assert gross <= 1.000001, f"Gross exposure {gross} exceeds 1.0 limit!"
        assert abs(net) <= 1e-6, f"Net exposure {net} is not zero for dollar-neutral strategy!"
        assert max_pos <= 0.100001, f"Max position {max_pos} exceeds 0.10 limit!"


def test_signal_weighted_constraints():
    """
    Verify SignalWeightedLongShort satisfies all hard exposure constraints.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = [f"TICKER_{i}" for i in range(30)]
    rng = np.random.default_rng(42)
    records = []
    for d in dates:
        for t in tickers:
            records.append({"date": d, "ticker": t, "signal": rng.uniform(-1.0, 1.0)})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constructor = SignalWeightedLongShort()
    weights_df = constructor.construct_weights(df["signal"])

    for d in dates:
        w_d = weights_df.xs(d, level="date")["target_weight"].dropna()
        gross = np.sum(np.abs(w_d))
        net = np.sum(w_d)
        max_pos = np.max(np.abs(w_d))

        assert gross <= 1.000001, f"Gross exposure {gross} exceeds 1.0!"
        assert abs(net) <= 0.200001, f"Net exposure {net} exceeds 0.20!"
        assert max_pos <= 0.100001, f"Max position {max_pos} exceeds 0.10!"


def test_inverse_volatility_signal_weighted():
    """
    Mandatory Rule 4:
    Verify inverse-volatility weighting: w_i proportional to signal_i / volatility_i,
    followed by normalization to satisfy gross exposure <= 1.0 with zero leverage.
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    tickers = [f"TICKER_{i}" for i in range(20)]
    records = []
    for d in dates:
        for i, t in enumerate(tickers):
            if i >= 10:
                sig = 1.0
                vol = 0.10 + 0.03 * (i - 10)
            else:
                sig = -1.0
                vol = 0.10 + 0.03 * i
            records.append({"date": d, "ticker": t, "signal": sig, "realized_vol_60": vol})

    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constructor = InverseVolatilitySignalWeighted()
    weights_df = constructor.construct_weights(df["signal"], features=df)

    w0 = weights_df.xs(dates[0], level="date")["target_weight"]

    assert w0["TICKER_10"] > w0["TICKER_19"]
    assert w0["TICKER_0"] < w0["TICKER_9"]

    gross = np.sum(np.abs(w0.dropna()))
    net = np.sum(w0.dropna())
    max_pos = np.max(np.abs(w0.dropna()))

    assert gross <= 1.000001
    assert abs(net) <= 0.200001
    assert max_pos <= 0.100001


def test_inverse_volatility_floor_and_nan_handling():
    """
    Verification Check 2:
    Tests zero, near-zero, NaN, and normal volatility:
    - sigma = 0.0 or near-zero is floored at sigma_min (0.05)
    - NaN volatility excludes that security from allocation (weight = NaN / 0)
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    records = [
        # Normal positive signal and normal vol
        {"date": dates[0], "ticker": "NORMAL_A", "signal": 1.0, "realized_vol_60": 0.20},
        {"date": dates[0], "ticker": "NORMAL_B", "signal": -1.0, "realized_vol_60": 0.20},
        # Near-zero vol (should be floored at 0.05, not explode)
        {"date": dates[0], "ticker": "NEAR_ZERO_VOL", "signal": 0.8, "realized_vol_60": 0.00001},
        # Zero vol (should be floored at 0.05)
        {"date": dates[0], "ticker": "ZERO_VOL", "signal": -0.8, "realized_vol_60": 0.0},
        # NaN vol (should be excluded)
        {"date": dates[0], "ticker": "NAN_VOL", "signal": 1.0, "realized_vol_60": np.nan},
        # Other normal stocks to form valid universe >= 5
        {"date": dates[0], "ticker": "NORMAL_C", "signal": 0.5, "realized_vol_60": 0.15},
        {"date": dates[0], "ticker": "NORMAL_D", "signal": -0.5, "realized_vol_60": 0.15},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constructor = InverseVolatilitySignalWeighted()
    weights_df = constructor.construct_weights(df["signal"], features=df, params={"sigma_min": 0.05})

    w0 = weights_df.xs(dates[0], level="date")["target_weight"]

    # NAN_VOL must be excluded (NaN or 0.0)
    assert pd.isna(w0["NAN_VOL"]) or w0["NAN_VOL"] == 0.0
    # ZERO_VOL and NEAR_ZERO_VOL must not explode, bounded by max_position 0.10
    assert abs(w0["NEAR_ZERO_VOL"]) <= 0.100001
    assert abs(w0["ZERO_VOL"]) <= 0.100001


def test_small_eligible_universe():
    """
    Verification Check 3:
    When fewer than min_eligible securities have valid signals, allocate 0 weight (cash).
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    records = [
        {"date": dates[0], "ticker": "A", "signal": 1.0},
        {"date": dates[0], "ticker": "B", "signal": -1.0},
        {"date": dates[0], "ticker": "C", "signal": 0.5},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constructor = EqualWeightLongShort()
    weights_df = constructor.construct_weights(df["signal"], params={"min_eligible": 5})

    w0 = weights_df.xs(dates[0], level="date")["target_weight"].dropna()
    np.testing.assert_allclose(w0.values, 0.0, atol=1e-8)


def test_signal_weighted_zero_denominator_identical_signals():
    """
    Verification Check 4:
    If all signals are identical (sum(|s - mean(s)|) == 0), return all-zero weights (cash).
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    records = [{"date": dates[0], "ticker": f"T_{i}", "signal": 0.50} for i in range(10)]
    df = pd.DataFrame(records).set_index(["date", "ticker"])

    constructor = SignalWeightedLongShort()
    weights_df = constructor.construct_weights(df["signal"])

    w0 = weights_df.xs(dates[0], level="date")["target_weight"].dropna()
    np.testing.assert_allclose(w0.values, 0.0, atol=1e-8)


def test_enforce_portfolio_constraints_extreme_clipping():
    """
    Verify enforce_portfolio_constraints clamps gross > 1.0 and positions > 0.10
    while satisfying all constraints simultaneously via Euclidean projection.
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    records = [
        {"date": dates[0], "ticker": "A", "target_weight": 0.80},
        {"date": dates[0], "ticker": "B", "target_weight": 0.50},
        {"date": dates[0], "ticker": "C", "target_weight": -0.60},
        {"date": dates[0], "ticker": "D", "target_weight": -0.40},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    constrained = enforce_portfolio_constraints(df["target_weight"])

    w = constrained.dropna().values
    assert np.all(np.abs(w) <= 0.100001), f"Positions exceed 0.10: {w}"
    assert np.sum(np.abs(w)) <= 1.000001, f"Gross exceeds 1.0: {np.sum(np.abs(w))}"
    assert abs(np.sum(w)) <= 0.200001, f"Net exceeds 0.20: {abs(np.sum(w))}"


def test_portfolio_constructor_factory():
    """Verify factory initializes registered portfolio constructors."""
    for name in ["equal_weight_long_short", "signal_weighted", "inverse_volatility_signal_weighted"]:
        c = get_portfolio_constructor(name)
        assert c is not None

    with pytest.raises(ValueError, match="Unknown portfolio constructor"):
        get_portfolio_constructor("invalid_portfolio_name")
