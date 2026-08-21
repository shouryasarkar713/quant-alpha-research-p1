"""Unit tests for the four core trading signals and combined statistical baseline."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_synthetic_ohlcv
from src.features.engine import compute_features
from src.signals import (
    AbnormalVolumeSignal,
    BaseSignal,
    CombinedSignal,
    MeanReversionSignal,
    MomentumSignal,
    VolatilitySignal,
    get_signal,
)


def test_momentum_ranking():
    """
    3 stocks with 252-day returns:
    A: +0.20, B: -0.05, C: +0.10
    Expected rank ordering: A (+1.0) > C (0.0) > B (-1.0)
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {"date": dates[0], "ticker": "A", "ret_252d_skip21d": 0.20},
        {"date": dates[0], "ticker": "B", "ret_252d_skip21d": -0.05},
        {"date": dates[0], "ticker": "C", "ret_252d_skip21d": 0.10},
        {"date": dates[1], "ticker": "A", "ret_252d_skip21d": 0.20},
        {"date": dates[1], "ticker": "B", "ret_252d_skip21d": -0.05},
        {"date": dates[1], "ticker": "C", "ret_252d_skip21d": 0.10},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    sig = MomentumSignal().compute(df)

    s_A = sig.loc[(dates[0], "A")]
    s_B = sig.loc[(dates[0], "B")]
    s_C = sig.loc[(dates[0], "C")]

    assert s_A > s_C > s_B
    assert np.isclose(s_A, 1.0, atol=1e-6)
    assert np.isclose(s_B, -1.0, atol=1e-6)
    assert np.isclose(s_C, 0.0, atol=1e-6)


def test_mean_reversion_direction():
    """
    Mean reversion directionality:
    Stock with z-score = +2.0 (overbought) should have negative signal (expect drop).
    Stock with z-score = -2.0 (oversold) should have positive signal (expect bounce).
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {"date": dates[0], "ticker": "OVERSOLD", "zscore_price_20": -2.0},
        {"date": dates[0], "ticker": "NEUTRAL", "zscore_price_20": 0.0},
        {"date": dates[0], "ticker": "OVERBOUGHT", "zscore_price_20": 2.0},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    sig = MeanReversionSignal().compute(df)

    s_oversold = sig.loc[(dates[0], "OVERSOLD")]
    s_overbought = sig.loc[(dates[0], "OVERBOUGHT")]

    assert s_oversold > 0.0, "Oversold stock must produce positive signal."
    assert s_overbought < 0.0, "Overbought stock must produce negative signal."
    assert s_oversold > s_overbought


def test_volatility_signal_direction():
    """
    Low volatility anomaly directionality:
    Stock with lower volatility must receive higher signal value.
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {"date": dates[0], "ticker": "LOW_VOL", "realized_vol_60": 0.12},
        {"date": dates[0], "ticker": "MID_VOL", "realized_vol_60": 0.25},
        {"date": dates[0], "ticker": "HIGH_VOL", "realized_vol_60": 0.50},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    sig = VolatilitySignal().compute(df)

    s_low = sig.loc[(dates[0], "LOW_VOL")]
    s_high = sig.loc[(dates[0], "HIGH_VOL")]

    assert s_low > 0.0, "Low volatility stock must produce positive signal."
    assert s_high < 0.0, "High volatility stock must produce negative signal."
    assert s_low > s_high


def test_volume_signal_direction():
    """
    Abnormal volume + price direction:
    Positive return + high volume -> positive continuation signal.
    Negative return + high volume -> negative continuation signal.
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {"date": dates[0], "ticker": "UP_HIGHVOL", "ret_1d": 0.05, "relative_volume": 3.0},
        {"date": dates[0], "ticker": "FLAT", "ret_1d": 0.00, "relative_volume": 1.0},
        {"date": dates[0], "ticker": "DOWN_HIGHVOL", "ret_1d": -0.05, "relative_volume": 3.0},
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    sig = AbnormalVolumeSignal().compute(df)

    s_up = sig.loc[(dates[0], "UP_HIGHVOL")]
    s_down = sig.loc[(dates[0], "DOWN_HIGHVOL")]

    assert s_up > 0.0
    assert s_down < 0.0
    assert s_up > s_down


def test_combined_signal_equal_weight():
    """
    Verify pre-specified equal-weight combined signal produces valid range in [-1, 1].
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        {
            "date": dates[0], "ticker": "WINNER",
            "ret_252d_skip21d": 0.30, "zscore_price_20": -1.5,
            "realized_vol_60": 0.10, "ret_1d": 0.02, "relative_volume": 2.5,
        },
        {
            "date": dates[0], "ticker": "LOSER",
            "ret_252d_skip21d": -0.30, "zscore_price_20": 2.0,
            "realized_vol_60": 0.45, "ret_1d": -0.02, "relative_volume": 2.5,
        },
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    comb = CombinedSignal().compute(df)

    assert comb.loc[(dates[0], "WINNER")] > comb.loc[(dates[0], "LOSER")]
    assert np.isclose(comb.loc[(dates[0], "WINNER")], 1.0, atol=1e-6)
    assert np.isclose(comb.loc[(dates[0], "LOSER")], -1.0, atol=1e-6)


def test_combined_signal_nan_propagation():
    """
    Strict NaN policy:
    If ANY component signal is NaN for a security on date t,
    combined_signal[ticker, t] MUST be NaN (do NOT silently renormalize remaining signals).
    """
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    records = [
        # Stock A has all 4 features valid
        {
            "date": dates[0], "ticker": "A",
            "ret_252d_skip21d": 0.20, "zscore_price_20": -1.0,
            "realized_vol_60": 0.15, "ret_1d": 0.01, "relative_volume": 1.5,
        },
        # Stock B has missing momentum (ret_252d_skip21d is NaN)
        {
            "date": dates[0], "ticker": "B",
            "ret_252d_skip21d": np.nan, "zscore_price_20": -1.0,
            "realized_vol_60": 0.15, "ret_1d": 0.01, "relative_volume": 1.5,
        },
        # Stock C has all 4 features valid
        {
            "date": dates[0], "ticker": "C",
            "ret_252d_skip21d": -0.10, "zscore_price_20": 1.5,
            "realized_vol_60": 0.35, "ret_1d": -0.02, "relative_volume": 1.1,
        },
    ]
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    comb = CombinedSignal().compute(df)

    # Stock B must be strictly NaN
    assert pd.isna(comb.loc[(dates[0], "B")]), "Combined signal for stock with missing component must be NaN."
    # Stock A and Stock C must have valid non-NaN signals
    assert not pd.isna(comb.loc[(dates[0], "A")])
    assert not pd.isna(comb.loc[(dates[0], "C")])


def test_signal_values_in_range():
    """
    Verify all 4 core signals and combined signal produce values strictly in [-1, 1]
    when evaluated on data with sufficient lookback.
    """
    df = generate_synthetic_ohlcv(
        tickers=["AAPL", "MSFT", "JPM", "XOM"],
        start_date="2020-01-01",
        end_date="2021-04-30",
        seed=42,
    )
    features = compute_features(df, lag=0, include_forward_targets=False)

    signals = [
        MomentumSignal(),
        MeanReversionSignal(),
        VolatilitySignal(),
        AbnormalVolumeSignal(),
        CombinedSignal(),
    ]

    for sig in signals:
        out = sig.compute(features)
        valid_vals = out.dropna()
        assert len(valid_vals) > 0, f"Signal {sig.name} produced empty output."
        assert (valid_vals >= -1.0 - 1e-6).all(), f"Signal {sig.name} produced values < -1.0"
        assert (valid_vals <= 1.0 + 1e-6).all(), f"Signal {sig.name} produced values > +1.0"


def test_signal_factory():
    """Verify get_signal correctly instantiates all registered signals."""
    for name in ["momentum_12_1", "mean_reversion_zscore", "low_vol", "abnormal_volume", "combined_signal"]:
        sig = get_signal(name)
        assert isinstance(sig, BaseSignal)
        assert sig.name in [name, "momentum_12_1", "mean_reversion_zscore", "low_vol", "abnormal_volume", "combined_signal"]

    with pytest.raises(ValueError, match="Unknown signal"):
        get_signal("unregistered_random_signal")
