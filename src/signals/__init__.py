"""Trading signals: momentum, mean reversion, low volatility, abnormal volume, and combined baseline."""

from src.signals.base import BaseSignal
from src.signals.combined import CombinedSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.momentum import MomentumSignal
from src.signals.volatility import VolatilitySignal
from src.signals.volume_signal import AbnormalVolumeSignal

_SIGNAL_REGISTRY = {
    "momentum": MomentumSignal,
    "momentum_12_1": MomentumSignal,
    "mean_reversion": MeanReversionSignal,
    "mean_reversion_zscore": MeanReversionSignal,
    "volatility": VolatilitySignal,
    "low_vol": VolatilitySignal,
    "volume": AbnormalVolumeSignal,
    "volume_signal": AbnormalVolumeSignal,
    "abnormal_volume": AbnormalVolumeSignal,
    "combined": CombinedSignal,
    "combined_signal": CombinedSignal,
}


def get_signal(signal_name: str) -> BaseSignal:
    """
    Factory function to instantiate a trading signal by name.

    Parameters
    ----------
    signal_name : str
        Name of the signal (e.g. 'momentum_12_1', 'mean_reversion_zscore', 'low_vol', 'abnormal_volume', 'combined_signal').

    Returns
    -------
    BaseSignal
        Instantiated signal object conforming to BaseSignal interface.
    """
    key = signal_name.strip().lower()
    if key not in _SIGNAL_REGISTRY:
        raise ValueError(
            f"Unknown signal '{signal_name}'. Available signals: {list(_SIGNAL_REGISTRY.keys())}"
        )
    return _SIGNAL_REGISTRY[key]()


__all__ = [
    "BaseSignal",
    "MomentumSignal",
    "MeanReversionSignal",
    "VolatilitySignal",
    "AbnormalVolumeSignal",
    "CombinedSignal",
    "get_signal",
]
