"""Portfolio construction and weight allocation module."""

from src.portfolio.base import BasePortfolioConstructor
from src.portfolio.constraints import enforce_portfolio_constraints
from src.portfolio.equal_weight import EqualWeightLongShort
from src.portfolio.signal_weighted import SignalWeightedLongShort
from src.portfolio.volatility_scaled import InverseVolatilitySignalWeighted

_CONSTRUCTOR_REGISTRY = {
    "equal_weight": EqualWeightLongShort,
    "equal_weight_long_short": EqualWeightLongShort,
    "signal_weighted": SignalWeightedLongShort,
    "volatility_scaled": InverseVolatilitySignalWeighted,
    "inverse_volatility": InverseVolatilitySignalWeighted,
    "inverse_volatility_signal_weighted": InverseVolatilitySignalWeighted,
}


def get_portfolio_constructor(name: str) -> BasePortfolioConstructor:
    """
    Factory function to instantiate a portfolio constructor by name.

    Parameters
    ----------
    name : str
        Constructor identifier (e.g. 'equal_weight_long_short', 'signal_weighted', 'inverse_volatility_signal_weighted').

    Returns
    -------
    BasePortfolioConstructor
    """
    key = name.strip().lower()
    if key not in _CONSTRUCTOR_REGISTRY:
        raise ValueError(
            f"Unknown portfolio constructor '{name}'. Available constructors: {list(_CONSTRUCTOR_REGISTRY.keys())}"
        )
    return _CONSTRUCTOR_REGISTRY[key]()


__all__ = [
    "BasePortfolioConstructor",
    "enforce_portfolio_constraints",
    "EqualWeightLongShort",
    "SignalWeightedLongShort",
    "InverseVolatilitySignalWeighted",
    "get_portfolio_constructor",
]
