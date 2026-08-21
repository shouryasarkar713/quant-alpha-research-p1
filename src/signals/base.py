"""Abstract base class interface for trading signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseSignal(ABC):
    """
    Abstract base class for all trading signals.

    Signal convention:
    - Positive values -> Expected positive future return (long bias)
    - Negative values -> Expected negative future return (short bias)
    - Scale: Cross-sectionally normalized to [-1.0, +1.0]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique signal identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the signal and rationale."""
        ...

    @abstractmethod
    def compute(self, features: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        """
        Compute signal values from pre-computed features.

        Parameters
        ----------
        features : pd.DataFrame
            DataFrame with MultiIndex (date, ticker) containing pre-computed features.
        params : dict[str, Any] | None
            Signal-specific parameters (lookback, thresholds, ranking methods).

        Returns
        -------
        pd.Series
            Series with MultiIndex (date, ticker), values in [-1, 1] after
            cross-sectional ranking / normalization.
        """
        ...
