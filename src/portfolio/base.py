"""Abstract base class interface for portfolio weight construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BasePortfolioConstructor(ABC):
    """
    Abstract base class for all portfolio construction methodologies.

    Mandatory Constraints:
    - Gross exposure <= 1.0 (Zero leverage guarantee)
    - Net exposure <= 0.20
    - Position weight <= 0.10
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique portfolio construction scheme identifier."""
        ...

    @abstractmethod
    def construct_weights(
        self,
        signals: pd.DataFrame | pd.Series,
        features: pd.DataFrame | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Construct target portfolio weights from normalized trading signals.

        Parameters
        ----------
        signals : pd.DataFrame | pd.Series
            MultiIndex (date, ticker) with signal values in [-1, 1].
        features : pd.DataFrame | None
            Pre-computed feature panel (needed for inverse-volatility weighting).
        params : dict[str, Any] | None
            Constructor-specific parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (date, ticker) and column 'target_weight'.
        """
        ...
