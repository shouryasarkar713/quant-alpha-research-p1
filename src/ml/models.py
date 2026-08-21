"""Machine learning forecasting models: OLS, Ridge, Lasso, and XGBoost (Specification Section 24)."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
import xgboost as xgb


class BaseMLModel:
    """Base interface for alpha prediction ML models."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseMLModel:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OLSModel(BaseMLModel):
    """Ordinary Least Squares Linear Regression baseline."""

    def __init__(self) -> None:
        self.model = LinearRegression(fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> OLSModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RidgeModel(BaseMLModel):
    """L2 Regularized Ridge Regression."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LassoModel(BaseMLModel):
    """L1 Regularized Lasso Regression."""

    def __init__(self, alpha: float = 0.001) -> None:
        self.alpha = alpha
        self.model = Lasso(alpha=alpha, fit_intercept=True, max_iter=2000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LassoModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBoostModel(BaseMLModel):
    """Gradient Boosted Decision Tree (XGBoost Regressor)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def get_ml_model(name: str, **kwargs: Any) -> BaseMLModel:
    """Factory to instantiate ML forecasting models."""
    key = name.strip().lower()
    if key == "ols":
        return OLSModel()
    elif key == "ridge":
        return RidgeModel(**kwargs)
    elif key == "lasso":
        return LassoModel(**kwargs)
    elif key in ["xgboost", "xgb"]:
        return XGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown ML model '{name}'. Available: ['ols', 'ridge', 'lasso', 'xgboost']")
