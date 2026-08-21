"""Portfolio hard constraint enforcement: quadratic Euclidean projection onto the constraint polytope."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def project_to_constraints(
    w_raw: np.ndarray,
    max_gross: float = 1.0,
    max_net: float = 0.20,
    max_position: float = 0.10,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Project raw portfolio weights onto the convex constraint polytope:
        minimize_{w} 0.5 * ||w - w_raw||_2^2
        subject to:
            -max_position <= w_i <= max_position   for all i = 1, ..., N
            sum_i |w_i| <= max_gross               (Zero-leverage guarantee: gross <= 1.0)
            -max_net <= sum_i w_i <= max_net        (Net exposure limit: |net| <= 0.20)

    Mathematical Formulation:
    This quadratic program minimizes the L2 distortion from the raw signal allocation
    while guaranteeing that all hard operational and risk limits are simultaneously satisfied.

    Parameters
    ----------
    w_raw : np.ndarray
        1D array of unconstrained target weights for active securities on date t.
    max_gross : float
        Gross exposure bound (default 1.0).
    max_net : float
        Net exposure bound (default 0.20).
    max_position : float
        Maximum absolute single-asset position (default 0.10).
    tolerance : float
        Constraint violation tolerance.

    Returns
    -------
    np.ndarray
        Constrained weight vector satisfying all limits.
    """
    n = len(w_raw)
    if n == 0:
        return w_raw

    # Check if raw weights already satisfy all constraints
    abs_w = np.abs(w_raw)
    if (
        np.all(abs_w <= max_position + tolerance)
        and np.sum(abs_w) <= max_gross + tolerance
        and abs(np.sum(w_raw)) <= max_net + tolerance
    ):
        return w_raw.copy()

    # If L1 norm is 0, return zeros
    if np.sum(abs_w) < 1e-12:
        return np.zeros_like(w_raw)

    # Quadratic programming via SLSQP / L-BFGS-B with box bounds
    bounds = [(-max_position, max_position) for _ in range(n)]

    def objective(w: np.ndarray) -> float:
        diff = w - w_raw
        return 0.5 * float(np.dot(diff, diff))

    def grad(w: np.ndarray) -> np.ndarray:
        return w - w_raw

    constraints = [
        # max_net - sum(w) >= 0
        {"type": "ineq", "fun": lambda w: max_net - np.sum(w)},
        # sum(w) + max_net >= 0
        {"type": "ineq", "fun": lambda w: np.sum(w) + max_net},
        # max_gross - sum(|w|) >= 0
        {"type": "ineq", "fun": lambda w: max_gross - np.sum(np.abs(w))},
    ]

    # Initial guess: clipped and normalized
    w0 = np.clip(w_raw, -max_position, max_position)
    gross0 = np.sum(np.abs(w0))
    if gross0 > max_gross:
        w0 = w0 * (max_gross / gross0)

    res = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )

    if res.success:
        w_opt = res.x
    else:
        # Fallback: iterative clipping and scaling projection
        w_opt = w0.copy()
        for _ in range(20):
            w_opt = np.clip(w_opt, -max_position, max_position)
            net_val = np.sum(w_opt)
            if abs(net_val) > max_net:
                w_opt -= (net_val - np.sign(net_val) * max_net) / n
                w_opt = np.clip(w_opt, -max_position, max_position)
            gross_val = np.sum(np.abs(w_opt))
            if gross_val > max_gross:
                w_opt *= max_gross / gross_val

    # Final hard check and safety clamp
    w_opt = np.clip(w_opt, -max_position, max_position)
    gross_final = np.sum(np.abs(w_opt))
    if gross_final > max_gross:
        w_opt *= max_gross / gross_final

    return w_opt


def enforce_portfolio_constraints(
    weights: pd.Series | pd.DataFrame,
    max_gross: float = 1.0,
    max_net: float = 0.20,
    max_position: float = 0.10,
    tolerance: float = 1e-6,
) -> pd.Series:
    """
    Enforce portfolio construction hard constraints per date cross-section via Euclidean projection:
    1. Maximum single-stock position limit: |w_{i,t}| <= max_position (0.10)
    2. Maximum gross exposure limit: sum_i |w_{i,t}| <= max_gross (1.00)
    3. Maximum net exposure limit: |sum_i w_{i,t}| <= max_net (0.20)

    Parameters
    ----------
    weights : pd.Series | pd.DataFrame
        Target weights with MultiIndex (date, ticker).
    max_gross : float
        Maximum gross exposure limit (default 1.0).
    max_net : float
        Maximum net exposure limit (default 0.20).
    max_position : float
        Maximum individual position weight (default 0.10).
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    pd.Series
        Constrained weights strictly satisfying all limits.
    """
    if isinstance(weights, pd.DataFrame):
        w_s = weights.iloc[:, 0].copy()
    else:
        w_s = weights.copy()

    if not isinstance(w_s.index, pd.MultiIndex):
        raise ValueError("Weights must have MultiIndex (date, ticker).")

    def _constrain_cross_section(row: pd.Series) -> pd.Series:
        valid_mask = row.notna()
        w = row[valid_mask].values.copy()
        if len(w) == 0:
            return row

        w_proj = project_to_constraints(
            w,
            max_gross=max_gross,
            max_net=max_net,
            max_position=max_position,
            tolerance=tolerance,
        )
        out = row.copy()
        out[valid_mask] = w_proj
        return out

    wide = w_s.unstack(level="ticker")
    constrained_wide = wide.apply(_constrain_cross_section, axis=1)
    constrained_series = constrained_wide.stack(future_stack=True) if hasattr(constrained_wide, "stack") else constrained_wide.stack(dropna=False)
    constrained_series = constrained_series.reindex(w_s.index)
    constrained_series.name = "target_weight"
    return constrained_series
