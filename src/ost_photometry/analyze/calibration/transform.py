"""Linear transformation fitting helpers."""

from __future__ import annotations

import numpy as np


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return T, ZP, T_err, ZP_err from y = T*x + ZP."""
    w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, 1.0)
    sw = np.sqrt(w)
    a = np.column_stack([x * sw, sw])
    b = y * sw
    coeffs, residuals, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    t_val, zp_val = float(coeffs[0]), float(coeffs[1])
    n = len(x)
    if rank < 2 or n < 3:
        return t_val, zp_val, 0.0, 0.0
    if len(residuals) > 0:
        s_sq = residuals[0] / max(n - 2, 1)
    else:
        fit = t_val * x + zp_val
        s_sq = np.sum((y - fit) ** 2) / max(n - 2, 1)
    cov = s_sq * np.linalg.inv(a.T @ a)
    return t_val, zp_val, float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def unweighted_linear_fit(
    color: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, float]:
    """Simple legacy-style fit: delta = T * color + ZP."""
    a = np.column_stack([color, np.ones_like(color)])
    coeffs, _, _, _ = np.linalg.lstsq(a, delta, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


__all__ = ["unweighted_linear_fit", "weighted_linear_fit"]
