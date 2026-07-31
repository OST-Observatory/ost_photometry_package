"""Linear transformation fitting helpers."""

from __future__ import annotations

import numpy as np


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Weighted least-squares fit ``y = T*x + ZP``.

    Returns
    -------
    T, ZP, T_err, ZP_err, cov_tz
        Slope, intercept, diagonal formal errors, and off-diagonal covariance
        ``Cov(T, ZP)`` from the residual variance times ``(AᵀA)⁻¹``.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, 1.0)
    sw = np.sqrt(w)
    a = np.column_stack([x * sw, sw])
    b = y * sw
    coeffs, residuals, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    t_val, zp_val = float(coeffs[0]), float(coeffs[1])
    n = len(x)
    if rank < 2 or n < 3:
        return t_val, zp_val, 0.0, 0.0, 0.0
    if len(residuals) > 0:
        s_sq = residuals[0] / max(n - 2, 1)
    else:
        fit = t_val * x + zp_val
        s_sq = np.sum((y - fit) ** 2) / max(n - 2, 1)
    try:
        cov = s_sq * np.linalg.inv(a.T @ a)
    except np.linalg.LinAlgError:
        return t_val, zp_val, 0.0, 0.0, 0.0
    t_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    zp_err = float(np.sqrt(max(cov[1, 1], 0.0)))
    cov_tz = float(cov[0, 1])
    return t_val, zp_val, t_err, zp_err, cov_tz


def unweighted_linear_fit(
    color: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, float]:
    """Unweighted least-squares fit: delta = T * color + ZP."""
    a = np.column_stack([color, np.ones_like(color)])
    coeffs, _, _, _ = np.linalg.lstsq(a, delta, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def calibrated_magnitude_variance(
    inst_err: np.ndarray | float,
    color: np.ndarray | float,
    *,
    color_term: float,
    color_term_err: float,
    zero_point_err: float,
    cov_tz: float = 0.0,
    sigma_color_sq: np.ndarray | float = 0.0,
) -> np.ndarray:
    """
    Variance of ``m_cal = m_inst + T·color + ZP`` (first-order).

    Includes the fit covariance term ``2·color·Cov(T, ZP)``.
    """
    inst_err = np.asarray(inst_err, dtype=float)
    color = np.asarray(color, dtype=float)
    sigma_color_sq = np.asarray(sigma_color_sq, dtype=float)
    var = (
        inst_err**2
        + float(zero_point_err) ** 2
        + (color * float(color_term_err)) ** 2
        + (float(color_term) ** 2) * sigma_color_sq
        + 2.0 * color * float(cov_tz)
    )
    return np.asarray(var, dtype=float)


__all__ = [
    "calibrated_magnitude_variance",
    "unweighted_linear_fit",
    "weighted_linear_fit",
]
