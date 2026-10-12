"""Non-linear stretches of linear image data to the range 0 ... 1."""

from __future__ import annotations

import numpy as np

from .background import sky_level

#: Values of ``stretch``.
STRETCHES = ("auto", "asinh", "linear")


def midtone_transfer(x: np.ndarray, midtone: float) -> np.ndarray:
    """Midtone transfer function: maps ``midtone`` to 0.5, keeps 0 and 1."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    m = float(midtone)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (m - 1.0) * x / ((2.0 * m - 1.0) * x - m)
    return np.nan_to_num(out, nan=0.0)


def stretch_parameters(
    data: np.ndarray,
    *,
    method: str = "auto",
    target_background: float = 0.25,
    shadows_sigma: float = 2.8,
    highlight_percentile: float = 99.995,
) -> dict[str, float]:
    """Black point, white point and shape of a stretch measured on ``data``.

    ``auto`` (screen-transfer like): black point at sky - ``shadows_sigma``
    sigma, the sky lands at ``target_background``. ``asinh``: softening of
    a few sigma of the sky noise. ``linear``: black at the sky, white at
    ``highlight_percentile``.
    """
    if method not in STRETCHES:
        raise ValueError(f"stretch must be one of {STRETCHES}, got {method!r}")
    sky, std = sky_level(data)
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    white = float(np.percentile(finite[:: max(1, finite.size // 2_000_000)],
                                highlight_percentile)) if finite.size else sky + 1.0
    if method == "linear":
        return {"black": sky, "white": max(white, sky + std)}
    black = sky - shadows_sigma * std
    white = max(white, sky + 10.0 * std)
    if method == "asinh":
        return {"black": black, "white": white, "softening": 3.0 * std / (white - black)}
    level = (sky - black) / (white - black)
    # Midtone such that the sky level is mapped to target_background
    t = float(target_background)
    midtone = level * (t - 1.0) / (2.0 * t * level - t - level) if level > 0 else 0.5
    return {"black": black, "white": white, "midtone": float(np.clip(midtone, 1e-4, 0.9999))}


def apply_stretch(data: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Apply :func:`stretch_parameters` output; result in 0 ... 1 (float32)."""
    black, white = params["black"], params["white"]
    x = (np.asarray(data, dtype=np.float32) - np.float32(black)) / np.float32(white - black)
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
    if "midtone" in params:
        return midtone_transfer(x, params["midtone"]).astype(np.float32)
    if "softening" in params:
        a = max(float(params["softening"]), 1e-6)
        return (np.arcsinh(x / a) / np.arcsinh(1.0 / a)).astype(np.float32)
    return x.astype(np.float32)


__all__ = ["STRETCHES", "apply_stretch", "midtone_transfer", "stretch_parameters"]
