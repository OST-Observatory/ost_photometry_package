"""Pixel-mask helpers shared by reduction and analysis.

A bad-pixel mask must survive resampling (shift / warp / reprojection)
without growing, and defects should be filled before resampling so they do
not contaminate their neighbours. These helpers implement that once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .. import terminal_output

__all__ = [
    "aperture_masked_fraction",
    "fill_masked_pixels",
    "interior_mask_fraction",
    "negative_outlier_mask",
    "resample_mask",
    "warn_if_mask_too_large",
]


def negative_outlier_mask(
    data: np.ndarray,
    n_sigma: float = 5.0,
    max_fraction_warn: float = 0.01,
) -> np.ndarray:
    """Mask pixels that are *significantly* negative after dark subtraction.

    Ordinary noise makes a large share of faint-sky pixels slightly negative;
    masking every ``data < 0`` pixel (the old behaviour) flagged 10–50 % of
    the background, biased sky estimates high, and — once such a mask is
    resampled during registration — dropped pixels from aperture sums. Only
    pixels below ``-n_sigma × σ_robust`` are defects worth masking.
    """
    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=bool)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros(values.shape, dtype=bool)
    mask = values < -float(n_sigma) * sigma
    frac = float(mask.mean())
    if frac > max_fraction_warn:
        terminal_output.print_to_terminal(
            f"WARNING: {frac:.1%} of the pixels are below -{n_sigma:g} sigma "
            "after dark subtraction; check the master dark / exposure matching.",
            style_name="WARNING",
            indent=2,
        )
    return mask


def fill_masked_pixels(
    data: np.ndarray,
    mask: np.ndarray | None,
    size: int = 5,
    max_fraction: float = 0.5,
) -> np.ndarray:
    """Replace masked pixels by the local median before resampling.

    Bilinear / bicubic warps smear a hot pixel into its neighbours and a
    resampled mask cannot exclude them cleanly. Filling defects first keeps
    the registered frame photometrically usable; the mask itself is still
    carried along (see :func:`resample_mask`). Returns ``data`` unchanged
    if there is no mask or more than ``max_fraction`` of the frame is masked.
    """
    if mask is None:
        return data
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.shape(data) or not mask.any() or mask.mean() > max_fraction:
        return data

    values = np.array(data, dtype=float, copy=True)
    good = np.where(mask, np.nan, values)
    half = int(size) // 2
    # Median of the *unmasked* neighbours, evaluated only at the masked
    # positions (a full median filter over the frame would cost seconds per
    # image for a few thousand defects).
    padded = np.pad(good, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (int(size), int(size)))
    rows, cols = np.nonzero(mask)
    local = np.empty(rows.size, dtype=float)
    chunk = 200_000
    with np.errstate(all="ignore"):
        for start in range(0, rows.size, chunk):
            sl = slice(start, start + chunk)
            block = windows[rows[sl], cols[sl]].reshape(rows[sl].size, -1)
            local[sl] = np.nanmedian(block, axis=1)
    # isolated windows without a single good neighbour fall back to the frame median
    fallback = float(np.nanmedian(good))
    local = np.where(np.isfinite(local), local, fallback)
    values[rows, cols] = local
    return values


def fill_masked_uncertainty(
    data: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray | None,
    size: int = 5,
    max_fraction: float = 0.5,
) -> np.ndarray:
    """σ for pixels that :func:`fill_masked_pixels` replaced.

    A filled value is the median of its unmasked ``size × size``
    neighbours. Its error is the neighbours' typical σ plus how far that
    median is from a second interpolator, the mean of the unmasked direct
    (4-connected) neighbours: both agree on flat sky and on a smooth
    gradient (a star's wing), and disagree near a peak, where the fill is a
    poor guess. The spread of the window itself would mistake the gradient
    for noise. Without this, a hot pixel's huge σ, or a dead pixel's tiny
    one, would enter the aperture error.
    """
    if mask is None:
        return sigma
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.shape(data) or not mask.any() or mask.mean() > max_fraction:
        return sigma
    values = np.where(mask, np.nan, np.asarray(data, dtype=float))
    var = np.where(mask, np.nan, np.asarray(sigma, dtype=float) ** 2)
    out = np.array(sigma, dtype=float, copy=True)
    half = int(size) // 2
    win_v = np.lib.stride_tricks.sliding_window_view(
        np.pad(values, half, mode="edge"), (int(size), int(size))
    )
    win_s = np.lib.stride_tricks.sliding_window_view(
        np.pad(var, half, mode="edge"), (int(size), int(size))
    )
    rows, cols = np.nonzero(mask)
    chunk = 200_000
    with np.errstate(all="ignore"):
        for start in range(0, rows.size, chunk):
            sl = slice(start, start + chunk)
            v = win_v[rows[sl], cols[sl]].reshape(rows[sl].size, int(size), int(size))
            w = win_s[rows[sl], cols[sl]].reshape(rows[sl].size, -1)
            med = np.nanmedian(v.reshape(rows[sl].size, -1), axis=1)
            cross = np.stack(
                [v[:, half - 1, half], v[:, half + 1, half], v[:, half, half - 1], v[:, half, half + 1]],
                axis=1,
            )
            mean4 = np.nanmean(cross, axis=1)
            interp = np.where(np.isfinite(mean4), med - mean4, 0.0)
            out[rows[sl], cols[sl]] = np.sqrt(np.nanmedian(w, axis=1) + interp**2)
    fallback = float(np.nanmedian(np.where(mask, np.nan, out)))
    bad = mask & ~np.isfinite(out)
    out[bad] = fallback
    return out


def resample_mask(
    mask: np.ndarray | None,
    resample: Callable[[np.ndarray], np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray | None:
    """Resample a boolean mask without growing it.

    ``resample`` maps a float array onto the new grid (e.g. a closure around
    ``reproject_interp`` or ``skimage.transform.warp``). Interpolated mask
    values are thresholded at ``threshold`` (nearest-neighbour semantics)
    instead of ``> 0``, which turned every defect into a 2×2–3×3 patch and
    multiplied the masked area several times.
    """
    if mask is None:
        return None
    values = resample(np.asarray(mask, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values > float(threshold), False)


def interior_mask_fraction(
    mask: np.ndarray | None,
    border_px: int = 20,
) -> float | None:
    """Masked fraction of the frame interior (``None`` if there is no mask).

    The outer ``border_px`` are ignored so a legitimate alignment footprint
    (NaNs along the edge after a shift or reproject) does not dominate.
    """
    if mask is None:
        return None
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return None
    border = max(int(border_px), 0)
    if values.ndim != 2 or min(values.shape) <= 2 * border:
        return float(values.mean())
    interior = values[border:-border, border:-border]
    return float(interior.mean())


def warn_if_mask_too_large(
    mask: np.ndarray | None,
    *,
    label: str,
    footprint: np.ndarray | None = None,
    limit: float = 0.10,
    border_px: int = 20,
    indent: int = 2,
) -> float | None:
    """Warn when too much of the frame's own field is masked (default > 10 %).

    A grown mask (every ``data < 0`` pixel plus bilinear resampling) punched
    holes into apertures and inflated the instrumental scatter. ``footprint``
    marks pixels of the output grid this frame does not cover (alignment
    onto a reference with a pointing offset, dither or meridian flip); they
    are counted separately and reported as a note, not as defects. The outer
    ``border_px`` are ignored. Returns the masked fraction inside the field.
    """
    if mask is None:
        return None
    values = np.asarray(mask, dtype=bool)
    outside = (np.zeros_like(values) if footprint is None
               else np.asarray(footprint, dtype=bool).reshape(values.shape))
    border = max(int(border_px), 0)
    if values.ndim == 2 and min(values.shape) > 2 * border and border:
        values = values[border:-border, border:-border]
        outside = outside[border:-border, border:-border]
    if values.size == 0:
        return None
    inside = ~outside
    n_inside = int(inside.sum())
    frac = float((values & inside).sum() / n_inside) if n_inside else 1.0
    outside_frac = float(outside.mean())
    if outside_frac > float(limit):
        terminal_output.print_to_terminal(
            f"{label}: {outside_frac:.1%} of the reference field is not covered by "
            "this frame (pointing offset, dither or meridian flip); only the "
            "overlap contributes to the stack.",
            indent=indent,
        )
    if frac <= float(limit):
        return frac
    terminal_output.print_to_terminal(
        f"WARNING: {label}: {frac:.1%} of the pixels inside the frame's field are "
        f"masked (limit {float(limit):.0%}). Aperture photometry will drop a "
        "changing set of pixels from each star. Typical causes: masking "
        "every negative pixel after dark subtraction, bilinear resampling of "
        "the mask during alignment, or a flat / dark mask flagging large areas.",
        style_name="WARNING",
        indent=indent,
    )
    return frac


def aperture_masked_fraction(
    apertures,
    data: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Fraction of each aperture's on-frame area that is masked (0 … 1)."""
    full = np.asarray(apertures.area_overlap(data), dtype=float)
    if mask is None:
        return np.zeros_like(full)
    kept = np.asarray(apertures.area_overlap(data, mask=mask), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 1.0 - kept / full
    return np.where(np.isfinite(frac), np.clip(frac, 0.0, 1.0), 1.0)
