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


#: A masked pixel whose median fill lies this many sky σ above the sky sits
#: in a source and is filled from a profile fit instead.
SOURCE_FILL_SIGMA = 5.0

#: Masked regions larger than this (pixels) are not fitted (saturated cores,
#: columns): they keep the median fill.
MAX_FIT_REGION = 25


def fill_masked_pixels(
    data: np.ndarray,
    mask: np.ndarray | None,
    size: int = 5,
    max_fraction: float = 0.5,
    fit_sources: bool = True,
) -> np.ndarray:
    """Replace masked pixels by the local median before resampling.

    Bilinear / bicubic warps smear a hot pixel into its neighbours and a
    resampled mask cannot exclude them cleanly. Filling defects first keeps
    the registered frame photometrically usable; the mask itself is still
    carried along (see :func:`resample_mask`). Returns ``data`` unchanged
    if there is no mask or more than ``max_fraction`` of the frame is masked.

    With ``fit_sources`` a masked region inside a star (median fill more
    than :data:`SOURCE_FILL_SIGMA` sky σ above the sky, at most
    :data:`MAX_FIT_REGION` pixels) is filled from a Moffat fit (free β and
    background) to the unmasked pixels around it: the median of the
    neighbours misses a star core by several per cent of the star's flux
    (FWHM 3 px: −8 %), the fit by < 0.3 %.
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
    if fit_sources:
        _fit_masked_sources(values, good, mask, rows, cols)
    return values


def _fit_masked_sources(
    values: np.ndarray, good: np.ndarray, mask: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> None:
    """Refill masked regions inside sources from a local Moffat fit (in place)."""
    from scipy import ndimage
    from scipy.optimize import least_squares

    sample = good[::4, ::4]
    sample = sample[np.isfinite(sample)]
    if sample.size < 100:
        return
    sky = float(np.median(sample))
    std = 1.4826 * float(np.median(np.abs(sample - sky)))
    if not np.isfinite(std) or std <= 0:
        return
    in_source = values[rows, cols] - sky > SOURCE_FILL_SIGMA * std
    if not np.any(in_source):
        return
    labels, _n = ndimage.label(mask)
    slices = ndimage.find_objects(labels)
    ny, nx = values.shape
    for rid in np.unique(labels[rows[in_source], cols[in_source]]):
        sl = slices[rid - 1]
        region = labels[sl] == rid
        if region.sum() > MAX_FIT_REGION:
            continue
        cy = 0.5 * (sl[0].start + sl[0].stop - 1)
        cx = 0.5 * (sl[1].start + sl[1].stop - 1)
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        r_fit = 6.0 + 0.5 * extent
        half = int(np.ceil(r_fit))
        y0, y1 = max(0, int(cy) - half), min(ny, int(cy) + half + 2)
        x0, x1 = max(0, int(cx) - half), min(nx, int(cx) + half + 2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cut = good[y0:y1, x0:x1]
        use = np.isfinite(cut) & (np.hypot(xx - cx, yy - cy) <= r_fit)
        if np.count_nonzero(use) < 12:
            continue
        xs, ys, zs = xx[use].astype(float), yy[use].astype(float), cut[use]
        b0 = float(np.percentile(zs, 10))
        k = int(np.argmax(zs))
        amp = max(float(zs[k]) - b0, 1e-6)
        #   star centre near the brightest unmasked pixel (the masked core,
        #   if the core is masked, is within ~1.5 px of it)
        xc0, yc0 = float(xs[k]), float(ys[k])

        def model(p, x, y):
            a, xc, yc, alpha, beta, b = p
            return a * (1.0 + ((x - xc) ** 2 + (y - yc) ** 2) / alpha**2) ** (-beta) + b

        try:
            fit = least_squares(
                lambda p, x=xs, y=ys, z=zs: model(p, x, y) - z,
                [amp, xc0, yc0, 2.0, 3.0, b0],
                bounds=([0.0, xc0 - 2, yc0 - 2, 0.3, 1.2, -np.inf],
                        [np.inf, xc0 + 2, yc0 + 2, 20.0, 30.0, np.inf]),
            )
        except ValueError:
            continue
        if not fit.success or not np.all(np.isfinite(fit.x)):
            continue
        ry, rx = np.nonzero(region)
        ry, rx = ry + sl[0].start, rx + sl[1].start
        filled = model(fit.x, rx.astype(float), ry.astype(float))
        if np.all(np.isfinite(filled)):
            values[ry, rx] = filled


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


def resample_uncertainty(
    sigma: np.ndarray,
    mask: np.ndarray | None,
    resample_linear: Callable[[np.ndarray], np.ndarray],
    area_ratio: float = 1.0,
) -> np.ndarray:
    """Per-pixel σ on a new pixel grid.

    The *variance* is resampled with a linear (non-negative) kernel, after
    masked pixels got the median variance of their neighbours (a hot pixel's
    σ must not leak into its neighbours), and scaled by ``area_ratio`` = new
    / old pixel area: ``σ_out = √(k · interp(σ²))``. Summed over an aperture
    this keeps the variance of the flux it covers, also when the new pixels
    are smaller (k < 1, finer grid). Interpolating σ itself with a cubic
    kernel overshoots near sharp σ spikes (negative σ is possible).
    """
    var = np.asarray(sigma, dtype=float) ** 2
    var = fill_masked_pixels(var, mask, fit_sources=False)
    out = np.asarray(resample_linear(var), dtype=float)
    return np.sqrt(float(area_ratio) * np.maximum(out, 0.0))


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
    fwhm: float | None = None,
    beta: float = 2.5,
) -> np.ndarray:
    """Masked fraction of each aperture (0 … 1).

    Without ``fwhm``: of the on-frame area. With ``fwhm``: of the star's
    light in the aperture (pixels weighted by a Moffat profile of that FWHM
    centred on the star): a masked pixel in the core counts far more than
    one at the edge, and the core is what the fill (see
    :func:`fill_masked_pixels`) can get wrong. Off-frame pixels count as
    masked.
    """
    full = np.asarray(apertures.area_overlap(data), dtype=float)
    if mask is None:
        return np.zeros_like(full)
    if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        kept = np.asarray(apertures.area_overlap(data, mask=mask), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = 1.0 - kept / full
        return np.where(np.isfinite(frac), np.clip(frac, 0.0, 1.0), 1.0)
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    radius = float(np.atleast_1d(apertures.r)[0])
    alpha = float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    half = int(np.ceil(radius)) + 1
    dy, dx = np.mgrid[-half : half + 1, -half : half + 1]
    positions = np.atleast_2d(apertures.positions)
    out = np.ones(len(positions))
    for k, (x, y) in enumerate(positions):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ix, iy = int(round(x)), int(round(y))
        r2 = (dx + ix - x) ** 2 + (dy + iy - y) ** 2
        inside = r2 <= radius**2
        weight = np.where(inside, (1.0 + r2 / alpha**2) ** (-beta), 0.0)
        yy, xx = dy + iy, dx + ix
        on = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        bad = np.ones(weight.shape, dtype=bool)
        bad[on] = mask[yy[on], xx[on]]
        total = float(weight.sum())
        if total > 0:
            out[k] = float(np.sum(weight[bad]) / total)
    return out
