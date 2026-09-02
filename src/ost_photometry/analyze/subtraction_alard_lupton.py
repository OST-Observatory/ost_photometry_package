"""Alard–Lupton PSF-matching kernel (no HOTPANTS binary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from scipy.signal import fftconvolve

from .. import terminal_output

# Three Gaussians × polynomials (Alard & Lupton 1998 / HOTPANTS-like).
_DEFAULT_SIGMA_SCALE = (0.8, 2.0, 4.5)
_DEFAULT_DEGREES = (4, 2, 1)


def _as_2d(data) -> np.ndarray:
    if data is None:
        raise ValueError("image data is None")
    raw = data
    if hasattr(raw, "value") and not isinstance(raw, np.ndarray):
        raw = raw.value
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr.mean(axis=-1)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _match_shapes(sci: np.ndarray, tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sci.shape == tmpl.shape:
        return sci, tmpl
    h = min(sci.shape[0], tmpl.shape[0])
    w = min(sci.shape[1], tmpl.shape[1])
    return sci[:h, :w], tmpl[:h, :w]


def _sky_level(data: np.ndarray) -> tuple[float, float]:
    _mean, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    if not np.isfinite(med):
        med = float(np.nanmedian(data))
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(data))
    return float(med), float(max(std, 1e-8))


def _border_sky(stamp: np.ndarray, border: int = 3) -> float:
    h, w = stamp.shape
    b = min(max(int(border), 1), h // 3, w // 3)
    frame = np.concatenate(
        [
            stamp[:b, :].ravel(),
            stamp[-b:, :].ravel(),
            stamp[b:-b, :b].ravel(),
            stamp[b:-b, -b:].ravel(),
        ]
    )
    frame = frame[np.isfinite(frame)]
    if frame.size == 0:
        return 0.0
    return float(np.median(frame))


def kernel_basis(
    ksize: int,
    sigmas: tuple[float, ...] = _DEFAULT_SIGMA_SCALE,
    degrees: tuple[int, ...] = _DEFAULT_DEGREES,
) -> np.ndarray:
    """Stack of kernel basis images, shape ``(n_basis, ksize, ksize)``."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    if len(sigmas) != len(degrees):
        raise ValueError("sigmas and degrees must have the same length")
    hw = ksize // 2
    yy, xx = np.mgrid[-hw : hw + 1, -hw : hw + 1]
    xn = xx / max(hw, 1)
    yn = yy / max(hw, 1)
    bases: list[np.ndarray] = []
    for sigma, deg in zip(sigmas, degrees, strict=True):
        sig = max(float(sigma), 0.3)
        g = np.exp(-(xx.astype(float) ** 2 + yy.astype(float) ** 2) / (2.0 * sig**2))
        g /= g.sum()
        for p in range(int(deg) + 1):
            for q in range(int(deg) + 1 - p):
                bases.append(g * (xn**p) * (yn**q))
    return np.stack(bases, axis=0)


def _cutout(data: np.ndarray, x: float, y: float, half: int) -> np.ndarray | None:
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = yi - half, yi + half + 1
    x0, x1 = xi - half, xi + half + 1
    if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
        return None
    return data[y0:y1, x0:x1]


def _stamp_snr(stamp: np.ndarray) -> float:
    sky = _border_sky(stamp)
    noise = float(np.nanstd(stamp - sky))
    peak = float(np.nanmax(stamp) - sky)
    if not np.isfinite(noise) or noise <= 0:
        return 0.0
    return peak / noise


def _aperture_flux(stamp: np.ndarray, radius: float = 5.0) -> float:
    sky = _border_sky(stamp)
    cy, cx = stamp.shape[0] / 2.0 - 0.5, stamp.shape[1] / 2.0 - 0.5
    yy, xx = np.ogrid[: stamp.shape[0], : stamp.shape[1]]
    ap = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(radius) ** 2
    vals = stamp[ap] - sky
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.sum(vals))


def find_kernel_stars(
    science: np.ndarray,
    *,
    n_stars: int = 40,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    min_separation: float | None = None,
) -> np.ndarray:
    """Bright unsaturated star positions ``(n, 2)`` as ``(x, y)``."""
    from photutils.detection import DAOStarFinder

    sci = _as_2d(science)
    _mean, med, std = sigma_clipped_stats(sci, sigma=3.0, maxiters=5)
    if not np.isfinite(std) or std <= 0:
        return np.empty((0, 2))
    finder = DAOStarFinder(fwhm=float(max(fwhm, 1.0)), threshold=threshold_sigma * std)
    tbl = finder(sci - med)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2))
    from .utils.photometry import xy_column_names

    xy_cols = xy_column_names(tbl)
    if xy_cols is None:
        return np.empty((0, 2))
    x_col, y_col = xy_cols
    if "flux" in tbl.colnames:
        tbl.sort("flux")
        tbl.reverse()
    skip_bright = max(1, int(0.1 * len(tbl))) if len(tbl) > 8 else 0
    sep = float(min_separation) if min_separation is not None else max(8.0, 3.0 * fwhm)
    xy: list[tuple[float, float]] = []
    for i, row in enumerate(tbl):
        if i < skip_bright:
            continue
        x, y = float(row[x_col]), float(row[y_col])
        if any((x - x0) ** 2 + (y - y0) ** 2 < sep**2 for x0, y0 in xy):
            continue
        xy.append((x, y))
        if len(xy) >= int(n_stars):
            break
    if not xy:
        return np.empty((0, 2))
    return np.asarray(xy, dtype=float)


def flux_scale_from_stamps(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray,
    *,
    half: int = 11,
) -> float:
    """Robust science/template flux ratio from star stamps (after local sky)."""
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    ratios: list[float] = []
    for x, y in np.asarray(star_xy, dtype=float).reshape(-1, 2):
        s = _cutout(sci, x, y, half)
        t = _cutout(tmpl, x, y, half)
        if s is None or t is None:
            continue
        if _stamp_snr(s) < 5.0 or _stamp_snr(t) < 4.0:
            continue
        fs = _aperture_flux(s)
        ft = _aperture_flux(t)
        if ft > 0 and fs > 0 and np.isfinite(fs) and np.isfinite(ft):
            ratios.append(fs / ft)
    if len(ratios) < 3:
        raise RuntimeError("Need at least 3 star stamps to estimate the flux scale")
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0 or scale > 1e6:
        raise RuntimeError(f"Unusable flux scale {scale}")
    return scale


def flux_scale_subtract(
    science: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Science minus a robust scalar times the template, plus a sky offset."""
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    ok = np.isfinite(sci) & np.isfinite(tmpl)
    if np.count_nonzero(ok) < 16:
        return sci - tmpl, 1.0
    sci_med, _ = _sky_level(sci)
    tmpl_med, _ = _sky_level(tmpl)
    sci0 = sci - sci_med
    tmpl0 = tmpl - tmpl_med
    peak = float(np.nanpercentile(np.abs(tmpl0[ok]), 99.0))
    bright = ok & (np.abs(tmpl0) > max(0.1 * peak, 1e-8))
    if np.count_nonzero(bright) < 16:
        scale = 1.0
    else:
        scale = float(np.nanmedian(sci0[bright] / tmpl0[bright]))
    if not np.isfinite(scale) or abs(scale) > 1e6:
        scale = 1.0
    return sci0 - scale * tmpl0, scale


def fit_alard_lupton_kernel(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray,
    *,
    ksize: int = 21,
    sigmas: tuple[float, ...] = _DEFAULT_SIGMA_SCALE,
    degrees: tuple[int, ...] = _DEFAULT_DEGREES,
    n_stars: int = 40,
) -> tuple[np.ndarray, int]:
    """
    Fit a spatially constant AL kernel on sky-subtracted, flux-matched images.

    Returns ``(kernel, n_stamps_used)``.
    """
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    bases = kernel_basis(ksize, sigmas, degrees)
    hw = ksize // 2
    half_stamp = hw + 6
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    used = 0
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    for x, y in xy:
        s = _cutout(sci, x, y, half_stamp)
        t = _cutout(tmpl, x, y, half_stamp)
        if s is None or t is None:
            continue
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(t)):
            continue
        if _stamp_snr(s) < 5.0 or _stamp_snr(t) < 4.0:
            continue
        s = s - _border_sky(s)
        t = t - _border_sky(t)
        if float(np.std(s)) < 1e-8 or float(np.std(t)) < 1e-8:
            continue
        convs = [fftconvolve(t, b, mode="same")[hw:-hw, hw:-hw].ravel() for b in bases]
        rows.append(np.column_stack(convs))
        rhs.append(s[hw:-hw, hw:-hw].ravel())
        used += 1
        if used >= int(n_stars):
            break
    if used < 3:
        raise RuntimeError(f"Need at least 3 valid stamps to fit the kernel (got {used})")
    design = np.vstack(rows)
    target = np.concatenate(rhs)
    coeff, *_ = np.linalg.lstsq(design, target, rcond=1e-6)
    kernel = np.tensordot(coeff, bases, axes=(0, 0))
    return kernel, used


def alard_lupton_difference(
    science: np.ndarray,
    template: np.ndarray,
    *,
    star_xy: np.ndarray | None = None,
    n_stars: int = 40,
    fwhm: float = 3.0,
    ksize: int | None = None,
) -> tuple[np.ndarray, str]:
    """
    ``science - (kernel ⊗ scaled_template)`` after matching sky and flux.

    Falls back to a scalar flux scale if the kernel fit has too few stamps.
    Returns ``(difference, method)`` with ``method`` ``alard_lupton`` or ``flux_scale``.
    """
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    sci_sky, _ = _sky_level(sci)
    tmpl_sky, _ = _sky_level(tmpl)
    sci0 = sci - sci_sky
    tmpl0 = tmpl - tmpl_sky
    if ksize is None:
        ksize = int(2 * np.ceil(3.0 * max(_DEFAULT_SIGMA_SCALE) * max(fwhm / 3.0, 0.5)) + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = int(np.clip(ksize, 15, 41))
    xy = star_xy
    try:
        if xy is None or len(np.asarray(xy).reshape(-1, 2)) == 0:
            xy = find_kernel_stars(
                sci0,
                n_stars=max(n_stars, 20),
                fwhm=fwhm,
                min_separation=max(8.0, 0.9 * ksize),
            )
        scale = flux_scale_from_stamps(sci0, tmpl0, xy, half=max(7, ksize // 2))
        tmpl_s = scale * tmpl0
        kernel, n_used = fit_alard_lupton_kernel(
            sci0, tmpl_s, xy, ksize=ksize, n_stars=n_stars
        )
        matched = fftconvolve(tmpl_s, kernel, mode="same")
        residual = sci0 - matched
        _, resid_sky, _ = sigma_clipped_stats(residual, sigma=3.0, maxiters=5)
        if not np.isfinite(resid_sky):
            resid_sky = 0.0
        ksum = float(np.sum(kernel))
        terminal_output.print_to_terminal(
            f"Alard–Lupton kernel ({n_used} stamps, ksize={ksize}, "
            f"flux_scale={scale:.4g}, kernel_sum={ksum:.3f})",
            indent=2,
            style_name="NORMAL",
        )
        return residual - float(resid_sky), "alard_lupton"
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Alard–Lupton kernel fit failed ({exc}); using flux-scale subtraction",
            indent=2,
            style_name="WARNING",
        )
        diff, scale = flux_scale_subtract(sci, tmpl)
        terminal_output.print_to_terminal(
            f"Flux-scale subtraction (scale={scale:.4g})",
            indent=2,
            style_name="NORMAL",
        )
        return diff, "flux_scale"


def run_alard_lupton(
    science_ccd: CCDData,
    template_hdu: fits.ImageHDU | fits.PrimaryHDU,
    *,
    workdir: str | Path,
    output_filename: str = "diff.fits",
    star_xy: np.ndarray | None = None,
    n_stars: int = 40,
    fwhm: float = 3.0,
    ksize: int | None = None,
) -> Path:
    """Write ``science − matched template`` under ``workdir`` (same contract as HOTPANTS)."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    sci = _as_2d(science_ccd.data)
    tmpl = _as_2d(template_hdu.data)
    diff, _method = alard_lupton_difference(
        sci, tmpl, star_xy=star_xy, n_stars=n_stars, fwhm=fwhm, ksize=ksize
    )
    out_path = work / output_filename
    out = CCDData(diff, unit=getattr(science_ccd, "unit", None) or "adu")
    if getattr(science_ccd, "wcs", None) is not None:
        out.wcs = science_ccd.wcs
    if getattr(science_ccd, "mask", None) is not None:
        mask = np.asarray(science_ccd.mask)
        if mask.shape == diff.shape:
            out.mask = mask
    out.write(out_path, overwrite=True)
    return out_path


__all__ = [
    "alard_lupton_difference",
    "find_kernel_stars",
    "fit_alard_lupton_kernel",
    "flux_scale_from_stamps",
    "flux_scale_subtract",
    "kernel_basis",
    "run_alard_lupton",
]
