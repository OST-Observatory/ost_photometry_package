"""Alard–Lupton PSF-matching kernel (no HOTPANTS binary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from scipy.signal import fftconvolve

from .. import terminal_output

# Three Gaussians × polynomials (Alard & Lupton 1998). Degree 2 on the
# narrowest Gaussian rings (donuts with a core spike); keep this modest.
_DEFAULT_SIGMA_SCALE = (0.7, 1.5, 3.0)
_DEFAULT_DEGREES = (1, 0, 0)


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


def _fill_image(arr: np.ndarray) -> np.ndarray:
    good = np.isfinite(arr)
    if np.all(good):
        return arr
    fill = float(np.nanmedian(arr))
    if not np.isfinite(fill):
        fill = 0.0
    out = np.array(arr, dtype=np.float64, copy=True)
    out[~good] = fill
    return out


def _match_shapes(sci: np.ndarray, tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sci.shape == tmpl.shape:
        return sci, tmpl
    h = min(sci.shape[0], tmpl.shape[0])
    w = min(sci.shape[1], tmpl.shape[1])

    def _center(arr: np.ndarray) -> np.ndarray:
        y0 = (arr.shape[0] - h) // 2
        x0 = (arr.shape[1] - w) // 2
        return arr[y0 : y0 + h, x0 : x0 + w]

    return _center(sci), _center(tmpl)


def _sky_level(data: np.ndarray) -> tuple[float, float]:
    _mean, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    if not np.isfinite(med):
        med = float(np.nanmedian(data))
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(data))
    return float(med), float(max(std, 1e-8))


def _border_pixels(stamp: np.ndarray, border: int = 3) -> np.ndarray:
    h, w = stamp.shape
    b = min(max(int(border), 1), max(h // 3, 1), max(w // 3, 1))
    frame = np.concatenate(
        [
            stamp[:b, :].ravel(),
            stamp[-b:, :].ravel(),
            stamp[b:-b, :b].ravel() if h > 2 * b else np.array([]),
            stamp[b:-b, -b:].ravel() if h > 2 * b else np.array([]),
        ]
    )
    return frame[np.isfinite(frame)]


def _border_sky(stamp: np.ndarray, border: int = 3) -> float:
    frame = _border_pixels(stamp, border=border)
    if frame.size == 0:
        finite = stamp[np.isfinite(stamp)]
        return float(np.median(finite)) if finite.size else 0.0
    return float(np.median(frame))


def _border_std(stamp: np.ndarray, border: int = 3) -> float:
    frame = _border_pixels(stamp, border=border)
    if frame.size < 4:
        finite = stamp[np.isfinite(stamp)]
        if finite.size < 4:
            return 1e-8
        return float(max(np.std(finite), 1e-8))
    return float(max(np.std(frame), 1e-8))


def _replace_nonfinite(stamp: np.ndarray) -> np.ndarray | None:
    """Fill NaN/Inf with local sky so a HiPS mask does not drop the stamp."""
    out = np.array(stamp, dtype=np.float64, copy=True)
    good = np.isfinite(out)
    if not np.any(good):
        return None
    if np.all(good):
        return out
    out[~good] = _border_sky(stamp)
    return out


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
    noise = _border_std(stamp)
    finite = stamp[np.isfinite(stamp)]
    if finite.size == 0:
        return 0.0
    hi = float(np.max(finite) - sky)
    lo = float(sky - np.min(finite))
    peak = max(hi, lo)
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


def _shift_to_peak(
    data: np.ndarray,
    x: float,
    y: float,
    half: int,
    *,
    max_shift: float = 10.0,
) -> tuple[float, float]:
    """Move ``(x, y)`` to the local positive peak (handles a few pixels of WCS error)."""
    stamp = _cutout(data, x, y, half)
    if stamp is None:
        return x, y
    sky = _border_sky(stamp)
    work = stamp - sky
    if float(np.nanmax(work)) < float(-np.nanmin(work)):
        work = -work
    cy = stamp.shape[0] / 2.0 - 0.5
    cx = stamp.shape[1] / 2.0 - 0.5
    yy, xx = np.indices(stamp.shape)
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    work = np.where((dist2 <= max_shift**2) & (work > 0) & np.isfinite(work), work, 0.0)
    total = float(work.sum())
    if total <= 0:
        return x, y
    dx = float((work * xx).sum() / total - cx)
    dy = float((work * yy).sum() / total - cy)
    return x + dx, y + dy


def _template_polarity(template: np.ndarray, star_xy: np.ndarray, half: int = 9) -> float:
    """``+1`` if stars are brighter than sky, ``-1`` if they are dips (photographic)."""
    deltas: list[float] = []
    for x, y in np.asarray(star_xy, dtype=float).reshape(-1, 2)[:40]:
        xt, yt = _shift_to_peak(template, x, y, half, max_shift=float(min(half - 2, 10)))
        stamp = _cutout(template, xt, yt, half)
        if stamp is None:
            continue
        sky = _border_sky(stamp)
        cy, cx = stamp.shape[0] // 2, stamp.shape[1] // 2
        r = 2
        core = stamp[cy - r : cy + r + 1, cx - r : cx + r + 1]
        delta = float(np.nanmedian(core) - sky)
        if np.isfinite(delta):
            deltas.append(delta)
    if len(deltas) < 3:
        return 1.0
    return 1.0 if float(np.median(deltas)) >= 0.0 else -1.0


def _stamp_fwhm(data: np.ndarray, star_xy: np.ndarray, half: int = 11) -> float:
    """Median Gaussian-equivalent FWHM from star stamps (NaN if too few)."""
    fwhms: list[float] = []
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    for x, y in xy[:25]:
        xt, yt = _shift_to_peak(data, x, y, half, max_shift=float(min(half - 2, 10)))
        stamp = _cutout(data, xt, yt, half)
        if stamp is None or _stamp_snr(stamp) < 3.0:
            continue
        sky = _border_sky(stamp)
        work = stamp - sky
        if float(np.nanmax(work)) < float(-np.nanmin(work)):
            work = -work
        work = np.where(np.isfinite(work), work, 0.0)
        work = np.clip(work, 0.0, None)
        total = float(work.sum())
        if total <= 0:
            continue
        yy, xx = np.indices(stamp.shape)
        cy = stamp.shape[0] / 2.0 - 0.5
        cx = stamp.shape[1] / 2.0 - 0.5
        varx = float(((xx - cx) ** 2 * work).sum() / total)
        vary = float(((yy - cy) ** 2 * work).sum() / total)
        sigma = float(np.sqrt(max((varx + vary) / 2.0, 0.0)))
        if sigma > 0.3:
            fwhms.append(2.355 * sigma)
    if len(fwhms) < 2:
        return float("nan")
    return float(np.median(fwhms))


def _collect_peak_offsets(
    data: np.ndarray,
    star_xy: np.ndarray,
    half: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Science ``(x, y)`` vs template-peak offsets ``(dx, dy)``."""
    xs: list[float] = []
    ys: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    for x, y in xy[:80]:
        xt, yt = _shift_to_peak(data, x, y, half, max_shift=float(min(half - 2, 10)))
        dx, dy = xt - x, yt - y
        if abs(dx) > 8.0 or abs(dy) > 8.0:
            continue
        xs.append(float(x))
        ys.append(float(y))
        dxs.append(float(dx))
        dys.append(float(dy))
    if len(xs) < 3:
        return (
            np.empty((0, 2), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    return (
        np.column_stack([xs, ys]),
        np.asarray(dxs, dtype=float),
        np.asarray(dys, dtype=float),
    )


def _median_peak_offset(
    data: np.ndarray,
    star_xy: np.ndarray,
    half: int = 11,
) -> tuple[float, float]:
    """Median ``(dx, dy)`` from given positions to the local template peak."""
    _xy, dxs, dys = _collect_peak_offsets(data, star_xy, half=half)
    if len(dxs) < 3:
        return 0.0, 0.0
    return float(np.median(dxs)), float(np.median(dys))


def _fit_affine_offsets(
    xy: np.ndarray, dx: np.ndarray, dy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``d = a + b x + c y`` for each axis."""
    design = np.column_stack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])
    cx, *_ = np.linalg.lstsq(design, dx, rcond=None)
    cy, *_ = np.linalg.lstsq(design, dy, rcond=None)
    return np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)


def _warp_by_affine(
    image: np.ndarray, coeff_x: np.ndarray, coeff_y: np.ndarray
) -> np.ndarray:
    """Sample ``image`` at ``(x+dx, y+dy)`` so template peaks move onto science xy."""
    from scipy.ndimage import map_coordinates

    ny, nx = image.shape
    yy, xx = np.indices((ny, nx))
    dx_map = coeff_x[0] + coeff_x[1] * xx + coeff_x[2] * yy
    dy_map = coeff_y[0] + coeff_y[1] * xx + coeff_y[2] * yy
    coords = np.stack([yy + dy_map, xx + dx_map])
    warped = map_coordinates(
        image, coords, order=1, mode="nearest", prefilter=False
    )
    return np.ascontiguousarray(warped, dtype=np.float64)


def _align_template_to_stars(
    template: np.ndarray,
    star_xy: np.ndarray,
    half: int = 11,
) -> tuple[np.ndarray, float, float, str]:
    """Align the template: affine warp if the residual is field-dependent, else a shift."""
    from scipy.ndimage import shift as nd_shift

    xy, dxs, dys = _collect_peak_offsets(template, star_xy, half=half)
    if len(dxs) < 3:
        return template, 0.0, 0.0, ""
    dx_med = float(np.median(dxs))
    dy_med = float(np.median(dys))
    resid0 = np.hypot(dxs - dx_med, dys - dy_med)
    rms0 = float(np.sqrt(np.mean(resid0**2)))
    note_shift = f"Δx={dx_med:+.2f}, Δy={dy_med:+.2f} px"
    use_affine = len(dxs) >= 8 and rms0 > 0.25
    if use_affine:
        cx, cy = _fit_affine_offsets(xy, dxs, dys)
        pred_x = cx[0] + cx[1] * xy[:, 0] + cx[2] * xy[:, 1]
        pred_y = cy[0] + cy[1] * xy[:, 0] + cy[2] * xy[:, 1]
        resid1 = np.hypot(dxs - pred_x, dys - pred_y)
        rms1 = float(np.sqrt(np.mean(resid1**2)))
        if rms1 < 0.9 * rms0:
            ny, nx = template.shape
            corners = np.array(
                [[0.0, 0.0], [nx - 1.0, 0.0], [0.0, ny - 1.0], [nx - 1.0, ny - 1.0]]
            )
            cdx = cx[0] + cx[1] * corners[:, 0] + cx[2] * corners[:, 1]
            cdy = cy[0] + cy[1] * corners[:, 0] + cy[2] * corners[:, 1]
            corner = float(np.max(np.hypot(cdx, cdy)))
            aligned = _warp_by_affine(template, cx, cy)
            note = (
                f"{note_shift}, affine residual rms {rms0:.2f}→{rms1:.2f} px, "
                f"corner shift {corner:.2f} px"
            )
            return aligned, dx_med, dy_med, note
        note_shift += (
            f", affine does not help ({rms0:.2f}→{rms1:.2f} px); "
            f"star-to-star scatter {rms0:.2f} px"
        )
    if abs(dx_med) < 0.05 and abs(dy_med) < 0.05:
        return template, 0.0, 0.0, ""
    aligned = nd_shift(template, shift=(-dy_med, -dx_med), order=1, mode="nearest")
    extra = f", field rms {rms0:.2f} px" if rms0 > 0.2 else ""
    return (
        np.ascontiguousarray(aligned, dtype=np.float64),
        dx_med,
        dy_med,
        note_shift + extra,
    )


def find_kernel_stars(
    science: np.ndarray,
    *,
    n_stars: int = 40,
    fwhm: float = 4.0,
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
    img_max = float(np.nanmax(sci))
    sat = 0.95 * img_max if np.isfinite(img_max) and img_max > 0 else np.inf
    sep = float(min_separation) if min_separation is not None else max(8.0, 2.5 * fwhm)
    ny, nx = sci.shape
    margin = max(12.0, 3.0 * fwhm)
    xy: list[tuple[float, float]] = []
    for row in tbl:
        if "peak" in tbl.colnames and float(row["peak"]) > sat:
            continue
        x, y = float(row[x_col]), float(row[y_col])
        if x < margin or y < margin or x >= nx - margin or y >= ny - margin:
            continue
        if any((x - x0) ** 2 + (y - y0) ** 2 < sep**2 for x0, y0 in xy):
            continue
        xy.append((x, y))
        if len(xy) >= int(n_stars):
            break
    if not xy:
        # Interior cut was empty (small image or stars near the edge).
        for row in tbl:
            x, y = float(row[x_col]), float(row[y_col])
            if any((x - x0) ** 2 + (y - y0) ** 2 < sep**2 for x0, y0 in xy):
                continue
            xy.append((x, y))
            if len(xy) >= int(n_stars):
                break
    if not xy:
        return np.empty((0, 2))
    return np.asarray(xy, dtype=float)


def _isolate_xy(star_xy: np.ndarray, min_separation: float) -> np.ndarray:
    """Greedy keep-first isolation of ``(x, y)`` positions."""
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return xy
    sep2 = float(min_separation) ** 2
    kept: list[tuple[float, float]] = []
    for x, y in xy:
        if any((x - x0) ** 2 + (y - y0) ** 2 < sep2 for x0, y0 in kept):
            continue
        kept.append((x, y))
    return np.asarray(kept, dtype=float) if kept else xy


def _phot_geometry(fwhm: float) -> tuple[int, float]:
    """Stamp half-width and aperture radius from seeing (kept small for crowding)."""
    radius = float(np.clip(1.4 * max(fwhm, 2.0), 5.0, 16.0))
    half = int(max(np.ceil(radius + 4.0), 9))
    return half, radius


def _seeing_kernel_params(
    sci_fwhm: float,
    tmpl_fwhm: float,
    fallback_fwhm: float = 4.0,
) -> tuple[int, tuple[float, float, float]]:
    """Kernel size and Gaussian widths from PSF scale and FWHM mismatch.

    The matching Gaussian ``sqrt(FWHM_wide² − FWHM_narrow²)`` is only the
    *difference* of two Gaussians. Science and HiPS PSFs also differ in shape
    (wings, colour), so the bases must live on the scale of the stars. A
    0.9 px Gaussian on a 13 px FWHM star fits the core, rings in the wings,
    and leaves a central spike — the classic donut-with-a-dot residual.
    """
    vals = [v for v in (sci_fwhm, tmpl_fwhm) if np.isfinite(v) and v > 0.5]
    if len(vals) == 2:
        narrow, wide = min(vals), max(vals)
        extra = float(np.sqrt(max(wide**2 - narrow**2, 0.0)) / 2.355)
        fwhm_typ = 0.5 * (narrow + wide)
    elif len(vals) == 1:
        extra = float(vals[0] / 2.355)
        fwhm_typ = float(vals[0])
    else:
        extra = float(max(fallback_fwhm, 1.0) / 2.355)
        fwhm_typ = float(max(fallback_fwhm, 1.0))
    extra = float(np.clip(extra, 0.6, 10.0))
    pix = float(max(fwhm_typ, 1.0) / 2.355)
    sigmas = (
        float(max(0.35 * pix, 0.5 * extra, 0.5)),
        float(max(0.60 * pix, 1.0 * extra, 0.6)),
        float(max(0.90 * pix, 1.7 * extra, 0.8)),
    )
    ksize = int(2 * np.ceil(3.2 * max(sigmas)) + 1)
    k_psf = int(2 * np.ceil(1.15 * fwhm_typ) + 1)
    ksize = max(ksize, k_psf)
    if ksize % 2 == 0:
        ksize += 1
    ksize = int(np.clip(ksize, 15, 61))
    max_sig = (ksize / 2.0) / 3.0
    sigmas = tuple(float(min(s, max_sig)) for s in sigmas)
    return ksize, sigmas


def _robust_ratio_median(ratios: list[float], fluxes: list[float] | None = None) -> float:
    """Median flux ratio, clipping outlier *ratios* (not the brightest stars)."""
    arr = np.asarray(ratios, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Need at least 1 star stamp to estimate the flux scale")
    if arr.size >= 8:
        lo, hi = np.percentile(arr, [16.0, 84.0])
        clipped = arr[(arr >= lo) & (arr <= hi)]
        if clipped.size >= 3:
            arr = clipped
    return float(np.median(arr))


def _aperture_flux_pairs(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray,
    *,
    half: int,
    aperture_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Science and template aperture fluxes at ``star_xy`` (same sky treatment)."""
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    fs_list: list[float] = []
    ft_list: list[float] = []
    for x, y in np.asarray(star_xy, dtype=float).reshape(-1, 2):
        s = _cutout(sci, x, y, half)
        t = _cutout(tmpl, x, y, half)
        if s is None or t is None:
            continue
        if _stamp_snr(s) < 2.5 or _stamp_snr(t) < 1.5:
            continue
        fs = _aperture_flux(s, radius=aperture_radius)
        ft = _aperture_flux(t, radius=aperture_radius)
        if abs(ft) > 1e-6 and abs(fs) > 1e-6 and np.isfinite(fs) and np.isfinite(ft):
            fs_list.append(fs)
            ft_list.append(ft)
    if not fs_list:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.asarray(fs_list, dtype=float), np.asarray(ft_list, dtype=float)


def _core_residual_terciles(
    residual: np.ndarray,
    brightness: np.ndarray,
    star_xy: np.ndarray,
    half: int = 3,
) -> tuple[float, float, float]:
    """Median residual in a small core, split by brightness terciles (faint, mid, bright)."""
    cores: list[float] = []
    peaks: list[float] = []
    for x, y in np.asarray(star_xy, dtype=float).reshape(-1, 2):
        r = _cutout(residual, x, y, half)
        b = _cutout(brightness, x, y, half)
        if r is None or b is None:
            continue
        cores.append(float(np.nanmedian(r)))
        peaks.append(float(np.nanmax(np.abs(b))))
    if len(cores) < 3:
        nan = float("nan")
        return nan, nan, nan
    cores_a = np.asarray(cores, dtype=float)
    order = np.argsort(np.asarray(peaks, dtype=float))
    n = len(order)
    t = max(n // 3, 1)
    faint = cores_a[order[:t]]
    mid = cores_a[order[t : n - t]]
    bright = cores_a[order[n - t :]]
    return float(np.median(faint)), float(np.median(mid)), float(np.median(bright))


def flux_scale_from_stamps(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray,
    *,
    half: int = 11,
    aperture_radius: float = 6.0,
) -> float:
    """Robust science/template flux ratio from star stamps (after local sky)."""
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        raise RuntimeError("Need star positions to estimate the flux scale")
    ratios: list[float] = []
    fluxes: list[float] = []
    for x, y in xy:
        xt, yt = _shift_to_peak(tmpl, x, y, half, max_shift=float(min(half - 2, 10)))
        s = _cutout(sci, x, y, half)
        t = _cutout(tmpl, xt, yt, half)
        if s is None or t is None:
            continue
        if _stamp_snr(s) < 2.5 or _stamp_snr(t) < 1.5:
            continue
        fs = _aperture_flux(s, radius=aperture_radius)
        ft = _aperture_flux(t, radius=aperture_radius)
        # DSS / photographic templates have negative star cores; the ratio may
        # be negative and is applied as a signed scale (or an invert + |scale|).
        if (
            abs(ft) > 1e-6
            and abs(fs) > 1e-6
            and np.isfinite(fs)
            and np.isfinite(ft)
        ):
            ratios.append(fs / ft)
            fluxes.append(fs)
    if not ratios:
        raise RuntimeError("Need at least 1 star stamp to estimate the flux scale")
    scale = _robust_ratio_median(ratios, fluxes)
    if not np.isfinite(scale) or abs(scale) < 1e-8 or abs(scale) > 1e6:
        raise RuntimeError(f"Unusable flux scale {scale}")
    return scale


def flux_scale_subtract(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Science minus a robust scalar times the template, plus a sky offset."""
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    sci = _fill_image(sci)
    tmpl = _fill_image(tmpl)
    sci_med, sci_std = _sky_level(sci)
    tmpl_med, tmpl_std = _sky_level(tmpl)
    sci0 = sci - sci_med
    tmpl0 = tmpl - tmpl_med
    xy = None if star_xy is None else np.asarray(star_xy, dtype=float).reshape(-1, 2)
    if xy is not None and len(xy) > 0:
        try:
            scale = flux_scale_from_stamps(sci0, tmpl0, xy)
            return sci0 - scale * tmpl0, scale
        except RuntimeError:
            pass
    ok = np.isfinite(sci0) & np.isfinite(tmpl0)
    bright = ok & (sci0 > max(5.0 * sci_std, float(np.nanpercentile(sci0[ok], 90.0))))
    both = bright & (np.abs(tmpl0) > 3.0 * tmpl_std)
    if np.count_nonzero(both) < 16:
        scale = float(sci_std / tmpl_std) if tmpl_std > 0 else 1.0
        if xy is not None and len(xy) > 0:
            scale = abs(scale) * _template_polarity(tmpl0, xy)
    else:
        scale = float(np.nanmedian(sci0[both] / tmpl0[both]))
    if not np.isfinite(scale) or abs(scale) > 1e6 or abs(scale) < 1e-4:
        scale = float(sci_std / tmpl_std) if tmpl_std > 0 else 1.0
        if xy is not None and len(xy) > 0:
            scale = abs(scale) * _template_polarity(tmpl0, xy)
    return sci0 - scale * tmpl0, scale


def _norm_xy(x: float, y: float, shape: tuple[int, int]) -> tuple[float, float]:
    """Map pixel ``(x, y)`` to ``[-1, 1]`` across the frame."""
    ny, nx = int(shape[0]), int(shape[1])
    xn = 2.0 * float(x) / max(nx - 1, 1) - 1.0
    yn = 2.0 * float(y) / max(ny - 1, 1) - 1.0
    return xn, yn


def _spatial_poly(xn, yn, n_poly: int) -> tuple:
    """Polynomial terms ``(1, x, y[, x², xy, y²])``; ``xn``/``yn`` may be arrays."""
    if n_poly <= 1:
        return (1.0,)
    terms: tuple = (1.0, xn, yn)
    if n_poly >= 6:
        terms = terms + (xn * xn, xn * yn, yn * yn)
    return terms[:n_poly]


def _spatial_n_poly(spatial_order: int, n_stamps: int) -> int:
    """Spatial polynomial length: 1 (const), 3 (linear), or 6 (quadratic)."""
    order = int(spatial_order)
    used = int(n_stamps)
    if order >= 2 and used >= 16:
        return 6
    if order >= 1 and used >= 8:
        return 3
    return 1


class SpatialKernel:
    """Alard–Lupton kernel with optional spatial variation.

    Basis ``i`` is scaled by a polynomial in ``x̂, ŷ`` (pixels mapped to
    ``[-1, 1]``): constant, linear ``(1, x, y)``, or quadratic
    ``(1, x, y, x², xy, y²)``.
    """

    def __init__(self, bases: np.ndarray, coeff: np.ndarray):
        self.bases = np.asarray(bases, dtype=np.float64)
        self.coeff = np.asarray(coeff, dtype=np.float64)
        if self.coeff.ndim == 1:
            self.coeff = self.coeff.reshape(-1, 1)
        if self.coeff.shape[0] != self.bases.shape[0]:
            raise ValueError("coeff rows must match the number of basis images")
        if self.coeff.shape[1] not in (1, 3, 6):
            raise ValueError("coeff must have 1, 3, or 6 polynomial columns")

    @property
    def n_poly(self) -> int:
        return int(self.coeff.shape[1])

    @property
    def spatial(self) -> bool:
        return self.n_poly >= 3

    def kernel_sum(self, x: float = 0.0, y: float = 0.0, shape: tuple[int, int] = (2, 2)) -> float:
        if self.n_poly == 1:
            poly = np.array([1.0])
        else:
            xn, yn = _norm_xy(x, y, shape)
            poly = np.array(_spatial_poly(xn, yn, self.n_poly), dtype=np.float64)
        weights = self.coeff @ poly
        bsum = self.bases.reshape(self.bases.shape[0], -1).sum(axis=1)
        return float(np.dot(weights, bsum))

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = _as_2d(image)
        ny, nx = img.shape
        out = np.zeros((ny, nx), dtype=np.float64)
        if self.n_poly == 1:
            terms = None
        else:
            yy, xx = np.indices((ny, nx))
            xn = 2.0 * xx / max(nx - 1, 1) - 1.0
            yn = 2.0 * yy / max(ny - 1, 1) - 1.0
            terms = _spatial_poly(xn, yn, self.n_poly)
        for i, basis in enumerate(self.bases):
            conv = fftconvolve(img, basis, mode="same")
            if terms is None:
                out += self.coeff[i, 0] * conv
            else:
                w = sum(self.coeff[i, j] * terms[j] for j in range(self.n_poly))
                out += w * conv
        return out


def _lstsq_spatial_kernel(
    design: np.ndarray,
    target: np.ndarray,
    bases: np.ndarray,
    n_poly: int,
) -> np.ndarray:
    """Least-squares coefficients with flux conservation at every polynomial order."""
    n_basis = int(bases.shape[0])
    bsum = bases.reshape(n_basis, -1).sum(axis=1)
    weight = float(np.sqrt(max(design.shape[0], 1)))
    extra_rows = []
    extra_rhs = []
    for j in range(n_poly):
        row = np.zeros(n_basis * n_poly, dtype=np.float64)
        for i in range(n_basis):
            row[i * n_poly + j] = bsum[i]
        extra_rows.append(weight * row)
        extra_rhs.append(weight if j == 0 else 0.0)
    design_c = np.vstack([design, *extra_rows])
    target_c = np.concatenate([target, extra_rhs])
    coeff, *_ = np.linalg.lstsq(design_c, target_c, rcond=1e-6)
    out = coeff.reshape(n_basis, n_poly)
    # Normalize the spatially constant part so the kernel sums to 1 at the origin.
    ksum = float(np.dot(out[:, 0], bsum))
    if abs(ksum) > 1e-8:
        out[:, 0] = out[:, 0] / ksum
    return out


def fit_alard_lupton_kernel(
    science: np.ndarray,
    template: np.ndarray,
    star_xy: np.ndarray,
    *,
    ksize: int = 21,
    sigmas: tuple[float, ...] = _DEFAULT_SIGMA_SCALE,
    degrees: tuple[int, ...] = _DEFAULT_DEGREES,
    n_stars: int = 40,
    spatial_order: int = 2,
) -> tuple[SpatialKernel, int]:
    """
    Fit an AL kernel on sky-subtracted, flux-matched images.

    ``spatial_order=2`` (default) is quadratic in x and y when enough stamps
    exist; ``1`` is linear, ``0`` is spatially constant.
    """
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    bases = kernel_basis(ksize, sigmas, degrees)
    hw = ksize // 2
    half_stamp = hw + 6
    packed: list[tuple[list[np.ndarray], np.ndarray, float, float]] = []
    xy = np.asarray(star_xy, dtype=float).reshape(-1, 2)
    for x, y in xy:
        xt, yt = _shift_to_peak(tmpl, x, y, half_stamp, max_shift=10.0)
        s = _cutout(sci, x, y, half_stamp)
        t = _cutout(tmpl, xt, yt, half_stamp)
        if s is None or t is None:
            continue
        s = _replace_nonfinite(s)
        t = _replace_nonfinite(t)
        if s is None or t is None:
            continue
        inner_s = s[hw:-hw, hw:-hw]
        if inner_s.size == 0:
            continue
        if _stamp_snr(s) < 2.0 or _stamp_snr(t) < 1.5:
            continue
        s = s - _border_sky(s, border=2)
        t = t - _border_sky(t, border=2)
        if float(np.std(s)) < 1e-8 or float(np.std(t)) < 1e-8:
            continue
        convs = [fftconvolve(t, b, mode="same")[hw:-hw, hw:-hw].ravel() for b in bases]
        packed.append((convs, inner_s.ravel(), float(x), float(y)))
        if len(packed) >= int(n_stars):
            break
    used = len(packed)
    if used < 3:
        raise RuntimeError(f"Need at least 3 valid stamps to fit the kernel (got {used})")
    n_poly = _spatial_n_poly(spatial_order, used)
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for convs, inner, x, y in packed:
        xn, yn = _norm_xy(x, y, sci.shape)
        poly = _spatial_poly(xn, yn, n_poly)
        cols = [c * p for c in convs for p in poly]
        rows.append(np.column_stack(cols))
        rhs.append(inner)
    coeff = _lstsq_spatial_kernel(
        np.vstack(rows), np.concatenate(rhs), bases, n_poly
    )
    return SpatialKernel(bases, coeff), used


def alard_lupton_difference(
    science: np.ndarray,
    template: np.ndarray,
    *,
    star_xy: np.ndarray | None = None,
    n_stars: int = 40,
    fwhm: float = 4.0,
    ksize: int | None = None,
) -> tuple[np.ndarray, str]:
    """
    Science minus a PSF-matched, flux-scaled template.

    Convolves whichever image is sharper. Falls back to a scalar flux scale if
    the kernel fit has too few stamps. Returns ``(difference, method)`` with
    ``method`` ``alard_lupton`` or ``flux_scale``.
    """
    sci, tmpl = _match_shapes(_as_2d(science), _as_2d(template))
    sci = _fill_image(sci)
    tmpl = _fill_image(tmpl)
    sci_sky, _ = _sky_level(sci)
    tmpl_sky, _ = _sky_level(tmpl)
    sci0 = sci - sci_sky
    tmpl0 = tmpl - tmpl_sky
    ksize_user = ksize
    xy = star_xy
    xy_was_found = False
    try:
        if xy is None or len(np.asarray(xy).reshape(-1, 2)) == 0:
            xy_was_found = True
            xy = find_kernel_stars(
                sci0,
                n_stars=max(n_stars, 20),
                fwhm=fwhm,
                threshold_sigma=3.5,
                min_separation=max(8.0, 2.5 * fwhm),
            )
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(xy) == 0:
            raise RuntimeError("No kernel stars found on the science image")
        terminal_output.print_to_terminal(
            f"Alard–Lupton: {len(xy)} kernel-star positions",
            indent=2,
            style_name="NORMAL",
        )
        scale = flux_scale_from_stamps(
            sci0, tmpl0, xy, half=16, aperture_radius=12.0
        )
        if scale < 0:
            tmpl0 = -tmpl0
            scale = -scale
            terminal_output.print_to_terminal(
                "HiPS/template stars are dips; inverting template before the kernel fit",
                indent=2,
                style_name="WARNING",
            )
        tmpl0, _, _, align_note = _align_template_to_stars(tmpl0, xy, half=16)
        if align_note:
            terminal_output.print_to_terminal(
                f"Aligned template to science stars ({align_note})",
                indent=2,
                style_name="NORMAL",
            )
        sci_fw = _stamp_fwhm(sci0, xy, half=20)
        tmpl_fw = _stamp_fwhm(tmpl0, xy, half=20)
        meas_fw = sci_fw if np.isfinite(sci_fw) and sci_fw > 1.0 else fwhm
        sep = max(8.0, 3.5 * meas_fw)
        if xy_was_found:
            xy_refound = find_kernel_stars(
                sci0,
                n_stars=max(n_stars, 20),
                fwhm=meas_fw,
                threshold_sigma=3.5,
                min_separation=sep,
            )
            if len(xy_refound) >= 8:
                xy = xy_refound
                terminal_output.print_to_terminal(
                    f"Kernel stars re-selected for seeing {meas_fw:.1f} px "
                    f"({len(xy)} isolated, min sep {sep:.0f} px)",
                    indent=2,
                    style_name="NORMAL",
                )
        else:
            xy_iso = _isolate_xy(xy, sep)
            if len(xy_iso) >= 8:
                xy = xy_iso
        if ksize_user is None:
            ksize, sigmas = _seeing_kernel_params(sci_fw, tmpl_fw, fallback_fwhm=fwhm)
        else:
            ksize = int(ksize_user) | 1
            _, sigmas = _seeing_kernel_params(sci_fw, tmpl_fw, fallback_fwhm=fwhm)
        wide_fw = max(
            v for v in (sci_fw, tmpl_fw, fwhm) if np.isfinite(v) and v > 0
        )
        phot_half, phot_radius = _phot_geometry(wide_fw)
        seeing_ratio = 1.0
        if np.isfinite(sci_fw) and np.isfinite(tmpl_fw) and min(sci_fw, tmpl_fw) > 0.5:
            seeing_ratio = max(sci_fw, tmpl_fw) / min(sci_fw, tmpl_fw)
        # Similar seeing: a spatially varying kernel fits stamp noise, not PSF.
        if seeing_ratio >= 1.25:
            spatial_order = 2
        elif seeing_ratio >= 1.12:
            spatial_order = 1
        else:
            spatial_order = 0
            terminal_output.print_to_terminal(
                f"Seeing almost equal ({sci_fw:.1f}/{tmpl_fw:.1f} px); "
                "using a spatially constant kernel.",
                indent=2,
                style_name="NORMAL",
            )
        scale = flux_scale_from_stamps(
            sci0, tmpl0, xy, half=phot_half, aperture_radius=phot_radius
        )
        if scale < 0:
            tmpl0 = -tmpl0
            scale = -scale
        tmpl_s = scale * tmpl0
        xy_kernel = xy
        prefer_convolve_template = True
        if np.isfinite(sci_fw) and np.isfinite(tmpl_fw) and tmpl_fw > 1.15 * sci_fw:
            prefer_convolve_template = False
        if prefer_convolve_template:
            directions = [
                ("template→science", sci0, tmpl_s),
                ("science→template", tmpl_s, sci0),
            ]
        else:
            directions = [
                ("science→template", tmpl_s, sci0),
                ("template→science", sci0, tmpl_s),
            ]
        ksizes: list[int] = []
        for k_try in (ksize, min(ksize, 41), min(ksize, 31)):
            k_odd = int(k_try) | 1
            if k_odd >= 15 and k_odd not in ksizes:
                ksizes.append(k_odd)
        kernel = None
        n_used = 0
        which = directions[0][0]
        last_err: BaseException | None = None
        used_sigmas = sigmas
        for k_try in ksizes:
            sig_try = sigmas
            if k_try != ksize:
                max_sig = (k_try / 2.0) / 3.0
                sig_try = tuple(float(min(s, max_sig)) for s in sigmas)
            for label, target, source in directions:
                try:
                    kernel, n_used = fit_alard_lupton_kernel(
                        target,
                        source,
                        xy_kernel,
                        ksize=k_try,
                        sigmas=sig_try,
                        n_stars=n_stars,
                        spatial_order=spatial_order,
                    )
                    ksize = k_try
                    used_sigmas = sig_try
                    which = label
                    last_err = None
                    break
                except RuntimeError as exc:
                    last_err = exc
            if kernel is not None:
                break
        if kernel is None:
            raise last_err or RuntimeError("Kernel fit failed")
        if which == "template→science":
            matched = kernel.apply(tmpl_s)
            left, right = sci0, matched
        else:
            matched = kernel.apply(sci0)
            left, right = matched, tmpl_s
        try:
            fs, ft = _aperture_flux_pairs(
                left, right, xy_kernel, half=phot_half, aperture_radius=phot_radius
            )
            ratios = fs / ft
            phot = _robust_ratio_median(list(ratios), list(fs))
            p16, p84 = np.percentile(ratios, [16.0, 84.0])
        except Exception:
            phot, p16, p84, ratios = 1.0, 1.0, 1.0, np.array([1.0])
        if not np.isfinite(phot) or abs(phot) < 1e-6 or abs(phot) > 1e3:
            phot = 1.0
        spread = float(p84 / max(abs(p16), 1e-6))
        if spread > 2.0:
            terminal_output.print_to_terminal(
                f"Aperture flux ratios are too scattered (p84/p16={spread:.2f}); "
                "keeping the kernel flux scale (phot_match=1). Crowding and "
                "B vs PanSTARRS-g colour both limit a single factor.",
                indent=2,
                style_name="WARNING",
            )
            phot = 1.0
        residual = left - phot * right
        _, resid_sky, _ = sigma_clipped_stats(residual, sigma=3.0, maxiters=5)
        if not np.isfinite(resid_sky):
            resid_sky = 0.0
        residual = residual - float(resid_sky)
        faint_c, mid_c, bright_c = _core_residual_terciles(residual, left, xy)
        ksum = kernel.kernel_sum(
            x=sci0.shape[1] / 2.0, y=sci0.shape[0] / 2.0, shape=sci0.shape
        )
        if kernel.n_poly >= 6:
            spatial_note = "spatial=x,y,quad"
        elif kernel.spatial:
            spatial_note = "spatial=x,y"
        else:
            spatial_note = "spatial=const"
        seeing = ""
        if np.isfinite(sci_fw) and np.isfinite(tmpl_fw):
            seeing = f", seeing={sci_fw:.2f}/{tmpl_fw:.2f}px"
        terminal_output.print_to_terminal(
            f"Alard–Lupton kernel ({n_used} stamps, ksize={ksize}, "
            f"sigmas={used_sigmas[0]:.1f}/{used_sigmas[1]:.1f}/{used_sigmas[2]:.1f}, "
            f"{spatial_note}, flux_scale={scale:.4g}, phot_match={phot:.3f} "
            f"(p16={p16:.3f}, p84={p84:.3f}, n={len(ratios)}), "
            f"kernel_sum={ksum:.3f}, {which}{seeing})",
            indent=2,
            style_name="NORMAL",
        )
        if np.isfinite(faint_c) and np.isfinite(bright_c):
            terminal_output.print_to_terminal(
                f"Star-core residuals (faint/mid/bright tercile): "
                f"{faint_c:+.3g} / {mid_c:+.3g} / {bright_c:+.3g}",
                indent=2,
                style_name="NORMAL",
            )
        return residual, "alard_lupton"
    except Exception as exc:
        terminal_output.print_to_terminal(
            f"Alard–Lupton kernel fit failed ({exc}); using flux-scale subtraction",
            indent=2,
            style_name="WARNING",
        )
        diff, scale = flux_scale_subtract(sci, tmpl, star_xy=xy)
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
    fwhm: float = 4.0,
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
    "SpatialKernel",
    "alard_lupton_difference",
    "find_kernel_stars",
    "fit_alard_lupton_kernel",
    "flux_scale_from_stamps",
    "flux_scale_subtract",
    "kernel_basis",
    "run_alard_lupton",
]
