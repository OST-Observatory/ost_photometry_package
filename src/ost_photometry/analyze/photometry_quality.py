"""Background noise, saturation and aperture corrections for the extraction.

The pixel uncertainty map (``UNCERT``) holds photon, read and dark noise.
What it cannot know is measured here on the image itself:

* :func:`background_noise_factor` places apertures on empty sky and compares
  the scatter of their sky-subtracted sums with the map prediction. Faint
  undetected stars, flat-field structure, scattered light and correlated
  noise all raise it; the background part of every flux error is scaled by
  ``max(1, factor)``.
* :func:`saturated_sources` flags stars with a pixel near ``SATLEVEL`` in
  the frame *before* the 2D background subtraction.
* :func:`aperture_correction` measures, per image, how much flux bright
  isolated stars have outside the measurement aperture (or outside the PSF
  normalisation). It changes with seeing, so without it per-night or
  ensemble calibrations see seeing as a brightness change.
"""

from __future__ import annotations

import math

import numpy as np
from astropy.stats import SigmaClip
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)

#: Error of a median relative to a mean for Gaussian data, sqrt(pi/2).
MEDIAN_ERROR_FACTOR = math.sqrt(math.pi / 2.0)
_MAD_TO_SIGMA = 1.4826


def _robust_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    return float(_MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def source_disk_mask(
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
) -> np.ndarray:
    """True within ``radius`` of every detected source (for the sky only).

    A neighbour's wings in the annulus raise even a clipped median; masking
    the detected sources removes that bias (the star itself is never in its
    own annulus as long as ``radius`` < annulus inner radius).
    """
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    r = float(radius)
    ri = int(np.ceil(r))
    dy, dx = np.mgrid[-ri : ri + 1, -ri : ri + 1]
    disk = dx**2 + dy**2 <= r * r
    for xc, yc in zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float), strict=True):
        if not (np.isfinite(xc) and np.isfinite(yc)):
            continue
        ix, iy = int(round(xc)), int(round(yc))
        x0, x1 = ix - ri, ix + ri + 1
        y0, y1 = iy - ri, iy + ri + 1
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, nx), min(y1, ny)
        if x0c >= x1c or y0c >= y1c:
            continue
        mask[y0c:y1c, x0c:x1c] |= disk[sy0 : sy0 + (y1c - y0c), sx0 : sx0 + (x1c - x0c)]
    return mask


def annulus_sky(
    data: np.ndarray,
    sigma_map: np.ndarray,
    annulus: CircularAnnulus,
    mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sky level and its map-predicted noise in each annulus.

    ``sky`` is the 3σ-clipped median per pixel of the annulus pixels outside
    ``source_mask`` (:func:`source_disk_mask` of the detected stars), so
    neighbours and cosmic rays in the annulus do not pull it up;
    ``sigma_map`` is the median map σ of the annulus pixels; ``n_sky`` the
    number of pixels used.
    """
    clip = SigmaClip(sigma=3.0, maxiters=10)
    sky_mask = mask
    if source_mask is not None:
        sky_mask = source_mask if mask is None else (np.asarray(mask, dtype=bool) | source_mask)
    stats = ApertureStats(data, annulus, mask=sky_mask, sigma_clip=clip)
    sky = np.atleast_1d(np.asarray(stats.median, dtype=float))
    n_sky = np.atleast_1d(np.asarray(stats.sum_aper_area.value, dtype=float))
    map_stats = ApertureStats(sigma_map**2, annulus, mask=mask)
    sigma = np.sqrt(np.atleast_1d(np.asarray(map_stats.median, dtype=float)))
    return {"sky": sky, "sigma_map": sigma, "n_sky": n_sky}


def aperture_flux_errors(
    aperture_sum_err: np.ndarray,
    sigma_sky_map: np.ndarray,
    aperture_area: np.ndarray,
    n_sky: np.ndarray,
    noise_factor: float = 1.0,
) -> np.ndarray:
    """1σ net flux error of aperture photometry with a median sky.

    ``aperture_sum_err²`` (map) holds source and background noise. Its
    background part ``A·σ_sky²`` is replaced by ``f²·A·σ_sky²`` and the error
    of the median sky ``f²·(π/2)·A²·σ_sky²/n_sky`` is added, with the
    empirical ``noise_factor`` ``f`` (:func:`background_noise_factor`).
    """
    a = np.asarray(aperture_area, dtype=float)
    s2 = np.asarray(sigma_sky_map, dtype=float) ** 2
    total = np.asarray(aperture_sum_err, dtype=float) ** 2
    source = np.maximum(total - a * s2, 0.0)
    f2 = float(noise_factor) ** 2
    n = np.maximum(np.asarray(n_sky, dtype=float), 1.0)
    background = f2 * (a * s2 + MEDIAN_ERROR_FACTOR**2 * a**2 * s2 / n)
    return np.sqrt(source + background)


def _empty_positions(
    shape: tuple[int, int],
    sources_xy: np.ndarray,
    exclusion_radius: float,
    edge: float,
    mask: np.ndarray | None,
    n_wanted: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ny, nx = shape
    if nx <= 2 * edge or ny <= 2 * edge:
        return np.empty((0, 2))
    cand = np.column_stack(
        [rng.uniform(edge, nx - edge, 20 * n_wanted), rng.uniform(edge, ny - edge, 20 * n_wanted)]
    )
    if sources_xy is not None and len(sources_xy):
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(np.asarray(sources_xy, dtype=float)).query(cand, k=1)
        cand = cand[dist > exclusion_radius]
    if mask is not None and cand.size:
        ix = np.clip(np.round(cand[:, 0]).astype(int), 0, nx - 1)
        iy = np.clip(np.round(cand[:, 1]).astype(int), 0, ny - 1)
        cand = cand[~np.asarray(mask, dtype=bool)[iy, ix]]
    return cand[:n_wanted]


def background_noise_factor(
    data: np.ndarray,
    sigma_map: np.ndarray,
    sources_xy: np.ndarray,
    radius: float,
    r_in: float,
    r_out: float,
    *,
    mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    n_apertures: int = 400,
    seed: int = 0,
) -> tuple[float, int]:
    """Measured / predicted scatter of sky-subtracted sums on empty sky.

    Apertures of the measurement size are placed at random positions at
    least ``r_out + radius`` from every detected source and measured exactly
    like a star (clipped-median sky in the annulus). The robust scatter of
    their net sums is compared with the map prediction of
    :func:`aperture_flux_errors` for zero source flux. Returns
    ``(factor, n_used)``; ``(nan, n)`` with fewer than 30 empty apertures.
    """
    rng = np.random.default_rng(seed)
    pos = _empty_positions(
        np.shape(data),
        sources_xy,
        exclusion_radius=r_out + radius,
        edge=r_out + 2.0,
        mask=mask,
        n_wanted=n_apertures,
        rng=rng,
    )
    if len(pos) < 30:
        return float("nan"), int(len(pos))
    ap = CircularAperture(pos, radius)
    ann = CircularAnnulus(pos, r_in, r_out)
    phot = aperture_photometry(data, ap, error=sigma_map, mask=mask)
    sky = annulus_sky(data, sigma_map, ann, mask=mask, source_mask=source_mask)
    area = np.asarray(ap.area_overlap(data, mask=mask), dtype=float)
    net = np.asarray(phot["aperture_sum"], dtype=float) - sky["sky"] * area
    predicted = aperture_flux_errors(
        np.asarray(phot["aperture_sum_err"], dtype=float),
        sky["sigma_map"],
        area,
        sky["n_sky"],
        noise_factor=1.0,
    )
    measured = _robust_std(net)
    expected = float(np.nanmedian(predicted))
    if not (np.isfinite(measured) and np.isfinite(expected) and expected > 0):
        return float("nan"), int(len(pos))
    return measured / expected, int(len(pos))


def saturated_sources(
    raw_counts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    saturation_level: float,
    fraction: float = 0.9,
) -> np.ndarray:
    """True where a pixel within ``radius`` reaches ``fraction·saturation``.

    ``raw_counts`` must be in the unit of ``saturation_level`` (electrons
    for the image times ``EXPTIME``, as ``SATLEVEL`` is written by the
    reduction) and include the sky, i.e. the frame before the 2D background
    subtraction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(x.size, dtype=bool)
    if x.size == 0 or not np.isfinite(saturation_level):
        return out
    ap = CircularAperture(np.column_stack([x, y]), max(float(radius), 1.0))
    stats = ApertureStats(np.asarray(raw_counts, dtype=float), ap)
    peak = np.atleast_1d(np.asarray(stats.max, dtype=float))
    return np.isfinite(peak) & (peak >= float(fraction) * float(saturation_level))


def aperture_correction(
    data: np.ndarray,
    sigma_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    flux_measured: np.ndarray,
    flux_err: np.ndarray,
    r_large: float,
    r_in: float,
    r_out: float,
    *,
    usable: np.ndarray | None = None,
    isolation_radius: float | None = None,
    mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    min_snr: float = 50.0,
    min_stars: int = 5,
) -> tuple[float, float, int]:
    """Factor from the measured flux to the flux within ``r_large``.

    Uses bright (S/N > ``min_snr``), isolated, ``usable`` stars:
    ``c = median(F(r_large) / F_measured)``, error ``1.2533·σ_robust/√n``.
    Returns ``(nan, nan, n)`` with fewer than ``min_stars`` stars.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    flux = np.asarray(flux_measured, dtype=float)
    err = np.asarray(flux_err, dtype=float)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux / err > min_snr)
    if usable is not None:
        ok &= np.asarray(usable, dtype=bool)
    if isolation_radius is not None and x.size > 1:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=2)
        ok &= dist[:, 1] > float(isolation_radius)
    idx = np.flatnonzero(ok)
    if idx.size < min_stars:
        return float("nan"), float("nan"), int(idx.size)
    pos = np.column_stack([x[idx], y[idx]])
    phot = aperture_photometry(data, CircularAperture(pos, r_large), mask=mask)
    sky = annulus_sky(
        data, sigma_map, CircularAnnulus(pos, r_in, r_out), mask=mask, source_mask=source_mask
    )
    area = np.pi * r_large**2
    big = np.asarray(phot["aperture_sum"], dtype=float) - sky["sky"] * area
    ratio = big / flux[idx]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < min_stars:
        return float("nan"), float("nan"), int(ratio.size)
    value = float(np.median(ratio))
    spread = _robust_std(ratio)
    error = MEDIAN_ERROR_FACTOR * spread / math.sqrt(ratio.size)
    return value, float(error), int(ratio.size)


__all__ = [
    "MEDIAN_ERROR_FACTOR",
    "annulus_sky",
    "aperture_correction",
    "aperture_flux_errors",
    "background_noise_factor",
    "saturated_sources",
    "source_disk_mask",
]
