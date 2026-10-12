"""White balance on the stars of the field."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .background import sky_level

#: Fewer stars than this: no star-based balance (the factors stay 1).
MIN_STARS = 5


def star_white_balance(
    channels: Mapping[str, np.ndarray],
    *,
    fwhm: float = 4.0,
    reference: str = "G",
    min_snr: float = 10.0,
    max_stars: int = 500,
    saturation: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], int]:
    """Channel factors that make the median star neutral (``reference`` = 1).

    Stars are found on the sum of the noise-normalised channels; their flux
    is measured in each (sky-subtracted) channel against a local annulus.
    Skipped: stars with a neighbour within two apertures, stars whose peak
    reaches 80 % of ``saturation`` (per channel, same units as the data;
    without it the brightest 2 %), and stars below ``min_snr`` in any
    channel. Returns ``(factors, n_stars)``; with fewer than
    :data:`MIN_STARS` usable stars all factors are 1.
    """
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
    from photutils.detection import DAOStarFinder

    names = list(channels)
    unit = {n: 1.0 for n in names}
    data = {n: np.nan_to_num(np.asarray(channels[n], dtype=float)) for n in names}
    noise = {n: sky_level(data[n])[1] for n in names}
    detection = sum(data[n] / noise[n] for n in names) / np.sqrt(len(names))
    finder = DAOStarFinder(fwhm=float(fwhm), threshold=10.0, exclude_border=True)
    sources = finder(detection - float(np.median(detection)))
    if sources is None or len(sources) < MIN_STARS:
        return unit, 0
    xy = np.column_stack([np.asarray(sources["x_centroid"], dtype=float),
                          np.asarray(sources["y_centroid"], dtype=float)])
    radius = 2.0 * float(fwhm)
    distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    keep = distance.min(axis=1) > 2.0 * radius

    iy = np.clip(np.round(xy[:, 1]).astype(int), 0, detection.shape[0] - 1)
    ix = np.clip(np.round(xy[:, 0]).astype(int), 0, detection.shape[1] - 1)
    if saturation:
        for n in names:
            limit = float(saturation.get(n, np.inf))
            if np.isfinite(limit):
                keep &= data[n][iy, ix] < 0.8 * limit
    else:
        peaks = np.asarray(sources["peak"], dtype=float)
        keep &= peaks < np.percentile(peaks, 98)
    xy = xy[keep]
    if len(xy) < MIN_STARS:
        return unit, 0
    aperture = CircularAperture(xy, r=radius)
    annulus = CircularAnnulus(xy, r_in=2.0 * radius, r_out=3.0 * radius)
    fluxes, good = {}, np.ones(len(xy), dtype=bool)
    for n in names:
        local = ApertureStats(data[n], annulus).median
        stats = ApertureStats(data[n], aperture, local_bkg=local)
        flux = np.asarray(stats.sum, dtype=float)
        npix = np.asarray(stats.sum_aper_area.value, dtype=float)
        snr = flux / (noise[n] * np.sqrt(np.maximum(npix, 1.0)))
        good &= np.isfinite(flux) & (flux > 0) & (snr >= min_snr)
        fluxes[n] = flux
    if good.sum() > max_stars:
        brightest = np.argsort(-np.where(good, fluxes[reference], -np.inf))[:max_stars]
        good = np.zeros_like(good)
        good[brightest] = True
    n_good = int(np.count_nonzero(good))
    if n_good < MIN_STARS or reference not in fluxes:
        return unit, n_good
    factors = {n: float(1.0 / np.median(fluxes[n][good] / fluxes[reference][good]))
               for n in names}
    return factors, n_good


__all__ = ["MIN_STARS", "star_white_balance"]
