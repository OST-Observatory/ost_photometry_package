"""Do stack uncertainties describe the difference between two independent stacks?

Usage::

    python scripts/validation/substack_consistency.py <stack_a.fit> <stack_b.fit>
        [<stack_full.fit>] [--radius 6] [--annulus 10 15]

Stars are found in stack A, mapped onto stack B with astroalign and measured
in both with the pipeline's aperture photometry (clipped-median sky with
detected stars masked, empty-sky noise factor). With a common flux factor c
for seeing / transparency differences, the pulls

    (F_a - c·F_b) / √(σ_a² + c²·σ_b²)

must scatter by 1 if the propagated stack errors are right. Printed per flux
bin as a robust σ (1.4826·MAD). Saturated and masked stars are left out.
With a third (full) stack its empty-sky noise factor is printed too.
"""

from __future__ import annotations

import argparse

import astroalign as aa
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

from ost_photometry.analyze.photometry_quality import (
    annulus_sky,
    aperture_flux_errors,
    background_noise_factor,
    saturated_sources,
    source_disk_mask,
)


def _load(path: str):
    ccd = CCDData.read(path)
    data = np.asarray(ccd.data, dtype=float)
    sigma = np.asarray(ccd.uncertainty.array, dtype=float)
    mask = ~np.isfinite(data) | ~np.isfinite(sigma)
    if ccd.mask is not None:
        mask |= np.asarray(ccd.mask, dtype=bool)
    bkg = Background2D(np.where(mask, np.nan, data), (80, 80), filter_size=(3, 3),
                       sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground(),
                       mask=mask)
    return ccd, data - bkg.background, data, sigma, mask


def _measure(data, sigma, mask, xy, detected, r, r_in, r_out):
    src = source_disk_mask(data.shape, detected[:, 0], detected[:, 1], min(2.0 * 2.5, 0.9 * r_in))
    ap = CircularAperture(xy, r)
    phot = aperture_photometry(data, ap, error=sigma, mask=mask)
    sky = annulus_sky(data, sigma, CircularAnnulus(xy, r_in, r_out), mask=mask, source_mask=src)
    area = np.asarray(ap.area_overlap(data, mask=mask), dtype=float)
    factor, n = background_noise_factor(data, sigma, detected, r, r_in, r_out, mask=mask,
                                        source_mask=src)
    f_used = max(1.0, factor) if np.isfinite(factor) else 1.0
    flux = np.asarray(phot["aperture_sum"], dtype=float) - sky["sky"] * area
    err = aperture_flux_errors(np.asarray(phot["aperture_sum_err"], dtype=float),
                               sky["sigma_map"], area, sky["n_sky"], f_used)
    masked = _aperture_masked(mask, xy, r)
    return flux, err, factor, masked


def _aperture_masked(mask, xy, r):
    ap = CircularAperture(xy, r)
    frac = aperture_photometry(mask.astype(float), ap)["aperture_sum"] / ap.area
    return np.asarray(frac, dtype=float) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stack_a")
    parser.add_argument("stack_b")
    parser.add_argument("stack_full", nargs="?")
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--annulus", type=float, nargs=2, default=(10.0, 15.0))
    args = parser.parse_args()
    r, (r_in, r_out) = args.radius, args.annulus

    ccd_a, data_a, raw_a, sig_a, mask_a = _load(args.stack_a)
    ccd_b, data_b, raw_b, sig_b, mask_b = _load(args.stack_b)
    _m, _med, std = sigma_clipped_stats(data_a[~mask_a], sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=10.0 * std, exclude_border=True)
    found = finder(np.where(mask_a, 0.0, data_a))
    xy_a = np.column_stack([found["x_centroid"], found["y_centroid"]])
    edge = r_out + 5
    ny, nx = data_a.shape
    keep = (xy_a[:, 0] > edge) & (xy_a[:, 0] < nx - edge) & (xy_a[:, 1] > edge) & (xy_a[:, 1] < ny - edge)
    xy_a = xy_a[keep]

    transform, _ = aa.find_transform(np.nan_to_num(data_a), np.nan_to_num(data_b))
    xy_b = transform(xy_a)
    inside = (xy_b[:, 0] > edge) & (xy_b[:, 0] < nx - edge) & (xy_b[:, 1] > edge) & (xy_b[:, 1] < ny - edge)
    xy_a, xy_b = xy_a[inside], xy_b[inside]

    f_a, e_a, fac_a, m_a = _measure(data_a, sig_a, mask_a, xy_a, xy_a, r, r_in, r_out)
    f_b, e_b, fac_b, m_b = _measure(data_b, sig_b, mask_b, xy_b, xy_b, r, r_in, r_out)
    good = ~m_a & ~m_b & np.isfinite(f_a) & np.isfinite(f_b) & (e_a > 0) & (e_b > 0)
    for ccd, raw, xy in ((ccd_a, raw_a, xy_a), (ccd_b, raw_b, xy_b)):
        sat = ccd.meta.get("SATLEVEL")
        if sat is not None:
            counts = raw * float(ccd.meta.get("EXPTIME", 1.0))
            good &= ~saturated_sources(counts, xy[:, 0], xy[:, 1], r, float(sat))
    snr = f_a / e_a
    bright = good & (snr > 100)
    c = float(np.median(f_a[bright] / f_b[bright]))
    pull = (f_a - c * f_b) / np.sqrt(e_a**2 + (c * e_b) ** 2)
    print(f"{int(good.sum())} stars; flux factor c = {c:.4f}; "
          f"empty-sky noise factor A {fac_a:.2f}, B {fac_b:.2f}")
    edges = np.quantile(snr[good], np.linspace(0, 1, 7))
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = good & (snr >= lo) & (snr <= hi)
        p = pull[m]
        robust = 1.4826 * np.median(np.abs(p - np.median(p)))
        print(f"  S/N {lo:8.1f} .. {hi:8.1f}: n={m.sum():4d}  robust σ(pull) {robust:.2f}  "
              f"median pull {np.median(p):+.2f}")
    if args.stack_full:
        _c, data_f, _raw, sig_f, mask_f = _load(args.stack_full)
        found_f = finder(np.where(mask_f, 0.0, data_f))
        xy_f = np.column_stack([found_f["x_centroid"], found_f["y_centroid"]])
        src = source_disk_mask(data_f.shape, xy_f[:, 0], xy_f[:, 1], 5.0)
        fac, n = background_noise_factor(data_f, sig_f, xy_f, r, r_in, r_out, mask=mask_f,
                                         source_mask=src)
        print(f"full stack: empty-sky noise factor {fac:.2f} ({n} apertures)")


if __name__ == "__main__":
    main()
