"""Inject artificial stars into a reduced image and measure them back.

Usage:
    python scripts/validation/injection_recovery.py <image.fits> <filter> <work_dir>
        [--method APER|PSF] [--radii 4 7 10] [--unit arcsec|pixel|fwhm]
        [--n 150] [--runs 2] [--seed 1]

1. The image is extracted once (``extraction.main_extract``) to get the FWHM
   and the positions and fluxes of the real stars.
2. Moffat stars (β = 2.5, the measured FWHM, Poisson noise from the image's
   EXPTIME) are added at isolated positions (no real star within 6 FWHM) and,
   as a second set, next to bright real stars (2.5–4 FWHM away). Their fluxes
   span the 10th–99th percentile of the real stars' fluxes. ``UNCERT`` gets
   the added photon noise.
3. The injected copy is extracted with the same settings; injected stars are
   matched within 1 px.

Printed: median recovered / injected flux (in mmag) per flux bin for both
sets. A flat curve means the photometry is linear; an offset common to all
bins is the profile difference between the Moffat and the real PSF (aperture
correction) and does not matter for the trend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling.functional_models import Moffat2D


def _moffat_stamp(flux: float, x: float, y: float, fwhm: float, half: int) -> tuple[np.ndarray, int, int]:
    beta = 2.5
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    x0, y0 = int(round(x)), int(round(y))
    yy, xx = np.mgrid[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    model = Moffat2D(amplitude=1.0, x_0=x, y_0=y, gamma=alpha, alpha=beta)
    # 5x5 subpixel sampling for undersampled PSFs
    sub = np.zeros(xx.shape)
    offsets = (np.arange(5) - 2) / 5.0
    for dy in offsets:
        for dx in offsets:
            sub += model(xx + dx, yy + dy)
    sub /= 25.0
    # normalise to the analytic total so the stamp's wings beyond `half` count
    total = np.pi * alpha**2 / (beta - 1.0)
    return flux * sub / total, y0 - half, x0 - half


def _extract(path: Path, filter_: str, out: Path, args):
    from ost_photometry.analyze.extraction import main_extract
    from ost_photometry.analyze.image import AnalysisImage

    out.mkdir(parents=True, exist_ok=True)
    image = AnalysisImage(0, filter_, path, out)
    main_extract(
        image,
        photometry_extraction_method=args.method,
        radius_aperture=args.radii[0],
        inner_annulus_radius=args.radii[1],
        outer_annulus_radius=args.radii[2],
        radii_unit=args.unit,
        cosmic_ray_removal=False,
        plots_for_all_images=False,
    )
    return image


def _columns(phot):
    x = np.asarray(phot["x_fit"] if "x_fit" in phot.colnames else phot["x"], float)
    y = np.asarray(phot["y_fit"] if "y_fit" in phot.colnames else phot["y"], float)
    f = np.asarray(phot["flux_fit"], float)
    return x, y, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("filter")
    ap.add_argument("work_dir")
    ap.add_argument("--method", default="APER")
    ap.add_argument("--radii", nargs=3, type=float, default=[4.0, 7.0, 10.0])
    ap.add_argument("--unit", default="arcsec")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base = _extract(Path(args.image), args.filter, work / "base", args)
    fwhm = float(base.fwhm)
    x0, y0, f0 = _columns(base.photometry)
    good = np.isfinite(f0) & (f0 > 0)
    x0, y0, f0 = x0[good], y0[good], f0[good]
    lo, hi = np.percentile(f0, [10, 99])
    print(f"FWHM {fwhm:.2f} px, {f0.size} real stars, injected flux {lo:.3g}-{hi:.3g}")

    with fits.open(args.image) as hdul:
        data0 = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()
        extra = [h.copy() for h in hdul[1:]]
    exptime = float(header.get("EXPTIME", 1.0)) or 1.0
    ny, nx = data0.shape
    half = int(np.ceil(6 * fwhm))
    bright = np.flatnonzero(f0 > np.percentile(f0, 90))

    rows = []
    for run in range(args.runs):
        data = data0.copy()
        added_var = np.zeros_like(data)
        injected = []
        tries = 0
        while len(injected) < args.n and tries < 200 * args.n:
            tries += 1
            near = len(injected) % 2 == 1 and bright.size > 0
            if near:
                k = rng.choice(bright)
                ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(2.5, 4.0) * fwhm
                x, y = x0[k] + dist * np.cos(ang), y0[k] + dist * np.sin(ang)
            else:
                x, y = rng.uniform(half + 60, nx - half - 60), rng.uniform(half + 60, ny - half - 60)
            if not (half + 60 < x < nx - half - 60 and half + 60 < y < ny - half - 60):
                continue
            d_real = np.hypot(x0 - x, y0 - y)
            others = [(ix, iy) for ix, iy, *_ in injected]
            d_inj = np.hypot(*(np.array(others) - [x, y]).T) if others else np.array([np.inf])
            if np.min(d_inj) < 8 * fwhm:
                continue
            if near and np.sum(d_real < 2.0 * fwhm) > 0:
                continue
            if not near and np.min(d_real) < 6 * fwhm:
                continue
            flux = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            stamp, sy, sx = _moffat_stamp(flux, x, y, fwhm, half)
            counts = np.clip(stamp * exptime, 0, None)
            noisy = rng.poisson(counts) / exptime
            data[sy:sy + stamp.shape[0], sx:sx + stamp.shape[1]] += noisy
            added_var[sy:sy + stamp.shape[0], sx:sx + stamp.shape[1]] += counts / exptime**2
            injected.append((x, y, flux, near))
        out_path = work / f"injected_{run}.fits"
        hdus = [fits.PrimaryHDU(data.astype(np.float32), header=header)]
        for h in extra:
            if h.name == "UNCERT" and h.data is not None:
                h.data = np.sqrt(h.data.astype(np.float64) ** 2 + added_var).astype(np.float32)
            hdus.append(h)
        fits.HDUList(hdus).writeto(out_path, overwrite=True)
        res = _extract(out_path, args.filter, work / f"run_{run}", args)
        xr, yr, fr = _columns(res.photometry)
        for x, y, flux, near in injected:
            d = np.hypot(xr - x, yr - y)
            j = int(np.argmin(d)) if d.size else -1
            if j >= 0 and d[j] < 1.0 and np.isfinite(fr[j]) and fr[j] > 0:
                rows.append((flux, fr[j], near))
    rows = np.array(rows, dtype=float)
    if rows.size == 0:
        print("no injected star recovered")
        return
    dm = -2.5 * np.log10(rows[:, 1] / rows[:, 0])
    np.savetxt(work / "injection_results.txt", np.column_stack([rows, dm]),
               header="flux_injected flux_recovered near_bright_star dmag")
    for label, sel in (("isolated", rows[:, 2] == 0), ("near bright star", rows[:, 2] == 1)):
        m = sel & np.isfinite(dm)
        if m.sum() < 10:
            continue
        ref = np.median(dm[m])
        edges = np.quantile(np.log10(rows[m, 0]), np.linspace(0, 1, 6))
        parts = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            k = m & (np.log10(rows[:, 0]) >= a) & (np.log10(rows[:, 0]) <= b)
            med = np.median(dm[k])
            err = 1.2533 * 1.4826 * np.median(np.abs(dm[k] - med)) / np.sqrt(k.sum())
            parts.append(f"F 10^{a:.2f}-10^{b:.2f}: {1000 * (med - ref):+5.1f}±{1000 * err:.1f}")
        print(f"{label}: n={int(m.sum())}, median Δm {1000 * ref:+.1f} mmag; trend (rel. to median):")
        print("   " + " | ".join(parts))


if __name__ == "__main__":
    main()
