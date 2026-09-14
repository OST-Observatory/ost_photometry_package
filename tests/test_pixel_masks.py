"""Bad-pixel masks must not grow during registration nor bias aperture sums."""

from __future__ import annotations

import numpy as np
import pytest

from ost_photometry.core.pixel_masks import (
    aperture_masked_fraction,
    fill_masked_pixels,
    negative_outlier_mask,
    resample_mask,
)


def _sky_with_defects(rng, shape=(120, 120), n_hot=60):
    data = rng.normal(100.0, 5.0, shape)
    mask = np.zeros(shape, dtype=bool)
    ys = rng.integers(2, shape[0] - 2, n_hot)
    xs = rng.integers(2, shape[1] - 2, n_hot)
    mask[ys, xs] = True
    data[mask] = 5000.0
    return data, mask


def test_negative_outlier_mask_ignores_ordinary_noise():
    rng = np.random.default_rng(1)
    # faint sky: ~40 % of the pixels are negative through noise alone
    data = rng.normal(2.0, 8.0, (200, 200))
    assert np.mean(data < 0) > 0.3
    mask = negative_outlier_mask(data)
    assert mask.mean() < 1e-3
    data[10, 10] = -500.0
    assert negative_outlier_mask(data)[10, 10]


def test_fill_masked_pixels_uses_local_median():
    rng = np.random.default_rng(2)
    data, mask = _sky_with_defects(rng)
    filled = fill_masked_pixels(data, mask)
    assert np.all(np.abs(filled[mask] - 100.0) < 20.0)
    # untouched elsewhere and input not modified
    assert np.array_equal(filled[~mask], data[~mask])
    assert np.all(data[mask] == 5000.0)
    assert fill_masked_pixels(data, None) is data


def test_resample_mask_keeps_area_with_nearest_semantics():
    from scipy.ndimage import shift

    rng = np.random.default_rng(3)
    _, mask = _sky_with_defects(rng)
    shifted = resample_mask(mask, lambda m: shift(m, (0.4, -0.7), order=0))
    assert shifted.dtype == bool
    # nearest-neighbour keeps the masked area; the old bilinear ``> 0``
    # threshold roughly quadrupled it
    assert shifted.sum() == mask.sum()
    grown = shift(mask.astype(float), (0.4, -0.7), order=1) > 0
    assert grown.sum() >= 3 * mask.sum()
    assert resample_mask(None, lambda m: m) is None


def test_aperture_masked_fraction():
    pytest.importorskip("photutils")
    from photutils.aperture import CircularAperture

    data = np.zeros((50, 50))
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, :] = True
    ap = CircularAperture([(25.0, 25.0), (10.0, 10.0)], r=4.0)
    frac = aperture_masked_fraction(ap, data, mask)
    assert frac[1] == pytest.approx(0.0)
    assert frac[0] == pytest.approx(1.0, abs=0.05)
    assert np.all(aperture_masked_fraction(ap, data, None) == 0.0)


def test_scipy_shift_of_bool_mask_is_lost_without_helper():
    # documents the failure mode fixed in trim.py (``transform_image`` path)
    from scipy.ndimage import shift

    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    assert not shift(mask, (0.3, 0.2), order=1).any()


def test_wcs_reproject_keeps_mask_small_and_fills_defects():
    pytest.importorskip("reproject")
    pytest.importorskip("ccdproc")
    from astropy.nddata import CCDData, StdDevUncertainty
    from astropy.wcs import WCS

    from ost_photometry.reduce.registration.wcs_align import reproject_ccd_onto_wcs

    def tan(crpix):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [10.0, 20.0]
        w.wcs.crpix = list(crpix)
        w.wcs.cdelt = [-0.0002, 0.0002]
        return w

    rng = np.random.default_rng(4)
    data, mask = _sky_with_defects(rng)
    ccd = CCDData(
        data,
        unit="adu",
        mask=mask,
        wcs=tan((60.0, 60.0)),
        uncertainty=StdDevUncertainty(np.full(data.shape, 5.0)),
    )
    out = reproject_ccd_onto_wcs(ccd, tan((60.4, 59.3)), data.shape)
    interior = (slice(5, -5), slice(5, -5))
    assert out.mask[interior].sum() <= 1.5 * mask[interior].sum()
    # defects were filled before the bilinear warp: no 5000-ADU smear left
    assert np.nanmax(out.data[interior]) < 200.0


def test_astroalign_path_does_not_punch_nan_holes():
    pytest.importorskip("astroalign")
    from astropy.nddata import CCDData, StdDevUncertainty

    from ost_photometry.reduce.registration.shifts import astro_align

    rng = np.random.default_rng(5)
    shape = (160, 160)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    stars = rng.uniform(20, 140, (25, 2))

    def frame(dx, dy):
        img = rng.normal(100.0, 2.0, shape)
        for x0, y0 in stars:
            img += 3000.0 * np.exp(-((xx - x0 - dx) ** 2 + (yy - y0 - dy) ** 2) / 4.5)
        return img

    ref = CCDData(frame(0, 0), unit="adu", uncertainty=StdDevUncertainty(np.full(shape, 2.0)))
    cur_data = frame(1.7, -2.3)
    mask = np.zeros(shape, dtype=bool)
    mask[rng.integers(5, 155, 80), rng.integers(5, 155, 80)] = True
    cur_data[mask] = 60000.0
    cur = CCDData(
        cur_data,
        unit="adu",
        mask=mask,
        uncertainty=StdDevUncertainty(np.full(shape, 2.0)),
    )
    aligned, _ = astro_align(ref, cur)
    interior = (slice(8, -8), slice(8, -8))
    # no NaN holes at the defect positions, mask about as large as before
    assert np.isfinite(aligned.data[interior]).all()
    assert aligned.mask[interior].sum() <= 1.5 * mask[interior].sum()
    assert np.nanmax(aligned.data[interior]) < 5000.0
