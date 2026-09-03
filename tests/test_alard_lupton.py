"""Alard–Lupton kernel subtractor and backend dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from scipy.ndimage import gaussian_filter

from ost_photometry.analyze.subtraction import (
    resolve_subtract_backend,
    subtract_science_template,
)
from ost_photometry.analyze.subtraction_alard_lupton import (
    SpatialKernel,
    alard_lupton_difference,
    find_kernel_stars,
    fit_alard_lupton_kernel,
    kernel_basis,
    run_alard_lupton,
)

_STAR_XY16 = np.array(
    [[30.0 + 45.0 * (i % 4), 30.0 + 45.0 * (i // 4)] for i in range(16)],
    dtype=float,
)

_STAR_XY12 = np.array(
    [[30.0 + 50.0 * (i % 4), 30.0 + 50.0 * (i // 4)] for i in range(12)],
    dtype=float,
)

_STAR_XY = np.array(
    [
        [40.0, 40.0],
        [90.0, 50.0],
        [55.0, 100.0],
        [125.0, 110.0],
        [70.0, 145.0],
        [155.0, 75.0],
    ]
)


def _star_field(shape: tuple[int, int], xy: np.ndarray, fwhm: float) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.355
    img = np.zeros(shape, dtype=np.float64)
    amps = np.linspace(800.0, 1800.0, len(xy))
    for (x, y), amp in zip(xy, amps, strict=True):
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return img


def test_as_2d_accepts_uint8_and_rgb():
    from ost_photometry.analyze.subtraction_alard_lupton import _as_2d

    plane = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = _as_2d(plane)
    assert out.dtype == np.float64
    assert out.shape == (3, 4)
    rgb = np.stack([plane, plane, plane], axis=-1)
    assert _as_2d(rgb).shape == (3, 4)


def test_find_kernel_stars_uses_photutils_centroid_columns():
    tmpl = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    xy = find_kernel_stars(tmpl, n_stars=6, fwhm=3.0, threshold_sigma=3.0)
    assert xy.ndim == 2 and xy.shape[1] == 2
    assert len(xy) >= 3
    for x, y in xy:
        dist = np.hypot(_STAR_XY[:, 0] - x, _STAR_XY[:, 1] - y)
        assert float(np.min(dist)) < 3.0


def test_alard_lupton_without_star_xy_uses_finder():
    tmpl = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.25 * gaussian_filter(tmpl, sigma=1.1) + 12.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=None, fwhm=4.0)
    assert method in ("alard_lupton", "flux_scale")
    assert diff.shape == sci.shape


def test_seeing_kernel_params_tracks_fwhm_mismatch():
    from ost_photometry.analyze.subtraction_alard_lupton import _seeing_kernel_params

    ksize, sigmas = _seeing_kernel_params(7.5, 18.0)
    assert ksize >= 41
    assert max(sigmas) >= 6.0
    k_small, sig_small = _seeing_kernel_params(3.0, 3.2)
    assert k_small < ksize
    assert max(sig_small) < max(sigmas)


def test_seeing_kernel_params_follows_absolute_psf():
    """Similar, broad PSFs still need a kernel on the star scale, not 0.9 px."""
    from ost_photometry.analyze.subtraction_alard_lupton import _seeing_kernel_params

    ksize, sigmas = _seeing_kernel_params(13.5, 14.2)
    assert ksize >= 31
    assert min(sigmas) >= 1.8
    assert max(sigmas) >= 4.5


def test_kernel_basis_shape_and_normalization():
    bases = kernel_basis(15)
    assert bases.shape == (10, 15, 15)
    assert np.all(np.isfinite(bases))


def test_alard_lupton_cancels_convolved_template():
    tmpl = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.25 * gaussian_filter(tmpl, sigma=1.1) + 12.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    naive = sci - tmpl
    assert float(np.std(diff)) < 0.2 * float(np.std(naive))
    assert float(np.std(diff)) < 15.0
    assert abs(float(np.median(diff))) < 2.0


def test_alard_lupton_handles_sky_and_flux_mismatch():
    """HiPS-like: template sky/units differ strongly from the science CCD."""
    stars = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    tmpl = 0.04 * stars + 3500.0
    sci = 1.8 * gaussian_filter(stars, sigma=1.0) + 220.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    assert abs(float(np.median(diff))) < 8.0
    assert float(np.std(diff)) < 0.15 * float(np.std(sci - tmpl))


def test_alard_lupton_handles_inverted_and_shifted_template():
    stars = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.6 * gaussian_filter(stars, sigma=1.0) + 80.0
    shifted = np.roll(np.roll(stars, 2, axis=1), 1, axis=0)
    tmpl = -0.03 * shifted + 1200.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    assert abs(float(np.median(diff))) < 10.0
    assert float(np.std(diff)) < 0.25 * float(np.std(sci - tmpl))


def test_alard_lupton_matches_when_template_is_broader():
    """DSS/HiPS-like: template seeing is worse than the science CCD."""
    stars = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.6 * stars + 80.0
    tmpl = -0.03 * gaussian_filter(stars, sigma=1.8) + 1200.0
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    assert abs(float(np.median(diff))) < 12.0
    cores = []
    signed = []
    for x, y in _STAR_XY:
        cut = diff[int(y) - 3 : int(y) + 4, int(x) - 3 : int(x) + 4]
        cores.append(float(np.nanmax(np.abs(cut))))
        signed.append(float(np.nanmedian(cut)))
    assert float(np.median(cores)) < 0.25 * 1.6 * 1800.0
    # Systematic negative holes at every star would mean the flux scale is high.
    assert float(np.median(signed)) > -40.0


def test_alard_lupton_fits_kernel_with_nan_hips_mask():
    """HiPS cutouts often have NaN borders and holes; the kernel must still fit."""
    stars = _star_field((220, 220), _STAR_XY, fwhm=3.0)
    sci = 1.5 * stars + 90.0
    tmpl = -0.02 * gaussian_filter(stars, sigma=1.6) + 4000.0
    rng = np.random.default_rng(1)
    tmpl[rng.random(tmpl.shape) < 0.08] = np.nan
    tmpl[:18, :] = np.nan
    tmpl[-18:, :] = np.nan
    tmpl[:, :18] = np.nan
    tmpl[:, -18:] = np.nan
    diff, method = alard_lupton_difference(sci, tmpl, star_xy=_STAR_XY, fwhm=4.0)
    assert method == "alard_lupton"
    assert diff.shape == sci.shape
    assert abs(float(np.nanmedian(diff))) < 15.0


def test_phot_geometry_keeps_apertures_small():
    from ost_photometry.analyze.subtraction_alard_lupton import _phot_geometry

    half, radius = _phot_geometry(13.5)
    assert radius <= 16.0
    assert half < 30


def test_isolate_xy_enforces_separation():
    from ost_photometry.analyze.subtraction_alard_lupton import _isolate_xy

    xy = np.array([[10.0, 10.0], [12.0, 11.0], [80.0, 80.0]])
    out = _isolate_xy(xy, 10.0)
    assert len(out) == 2
    assert float(np.hypot(*(out[1] - out[0]))) >= 10.0
    from ost_photometry.analyze.subtraction_alard_lupton import _robust_ratio_median

    ft = np.linspace(10.0, 1280.0, 12)
    ratio = 4.0 - 3.0 * (ft / ft.max())
    phot = _robust_ratio_median(list(ratio), None)
    x = ft / float(np.median(ft))
    intercept = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), ratio, rcond=None)[0][0])
    assert abs(phot - float(np.median(ratio))) < 0.35
    assert phot < intercept


def test_flux_scale_from_stamps_accepts_photographic_dips():
    from ost_photometry.analyze.subtraction_alard_lupton import flux_scale_from_stamps

    stars = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = 1.5 * stars + 40.0
    tmpl = -0.02 * stars + 900.0
    scale = flux_scale_from_stamps(sci - np.median(sci), tmpl - np.median(tmpl), _STAR_XY)
    assert scale < 0
    assert abs(abs(scale) - (1.5 / 0.02)) / (1.5 / 0.02) < 0.25


def test_alard_lupton_falls_back_when_stamps_are_missing():
    xy = np.array([[40.0, 40.0], [55.0, 50.0], [65.0, 60.0]])
    tmpl = _star_field((80, 80), xy, fwhm=3.0)
    sci = 1.1 * tmpl + 5.0
    diff, method = alard_lupton_difference(
        sci, tmpl, star_xy=np.array([[2.0, 2.0]]), fwhm=3.0
    )
    assert method == "flux_scale"
    assert diff.shape == sci.shape
    assert float(np.std(diff)) < 1.0


def test_run_alard_lupton_writes_fits(tmp_path: Path):
    tmpl = _star_field((120, 120), _STAR_XY[:4], fwhm=3.0)
    sci = 1.2 * gaussian_filter(tmpl, sigma=0.8) + 4.0
    ccd = CCDData(sci, unit="adu")
    hdu = fits.PrimaryHDU(tmpl)
    out = run_alard_lupton(
        ccd,
        hdu,
        workdir=tmp_path,
        output_filename="diff.fits",
        star_xy=_STAR_XY[:4],
    )
    assert out == tmp_path / "diff.fits"
    assert out.is_file()
    written = CCDData.read(out)
    assert written.data.shape == sci.shape


def test_resolve_subtract_backend_auto(monkeypatch):
    import ost_photometry.analyze.subtraction as sub

    monkeypatch.setattr(sub.shutil, "which", lambda name: None)
    assert resolve_subtract_backend("auto") == "alard_lupton"
    assert resolve_subtract_backend("alard_lupton") == "alard_lupton"
    with pytest.raises(RuntimeError, match="hotpants"):
        resolve_subtract_backend("hotpants")

    monkeypatch.setattr(sub.shutil, "which", lambda name: "/usr/bin/hotpants")
    assert resolve_subtract_backend("auto") == "hotpants"
    assert resolve_subtract_backend("alard_lupton") == "alard_lupton"
    assert resolve_subtract_backend("hotpants") == "hotpants"


def test_spatial_kernel_constant_apply_matches_fftconvolve():
    from scipy.signal import fftconvolve

    bases = kernel_basis(11)
    coeff = np.zeros((bases.shape[0], 1))
    coeff[0, 0] = 1.0
    sk = SpatialKernel(bases, coeff)
    img = np.zeros((40, 40))
    img[20, 20] = 1.0
    assert np.allclose(sk.apply(img), fftconvolve(img, bases[0], mode="same"), atol=1e-9)


def test_spatial_kernel_linear_weights_follow_x():
    bases = kernel_basis(11)
    coeff = np.zeros((bases.shape[0], 3))
    coeff[0, 0] = 1.0
    coeff[0, 1] = 0.5
    sk = SpatialKernel(bases, coeff)
    img = np.zeros((21, 41))
    img[10, 10] = 1.0
    img[10, 30] = 1.0
    out = sk.apply(img)
    assert float(np.max(out[:, :20])) < float(np.max(out[:, 20:]))


def test_spatial_kernel_quadratic_weights_follow_x2():
    bases = kernel_basis(11)
    coeff = np.zeros((bases.shape[0], 6))
    coeff[0, 0] = 1.0
    coeff[0, 3] = 0.5
    sk = SpatialKernel(bases, coeff)
    img = np.zeros((21, 41))
    img[10, 5] = 1.0
    img[10, 20] = 1.0
    img[10, 35] = 1.0
    out = sk.apply(img)
    mid = float(np.max(out[:, 12:28]))
    assert float(np.max(out[:, :12])) > mid
    assert float(np.max(out[:, 28:])) > mid


def test_fit_falls_back_to_constant_kernel_with_few_stars():
    tmpl = _star_field((180, 180), _STAR_XY, fwhm=3.0)
    sci = gaussian_filter(tmpl, sigma=1.0)
    sk, n_used = fit_alard_lupton_kernel(sci, tmpl, _STAR_XY, ksize=15)
    assert n_used == 6
    assert not sk.spatial
    assert abs(sk.kernel_sum() - 1.0) < 1e-6


def test_fit_uses_spatial_kernel_with_enough_stars():
    tmpl = _star_field((240, 240), _STAR_XY12, fwhm=3.0)
    sci = gaussian_filter(tmpl, sigma=1.0)
    sk, n_used = fit_alard_lupton_kernel(sci, tmpl, _STAR_XY12, ksize=15)
    assert n_used >= 8
    assert sk.spatial
    assert sk.n_poly == 3
    ksum = sk.kernel_sum(x=120.0, y=120.0, shape=(240, 240))
    assert abs(ksum - 1.0) < 0.05


def test_fit_uses_quadratic_spatial_kernel_with_many_stars():
    tmpl = _star_field((260, 260), _STAR_XY16, fwhm=3.0)
    sci = gaussian_filter(tmpl, sigma=1.0)
    sk, n_used = fit_alard_lupton_kernel(sci, tmpl, _STAR_XY16, ksize=15)
    assert n_used >= 16
    assert sk.n_poly == 6
    ksum = sk.kernel_sum(x=130.0, y=130.0, shape=(260, 260))
    assert abs(ksum - 1.0) < 0.05


def test_spatial_kernel_beats_constant_on_psf_gradient():
    """Left-to-right seeing gradient: linear spatial terms should help."""
    from ost_photometry.analyze.subtraction_alard_lupton import (
        _sky_level,
        flux_scale_from_stamps,
    )

    shape = (240, 240)
    tmpl = _star_field(shape, _STAR_XY12, fwhm=3.0)
    xx = np.indices(shape)[1]
    w = xx / (shape[1] - 1)
    sci = (1.0 - w) * gaussian_filter(tmpl, sigma=0.4) + w * gaussian_filter(
        tmpl, sigma=1.6
    )
    sci0 = sci - _sky_level(sci)[0]
    tmpl0 = tmpl - _sky_level(tmpl)[0]
    tmpl_s = flux_scale_from_stamps(sci0, tmpl0, _STAR_XY12) * tmpl0
    sk_s, _ = fit_alard_lupton_kernel(
        sci0, tmpl_s, _STAR_XY12, ksize=21, spatial_order=1
    )
    sk_c, _ = fit_alard_lupton_kernel(
        sci0, tmpl_s, _STAR_XY12, ksize=21, spatial_order=0
    )
    assert sk_s.spatial
    assert not sk_c.spatial
    std_s = float(np.std(sci0 - sk_s.apply(tmpl_s)))
    std_c = float(np.std(sci0 - sk_c.apply(tmpl_s)))
    assert std_s < 0.9 * std_c


def test_quadratic_spatial_beats_linear_on_radial_psf():
    """Corner-worse seeing (r²): a quadratic kernel should beat a plane."""
    from ost_photometry.analyze.subtraction_alard_lupton import (
        _sky_level,
        flux_scale_from_stamps,
    )

    shape = (260, 260)
    tmpl = _star_field(shape, _STAR_XY16, fwhm=3.0)
    yy, xx = np.indices(shape)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r = np.hypot((xx - cx) / cx, (yy - cy) / cy)
    w = np.clip(r / np.sqrt(2.0), 0.0, 1.0)
    sci = (1.0 - w) * gaussian_filter(tmpl, sigma=0.4) + w * gaussian_filter(
        tmpl, sigma=1.6
    )
    sci0 = sci - _sky_level(sci)[0]
    tmpl0 = tmpl - _sky_level(tmpl)[0]
    tmpl_s = flux_scale_from_stamps(sci0, tmpl0, _STAR_XY16) * tmpl0
    sk_q, _ = fit_alard_lupton_kernel(
        sci0, tmpl_s, _STAR_XY16, ksize=21, spatial_order=2
    )
    sk_l, _ = fit_alard_lupton_kernel(
        sci0, tmpl_s, _STAR_XY16, ksize=21, spatial_order=1
    )
    assert sk_q.n_poly == 6
    assert sk_l.n_poly == 3
    std_q = float(np.std(sci0 - sk_q.apply(tmpl_s)))
    std_l = float(np.std(sci0 - sk_l.apply(tmpl_s)))
    assert std_q < 0.9 * std_l


def test_subtract_science_template_dispatches_to_alard_lupton(tmp_path: Path):
    tmpl = _star_field((120, 120), _STAR_XY[:4], fwhm=3.0)
    sci = 1.2 * gaussian_filter(tmpl, sigma=0.8) + 4.0
    path = subtract_science_template(
        CCDData(sci, unit="adu"),
        fits.PrimaryHDU(tmpl),
        workdir=tmp_path,
        backend="alard_lupton",
        star_xy=_STAR_XY[:4],
    )
    assert path.is_file()
    assert path.name == "diff.fits"
