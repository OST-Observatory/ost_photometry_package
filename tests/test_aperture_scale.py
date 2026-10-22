"""APER radii in FWHM, arcsec or pixel."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src
from ost_photometry.analyze.extraction import (
    aperture_radius_for_seeing,
    apertures_from_xy,
    resolve_aperture_radii,
)


def _resolve(**kw):
    args = dict(radius_aperture=None, inner_annulus_radius=None, outer_annulus_radius=None,
                radii_unit="fwhm", fwhm_pix=3.5, pixel_scale=0.5)
    args.update(kw)
    return resolve_aperture_radii(**args)


def test_default_radii_are_fwhm_multiples():
    assert _resolve() == pytest.approx((3.5, 8.75, 14.0))


def test_arcsec_radii_keep_their_meaning():
    assert _resolve(radii_unit="arcsec", radius_aperture=4.0, inner_annulus_radius=7.0,
                    outer_annulus_radius=10.0) == pytest.approx((8.0, 14.0, 20.0))
    assert _resolve(radii_unit="arcsec") == pytest.approx((10.0, 14.0, 20.0))


def test_pixel_radii_need_values():
    assert _resolve(radii_unit="pixel", radius_aperture=4.0, inner_annulus_radius=7.0,
                    outer_annulus_radius=10.0) == pytest.approx((4.0, 7.0, 10.0))
    with pytest.raises(ValueError, match="needs radius_aperture"):
        _resolve(radii_unit="pixel")


@pytest.mark.parametrize("fwhm_pix", [None, 0.0, -1.0, np.nan])
def test_fwhm_radii_require_positive_fwhm(fwhm_pix):
    with pytest.raises(ValueError, match="positive image FWHM"):
        _resolve(fwhm_pix=fwhm_pix)


def test_unordered_radii_are_rejected():
    with pytest.raises(ValueError, match="annulus inner"):
        _resolve(radius_aperture=3.0, inner_annulus_radius=2.0, outer_annulus_radius=4.0)


def test_removed_fwhm_scaling_options_raise():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    kw = cfg_mod.ExtractionConfig().main_extract_kwargs()
    assert kw["radii_unit"] == "fwhm" and kw["radius_aperture"] is None
    for key in ("aperture_scale_with_fwhm", "aperture_fwhm_factor"):
        with pytest.raises(ValueError, match="removed option"):
            cfg_mod.PipelineConfig(**{key: 1.8})
    cfg = cfg_mod.PipelineConfig(radii_unit="arcsec", radius_aperture=4.0)
    assert cfg.extraction.main_extract_kwargs()["radius_aperture"] == 4.0
    with pytest.raises(ValueError, match="radii_unit"):
        cfg_mod.PipelineConfig(radii_unit="arcmin")


def test_apertures_from_xy_matches_surviving_positions():
    aps = apertures_from_xy(
        [5.0, 50.0, 95.0],
        [50.0, 50.0, np.nan],
        radius_aperture=4.0,
        inner_annulus_radius=7.0,
        outer_annulus_radius=10.0,
    )
    assert aps is not None
    aperture, annulus = aps
    assert aperture.positions.shape[0] == 2
    assert annulus.positions.shape[0] == 2
    np.testing.assert_allclose(aperture.positions[1], [50.0, 50.0])
    assert apertures_from_xy([], [], 4.0, 7.0, 10.0) is None


@pytest.mark.parametrize("beta", [2.0, 2.5, 4.0])
def test_total_flux_correction_recovers_missing_wings(beta):
    """Fluxes measured in 2 FWHM come back to the total flux (Moffat stars)."""
    from ost_photometry.analyze.photometry_quality import total_flux_correction

    rng = np.random.default_rng(int(beta * 10))
    fwhm = 4.0
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    shape = (600, 600)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    xs, ys = np.meshgrid(np.arange(60, 560, 75), np.arange(60, 560, 75))
    xs = xs.ravel() + rng.uniform(-0.5, 0.5, xs.size)
    ys = ys.ravel() + rng.uniform(-0.5, 0.5, ys.size)
    flux = 10 ** rng.uniform(4.0, 5.0, xs.size)
    img = np.zeros(shape)
    for x, y, f in zip(xs, ys, flux, strict=True):
        img += f * (beta - 1) / (np.pi * alpha**2) * (1 + ((xx - x) ** 2 + (yy - y) ** 2) / alpha**2) ** (-beta)
    img += rng.normal(0, 1.0, shape)
    sigma = np.ones(shape)
    r2 = 2.0 * fwhm
    ee2 = 1 - (1 + (r2 / alpha) ** 2) ** (1 - beta)
    measured = flux * ee2
    out = total_flux_correction(img, sigma, xs, ys, measured, np.sqrt(measured), fwhm)
    assert out["n"] >= 20
    assert out["value"] == pytest.approx(1 / ee2, rel=0.004)


def test_neighbour_flux_fraction_matches_direct_integration():
    from ost_photometry.analyze.photometry_quality import neighbour_flux_fraction

    fwhm, beta, r = 4.0, 2.5, 8.0
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    # star at the origin, a 10x brighter neighbour 3 FWHM away, a far one
    x = np.array([100.0, 112.0, 400.0])
    y = np.array([100.0, 100.0, 400.0])
    f = np.array([1000.0, 10000.0, 5000.0])
    out = neighbour_flux_fraction(x, y, f, fwhm, r, beta=beta)
    g = np.arange(-r, r + 0.02, 0.02)
    gx, gy = np.meshgrid(g, g)
    inside = gx**2 + gy**2 <= r * r
    prof = (beta - 1) / (np.pi * alpha**2) * (1 + ((gx[inside] - 12.0) ** 2 + gy[inside] ** 2) / alpha**2) ** (-beta)
    expected = 10.0 * np.sum(prof) * 0.02**2
    assert out[0] == pytest.approx(expected, rel=0.02)
    assert out[2] == 0.0
    assert out[1] < out[0] / 50


def test_poor_seeing_radius_only_above_the_threshold():
    # FWHM 10 px at 0.5"/px = 5" > 4": the poor-seeing radius
    r, note = aperture_radius_for_seeing(None, 0.7, 4.0, 10.0, 0.5)
    assert r == 0.7 and "poor-seeing" in note
    # 3" seeing: the normal radius, no note
    assert aperture_radius_for_seeing(1.2, 0.7, 4.0, 6.0, 0.5) == (1.2, None)
    # option off
    assert aperture_radius_for_seeing(None, None, 4.0, 10.0, 0.5) == (None, None)
    # seeing unknown: normal radius with a note
    r, note = aperture_radius_for_seeing(None, 0.7, 4.0, 10.0, None)
    assert r is None and "unknown" in note


def test_poor_seeing_options_are_validated():
    from ost_photometry.analyze.pipeline import PipelineConfig

    cfg = PipelineConfig(poor_seeing_radius_aperture=0.7)
    assert cfg.main_extract_kwargs()["poor_seeing_radius_aperture"] == 0.7
    assert cfg.main_extract_kwargs()["poor_seeing_fwhm_arcsec"] == 4.0
    with pytest.raises(ValueError, match="poor_seeing_radius_aperture"):
        PipelineConfig(poor_seeing_radius_aperture=0.0)
    with pytest.raises(ValueError, match="poor_seeing_fwhm_arcsec"):
        PipelineConfig(poor_seeing_fwhm_arcsec=-1.0)
