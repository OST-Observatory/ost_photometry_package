"""Image-FWHM scaling of APER radii."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import load_module_from_path, pkg_src
from ost_photometry.analyze.extraction import resolve_aperture_radii


def test_resolve_aperture_radii_passthrough():
    assert resolve_aperture_radii(
        scale_with_fwhm=False,
        fwhm_pix=3.0,
        radius_aperture=4.0,
        inner_annulus_radius=7.0,
        outer_annulus_radius=10.0,
        radii_unit="arcsec",
    ) == (4.0, 7.0, 10.0, "arcsec")


def test_resolve_aperture_radii_scales_to_fwhm_pixels():
    r, r_in, r_out, unit = resolve_aperture_radii(
        scale_with_fwhm=True,
        fwhm_pix=3.5,
        radius_aperture=4.0,
        inner_annulus_radius=7.0,
        outer_annulus_radius=10.0,
        radii_unit="arcsec",
        aperture_fwhm_factor=2.0,
        inner_annulus_fwhm_factor=2.8,
        outer_annulus_fwhm_factor=4.0,
    )
    assert unit == "pixel"
    assert r == pytest.approx(7.0)
    assert r_in == pytest.approx(9.8)
    assert r_out == pytest.approx(14.0)


@pytest.mark.parametrize("fwhm_pix", [None, 0.0, -1.0, np.nan])
def test_resolve_aperture_radii_requires_positive_fwhm(fwhm_pix):
    with pytest.raises(ValueError, match="positive image FWHM"):
        resolve_aperture_radii(
            scale_with_fwhm=True,
            fwhm_pix=fwhm_pix,
            radius_aperture=4.0,
            inner_annulus_radius=7.0,
            outer_annulus_radius=10.0,
            radii_unit="pixel",
        )


def test_resolve_aperture_radii_rejects_unordered_factors():
    with pytest.raises(ValueError, match="FWHM factors"):
        resolve_aperture_radii(
            scale_with_fwhm=True,
            fwhm_pix=3.0,
            radius_aperture=4.0,
            inner_annulus_radius=7.0,
            outer_annulus_radius=10.0,
            radii_unit="pixel",
            aperture_fwhm_factor=3.0,
            inner_annulus_fwhm_factor=2.0,
            outer_annulus_fwhm_factor=4.0,
        )


def test_extraction_config_passes_aperture_scale_kwargs():
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        pkg_src() / "ost_photometry" / "analyze" / "pipeline" / "config.py",
    )
    ext = cfg_mod.ExtractionConfig()
    kw = ext.main_extract_kwargs()
    assert kw["aperture_scale_with_fwhm"] is False
    assert kw["aperture_fwhm_factor"] == 2.0
    assert kw["inner_annulus_fwhm_factor"] == 2.8
    assert kw["outer_annulus_fwhm_factor"] == 4.0

    cfg = cfg_mod.PipelineConfig(aperture_scale_with_fwhm=True, aperture_fwhm_factor=1.8)
    assert cfg.extraction.aperture_scale_with_fwhm is True
    assert cfg.aperture_fwhm_factor == 1.8
    assert cfg.extraction.main_extract_kwargs()["aperture_fwhm_factor"] == 1.8
