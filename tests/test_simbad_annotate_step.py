"""Simbad annotation as a post-processing pipeline step."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src

_PRESENT_WCS = object()
_SRC = pkg_src() / "ost_photometry"


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _stub_if_dummy(mod, **attrs) -> None:
    if getattr(mod, "__file__", None) is None:
        for key, val in attrs.items():
            setattr(mod, key, val)


def _stub_simbad_deps(*, annotate=None) -> None:
    analyze = _SRC / "analyze"
    ensure_stub_package("ost_photometry", path=_SRC)
    ensure_stub_package("ost_photometry.analyze", path=analyze)
    ensure_stub_package("ost_photometry.analyze.pipeline", path=analyze / "pipeline")
    ensure_stub_package(
        "ost_photometry.analyze.pipeline.steps",
        path=analyze / "pipeline" / "steps",
    )
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze / "post_processing",
    )
    ensure_stub_package("ost_photometry.analyze.plots")
    ensure_stub_package("ost_photometry.terminal_output")
    ensure_stub_package("astroquery")
    ensure_stub_package("astroquery.exceptions")
    ensure_stub_package("astroquery.simbad")

    _stub_if_dummy(
        sys.modules["ost_photometry.terminal_output"],
        print_to_terminal=lambda *a, **k: None,
    )
    _stub_if_dummy(
        sys.modules["astroquery.exceptions"],
        TableParseError=type("TableParseError", (Exception,), {}),
    )
    _stub_if_dummy(sys.modules["astroquery.simbad"], Simbad=MagicMock())
    if annotate is not None:
        helper = ensure_stub_package(
            "ost_photometry.analyze.post_processing.simbad_annotate"
        )
        helper.annotate_reference_image_with_simbad = annotate


def _load_config_context_step(*, annotate=None):
    _stub_simbad_deps(annotate=annotate or (lambda *a, **k: None))
    cfg_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.config",
        _SRC / "analyze" / "pipeline" / "config.py",
    )
    ctx_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.context",
        _SRC / "analyze" / "pipeline" / "context.py",
    )
    load_module_from_path(
        "ost_photometry.analyze.pipeline.base",
        _SRC / "analyze" / "pipeline" / "base.py",
    )
    step_mod = load_module_from_path(
        "ost_photometry.analyze.pipeline.steps.simbad_annotate",
        _SRC / "analyze" / "pipeline" / "steps" / "simbad_annotate.py",
    )
    return cfg_mod, ctx_mod, step_mod


def _load_helper():
    _stub_simbad_deps()
    return load_module_from_path(
        "ost_photometry.analyze.post_processing.simbad_annotate",
        _SRC / "analyze" / "post_processing" / "simbad_annotate.py",
    )


def _image(*, wcs=_PRESENT_WCS, filter_="V"):
    image = SimpleNamespace(
        wcs=wcs,
        out_path=Path("/tmp/out"),
        filter_=filter_,
        image_id=0,
    )
    image.get_data = MagicMock(return_value=np.ones((8, 8)))
    return image


def _series(*images):
    return SimpleNamespace(image_list=list(images), reference_image_index=0)


def test_default_pipeline_includes_simbad_annotate_after_extraction():
    text = (_SRC / "analyze" / "pipeline" / "orchestrator.py").read_text()
    extract_at = text.find("ExtractionStep(),")
    simbad_at = text.find("SimbadAnnotateStep(),")
    intra_at = text.find("CorrelationIntraStep(),")
    assert 0 <= extract_at < simbad_at < intra_at


def test_skip_single_mode_when_annotate_image_false():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig(extraction_mode="single", annotate_image=False)
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is True


def test_does_not_skip_by_default():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig()
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is False


def test_skip_multi_mode_when_annotate_reference_image_false():
    cfg_mod, ctx_mod, step_mod = _load_config_context_step()
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={"V": _series(_image(), _image())},
        filter_list=["V"],
        output_dir="/tmp/out",
    )
    cfg = cfg_mod.PipelineConfig(
        extraction_mode="multi",
        annotate_reference_image=False,
    )
    assert step_mod.SimbadAnnotateStep().skip(ctx, cfg) is True


def test_run_annotates_each_filter_reference_image():
    called = []

    def _fake(image, **kwargs):
        called.append(image.filter_)

    cfg_mod, ctx_mod, step_mod = _load_config_context_step(annotate=_fake)
    ctx = ctx_mod.AnalysisContext(
        image_series_dict={
            "V": _series(_image(filter_="V")),
            "B": _series(_image(filter_="B")),
        },
        filter_list=["V", "B"],
        output_dir="/tmp/out",
    )
    step_mod.SimbadAnnotateStep().run(ctx, cfg_mod.PipelineConfig())
    assert called == ["V", "B"]


def test_annotate_skips_when_wcs_missing():
    helper = _load_helper()
    marked = MagicMock()
    helper.mark_simbad_objects_on_image = marked
    helper.annotate_reference_image_with_simbad(_image(wcs=None))
    marked.assert_not_called()


def test_annotate_swallows_query_errors():
    helper = _load_helper()
    helper.mark_simbad_objects_on_image = MagicMock(
        side_effect=RuntimeError("simbad down")
    )
    helper.annotate_reference_image_with_simbad(_image())


def _tan_wcs(*, crval1: float, crval2: float, npix: int = 200, cdelt: float = 0.001):
    pytest.importorskip("astropy")
    from astropy.wcs import WCS

    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": npix,
            "NAXIS2": npix,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": crval1,
            "CRVAL2": crval2,
            "CRPIX1": npix / 2 + 0.5,
            "CRPIX2": npix / 2 + 0.5,
            "CDELT1": -cdelt,
            "CDELT2": cdelt,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }
    )


def test_search_cone_handles_ra_wrap_near_zero_hours():
    """NGC 7789-like fields must not yield a ~180 deg Simbad radius from RA min/max."""
    pytest.importorskip("astropy.units")
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    helper = _load_helper()
    wcs_image = _tan_wcs(crval1=359.35, crval2=56.7, npix=1000, cdelt=0.001)
    center, radius = helper.search_cone_from_wcs(wcs_image, (1000, 1000))
    expected = SkyCoord(ra=359.35 * u.deg, dec=56.7 * u.deg)
    assert center.separation(expected).to_value(u.deg) < 0.05
    radius_deg = float(radius.to_value(u.deg))
    assert 0.3 < radius_deg < 2.0


def test_search_cone_equator_matches_half_diagonal():
    pytest.importorskip("astropy.units")
    import astropy.units as u

    helper = _load_helper()
    wcs_image = _tan_wcs(crval1=180.0, crval2=0.0, npix=101, cdelt=0.01)
    center, radius = helper.search_cone_from_wcs(wcs_image, (101, 101))
    assert abs(center.ra.degree - 180.0) < 0.01
    assert abs(center.dec.degree) < 0.01
    assert 0.6 < float(radius.to_value(u.deg)) < 0.8


def test_simbad_query_radius_clips_above_90_and_rejects_nonpositive():
    pytest.importorskip("astropy.units")
    import astropy.units as u

    helper = _load_helper()
    assert helper._simbad_query_radius_deg(120 * u.deg) == 90.0
    assert helper._simbad_query_radius_deg(0 * u.deg) is None
    assert helper._simbad_query_radius_deg(1.5 * u.deg) == pytest.approx(1.5)


def test_simbad_query_criteria_combines_mag_type_and_common_name():
    helper = _load_helper()
    assert helper.simbad_query_criteria() is None
    mag = helper.simbad_query_criteria(mag_limit=16)
    assert "allfluxes.V < 16.0" in mag
    assert "otype IN ('OpC', 'GlC', 'Cl*', 'As*')" in mag
    mag_b = helper.simbad_query_criteria(filter_mag="B", mag_limit=15.5)
    assert "allfluxes.B < 15.5" in mag_b
    assert "OpC" in mag_b
    criteria = helper.simbad_query_criteria(
        mag_limit=16,
        otypes=["Star", "Galaxy"],
        require_common_name=True,
    )
    assert "allfluxes.V < 16.0" in criteria
    assert "otype = 'Star..'" in criteria
    assert "otype = 'Galaxy..'" in criteria
    assert "ids.ids LIKE '%NAME %'" in criteria


def test_filter_simbad_objects_magnitude_and_common_name():
    pytest.importorskip("astropy.table")
    from astropy.table import Table

    helper = _load_helper()
    table = Table(
        {
            "main_id": ["NGC 7789", "NAME Bright Star", "TYC 123"],
            "V": [8.1, 10.2, 18.4],
            "otype": ["OpC", "Star", "Star"],
            "ids": [
                "NGC 7789|Cl Melotte 245",
                "HD 1|NAME Bright Star",
                "TYC 123-1-1",
            ],
        }
    )
    bright = helper.filter_simbad_objects(table, filter_mag="V", mag_limit=16.0)
    assert list(bright["main_id"]) == ["NGC 7789", "NAME Bright Star"]
    named = helper.filter_simbad_objects(table, require_common_name=True)
    assert list(named["main_id"]) == ["NAME Bright Star"]
    stars = helper.filter_simbad_objects(table, otypes=["Star"])
    assert list(stars["main_id"]) == ["NAME Bright Star", "TYC 123"]

    from astropy.table import MaskedColumn

    no_v = Table(
        {
            "main_id": ["NGC 7789", "TYC 123"],
            "V": MaskedColumn([np.nan, 18.4], mask=[True, False]),
            "otype": ["OpC", "*"],
        }
    )
    kept = helper.filter_simbad_objects(no_v, filter_mag="V", mag_limit=16.0)
    assert list(kept["main_id"]) == ["NGC 7789"]


def test_simbad_magnitude_column_accepts_tap_and_legacy_names():
    pytest.importorskip("astropy.table")
    from astropy.table import Table

    helper = _load_helper()
    tap = Table({"V": [10.0]})
    legacy = Table({"FLUX_V": [10.0]})
    assert helper.simbad_magnitude_column(tap, "V") == "V"
    assert helper.simbad_magnitude_column(legacy, "V") == "FLUX_V"


def test_left_join_allfluxes_does_not_drop_objects_without_v():
    from dataclasses import dataclass

    helper = _load_helper()

    @dataclass(frozen=True)
    class _Join:
        table: str
        join_type: str = "JOIN"

    simbad = SimpleNamespace(
        joins=[_Join("allfluxes"), _Join("ident")],
    )
    helper._left_join_allfluxes(simbad)
    assert simbad.joins[0].join_type == "LEFT JOIN"
    assert simbad.joins[1].join_type == "JOIN"


def test_tap_degrees_land_on_ngc7789_frame_hourangle_does_not():
    """Mis-reading TAP RA as hourangle throws NGC 7789 off the image."""
    pytest.importorskip("astropy.coordinates")
    from astropy.coordinates import SkyCoord

    galaxy = load_module_from_path(
        "ost_photometry.analyze.plots.simbad_galaxy",
        _SRC / "analyze" / "plots" / "simbad_galaxy.py",
    )
    wcs_image = _tan_wcs(crval1=359.35, crval2=56.73, npix=1000, cdelt=0.001)
    x, y = wcs_image.world_to_pixel(galaxy.skycoord_from_simbad(359.35, 56.73))
    assert 400 < float(x) < 600
    assert 400 < float(y) < 600
    wrong = SkyCoord(ra=359.35, dec=56.73, unit=("hourangle", "deg"))
    xw, yw = wcs_image.world_to_pixel(wrong)
    assert not (0 <= float(xw) < 1000 and 0 <= float(yw) < 1000)
