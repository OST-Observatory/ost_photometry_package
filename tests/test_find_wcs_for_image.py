"""Tests for single-image WCS resolution and HiPS subtraction wiring."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import wcs as astropy_wcs
from astropy.io import fits

from helpers import ensure_stub_package, isolated_sys_modules, load_module_from_path, pkg_src
from ost_photometry.wcs import (
    _wcs_maps_distinct_sky_positions,
    find_wcs_for_image,
)


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    with isolated_sys_modules():
        yield


def _write_science_fits(path: Path) -> astropy_wcs.WCS:
    data = np.ones((64, 64), dtype=np.float32)
    header = fits.Header()
    header["BITPIX"] = -32
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 11.776
    header["CRVAL2"] = 85.29
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CDELT1"] = 0.00018
    header["CDELT2"] = 0.00018
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["BUNIT"] = "adu"
    fits.writeto(path, data, header, overwrite=True)
    return astropy_wcs.WCS(header)


def test_find_wcs_for_image_reuses_header_wcs(tmp_path: Path):
    fits_path = tmp_path / "science.fit"
    expected = _write_science_fits(fits_path)

    image = MagicMock()
    image.path = fits_path
    image.out_path = tmp_path

    resolved = find_wcs_for_image(image, method="astrometry", indent=2)

    assert _wcs_maps_distinct_sky_positions(resolved, (64, 64))
    assert resolved.wcs.crval[0] == expected.wcs.crval[0]
    assert resolved.wcs.crval[1] == expected.wcs.crval[1]


def _hips_module(monkeypatch):
    src = pkg_src()
    load_module_from_path("ost_photometry.style", src / "ost_photometry" / "style.py")
    load_module_from_path(
        "ost_photometry.terminal_output",
        src / "ost_photometry" / "terminal_output.py",
    )
    load_module_from_path("ost_photometry.checks", src / "ost_photometry" / "checks.py")
    load_module_from_path("ost_photometry.wcs", src / "ost_photometry" / "wcs.py")

    utilities_mod = types.ModuleType("ost_photometry.utilities")

    class _Image:
        def __init__(self, image_id, filter_, path, output_dir):
            self.image_id = image_id
            self.filter_ = filter_
            self.path = Path(path)
            self.out_path = Path(output_dir)

    utilities_mod.Image = _Image
    utilities_mod.get_basename = lambda path: Path(path).name
    sys.modules["ost_photometry.utilities"] = utilities_mod

    analyze_dir = src / "ost_photometry" / "analyze"
    ensure_stub_package("ost_photometry.analyze", path=analyze_dir)
    ensure_stub_package(
        "ost_photometry.analyze.post_processing",
        path=analyze_dir / "post_processing",
    )

    plots_mod = types.ModuleType("ost_photometry.analyze.plots")
    plots_mod.compare_images = MagicMock()
    sys.modules["ost_photometry.analyze.plots"] = plots_mod

    subtraction_mod = types.ModuleType("ost_photometry.analyze.subtraction")
    subtraction_mod.run_hotpants = MagicMock(
        return_value=Path("/tmp/diff.fits")
    )
    subtraction_mod.subtract_science_template = MagicMock(
        return_value=Path("/tmp/diff.fits")
    )

    def _resolve_backend(backend="auto", hotpants_executable=None):
        name = (backend or "auto").strip().lower()
        if name in ("alard_lupton", "alard-lupton", "python"):
            return "alard_lupton"
        if name == "hotpants":
            return "hotpants"
        return "alard_lupton"

    subtraction_mod.resolve_subtract_backend = _resolve_backend
    sys.modules["ost_photometry.analyze.subtraction"] = subtraction_mod

    models_mod = types.ModuleType("ost_photometry.analyze.models")
    models_mod.ImageSeries = MagicMock
    sys.modules["ost_photometry.analyze.models"] = models_mod

    hips_query_mod = types.ModuleType("astroquery.hips2fits")
    hips_query_mod.hips2fitsClass = MagicMock
    sys.modules["astroquery.hips2fits"] = hips_query_mod

    ccdproc_mod = types.ModuleType("ccdproc")
    ccdproc_mod.trim_image = lambda ccd: ccd
    monkeypatch.setitem(sys.modules, "ccdproc", ccdproc_mod)

    return load_module_from_path(
        "ost_photometry.analyze.post_processing.hips_reference_subtract",
        src
        / "ost_photometry"
        / "analyze"
        / "post_processing"
        / "hips_reference_subtract.py",
    )


def test_run_hips_reference_subtraction_reuses_image_wcs(
    monkeypatch,
    tmp_path: Path,
):
    hips_mod = _hips_module(monkeypatch)
    run_hips = hips_mod.run_hips_reference_subtraction

    science_path = tmp_path / "science.fit"
    _write_science_fits(science_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    reused_wcs = astropy_wcs.WCS(naxis=2)
    reused_wcs.wcs.crpix = [32.5, 32.5]
    reused_wcs.wcs.crval = [180.0, 45.0]
    reused_wcs.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    reused_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    reuse_image = MagicMock()
    reuse_image.wcs = reused_wcs

    calls: list[str] = []

    def _fake_find_wcs_for_image(image, **kwargs):
        calls.append("find_wcs_for_image")
        return astropy_wcs.WCS(naxis=2)

    captured: dict = {}

    def _fake_subtract(science_ccd, template_hdu, **kwargs):
        captured["science_ccd"] = science_ccd
        captured["template_hdu"] = template_hdu
        captured["kwargs"] = kwargs
        return workdir / "diff.fits"

    class _FakeHips:
        last_timeout = None
        last_hips = None
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            type(self).last_timeout = self.timeout
            type(self).last_hips = kwargs.get("hips")
            primary = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
            return fits.HDUList([primary])

    monkeypatch.setattr(hips_mod, "find_wcs_for_image", _fake_find_wcs_for_image)
    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FakeHips)
    monkeypatch.setattr(
        hips_mod.subtraction, "subtract_science_template", _fake_subtract
    )

    result = run_hips(
        "B",
        str(science_path),
        workdir,
        reuse_wcs_image_series=reuse_image,
        plot_comp=False,
    )

    assert calls == []
    assert result.difference_fits == workdir / "diff.fits"
    assert result.hips_source == "CDS/P/DSS2/blue"
    assert result.hips_from_cache is False
    assert result.subtract_backend == "alard_lupton"
    assert captured["science_ccd"].wcs.wcs.crval[0] == pytest.approx(180.0)
    assert captured["science_ccd"].wcs.wcs.crval[1] == pytest.approx(45.0)
    assert _FakeHips.last_timeout == pytest.approx(120.0)
    assert _FakeHips.last_hips == "CDS/P/DSS2/blue"


def test_run_hips_reference_subtraction_solves_wcs_for_single_image(
    monkeypatch,
    tmp_path: Path,
):
    hips_mod = _hips_module(monkeypatch)
    run_hips = hips_mod.run_hips_reference_subtraction

    science_path = tmp_path / "science.fit"
    _write_science_fits(science_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    calls: list[str] = []

    def _fake_find_wcs_for_image(image, **kwargs):
        calls.append(image.path.name)
        solved = astropy_wcs.WCS(naxis=2)
        solved.wcs.crpix = [32.5, 32.5]
        solved.wcs.crval = [180.0, 45.0]
        solved.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
        solved.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return solved

    class _FakeHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            primary = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
            return fits.HDUList([primary])

    monkeypatch.setattr(hips_mod, "find_wcs_for_image", _fake_find_wcs_for_image)
    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FakeHips)
    monkeypatch.setattr(
        hips_mod.subtraction,
        "subtract_science_template",
        lambda *args, **kwargs: workdir / "diff.fits",
    )

    run_hips(
        "B",
        str(science_path),
        workdir,
        reuse_wcs_image_series=None,
        plot_comp=False,
    )

    assert calls == ["science.fit"]


def test_hips_source_for_filter_bandpass_map(monkeypatch):
    hips_mod = _hips_module(monkeypatch)
    fn = hips_mod.hips_source_for_filter
    assert fn("B") == "CDS/P/DSS2/blue"
    assert fn("U") == "CDS/P/DSS2/blue"
    assert fn("V") == "CDS/P/DSS2/red"
    assert fn("R") == "CDS/P/DSS2/red"
    assert fn("I") == "CDS/P/DSS2/red"
    assert fn("g") == "CDS/P/PanSTARRS/DR1/g"
    assert fn("r") == "CDS/P/PanSTARRS/DR1/r"
    assert fn("unknown") == "CDS/P/DSS2/red"
    assert fn("V", explicit="CDS/P/PanSTARRS/DR1/g") == "CDS/P/PanSTARRS/DR1/g"
    assert fn("V", explicit="auto") == "CDS/P/DSS2/red"
    assert fn("V", explicit=None) == "CDS/P/DSS2/red"


def test_hips_timeout_seconds_converts_milliseconds(monkeypatch):
    hips_mod = _hips_module(monkeypatch)
    fn = hips_mod.hips_timeout_seconds
    assert fn(120_000) == pytest.approx(120.0)
    assert fn(30) == pytest.approx(30.0)
    assert fn(0) == pytest.approx(120.0)
    assert fn(-1) == pytest.approx(120.0)


def _dummy_wcs() -> astropy_wcs.WCS:
    wcs_obj = astropy_wcs.WCS(naxis=2)
    wcs_obj.wcs.crpix = [32.5, 32.5]
    wcs_obj.wcs.crval = [180.0, 45.0]
    wcs_obj.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    wcs_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs_obj


def _ok_hdus() -> fits.HDUList:
    return fits.HDUList([fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))])


def test_fetch_hips_cutout_uses_cache(monkeypatch, tmp_path: Path):
    hips_mod = _hips_module(monkeypatch)
    monkeypatch.setattr(hips_mod.time, "sleep", lambda *a, **k: None)
    wcs_obj = _dummy_wcs()
    shape = (64, 64)
    n_query = {"n": 0}

    class _CountingHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            n_query["n"] += 1
            return _ok_hdus()

    monkeypatch.setattr(hips_mod, "hips2fitsClass", _CountingHips)

    first = hips_mod.fetch_hips_cutout(
        wcs_obj,
        "CDS/P/DSS2/blue",
        workdir=tmp_path,
        shape=shape,
        retries=1,
        fallback_servers=(),
    )
    first[0].close()
    second = hips_mod.fetch_hips_cutout(
        wcs_obj,
        "CDS/P/DSS2/blue",
        workdir=tmp_path,
        shape=shape,
        retries=1,
        fallback_servers=(),
    )
    second[0].close()

    assert n_query["n"] == 1
    assert first[2] is False
    assert second[2] is True
    assert first[1] == second[1]
    assert first[1].is_file()


def test_fetch_hips_cutout_retries_then_succeeds(monkeypatch, tmp_path: Path):
    hips_mod = _hips_module(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(hips_mod.time, "sleep", lambda s: sleeps.append(s))
    wcs_obj = _dummy_wcs()

    class _FlakyHips:
        calls = 0
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            type(self).calls += 1
            if type(self).calls < 3:
                raise ConnectionError("temporary")
            return _ok_hdus()

    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FlakyHips)

    hdus, path, from_cache = hips_mod.fetch_hips_cutout(
        wcs_obj,
        "CDS/P/DSS2/red",
        workdir=tmp_path,
        shape=(64, 64),
        retries=3,
        retry_backoff_s=1.5,
        fallback_servers=(),
        use_cache=False,
    )
    hdus.close()

    assert _FlakyHips.calls == 3
    assert from_cache is False
    assert path.is_file()
    assert sleeps == [1.5, 3.0]


def test_fetch_hips_cutout_falls_back_to_second_server(monkeypatch, tmp_path: Path):
    hips_mod = _hips_module(monkeypatch)
    monkeypatch.setattr(hips_mod.time, "sleep", lambda *a, **k: None)
    wcs_obj = _dummy_wcs()
    primary = "https://alaskybis.example/hips2fits"
    fallback = "https://alasky.example/hips2fits"
    tried: list[str] = []

    class _ServerAwareHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            tried.append(self.server)
            if self.server == primary:
                raise ConnectionError("primary down")
            return _ok_hdus()

    monkeypatch.setattr(hips_mod, "hips2fitsClass", _ServerAwareHips)

    hdus, _path, from_cache = hips_mod.fetch_hips_cutout(
        wcs_obj,
        "CDS/P/DSS2/blue",
        workdir=tmp_path,
        shape=(64, 64),
        server=primary,
        fallback_servers=(fallback,),
        retries=1,
        use_cache=False,
    )
    hdus.close()

    assert tried == [primary, fallback]
    assert from_cache is False


def test_fetch_hips_cutout_raises_after_all_servers_fail(monkeypatch, tmp_path: Path):
    hips_mod = _hips_module(monkeypatch)
    monkeypatch.setattr(hips_mod.time, "sleep", lambda *a, **k: None)

    class _DeadHips:
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            raise ConnectionError("offline")

    monkeypatch.setattr(hips_mod, "hips2fitsClass", _DeadHips)

    with pytest.raises(RuntimeError, match="after retries"):
        hips_mod.fetch_hips_cutout(
            _dummy_wcs(),
            "CDS/P/DSS2/blue",
            workdir=tmp_path,
            shape=(64, 64),
            retries=2,
            fallback_servers=(),
            use_cache=False,
        )


def test_run_hips_maps_v_filter_to_dss2_red(monkeypatch, tmp_path: Path):
    hips_mod = _hips_module(monkeypatch)
    science_path = tmp_path / "science.fit"
    _write_science_fits(science_path)
    workdir = tmp_path / "work"
    workdir.mkdir()

    class _FakeHips:
        last_hips = None
        timeout = 0
        server = ""

        def query_with_wcs(self, **kwargs):
            type(self).last_hips = kwargs.get("hips")
            return _ok_hdus()

    monkeypatch.setattr(
        hips_mod,
        "find_wcs_for_image",
        lambda *a, **k: _dummy_wcs(),
    )
    monkeypatch.setattr(hips_mod, "hips2fitsClass", _FakeHips)
    monkeypatch.setattr(
        hips_mod.subtraction,
        "subtract_science_template",
        lambda *a, **k: workdir / "diff.fits",
    )

    result = hips_mod.run_hips_reference_subtraction(
        "V",
        str(science_path),
        workdir,
        reuse_wcs_image_series=None,
        plot_comp=False,
    )

    assert result.hips_source == "CDS/P/DSS2/red"
    assert _FakeHips.last_hips == "CDS/P/DSS2/red"
