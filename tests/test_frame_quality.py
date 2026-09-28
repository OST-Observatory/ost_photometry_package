"""Tests for per-frame quality measurement (reduce.quality)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TRUE_FWHM = 3.5
FOCAL_MM = 1000.0
PIXEL_UM = 5.0
PIXEL_SCALE = 206.265 * PIXEL_UM / FOCAL_MM


def _star_positions(n: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    ny, nx = shape
    xy: list[tuple[float, float]] = []
    while len(xy) < n:
        x = rng.uniform(20, nx - 20)
        y = rng.uniform(20, ny - 20)
        if all(np.hypot(x - a, y - b) > 14.0 for a, b in xy):
            xy.append((x, y))
    return np.asarray(xy)


def _star_field(shape: tuple[int, int], xy: np.ndarray, fwhm: float, rng) -> np.ndarray:
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.355
    img = rng.normal(100.0, 1.0, shape)
    amps = np.linspace(300.0, 3000.0, len(xy))
    for (x, y), amp in zip(xy, amps, strict=True):
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return img


def _write_light(
    path: Path,
    data: np.ndarray,
    *,
    filt: str = "V",
    jd: float = 2460000.0,
    with_geometry: bool = True,
) -> None:
    from astropy.nddata import CCDData

    ccd = CCDData(data.astype(np.float32), unit="electron/s")
    ccd.meta["IMAGETYP"] = "LIGHT"
    ccd.meta["FILTER"] = filt
    ccd.meta["JD"] = jd
    ccd.meta["EXPTIME"] = 60.0
    ccd.meta["DATE-OBS"] = "2024-01-01T00:00:00"
    ccd.meta["AIRMASS"] = 1.2
    if with_geometry:
        ccd.meta["FOCALLEN"] = FOCAL_MM
        ccd.meta["XPIXSZ"] = PIXEL_UM
    ccd.write(path, overwrite=True)


def _quality():
    pytest.importorskip("photutils")
    pytest.importorskip("ccdproc")
    from ost_photometry.reduce import quality

    return quality


@pytest.fixture
def star_frame(tmp_path):
    rng = np.random.default_rng(1)
    shape = (220, 260)
    xy = _star_positions(40, shape, rng)
    data = _star_field(shape, xy, TRUE_FWHM, rng)
    path = tmp_path / "light_V_000.fit"
    _write_light(path, data)
    return path


def test_measure_frame_quality_on_synthetic_field(star_frame):
    quality = _quality()
    row = quality.measure_frame_quality(star_frame)
    assert row["status"] == "ok"
    assert row["file"] == star_frame.name
    assert row["filter"] == "V"
    assert row["n_stars"] >= 30
    assert row["fwhm_px"] == pytest.approx(TRUE_FWHM, rel=0.25)
    assert abs(row["roundness"]) < 0.3
    assert row["pixel_scale"] == pytest.approx(PIXEL_SCALE)
    assert row["fwhm_arcsec"] == pytest.approx(row["fwhm_px"] * PIXEL_SCALE)
    assert row["background"] == pytest.approx(100.0, abs=1.0)
    assert 0.5 < row["background_rms"] < 2.0
    assert row["fwhm_source"] in {"finder_column", "psf_fit"}
    assert np.isnan(row["masked_fraction"])  # no mask stored


def test_noise_only_frame_reports_no_stars(tmp_path):
    quality = _quality()
    rng = np.random.default_rng(2)
    path = tmp_path / "cloudy.fit"
    _write_light(path, rng.normal(100.0, 1.0, (120, 120)), with_geometry=False)
    row = quality.measure_frame_quality(path)
    assert row["status"] == "no_stars"
    assert row.get("n_stars", 0) == 0
    assert "fwhm_px" not in row or np.isnan(row["fwhm_px"])
    assert np.isnan(row["pixel_scale"])


def test_pixel_scale_from_header_wcs_geometry_none():
    quality = _quality()
    from astropy.io import fits

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 100
    header["NAXIS2"] = 100
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 50.0
    header["CRPIX2"] = 50.0
    header["CDELT1"] = -0.0005
    header["CDELT2"] = 0.0005
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    assert quality.pixel_scale_from_header(header) == pytest.approx(1.8)

    geometry = fits.Header()
    geometry["FOCALLEN"] = FOCAL_MM
    geometry["PIXSIZE1"] = PIXEL_UM
    assert quality.pixel_scale_from_header(geometry) == pytest.approx(PIXEL_SCALE)

    assert quality.pixel_scale_from_header({"FOCALLEN": FOCAL_MM}) is None
    assert quality.pixel_scale_from_header({"XPIXSZ": 0.0, "FOCALLEN": 100}) is None


def test_header_roundtrip_and_rejection_move(tmp_path, star_frame):
    quality = _quality()
    from astropy.io import fits

    from ost_photometry.fits_headers import frame_rejected, frame_weight
    from ost_photometry.reduce.frame_selection import quality_table_from_rows

    row = quality.measure_frame_quality(star_frame)
    table = quality_table_from_rows([row])
    assert quality.write_quality_to_headers(table, tmp_path) == 1
    header = fits.getheader(star_frame)
    assert header["FWHM"] == pytest.approx(row["fwhm_px"])
    assert header["NSTARS"] == row["n_stars"]
    assert header["QCSTAT"] == "ok"
    assert header["PIXSCALE"] == pytest.approx(PIXEL_SCALE)

    table["stack_weight"] = np.array([1.7])
    assert quality.write_weights_to_headers(table, tmp_path) == 1
    assert frame_weight(fits.getheader(star_frame)) == pytest.approx(1.7)

    table["rejected"] = np.array([True])
    table["reject_reason"] = np.array(["fwhm 9.99 > 3.00 px"], dtype=str)
    moved = quality.move_rejected_frames(table, tmp_path, tmp_path / "rejected")
    assert moved == [tmp_path / "rejected" / star_frame.name]
    assert not star_frame.exists()
    header = fits.getheader(moved[0])
    assert frame_rejected(header)
    assert header["QCREASON"] == "fwhm 9.99 > 3.00 px"


def test_measure_directory_quality_sorts_by_filter_and_time(tmp_path):
    quality = _quality()
    rng = np.random.default_rng(3)
    shape = (160, 180)
    for name, filt, jd, fwhm in (
        ("c.fit", "V", 2460000.3, 3.0),
        ("a.fit", "V", 2460000.1, 4.0),
        ("b.fit", "B", 2460000.2, 3.5),
    ):
        xy = _star_positions(30, shape, rng)
        _write_light(tmp_path / name, _star_field(shape, xy, fwhm, rng), filt=filt, jd=jd)
    table = quality.measure_directory_quality(
        tmp_path, image_type_list=["LIGHT"], n_cores_multiprocessing=1
    )
    assert list(table["file"]) == ["b.fit", "a.fit", "c.fit"]
    assert list(table["filter"]) == ["B", "V", "V"]
    assert np.all(np.asarray(table["status"]) == "ok")
    assert table["fwhm_px"][1] > table["fwhm_px"][2]
    empty = quality.measure_directory_quality(tmp_path, image_type_list=["DARK"])
    assert len(empty) == 0


def test_assess_frame_quality_end_to_end(tmp_path):
    quality = _quality()
    from ost_photometry.fits_headers import frame_weight
    from ost_photometry.reduce.frame_selection import read_quality_table

    rng = np.random.default_rng(4)
    shape = (160, 180)
    light = tmp_path / "light"
    light.mkdir()
    fwhms = [2.8, 3.2, 3.6, 6.5]
    for i, fwhm in enumerate(fwhms):
        xy = _star_positions(30, shape, rng)
        _write_light(
            light / f"img_{i}.fit", _star_field(shape, xy, fwhm, rng), jd=2460000.0 + i
        )
    table, references = quality.assess_frame_quality(
        light,
        image_type_list=["LIGHT"],
        selection={"best_fraction": 0.75},
        stack_weighting="fwhm",
        per_filter_reference=True,
        n_cores_multiprocessing=1,
    )
    assert len(table) == 4
    assert np.asarray(table["rejected"]).tolist() == [False, False, False, True]
    assert references == {"V": "img_0.fit"}
    assert not (light / "img_3.fit").exists()
    assert (tmp_path / "rejected_lights" / "img_3.fit").exists()
    assert (tmp_path / "frame_quality.ecsv").exists()
    back = read_quality_table(tmp_path / "frame_quality.ecsv")
    assert back["is_reference"][0]
    from astropy.io import fits

    ref_header = fits.getheader(light / "img_0.fit")
    assert ref_header["QCREF"] is True
    weights = [frame_weight(fits.getheader(light / f"img_{i}.fit")) for i in range(3)]
    assert weights[0] > weights[1] > weights[2]
    assert np.mean(weights) == pytest.approx(1.0)
