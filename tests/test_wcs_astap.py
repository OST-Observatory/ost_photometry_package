"""Tests for ASTAP WCS preprocessing helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from astropy import wcs as astropy_wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ost_photometry.wcs import (
    _ASTAP_COPY_TOP,
    _apply_wcs_to_fits,
    _astap_field_of_view_degrees,
    _needs_astap_preprocessing,
    _prepare_astap_fits,
    _scale_image_for_astap,
    wcs_maps_distinct_sky_positions,
)


def test_astap_field_of_view_uses_height():
    image = MagicMock()
    image.field_of_view_y = 13.5
    image.field_of_view_x = 18.0

    assert _astap_field_of_view_degrees(image) == 13.5 / 60.0


def test_needs_astap_preprocessing_for_float_and_negative_values():
    float_data = np.ones((4, 4), dtype=np.float32)
    uint_data = np.arange(16, dtype=np.uint16).reshape(4, 4) + 100

    assert _needs_astap_preprocessing(float_data, -32) is True
    assert _needs_astap_preprocessing(uint_data, 16) is False
    negative_values = uint_data.astype(np.int32)
    negative_values[0, 0] = -5
    assert _needs_astap_preprocessing(negative_values, 16) is True


def test_scale_image_for_astap_removes_negative_values(tmp_path: Path):
    data = np.array([[-10.0, 0.0, 5.0], [10.0, 20.0, np.nan]], dtype=np.float32)
    scaled = _scale_image_for_astap(data)

    assert scaled.dtype == np.uint16
    assert scaled.min() >= 0
    assert scaled.max() <= 60000
    assert np.isfinite(scaled).all()


def test_prepare_astap_fits_keeps_raw_uint16(tmp_path: Path):
    source = tmp_path / "raw.fit"
    data = (np.arange(16, dtype=np.uint16).reshape(4, 4) + 500)
    fits.writeto(source, data, overwrite=True)

    astap_path, is_temporary = _prepare_astap_fits(source, tmp_path)

    assert astap_path == source
    assert is_temporary is False


def test_prepare_astap_fits_converts_calibrated_float(tmp_path: Path):
    source = tmp_path / "reduced.fit"
    data = np.array([[100.0, -5.0], [200.0, 50.0]], dtype=np.float32)
    header = fits.Header()
    header["BITPIX"] = -32
    header["OBJCTRA"] = "12 30 00"
    header["OBJCTDEC"] = "+45 00 00"
    fits.writeto(source, data, header, overwrite=True)

    astap_path, is_temporary = _prepare_astap_fits(source, tmp_path)

    assert is_temporary is True
    assert astap_path != source
    assert astap_path.exists()

    with fits.open(astap_path) as hdul:
        assert hdul[0].header["BITPIX"] == 16
        assert hdul[0].data.dtype == np.uint16
        assert hdul[0].data.min() >= 0

    astap_path.unlink()


def test_apply_wcs_to_fits_replaces_conflicting_keywords(tmp_path: Path):
  source = tmp_path / "combined.fit"
  data = np.ones((64, 64), dtype=np.float32)
  header = fits.Header()
  header["BITPIX"] = -32
  header["CTYPE1"] = "RA---TAN"
  header["CTYPE2"] = "DEC--TAN"
  header["CRVAL1"] = 10.0
  header["CRVAL2"] = 20.0
  header["CRPIX1"] = 1.0
  header["CRPIX2"] = 1.0
  header["CDELT1"] = 1.0
  header["CDELT2"] = 1.0
  fits.writeto(source, data, header, overwrite=True)

  solved_header = fits.Header()
  solved_header["CTYPE1"] = "RA---TAN"
  solved_header["CTYPE2"] = "DEC--TAN"
  solved_header["CRVAL1"] = 11.776
  solved_header["CRVAL2"] = 85.29
  solved_header["CRPIX1"] = 32.5
  solved_header["CRPIX2"] = 32.5
  solved_header["CDELT1"] = 0.00018
  solved_header["CDELT2"] = 0.00018
  solved_header["CUNIT1"] = "deg"
  solved_header["CUNIT2"] = "deg"
  solved_wcs = astropy_wcs.WCS(solved_header)

  applied = _apply_wcs_to_fits(source, solved_wcs, solved_header, image_shape=(64, 64))
  corner = SkyCoord.from_pixel(0, 0, applied)
  center = SkyCoord.from_pixel(32, 32, applied)

  assert abs(corner.ra.deg - center.ra.deg) > 0.001
  assert abs(corner.dec.deg - center.dec.deg) > 0.001

  with fits.open(source) as hdul:
    reloaded = astropy_wcs.WCS(hdul[0].header)
    assert wcs_maps_distinct_sky_positions(reloaded, (64, 64))
    assert hdul[0].header["CRVAL1"] == solved_header["CRVAL1"]
    assert hdul[0].header["CDELT1"] == solved_header["CDELT1"]


def test_prepare_astap_fits_strips_existing_wcs_from_temp_header(tmp_path: Path):
    source = tmp_path / "reduced.fit"
    data = np.array([[100.0, -5.0], [200.0, 50.0]], dtype=np.float32)
    header = fits.Header()
    header["BITPIX"] = -32
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    fits.writeto(source, data, header, overwrite=True)

    astap_path, is_temporary = _prepare_astap_fits(source, tmp_path)

    assert is_temporary is True
    with fits.open(astap_path) as hdul:
        assert "CTYPE1" not in hdul[0].header

    astap_path.unlink()


def test_scale_image_for_astap_ignores_a_masked_hot_pixel():
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 5.0, (200, 200))
    data[100, 100] += 5000.0  # a star
    data[10, 10] = 1e9  # dead flat pixel, masked
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True
    scaled = _scale_image_for_astap(data, mask, "asinh")
    sky = float(np.median(scaled))
    assert abs(float(scaled[10, 10]) - sky) < 0.05 * _ASTAP_COPY_TOP  # masked: set to the sky
    assert scaled[100, 100] > 0.9 * _ASTAP_COPY_TOP  # the star keeps its dynamic range
    # unmasked hot pixel: capped at the star, which keeps its full range
    for stretch in ("linear", "asinh"):
        scaled = _scale_image_for_astap(data, stretch=stretch)
        assert scaled[100, 100] > 0.9 * _ASTAP_COPY_TOP


def test_scale_image_for_astap_keeps_bright_star_cores_apart():
    """Bright stars must keep their order: no cap at a percentile of the image."""
    yy, xx = np.mgrid[:300, :300]
    data = np.full((300, 300), 100.0)
    peaks = [20000.0, 15000.0, 10000.0, 6000.0]
    for k, peak in enumerate(peaks):
        data += peak * np.exp(-((xx - 50 - 60 * k) ** 2 + (yy - 150) ** 2) / (2 * 2.0**2))
    scaled = _scale_image_for_astap(data).astype(float)
    cores = [scaled[150, 50 + 60 * k] for k in range(len(peaks))]
    assert cores == sorted(cores, reverse=True) and len(set(cores)) == len(cores)



def test_scale_image_for_astap_resolves_the_noise_of_deep_stacks():
    """Dynamic range 10⁶ σ: the sky noise must still span many ADU."""
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 0.01, (300, 300))
    yy, xx = np.mgrid[:300, :300]
    data += 1e4 * np.exp(-((xx - 150) ** 2 + (yy - 150) ** 2) / (2 * 2.0**2))  # 10⁶ σ star
    data += 0.2 * np.exp(-((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 2.0**2))   # 20 σ star
    scaled = _scale_image_for_astap(data, stretch="asinh").astype(float)
    sky = scaled[200:290, 200:290]
    assert np.std(sky) > 100
    assert scaled[60, 60] - np.median(sky) > 5 * np.std(sky)
    assert scaled[150, 150] > scaled[60, 60]


def test_astap_retries_with_fewer_stars(monkeypatch):
    import subprocess

    from ost_photometry import wcs as wcs_module

    calls = []

    def fake_run(cmd, timeout):
        calls.append(list(cmd))
        solved = "-s" in cmd and cmd[cmd.index("-s") + 1] == "100"
        return subprocess.CompletedProcess(cmd, 0, stdout="Solution found: 23 57" if solved else "No solution found!", stderr="")

    monkeypatch.setattr(wcs_module, "_run_astap", fake_run)
    monkeypatch.setattr(wcs_module, "_astap_solution_plausible", lambda path: True)
    result = wcs_module._run_astap_solve(["astap_cli", "-f", "x.fit"], 10)
    assert wcs_module._astap_solved(result)
    assert calls == [["astap_cli", "-f", "x.fit"], ["astap_cli", "-f", "x.fit", "-s", "100"]]



def test_skewed_astap_solution_is_rejected(tmp_path):
    from astropy.io import fits

    from ost_photometry.wcs import _astap_solution_plausible

    good = tmp_path / "good.fit"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.uint16), header=fits.Header({
        "CDELT1": 1.2542e-4, "CDELT2": 1.2537e-4, "CROTA1": 43.007, "CROTA2": 43.023})).writeto(good)
    bad = tmp_path / "bad.fit"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.uint16), header=fits.Header({
        "CDELT1": 1.2548e-4, "CDELT2": 1.2614e-4, "CROTA1": 43.300, "CROTA2": 43.016})).writeto(bad)
    assert _astap_solution_plausible(str(good))
    assert not _astap_solution_plausible(str(bad))



def test_astap_tries_the_linear_copy_first_then_asinh(tmp_path, monkeypatch):
    import subprocess

    from astropy.io import fits

    from ost_photometry import wcs as wcs_module

    src = tmp_path / "stack.fit"
    fits.PrimaryHDU(np.random.default_rng(0).normal(10.0, 1.0, (64, 64)).astype(np.float32)).writeto(src)
    seen = []

    def fake_solve(cmd, timeout):
        seen.append(fits.getheader(cmd[2]).get("HISTORY"))
        solved = len(seen) == 2  # the second copy (asinh) solves
        return subprocess.CompletedProcess(cmd, 0, stdout="Solution found: 23 57" if solved else "No solution", stderr="")

    monkeypatch.setattr(wcs_module, "_run_astap_solve", fake_solve)
    result, path, is_temporary, _cmd = wcs_module._solve_with_astap(
        src, tmp_path, lambda p: ["astap_cli", "-f", str(p)], 10
    )
    assert wcs_module._astap_solved(result) and is_temporary and path.exists()
    assert len(seen) == 2
    assert len(list(tmp_path.glob("stack_astap_*.fit"))) == 1  # the failed copy was removed
