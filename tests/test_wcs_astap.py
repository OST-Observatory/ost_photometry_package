"""Tests for ASTAP WCS preprocessing helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import wcs as astropy_wcs

from ost_photometry.wcs import (
    _apply_wcs_to_fits,
    _astap_field_of_view_degrees,
    _needs_astap_preprocessing,
    _prepare_astap_fits,
    _scale_image_for_astap,
    _strip_wcs_keywords,
    _wcs_maps_distinct_sky_positions,
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
    assert _wcs_maps_distinct_sky_positions(reloaded, (64, 64))
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
