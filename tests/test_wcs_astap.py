"""Tests for ASTAP WCS preprocessing helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from astropy.io import fits

from ost_photometry.wcs import (
    _astap_field_of_view_degrees,
    _needs_astap_preprocessing,
    _prepare_astap_fits,
    _scale_image_for_astap,
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
