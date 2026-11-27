"""Field of view of an Image when FOCALLEN is missing or 0 (old SBIG frames)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.image import DEFAULT_FOCAL_LENGTH_MM, Image

NX, NY, PIXEL_UM = 1530, 1020, 9.0


def _image(tmp_path, name, **cards):
    header = fits.Header({"XPIXSZ": PIXEL_UM, "INSTRUME": "SBIG ST-8",
                          "DATE-OBS": "2008-10-01T20:00:00", "EXPTIME": 60.0, **cards})
    path = tmp_path / name
    fits.writeto(path, np.zeros((NY, NX), dtype=np.float32), header)
    return Image(0, "V", str(path), str(tmp_path))


def _fov_x(focal_mm):
    return math.degrees(2 * math.atan(NX * PIXEL_UM / 1000 / 2 / focal_mm)) * 60.0


def test_focal_length_from_the_header(tmp_path):
    assert _image(tmp_path, "a.fit", FOCALLEN=2800.0).field_of_view_x == pytest.approx(_fov_x(2800.0))


def test_zero_focal_length_uses_the_wcs(tmp_path):
    """FOCALLEN = 0 divided by zero (ZeroDivisionError in the alignment)."""
    scale = 0.4366 / 3600.0
    image = _image(tmp_path, "b.fit", FOCALLEN=0, CTYPE1="RA---TAN", CTYPE2="DEC--TAN",
                   CRVAL1=359.33, CRVAL2=56.72, CRPIX1=765.0, CRPIX2=510.0,
                   CD1_1=-scale, CD1_2=0.0, CD2_1=0.0, CD2_2=scale)
    assert image.field_of_view_x == pytest.approx(NX * 0.4366 / 60.0, rel=1e-3)
    assert image.field_of_view_y == pytest.approx(NY * 0.4366 / 60.0, rel=1e-3)
    assert image.pixel_scale == pytest.approx(0.4366, rel=1e-3)


def test_zero_focal_length_uses_the_telescope_or_the_default(tmp_path, monkeypatch):
    from ost_photometry import terminal_output

    messages = []
    monkeypatch.setattr(terminal_output, "print_to_terminal",
                        lambda text, **kw: messages.append(text))
    image = _image(tmp_path, "c.fit", FOCALLEN=0, TELESCOP="Celestron C11")
    assert image.field_of_view_x == pytest.approx(_fov_x(2800.0))
    assert not any("No focal length" in m for m in messages)
    image = _image(tmp_path, "d.fit", FOCALLEN=0)
    assert image.field_of_view_x == pytest.approx(_fov_x(DEFAULT_FOCAL_LENGTH_MM))
    assert any("No focal length" in m for m in messages)
