"""ASTAP search centre: pointing keywords in the header, target position."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ost_photometry.reduce.workflow.combine import add_pointing_hints
from ost_photometry.wcs import pointing_from_header

NGC7789 = (359.33, 56.72)


@pytest.mark.parametrize(
    "cards, expected",
    [
        ({"OBJCTRA": "23 57 19.2", "OBJCTDEC": "+56 43 12"}, NGC7789),
        ({"OBJCTRA": "", "OBJCTDEC": "", "RA": 359.33, "DEC": 56.72}, NGC7789),
        ({"RA": "23:57:19.2", "DEC": "+56:43:12"}, NGC7789),
        ({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 359.33, "CRVAL2": 56.72}, NGC7789),
        ({"DATE-OBS": "2024-07-24T00:00:00"}, None),
        ({"RA": 12.0, "DEC": 95.0}, None),
    ],
)
def test_pointing_from_header(cards, expected):
    pointing = pointing_from_header(fits.Header(cards))
    if expected is None:
        assert pointing is None
    else:
        assert pointing == pytest.approx(expected, abs=1e-3)


def test_frames_without_pointing_get_the_target_position(tmp_path):
    """SBIG frames without OBJCTRA: ASTAP solved blind (300 s timeouts)."""
    cards = {"with.fit": {"OBJCTRA": "00 10 00", "OBJCTDEC": "+10 00 00"},
             "radec.FIT": {"RA": 1.0, "DEC": 2.0},
             "none.FIT": {"DATE-OBS": "2024-07-24T00:00:00"}}
    for name, header in cards.items():
        fits.writeto(tmp_path / name, np.zeros((4, 4), dtype=np.float32), fits.Header(header))
    (tmp_path / "notes.txt").write_text("not a frame")

    assert add_pointing_hints(tmp_path, *NGC7789) == 1
    header = fits.getheader(tmp_path / "none.FIT")
    assert pointing_from_header(header) == pytest.approx(NGC7789, abs=1e-3)
    assert "ASTAP" in header.comments["OBJCTRA"]
    assert fits.getheader(tmp_path / "with.fit")["OBJCTRA"] == "00 10 00"
    assert "OBJCTRA" not in fits.getheader(tmp_path / "radec.FIT")
    assert add_pointing_hints(tmp_path, float("nan"), 56.0) == 0
    assert add_pointing_hints(tmp_path, None, None) == 0
