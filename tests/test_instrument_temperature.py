"""Sensor temperature of a frame collection; frames without CCD-TEMP."""

from __future__ import annotations

import math

import ccdproc as ccdp
import numpy as np
from astropy.io import fits

from ost_photometry.reduce.instrument import get_instrument_info


def _collection(tmp_path, **temperature):
    for i in range(2):
        header = fits.Header({"INSTRUME": "SBIG STF-8300 CCD Camera", "IMAGETYP": "Light Frame",
                              "EXPTIME": 60.0, "READOUTM": "Raw", **temperature})
        fits.writeto(tmp_path / f"l{i}.fit", np.zeros((8, 8), dtype=np.uint16), header,
                     overwrite=True)
    return ccdp.ImageFileCollection(tmp_path)


def test_frames_without_temperature_give_nan(tmp_path, monkeypatch):
    """SBIG STF-8300 frames have no CCD-TEMP: the summary lacks the column
    (KeyError 'ccd-temp' before); the temperature is unknown."""
    from ost_photometry import terminal_output

    messages = []
    monkeypatch.setattr(terminal_output, "print_to_terminal",
                        lambda text, **kw: messages.append(text))
    *_, temperature = get_instrument_info(_collection(tmp_path), 2.0)
    assert math.isnan(temperature)
    assert any("No sensor temperature (CCD-TEMP)" in m for m in messages)


def test_temperature_from_ccd_temp_or_temperat(tmp_path):
    for name, keyword, value in (("a", "CCD-TEMP", -10.0), ("b", "TEMPERAT", -5.0)):
        (tmp_path / name).mkdir()
        *_, temperature = get_instrument_info(_collection(tmp_path / name, **{keyword: value}), 2.0)
        assert temperature == value
