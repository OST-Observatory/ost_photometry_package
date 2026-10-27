"""Header keywords added to reduced frames and stacks (update_header_information)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData

from ost_photometry.calibration_parameters import filter_system
from ost_photometry.reduce.utilities import update_header_information


@pytest.mark.parametrize(
    ("name", "system"),
    [
        ("V", "bessell"), ("R", "bessell"), ("r", "sdss"), ("g`", "sdss"), ("i'", "sdss"),
        ("z-s`", "sdss"), ("H alpha", "narrow_band"), ("Halpha`", "narrow_band"),
        ("O III", "narrow_band"), ("S II", "narrow_band"), ("Clear", "clear"), ("L", "clear"),
        ("Spectrum", None), ("", None),
    ],
)
def test_filter_system(name, system):
    assert filter_system(name) == system


@pytest.fixture
def messages(monkeypatch):
    """Terminal messages of update_header_information."""
    from ost_photometry.reduce import utilities

    seen: list[str] = []
    monkeypatch.setattr(utilities.terminal_output, "print_to_terminal",
                        lambda text, *a, **k: seen.append(str(text)))
    return seen


def test_start_and_mid_time_from_date_obs(messages):
    """Archive frames carry DATE-OBS but no JD: no warning, MJD_MID set."""
    header = fits.Header({"DATE-OBS": "2026-03-01T23:32:33", "EXPTIME": 300.0, "FILTER": "H alpha"})
    ccd = CCDData(np.zeros((4, 4), dtype=np.float32), unit="electron/s", meta=header)
    update_header_information(ccd)
    assert not any("could not" in m for m in messages)
    assert ccd.meta["MJD_STA"] == pytest.approx(61100.981, abs=1e-3)
    assert ccd.meta["MJD_MID"] - ccd.meta["MJD_STA"] == pytest.approx(150 / 86400)
    assert ccd.meta["FILTER-S"] == "narrow_band"


def test_no_time_at_all_warns(messages):
    ccd = CCDData(np.zeros((4, 4), dtype=np.float32), unit="electron/s",
                  meta=fits.Header({"EXPTIME": 10.0, "FILTER": "Spectrum"}))
    update_header_information(ccd)
    assert any("MJD could not be added" in m for m in messages)
    assert "FILTER-S" not in ccd.meta
