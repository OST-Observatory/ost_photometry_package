"""Time order of the frames to align when only some have a JD."""

from __future__ import annotations

from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.io import fits
from astropy.table import MaskedColumn, Table

from ost_photometry.reduce.registration.align import observation_jds, sort_by_observation_time


def test_observation_jds_from_jd_date_obs_or_mjd_obs():
    table = Table({
        "jd": MaskedColumn([2460515.6, 0.0, 0.0, 0.0, 0.0], mask=[False, True, True, True, True],
                           dtype=object),
        "date-obs": ["2024-07-24T12:00:00", "2024-07-24T00:00:00", "", "bad", ""],
        "mjd-obs": MaskedColumn([0.0, 0.0, 60515.25, 0.0, 0.0], mask=[True, True, False, True, True]),
    })
    jds = observation_jds(table)
    assert jds[0] == 2460515.6  # JD wins over DATE-OBS
    assert np.isclose(jds[1], 2460515.5)  # from DATE-OBS
    assert np.isclose(jds[2], 2460515.75)  # from MJD-OBS
    assert np.isinf(jds[3]) and np.isinf(jds[4])  # unparsable / nothing: last


def test_frames_with_and_without_jd_sort_by_time(tmp_path):
    """QHY frames with JD and SBIG frames with only DATE-OBS in one target:
    sorting the 'jd' column failed (None / str against float)."""
    headers = {
        "a.fit": {"IMAGETYP": "Light Frame", "JD": 2460515.70},
        "b.fit": {"IMAGETYP": "Light Frame", "DATE-OBS": "2024-07-24T00:00:00"},  # JD 2460515.5
        "c.fit": {"IMAGETYP": "Light Frame", "JD": "2460515.60"},  # JD as text
        "d.fit": {"IMAGETYP": "Light Frame"},
    }
    for name, cards in headers.items():
        fits.writeto(tmp_path / name, np.zeros((4, 4), dtype=np.uint16), fits.Header(cards))
    collection = ccdp.ImageFileCollection(tmp_path)
    sort_by_observation_time(collection)
    assert list(collection.files) == ["b.fit", "c.fit", "a.fit", "d.fit"]
    filtered = collection.filter(imagetyp="Light Frame")
    assert [Path(f).name for f in filtered.files] == ["b.fit", "c.fit", "a.fit", "d.fit"]
