"""Flux-fallback light curves get RA/Dec from the frames' WCS (for BJD_TDB)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from ost_photometry.analyze.pipeline.steps.light_curve import _series_id_coordinates


def _wcs() -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 30.0]
    w.wcs.crpix = [51.0, 51.0]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    return w


def test_ids_get_positions_from_reference_then_other_frames():
    w = _wcs()
    ref = SimpleNamespace(wcs=w, photometry=Table({"id": [1, 2], "x_fit": [50.0, 60.0], "y_fit": [50.0, 50.0]}))
    other = SimpleNamespace(wcs=w, photometry=Table({"id": [2, 3], "x_fit": [0.0, 50.0], "y_fit": [0.0, 60.0]}))
    series = SimpleNamespace(image_list=[other, ref], reference_image_index=1, wcs=w)
    ra, dec = _series_id_coordinates(series, np.array([1, 2, 3, 4]))
    assert np.isclose(ra[0], 150.0) and np.isclose(dec[0], 30.0)
    # id 2 from the reference frame (x = 60, 10 px to the west), not from the other one
    assert np.isclose(dec[1], 30.0, atol=1e-6) and ra[1] < 150.0
    # id 3 only in the other frame
    assert np.isclose(dec[2], 30.0 + 10.0 / 3600, atol=1e-6)
    assert np.isnan(ra[3])


def test_no_wcs_gives_none():
    img = SimpleNamespace(wcs=None, photometry=Table({"id": [1], "x_fit": [1.0], "y_fit": [1.0]}))
    series = SimpleNamespace(image_list=[img], reference_image_index=0, wcs=None)
    assert _series_id_coordinates(series, np.array([1])) == (None, None)
