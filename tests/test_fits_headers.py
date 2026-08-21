"""Tests for FITS MJD-OBS / WCS datfix helpers."""

from __future__ import annotations

import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning

from ost_photometry.fits_headers import (
    ensure_mjd_obs_in_header,
    wcs_from_header,
)


def _tan_header() -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 11.776
    header["CRVAL2"] = 85.29
    header["CRPIX1"] = 32.5
    header["CRPIX2"] = 32.5
    header["CDELT1"] = 0.00018
    header["CDELT2"] = 0.00018
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["NAXIS"] = 2
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    return header


def test_ensure_mjd_obs_from_date_obs():
    header = fits.Header()
    header["DATE-OBS"] = "2024-06-15T22:30:00.000"
    ensure_mjd_obs_in_header(header)
    expected = Time(header["DATE-OBS"], format="fits").mjd
    np.testing.assert_allclose(float(header["MJD-OBS"]), expected, rtol=0, atol=1e-9)


def test_ensure_mjd_obs_from_jd_when_date_obs_missing():
    header = fits.Header()
    header["JD"] = 2460000.5
    ensure_mjd_obs_in_header(header)
    np.testing.assert_allclose(float(header["MJD-OBS"]), 60000.0, rtol=0, atol=1e-9)


def test_ensure_mjd_obs_keeps_existing_value():
    header = fits.Header()
    header["DATE-OBS"] = "2024-06-15T22:30:00.000"
    header["MJD-OBS"] = 12345.0
    ensure_mjd_obs_in_header(header)
    assert float(header["MJD-OBS"]) == 12345.0


def test_wcs_from_header_sets_mjd_obs_and_builds_wcs():
    header = _tan_header()
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    derived = wcs_from_header(header)
    assert "MJD-OBS" in header
    assert derived.wcs.crval[0] == header["CRVAL1"]
    assert derived.wcs.crval[1] == header["CRVAL2"]


def test_wcs_from_header_does_not_emit_datfix_warning():
    header = _tan_header()
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", FITSFixedWarning)
        wcs_from_header(header)
    datfix = [
        item
        for item in recorded
        if issubclass(item.category, FITSFixedWarning)
        and "datfix" in str(item.message)
    ]
    assert datfix == []
