"""FITS time/WCS header hygiene (MJD-OBS, datfix warnings)."""

from __future__ import annotations

import math
import warnings

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

_DATFIX_FILTER_INSTALLED = False


def ignore_wcs_datfix_warnings() -> None:
    """Ignore informational WCS ``datfix`` messages (DATE-OBS / MJD-OBS)."""
    global _DATFIX_FILTER_INSTALLED
    if _DATFIX_FILTER_INSTALLED:
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*'datfix' made the change.*",
        category=FITSFixedWarning,
    )
    _DATFIX_FILTER_INSTALLED = True


def ensure_mjd_obs_in_header(header: fits.Header) -> fits.Header:
    """Set ``MJD-OBS`` from ``DATE-OBS`` or ``JD`` when it is missing.

    Astropy WCS otherwise emits ``FITSFixedWarning`` (``datfix``) for the
    same conversion. The header is updated in place and returned.
    """
    if _finite_mjd_obs(header) is not None:
        return header
    mjd = _mjd_obs_from_header(header)
    if mjd is not None:
        header["MJD-OBS"] = (float(mjd), "MJD of observation start")
    return header


def wcs_from_header(header: fits.Header) -> WCS:
    """Build a WCS after filling ``MJD-OBS`` so ``datfix`` has nothing to do."""
    ensure_mjd_obs_in_header(header)
    return WCS(header)


def _finite_mjd_obs(header: fits.Header) -> float | None:
    value = header.get("MJD-OBS")
    if value is None:
        return None
    try:
        mjd = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mjd):
        return None
    return mjd


def _mjd_obs_from_header(header: fits.Header) -> float | None:
    date_obs = header.get("DATE-OBS")
    if date_obs:
        for kwargs in ({"format": "fits"}, {}):
            try:
                return float(Time(date_obs, **kwargs).mjd)
            except Exception:  # noqa: BLE001 — DATE-OBS can be instrument-specific
                continue
    jd = header.get("JD")
    if jd is None:
        return None
    try:
        return float(jd) - 2400000.5
    except (TypeError, ValueError):
        return None


ignore_wcs_datfix_warnings()
