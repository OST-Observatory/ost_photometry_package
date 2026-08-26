"""FITS time/WCS header hygiene (MJD-OBS, datfix warnings) and cosmic flags."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

_DATFIX_FILTER_INSTALLED = False

CosmicHandling = Literal["interpolated", "masked"]
CosmicRayRemovalMode = Literal["auto", "always", "never"]

# Canonical + legacy keys that mean cosmics were already identified.
_COSMIC_IDENTIFIED_KEYS = ("CRIDENT", "cosmics_rm", "cosmics_msk", "cosmic_mas")


def _header_truthy(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in {"t", "true", "yes", "1"}:
        return True
    return False


def cosmics_identified(header: Mapping[str, Any] | fits.Header) -> bool:
    """Return True if cosmics were identified (masked and/or interpolated).

    Recognizes ``CRIDENT``, ``cosmics_rm``, ``cosmics_msk``, and legacy
    ``cosmic_mas``.
    """
    for key in _COSMIC_IDENTIFIED_KEYS:
        if key in header and _header_truthy(header.get(key)):
            return True
    return False


def mark_cosmics_identified(
    header: MutableMapping[str, Any] | fits.Header,
    *,
    handling: CosmicHandling,
) -> None:
    """Set cosmic-ray identification keywords on a FITS header / CCD meta.

    Always sets ``CRIDENT=True``. Additionally sets ``cosmics_rm`` when
    cosmics were interpolated out, or ``cosmics_msk`` when only masked.
    """
    header["CRIDENT"] = True
    if handling == "interpolated":
        header["cosmics_rm"] = True
    elif handling == "masked":
        header["cosmics_msk"] = True
    else:
        raise ValueError(f"Unknown cosmic handling {handling!r}")


def normalize_cosmic_ray_removal(
    value: bool | CosmicRayRemovalMode | str,
) -> CosmicRayRemovalMode:
    """Map bool aliases and strings to ``auto`` / ``always`` / ``never``."""
    if value is True or value == "always":
        return "always"
    if value is False or value == "never":
        return "never"
    if value == "auto":
        return "auto"
    raise ValueError(
        f"cosmic_ray_removal must be 'auto', 'always', 'never', or bool; got {value!r}"
    )


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
