"""Parse SIMBAD galaxy angular size, coordinates, and overlay class."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


def simbad_unmasked_value(row, *names: str):
    """Return the first unmasked SIMBAD cell among ``names`` (any case)."""
    colnames = {str(column).lower(): column for column in row.colnames}
    for name in names:
        column = colnames.get(name.lower())
        if column is None:
            continue
        value = row[column]
        if value is None or isinstance(value, np.ma.core.MaskedConstant):
            continue
        if np.ma.is_masked(value):
            continue
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def simbad_galaxy_axes_arcmin(row) -> tuple[float, float, float] | None:
    """Major/minor axis in arcmin and position angle in degrees.

    Prefers TAP ``galdim_*`` columns from the SIMBAD ``dimensions`` field;
    falls back to a ``DIMENSIONS`` string ``a x b``. The angle is East of
    North; ``0`` if the PA is missing or masked.
    """
    major = simbad_unmasked_value(row, "galdim_majaxis")
    minor = simbad_unmasked_value(row, "galdim_minaxis")
    angle = simbad_unmasked_value(row, "galdim_angle")
    pa = 0.0 if angle is None else float(angle)

    if major is not None and minor is not None:
        return float(major), float(minor), pa

    dimensions = simbad_unmasked_value(row, "DIMENSIONS")
    if dimensions is None:
        return None
    try:
        major_s, minor_s = [part.strip() for part in str(dimensions).split("x")]
        return float(major_s), float(minor_s), pa
    except ValueError:
        return None


def skycoord_from_simbad(ra, dec) -> SkyCoord:
    """
    SkyCoord from a SIMBAD RA/Dec cell.

    TAP ``basic.ra`` / ``basic.dec`` are degrees. Legacy script output used
    sexagesimal hourangle strings. Interpreting TAP degrees as hourangle
    shifts objects by tens of degrees (NGC 7789 would miss the frame).
    """
    ra_u = getattr(ra, "unit", None)
    dec_u = getattr(dec, "unit", None)
    if (ra_u is not None and ra_u != u.dimensionless_unscaled) or (
        dec_u is not None and dec_u != u.dimensionless_unscaled
    ):
        return SkyCoord(ra=ra, dec=dec)

    if isinstance(ra, bytes):
        ra = ra.decode()
    if isinstance(dec, bytes):
        dec = dec.decode()

    if isinstance(ra, str) or isinstance(dec, str):
        ra_s, dec_s = str(ra).strip(), str(dec).strip()
        if any(sep in ra_s for sep in (":", " ")) or any(
            sep in dec_s for sep in (":", " ")
        ):
            return SkyCoord(ra=ra_s, dec=dec_s, unit=("hourangle", "deg"))
        ra = float(ra_s)
        dec = float(dec_s)

    return SkyCoord(
        ra=float(np.asarray(ra, dtype=float)) * u.deg,
        dec=float(np.asarray(dec, dtype=float)) * u.deg,
    )


def simbad_overlay_kind(otype) -> str:
    """Map a SIMBAD ``otype`` code or label to an overlay class."""
    if otype is None:
        return "other"
    if isinstance(otype, bytes):
        otype = otype.decode()
    text = str(otype).strip()
    if not text:
        return "other"
    cluster_tokens = ("OpC", "GlC", "Cl*", "As*", "Cluster", "MCl", "SGr")
    if any(token in text for token in cluster_tokens):
        return "cluster"
    galaxy_exact = {
        "G",
        "GiC",
        "GiG",
        "BiC",
        "SBG",
        "AGN",
        "QSO",
        "Sy1",
        "Sy2",
        "Galaxy",
        "Seyfert1",
        "Seyfert2",
        "AGN_Candidate",
    }
    if text in galaxy_exact or "Galaxy" in text:
        return "galaxy"
    if "Nebula" in text or text in {"HII", "PN", "SNR", "RNe"}:
        return "nebula"
    if text == "*" or text.endswith("*") or "Star" in text:
        return "star"
    return "other"
