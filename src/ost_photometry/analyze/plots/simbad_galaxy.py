"""Parse SIMBAD galaxy angular size and position angle from a table row."""

from __future__ import annotations

import numpy as np


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
