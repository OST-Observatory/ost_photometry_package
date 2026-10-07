"""Normalised camera / telescope identifiers and the electronic setup key.

All functions take manifest rows (mappings with the columns of
:mod:`ost_photometry.archive.manifest`) and depend on numpy only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

from ...camera_specs import (
    normalize_camera_id,
    normalize_instrument_name,
    normalize_readout_mode,
)

#: Archive / header spellings of the same instrument or telescope.
_CAMERA_SUFFIXES = re.compile(r"(\s+\d+)?\s+ccd\s+camera$", re.IGNORECASE)
_TELESCOPE_ALIASES = {
    "planewavecdk20": "CDK20",
    "ostcdk20": "CDK20",
    "cdk20": "CDK20",
    "meadelx200": "LX200",
    "lx200": "LX200",
    "skywatcher": "SkyWatcher",
}

#: Model names recognised inside longer telescope names ("OST CDK20",
#: "Planewave CDK 20"): (compact substring, telescope id).
_TELESCOPE_MODELS = (("cdk20", "CDK20"), ("lx200", "LX200"), ("skywatcher", "SkyWatcher"))


def _text(value: object) -> str:
    if value is None or value is np.ma.masked:
        return ""
    return str(value).strip()


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def camera_id(row: Mapping[str, object]) -> str:
    """Stable camera identifier, e.g. ``qhy600m``, ``qhy268``, ``sbig-st-8``.

    The FITS ``INSTRUME`` (with chip-size resolution of generic driver
    names) wins over the archive's instrument string. Unknown cameras get a
    slug of their name; frames without any instrument get ``unknown``.
    """
    header_name = normalize_instrument_name(
        _text(row.get("instrume")),
        row.get("naxis1"),
        row.get("naxis2"),
        row.get("xbinning", 1),
        row.get("ybinning", 1),
    )
    name = header_name or _text(row.get("instrument_archive"))
    if not name:
        return "unknown"
    catalog_id = normalize_camera_id(name)
    if catalog_id:
        return catalog_id
    return _slug(_CAMERA_SUFFIXES.sub("", name)) or "unknown"


def telescope_id(row: Mapping[str, object]) -> str:
    """Telescope from ``TELESCOP`` / archive, else the focal length (``f3454``)."""
    for key in ("telescop", "telescope"):
        name = _text(row.get(key))
        if name and name.upper() not in {"UK", "UNKNOWN"}:
            compact = re.sub(r"[^a-z0-9]", "", name.lower())
            if compact in _TELESCOPE_ALIASES:
                return _TELESCOPE_ALIASES[compact]
            for model, telescope in _TELESCOPE_MODELS:
                if model in compact:
                    return telescope
            return name
    focal = _float(row.get("focallen"))
    if math.isfinite(focal) and focal > 0:
        return f"f{int(round(focal))}"
    return "unknown"


def binning(row: Mapping[str, object]) -> str:
    return f"{int(_float(row.get('xbinning')) or 1)}x{int(_float(row.get('ybinning')) or 1)}"


def readout_mode(row: Mapping[str, object]) -> str:
    return normalize_readout_mode(_text(row.get("readoutm"))) or "default"


def _fmt(value: object) -> str:
    number = _float(value)
    return "-" if not math.isfinite(number) else f"{number:g}"


def sensor_temperature(row: Mapping[str, object]) -> float:
    """Set point if known, else the measured sensor temperature."""
    set_temp = _float(row.get("set_temp"))
    return set_temp if math.isfinite(set_temp) else _float(row.get("ccd_temp"))


def electronic_key(row: Mapping[str, object]) -> str:
    """Camera, binning, readout mode, gain and offset (temperature excluded).

    Temperature is clustered separately with a tolerance, see
    :func:`temperature_clusters`.
    """
    return "|".join(
        (
            camera_id(row),
            binning(row),
            readout_mode(row),
            f"g{_fmt(row.get('gain'))}",
            f"o{_fmt(row.get('offset'))}",
        )
    )


def filter_name(row: Mapping[str, object]) -> str:
    name = _text(row.get("filter"))
    return name if name and name not in {"**LEER**", "-"} else "none"


def optical_key(row: Mapping[str, object]) -> str:
    """Camera, binning, filter and telescope (no mount session)."""
    return "|".join((camera_id(row), binning(row), filter_name(row), telescope_id(row)))


def temperature_clusters(values: Sequence[float], tolerance: float = 2.0) -> list[int]:
    """Cluster index per value: sorted values split where a gap exceeds ``tolerance``.

    Unknown temperatures (``nan``) form their own cluster ``-1``.
    """
    temps = np.asarray(values, dtype=float)
    labels = np.full(temps.size, -1, dtype=int)
    finite = np.flatnonzero(np.isfinite(temps))
    if finite.size == 0:
        return labels.tolist()
    order = finite[np.argsort(temps[finite], kind="stable")]
    cluster = 0
    labels[order[0]] = 0
    for prev, cur in zip(order[:-1], order[1:], strict=True):
        if temps[cur] - temps[prev] > tolerance:
            cluster += 1
        labels[cur] = cluster
    return labels.tolist()


__all__ = [
    "binning",
    "camera_id",
    "electronic_key",
    "filter_name",
    "optical_key",
    "readout_mode",
    "sensor_temperature",
    "telescope_id",
    "temperature_clusters",
]
