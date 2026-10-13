"""Telescope catalog: aperture and focal length of the OST telescopes.

``ost_photometry/data/telescopes.json`` holds the values of the OST wiki
(CDK20 and the mobile telescopes). Images are matched by the FITS
``TELESCOP`` name (catalog aliases, also inside longer names such as
"OST CDK20"), else by ``FOCALLEN`` within 2 %.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from importlib.resources import files

_CATALOG_NAME = "telescopes.json"
_FOCAL_LENGTH_TOLERANCE = 0.02


@lru_cache(maxsize=1)
def telescope_catalog() -> dict:
    """Load the runtime telescope catalog shipped with the package."""
    resource = files("ost_photometry.data").joinpath(_CATALOG_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalize_telescope_id(name: str | None) -> str | None:
    """Catalog id for a ``TELESCOP`` string, or ``None`` if unknown.

    Exact aliases win; otherwise the longest alias contained in the name
    (so "Planewave CDK 20" is the CDK20 and "Celestron RASA 11" not a C11).
    """
    if not name:
        return None
    compact = _compact(name)
    if not compact:
        return None
    telescopes = telescope_catalog()["telescopes"]
    pairs = [
        (alias, tid)
        for tid, rec in telescopes.items()
        for alias in rec.get("aliases", [])
    ]
    for alias, tid in pairs:
        if compact == alias:
            return tid
    for alias, tid in sorted(pairs, key=lambda p: -len(p[0])):
        if alias in compact:
            return tid
    return None


def telescope_from_focal_length(focal_length_mm: float | None) -> str | None:
    """Catalog id whose focal length is within 2 % of ``focal_length_mm``."""
    try:
        focal = float(focal_length_mm)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(focal) or focal <= 0:
        return None
    best: tuple[float, str] | None = None
    for tid, rec in telescope_catalog()["telescopes"].items():
        ref = rec.get("focal_length_mm")
        if not ref:
            continue
        rel = abs(focal - float(ref)) / float(ref)
        if rel <= _FOCAL_LENGTH_TOLERANCE and (best is None or rel < best[0]):
            best = (rel, tid)
    return None if best is None else best[1]


def resolve_telescope(
    telescope_name: str | None = None,
    focal_length_mm: float | None = None,
) -> str | None:
    """Catalog id from ``TELESCOP``, else from ``FOCALLEN``."""
    return normalize_telescope_id(telescope_name) or telescope_from_focal_length(
        focal_length_mm
    )


def telescope_record(telescope_id: str) -> dict | None:
    return telescope_catalog()["telescopes"].get(telescope_id)


def aperture_diameter_m(telescope_id: str | None) -> float | None:
    """Clear aperture in metres, or ``None`` if unknown."""
    if telescope_id is None:
        return None
    rec = telescope_record(telescope_id)
    if rec is None or not rec.get("aperture_mm"):
        return None
    return float(rec["aperture_mm"]) / 1000.0


__all__ = [
    "aperture_diameter_m",
    "normalize_telescope_id",
    "resolve_telescope",
    "telescope_catalog",
    "telescope_from_focal_length",
    "telescope_record",
]
