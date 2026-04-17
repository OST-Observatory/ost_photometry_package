"""
Calibration sources: shared download, normalization, and matching for photometry.

This package is the **single implementation** for fetching external calibration
catalogs used by:

* **Legacy pipeline** — :mod:`ost_photometry.analyze.calibration_data` calls
  :func:`fetch_standard_calibration_catalog`, then :func:`standard_catalog_to_legacy`
  to produce the ``(Table, column_dict, ra_unit)`` contract expected by
  :func:`~ost_photometry.analyze.calibration_data.derive_calibration`.

* **Differential pipeline** — :class:`~ost_photometry.analyze.differential_photometry.PhotometryCalibrator`
  keeps the catalog in **standard schema** and cross-matches via
  :func:`crossmatch_standard_catalog`.

**Standard schema** (internal, degrees on the sky):

* ``ra``, ``dec`` — ICRS, decimal degrees.
* ``mag_std_{f}``, ``err_std_{f}`` — standard magnitudes per filter letter ``f``
  (Johnson-Cousins letters and/or Sloan ``g``, ``r``, ``i`` as lowercase).
* Optional metadata, e.g. ``id``, ``apass_id``.

**Lupton (2005)** Sloan → Johnson R/I lives in :mod:`transforms` and is invoked
from :mod:`fetch` where appropriate (APASS and optional heuristic for other Vizier tables).

Public re-exports below; see each submodule for details.
"""

from .crossmatch import crossmatch_standard_catalog
from .fetch import fetch_standard_calibration_catalog, vizier_result_to_standard_table
from .legacy_adapter import standard_catalog_to_legacy
from .transforms import (
    add_johnson_ri_from_sloan,
    add_johnson_ri_to_standard_table,
    johnson_ri_from_sloan_ri,
)
from .vizier_query import get_vizier_catalog

__all__ = [
    "add_johnson_ri_from_sloan",
    "add_johnson_ri_to_standard_table",
    "crossmatch_standard_catalog",
    "fetch_standard_calibration_catalog",
    "get_vizier_catalog",
    "johnson_ri_from_sloan_ri",
    "standard_catalog_to_legacy",
    "vizier_result_to_standard_table",
]
