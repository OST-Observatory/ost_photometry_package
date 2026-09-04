"""
Calibration sources: shared download, normalization, and matching for photometry.

This package is the **single implementation** for fetching external calibration
catalogs used by:

* **Protect-calibrators / correlation** — :func:`fetch_standard_calibration_catalog`
  (standard schema) via :mod:`ost_photometry.analyze.correlate.protection`.
* **Epoch-native calibration** — :class:`~ost_photometry.analyze.calibration.PhotometryCalibrator`
  (via :class:`~ost_photometry.analyze.calibration.CalibrationEngine`) keeps the
  catalog in **standard schema** and cross-matches via :func:`crossmatch_standard_catalog`.

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
from .flags import flag_comparison_stars, mark_used_calibrators
from .known_variables import drop_catalog_rows_near_known_variables
from .transforms import (
    add_johnson_ri_from_sloan,
    add_johnson_ri_to_standard_table,
    johnson_bv_from_sloan_gr,
    johnson_ri_from_sloan_ri,
    johnson_u_from_sloan_ug_and_b,
    johnson_ubvri_from_sloan_arrays,
)
from .vizier_query import get_vizier_catalog

__all__ = [
    "add_johnson_ri_from_sloan",
    "add_johnson_ri_to_standard_table",
    "crossmatch_standard_catalog",
    "drop_catalog_rows_near_known_variables",
    "fetch_standard_calibration_catalog",
    "flag_comparison_stars",
    "mark_used_calibrators",
    "get_vizier_catalog",
    "johnson_bv_from_sloan_gr",
    "johnson_ri_from_sloan_ri",
    "johnson_u_from_sloan_ug_and_b",
    "johnson_ubvri_from_sloan_arrays",
    "vizier_result_to_standard_table",
]
