"""
Bridge from the internal **standard** calibration table to the **legacy** layout.

Legacy code (``derive_calibration``, correlation helpers) was written around:

* A flat ``column_dict`` mapping logical names (``ra``, ``magB``, …) to actual
  column names in the table (often the same string after this adapter).
* ``ra_unit`` passed to :class:`astropy.coordinates.SkyCoord` for the RA column.

After centralizing fetch in :mod:`fetch`, all catalogs are normalized to
``ra``/``dec`` in **degrees** and ``mag_std_*`` / ``err_*``. This module only
renames/flattens for backward compatibility; it does **not** re-download data.
"""

from __future__ import annotations

import astropy.units as u
from astropy.table import Table


def standard_catalog_to_legacy(
    table: Table,
    filter_list: list[str],
) -> tuple[Table, dict[str, str], u.Unit]:
    """
    Build ``(Table, column_dict, ra_unit)`` expected by ``derive_calibration``.

    Standard input uses ``ra``/``dec`` in degrees and ``mag_std_{f}`` / ``err_std_{f}``.
    Legacy output uses plain names ``mag{f}``, ``err{f}`` with ``ra_unit`` = deg.

    Notes
    -----
    * Only filters present in ``filter_list`` are copied if the corresponding
      ``mag_std_*`` columns exist; missing bands are omitted (same as old Vizier path).
    * ``id`` is forwarded when present (e.g. AAVSO VSP).
    * RA/Dec are always degrees here; legacy ``hourangle`` paths for raw VSP/Simbad
      are replaced by degree-based standard tables upstream in :mod:`fetch`.
    """
    out = Table()
    # Positions: already ICRS degrees in the standard table
    out["ra"] = table["ra"]
    out["dec"] = table["dec"]
    # Self-referential mapping: column_dict values equal table column names
    column_dict: dict[str, str] = {"ra": "ra", "dec": "dec"}

    for f in filter_list:
        mcol, ecol = f"mag_std_{f}", f"err_std_{f}"
        if mcol in table.colnames:
            out[f"mag{f}"] = table[mcol]
            column_dict[f"mag{f}"] = f"mag{f}"
        if ecol in table.colnames:
            out[f"err{f}"] = table[ecol]
            column_dict[f"err{f}"] = f"err{f}"

    if "id" in table.colnames:
        out["id"] = table["id"]
        column_dict["id"] = "id"

    # Uniform degree-based RA for SkyCoord(…, unit=(ra_unit, u.deg))
    return out, column_dict, u.deg
