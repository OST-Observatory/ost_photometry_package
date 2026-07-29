"""Schema-aware I/O for epoch-native photometry tables (ECSV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from . import schema


def write_epoch_native_magnitudes(
    observation: Any,
    tbl: Table,
    *,
    object_id: int | None = None,
    rts: str = "",
    photometry_extraction_method: str = "",
    validate_schema: bool = False,
    file_stem: str = "calibrated_magnitudes",
) -> Path:
    """
    Write an epoch-native magnitude table as ``ascii.ecsv`` with schema metadata.

    Used from calibration steps and from cluster-field post-processing, not only
    after post-process filters.

    Column dtypes and units are preserved where present.

    Parameters
    ----------
    file_stem
        Base name before ``_img_`` / ``rts`` / ``.ecsv``. Use ``extracted_magnitudes``
        for instrumental (uncalibrated) epoch-native dumps.
    """
    output_dir = list(observation.image_series_dict.values())[0].out_path
    from ... import checks

    checks.check_output_directories(
        output_dir,
        output_dir / "tables",
    )

    if object_id is not None:
        object_id_suffix = f"_img_{object_id}"
    else:
        object_id_suffix = ""
    if photometry_extraction_method:
        photometry_extraction_method = f"_{photometry_extraction_method}"

    filename = (
        f"{file_stem}{photometry_extraction_method}{object_id_suffix}{rts}.ecsv"
    )
    out_path = output_dir / "tables" / filename

    if validate_schema:
        schema.validate_epoch_native_table(tbl, require_mag_columns=False)

    out_tbl = tbl.copy()
    meta = dict(out_tbl.meta) if out_tbl.meta else {}
    meta["photometry_schema"] = schema.PHOTOMETRY_TABLE_SCHEMA_ID
    out_tbl.meta = meta

    for column_name in out_tbl.colnames:
        if column_name in ("ra (deg)", "dec (deg)"):
            continue
        col = out_tbl[column_name]
        if not np.issubdtype(col.dtype, np.number):
            continue
        col.info.format = "{:12.3f}"

    formats = schema.ascii_write_formats_for_columns(out_tbl.colnames)

    out_tbl.write(
        str(out_path),
        format="ascii.ecsv",
        overwrite=True,
        formats=formats,
    )
    return out_path


def read_epoch_native_magnitudes(path: str | Path) -> Table:
    """Read a table written by :func:`write_epoch_native_magnitudes`."""
    return Table.read(str(path), format="ascii.ecsv")
