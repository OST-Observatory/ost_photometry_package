"""Epoch-native photometry table schema (post-processing pipeline)."""

from __future__ import annotations

from astropy.table import Table

REQUIRED_EPOCH_NATIVE_COLUMNS: tuple[str, ...] = (
    "id",
    "x",
    "y",
    "ra",
    "dec",
    "epoch_id",
)

PHOTOMETRY_TABLE_SCHEMA_ID: str = "ost_photometry.epoch_native.v1"

# Only keys that exist on the table should be passed to Table.write(formats=...).
_ASCII_COLUMN_FORMATS: dict[str, str] = {
    "i": "{:5.0f}",
    "id": "{:5.0f}",
    "x": "{:12.2f}",
    "y": "{:12.2f}",
}


def ascii_write_formats_for_columns(colnames: list[str] | tuple[str, ...] | set[str]) -> dict[str, str]:
    """
    Format dict for ``Table.write(..., formats=...)`` limited to existing columns.

    Avoids Astropy warnings when a formats key (e.g. legacy ``i``) is absent
    because the table uses epoch-native ``id`` (or the reverse).
    """
    names = set(colnames)
    return {key: fmt for key, fmt in _ASCII_COLUMN_FORMATS.items() if key in names}


def validate_epoch_native_table(tbl: Table, *, require_mag_columns: bool = False) -> None:
    """Raise ValueError if required epoch-native columns are missing."""
    missing = [c for c in REQUIRED_EPOCH_NATIVE_COLUMNS if c not in tbl.colnames]
    if missing:
        raise ValueError(
            f"Table missing epoch-native columns {missing!r}; "
            f"expected at least {list(REQUIRED_EPOCH_NATIVE_COLUMNS)!r}."
        )
    if require_mag_columns and not any(
        n.startswith("mag_cal_") or n.startswith("mag_inst_") for n in tbl.colnames
    ):
        raise ValueError(
            "Table has no ``mag_cal_*`` or ``mag_inst_*`` magnitude columns."
        )
