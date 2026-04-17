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
