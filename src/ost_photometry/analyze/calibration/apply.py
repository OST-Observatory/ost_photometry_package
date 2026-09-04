"""Apply calibration results to epoch tables."""

from __future__ import annotations

from astropy.table import Table

from .photometer import DifferentialPhotometer
from .result import CalibrationResult


def apply_calibration_to_table(
    data: Table,
    calibration: CalibrationResult,
    filters: list[str],
    photometer: DifferentialPhotometer | None = None,
    *,
    mag_col_prefix: str = "mag_",
    std_col_prefix: str = "mag_std_",
    output_prefix: str = "mag_cal_",
    err_col_prefix: str = "err_",
    output_err_prefix: str = "err_cal_",
    fallback_airmass_col: str = "airmass",
) -> Table:
    """Apply T/ZP from ``calibration`` to ``data`` using DifferentialPhotometer.apply_transform_to_table."""
    phot = photometer or DifferentialPhotometer()
    phot.calibrations[calibration.identifier] = calibration
    return phot.apply_transform_to_table(
        data,
        calibration,
        filters=filters,
        mag_col_prefix=mag_col_prefix,
        std_col_prefix=std_col_prefix,
        output_prefix=output_prefix,
        err_col_prefix=err_col_prefix,
        output_err_prefix=output_err_prefix,
        fallback_airmass_col=fallback_airmass_col,
        inplace=False,
    )


def apply_calibration_epochs(
    epochs: dict[str, Table],
    results: dict[str, CalibrationResult],
    filters: list[str],
    photometer: DifferentialPhotometer | None = None,
    *,
    output_prefix: str = "mag_cal_",
) -> Table:
    """Apply per-epoch calibration and vstack rows with ``epoch_id`` column."""
    from astropy.table import vstack

    phot = photometer or DifferentialPhotometer()
    tables = []
    for epoch_id, data in epochs.items():
        cal = results.get(epoch_id)
        if cal is None:
            continue
        out = apply_calibration_to_table(
            data,
            cal,
            filters,
            photometer=phot,
            output_prefix=output_prefix,
        )
        out["epoch_id"] = epoch_id
        tables.append(out)
    return vstack(tables) if tables else Table()


__all__ = ["apply_calibration_epochs", "apply_calibration_to_table"]
