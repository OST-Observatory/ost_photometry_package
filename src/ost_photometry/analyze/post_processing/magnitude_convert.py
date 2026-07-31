"""Photometric system conversion on observation tables (not cluster-field specific)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table

from . import convert as convert_mod
from .magnitude_systems import (
    annotate_table_magnitude_meta,
    infer_filter_set,
    log_magnitude_output,
    resolve_catalog_magnitude_system,
    resolve_effective_output,
    table_magnitude_system,
)

if TYPE_CHECKING:
    from .. import analyze


def apply_magnitude_system_convert_on_observation(
    observation: "analyze.Observation",
    *,
    target_filter_system: str | None = None,
    output_filter_set: str = "auto",
    output_magnitude_system: str = "auto",
    convert_magnitudes: bool = True,
    distribution_samples: int = 1000,
    calibration_source: str | None = None,
    input_table: Table | None = None,
) -> None:
    """
    Convert magnitudes in ``observation.table_magnitudes`` toward the requested output.

    Always annotates table meta with the effective filter set / magnitude system.
    When ``convert_magnitudes`` is False, only meta/labels are updated (no numerical convert).
    """
    tbl = input_table if input_table is not None else observation.table_magnitudes
    if tbl is None or len(tbl) == 0:
        return

    filters = []
    for name in tbl.colnames:
        if name.startswith("mag_cal_"):
            filters.append(name[len("mag_cal_") :])
    if not filters:
        for name in tbl.colnames:
            if name.startswith("mag_inst_"):
                filters.append(name[len("mag_inst_") :])

    cal_fs = infer_filter_set(filters)
    cat_ms = resolve_catalog_magnitude_system(calibration_source)
    if cat_ms == "unknown":
        cat_ms = "ab" if cal_fs == "sdss" else "vega" if cal_fs == "bessell" else "unknown"

    effective = resolve_effective_output(
        output_filter_set=output_filter_set,  # type: ignore[arg-type]
        output_magnitude_system=output_magnitude_system,  # type: ignore[arg-type]
        calibrated_filter_set=cal_fs,
        catalog_magnitude_system=cat_ms,  # type: ignore[arg-type]
        convert_magnitudes=convert_magnitudes,
    )
    log_magnitude_output(effective, calibration_source)

    if convert_magnitudes and (
        effective.needs_convert
        or target_filter_system
        or output_filter_set != "auto"
        or output_magnitude_system != "auto"
    ):
        work = convert_mod.convert_magnitudes_to_other_system(
            tbl,
            target_filter_system=target_filter_system,
            distribution_samples=distribution_samples,
            output_filter_set=output_filter_set,
            output_magnitude_system=output_magnitude_system,
            calibration_source=calibration_source,
            source_magnitude_system=cat_ms,
            source_filter_set=cal_fs,
        )
    else:
        work = tbl.copy() if tbl is observation.table_magnitudes else tbl
        annotate_table_magnitude_meta(
            work,
            filter_set=effective.filter_set,
            magnitude_system=effective.magnitude_system,
            catalog_magnitude_system=cat_ms,
            calibration_source=calibration_source,
            conversion_note="identity",
        )

    observation.table_magnitudes = work


__all__ = [
    "apply_magnitude_system_convert_on_observation",
    "table_magnitude_system",
]
