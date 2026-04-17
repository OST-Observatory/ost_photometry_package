"""Photometric system conversion on observation tables (not cluster-field specific)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table

from . import convert as convert_mod

if TYPE_CHECKING:
    from .. import analyze


def apply_magnitude_system_convert_on_observation(
    observation: "analyze.Observation",
    *,
    target_filter_system: str = "SDSS",
    distribution_samples: int = 1000,
    input_table: Table | None = None,
) -> None:
    """
    Convert magnitudes in ``observation.table_magnitudes`` to another system in place.

    The full table is passed to ``convert_magnitudes_to_other_system``, including every
    row for multi-epoch tables (all ``epoch_id`` values).
    """
    tbl = input_table if input_table is not None else observation.table_magnitudes
    if tbl is None or len(tbl) == 0:
        return

    work = convert_mod.convert_magnitudes_to_other_system(
        tbl,
        target_filter_system,
        distribution_samples=distribution_samples,
    )

    observation.table_magnitudes = work
