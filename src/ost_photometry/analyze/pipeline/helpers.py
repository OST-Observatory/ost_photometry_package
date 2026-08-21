"""Pipeline helpers (decoupled from Observation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import terminal_output
from ..ooi_ids import ooi_photometry_ids

if TYPE_CHECKING:
    from .context import AnalysisContext


def get_ids_object_of_interest(
    context: AnalysisContext,
    filter_: str | None = None,
    reference_image_series_id: int | None = None,
) -> list[int]:
    """Resolve OOI photometry IDs from context (correlated ``id`` when set)."""
    if filter_ is None and reference_image_series_id is None:
        terminal_output.print_to_terminal(
            "Neither a filter nor an image series ID was provided to "
            "compile the IDs for the objects of interest. The image series ID "
            "is assumed to be 0.",
            style_name="WARNING",
        )
        reference_image_series_id = 0

    return ooi_photometry_ids(
        context.objects_of_interest,
        filter_=filter_,
        reference_image_series_id=reference_image_series_id,
    )
