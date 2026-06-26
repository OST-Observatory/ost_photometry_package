"""Pipeline helpers (decoupled from Observation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import terminal_output

if TYPE_CHECKING:
    from .context import AnalysisContext


def get_ids_object_of_interest(
    context: AnalysisContext,
    filter_: str | None = None,
    reference_image_series_id: int | None = None,
) -> list[int]:
    """Resolve OOI indices from context without requiring Observation methods."""
    if filter_ is None and reference_image_series_id is None:
        terminal_output.print_to_terminal(
            "Neither a filter nor an image series ID was provided to "
            "compile the IDs for the objects of interest. The image series ID "
            "is assumed to be 0.",
            style_name="WARNING",
        )
        reference_image_series_id = 0

    ids: list[int] = []
    for object_ in context.objects_of_interest:
        id_map = object_.id_in_image_series
        if not id_map:
            continue
        if filter_ is not None:
            ids.append(id_map[filter_])
        else:
            keys = list(id_map.keys())
            ids.append(id_map[keys[reference_image_series_id]])
    return ids
