"""Compatibility re-export; implementation lives in ``post_processing.simbad_annotate``."""

from ..post_processing.simbad_annotate import (
    annotate_reference_image_with_simbad,
    mark_simbad_objects_on_image,
    query_simbad_objects,
)

__all__ = [
    "annotate_reference_image_with_simbad",
    "mark_simbad_objects_on_image",
    "query_simbad_objects",
]
