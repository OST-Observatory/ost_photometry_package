"""Resolve object-of-interest photometry IDs after correlation.

After intra- and (when used) inter-filter correlation, photometry table ``id``
is the aligned row index. ``ObjectOfInterest.correlated_id`` is that same
index. Per-filter ``id_in_image_series`` is only the pre-alignment row map
used while tables are still filter-local.
"""

from __future__ import annotations

from typing import Any


def ooi_photometry_id(
        obj: Any,
        filter_: str | None = None,
        reference_image_series_id: int | None = None,
    ) -> int | None:
    """Return the correlated photometry ``id``, or a pre-alignment row index.

    Prefers ``correlated_id`` once tables are aligned. Otherwise uses
    ``id_in_image_series[filter_]``, or the entry at ``reference_image_series_id``
    (default 0) when no filter is given.
    """
    correlated = getattr(obj, "correlated_id", None)
    if correlated is not None:
        return int(correlated)

    id_map = getattr(obj, "id_in_image_series", None) or {}
    if not id_map:
        return None

    if filter_ is not None:
        value = id_map.get(filter_)
    else:
        keys = list(id_map.keys())
        index = 0 if reference_image_series_id is None else int(reference_image_series_id)
        if index < 0 or index >= len(keys):
            return None
        value = id_map[keys[index]]

    if value is None:
        return None
    return int(value)


def ooi_photometry_ids(
        objects: list[Any],
        filter_: str | None = None,
        reference_image_series_id: int | None = None,
    ) -> list[int]:
    """Collect finite photometry IDs for a list of objects of interest."""
    ids: list[int] = []
    for obj in objects:
        value = ooi_photometry_id(
            obj,
            filter_=filter_,
            reference_image_series_id=reference_image_series_id,
        )
        if value is not None:
            ids.append(value)
    return ids


def set_ooi_correlated_ids_from_filter(objects: list[Any], filter_: str) -> None:
    """Copy aligned ``id_in_image_series[filter_]`` onto ``correlated_id``.

    Call after photometry rows are the same physical objects across the
    tables used downstream (inter-filter correlation, or intra for a
    single-filter series).
    """
    for obj in objects:
        id_map = getattr(obj, "id_in_image_series", None) or {}
        raw = id_map.get(filter_)
        obj.correlated_id = None if raw is None else int(raw)
