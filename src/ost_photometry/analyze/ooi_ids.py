"""Resolve object-of-interest photometry IDs after correlation.

Identify-on-sky writes a **table row** into ``id_in_image_series[filter_]``.
After correlation that row is not always the join key: dense intersection
tables are sliced so row ``k`` is object ``k``, but sparse tracks keep native
detections and set photometry ``id`` to the track id.

``bind_ooi_ids_from_photometry`` maps that row onto photometry ``id``
(the value light curves and calibration join on). ``correlated_id`` is the
cross-filter join key once tables share ids.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    """Copy ``id_in_image_series[filter_]`` onto ``correlated_id``.

    Use :func:`bind_ooi_ids_from_photometry` after correlation so the copied
    value is the photometry ``id`` (track), not a leftover native row index.
    """
    for obj in objects:
        id_map = getattr(obj, "id_in_image_series", None) or {}
        raw = id_map.get(filter_)
        obj.correlated_id = None if raw is None else int(raw)


def bind_ooi_ids_from_photometry(
    objects: list[Any],
    filter_: str,
    photometry: Any,
    *,
    set_correlated_id: bool = True,
) -> None:
    """Map identify() table rows to photometry ``id`` after correlation apply.

    ``identify_object_of_interest_in_dataset`` stores the current table row.
    Sparse apply does not reorder rows, so that row is not the track id used
    by :meth:`ImageSeries.get_flux_array` and epoch tables. When ``id`` is
    present, both ``id_in_image_series[filter_]`` and (optionally)
    ``correlated_id`` become ``photometry['id'][row]``.
    """
    n = 0 if photometry is None else int(len(photometry))
    id_col = None
    if photometry is not None and n and "id" in getattr(photometry, "colnames", ()):
        id_col = np.asarray(photometry["id"], dtype=np.int64)

    for obj in objects:
        id_map = getattr(obj, "id_in_image_series", None)
        if id_map is None:
            obj.id_in_image_series = {}
            id_map = obj.id_in_image_series
        raw = id_map.get(filter_)
        if raw is None:
            if set_correlated_id:
                obj.correlated_id = None
            continue
        row = int(raw)
        if row < 0 or row >= n:
            id_map[filter_] = None
            if set_correlated_id:
                obj.correlated_id = None
            continue
        bound = int(id_col[row]) if id_col is not None else row
        id_map[filter_] = bound
        if set_correlated_id:
            obj.correlated_id = bound
