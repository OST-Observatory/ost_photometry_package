"""Sparse track helpers: apply correlation indexes without intersection slicing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from ..image import AnalysisImage
    from ..models import ImageSeries


def effective_miss_limit(
    n_images: int,
    n_allowed_non_detections_object: int,
    min_detection_fraction: float | None,
) -> int:
    """Miss count at which a track is dropped.

    When ``min_detection_fraction`` is set, the limit is at least
    ``int((1 - fraction) * n_images)`` so a small integer ``n_allowed`` does
    not force near-completeness on long series.
    """
    limit = max(int(n_allowed_non_detections_object), 0)
    if min_detection_fraction is not None:
        frac = float(min_detection_fraction)
        frac = min(max(frac, 0.0), 1.0)
        frac_limit = int((1.0 - frac) * int(n_images))
        limit = max(limit, frac_limit)
    return limit


def pick_auto_reference_image(image_series: ImageSeries) -> int:
    """Index of the frame with the most finite detections, then smallest FWHM."""
    best_i = 0
    best_key: tuple[int, float] = (-1, float("inf"))
    for i, image in enumerate(image_series.image_list):
        phot = getattr(image, "photometry", None)
        if phot is None or len(phot) == 0:
            n_det = 0
        elif "flux_fit" in phot.colnames:
            n_det = int(np.sum(np.isfinite(np.asarray(phot["flux_fit"], dtype=float))))
        else:
            n_det = int(len(phot))
        fwhm = float(getattr(image, "fwhm", np.inf) or np.inf)
        key = (n_det, -fwhm)
        if key > best_key:
            best_key = key
            best_i = i
    return best_i


def coerce_reference_image_index(
    value: int | str | None,
    n_images: int,
    *,
    default: int = 0,
) -> int:
    """Map ``'auto'`` / ``None`` to ``default``; clamp an int into ``[0, n)``."""
    if value is None or value == "auto":
        idx = int(default)
    else:
        idx = int(value)
    if n_images <= 0:
        return 0
    if idx < 0 or idx >= n_images:
        return 0
    return idx


def apply_sparse_track_ids_to_table(
    photometry: Table,
    row_index: np.ndarray,
) -> Table:
    """Set ``id`` from a correlation row (``-1`` = miss) and drop unmatched rows."""
    n = len(photometry)
    n_tracks = int(row_index.size)
    ids = np.full(n, -1, dtype=np.int64)
    row = np.asarray(row_index, dtype=int).ravel()
    valid = (row >= 0) & (row < n)
    if np.any(valid):
        ids[row[valid]] = np.arange(n_tracks, dtype=np.int64)[valid]
    out = photometry.copy()
    out["id"] = ids
    return out[ids >= 0]


def apply_correlation_index_to_images(
    image_list: list[AnalysisImage],
    correlation_index: np.ndarray,
    *,
    require_complete_intersection: bool,
) -> None:
    """Apply intra-series correlation: slice (dense) or tag ``id`` (sparse)."""
    if require_complete_intersection:
        for j, image in enumerate(image_list):
            image.photometry = image.photometry[correlation_index[j, :]]
        return
    for j, image in enumerate(image_list):
        image.photometry = apply_sparse_track_ids_to_table(
            image.photometry,
            correlation_index[j, :],
        )


def remap_series_ids_from_reference_index(
    image_list: list[AnalysisImage],
    reference_image_index: int,
    correlation_row: np.ndarray,
    *,
    require_complete_intersection: bool,
) -> None:
    """Apply inter-filter tracks to every image in a series.

    Dense mode slices every table with the same row index (aligned intra
    tables). Sparse mode remaps existing ``id`` values via the reference
    image's rows, so non-reference frames keep their native detections.
    """
    if require_complete_intersection:
        for image in image_list:
            image.photometry = image.photometry[correlation_row]
        return

    ref_phot = image_list[reference_image_index].photometry
    if "id" in ref_phot.colnames:
        ref_old_ids = np.asarray(ref_phot["id"], dtype=np.int64)
    else:
        ref_old_ids = np.arange(len(ref_phot), dtype=np.int64)

    old_to_new: dict[int, int] = {}
    row = np.asarray(correlation_row, dtype=int).ravel()
    for k, idx in enumerate(row):
        if 0 <= int(idx) < len(ref_old_ids):
            old_to_new[int(ref_old_ids[int(idx)])] = int(k)

    for image in image_list:
        phot = image.photometry
        if "id" in phot.colnames:
            old_ids = np.asarray(phot["id"], dtype=np.int64)
        else:
            old_ids = np.arange(len(phot), dtype=np.int64)
        new_ids = np.full(len(phot), -1, dtype=np.int64)
        for i, oid in enumerate(old_ids):
            mapped = old_to_new.get(int(oid))
            if mapped is not None:
                new_ids[i] = mapped
        out = phot.copy()
        out["id"] = new_ids
        image.photometry = out[new_ids >= 0]


def flux_arrays_from_photometry_tables(
    tables: list[Table | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``flux_fit`` / ``flux_err`` on ``id``, padding misses with NaN."""
    n_images = len(tables)
    ids_per: list[np.ndarray] = []
    for tbl in tables:
        if tbl is None or len(tbl) == 0:
            ids_per.append(np.array([], dtype=np.int64))
            continue
        if "id" in tbl.colnames:
            ids_per.append(np.asarray(tbl["id"], dtype=np.int64))
        else:
            ids_per.append(np.arange(len(tbl), dtype=np.int64))

    nonempty = [ids[ids >= 0] for ids in ids_per if ids.size]
    if not nonempty:
        return np.zeros((n_images, 0)), np.zeros((n_images, 0))
    all_ids = np.unique(np.concatenate(nonempty))
    id_to_col = {int(sid): k for k, sid in enumerate(all_ids)}
    n_objects = len(all_ids)
    flux = np.full((n_images, n_objects), np.nan)
    flux_err = np.full((n_images, n_objects), np.nan)
    for i, tbl in enumerate(tables):
        if tbl is None or len(tbl) == 0:
            continue
        ids = ids_per[i]
        flux_col = np.asarray(tbl["flux_fit"], dtype=float)
        err_col = np.asarray(tbl["flux_err"], dtype=float)
        for r, sid in enumerate(ids):
            if int(sid) < 0:
                continue
            col = id_to_col.get(int(sid))
            if col is None:
                continue
            flux[i, col] = flux_col[r]
            flux_err[i, col] = err_col[r]
    return flux, flux_err


__all__ = [
    "apply_correlation_index_to_images",
    "apply_sparse_track_ids_to_table",
    "coerce_reference_image_index",
    "effective_miss_limit",
    "flux_arrays_from_photometry_tables",
    "pick_auto_reference_image",
    "remap_series_ids_from_reference_index",
]
