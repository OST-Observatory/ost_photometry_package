"""
Bridge: convert Observation/Context data to differential calibration epochs.

observation_to_calibration_epochs(context, config) fills context.calibration_epochs,
context.calibration_epoch_meta, and context.calibration_epochs_skipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from astropy.table import Table

from .config import PipelineConfig
from .context import AnalysisContext


def _photometry_table_from_image(image, filter_: str, wcs_obj) -> Optional[Table]:
    """One band: id, ra, dec, x, y, mag_<f>, err_<f>. Returns None if unusable."""
    if image.photometry is None:
        return None
    phot = image.photometry
    x = np.asarray(phot["x_fit"])
    y = np.asarray(phot["y_fit"])
    try:
        sky = wcs_obj.pixel_to_world(x, y)
        ra = sky.ra.deg
        dec = sky.dec.deg
    except Exception:
        return None

    mag_col = f"mag_{filter_}"
    err_col = f"err_{filter_}"
    n = len(phot)
    tbl = Table()
    tbl["id"] = np.arange(n) if "id" not in phot.colnames else phot["id"]
    tbl["ra"] = ra
    tbl["dec"] = dec
    tbl["x"] = np.asarray(x, dtype=float)
    tbl["y"] = np.asarray(y, dtype=float)
    mag_vals = phot["mags_fit"]
    err_vals = phot["mags_unc"]
    if hasattr(mag_vals, "value"):
        mag_vals = mag_vals.value
    if hasattr(err_vals, "value"):
        err_vals = err_vals.value
    tbl[mag_col] = np.asarray(mag_vals, dtype=float)
    tbl[err_col] = np.asarray(err_vals, dtype=float)
    return tbl


def _airmass_for_image(image) -> float:
    am = getattr(image, "air_mass", None)
    if am is None:
        return 1.0
    return float(am)


def _jd_for_image(image) -> Optional[float]:
    jd = getattr(image, "jd", None)
    if jd is None:
        return None
    return float(jd)


def _merge_epoch_on_id(
    tables: Dict[str, Table],
    ref_filter: str,
    filter_order: List[str],
    airmasses: Dict[str, float],
) -> Table:
    """Left-join magnitudes onto reference filter row order by ``id``."""
    base = tables[ref_filter].copy()
    n = len(base)
    base_ids = np.asarray(base["id"], dtype=np.int64)

    for f in filter_order:
        am_col = f"airmass_{f}"
        base[am_col] = np.full(n, airmasses[f], dtype=float)

    for f in filter_order:
        if f == ref_filter:
            continue
        t = tables[f]
        mag_col = f"mag_{f}"
        err_col = f"err_{f}"
        id_to_row = {int(np.asarray(t["id"])[i]): i for i in range(len(t))}
        mag_arr = np.full(n, np.nan, dtype=float)
        err_arr = np.full(n, np.nan, dtype=float)
        for i in range(n):
            j = id_to_row.get(int(base_ids[i]))
            if j is not None:
                mag_arr[i] = float(np.asarray(t[mag_col])[j])
                err_arr[i] = float(np.asarray(t[err_col])[j])
        base[mag_col] = mag_arr
        base[err_col] = err_arr

    ams = [airmasses[f] for f in filter_order]
    base["airmass"] = np.full(n, float(np.mean(ams)), dtype=float)
    return base


def _pairing_index(
    context: AnalysisContext,
    filter_list: List[str],
    skipped: List[dict],
) -> List[Dict[str, Any]]:
    """
    Build list of pair dicts: {filter: image} per epoch.
    Uses same list index in each series.
    """
    series_by_f = {
        f: context.image_series_dict[f]
        for f in filter_list
        if f in context.image_series_dict
    }
    if len(series_by_f) != len(filter_list):
        return []

    lengths = [len(s.image_list) for s in series_by_f.values()]
    n_epoch = min(lengths)
    if min(lengths) != max(lengths):
        skipped.append(
            {
                "reason": "index_unequal_lengths",
                "message": (
                    f"Image counts differ across filters {filter_list}: {dict(zip(filter_list, lengths))}. "
                    f"Using first {n_epoch} index slots only."
                ),
            }
        )

    pairs: List[Dict[str, Any]] = []
    for i in range(n_epoch):
        group: Dict[str, Any] = {}
        ok = True
        for f in filter_list:
            s = series_by_f[f]
            if i >= len(s.image_list):
                ok = False
                break
            im = s.image_list[i]
            if im.photometry is None:
                ok = False
                skipped.append(
                    {
                        "reason": "missing_photometry",
                        "index": i,
                        "filter": f,
                        "pd": getattr(im, "pd", None),
                    }
                )
                break
            group[f] = im
        if ok and len(group) == len(filter_list):
            pairs.append(group)
    return pairs


def _pairing_jd_nearest(
    context: AnalysisContext,
    filter_list: List[str],
    ref_filter: str,
    jd_tolerance: float,
    skipped: List[dict],
) -> List[Dict[str, Any]]:
    series_by_f = {
        f: context.image_series_dict[f]
        for f in filter_list
        if f in context.image_series_dict
    }
    if ref_filter not in series_by_f:
        return []

    ref_images = [
        im
        for im in series_by_f[ref_filter].image_list
        if im.photometry is not None and _jd_for_image(im) is not None
    ]
    ref_images.sort(key=lambda im: _jd_for_image(im) or 0.0)

    other_filters = [f for f in filter_list if f != ref_filter]

    # Pool of candidates per filter (jd known, photometry ok)
    pool: Dict[str, List] = {}
    for f in other_filters:
        pool[f] = [
            im
            for im in series_by_f[f].image_list
            if im.photometry is not None and _jd_for_image(im) is not None
        ]
        pool[f].sort(key=lambda im: _jd_for_image(im) or 0.0)

    used_idx = {f: set() for f in other_filters}
    pairs: List[Dict[str, Any]] = []

    for ref_im in ref_images:
        jd0 = _jd_for_image(ref_im)
        if jd0 is None:
            continue
        group: Dict[str, Any] = {ref_filter: ref_im}
        ok = True
        for f in other_filters:
            best_im = None
            best_dj = float("inf")
            best_j = None
            for j, im in enumerate(pool[f]):
                if j in used_idx[f]:
                    continue
                jdi = _jd_for_image(im)
                if jdi is None:
                    continue
                dj = abs(jdi - jd0)
                if dj < best_dj:
                    best_dj = dj
                    best_im = im
                    best_j = j
            if best_im is None or best_dj > jd_tolerance or best_j is None:
                ok = False
                skipped.append(
                    {
                        "reason": "jd_no_partner" if best_im is None else "jd_exceeds_tolerance",
                        "reference_filter": ref_filter,
                        "reference_pd": getattr(ref_im, "pd", None),
                        "reference_jd": jd0,
                        "failed_filter": f,
                        "best_delta_jd": best_dj if best_im is not None else None,
                        "jd_tolerance": jd_tolerance,
                    }
                )
                break
            used_idx[f].add(best_j)
            group[f] = best_im
        if ok:
            pairs.append(group)

    return pairs


def observation_to_calibration_epochs(
    context: AnalysisContext,
    config: PipelineConfig,
) -> Dict[str, Table]:
    """
    Build multi-band calibration epoch tables and store them on ``context``.

    Each epoch is one table with ``mag_<f>``, ``err_<f>``, ``airmass_<f>``,
    mean ``airmass``, aligned rows by correlated ``id`` on the reference filter.

    Parameters
    ----------
    context
        Must have ``image_series_dict``, ``filter_list``, WCS per series.
    config
        Uses ``differential_exposure_pairing``, ``differential_exposure_jd_tolerance``,
        ``differential_reference_filter``.

    Returns
    -------
    dict[str, Table]
        Same as ``context.calibration_epochs`` after the call.
    """
    context.calibration_epochs = {}
    context.calibration_epoch_meta = {}
    context.calibration_epochs_skipped = []

    filter_list = list(context.filter_list)
    if not filter_list:
        return {}

    ref_filter = config.differential_reference_filter or filter_list[0]
    if ref_filter not in context.image_series_dict:
        ref_filter = filter_list[0]

    pairing = config.differential_exposure_pairing
    skipped = context.calibration_epochs_skipped

    if pairing == "index":
        image_groups = _pairing_index(context, filter_list, skipped)
    else:
        image_groups = _pairing_jd_nearest(
            context,
            filter_list,
            ref_filter,
            config.differential_exposure_jd_tolerance,
            skipped,
        )

    epoch_idx = 0
    for group in image_groups:
        tables: Dict[str, Table] = {}
        airmasses: Dict[str, float] = {}
        filter_jds: Dict[str, Optional[float]] = {}
        filter_pds: Dict[str, Any] = {}

        failed = False
        for f in filter_list:
            im = group.get(f)
            if im is None:
                failed = True
                break
            series = context.image_series_dict.get(f)
            wcs_obj = getattr(series, "wcs", None) if series is not None else None
            if wcs_obj is None:
                failed = True
                skipped.append({"reason": "no_wcs", "filter": f})
                break
            t = _photometry_table_from_image(im, f, wcs_obj)
            if t is None:
                failed = True
                skipped.append(
                    {"reason": "bad_photometry_or_wcs", "filter": f, "pd": getattr(im, "pd", None)}
                )
                break
            tables[f] = t
            airmasses[f] = _airmass_for_image(im)
            filter_jds[f] = _jd_for_image(im)
            filter_pds[f] = getattr(im, "pd", None)

        if failed:
            continue

        try:
            merged = _merge_epoch_on_id(tables, ref_filter, filter_list, airmasses)
        except Exception as exc:
            skipped.append({"reason": "merge_failed", "error": str(exc)})
            continue

        epoch_id = f"epoch_{epoch_idx:03d}"
        epoch_idx += 1

        context.calibration_epochs[epoch_id] = merged
        context.calibration_epoch_meta[epoch_id] = {
            "filter_pds": filter_pds,
            "filter_jds": filter_jds,
            "reference_filter": ref_filter,
            "pairing_mode": pairing,
            "airmasses": airmasses,
        }

    return context.calibration_epochs


def observation_to_epoch_tables(context: AnalysisContext) -> Dict[str, Table]:
    """
    Return ``context.calibration_epochs`` (copy as dict).

    Prefer :func:`observation_to_calibration_epochs` with ``PipelineConfig`` to fill
    the context. If epochs were never built, returns an empty dict.
    """
    return dict(context.calibration_epochs)


# Backward-compatible alias
observation_to_frame_tables = observation_to_epoch_tables
