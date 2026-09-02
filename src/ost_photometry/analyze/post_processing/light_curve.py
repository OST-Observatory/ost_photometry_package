"""Light curve preparation from calibrated tables and time series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from astropy import uncertainty as unc
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.timeseries import TimeSeries

from ... import terminal_output
from ...output_layout import tables_dir

LightCurveQuantity = Literal["magnitude", "flux"]
LightCurveCalibrationRows = Literal["auto", "transformed", "simple"]


def epoch_native_flux_err_columns(tbl: Table, filter_: str) -> tuple[str, str] | None:
    """``(flux_col, flux_err_col)`` for extracted flux in epoch-native tables."""
    for base_a, base_b in (
        (f"flux_inst_{filter_}", f"flux_err_inst_{filter_}"),
        (f"flux_{filter_}", f"flux_err_{filter_}"),
    ):
        if base_a in tbl.colnames and base_b in tbl.colnames:
            return base_a, base_b
    return None


def epoch_native_mag_err_columns(tbl: Table, filter_: str) -> tuple[str, str] | None:
    """
    Return ``(mag_column, err_column)`` for epoch-native rows.

    Prefers ``mag_cal_<filter>`` when present; otherwise ``mag_inst_<filter>``
    (instrumental-only tables from the pipeline).
    """
    mc, ec = f"mag_cal_{filter_}", f"err_cal_{filter_}"
    if mc in tbl.colnames and ec in tbl.colnames:
        return mc, ec
    mi, ei = f"mag_inst_{filter_}", f"err_inst_{filter_}"
    if mi in tbl.colnames and ei in tbl.colnames:
        return mi, ei
    return None


def is_epoch_native_photometry_table(
    data: Table,
    filter_: str,
    *,
    quantity: LightCurveQuantity = "magnitude",
) -> bool:
    """True if ``data`` has per-epoch rows with mag or flux columns for ``filter_``."""
    if not isinstance(data, Table):
        return False
    if "epoch_id" not in data.colnames or "id" not in data.colnames:
        return False
    if quantity == "flux":
        return epoch_native_flux_err_columns(data, filter_) is not None
    return epoch_native_mag_err_columns(data, filter_) is not None


def _subset_drop_simple_when_transformed_same_jd(
    sub: Table,
    epoch_meta: dict | None,
    filter_: str,
) -> Table:
    """
    For one source, drop ``epoch_*_simple`` rows when another row has the same JD
    and a non-``_simple`` ``epoch_id`` (transformed calibration).

    Uses ``observation_jd`` / ``jd`` columns when present, else ``epoch_meta``.
    Keeps ``*_simple``-only tables (``apply_transformation=False``) unchanged.
    """
    if len(sub) == 0:
        return sub
    eid_col = np.asarray(sub["epoch_id"]).astype(str)
    n = len(sub)
    jds = _row_jds_from_table(sub, epoch_meta, filter_)
    drop = np.zeros(n, dtype=bool)
    simple_rows = [i for i in range(n) if str(eid_col[i]).endswith("_simple")]
    trans_rows = [i for i in range(n) if not str(eid_col[i]).endswith("_simple")]
    for i in simple_rows:
        if not np.isfinite(jds[i]):
            continue
        for j in trans_rows:
            if not np.isfinite(jds[j]):
                continue
            if np.isclose(jds[i], jds[j], rtol=0.0, atol=1e-9):
                drop[i] = True
                break
    if not np.any(drop):
        return sub
    return sub[~drop]


def _subset_for_calibration_row_mode(
    sub: Table,
    mode: LightCurveCalibrationRows,
    epoch_meta: dict | None,
    filter_: str,
) -> Table:
    """Select transformed-only, simple-only, or auto-deduplicated rows."""
    if len(sub) == 0:
        return sub
    eid = np.asarray(sub["epoch_id"]).astype(str)
    if mode == "auto":
        return _subset_drop_simple_when_transformed_same_jd(sub, epoch_meta, filter_)
    if mode == "transformed":
        keep = np.array([not str(s).endswith("_simple") for s in eid], dtype=bool)
        if not np.any(keep):
            return sub
        return sub[keep]
    if mode == "simple":
        keep = np.array([str(s).endswith("_simple") for s in eid], dtype=bool)
        if not np.any(keep):
            return sub
        return sub[keep]
    return sub


def epoch_native_flux_matrix_for_pipeline_normalization(
    tbl: Table,
    filter_: str,
    epoch_meta: dict | None,
    calibration_rows: LightCurveCalibrationRows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a dense ``(n_epochs, n_sources)`` flux matrix from an epoch-native table.

    Rows are ordered by sorted unique JD; columns by sorted unique correlated ``id``.
    Missing ``(epoch, id)`` cells are ``0`` (same masking convention as
    :func:`ost_photometry.analyze.calibration.quasi_flux_calibration_flux_arrays`).
    """
    cols = epoch_native_flux_err_columns(tbl, filter_)
    if cols is None:
        raise ValueError(
            f"epoch_native_flux_matrix_for_pipeline_normalization: no flux columns "
            f"for filter {filter_!r} (need flux_inst_* / flux_err_inst_* or flux_* / flux_err_*)."
        )
    val_col, err_col = cols
    uid = np.unique(np.asarray(tbl["id"]).astype(int))
    parts: list[Table] = []
    for sid in uid:
        m = np.asarray(tbl["id"]).astype(int) == int(sid)
        sub = _subset_for_calibration_row_mode(
            tbl[m], calibration_rows, epoch_meta, filter_
        )
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        raise ValueError(
            "epoch_native_flux_matrix_for_pipeline_normalization: no rows after "
            "calibration_rows filtering."
        )
    stacked = vstack(parts)
    nrows = len(stacked)
    jds = _row_jds_from_table(stacked, epoch_meta, filter_)
    if not np.any(np.isfinite(jds)):
        raise ValueError(
            "epoch_native_flux_matrix_for_pipeline_normalization: no finite JDs "
            "(need observation_jd / jd or epoch_meta)."
        )
    finite_j = jds[np.isfinite(jds)]
    unique_jds = np.sort(np.unique(finite_j))
    id_list = np.sort(uid)
    n_ep, n_src = len(unique_jds), len(id_list)
    flux_m = np.zeros((n_ep, n_src), dtype=float)
    err_m = np.zeros((n_ep, n_src), dtype=float)
    row_ids = np.asarray(stacked["id"]).astype(int)
    vals = np.asarray(stacked[val_col], dtype=float)
    errs = np.abs(np.asarray(stacked[err_col], dtype=float))
    atol = max(1e-9, np.finfo(float).eps * max(1.0, np.max(np.abs(unique_jds))))
    for i in range(nrows):
        jd = float(jds[i])
        if not np.isfinite(jd):
            continue
        ei = int(np.searchsorted(unique_jds, jd))
        if ei >= n_ep or not np.isclose(unique_jds[ei], jd, rtol=0.0, atol=atol):
            continue
        ci = int(np.searchsorted(id_list, row_ids[i]))
        if ci >= n_src or id_list[ci] != row_ids[i]:
            continue
        flux_m[ei, ci] = vals[i]
        err_m[ei, ci] = errs[i]
    return flux_m, err_m, unique_jds, id_list


def object_id_from_epoch_native_sky(
    tbl: Table,
    coord: SkyCoord,
    *,
    max_sep: u.Quantity = 5 * u.arcsec,
) -> int:
    """
    Resolve ``id`` by nearest sky position (median ``ra``/``dec`` per ``id`` in ``tbl``).

    Parameters
    ----------
    tbl
        Epoch-native table with ``id``, ``ra``, ``dec`` (degrees, ICRS-style).
    coord
        Target position (any frame; compared after transformation to ICRS).
    max_sep
        Match rejected if the nearest neighbour exceeds this on-sky separation.
    """
    if len(tbl) == 0:
        raise ValueError("object_id_from_epoch_native_sky: empty table")
    for col in ("id", "ra", "dec"):
        if col not in tbl.colnames:
            raise ValueError(
                f"object_id_from_epoch_native_sky: table missing column {col!r}"
            )
    ids_u = np.unique(np.asarray(tbl["id"]).astype(int))
    id_list: list[int] = []
    ra_list: list[float] = []
    dec_list: list[float] = []
    for sid in ids_u:
        m = np.asarray(tbl["id"]).astype(int) == int(sid)
        ra_m = float(np.nanmedian(np.asarray(tbl["ra"][m], dtype=float)))
        dec_m = float(np.nanmedian(np.asarray(tbl["dec"][m], dtype=float)))
        id_list.append(int(sid))
        ra_list.append(ra_m)
        dec_list.append(dec_m)
    cat = SkyCoord(
        ra=np.asarray(ra_list) * u.deg,
        dec=np.asarray(dec_list) * u.deg,
        frame="icrs",
    )
    c = coord.icrs
    if not c.isscalar:
        c = c[0]
    idx, sep2d, _d3d = match_coordinates_sky(c, cat)
    j = int(np.asarray(idx).ravel()[0])
    s = sep2d.ravel()[0]
    if s > max_sep:
        raise ValueError(
            f"No table source within {max_sep}: nearest separation is {s.to(u.arcsec)}"
        )
    terminal_output.print_to_terminal(
        f"Sky match: id={id_list[j]} (offset {s.to(u.arcsec).value:.3f} arcsec)",
        style_name="INFO",
    )
    return id_list[j]


def attach_observation_jd_column(
    tbl: Table,
    epoch_meta: dict,
    ref_filter: str,
) -> Table:
    """
    Add ``observation_jd`` (Julian Date) per row from ``epoch_meta`` and ``epoch_id``.

    Enables standalone light-curve plotting from ECSV without a full
    :class:`~ost_photometry.analyze.analyze.Observation`.
    """
    if len(tbl) == 0 or "epoch_id" not in tbl.colnames:
        return tbl
    out = tbl.copy()
    if "observation_jd" in out.colnames:
        return out
    eids = np.asarray(out["epoch_id"]).astype(str)
    jds = [_epoch_filter_jd(epoch_meta, eid, ref_filter) for eid in eids]
    out["observation_jd"] = np.asarray(jds, dtype=float)
    return out


def save_calibration_epoch_meta_json(meta: dict, path: str | Path) -> None:
    """Write ``calibration_epoch_meta``-style dict as JSON (for offline light curves)."""
    Path(path).write_text(
        json.dumps(_json_safe(meta), indent=2),
        encoding="utf-8",
    )


def write_epoch_meta_json(output_dir: str | Path, meta: dict | None) -> Path | None:
    """Write ``tables/epoch_meta.json`` when ``meta`` is non-empty."""
    if not meta:
        return None
    path = tables_dir(output_dir) / "epoch_meta.json"
    save_calibration_epoch_meta_json(meta, path)
    return path


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    return str(obj)


def load_calibration_epoch_meta_json(path: str | Path) -> dict:
    """Load JSON written by :func:`save_calibration_epoch_meta_json`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_jds_from_table(
    sub: Table,
    epoch_meta: dict | None,
    filter_: str,
) -> np.ndarray:
    """Per-row JDs: prefer ``observation_jd`` / ``jd`` columns, else ``epoch_meta``."""
    for col in ("observation_jd", "jd", "time_jd"):
        if col in sub.colnames:
            arr = np.asarray(sub[col], dtype=float)
            if np.any(np.isfinite(arr)):
                return arr
    eid_col = np.asarray(sub["epoch_id"]).astype(str)
    return np.array(
        [_epoch_filter_jd(epoch_meta, eid, filter_) for eid in eid_col],
        dtype=float,
    )


def _epoch_filter_jd(epoch_meta: dict | None, epoch_id: str, filter_: str) -> float:
    if not epoch_meta:
        return float("nan")
    meta = epoch_meta.get(epoch_id)
    if meta is None:
        meta = epoch_meta.get(str(epoch_id))
    if meta is None:
        return float("nan")
    jd_by_filter = meta.get("jd_by_filter") or meta.get("filter_jds") or {}
    jd = jd_by_filter.get(filter_)
    if jd is None:
        return float("nan")
    return float(jd)


def prepare_time_series_epoch_native(
    data: Table,
    filter_: str,
    source_id: int,
    epoch_meta: dict | None,
    *,
    quantity: LightCurveQuantity = "magnitude",
    calibration_rows: LightCurveCalibrationRows = "auto",
) -> tuple[np.ndarray, np.ndarray, Time]:
    """
    Build magnitude or flux arrays and observation times for one correlated source.

    Rows are filtered by ``id == source_id``, optionally restricted to transformed or
    simple ``epoch_id`` rows, sorted by JD. JDs come from ``observation_jd`` / ``jd``
    columns when present, otherwise from ``epoch_meta``.

    Returns
    -------
    values, errs, times
        Per-epoch magnitudes **or** flux (``quantity``), matching uncertainties, and JD times.
    """
    if quantity == "flux":
        cols = epoch_native_flux_err_columns(data, filter_)
    else:
        cols = epoch_native_mag_err_columns(data, filter_)
    if cols is None:
        return np.array([]), np.array([]), Time([], format="jd")
    val_col, err_col = cols
    ids = np.asarray(data["id"])
    mask = ids == int(source_id)
    sub = data[mask]
    if len(sub) == 0:
        return np.array([]), np.array([]), Time([], format="jd")

    sub = _subset_for_calibration_row_mode(sub, calibration_rows, epoch_meta, filter_)
    if len(sub) == 0:
        return np.array([]), np.array([]), Time([], format="jd")

    values = np.asarray(sub[val_col].ravel(), dtype=float)
    errs = np.abs(np.asarray(sub[err_col].ravel(), dtype=float))

    jds = _row_jds_from_table(sub, epoch_meta, filter_)
    valid = np.isfinite(jds)
    if not np.any(valid):
        terminal_output.print_to_terminal(
            f"No valid JDs (need ``observation_jd``/``jd`` column or "
            f"``calibration_epoch_meta``) for filter {filter_!r}, "
            f"source id {source_id}. Skipping light curve.",
            style_name="WARNING",
        )
        return np.array([]), np.array([]), Time([], format="jd")
    if not np.all(valid):
        values = values[valid]
        errs = errs[valid]
        jds = jds[valid]

    finite_val = np.isfinite(values) & np.isfinite(errs)
    if not np.any(finite_val):
        q = "flux" if quantity == "flux" else "magnitudes"
        terminal_output.print_to_terminal(
            f"No finite {q} for light curve: filter {filter_!r}, "
            f"source id {source_id}.",
            style_name="WARNING",
        )
        return np.array([]), np.array([]), Time([], format="jd")
    if not np.all(finite_val):
        n_drop = int(np.sum(~finite_val))
        terminal_output.print_to_terminal(
            f"Dropping {n_drop} row(s) with non-finite values for filter {filter_!r} "
            f"(source id {source_id}).",
            style_name="INFO",
        )
        values = values[finite_val]
        errs = errs[finite_val]
        jds = jds[finite_val]

    order = np.argsort(jds, kind="stable")
    jds = jds[order]
    values = values[order]
    errs = errs[order]
    return values, errs, Time(jds, format="jd")


def prepare_time_series_data(
        data: unc.core.NdarrayDistribution,
        filter_: str, object_id: int, calibration_type: str = 'transformed'
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare magnitude arrays from an ``NdarrayDistribution`` (e.g. normalized flux).

    Parameters
    ----------
    data
        Per-epoch, per-object distributions

    filter_
        Unused for distributions; kept for API compatibility.

    object_id
        Object (column) index in the distribution.

    calibration_type
        Unused for distributions; kept for API compatibility.
    """
    if isinstance(data, unc.core.NdarrayDistribution):
        return data.pdf_median()[:, object_id], data.pdf_std()[:, object_id]
    raise TypeError(
        f"prepare_time_series_data expects NdarrayDistribution; got {type(data)}. "
        "Use epoch-native tables with prepare_plot_time_series(..., epoch_meta=...)."
    )


def mk_time_series_flux(
    observation_times: Time,
    flux: np.ndarray,
    flux_errors: np.ndarray,
    filter_: str,
) -> TimeSeries:
    """Build a :class:`~astropy.timeseries.TimeSeries` for extracted flux."""
    fe = np.abs(np.asarray(flux_errors, dtype=float))
    col = f"flux_{filter_}"
    return TimeSeries(
        time=observation_times,
        data={
            col: np.asarray(flux, dtype=float) << u.one,
            f"{col}_err": fe << u.one,
        },
    )


def mk_time_series(
        observation_times: Time, magnitudes: np.ndarray,
        magnitude_errors: np.ndarray, filter_: str) -> TimeSeries:
    """
    Make a time series object

    Parameters
    ----------
    observation_times
        Observation times

    magnitudes
        Object magnitudes

    magnitude_errors
        Object uncertainties

    filter_
        Filter

    Returns
    -------
    ts
    """
    mag_errs = np.abs(np.asarray(magnitude_errors, dtype=float))
    ts = TimeSeries(
        time=observation_times,
        data={
            filter_: magnitudes << u.mag,
            filter_ + '_err': mag_errs << u.mag,
        }
    )
    return ts


LIGHT_CURVES_FILENAME = "light_curves.ecsv"
CALIBRATOR_STATS_FILENAME = "calibrator_lc_stats.ecsv"
JD_MINUS_OFFSET = 2450000.0


def _as_positive_period(period) -> float | None:
    if period is None or period == "?":
        return None
    try:
        val = float(period)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def night_id_from_jd(jd) -> np.ndarray:
    """Local night index: ``floor(JD - 0.5)`` so evening UT stays one night."""
    arr = np.asarray(jd, dtype=float)
    out = np.full(arr.shape, -1, dtype=np.int64)
    ok = np.isfinite(arr)
    out[ok] = np.floor(arr[ok] - 0.5).astype(np.int64)
    return out


def excess_rms(values, errors) -> float:
    """
    Scatter beyond photometric noise: ``sqrt(max(0, RMS^2 - median(err)^2))``.

    Computed about the sample median. Returns 0 if fewer than two finite points.
    """
    y = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(y)
    if int(np.count_nonzero(ok)) < 2:
        return 0.0
    y = y[ok]
    e = e[ok] if e.shape == np.asarray(values).shape else e[np.isfinite(e)]
    rms = float(np.sqrt(np.mean((y - np.median(y)) ** 2)))
    if e.size:
        e_ok = e[np.isfinite(e)]
        med_e = float(np.median(e_ok)) if e_ok.size else 0.0
    else:
        med_e = 0.0
    if not np.isfinite(med_e) or med_e < 0.0:
        med_e = 0.0
    return float(np.sqrt(max(0.0, rms * rms - med_e * med_e)))


def _airmass_for_rows(
    sub: Table,
    epoch_meta: dict | None,
    filter_: str,
) -> np.ndarray:
    col_f = f"airmass_{filter_}"
    if col_f in sub.colnames:
        return np.asarray(sub[col_f], dtype=float)
    if "airmass" in sub.colnames:
        return np.asarray(sub["airmass"], dtype=float)
    n = len(sub)
    out = np.full(n, np.nan, dtype=float)
    if not epoch_meta or "epoch_id" not in sub.colnames:
        return out
    eids = np.asarray(sub["epoch_id"]).astype(str)
    for i, eid in enumerate(eids):
        meta = epoch_meta.get(eid)
        if meta is None:
            meta = epoch_meta.get(str(eid))
        if not meta:
            continue
        ams = meta.get("airmasses") or {}
        val = ams.get(filter_)
        if val is None:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[i] = fv
    return out


def _empty_light_curves_table() -> Table:
    return Table(
        {
            "id": np.array([], dtype=np.int64),
            "object_name": np.array([], dtype="U64"),
            "filter": np.array([], dtype="U32"),
            "epoch_id": np.array([], dtype="U64"),
            "jd": np.array([], dtype=float),
            "bjd_tdb": np.array([], dtype=float),
            "airmass": np.array([], dtype=float),
            "night_id": np.array([], dtype=np.int64),
            "mag": np.array([], dtype=float),
            "mag_err": np.array([], dtype=float),
            "flux": np.array([], dtype=float),
            "flux_err": np.array([], dtype=float),
            "quantity": np.array([], dtype="U16"),
            "flag_outlier": np.array([], dtype=bool),
            "ra": np.array([], dtype=float),
            "dec": np.array([], dtype=float),
            "is_calibrator": np.array([], dtype=bool),
        }
    )


def _rows_from_epoch_native_source(
    data: Table,
    filter_: str,
    source_id: int,
    epoch_meta: dict | None,
    *,
    quantity: LightCurveQuantity,
    calibration_rows: LightCurveCalibrationRows,
    object_name: str = "",
    is_calibrator: bool = False,
) -> Table | None:
    if quantity == "flux":
        cols = epoch_native_flux_err_columns(data, filter_)
    else:
        cols = epoch_native_mag_err_columns(data, filter_)
    if cols is None:
        return None
    val_col, err_col = cols
    ids = np.asarray(data["id"]).astype(int)
    sub = data[ids == int(source_id)]
    if len(sub) == 0:
        return None
    sub = _subset_for_calibration_row_mode(sub, calibration_rows, epoch_meta, filter_)
    if len(sub) == 0:
        return None
    values = np.asarray(sub[val_col].ravel(), dtype=float)
    errs = np.abs(np.asarray(sub[err_col].ravel(), dtype=float))
    jds = _row_jds_from_table(sub, epoch_meta, filter_)
    airmass = _airmass_for_rows(sub, epoch_meta, filter_)
    if "epoch_id" in sub.colnames:
        eids = np.asarray(sub["epoch_id"]).astype(str)
    else:
        eids = np.array([""] * len(sub), dtype=str)
    if "ra" in sub.colnames:
        ra = np.asarray(sub["ra"], dtype=float)
    else:
        ra = np.full(len(sub), np.nan)
    if "dec" in sub.colnames:
        dec = np.asarray(sub["dec"], dtype=float)
    else:
        dec = np.full(len(sub), np.nan)
    n = len(sub)
    mag = np.full(n, np.nan)
    mag_err = np.full(n, np.nan)
    flux = np.full(n, np.nan)
    flux_err = np.full(n, np.nan)
    if quantity == "flux":
        flux[:] = values
        flux_err[:] = errs
    else:
        mag[:] = values
        mag_err[:] = errs
    name = str(object_name or "")
    return Table(
        {
            "id": np.full(n, int(source_id), dtype=np.int64),
            "object_name": np.full(n, name, dtype="U64"),
            "filter": np.full(n, str(filter_), dtype="U32"),
            "epoch_id": eids.astype("U64"),
            "jd": jds,
            "bjd_tdb": np.full(n, np.nan),
            "airmass": airmass,
            "night_id": night_id_from_jd(jds),
            "mag": mag,
            "mag_err": mag_err,
            "flux": flux,
            "flux_err": flux_err,
            "quantity": np.full(n, str(quantity), dtype="U16"),
            "flag_outlier": np.zeros(n, dtype=bool),
            "ra": ra,
            "dec": dec,
            "is_calibrator": np.full(n, bool(is_calibrator), dtype=bool),
        }
    )


def flag_outliers_in_light_curves(
    tbl: Table,
    sigma: float | None = 5.0,
) -> Table:
    """Per ``(id, filter)`` sigma-clip on mag or flux. Flags stay in the table."""
    out = tbl.copy()
    n = len(out)
    flag = np.zeros(n, dtype=bool)
    if sigma is None or n == 0:
        out["flag_outlier"] = flag
        return out
    from astropy.stats import sigma_clip

    ids = np.asarray(out["id"]).astype(int)
    filts = np.asarray(out["filter"]).astype(str)
    qty = np.asarray(out["quantity"]).astype(str)
    mag = np.asarray(out["mag"], dtype=float)
    flux = np.asarray(out["flux"], dtype=float)
    sig = float(sigma)
    for sid in np.unique(ids):
        for filt in np.unique(filts[ids == sid]):
            m = (ids == sid) & (filts == filt)
            q = qty[m]
            if np.any(q == "flux"):
                y = flux[m]
            else:
                y = mag[m]
            ok = np.isfinite(y)
            if int(np.count_nonzero(ok)) < 4:
                continue
            y_ok = y[ok]
            center = float(np.median(y_ok))
            mad = float(np.median(np.abs(y_ok - center)))
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale <= 0.0:
                clipped = sigma_clip(y_ok, sigma=sig, masked=True)
                local = np.zeros(int(np.count_nonzero(m)), dtype=bool)
                local[ok] = np.asarray(clipped.mask, dtype=bool)
            else:
                local = np.zeros(int(np.count_nonzero(m)), dtype=bool)
                local[ok] = np.abs(y_ok - center) > sig * scale
            flag[m] = local
    out["flag_outlier"] = flag
    return out


def add_bjd_tdb_column(tbl: Table, location) -> Table:
    """Barycentric JD (TDB) from ``jd`` + source RA/Dec + observatory location."""
    out = tbl.copy()
    n = len(out)
    bjd = np.full(n, np.nan, dtype=float)
    if location is None or n == 0:
        out["bjd_tdb"] = bjd
        return out
    jd = np.asarray(out["jd"], dtype=float)
    ra = np.asarray(out["ra"], dtype=float)
    dec = np.asarray(out["dec"], dtype=float)
    ok = np.isfinite(jd) & np.isfinite(ra) & np.isfinite(dec)
    if not np.any(ok):
        out["bjd_tdb"] = bjd
        return out
    t = Time(jd[ok], format="jd", scale="utc")
    coord = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg, frame="icrs")
    ltt = t.light_travel_time(coord, kind="barycentric", location=location)
    bjd[ok] = (t.tdb + ltt).jd
    out["bjd_tdb"] = bjd
    return out


def add_color_index_rows(tbl: Table, color: str | None) -> Table:
    """Append colour rows (e.g. ``B-V``) matched on ``id`` and ``epoch_id``."""
    if not color or color in ("?", "?-?", "-"):
        return tbl
    text = str(color).strip()
    if "-" not in text:
        return tbl
    f1, f2 = (p.strip() for p in text.split("-", 1))
    if not f1 or not f2:
        return tbl
    filts = np.asarray(tbl["filter"]).astype(str)
    has1 = np.any(filts == f1)
    has2 = np.any(filts == f2)
    if not has1 or not has2:
        terminal_output.print_to_terminal(
            f"Colour {text!r}: missing filter {f1!r} or {f2!r} in light-curve table.",
            style_name="WARNING",
        )
        return tbl
    ids = np.asarray(tbl["id"]).astype(int)
    eids = np.asarray(tbl["epoch_id"]).astype(str)
    mag = np.asarray(tbl["mag"], dtype=float)
    mag_err = np.asarray(tbl["mag_err"], dtype=float)
    parts: list[Table] = []
    for sid in np.unique(ids):
        m1 = (ids == sid) & (filts == f1)
        m2 = (ids == sid) & (filts == f2)
        if not np.any(m1) or not np.any(m2):
            continue
        map2 = {eids[i]: i for i in np.flatnonzero(m2)}
        for i in np.flatnonzero(m1):
            j = map2.get(eids[i])
            if j is None:
                continue
            c = mag[i] - mag[j]
            e = np.hypot(mag_err[i], mag_err[j])
            row = tbl[i : i + 1].copy()
            row["filter"] = text
            row["mag"] = c
            row["mag_err"] = e
            row["flux"] = np.nan
            row["flux_err"] = np.nan
            row["quantity"] = "magnitude"
            row["flag_outlier"] = False
            parts.append(row)
    if not parts:
        return tbl
    return vstack([tbl, *parts], metadata_conflicts="silent")


def build_light_curves_table(
    phot: Table,
    filter_list: list[str],
    *,
    epoch_meta: dict | None = None,
    quantity: LightCurveQuantity = "magnitude",
    calibration_rows: LightCurveCalibrationRows = "auto",
    object_names: dict[int, str] | None = None,
    calibrator_ids: set[int] | None = None,
    outlier_sigma: float | None = 5.0,
    observatory_location=None,
    color: str | None = None,
) -> Table:
    """Long light-curve table: one row per source × filter × epoch."""
    names = object_names or {}
    cal = calibrator_ids or set()
    if phot is None or len(phot) == 0 or "id" not in phot.colnames:
        return _empty_light_curves_table()
    parts: list[Table] = []
    for sid in np.unique(np.asarray(phot["id"]).astype(int)):
        sid_i = int(sid)
        for filt in filter_list:
            if not is_epoch_native_photometry_table(phot, filt, quantity=quantity):
                continue
            rows = _rows_from_epoch_native_source(
                phot,
                filt,
                sid_i,
                epoch_meta,
                quantity=quantity,
                calibration_rows=calibration_rows,
                object_name=names.get(sid_i, ""),
                is_calibrator=sid_i in cal,
            )
            if rows is not None and len(rows) > 0:
                parts.append(rows)
    if not parts:
        return _empty_light_curves_table()
    tbl = vstack(parts, metadata_conflicts="silent")
    tbl = flag_outliers_in_light_curves(tbl, sigma=outlier_sigma)
    tbl = add_bjd_tdb_column(tbl, observatory_location)
    tbl = add_color_index_rows(tbl, color)
    from .magnitude_systems import table_magnitude_system

    tbl.meta["ost_photometry.magnitude_system"] = table_magnitude_system(phot)
    return tbl


def build_light_curves_table_from_flux(
    flux_distribution: unc.core.NdarrayDistribution,
    observation_times: Time,
    filter_: str,
    *,
    source_ids: np.ndarray | None = None,
    object_names: dict[int, str] | None = None,
    calibrator_ids: set[int] | None = None,
    airmasses: np.ndarray | None = None,
    ra: np.ndarray | None = None,
    dec: np.ndarray | None = None,
    outlier_sigma: float | None = 5.0,
    observatory_location=None,
) -> Table:
    """Long table from a normalized ``(n_epochs, n_objects)`` flux distribution."""
    med = np.asarray(flux_distribution.pdf_median(), dtype=float)
    std = np.abs(np.asarray(flux_distribution.pdf_std(), dtype=float))
    n_ep, n_obj = med.shape
    jds = np.asarray(observation_times.jd, dtype=float)
    if jds.size != n_ep:
        raise ValueError(
            f"observation_times length {jds.size} != n_epochs {n_ep} in flux array"
        )
    if source_ids is None:
        ids = np.arange(n_obj, dtype=np.int64)
    else:
        ids = np.asarray(source_ids).astype(np.int64)
        if ids.size != n_obj:
            raise ValueError("source_ids length must match flux object axis")
    names = object_names or {}
    cal = calibrator_ids or set()
    if airmasses is None:
        am = np.full(n_ep, np.nan)
    else:
        am = np.asarray(airmasses, dtype=float)
        if am.size != n_ep:
            am = np.full(n_ep, np.nan)
    nights = night_id_from_jd(jds)
    rows: list[Table] = []
    for j in range(n_obj):
        sid = int(ids[j])
        ra_j = float(ra[j]) if ra is not None and j < len(ra) else np.nan
        dec_j = float(dec[j]) if dec is not None and j < len(dec) else np.nan
        rows.append(
            Table(
                {
                    "id": np.full(n_ep, sid, dtype=np.int64),
                    "object_name": np.full(n_ep, str(names.get(sid, "")), dtype="U64"),
                    "filter": np.full(n_ep, str(filter_), dtype="U32"),
                    "epoch_id": np.array([f"epoch_{k:03d}" for k in range(n_ep)], dtype="U64"),
                    "jd": jds,
                    "bjd_tdb": np.full(n_ep, np.nan),
                    "airmass": am,
                    "night_id": nights,
                    "mag": np.full(n_ep, np.nan),
                    "mag_err": np.full(n_ep, np.nan),
                    "flux": med[:, j],
                    "flux_err": std[:, j],
                    "quantity": np.full(n_ep, "flux", dtype="U16"),
                    "flag_outlier": np.zeros(n_ep, dtype=bool),
                    "ra": np.full(n_ep, ra_j),
                    "dec": np.full(n_ep, dec_j),
                    "is_calibrator": np.full(n_ep, sid in cal, dtype=bool),
                }
            )
        )
    tbl = vstack(rows, metadata_conflicts="silent") if rows else _empty_light_curves_table()
    tbl = flag_outliers_in_light_curves(tbl, sigma=outlier_sigma)
    tbl = add_bjd_tdb_column(tbl, observatory_location)
    return tbl


def write_light_curves_table(tbl: Table, output_dir: str | Path) -> Path:
    """Write ``tables/light_curves.ecsv``."""
    path = tables_dir(output_dir) / LIGHT_CURVES_FILENAME
    tbl.write(str(path), format="ascii.ecsv", overwrite=True)
    return path


def calibrator_variability_stats(
    lc: Table,
    calibrator_ids: set[int] | list[int],
    filter_: str,
) -> Table:
    """Per-calibrator RMS / excess-RMS / χ²/ν in one filter (unflagged points)."""
    cal = {int(i) for i in calibrator_ids}
    if len(lc) == 0 or not cal:
        return Table(
            {
                "id": np.array([], dtype=np.int64),
                "filter": np.array([], dtype="U32"),
                "n": np.array([], dtype=np.int64),
                "med_mag": np.array([], dtype=float),
                "rms": np.array([], dtype=float),
                "excess_rms": np.array([], dtype=float),
                "chi2_nu": np.array([], dtype=float),
            }
        )
    ids = np.asarray(lc["id"]).astype(int)
    filts = np.asarray(lc["filter"]).astype(str)
    flag = np.asarray(lc["flag_outlier"], dtype=bool)
    mag = np.asarray(lc["mag"], dtype=float)
    mag_err = np.asarray(lc["mag_err"], dtype=float)
    flux = np.asarray(lc["flux"], dtype=float)
    flux_err = np.asarray(lc["flux_err"], dtype=float)
    qty = np.asarray(lc["quantity"]).astype(str)
    rec_id: list[int] = []
    rec_n: list[int] = []
    rec_med: list[float] = []
    rec_rms: list[float] = []
    rec_exc: list[float] = []
    rec_chi: list[float] = []
    for sid in sorted(cal):
        m = (ids == sid) & (filts == str(filter_)) & (~flag)
        if not np.any(m):
            continue
        if np.any(qty[m] == "flux"):
            y = flux[m]
            e = flux_err[m]
        else:
            y = mag[m]
            e = mag_err[m]
        ok = np.isfinite(y)
        y = y[ok]
        e = e[ok]
        n = int(y.size)
        if n < 2:
            continue
        med = float(np.median(y))
        rms = float(np.sqrt(np.mean((y - med) ** 2)))
        exc = excess_rms(y, e)
        e_pos = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        if np.any(np.isfinite(e_pos)):
            chi = float(np.nansum(((y - med) / e_pos) ** 2) / max(n - 1, 1))
        else:
            chi = np.nan
        rec_id.append(int(sid))
        rec_n.append(n)
        rec_med.append(med)
        rec_rms.append(rms)
        rec_exc.append(exc)
        rec_chi.append(chi)
    return Table(
        {
            "id": np.asarray(rec_id, dtype=np.int64),
            "filter": np.full(len(rec_id), str(filter_), dtype="U32"),
            "n": np.asarray(rec_n, dtype=np.int64),
            "med_mag": np.asarray(rec_med, dtype=float),
            "rms": np.asarray(rec_rms, dtype=float),
            "excess_rms": np.asarray(rec_exc, dtype=float),
            "chi2_nu": np.asarray(rec_chi, dtype=float),
        }
    )


def ids_excluding(
    ids: set[int] | list[int] | tuple[int, ...],
    exclude: set[int] | list[int] | tuple[int, ...] | None,
) -> set[int]:
    """``ids`` without any member of ``exclude`` (e.g. drop OOI from calibrators)."""
    skip = {int(i) for i in (exclude or ())}
    return {int(i) for i in ids if int(i) not in skip}


def top_variable_calibrator_ids(
    stats: Table,
    n: int = 3,
    exclude: set[int] | list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """``id``s with the largest ``excess_rms`` (stable order for ties)."""
    if stats is None or len(stats) == 0 or n <= 0:
        return []
    skip = {int(i) for i in (exclude or ())}
    exc = np.asarray(stats["excess_rms"], dtype=float)
    ids = np.asarray(stats["id"]).astype(int)
    order = np.argsort(-exc, kind="stable")
    out: list[int] = []
    for i in order:
        sid = int(ids[i])
        if sid in skip:
            continue
        out.append(sid)
        if len(out) >= int(n):
            break
    return out


def slice_light_curve(
    lc: Table,
    source_id: int,
    filter_: str,
) -> Table:
    ids = np.asarray(lc["id"]).astype(int)
    filts = np.asarray(lc["filter"]).astype(str)
    return lc[(ids == int(source_id)) & (filts == str(filter_))]


def build_check_star_qc_panels(
    lc: Table,
    filter_: str,
    ooi_ids: list[tuple[int, str]],
    calibrator_ids: list[int],
) -> list[tuple[str, Table]]:
    """
    Panel specs for the check-star QC figure.

    Objects of interest come first. Catalog calibrators that share an OOI
    ``id`` are omitted so the science target is not listed twice.
    """
    panels: list[tuple[str, Table]] = []
    seen: set[int] = set()
    for oid, name in ooi_ids:
        oid_i = int(oid)
        if oid_i in seen:
            continue
        sub = slice_light_curve(lc, oid_i, filter_)
        if len(sub) == 0:
            continue
        seen.add(oid_i)
        panels.append((f"object of interest {name} (id={oid_i})", sub))
    rank = 0
    for cid in calibrator_ids:
        cid_i = int(cid)
        if cid_i in seen:
            continue
        sub = slice_light_curve(lc, cid_i, filter_)
        if len(sub) == 0:
            continue
        seen.add(cid_i)
        rank += 1
        panels.append(
            (f"catalog calibrator id={cid_i} (#{rank} by excess RMS)", sub)
        )
    return panels


def plot_from_light_curves_table(
    lc: Table,
    source_id: int,
    filter_: str,
    output_dir: str,
    *,
    name_object: str | None = None,
    file_type: str = "pdf",
    subdirectory: str = "",
    transit_time: str | None = None,
    period: float | None = None,
    binning_factor: float | None = None,
    time_scale: str = "bjd_tdb",
    phase_cycles: int = 1,
    show_airmass: bool = True,
    magnitude_system: str | None = None,
) -> None:
    """JD (and folded, if period/t0) plots for one source from the long table."""
    from .. import plots
    from .magnitude_systems import table_magnitude_system

    sub = slice_light_curve(lc, source_id, filter_)
    if len(sub) == 0:
        terminal_output.print_to_terminal(
            f"No light-curve rows for id={source_id}, filter={filter_!r}.",
            style_name="WARNING",
        )
        return
    name = name_object or str(sub["object_name"][0] or source_id)
    mag_sys = magnitude_system or table_magnitude_system(lc)
    plots.light_curve_jd_from_table(
        sub,
        output_dir,
        name_object=name,
        filter_=filter_,
        file_type=file_type,
        subdirectory=subdirectory,
        time_scale=time_scale,
        show_airmass=show_airmass,
        magnitude_system=mag_sys,
    )
    per = period
    if per is not None and per != "?" and float(per) > 0.0:
        if transit_time is not None and transit_time != "?":
            plots.light_curve_fold_from_table(
                sub,
                output_dir,
                transit_time=str(transit_time),
                period=float(per),
                name_object=name,
                filter_=filter_,
                file_type=file_type,
                subdirectory=subdirectory,
                binning_factor=binning_factor,
                time_scale=time_scale,
                phase_cycles=phase_cycles,
                magnitude_system=mag_sys,
            )


def prepare_plot_time_series(
        data: unc.core.NdarrayDistribution | Table,
        observation_times: Time | None,
        filter_: str, object_name: str, object_id: int, output_dir: str,
        binning_factor: float | None = None, transit_time: str | None = None,
        period: float | None = None, file_name_suffix: str = '',
        light_curve_save_format: str | None = None,
        subdirectory: str = '',
        file_type_plots: str = 'pdf', calibration_type: str = 'transformed',
        epoch_meta: dict | None = None,
        light_curve_quantity: LightCurveQuantity = "magnitude",
        light_curve_calibration_rows: LightCurveCalibrationRows = "auto",
        magnitude_system: str | None = None,
        time_scale: str = "bjd_tdb",
        phase_cycles: int = 1,
        show_airmass: bool = True,
        observatory_location=None,
        ) -> None:
    """Plot one object from an epoch-native table or flux distribution (no per-star CSV)."""
    if object_id is None:
        terminal_output.print_to_terminal(
            f"ID of object {object_name} is None. Failed to create "
            f"light curve.",
            style_name='WARNING',
        )
        return

    _ = file_name_suffix
    _ = calibration_type
    _ = light_curve_save_format

    if isinstance(data, Table):
        lc = build_light_curves_table(
            data,
            [filter_],
            epoch_meta=epoch_meta,
            quantity=light_curve_quantity,
            calibration_rows=light_curve_calibration_rows,
            object_names={int(object_id): object_name},
            observatory_location=observatory_location,
        )
        plot_from_light_curves_table(
            lc,
            int(object_id),
            filter_,
            output_dir,
            name_object=object_name,
            file_type=file_type_plots,
            subdirectory=subdirectory,
            transit_time=transit_time,
            period=_as_positive_period(period),
            binning_factor=binning_factor,
            time_scale=time_scale,
            phase_cycles=phase_cycles,
            show_airmass=show_airmass,
            magnitude_system=magnitude_system,
        )
        return

    if isinstance(data, unc.core.NdarrayDistribution):
        if observation_times is None:
            raise ValueError(
                "prepare_plot_time_series requires observation_times for NdarrayDistribution"
            )
        lc = build_light_curves_table_from_flux(
            data,
            observation_times,
            filter_,
            object_names={int(object_id): object_name},
            observatory_location=observatory_location,
        )
        plot_from_light_curves_table(
            lc,
            int(object_id),
            filter_,
            output_dir,
            name_object=object_name,
            file_type=file_type_plots,
            subdirectory=subdirectory,
            transit_time=transit_time,
            period=_as_positive_period(period),
            binning_factor=binning_factor,
            time_scale=time_scale,
            phase_cycles=phase_cycles,
            show_airmass=show_airmass,
            magnitude_system=magnitude_system,
        )
        return

    raise TypeError(
        f"prepare_plot_time_series: unsupported data type {type(data)}"
    )


def plot_light_curve_from_epoch_native_ecsv(
    ecsv_path: str | Path,
    output_dir: str | Path,
    *,
    filter_: str,
    object_name: str,
    object_id: int | None = None,
    object_ra_deg: float | None = None,
    object_dec_deg: float | None = None,
    max_match_sep_arcsec: float = 5.0,
    quantity: LightCurveQuantity = "magnitude",
    calibration_rows: LightCurveCalibrationRows = "auto",
    epoch_meta: dict | None = None,
    epoch_meta_json: str | Path | None = None,
    binning_factor: float | None = None,
    transit_time: str | None = None,
    period: float | str | None = None,
    light_curve_save_format: str = "csv",
    file_type_plots: str = "pdf",
    subdirectory: str = "",
    pipeline_flux_normalization: bool = False,
    distribution_samples: int = 1000,
) -> None:
    """
    Read an epoch-native ``calibrated_magnitudes_*.ecsv`` and run
    :func:`prepare_plot_time_series`.

    Provide either ``object_id`` or both ``object_ra_deg`` and ``object_dec_deg``
    (ICRS degrees); the latter uses :func:`object_id_from_epoch_native_sky`.

    JDs require an ``observation_jd`` or ``jd`` column in the table (written by
    recent pipeline calibration saves) or a ``calibration_epoch_meta`` dict /
    JSON file (same structure as :attr:`ost_photometry.analyze.pipeline.context.AnalysisContext.calibration_epoch_meta`).

    If ``pipeline_flux_normalization`` is True, replicate the pipeline flux
    fallback (quasi ZP per epoch, then per-object median normalization) using
    flux columns in the ECSV — same math as
    :func:`ost_photometry.analyze.calibration.quasi_flux_calibration_image_series`
    followed by
    :func:`ost_photometry.analyze.calibration.flux_normalization_image_series`.
    Requires ``quantity="flux"`` and a rectangular epoch×source matrix fillable
    from the table (see :func:`epoch_native_flux_matrix_for_pipeline_normalization`).
    """
    from ...output_layout import results_dir, tables_dir
    from .. import calibration
    from .io import read_epoch_native_magnitudes

    tbl = read_epoch_native_magnitudes(ecsv_path)
    meta = epoch_meta
    if meta is None and epoch_meta_json is not None:
        meta = load_calibration_epoch_meta_json(epoch_meta_json)
    has_jd_col = "observation_jd" in tbl.colnames or "jd" in tbl.colnames
    if meta is None and not has_jd_col:
        raise ValueError(
            "No JD column (observation_jd / jd) in ECSV and no epoch_meta / "
            "epoch_meta_json provided. Re-run calibration with a current "
            "ost_photometry version (adds observation_jd) or pass --epoch-meta-json "
            "exported via save_calibration_epoch_meta_json."
        )

    out = str(output_dir)
    results_dir(out, "lightcurves")
    tables_dir(out)

    has_sky = object_ra_deg is not None and object_dec_deg is not None
    if has_sky and object_id is not None:
        terminal_output.print_to_terminal(
            "Both object_id and sky coordinates given; using sky match (ignoring object_id).",
            style_name="WARNING",
        )
    if has_sky:
        c = SkyCoord(object_ra_deg * u.deg, object_dec_deg * u.deg, frame="icrs")
        resolved_id = object_id_from_epoch_native_sky(
            tbl,
            c,
            max_sep=max_match_sep_arcsec * u.arcsec,
        )
    elif object_id is not None:
        resolved_id = int(object_id)
    else:
        raise ValueError(
            "Provide object_id or both object_ra_deg and object_dec_deg (degrees, ICRS)."
        )

    per = period
    if per == "?" or per is None:
        per = None
    elif isinstance(per, str):
        try:
            per = float(per)
        except ValueError:
            per = None

    if pipeline_flux_normalization:
        if quantity != "flux":
            raise ValueError(
                'pipeline_flux_normalization requires quantity="flux" '
                "(instrumental / extracted flux columns in the ECSV)."
            )
        flux_m, err_m, jds_arr, id_list = (
            epoch_native_flux_matrix_for_pipeline_normalization(
                tbl,
                filter_,
                meta,
                calibration_rows,
            )
        )
        col_matches = np.nonzero(id_list.astype(int) == int(resolved_id))[0]
        if col_matches.size == 0:
            raise ValueError(
                f"pipeline_flux_normalization: resolved id {resolved_id} not in "
                f"flux matrix column ids {id_list.tolist()}."
            )
        col_idx = int(col_matches[0])
        terminal_output.print_to_terminal(
            "Using ECSV flux matrix + pipeline-style quasi flux calibration and "
            "per-object normalization (same recipe as LightCurveStep flux fallback).",
            style_name="INFO",
        )
        quasi = calibration.quasi_flux_calibration_flux_arrays(
            flux_m,
            err_m,
            distribution_samples=distribution_samples,
        )
        plot_quantity = calibration.flux_normalization_flux_distribution(quasi)
        obs_times = Time(jds_arr, format="jd")
        prepare_plot_time_series(
            plot_quantity,
            obs_times,
            filter_,
            object_name,
            col_idx,
            out,
            binning_factor=binning_factor,
            transit_time=transit_time,
            period=per,
            light_curve_save_format=light_curve_save_format,
            subdirectory=subdirectory,
            file_type_plots=file_type_plots,
            calibration_type="simple",
            epoch_meta={},
            light_curve_quantity="flux",
        )
        return

    prepare_plot_time_series(
        tbl,
        None,
        filter_,
        object_name,
        int(resolved_id),
        out,
        binning_factor=binning_factor,
        transit_time=transit_time,
        period=per,
        light_curve_save_format=light_curve_save_format,
        subdirectory=subdirectory,
        file_type_plots=file_type_plots,
        epoch_meta=meta if meta is not None else {},
        light_curve_quantity=quantity,
        light_curve_calibration_rows=calibration_rows,
    )
