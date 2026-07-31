"""Light curve preparation from calibrated tables and time series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.table import Table, vstack
from astropy import uncertainty as unc
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.timeseries import TimeSeries

from ... import terminal_output


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
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


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


def _light_curve_plot_filename_suffix(lc_suffix: str, ts_data_col: str) -> str:
    """
    :func:`plots.light_curve_jd` / ``light_curve_fold`` build
    ``..._{ts_data_col}{file_name_suffix}``. For flux, ``ts_data_col`` is already
    ``flux_<filter>``; strip a trailing ``_flux`` from the suffix to avoid
    ``flux_V_flux`` in the filename.
    """
    if not ts_data_col.startswith("flux_"):
        return lc_suffix
    if lc_suffix.endswith("_flux"):
        return lc_suffix[: -len("_flux")]
    return lc_suffix


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


def prepare_plot_time_series(
        data: unc.core.NdarrayDistribution | Table,
        observation_times: Time | None,
        filter_: str, object_name: str, object_id: int, output_dir: str,
        binning_factor: float | None = None, transit_time: str | None = None,
        period: float | None = None, file_name_suffix: str = '',
        light_curve_save_format: str = 'csv', subdirectory: str = '',
        file_type_plots: str = 'pdf', calibration_type: str = 'transformed',
        epoch_meta: dict | None = None,
        light_curve_quantity: LightCurveQuantity = "magnitude",
        light_curve_calibration_rows: LightCurveCalibrationRows = "auto",
        magnitude_system: str | None = None,
        ) -> None:
    """
    Prepares, plot, and saves a time series for the object with the
    object ID: ``object_id``

    Parameters
    ----------
    data
        Epoch-native ``Table`` (``mag_cal_*`` or ``mag_inst_*``, ``epoch_id``, ``id``) or
        ``NdarrayDistribution`` (legacy flux-style light curves).

    observation_times
        Required for ``NdarrayDistribution``. For epoch-native tables, pass ``None``
        and supply ``epoch_meta``.

    filter_
        Filter in which the magnitudes are taken

    object_name
        Name of the object

    object_id
        Correlated source ``id`` (epoch-native) or column index (distributions).

    epoch_meta
        Per-epoch JD map (e.g. ``context.calibration_epoch_meta``). Required for tables
        unless the table has an ``observation_jd`` or ``jd`` column.

    light_curve_quantity
        ``"magnitude"`` (``mag_cal_*`` / ``mag_inst_*``) or ``"flux"``
        (``flux_inst_*`` or ``flux_*``).

    light_curve_calibration_rows
        For legacy tables with both transformed and ``*_simple`` epochs: ``"auto"``
        drops simple rows when transformed exists at the same JD; ``"transformed"``
        or ``"simple"`` keeps only those rows.
    """
    from .. import plots

    if object_id is None:
        terminal_output.print_to_terminal(
            f"ID of object {object_name} is None. Failed to create "
            f"light curve.",
            style_name='WARNING',
        )
        return

    if isinstance(data, Table):
        if not is_epoch_native_photometry_table(
            data, filter_, quantity=light_curve_quantity
        ):
            terminal_output.print_to_terminal(
                f"Light curve for {object_name!r}: table is not epoch-native for "
                f"quantity={light_curve_quantity!r} (filter {filter_!r}; need mag or "
                f"flux columns, plus epoch_id, id). Skipping.",
                style_name="WARNING",
            )
            return
        need_meta = "observation_jd" not in data.colnames and "jd" not in data.colnames
        if need_meta and not epoch_meta:
            terminal_output.print_to_terminal(
                f"Epoch-native table for filter {filter_!r}: need ``observation_jd`` "
                f"or ``jd`` column, or ``epoch_meta`` (e.g. from pipeline "
                f"``calibration_epoch_meta`` / JSON). Skipping {object_name!r}.",
                style_name="WARNING",
            )
            return
        y_values, y_errs, obs_times = prepare_time_series_epoch_native(
            data,
            filter_,
            int(object_id),
            epoch_meta,
            quantity=light_curve_quantity,
            calibration_rows=light_curve_calibration_rows,
        )
    elif isinstance(data, unc.core.NdarrayDistribution):
        if observation_times is None:
            raise ValueError(
                "prepare_plot_time_series requires observation_times for NdarrayDistribution"
            )
        y_values, y_errs = prepare_time_series_data(
            data,
            filter_,
            object_id,
            calibration_type=calibration_type,
        )
        obs_times = observation_times
    else:
        raise TypeError(
            f"prepare_plot_time_series: unsupported data type {type(data)}"
        )

    if y_values.size == 0:
        terminal_output.print_to_terminal(
            f"No photometry points for light curve: {object_name!r}, filter {filter_!r}.",
            style_name="WARNING",
        )
        return

    y_style = "flux" if light_curve_quantity == "flux" else "magnitude"
    lc_suffix = file_name_suffix
    if light_curve_calibration_rows != "auto":
        lc_suffix = f"{lc_suffix}_{light_curve_calibration_rows}"
    if light_curve_quantity == "flux":
        lc_suffix = f"{lc_suffix}_flux"

    if light_curve_quantity == "flux":
        time_series = mk_time_series_flux(
            obs_times,
            y_values,
            y_errs,
            filter_,
        )
        ts_data_col = f"flux_{filter_}"
        ts_err_col = f"{ts_data_col}_err"
    else:
        time_series = mk_time_series(
            obs_times,
            y_values,
            y_errs,
            filter_,
        )
        ts_data_col = filter_
        ts_err_col = f"{filter_}_err"

    plot_suffix = _light_curve_plot_filename_suffix(lc_suffix, ts_data_col)

    #   Write time series
    if light_curve_save_format not in ['dat', 'csv']:
        terminal_output.print_to_terminal(
            f"Format to save the light curve not known. Assume csv. "
            f"The provided format was: {light_curve_save_format}",
            style_name='WARNING',
        )

    if light_curve_save_format == 'dat':
        time_series.write(
            f'{output_dir}/tables/light_curve_{object_name}_{filter_}'
            f'{lc_suffix}.dat',
            format='ascii',
            overwrite=True,
        )
    else:
        time_series.write(
            f'{output_dir}/tables/light_curve_{object_name}_{filter_}'
            f'{lc_suffix}.csv',
            format='ascii.csv',
            overwrite=True,
        )

    #   Plot light curve over JD
    from .magnitude_systems import table_magnitude_system

    mag_sys = magnitude_system
    if mag_sys is None and isinstance(data, Table):
        mag_sys = table_magnitude_system(data)
    if mag_sys is None:
        mag_sys = "vega"

    plots.light_curve_jd(
        time_series,
        ts_data_col,
        ts_err_col,
        output_dir,
        name_object=object_name,
        file_name_suffix=plot_suffix,
        subdirectory=subdirectory,
        file_type=file_type_plots,
        y_axis_style=y_style,
        magnitude_system=mag_sys,
    )

    #   Plot the light curve folded on the period
    if (transit_time is not None and transit_time != '?'
            and period is not None and period != '?' and period > 0.):
        plots.light_curve_fold(
            time_series,
            ts_data_col,
            ts_err_col,
            output_dir,
            transit_time,
            period,
            binning_factor=binning_factor,
            name_object=object_name,
            file_name_suffix=plot_suffix,
            subdirectory=subdirectory,
            file_type=file_type_plots,
            y_axis_style=y_style,
            magnitude_system=mag_sys,
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
    from .io import read_epoch_native_magnitudes

    from .. import calibration
    from ... import checks

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
    checks.check_output_directories(f"{out}/lightcurve", f"{out}/tables")

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
