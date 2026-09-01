"""CMD table loading: epoch slice, magnitudes/colour, distance modulus, isochrones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

# Keep in sync with ``post_processing.schema.PHOTOMETRY_TABLE_SCHEMA_ID``.
_EPOCH_NATIVE_SCHEMA = "ost_photometry.epoch_native.v1"
_LEGACY_IMAGE_TAG = "0"


@dataclass(frozen=True)
class CmdSeries:
    """One colour–magnitude series for :class:`~ost_photometry.analyze.plots.cmds.MakeCMDs`."""

    filter_1: str
    filter_2: str
    color: np.ndarray
    magnitude_filter_2: np.ndarray
    color_err: np.ndarray | None = None
    magnitude_filter_2_err: np.ndarray | None = None


@dataclass(frozen=True)
class IsochronePlotConfig:
    """Fields forwarded to :meth:`MakeCMDs.plot_absolute_cmd`."""

    isochrones: str = ""
    isochrone_type: str = ""
    isochrone_column_type: dict | str = ""
    isochrone_column: dict | str = ""
    isochrone_keyword: str = ""
    isochrone_log_age: bool | str = ""
    isochrone_legend: bool | str = ""
    isochrone_set: str | None = None
    feh: float | None = None
    z: float | None = None
    y: float | None = None
    alpha_fe: float | None = None


def _is_unset(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "?"}:
        return True
    return False


def _column_array(tbl: Table, name: str) -> np.ndarray:
    return np.asarray(tbl[name], dtype=float)


def table_is_epoch_native_cmd(tbl: Table) -> bool:
    """True for pipeline ECSV (schema meta or ``epoch_id`` + ``mag_cal_*`` / ``mag_inst_*``)."""
    meta = getattr(tbl, "meta", None) or {}
    if meta.get("photometry_schema") == _EPOCH_NATIVE_SCHEMA:
        return True
    if "epoch_id" not in tbl.colnames:
        return False
    return any(
        n.startswith("mag_cal_") or n.startswith("mag_inst_") for n in tbl.colnames
    )


def slice_cmd_table_to_single_epoch(
    tbl: Table, epoch_id: str | None = None
) -> Table:
    """Keep one epoch from a multi-epoch vstack (default: first ``epoch_id``)."""
    if "epoch_id" not in tbl.colnames:
        return tbl
    ids = np.unique(np.asarray(tbl["epoch_id"]))
    if len(ids) <= 1:
        return tbl
    if epoch_id is None:
        epoch_id = str(ids[0])
    return tbl[np.asarray(tbl["epoch_id"]).astype(str) == str(epoch_id)]


def load_cmd_table(
    path: str | Path, *, epoch_id: str | None = None
) -> Table:
    """Read ASCII/ECSV photometry and slice to a single epoch when needed."""
    path = Path(path)
    fmt = "ascii.ecsv" if path.suffix.lower() == ".ecsv" else "ascii"
    tbl = Table.read(path, format=fmt)
    return slice_cmd_table_to_single_epoch(tbl, epoch_id=epoch_id)


def distance_modulus(
    m_m: str | float | None, distance_kpc: str | float | None
) -> float:
    """Distance modulus from ``m_M``, or from distance in kpc; ``0`` if both unset."""
    if not _is_unset(m_m):
        return float(m_m)
    if not _is_unset(distance_kpc):
        return float(5.0 * np.log10(float(distance_kpc) * 100.0))
    return 0.0


def _native_mag_column(tbl: Table, filter_: str) -> str:
    cal = f"mag_cal_{filter_}"
    inst = f"mag_inst_{filter_}"
    if cal in tbl.colnames:
        return cal
    if inst in tbl.colnames:
        return inst
    raise KeyError(
        f"Epoch-native table missing {cal!r} or {inst!r}; "
        f"available: {tbl.colnames!r}"
    )


def _native_err_column(tbl: Table, filter_: str) -> str:
    mag_col = _native_mag_column(tbl, filter_)
    if mag_col.startswith("mag_cal_"):
        err = f"err_cal_{filter_}"
    else:
        err = f"err_inst_{filter_}"
    if err not in tbl.colnames:
        raise KeyError(
            f"Epoch-native table missing {err!r} for error bars; "
            f"available: {tbl.colnames!r}"
        )
    return err


def _legacy_wide_mag_column(filter_: str, *, transformed: bool) -> str:
    kind = "transformed" if transformed else "simple"
    return f"{filter_} ({kind}, image={_LEGACY_IMAGE_TAG})"


def _legacy_wide_err_column(filter_: str, *, transformed: bool) -> str:
    kind = "transformed" if transformed else "simple"
    return f"{filter_}_err ({kind}, image={_LEGACY_IMAGE_TAG})"


def _student_mag_column(filter_: str) -> str:
    return f"{filter_} [mag]"


def _student_err_column(filter_: str) -> str:
    return f"{filter_}_err"


def _require_columns(tbl: Table, *names: str) -> None:
    missing = [n for n in names if n not in tbl.colnames]
    if missing:
        raise KeyError(
            f"CMD table missing {missing!r}; available: {tbl.colnames!r}"
        )


def _mag_column(
    tbl: Table,
    filter_: str,
    *,
    native: bool,
    magnitude_transformation: bool | None,
) -> str:
    if native:
        return _native_mag_column(tbl, filter_)

    student_mag = _student_mag_column(filter_)
    if magnitude_transformation is None:
        if student_mag in tbl.colnames:
            return student_mag
        transformed = _legacy_wide_mag_column(filter_, transformed=True)
        simple = _legacy_wide_mag_column(filter_, transformed=False)
        if transformed in tbl.colnames:
            return transformed
        if simple in tbl.colnames:
            return simple
        raise KeyError(
            f"CMD table has no magnitudes for filter {filter_!r} "
            f"(tried {student_mag!r}, legacy transformed/simple); "
            f"available: {tbl.colnames!r}"
        )

    return _legacy_wide_mag_column(
        filter_, transformed=bool(magnitude_transformation)
    )


def _err_column_for_mag(
    tbl: Table,
    filter_: str,
    mag_col: str,
    *,
    native: bool,
    magnitude_transformation: bool | None,
) -> str:
    if native:
        return _native_err_column(tbl, filter_)
    if mag_col == _student_mag_column(filter_):
        return _student_err_column(filter_)
    if magnitude_transformation is None:
        if mag_col == _legacy_wide_mag_column(filter_, transformed=True):
            return _legacy_wide_err_column(filter_, transformed=True)
        return _legacy_wide_err_column(filter_, transformed=False)
    return _legacy_wide_err_column(
        filter_, transformed=bool(magnitude_transformation)
    )


def cmd_series_from_table(
    tbl: Table,
    filter_1: str,
    filter_2: str,
    *,
    do_error_bars: bool = False,
    magnitude_transformation: bool | None = None,
    cali: dict[str, float] | None = None,
) -> CmdSeries:
    """Magnitudes, colour, and optional errors for one filter pair."""
    native = table_is_epoch_native_cmd(tbl)
    mag_col_1 = _mag_column(
        tbl,
        filter_1,
        native=native,
        magnitude_transformation=magnitude_transformation,
    )
    mag_col_2 = _mag_column(
        tbl,
        filter_2,
        native=native,
        magnitude_transformation=magnitude_transformation,
    )
    _require_columns(tbl, mag_col_1, mag_col_2)

    zp = cali or {}
    mag_1 = _column_array(tbl, mag_col_1) + float(zp.get(filter_1, 0.0))
    mag_2 = _column_array(tbl, mag_col_2) + float(zp.get(filter_2, 0.0))
    color = mag_1 - mag_2

    if not do_error_bars:
        return CmdSeries(filter_1, filter_2, color, mag_2)

    err_col_1 = _err_column_for_mag(
        tbl,
        filter_1,
        mag_col_1,
        native=native,
        magnitude_transformation=magnitude_transformation,
    )
    err_col_2 = _err_column_for_mag(
        tbl,
        filter_2,
        mag_col_2,
        native=native,
        magnitude_transformation=magnitude_transformation,
    )
    _require_columns(tbl, err_col_1, err_col_2)
    err_1 = _column_array(tbl, err_col_1)
    err_2 = _column_array(tbl, err_col_2)
    color_err = np.sqrt(np.square(err_1) + np.square(err_2))
    return CmdSeries(
        filter_1,
        filter_2,
        color,
        mag_2,
        color_err=color_err,
        magnitude_filter_2_err=err_2,
    )


def mask_cmd_series(
    series: CmdSeries,
    *,
    max_photometric_err: float | None = None,
) -> CmdSeries:
    """Drop non-finite CMD points and, optionally, large photometric errors."""
    color = np.asarray(series.color, dtype=float)
    mag = np.asarray(series.magnitude_filter_2, dtype=float)
    keep = np.isfinite(color) & np.isfinite(mag)
    color_err = (
        None
        if series.color_err is None
        else np.asarray(series.color_err, dtype=float)
    )
    mag_err = (
        None
        if series.magnitude_filter_2_err is None
        else np.asarray(series.magnitude_filter_2_err, dtype=float)
    )
    if color_err is not None:
        keep &= np.isfinite(color_err)
    if mag_err is not None:
        keep &= np.isfinite(mag_err)

    cut: float | None
    try:
        cut = None if max_photometric_err is None else float(max_photometric_err)
    except (TypeError, ValueError):
        cut = None
    if cut is not None and np.isfinite(cut) and cut > 0:
        if color_err is not None:
            keep &= color_err <= cut
        if mag_err is not None:
            keep &= mag_err <= cut

    return CmdSeries(
        series.filter_1,
        series.filter_2,
        color[keep],
        mag[keep],
        color_err=None if color_err is None else color_err[keep],
        magnitude_filter_2_err=None if mag_err is None else mag_err[keep],
    )


def fiducial_fit_sigma(
    phot_err: np.ndarray | None,
    scatter: float,
    n: int,
) -> float:
    """1-sigma for a binned CMD fiducial: photometric IVW, else scatter/√N."""
    phot_term = 0.0
    if phot_err is not None:
        err = np.asarray(phot_err, dtype=float)
        err = err[np.isfinite(err) & (err > 0)]
        if err.size:
            phot_term = float(np.sqrt(1.0 / np.sum(1.0 / np.square(err))))
    if phot_term > 0:
        return phot_term
    count = max(int(n), 1)
    if np.isfinite(scatter) and scatter > 0:
        return float(scatter) / np.sqrt(count)
    return 1.0


def weighted_chi_square(
    residual: np.ndarray,
    sigma: np.ndarray | None = None,
) -> float:
    """Σ (δ/σ)², or unweighted Σ δ² if ``sigma`` is missing or unusable."""
    delta = np.asarray(residual, dtype=float)
    if sigma is None:
        finite = np.isfinite(delta)
        return float(np.square(delta[finite]).sum()) if np.any(finite) else 0.0
    sig = np.asarray(sigma, dtype=float)
    good = np.isfinite(delta) & np.isfinite(sig) & (sig > 0)
    if not np.any(good):
        finite = np.isfinite(delta)
        return float(np.square(delta[finite]).sum()) if np.any(finite) else 0.0
    return float(np.sum(np.square(delta[good] / sig[good])))


def _optional_yaml_str(data: dict, *keys: str) -> str | None:
    for key in keys:
        if key not in data or data[key] is None:
            continue
        value = str(data[key]).strip()
        if value in {"", "?"}:
            continue
        return value
    return None


def _optional_yaml_float(data: dict, *keys: str) -> float | None:
    for key in keys:
        if key not in data or data[key] is None:
            continue
        raw = data[key]
        if isinstance(raw, str) and raw.strip() in {"", "?"}:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        return value
    return None


def _fmt_annotation_number(value: object, *, digits: int = 4) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    text = f"{number:.{digits}g}"
    return text


def format_isochrone_annotation(
    *,
    isochrone_set: str | None = None,
    feh: float | None = None,
    z: float | None = None,
    y: float | None = None,
    alpha_fe: float | None = None,
    e_b_v: float | None = None,
    rv: float | None = None,
    m_m: float | None = None,
    apply_corrections_to: str = "observation",
    best_age: float | None = None,
    best_age_unit: str | None = None,
    chi_square: float | None = None,
) -> str:
    """Compact matplotlib mathtext block for the CMD isochrone info box."""
    lines: list[str] = []
    set_name = None if _is_unset(isochrone_set) else str(isochrone_set).strip()
    if set_name:
        lines.append(set_name)

    composition: list[str] = []
    feh_txt = _fmt_annotation_number(feh, digits=3)
    if feh_txt is not None:
        composition.append(rf"$[\mathrm{{Fe}}/\mathrm{{H}}]={feh_txt}$")
    z_txt = _fmt_annotation_number(z, digits=4)
    if z_txt is not None:
        composition.append(rf"$Z={z_txt}$")
    y_txt = _fmt_annotation_number(y, digits=4)
    if y_txt is not None:
        composition.append(rf"$Y={y_txt}$")
    alpha_txt = _fmt_annotation_number(alpha_fe, digits=3)
    if alpha_txt is not None:
        composition.append(rf"$[\alpha/\mathrm{{Fe}}]={alpha_txt}$")
    if composition:
        lines.append(", ".join(composition))

    cluster: list[str] = []
    ebv_txt = _fmt_annotation_number(e_b_v, digits=3)
    if ebv_txt is not None:
        cluster.append(rf"$E(B-V)={ebv_txt}$")
    rv_txt = _fmt_annotation_number(rv, digits=3)
    if rv_txt is not None:
        cluster.append(rf"$R_V={rv_txt}$")
    mm_txt = _fmt_annotation_number(m_m, digits=4)
    if mm_txt is not None:
        cluster.append(rf"$(m-M)={mm_txt}$")
    if cluster:
        lines.append(", ".join(cluster))

    target = str(apply_corrections_to).strip().lower()
    if target in ("isochrone", "isochrones"):
        lines.append("Corrections: isochrones")
    elif target in ("observation", "data", "stars"):
        lines.append("Corrections: stars")

    age_txt = _fmt_annotation_number(best_age, digits=3)
    if age_txt is not None:
        unit = "" if _is_unset(best_age_unit) else f" {best_age_unit}"
        fit = rf"Best age: ${age_txt}${unit}"
        chi_txt = _fmt_annotation_number(chi_square, digits=3)
        if chi_txt is not None:
            fit += rf", $\chi^2={chi_txt}$"
        lines.append(fit)

    return "\n".join(lines)


def _select_isochrone_grid(data: dict) -> tuple[dict, str | None]:
    """Return ``(grid_entry, name)``; ``({}, None)`` if the YAML has no catalog."""
    grids = data.get("grids")
    if grids is None:
        return {}, None
    if not isinstance(grids, dict) or not grids:
        raise ValueError("isochrone YAML 'grids' must be a non-empty mapping")
    use = data.get("use")
    if _is_unset(use):
        names = ", ".join(str(name) for name in grids)
        raise ValueError(
            "isochrone YAML has 'grids' but no 'use' key "
            f"(available: {names})"
        )
    if use in grids:
        selected = grids[use]
        name = str(use)
    else:
        name = str(use).strip()
        if name not in grids:
            names = ", ".join(str(key) for key in grids)
            raise ValueError(
                f"Unknown isochrone grid {name!r}; available: {names}"
            )
        selected = grids[name]
    if not isinstance(selected, dict):
        raise ValueError(
            f"isochrone grid {name!r} must be a mapping of path and metadata"
        )
    return selected, name


def _merged_optional_str(selected: dict, data: dict, *keys: str) -> str | None:
    return _optional_yaml_str(selected, *keys) or _optional_yaml_str(data, *keys)


def _merged_optional_float(selected: dict, data: dict, *keys: str) -> float | None:
    value = _optional_yaml_float(selected, *keys)
    if value is not None:
        return value
    return _optional_yaml_float(data, *keys)


def load_isochrone_config(
    path: str | Path | None,
    filter_list: list[str],
) -> IsochronePlotConfig:
    """Read an isochrone YAML; empty path or missing file yields blank fields.

    Optional ``grids`` / ``use`` select one catalog entry. That entry supplies
    ``isochrones`` and may override composition metadata; shared keys stay at
    file level. Without ``grids`` the top-level ``isochrones`` path is used.
    """
    if _is_unset(path):
        return IsochronePlotConfig()

    import yaml

    try:
        with Path(path).open() as file:
            data = yaml.safe_load(file)
    except (yaml.YAMLError, FileNotFoundError, OSError, TypeError):
        return IsochronePlotConfig()
    if not isinstance(data, dict) or not data:
        return IsochronePlotConfig()

    column_type = data["isochrone_column_type"]
    column = data["isochrone_column"]
    for filter_ in filter_list:
        if filter_ not in column_type:
            raise ValueError(
                f"No isochrone_column_type entry for filter {filter_!r}"
            )
        if filter_ not in column:
            raise ValueError(f"No isochrone_column entry for filter {filter_!r}")

    selected, grid_name = _select_isochrone_grid(data)
    isochrones = _merged_optional_str(selected, data, "isochrones") or ""
    if grid_name is not None and not isochrones:
        raise ValueError(f"isochrone grid {grid_name!r} has no 'isochrones' path")

    return IsochronePlotConfig(
        isochrones=isochrones,
        isochrone_type=data["isochrone_type"],
        isochrone_column_type=column_type,
        isochrone_column=column,
        isochrone_keyword=data["isochrone_keyword"],
        isochrone_log_age=data["isochrone_log_age"],
        isochrone_legend=data["isochrone_legend"],
        isochrone_set=_merged_optional_str(selected, data, "isochrone_set"),
        feh=_merged_optional_float(selected, data, "FeH", "feh"),
        z=_merged_optional_float(selected, data, "Z", "z"),
        y=_merged_optional_float(selected, data, "Y", "y"),
        alpha_fe=_merged_optional_float(selected, data, "alpha_Fe", "alpha_fe"),
    )


__all__ = [
    "CmdSeries",
    "IsochronePlotConfig",
    "cmd_series_from_table",
    "distance_modulus",
    "fiducial_fit_sigma",
    "format_isochrone_annotation",
    "load_cmd_table",
    "load_isochrone_config",
    "mask_cmd_series",
    "weighted_chi_square",
    "slice_cmd_table_to_single_epoch",
    "table_is_epoch_native_cmd",
]
