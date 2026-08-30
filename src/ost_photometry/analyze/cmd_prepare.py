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


def load_isochrone_config(
    path: str | Path | None,
    filter_list: list[str],
) -> IsochronePlotConfig:
    """Read an isochrone YAML; empty path or missing file yields blank fields."""
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
    return IsochronePlotConfig(
        isochrones=data.get("isochrones", ""),
        isochrone_type=data["isochrone_type"],
        isochrone_column_type=column_type,
        isochrone_column=column,
        isochrone_keyword=data["isochrone_keyword"],
        isochrone_log_age=data["isochrone_log_age"],
        isochrone_legend=data["isochrone_legend"],
    )


__all__ = [
    "CmdSeries",
    "IsochronePlotConfig",
    "cmd_series_from_table",
    "distance_modulus",
    "load_cmd_table",
    "load_isochrone_config",
    "slice_cmd_table_to_single_epoch",
    "table_is_epoch_native_cmd",
]
