"""
mk_calib field transformation via :class:`CalibrationEngine` (derive-transform path).

Produces ``trans_para_*.dat`` (and JSON sidecar) for second-order extinction campaigns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table, vstack

if TYPE_CHECKING:
    from .. import analyze
    from ..pipeline.config import PipelineConfig
    from .derive_transform import DeriveTransformFit


# Legacy second-order column names per filter (color index f0–f1).
LEGACY_C_COLUMNS: dict[str, list[str]] = {
    "U": ["Cuuv", "Cvuv"],
    "B": ["Cbbv", "Cvbv"],
    "V": ["Cbbv", "Cvbv"],
    "R": ["Crvr", "Cvvr"],
}

LEGACY_C_ERR_COLUMNS: dict[str, list[str]] = {
    filt: [f"{c}_err" for c in cols] for filt, cols in LEGACY_C_COLUMNS.items()
}


@dataclass
class TransformCoefficient:
    """One magnitude-transform coefficient (mk_calib ``C*`` column equivalent)."""

    column: str
    filter: str
    color_index: tuple[str, str]
    c: float
    c_err: float


@dataclass
class FieldTransformationRecord:
    """Transformation summary for one calibration field / cluster."""

    name: str
    filter_pair: tuple[str, str]
    jd: float
    airmass: dict[str, float] = field(default_factory=dict)
    coefficients: list[TransformCoefficient] = field(default_factory=list)
    n_comparison_stars: int = 0
    engine: str = "CalibrationEngine/derive_transform"

    def coefficient_by_column(self, column: str) -> TransformCoefficient | None:
        for coeff in self.coefficients:
            if coeff.column == column:
                return coeff
        return None


def _legacy_column_name(key_filter: str, f0: str, f1: str) -> str:
    return f"C{key_filter.lower()}{f0.lower()}{f1.lower()}"


def _ensemble_epochs(epochs: dict[str, Table]) -> dict[str, Table]:
    if not epochs:
        return {}
    if len(epochs) == 1:
        return dict(epochs)
    return {"ensemble": vstack(list(epochs.values()))}


def _slopes_with_errors(
    table: Table,
    filters: list[str],
    comparison_mask: np.ndarray,
    *,
    apply_weights: bool,
) -> dict[str, tuple[float, float]]:
    """Per-filter slope and stderr for (m_std - m_inst) vs catalog color."""
    from .transform import weighted_linear_fit

    f0, f1 = filters[0], filters[1]
    mask = np.asarray(comparison_mask, dtype=bool)
    m_std_0 = np.asarray(table[f"mag_std_{f0}"], dtype=float)[mask]
    m_std_1 = np.asarray(table[f"mag_std_{f1}"], dtype=float)[mask]
    m_inst_0 = np.asarray(table[f"mag_{f0}"], dtype=float)[mask]
    m_inst_1 = np.asarray(table[f"mag_{f1}"], dtype=float)[mask]
    color = m_std_0 - m_std_1
    diff_0 = m_std_0 - m_inst_0
    diff_1 = m_std_1 - m_inst_1

    out: dict[str, tuple[float, float]] = {}
    for filt, delta in ((f0, diff_0), (f1, diff_1)):
        err_col = f"err_{filt}"
        if apply_weights and err_col in table.colnames:
            sigma = np.asarray(table[err_col], dtype=float)[mask]
            weights = 1.0 / np.maximum(sigma**2, 1e-12)
        else:
            weights = np.ones(len(color), dtype=float)
        slope, _zp, slope_err, _, _ = weighted_linear_fit(color, delta, weights)
        out[filt] = (float(slope), float(slope_err))
    return out


def record_from_derive_fit(
    *,
    name: str,
    filter_pair: list[str],
    jd: float,
    airmass: dict[str, float],
    derive_fit: DeriveTransformFit,
    slope_err: dict[str, tuple[float, float]],
) -> FieldTransformationRecord:
    f0, f1 = filter_pair[0], filter_pair[1]
    raw_slopes = {f0: derive_fit.c_slope_f0, f1: derive_fit.c_slope_f1}
    coeffs: list[TransformCoefficient] = []
    seen_cols: set[str] = set()

    def _add(key_filter: str, column: str) -> None:
        if column in seen_cols:
            return
        c_val, c_err = slope_err.get(
            key_filter, (raw_slopes[key_filter], 0.0)
        )
        coeffs.append(
            TransformCoefficient(
                column=column,
                filter=key_filter,
                color_index=(f0, f1),
                c=c_val,
                c_err=c_err,
            )
        )
        seen_cols.add(column)

    for key_filter in filter_pair:
        _add(key_filter, _legacy_column_name(key_filter, f0, f1))

    return FieldTransformationRecord(
        name=name,
        filter_pair=(f0, f1),
        jd=jd,
        airmass=dict(airmass),
        coefficients=coeffs,
        n_comparison_stars=derive_fit.n_stars_used,
    )


def calibrate_mk_calib_filter_pair(
    observation: analyze.Observation,
    filter_pair: list[str],
    config: PipelineConfig,
    *,
    apply_weights: bool = True,
) -> FieldTransformationRecord:
    """
    Inter-correlate a two-filter pair and fit transformation via CalibrationEngine.

    Returns a :class:`FieldTransformationRecord` for second-order extinction analysis.
    """
    from ..correlate.inter import correlate_image_series
    from ..correlate.protection import resolve_protected_object_ids_for_inter
    from ..pipeline.bridge import observation_to_calibration_epochs
    from ..pipeline.context import AnalysisContext
    from ..pipeline.steps.calibration import _crossmatch_epochs
    from .derive_transform import fit_epoch_derive_transform
    from .engine import CalibrationEngine

    if len(filter_pair) != 2:
        raise ValueError("mk_calib calibration requires exactly two filters per pair")

    protected_ids = resolve_protected_object_ids_for_inter(
        observation,
        filter_pair,
        observation.image_series_dict,
        config,
    )
    correlate_image_series(
        observation,
        filter_pair,
        max_pixel_between_objects=config.max_pixel_between_objects,
        ooi_correlation_strategy=config.ooi_correlation_strategy,
        cross_identification_limit=config.cross_identification_limit,
        n_allowed_non_detections_object=config.n_allowed_non_detections_object,
        expected_bad_image_fraction=config.expected_bad_image_fraction,
        protected_object_ids=protected_ids,
        correlation_method=config.correlation_method,
        separation_limit=config.separation_limit,
        file_type_plots=config.file_type_plots,
        verbose=config.verbose,
    )

    context = AnalysisContext(
        image_series_dict={
            f: observation.image_series_dict[f] for f in filter_pair
        },
        filter_list=list(filter_pair),
        output_dir=str(observation.image_series_dict[filter_pair[0]].out_path),
        observation=observation,
    )
    observation_to_calibration_epochs(context, config)
    epochs = _crossmatch_epochs(dict(context.calibration_epochs), context, config)
    epochs = _ensemble_epochs(epochs)
    if not epochs:
        raise RuntimeError(
            f"No calibration epochs for filter pair {filter_pair}; "
            "check exposure pairing and correlation."
        )

    epoch_id, table = next(iter(epochs.items()))
    fitted = fit_epoch_derive_transform(
        table,
        epoch_id,
        filter_pair,
        color_indices=config.color_indices,
        min_comparisons=5,
        sigma_clip=config.fit_sigma_clip,
        zp_subsample_statistic=config.zp_subsample_statistic,
        distribution_samples=config.distribution_samples,
    )
    if fitted is None:
        raise RuntimeError(
            f"derive_transform fit failed for filter pair {filter_pair}"
        )
    _result, derive_fit = fitted

    # Also run engine for QC plots / consistency
    cal_config = config
    CalibrationEngine.fit(
        epochs,
        cal_config,
        filter_pair,
        color_indices=config.color_indices,
        output_dir=context.output_dir,
        file_type=config.file_type_plots,
    )

    slope_err = _slopes_with_errors(
        table,
        filter_pair,
        derive_fit.comparison_mask,
        apply_weights=apply_weights,
    )

    name = getattr(observation, "object_names", [""])[0] if getattr(
        observation, "object_names", None
    ) else ""
    if not name:
        name = Path(context.output_dir).name

    airmass = {
        f: float(observation.image_series_dict[f].median_air_mass())
        for f in filter_pair
    }
    jd = float(observation.image_series_dict[filter_pair[0]].median_observation_time())

    return record_from_derive_fit(
        name=str(name),
        filter_pair=filter_pair,
        jd=jd,
        airmass=airmass,
        derive_fit=derive_fit,
        slope_err=slope_err,
    )


def merge_field_transformation_records(
    records: list[FieldTransformationRecord],
) -> FieldTransformationRecord:
    """Merge per-pair records into one field table (e.g. B–V and V–R)."""
    if not records:
        raise ValueError("No records to merge")
    base = records[0]
    airmass = dict(base.airmass)
    coeffs: list[TransformCoefficient] = []
    seen: set[str] = set()
    n_stars = 0
    for rec in records:
        airmass.update(rec.airmass)
        n_stars = max(n_stars, rec.n_comparison_stars)
        for coeff in rec.coefficients:
            if coeff.column in seen:
                continue
            coeffs.append(coeff)
            seen.add(coeff.column)
    return FieldTransformationRecord(
        name=base.name,
        filter_pair=base.filter_pair,
        jd=base.jd,
        airmass=airmass,
        coefficients=coeffs,
        n_comparison_stars=n_stars,
    )


def write_field_transformation_json(
    record: FieldTransformationRecord,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["filter_pair"] = list(record.filter_pair)
    for coeff in payload["coefficients"]:
        coeff["color_index"] = list(coeff["color_index"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def write_trans_para_table(
    record: FieldTransformationRecord,
    path: str | Path,
) -> Path:
    """Write ``trans_para_*.dat`` ASCII table (mk_calib-compatible columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = Table()
    tbl["name"] = [record.name]
    tbl["jd"] = [record.jd]
    for filt, am in record.airmass.items():
        tbl[f"airmass_{filt}"] = [am]
    for coeff in record.coefficients:
        tbl[coeff.column] = [coeff.c]
        tbl[f"{coeff.column}_err"] = [coeff.c_err]
    tbl.write(path, format="ascii", overwrite=True)
    return path


def load_field_transformation_record(path: str | Path) -> FieldTransformationRecord:
    path = Path(path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        coeffs = [
            TransformCoefficient(
                column=c["column"],
                filter=c["filter"],
                color_index=tuple(c["color_index"]),
                c=float(c["c"]),
                c_err=float(c["c_err"]),
            )
            for c in data["coefficients"]
        ]
        return FieldTransformationRecord(
            name=data["name"],
            filter_pair=tuple(data["filter_pair"]),
            jd=float(data["jd"]),
            airmass={k: float(v) for k, v in data.get("airmass", {}).items()},
            coefficients=coeffs,
            n_comparison_stars=int(data.get("n_comparison_stars", 0)),
            engine=data.get("engine", "CalibrationEngine/derive_transform"),
        )

    # Legacy ASCII trans_para
    tbl = Table.read(path, format="ascii")
    name = str(tbl["name"][0])
    jd = float(tbl["jd"][0]) if "jd" in tbl.colnames else 0.0
    airmass: dict[str, float] = {}
    coeffs: list[TransformCoefficient] = []
    filter_pair: tuple[str, str] | None = None
    for col in tbl.colnames:
        if col.startswith("airmass_"):
            filt = col.split("_", 1)[1]
            airmass[filt] = float(tbl[col][0])
        if col.startswith("C") and not col.endswith("_err"):
            # Parse Cbbv -> filter B, pair b,v
            body = col[1:]
            if len(body) >= 3:
                key_f = body[0].upper()
                f0 = body[1].upper()
                f1 = body[2].upper()
                filter_pair = (f0, f1)
                err_col = f"{col}_err"
                c_err = float(tbl[err_col][0]) if err_col in tbl.colnames else 0.0
                coeffs.append(
                    TransformCoefficient(
                        column=col,
                        filter=key_f,
                        color_index=(f0, f1),
                        c=float(tbl[col][0]),
                        c_err=c_err,
                    )
                )
    if filter_pair is None:
        filter_pair = ("B", "V")
    return FieldTransformationRecord(
        name=name,
        filter_pair=filter_pair,
        jd=jd,
        airmass=airmass,
        coefficients=coeffs,
    )


def load_field_transformation_records(
    paths: list[str | Path],
) -> list[FieldTransformationRecord]:
    return [load_field_transformation_record(p) for p in paths]


__all__ = [
    "FieldTransformationRecord",
    "TransformCoefficient",
    "LEGACY_C_COLUMNS",
    "LEGACY_C_ERR_COLUMNS",
    "calibrate_mk_calib_filter_pair",
    "load_field_transformation_record",
    "load_field_transformation_records",
    "merge_field_transformation_records",
    "record_from_derive_fit",
    "write_field_transformation_json",
    "write_trans_para_table",
]
