"""CalibrationEngine: unified fit/apply API."""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from astropy.table import Table

from ..differential_photometry import DifferentialPhotometer, PhotometryCalibrator
from .apply import apply_calibration_epochs
from .backends import linear as linear_backend
from .backends import median as median_backend
from .result import CalibrationResult, TransformationCoefficients
from .zp import comparison_mask_from_std_columns, zp_subsample_statistic

if TYPE_CHECKING:
    from ..pipeline.config import PipelineConfig


class CalibrationEngine:
    """Fit and apply calibration using strategy backends."""

    @staticmethod
    def fit(
        epochs: Dict[str, Table],
        config: "PipelineConfig",
        filters: List[str],
        *,
        calibrator: Optional[PhotometryCalibrator] = None,
        color_indices: dict[str, tuple[str, str]] | None = None,
        output_dir: str | None = None,
        file_type: str = "pdf",
        calibration_summary_x_jd: dict[str, float] | None = None,
    ) -> Dict[str, CalibrationResult]:
        strategy = config.calibration_strategy
        if strategy == "median_zp":
            results = median_backend.fit_epochs(
                epochs,
                filters,
                config,
                color_indices=color_indices,
            )
        else:
            if config.derive_transform_from_data:
                from .backends import derive_transform as derive_transform_backend

                results = derive_transform_backend.fit_epochs(
                    epochs,
                    filters,
                    config,
                    color_indices=color_indices,
                    output_dir=output_dir,
                    file_type=file_type,
                )
                if results is None:
                    cal = calibrator or linear_backend.build_calibrator(
                        config,
                        color_indices=color_indices,
                    )
                    cal.epochs.clear()
                    for epoch_id, tbl in epochs.items():
                        cal.epochs[epoch_id] = tbl
                    results = linear_backend.fit_epochs(
                        cal,
                        epochs,
                        filters,
                        config,
                        output_dir=output_dir,
                        file_type=file_type,
                        calibration_summary_x_jd=calibration_summary_x_jd,
                    )
            else:
                cal = calibrator or linear_backend.build_calibrator(
                    config,
                    color_indices=color_indices,
                )
                cal.epochs.clear()
                results = linear_backend.fit_epochs(
                    cal,
                    epochs,
                    filters,
                    config,
                    output_dir=output_dir,
                    file_type=file_type,
                    calibration_summary_x_jd=calibration_summary_x_jd,
                )
        if config.zp_subsample_statistic and strategy == "median_zp":
            for epoch_id, result in results.items():
                tbl = epochs.get(epoch_id)
                if tbl is None:
                    continue
                for f in result.transformation:
                    inst_col, std_col = f"mag_{f}", f"mag_std_{f}"
                    if inst_col not in tbl.colnames or std_col not in tbl.colnames:
                        continue
                    mask = comparison_mask_from_std_columns(tbl, [f])
                    if not np.any(mask):
                        continue
                    stats = zp_subsample_statistic(
                        np.asarray(tbl[std_col][mask], dtype=float),
                        np.asarray(tbl[inst_col][mask], dtype=float),
                        n_subsamples=config.distribution_samples,
                    )
                    result.notes = (
                        f"{f} subsample_median={stats['median']:.4f} "
                        f"spread={stats['subsample_spread']:.4f}"
                    )
        return results

    @staticmethod
    def apply(
        epochs: Dict[str, Table],
        results: Dict[str, CalibrationResult],
        filters: List[str],
        photometer: Optional[DifferentialPhotometer] = None,
        *,
        output_prefix: str = "mag_cal_",
    ) -> Table:
        return apply_calibration_epochs(
            epochs,
            results,
            filters,
            photometer=photometer,
            output_prefix=output_prefix,
        )


def prepare_calibration_check_plots(
    output_dir: str,
    epochs: Dict[str, Table],
    results: Dict[str, CalibrationResult],
    filters: List[str],
    *,
    file_type: str = "pdf",
    filename_prefix: str = "calibration",
    title_prefix: str | None = None,
    fit_masks: Dict[str, np.ndarray] | None = None,
) -> None:
    """Write transformation diagnostic plots under ``output_dir/calibration/``.

    Parameters
    ----------
    fit_masks
        Optional ``{epoch_id: bool array}`` of stars actually used in the fit.
        When given, gray vs blue points reflect that mask (clipped outliers
        appear as excluded). Otherwise all finite comparison stars are shown
        as used.
    """
    from .. import plots
    from .zp import comparison_mask_from_std_columns

    for epoch_id, result in results.items():
        if not result.transformation:
            continue
        tbl = epochs.get(epoch_id)
        if tbl is None:
            continue
        comp_mask = comparison_mask_from_std_columns(tbl, filters)
        plot_data: dict = {}
        for f in filters:
            if f not in result.transformation:
                continue
            inst_col, std_col = f"mag_{f}", f"mag_std_{f}"
            if inst_col not in tbl.colnames or std_col not in tbl.colnames:
                continue
            m_inst = np.asarray(tbl[inst_col], dtype=float)
            m_std = np.asarray(tbl[std_col], dtype=float)
            delta = m_std - m_inst
            comparison = np.isfinite(m_inst) & np.isfinite(m_std)

            tc = result.transformation[f]
            ci_f1, ci_f2 = tc.color_index_filters
            ci_std_col1 = f"mag_std_{ci_f1}"
            ci_std_col2 = f"mag_std_{ci_f2}"
            if ci_std_col1 in tbl.colnames and ci_std_col2 in tbl.colnames:
                color = (
                    np.asarray(tbl[ci_std_col1], dtype=float)
                    - np.asarray(tbl[ci_std_col2], dtype=float)
                )
                comparison &= np.isfinite(color)
            else:
                color = np.zeros(len(tbl), dtype=float)

            comparison &= comp_mask
            if not np.any(comparison):
                continue

            if fit_masks is not None and epoch_id in fit_masks:
                used = np.asarray(fit_masks[epoch_id], dtype=bool)
                if used.shape != comparison.shape:
                    used = comparison.copy()
                else:
                    # Show all comparison candidates; mark only fit survivors as used.
                    used = used & comparison
            else:
                used = comparison.copy()

            # Hide non-candidates in the scatter (NaN); clipped comps stay visible as gray.
            color_plot = np.asarray(color, dtype=float).copy()
            delta_plot = np.asarray(delta, dtype=float).copy()
            color_plot[~comparison] = np.nan
            delta_plot[~comparison] = np.nan
            plot_data[f] = (color_plot, delta_plot, used)
        if plot_data:
            plots.plot_calibration_transformation(
                output_dir,
                epoch_id,
                plot_data,
                result.transformation,
                file_type=file_type,
                filename_prefix=filename_prefix,
                title_prefix=title_prefix,
            )


__all__ = [
    "CalibrationEngine",
    "CalibrationResult",
    "TransformationCoefficients",
    "prepare_calibration_check_plots",
]
