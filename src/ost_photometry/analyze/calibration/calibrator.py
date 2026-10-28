"""Catalog-driven photometry calibration orchestrator.

:class:`PhotometryCalibrator` fetches a standard catalog, cross-matches epochs,
fits extinction and T/ZP via :class:`~ost_photometry.analyze.calibration.photometer.DifferentialPhotometer`,
and returns calibrated tables.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

from ...core.parallel import start_plot_process
from ..calibration_sources import crossmatch_standard_catalog, fetch_standard_calibration_catalog
from ..extinction import (
    CoefficientMode,
    ExtinctionCoefficients,
    ExtinctionCorrector,
    ExtinctionOrder,
    calculate_airmass,
    extinction_airmass_ready,
    fit_extinction_from_comparison_stars,
)
from ..warnings_types import OstPhotometryAnalyzeWarning
from .photometer import (
    DifferentialPhotometer,
    _apply_rolling_smooth_to_per_image_results,
    _resolve_per_image_rolling_mode,
)
from .result import CalibrationResult
from .statistics import average_epochs_per_star, mark_shared_zero_point


class PhotometryCalibrator:
    """
    Main class for differential photometry calibration.

    Workflow:

    1. **Reference catalog** — :meth:`setup_calibration_source` calls
       :func:`~ost_photometry.analyze.calibration_sources.fetch_standard_calibration_catalog`
       (same ``calibration_source`` / ``vizier_dict`` / file options as the legacy pipeline).
       The table stays in **standard schema** (``ra``, ``dec``, ``mag_std_*``).

    2. **Epochs** — :meth:`add_epoch` cross-matches each epoch’s detections against
       that catalog via :func:`~ost_photometry.analyze.calibration_sources.crossmatch_standard_catalog`,
       attaching standard magnitudes for comparison stars.

    3. **Extinction + transforms** — :class:`ExtinctionCorrector` and
       :class:`DifferentialPhotometer`; :meth:`fit_transformation_parameters` fits
       T/ZP, then :meth:`~DifferentialPhotometer.apply_transform_to_table` applies them.
    """

    def __init__(
        self,
        mode: CoefficientMode = CoefficientMode.PER_NIGHT,
        extinction_order: ExtinctionOrder = ExtinctionOrder.FIRST,
        extinction_coefficients: dict[str, ExtinctionCoefficients] | None = None,
        observatory_location: object | None = None,
        color_indices: dict | None = None,
        match_radius=None,
    ):
        self.mode = mode
        self.location = observatory_location
        #: Sky radius for :func:`crossmatch_standard_catalog` (pipeline: ``calibration_match_radius``).
        self.match_radius = match_radius
        self.extinction = ExtinctionCorrector(
            coefficients=extinction_coefficients, order=extinction_order
        )
        # Standard-schema table from fetch; None until setup_calibration_source()
        self.reference_catalog: Table | None = None
        self.photometer = DifferentialPhotometer(
            color_indices=color_indices,
            extinction_corrector=self.extinction,
        )
        self.epochs: dict[str, Table] = {}
        self.epoch_metadata: dict[str, dict] = {}
        self.fixed_color_terms: dict[str, tuple[float, float]] = {}

    def setup_calibration_source(
        self,
        center: SkyCoord,
        filters: list[str],
        *,
        calibration_source: str = "auto",
        radius_arcmin: float = 15.0,
        calibration_catalog_mag_range: tuple[float, float] = (0.0, 18.5),
        vizier_dict: dict[str, str] | None = None,
        path_calibration_file: str | None = None,
        apply_sloan_to_johnson_ri: bool = True,
        indent: int = 0,
    ) -> None:
        """
        Download (or read) the calibration catalog and store it in standard schema.

        Mirrors :func:`~ost_photometry.analyze.calibration_sources.fetch_standard_calibration_catalog`:

        * ``calibration_source`` — ``"APASS"``, ``"simbad"``, ``"vsp"``, ``"simbad_vot"``,
          or any key present in ``vizier_dict`` (e.g. ``"GSC2.3"``, ``"SDSS_Release_16"``).
        * ``radius_arcmin`` — cone radius in arcminutes (pipeline: ``calibration_catalog_radius_arcmin``).
        * ``calibration_catalog_mag_range`` — inclusive bright/faint limits on the catalog’s reference band(s).
        * ``vizier_dict`` / ``path_calibration_file`` — same semantics as
          :func:`~ost_photometry.analyze.calibration_sources.fetch_standard_calibration_catalog`.

        Lupton Johnson R/I from Sloan bands is applied inside ``fetch`` when appropriate;
        set ``apply_sloan_to_johnson_ri=False`` to disable the optional Vizier heuristic only.
        """
        self.reference_catalog = fetch_standard_calibration_catalog(
            filters,
            center,
            calibration_source=calibration_source,
            field_of_view_arcmin=radius_arcmin,
            calibration_catalog_mag_range=calibration_catalog_mag_range,
            vizier_dict=vizier_dict,
            path_calibration_file=path_calibration_file,
            apply_sloan_to_johnson_ri=apply_sloan_to_johnson_ri,
            indent=indent,
        )

    def add_epoch(
        self,
        epoch_id: str,
        data: Table,
        obstime: Time | None = None,
        airmass: float | None = None,
        filter_obstimes: dict[str, Time] | None = None,
        ra_col: str = "ra",
        dec_col: str = "dec",
    ):
        """
        Add a calibration epoch table (multi-band rows, aligned ``id``).

        Expects ``mag_<filter>`` / ``err_<filter>``. If ``airmass_<filter>`` columns
        are missing, they are filled from ``filter_obstimes[f]``, then ``obstime``,
        then scalar ``airmass``, else NaN. A mean ``airmass`` column is added when
        absent if per-filter columns exist.
        """
        data = data.copy()
        mag_prefix = "mag_"
        std_prefix = "mag_std_"
        filters_here = [
            c[len(mag_prefix) :]
            for c in data.colnames
            if c.startswith(mag_prefix) and not c.startswith(std_prefix)
        ]
        for f in filters_here:
            col_a = f"airmass_{f}"
            if col_a in data.colnames:
                continue
            t_use: Time | None = None
            if filter_obstimes is not None and f in filter_obstimes:
                t_use = filter_obstimes[f]
            elif obstime is not None:
                t_use = obstime
            if t_use is not None and self.location is not None:
                coords = SkyCoord(data[ra_col], data[dec_col], unit="deg")
                data[col_a] = calculate_airmass(coords, t_use, self.location)
            elif airmass is not None:
                data[col_a] = float(airmass)
            else:
                data[col_a] = np.nan

        if "airmass" not in data.colnames:
            if filters_here:
                stacks = np.column_stack(
                    [
                        np.asarray(data[f"airmass_{f}"], dtype=float)
                        for f in filters_here
                    ]
                )
                data["airmass"] = np.nanmean(stacks, axis=1)
            elif airmass is not None:
                data["airmass"] = float(airmass)
            elif obstime is not None and self.location is not None:
                coords = SkyCoord(data[ra_col], data[dec_col], unit="deg")
                data["airmass"] = calculate_airmass(
                    coords, obstime, self.location
                )
            else:
                data["airmass"] = np.nan

        # Attach mag_std_* (and other numeric catalog columns) from setup_calibration_source
        if self.reference_catalog is not None and len(self.reference_catalog) > 0:
            match_kw = {}
            if self.match_radius is not None:
                match_kw["match_radius"] = self.match_radius
            data = crossmatch_standard_catalog(
                data, self.reference_catalog, ra_col, dec_col, **match_kw
            )

        self.epochs[epoch_id] = data
        self.epoch_metadata[epoch_id] = {
            "obstime": obstime,
            "filter_obstimes": filter_obstimes,
            "airmass_mean": float(np.nanmean(data["airmass"]))
            if "airmass" in data.colnames
            else None,
        }

    def fit_extinction_from_epochs(
        self,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        fallback_airmass_col: str = "airmass",
        output_dir: str | None = None,
        file_type: str = "pdf",
    ) -> dict[str, ExtinctionCoefficients]:
        """
        Fit extinction coefficients from catalog-matched comparison stars in epochs.

        Call after add_epoch() for all epochs. Updates internal ExtinctionCorrector.
        Requires epoch tables with mag_std_* from reference-catalog crossmatch.

        Parameters
        ----------
        output_dir : str, optional
            If provided, save diagnostic plots to output_dir/diagnostics/extinction/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        """
        fitted = fit_extinction_from_comparison_stars(
            self.epochs,
            mag_col_prefix=mag_col_prefix,
            std_col_prefix=std_col_prefix,
            fallback_airmass_col=fallback_airmass_col,
            output_dir=output_dir,
            file_type=file_type,
        )
        if fitted:
            self.extinction.coefficients.update(fitted)
        return fitted

    def _decide_color_terms(self, filters: list[str], comparison_selector) -> dict[str, str]:
        """``auto`` colour term, decided once per filter for all epochs.

        Fitted (``always``) when the comparison stars of all epochs together
        span more than 0.1 mag (std) in the filter's colour, else ``never``.
        A per-epoch decision mixed epochs with and without a colour term:
        magnitudes jumped between them and the per-night combination failed.
        """
        decision: dict[str, str] = {}
        for filter_ in filters:
            c1, c2 = self.photometer.color_indices.get(filter_, ("B", "V"))
            colours = []
            for data in self.epochs.values():
                s1, s2 = f"mag_std_{c1}", f"mag_std_{c2}"
                if s1 not in data.colnames or s2 not in data.colnames:
                    continue
                mask = np.asarray(comparison_selector(data), dtype=bool)
                colour = np.asarray(data[s1], dtype=float) - np.asarray(data[s2], dtype=float)
                colours.append(colour[mask & np.isfinite(colour)])
            pooled = np.concatenate(colours) if colours else np.array([])
            spread = float(np.std(pooled)) if pooled.size > 1 else 0.0
            decision[filter_] = "always" if spread > 0.1 else "never"
            if decision[filter_] == "never":
                warnings.warn(
                    f"Filter {filter_}: colour spread of the comparison stars std={spread:.3f} mag "
                    "<= 0.1 over all epochs, using ZP-only fits (no colour term)",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
        return decision

    def set_fixed_color_terms(self, color_terms: dict[str, tuple[float, float]]) -> None:
        """Colour terms ``{filter: (T, σ_T)}`` held fixed in FIXED mode."""
        self.fixed_color_terms = dict(color_terms)

    def fit_transformation_parameters(
        self,
        filters: list[str],
        comparison_selector=None,
        determine_color_terms: bool = True,
        min_comparisons: int = 5,
        sigma_clip: float = 2.5,
        output_dir: str | None = None,
        file_type: str = "pdf",
        per_image_rolling_median_color_term: bool = False,
        per_image_rolling_median_zero_point: bool = False,
        per_image_rolling_mean_color_term: bool = False,
        per_image_rolling_mean_zero_point: bool = False,
        per_image_rolling_window: int = 3,
        calibration_summary_x_jd: dict[str, float] | None = None,
        calibration_summary_use_jd_x: bool = False,
        color_term_fit: Literal["always", "auto", "never"] = "auto",
    ) -> dict[str, CalibrationResult]:
        """
        Fit color terms and zero points (per mode); store results in :attr:`calib_parameters`.

        This **determines** transformation parameters (T, ZP). Applying them to magnitudes is
        :meth:`~DifferentialPhotometer.apply_transform_to_table` via :meth:`get_calibrated_photometry`.

        Parameters
        ----------
        output_dir : str, optional
            If provided, save calibration diagnostic plots to output_dir/diagnostics/calibration/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        sigma_clip : float
            Outlier rejection in :meth:`DifferentialPhotometer.fit_transformation_epoch`
            (|residual| < ``sigma_clip`` × RMS). Pipeline: ``PipelineConfig.fit_sigma_clip``.
        per_image_rolling_median_color_term
            If True (``PER_IMAGE`` only), replace each epoch's color term by the centered
            rolling median along sorted epoch order.
        per_image_rolling_median_zero_point
            If True (``PER_IMAGE`` only), same for zero point.
        per_image_rolling_mean_color_term
            If True (``PER_IMAGE`` only), rolling mean for color term (same window).
        per_image_rolling_mean_zero_point
            If True (``PER_IMAGE`` only), rolling mean for zero point.
            Median and mean must not both be enabled for the same quantity; if they are,
            median is used and a warning is issued.
        per_image_rolling_window
            Rolling window length for median and mean (odd; even values are incremented).
            Default 3.
        """
        if not self.epochs:
            raise ValueError("No epochs added!")

        if comparison_selector is None:
            def comparison_selector(table):
                mask = np.ones(len(table), dtype=bool)
                for filter_ in filters:
                    std_col = f"mag_std_{filter_}"
                    if std_col in table.colnames:
                        mask &= np.isfinite(table[std_col])
                return mask

        if color_term_fit == "auto" and determine_color_terms:
            color_term_fit = self._decide_color_terms(filters, comparison_selector)

        results = {}
        if self.mode == CoefficientMode.FIXED:
            #   Known colour terms (campaign), zero point fitted per epoch;
            #   σ_T stays a systematic of every calibrated magnitude.
            fixed = getattr(self, "fixed_color_terms", None)
            if not fixed:
                raise ValueError(
                    "calibration_grouping='fixed' needs transformation_coefficients_dict "
                    "with the colour terms"
                )
            for epoch_id, data in self.epochs.items():
                mask = comparison_selector(data)
                results[epoch_id] = self.photometer.fit_transformation_epoch(
                    data, epoch_id, filters, mask,
                    determine_color_terms=False,
                    min_comparisons=min_comparisons,
                    sigma_clip=sigma_clip,
                    color_term_fit="never",
                    fixed_color_terms=fixed,
                    output_dir=output_dir,
                    file_type=file_type,
                )
        elif self.mode == CoefficientMode.PER_IMAGE:
            for epoch_id, data in self.epochs.items():
                mask = comparison_selector(data)
                result = self.photometer.fit_transformation_epoch(
                    data, epoch_id, filters, mask,
                    determine_color_terms=determine_color_terms,
                    min_comparisons=min_comparisons,
                    sigma_clip=sigma_clip,
                    color_term_fit=color_term_fit,
                    output_dir=output_dir,
                    file_type=file_type,
                )
                results[epoch_id] = result

            from .photometer import report_calibrator_counts

            ordered_ids = sorted(self.epochs.keys(), key=str)
            report_calibrator_counts(
                [(epoch_id, results[epoch_id].transformation) for epoch_id in ordered_ids],
                filters,
            )
            ct_mode = _resolve_per_image_rolling_mode(
                per_image_rolling_median_color_term,
                per_image_rolling_mean_color_term,
                quantity="color term",
            )
            zp_mode = _resolve_per_image_rolling_mode(
                per_image_rolling_median_zero_point,
                per_image_rolling_mean_zero_point,
                quantity="zero point",
            )
            if ct_mode != "none" or zp_mode != "none":
                _apply_rolling_smooth_to_per_image_results(
                    ordered_ids,
                    results,
                    filters,
                    window=per_image_rolling_window,
                    color_term_mode=ct_mode,
                    zero_point_mode=zp_mode,
                )

            if output_dir and len(self.epochs) > 1:
                from .. import plots

                ordered_results = [results[k] for k in ordered_ids]
                combined_preview = self.photometer.combine_epoch_calibration_results(
                    ordered_results,
                    filters,
                )
                x_jd_plot: list[float] | None = None
                if calibration_summary_use_jd_x and calibration_summary_x_jd:
                    seq = [
                        float(calibration_summary_x_jd.get(str(k), np.nan))
                        for k in ordered_ids
                    ]
                    if all(np.isfinite(seq)):
                        x_jd_plot = seq
                    else:
                        warnings.warn(
                            "calibration_summary_use_jd_x: missing or non-finite JD for "
                            "one or more epochs; using epoch index on per-image summary plot.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )

                start_plot_process(
                    plots.plot_calibration_night_summary,
                    (
                        output_dir,
                        ordered_ids,
                        [results[k].transformation for k in ordered_ids],
                        filters,
                    ),
                    {
                        "file_type": file_type,
                        "combined_per_filter": combined_preview,
                        "output_basename": "calibration_per_image_summary",
                        "x_jd": x_jd_plot,
                    },
                )
        elif self.mode == CoefficientMode.PER_NIGHT:
            result = self.photometer.fit_transformation_night(
                self.epochs, filters, comparison_selector,
                night_id="night_combined",
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip,
                color_term_fit=color_term_fit,
                output_dir=output_dir,
                file_type=file_type,
                calibration_summary_x_jd=calibration_summary_x_jd,
                calibration_summary_use_jd_x=calibration_summary_use_jd_x,
            )
            mark_shared_zero_point(result, list(self.epochs.values()), filters)
            for epoch_id in self.epochs:
                results[epoch_id] = result
        elif self.mode == CoefficientMode.ENSEMBLE:
            #   One point per star (extinction-corrected mean over epochs):
            #   stacking all epochs counted each star's catalog error once
            #   per epoch as if independent.
            corrected = []
            for data in self.epochs.values():
                if extinction_airmass_ready(data, filters, "airmass"):
                    data = self.photometer.extinction.correct(
                        data,
                        fallback_airmass_col="airmass",
                        mag_col_prefix="mag_",
                        output_prefix="mag_",
                        filters=filters,
                        inplace=False,
                        catalog_color_prefix="mag_std_",
                    )
                corrected.append(data)
            combined = average_epochs_per_star(corrected, filters)
            combined_mask = comparison_selector(combined)
            result = self.photometer.fit_transformation_epoch(
                combined, "ensemble", filters, combined_mask,
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip,
                color_term_fit=color_term_fit,
            )
            mark_shared_zero_point(result, list(self.epochs.values()), filters)
            for epoch_id in self.epochs:
                results[epoch_id] = result

        # epoch_id -> CalibrationResult (fitted T, ZP per filter); not legacy CalibParameters
        self.calib_parameters = results
        return results

    def get_calibrated_photometry(
        self,
        output_prefix: str = "mag_cal_",
        target_selector=None,
    ) -> Table:
        """Apply calibration and return calibrated table."""
        if not hasattr(self, "calib_parameters"):
            raise ValueError("Call fit_transformation_parameters() first!")

        all_results = []
        for epoch_id, data in self.epochs.items():
            cal = self.calib_parameters[epoch_id]
            calibrated = self.photometer.apply_transform_to_table(
                data, cal, output_prefix=output_prefix, inplace=False
            )
            calibrated["epoch_id"] = epoch_id
            if target_selector is not None:
                calibrated = calibrated[target_selector(calibrated)]
            all_results.append(calibrated)
        return vstack(all_results) if all_results else Table()



__all__ = ["PhotometryCalibrator"]
