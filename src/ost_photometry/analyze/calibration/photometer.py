"""Ensemble T/ZP fit and application for catalog-calibrated magnitudes.

:class:`DifferentialPhotometer` fits colour terms and zero points from
comparison stars and applies them to epoch tables. Catalog fetch / crossmatch
live in :mod:`~ost_photometry.analyze.calibration_sources`; orchestration is
:class:`~ost_photometry.analyze.calibration.calibrator.PhotometryCalibrator`.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from astropy.table import Table

from ... import terminal_output
from ...core.parallel import start_plot_process
from ..extinction import ExtinctionCorrector, extinction_airmass_ready
from ..warnings_types import OstPhotometryAnalyzeWarning
from .result import CalibrationResult, TransformationCoefficients

#: Catalog error per band when the catalog gives none (APASS convention,
#: see ``calibration_sources/fetch.py``).
_CATALOG_ERR_DEFAULT = 0.02


def _color_for_transformation(
    data: Table,
    tc,
    cal_mags: dict[str, np.ndarray],
    calibrated_filters: set[str],
    *,
    std_col_prefix: str = "mag_std_",
    err_col_prefix: str = "err_",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colour for ``m_cal = m_inst + T·colour + ZP``, its variance and source.

    Per row, in this order:

    ``calibrated``
        both colour filters are calibrated in this epoch (running estimate);
        variance from their instrumental errors.
    ``catalog``
        catalog colour ``mag_std_c1 - mag_std_c2`` (star without the other
        filter, or that filter has no transformation); variance from
        ``err_std_*`` (0.02 mag per band when the catalog gives none).
    ``none``
        no colour: NaN, so ``mag_cal`` is NaN rather than biased.

    The instrumental colour is never used: it carries the arbitrary
    zero-point difference of the two filters. With ``T = 0`` the colour does
    not enter and the source is ``not_needed``.
    """
    n = len(data)
    color = np.full(n, np.nan)
    sigma_sq = np.zeros(n)
    source = np.full(n, "none", dtype="U12")
    if abs(float(tc.color_term)) <= 1e-12 and float(tc.color_term_err) <= 0.0:
        source[:] = "not_needed"
        return np.zeros(n), sigma_sq, source
    c1, c2 = tc.color_index_filters
    if c1 in calibrated_filters and c2 in calibrated_filters:
        cal = cal_mags[c1] - cal_mags[c2]
        ok = np.isfinite(cal)
        color[ok] = cal[ok]
        source[ok] = "calibrated"
        e1, e2 = f"{err_col_prefix}{c1}", f"{err_col_prefix}{c2}"
        if e1 in data.colnames and e2 in data.colnames:
            sigma_sq[ok] = (
                np.asarray(data[e1].ravel(), dtype=float)[ok] ** 2
                + np.asarray(data[e2].ravel(), dtype=float)[ok] ** 2
            )
    s1, s2 = f"{std_col_prefix}{c1}", f"{std_col_prefix}{c2}"
    if s1 in data.colnames and s2 in data.colnames:
        cat = np.asarray(data[s1], dtype=float) - np.asarray(data[s2], dtype=float)
        use = ~np.isfinite(color) & np.isfinite(cat)
        color[use] = cat[use]
        source[use] = "catalog"
        es1 = std_col_prefix.replace("mag_", "err_") + c1
        es2 = std_col_prefix.replace("mag_", "err_") + c2
        cat_var = np.zeros(n)
        for col in (es1, es2):
            if col in data.colnames:
                e = np.asarray(data[col], dtype=float)
                e = np.where(np.isfinite(e) & (e > 0), e, _CATALOG_ERR_DEFAULT)
            else:
                e = np.full(n, _CATALOG_ERR_DEFAULT)
            cat_var += e**2
        sigma_sq[use] = cat_var[use]
    return color, sigma_sq, source


def _filters_with_instrumental_mags(
    table: Table,
    filters: list[str],
    mag_col_prefix: str = "mag_",
) -> list[str]:
    """Return filters that have an instrumental column ``{mag_col_prefix}{f}`` in ``table``."""
    return [f for f in filters if f"{mag_col_prefix}{f}" in table.colnames]


def _resolve_per_image_rolling_mode(
    use_median: bool,
    use_mean: bool,
    *,
    quantity: str,
) -> Literal["none", "median", "mean"]:
    if use_median and use_mean:
        warnings.warn(
            f"PER_IMAGE rolling median and mean both enabled for {quantity}; using median.",
            category=OstPhotometryAnalyzeWarning,
            stacklevel=3,
        )
        return "median"
    if use_median:
        return "median"
    if use_mean:
        return "mean"
    return "none"


def _apply_rolling_smooth_to_per_image_results(
    ordered_epoch_ids: list[str],
    results: dict[str, CalibrationResult],
    filters: list[str],
    *,
    window: int,
    color_term_mode: Literal["none", "median", "mean"] = "none",
    zero_point_mode: Literal["none", "median", "mean"] = "none",
) -> None:
    """
    Replace per-epoch T and/or ZP with a centered rolling statistic along ``ordered_epoch_ids``.

    Formal errors for smoothed quantities use the same rolling statistic on per-epoch
    errors in the window (ad hoc). Epochs without a transformation for a filter
    are left unchanged for that filter.
    """
    import pandas as pd

    w = max(1, int(window))
    if w % 2 == 0:
        w += 1

    if color_term_mode == "none" and zero_point_mode == "none":
        return

    def _roll(series: pd.Series, mode: Literal["none", "median", "mean"]) -> pd.Series:
        if mode == "none":
            return series
        r = series.rolling(w, center=True, min_periods=1)
        return r.median() if mode == "median" else r.mean()

    for filter_ in filters:
        t_vals: list[float] = []
        zp_vals: list[float] = []
        t_errs: list[float] = []
        zp_errs: list[float] = []
        has_tc: list[bool] = []
        for eid in ordered_epoch_ids:
            tc = results[eid].transformation.get(filter_)
            if tc is None:
                t_vals.append(np.nan)
                zp_vals.append(np.nan)
                t_errs.append(np.nan)
                zp_errs.append(np.nan)
                has_tc.append(False)
            else:
                t_vals.append(float(tc.color_term))
                zp_vals.append(float(tc.zero_point))
                t_errs.append(float(tc.color_term_err))
                zp_errs.append(float(tc.zero_point_err))
                has_tc.append(True)

        s_t = pd.Series(t_vals, dtype=float)
        s_zp = pd.Series(zp_vals, dtype=float)
        s_te = pd.Series(t_errs, dtype=float)
        s_ze = pd.Series(zp_errs, dtype=float)
        roll_t = _roll(s_t, color_term_mode)
        roll_te = _roll(s_te, color_term_mode)
        roll_zp = _roll(s_zp, zero_point_mode)
        roll_ze = _roll(s_ze, zero_point_mode)

        for i, eid in enumerate(ordered_epoch_ids):
            if not has_tc[i]:
                continue
            r = results[eid]
            tc = r.transformation[filter_]
            new_t = float(roll_t.iloc[i]) if color_term_mode != "none" else tc.color_term
            new_te = (
                float(roll_te.iloc[i]) if color_term_mode != "none" else tc.color_term_err
            )
            new_zp = (
                float(roll_zp.iloc[i]) if zero_point_mode != "none" else tc.zero_point
            )
            new_ze = (
                float(roll_ze.iloc[i])
                if zero_point_mode != "none"
                else tc.zero_point_err
            )
            # Rolling mixes epochs independently for T and ZP → drop fit covariance.
            new_cov = (
                0.0
                if color_term_mode != "none" or zero_point_mode != "none"
                else tc.cov_tz
            )
            r.transformation[filter_] = TransformationCoefficients(
                filter_name=filter_,
                color_term=new_t,
                color_term_err=new_te,
                zero_point=new_zp,
                zero_point_err=new_ze,
                cov_tz=new_cov,
                color_index_filters=tc.color_index_filters,
                n_stars_used=tc.n_stars_used,
                rms_residual=tc.rms_residual,
            )


def _apply_rolling_median_to_per_image_results(
    ordered_epoch_ids: list[str],
    results: dict[str, CalibrationResult],
    filters: list[str],
    *,
    window: int,
    smooth_color_term: bool,
    smooth_zero_point: bool,
) -> None:
    """Rolling median; see :func:`_apply_rolling_smooth_to_per_image_results`."""
    _apply_rolling_smooth_to_per_image_results(
        ordered_epoch_ids,
        results,
        filters,
        window=window,
        color_term_mode="median" if smooth_color_term else "none",
        zero_point_mode="median" if smooth_zero_point else "none",
    )


def _apply_rolling_mean_to_per_image_results(
    ordered_epoch_ids: list[str],
    results: dict[str, CalibrationResult],
    filters: list[str],
    *,
    window: int,
    smooth_color_term: bool,
    smooth_zero_point: bool,
) -> None:
    """Rolling mean; see :func:`_apply_rolling_smooth_to_per_image_results`."""
    _apply_rolling_smooth_to_per_image_results(
        ordered_epoch_ids,
        results,
        filters,
        window=window,
        color_term_mode="mean" if smooth_color_term else "none",
        zero_point_mode="mean" if smooth_zero_point else "none",
    )


class DifferentialPhotometer:
    """
    Differential ensemble photometry using comparison stars.

    Determines transformation coefficients from instrumental to standard magnitudes
    via the relation::

        m_std - m_inst = T * (CI1 - CI2) + ZP

    where T is the color term, ZP the zero point, and (CI1 - CI2) the color index
    (e.g. B-V). The color term accounts for passband differences between the
    instrumental and standard system (filter transmission, detector response).
    Uses APASS-matched comparison stars with known standard magnitudes.
    """

    DEFAULT_COLOR_INDICES = {
        "U": ("U", "B"),
        "B": ("B", "V"),
        "V": ("B", "V"),
        "R": ("V", "R"),
        "I": ("R", "I"),
    }

    def __init__(
        self,
        color_indices: dict | None = None,
        extinction_corrector: ExtinctionCorrector | None = None,
    ):
        """
        Parameters
        ----------
        color_indices : dict, optional
            {filter: (filter1, filter2)} defining the color index per filter,
            e.g. {"V": ("B", "V")} for B-V. Defaults to Johnson-Cousins conventions.
        extinction_corrector : ExtinctionCorrector, optional
            Applies airmass-dependent extinction correction before transformation.
        """
        self.color_indices = self.DEFAULT_COLOR_INDICES.copy()
        if color_indices:
            self.color_indices.update(color_indices)
        self.extinction = extinction_corrector
        self.calibrations: dict[str, CalibrationResult] = {}

    def fit_transformation_epoch(
        self,
        data: Table,
        epoch_id: str,
        filters: list[str],
        comparison_mask: np.ndarray,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        fallback_airmass_col: str = "airmass",
        sigma_clip: float = 2.5,
        min_comparisons: int = 3,
        determine_color_terms: bool = True,
        color_term_fit: Literal["always", "auto", "never"] = "auto",
        output_dir: str | None = None,
        file_type: str = "pdf",
    ) -> CalibrationResult:
        """
        Fit transformation parameters (color term T, zero point ZP) for one epoch.

        Does **not** apply calibration to the full table; use
        :meth:`apply_transform_to_table` with the returned :class:`CalibrationResult`.
        For each filter, fits m_std - m_inst = T * color + ZP using comparison stars.
        T (color term) corrects for passband mismatch; ZP (zero point) sets the scale.
        Applies extinction correction first, then iterative sigma-clip to reject outliers.
        Requires color spread > 0.1 mag for reliable T; otherwise uses ZP-only.

        Parameters
        ----------
        color_term_fit : {"always", "auto", "never"}
            ``never`` — median ZP, T=0.
            ``always`` — always attempt linear T/ZP when color columns exist.
            ``auto`` — linear when color spread > 0.1 mag, else median ZP (default).
            Extinction (when a corrector with a non-``NONE`` order is set) is
            removed before the fit in every mode, exactly as
            :meth:`apply_transform_to_table` does before adding T/ZP; a ZP
            fitted without that correction would be off by ``k'·X`` on apply.
        output_dir : str, optional
            If provided, save transformation fit plots to output_dir/diagnostics/calibration/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        """
        result = CalibrationResult(identifier=epoch_id)
        plot_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        filters_use = _filters_with_instrumental_mags(data, filters, mag_col_prefix)
        if not filters_use:
            warnings.warn(
                f"[{epoch_id}] No instrumental columns "
                f"{[f'{mag_col_prefix}{f}' for f in filters]}; nothing to calibrate.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=1,
            )
            self.calibrations[epoch_id] = result
            return result

        if self.extinction is not None and extinction_airmass_ready(
            data, filters_use, fallback_airmass_col
        ):
            data = self.extinction.correct(
                data,
                fallback_airmass_col=fallback_airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,
                filters=filters_use,
                inplace=False,
                catalog_color_prefix=std_col_prefix,
            )

        comps = data[comparison_mask]

        for filter_ in filters_use:
            # --- Column names for instrumental and standard magnitudes ---
            inst_col = f"{mag_col_prefix}{filter_}"
            std_col = f"{std_col_prefix}{filter_}"

            if inst_col not in comps.colnames or std_col not in comps.colnames:
                missing = [c for c in (inst_col, std_col) if c not in comps.colnames]
                warnings.warn(
                    f"[{epoch_id}] Filter {filter_} skipped: missing columns {missing}",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )
                continue

            # --- Valid comparison stars: finite mag_inst and mag_std ---
            m_inst = np.array(comps[inst_col], dtype=float)
            m_std = np.array(comps[std_col], dtype=float)
            valid = np.isfinite(m_inst) & np.isfinite(m_std)

            if np.sum(valid) < min_comparisons:
                warnings.warn(
                    f"[{epoch_id}] Filter {filter_} skipped: only {np.sum(valid)} valid "
                    f"comparisons (need >= {min_comparisons})",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )
                continue

            # --- Color index for transformation: m_std - m_inst = T * (CI1 - CI2) + ZP ---
            ci_filters = self.color_indices.get(filter_, ("B", "V"))
            ci_std_col1 = f"{std_col_prefix}{ci_filters[0]}"
            ci_std_col2 = f"{std_col_prefix}{ci_filters[1]}"
            has_color = ci_std_col1 in comps.colnames and ci_std_col2 in comps.colnames

            if determine_color_terms and has_color:
                color_std = np.array(comps[ci_std_col1]) - np.array(comps[ci_std_col2])
                valid &= np.isfinite(color_std)
            else:
                color_std = np.zeros(len(comps))

            if np.sum(valid) < min_comparisons:
                warnings.warn(
                    f"[{epoch_id}] Filter {filter_} skipped: after color-index cut "
                    f"only {np.sum(valid)} valid (need >= {min_comparisons})",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )
                continue

            # --- Iterative sigma-clip fit: T (color term), ZP (zero point) ---
            mask = valid.copy()
            T, ZP, T_err, ZP_err, cov_tz = 0.0, 0.0, 0.0, 0.0, 0.0
            rms = 0.0
            warned_color_spread = False

            for _ in range(5):
                c = color_std[mask]
                if color_term_fit == "always":
                    use_linear = determine_color_terms and has_color
                elif color_term_fit == "never":
                    use_linear = False
                else:
                    use_linear = (
                        determine_color_terms and has_color and np.std(c) > 0.1
                    )

                if use_linear:
                    T, ZP, T_err, ZP_err, cov_tz = self._weighted_linear_fit(
                        c, m_std[mask] - m_inst[mask], np.ones(np.sum(mask))
                    )
                else:
                    if (
                        color_term_fit == "auto"
                        and determine_color_terms
                        and has_color
                        and not warned_color_spread
                    ):
                        warnings.warn(
                            f"[{epoch_id}] Filter {filter_}: color spread std={np.std(c):.3f} mag "
                            f"<= 0.1, using ZP-only fit (no color term)",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=1,
                        )
                        warned_color_spread = True
                    T = 0.0
                    T_err = 0.0
                    cov_tz = 0.0
                    ZP = float(np.median((m_std - m_inst)[mask]))
                    ZP_err = float(
                        np.std((m_std - m_inst)[mask]) / np.sqrt(np.sum(mask))
                    )

                all_residuals = (m_std - m_inst) - (T * color_std + ZP)
                rms = np.nanstd(all_residuals[mask])
                new_mask = valid & (np.abs(all_residuals) < sigma_clip * rms)
                if np.sum(new_mask) == np.sum(mask) or np.sum(new_mask) < min_comparisons:
                    break
                mask = new_mask

            result.transformation[filter_] = TransformationCoefficients(
                filter_name=filter_,
                color_term=T,
                color_term_err=T_err,
                zero_point=ZP,
                zero_point_err=ZP_err,
                cov_tz=cov_tz,
                color_index_filters=ci_filters,
                n_stars_used=int(np.sum(mask)),
                rms_residual=rms,
            )
            used_full = np.zeros(len(data), dtype=bool)
            comp_idx = np.flatnonzero(np.asarray(comparison_mask, dtype=bool).ravel())
            keep = np.asarray(mask, dtype=bool).ravel()
            if keep.size == comp_idx.size:
                used_full[comp_idx[keep]] = True
            result.calibrator_mask_by_filter[filter_] = used_full
            if output_dir:
                plot_data[filter_] = (color_std, m_std - m_inst, mask)

        if output_dir and plot_data:
            from .. import plots
            start_plot_process(
                plots.plot_calibration_transformation,
                (output_dir, epoch_id, plot_data, result.transformation),
                {"file_type": file_type},
            )

        result.n_comparison_stars = int(np.sum(comparison_mask))
        self.calibrations[epoch_id] = result
        return result

    def combine_epoch_calibration_results(
        self,
        epoch_results: list[CalibrationResult],
        filters: list[str],
    ) -> dict[str, TransformationCoefficients]:
        """
        Combine per-epoch T and ZP per filter with inverse-variance weights
        (same recipe as :meth:`fit_transformation_night` night result).
        """
        out: dict[str, TransformationCoefficients] = {}
        for filter_ in filters:
            results_for_filter_ = [fr for fr in epoch_results if filter_ in fr.transformation]
            if not results_for_filter_:
                continue
            T_vals = np.array(
                [fr.transformation[filter_].color_term for fr in results_for_filter_],
                dtype=float,
            )
            ZP_vals = np.array(
                [fr.transformation[filter_].zero_point for fr in results_for_filter_],
                dtype=float,
            )
            T_errs = np.array(
                [fr.transformation[filter_].color_term_err for fr in results_for_filter_],
                dtype=float,
            )
            ZP_errs = np.array(
                [fr.transformation[filter_].zero_point_err for fr in results_for_filter_],
                dtype=float,
            )
            if np.all(T_errs == 0.0) and np.all(T_vals == T_vals[0]):
                # Colour term held fixed (ZP-only fits): not a measurement.
                T_mean, T_err = float(T_vals[0]), 0.0
            else:
                T_mean, T_err = self._inverse_variance_weighted_mean(T_vals, T_errs)
            ZP_mean, ZP_err = self._inverse_variance_weighted_mean(ZP_vals, ZP_errs)
            out[filter_] = TransformationCoefficients(
                filter_name=filter_,
                color_term=T_mean,
                color_term_err=T_err,
                zero_point=ZP_mean,
                zero_point_err=ZP_err,
                # Independent IV means for T and ZP — no joint covariance.
                cov_tz=0.0,
                color_index_filters=results_for_filter_[0].transformation[filter_].color_index_filters,
                n_stars_used=sum(
                    fr.transformation[filter_].n_stars_used for fr in results_for_filter_
                ),
                rms_residual=float(
                    np.mean(
                        [
                            fr.transformation[filter_].rms_residual
                            for fr in results_for_filter_
                        ]
                    )
                ),
            )
        return out

    def fit_transformation_night(
        self,
        epochs: dict[str, Table],
        filters: list[str],
        comparison_mask_func,
        night_id: str = "night",
        output_dir: str | None = None,
        file_type: str = "pdf",
        calibration_summary_x_jd: dict[str, float] | None = None,
        calibration_summary_use_jd_x: bool = False,
        **kwargs,
    ) -> CalibrationResult:
        """
        Fit transformation parameters for a full night by combining epoch fits.

        Runs :meth:`fit_transformation_epoch` per epoch, then combines T and ZP across epochs
        with inverse-variance weights (w_i = 1/σ_i²). The reported uncertainty
        on the combined value is 1/sqrt(Σ w_i). Per-epoch σ are
        ``color_term_err`` and ``zero_point_err``; epochs without a finite
        positive σ are left out (no floor). A colour term held at 0 in every
        epoch (ZP-only fits) stays 0.

        Assumes stable atmospheric conditions and instrument response over the night.
        """
        epoch_results = []
        for epoch_id, data in epochs.items():
            mask = comparison_mask_func(data)
            try:
                result = self.fit_transformation_epoch(
                    data, epoch_id, filters, mask,
                    output_dir=output_dir,
                    file_type=file_type,
                    **kwargs,
                )
                epoch_results.append(result)
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as e:
                warnings.warn(
                    f"Epoch {epoch_id} failed: {e}",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )

        if not epoch_results:
            raise ValueError("No epochs calibrated successfully!")

        epoch_results_sorted = sorted(
            epoch_results, key=lambda fr: str(fr.identifier)
        )
        for fr in epoch_results_sorted:
            parts = []
            for filter_ in filters:
                tc = fr.transformation.get(filter_)
                n_used = int(tc.n_stars_used) if tc is not None else 0
                parts.append(f"{filter_}={n_used}")
            terminal_output.print_to_terminal(
                f"Calibrators in epoch {fr.identifier}: {', '.join(parts)}",
                indent=2,
            )
        combined = self.combine_epoch_calibration_results(
            epoch_results_sorted,
            filters,
        )

        if output_dir and len(epoch_results_sorted) > 1:
            from .. import plots

            x_jd_plot: list[float] | None = None
            if calibration_summary_use_jd_x and calibration_summary_x_jd:
                ids = [str(fr.identifier) for fr in epoch_results_sorted]
                seq = [calibration_summary_x_jd.get(eid, np.nan) for eid in ids]
                if all(np.isfinite(seq)):
                    x_jd_plot = [float(x) for x in seq]
                else:
                    warnings.warn(
                        "calibration_summary_use_jd_x: missing or non-finite JD for "
                        "one or more calibrated epochs; using epoch index on summary plot.",
                        category=OstPhotometryAnalyzeWarning,
                        stacklevel=2,
                    )

            start_plot_process(
                plots.plot_calibration_night_summary,
                (
                    output_dir,
                    [fr.identifier for fr in epoch_results_sorted],
                    [fr.transformation for fr in epoch_results_sorted],
                    filters,
                ),
                {
                    "file_type": file_type,
                    "combined_per_filter": combined,
                    "x_jd": x_jd_plot,
                },
            )

        night_result = CalibrationResult(identifier=night_id)
        for filter_, tc in combined.items():
            night_result.transformation[filter_] = tc
        self.calibrations[night_id] = night_result
        return night_result

    def apply_transform_to_table(
        self,
        data: Table,
        calibration: str | CalibrationResult,
        filters: list[str] | None = None,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        output_prefix: str = "mag_cal_",
        err_col_prefix: str = "err_",
        output_err_prefix: str = "err_cal_",
        fallback_airmass_col: str = "airmass",
        max_iterations: int = 10,
        inplace: bool = False,
    ) -> Table:
        """
        Apply fitted T/ZP (and extinction) to a photometry table: m_cal = m_inst + T * color + ZP.

        First corrects extinction, then applies transformation. Iterative when
        color index uses magnitudes from other filters (e.g. B-V needs calibrated
        B and V); converges when changes < 0.0001 mag.

        Notes
        -----
        Reference magnitudes (e.g. APASS ``mag_std_*``) are used only when *fitting*
        T and ZP in ``fit_transformation_epoch`` / ``fit_transformation_night``. Here, the same
        global T, ZP are applied to **every** row that has instrumental
        ``mag_<filter>`` — including targets without catalog photometry. The
        internal dict ``iter_cal_mags`` is **not** catalog standards: it holds
        the running estimate of calibrated magnitudes per filter, seeded from
        (extinction-corrected) instrumental mags for the iteration.
        """
        cal = self.calibrations[calibration] if isinstance(calibration, str) else calibration
        if not inplace:
            data = data.copy()

        if filters is None:
            filters = list(cal.transformation.keys())
        filters = _filters_with_instrumental_mags(data, filters, mag_col_prefix)
        if not filters:
            return data

        if self.extinction is not None and extinction_airmass_ready(
            data, filters, fallback_airmass_col
        ):
            data = self.extinction.correct(
                data,
                fallback_airmass_col=fallback_airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,
                filters=filters,
                inplace=True,
                catalog_color_prefix=std_col_prefix,
            )
        # Running calibrated mags per filter (starts as instrumental, not APASS)
        iter_cal_mags: dict[str, np.ndarray] = {}
        for filter_ in filters:
            col = f"{mag_col_prefix}{filter_}"
            if col in data.colnames:
                iter_cal_mags[filter_] = np.array(data[col], dtype=float)

        warned_no_transform: set[str] = set()
        warned_no_inst: set[str] = set()
        #   Only filters with a transformation become calibrated; their
        #   running estimate may enter a colour. The others stay instrumental
        #   and must never be mixed into a colour (zero-point offset z1 - z2).
        calibrated_filters = {
            f for f in filters
            if f in cal.transformation and f"{mag_col_prefix}{f}" in data.colnames
        }
        converged = False
        last_max_change = 0.0

        for _ in range(max_iterations):
            max_change = 0.0
            for filter_ in filters:
                if filter_ not in cal.transformation:
                    if filter_ not in warned_no_transform:
                        warnings.warn(
                            f"apply_transform_to_table: filter {filter_} has no transformation "
                            f"in calibration '{getattr(cal, 'identifier', calibration)}'; skipping.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_transform.add(filter_)
                    continue
                tc = cal.transformation[filter_]
                inst_col = f"{mag_col_prefix}{filter_}"
                if inst_col not in data.colnames:
                    if filter_ not in warned_no_inst:
                        warnings.warn(
                            f"apply_transform_to_table: missing instrumental column '{inst_col}' "
                            f"for filter {filter_}; skipping.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_inst.add(filter_)
                    continue
                m_inst = np.array(data[inst_col], dtype=float)
                color, _sigma_sq, _source = _color_for_transformation(
                    data,
                    tc,
                    iter_cal_mags,
                    calibrated_filters,
                    std_col_prefix=std_col_prefix,
                    err_col_prefix=err_col_prefix,
                )
                m_cal_new = m_inst + tc.color_term * color + tc.zero_point
                if filter_ in iter_cal_mags:
                    max_change = max(max_change, np.nanmax(np.abs(m_cal_new - iter_cal_mags[filter_])))
                iter_cal_mags[filter_] = m_cal_new
            last_max_change = max_change
            if max_change < 0.0001:
                converged = True
                break

        if not converged and max_iterations > 0:
            cal_id = getattr(cal, "identifier", calibration)
            warnings.warn(
                f"apply_transform_to_table: calibration '{cal_id}' did not converge after "
                f"{max_iterations} iterations (last max_change={last_max_change:.6f} mag).",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )

        warned_no_err: set[str] = set()
        for filter_ in filters:
            if filter_ in iter_cal_mags:
                data[f"{output_prefix}{filter_}"] = iter_cal_mags[filter_]
                if filter_ in cal.transformation:
                    err_col = f"{err_col_prefix}{filter_}"
                    if err_col in data.colnames:
                        tc = cal.transformation[filter_]
                        inst_err = np.asarray(data[err_col].ravel(), dtype=float)
                        color, sigma_color_sq, source = _color_for_transformation(
                            data,
                            tc,
                            iter_cal_mags,
                            calibrated_filters,
                            std_col_prefix=std_col_prefix,
                            err_col_prefix=err_col_prefix,
                        )
                        data[f"cal_color_source_{filter_}"] = source
                        from .transform import calibrated_magnitude_variance

                        # m_cal = m_inst + T*color + ZP → first-order variance including
                        # 2·color·cov(T, ZP) from the joint linear fit.
                        var = calibrated_magnitude_variance(
                            inst_err,
                            color,
                            color_term=tc.color_term,
                            color_term_err=tc.color_term_err,
                            zero_point_err=tc.zero_point_err,
                            cov_tz=tc.cov_tz,
                            sigma_color_sq=sigma_color_sq,
                        )
                        data[f"{output_err_prefix}{filter_}"] = np.sqrt(
                            np.maximum(var, 0.0)
                        )
                    elif filter_ not in warned_no_err:
                        warnings.warn(
                            f"apply_transform_to_table: no column '{err_col}' for filter {filter_}; "
                            f"calibrated magnitude errors not written.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_err.add(filter_)
        return data

    @staticmethod
    def _inverse_variance_weighted_mean(
        values: np.ndarray,
        errors: np.ndarray,
    ) -> tuple[float, float]:
        """
        Combine measurements with inverse-variance weights.

        w_i = 1/σ_i², weighted mean ``Σ w_i x_i / Σ w_i``, formal uncertainty
        ``1/√(Σ w_i)``. Measurements without a finite positive σ are left out
        (a zero error would otherwise take all the weight); NaN if none is
        left. Reduces to the single-measurement case for one epoch.
        """
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        if values.size == 0:
            raise ValueError("_inverse_variance_weighted_mean: empty input")
        use = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        if not np.any(use):
            return float("nan"), float("nan")
        values = values[use]
        sigma = errors[use]
        w = 1.0 / (sigma**2)
        wsum = np.sum(w)
        mean = float(np.sum(w * values) / wsum)
        mean_err = float(1.0 / np.sqrt(wsum))
        return mean, mean_err

    @staticmethod
    def _weighted_linear_fit(x, y, weights):
        """
        Weighted least-squares fit y = T*x + ZP.

        Returns ``(T, ZP, T_err, ZP_err, cov_tz)``. Delegates to
        :func:`ost_photometry.analyze.calibration.transform.weighted_linear_fit`.
        """
        from .transform import weighted_linear_fit

        return weighted_linear_fit(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(weights, dtype=float),
        )



__all__ = ["DifferentialPhotometer"]
