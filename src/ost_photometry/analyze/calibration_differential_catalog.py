"""
Differential photometry calibration: APASSCatalog and PhotometryCalibrator.

Provides APASS-based differential photometry as alternative to legacy calibration.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
import astropy.units as u

from .extinction import (
    CoefficientMode,
    DEFAULT_EXTINCTION,
    ExtinctionCoefficients,
    ExtinctionCorrector,
    ExtinctionOrder,
    calculate_airmass,
    extinction_airmass_ready,
    fit_extinction_from_comparison_stars,
)
from .warnings_types import OstPhotometryAnalyzeWarning


@dataclass
class TransformationCoefficients:
    """Transformation coefficients for one filter."""

    filter_name: str
    color_term: float
    color_term_err: float = 0.0
    zero_point: float = 0.0
    zero_point_err: float = 0.0
    color_index_filters: Tuple[str, str] = ("B", "V")
    n_stars_used: int = 0
    rms_residual: float = 0.0

    def __repr__(self) -> str:
        ci = f"({self.color_index_filters[0]}-{self.color_index_filters[1]})"
        return (
            f"{self.filter_name}: T={self.color_term:.4f}±{self.color_term_err:.4f}, "
            f"ZP={self.zero_point:.4f}±{self.zero_point_err:.4f}, CI={ci}"
        )


def _filters_with_instrumental_mags(
    table: Table,
    filters: List[str],
    mag_col_prefix: str = "mag_",
) -> List[str]:
    """Return filters that have an instrumental column ``{mag_col_prefix}{f}`` in ``table``."""
    return [f for f in filters if f"{mag_col_prefix}{f}" in table.colnames]


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    identifier: str
    timestamp: Optional[Time] = None
    extinction: Dict[str, ExtinctionCoefficients] = field(default_factory=dict)
    transformation: Dict[str, TransformationCoefficients] = field(default_factory=dict)
    n_comparison_stars: int = 0
    quality_flag: str = "OK"
    notes: str = ""


class APASSCatalog:
    """
    APASS catalog interface for comparison stars.

    APASS provides B, V, g', r', i'. R, I converted via empirical relations.
    """

    #   Transformation form SDSS to Johnson-Cousins system
    #   Based on Lupton (2005):
    #       - https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
    #       - https://www.aavso.org/transformations-sdss-magnitudes
    # Lupton (2005): SDSS r',i' -> Johnson-Cousins R,I
    # https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Lupton2005
    APASS_TRANSFORMS = {
        "R_from_ri": lambda r, i: r - 0.2936 * (r - i) - 0.1439,
        "I_from_ri": lambda r, i: r - 1.2444 * (r - i) - 0.3820,
    }
    # Partial derivatives for error propagation: dR/dr, dR/di, dI/dr, dI/di
    APASS_TRANSFORM_GRADIENTS = {
        "R_from_ri": (0.7064, 0.2936),   # dR/dr=1-0.2936, dR/di=0.2936
        "I_from_ri": (-0.2444, 1.2444),  # dI/dr=1-1.2444, dI/di=1.2444
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def query_region(
        self,
        center: SkyCoord,
        radius: u.Quantity,
        mag_limit: float = 16.0,
    ) -> Table:
        """Query APASS for a sky region."""
        from astroquery.vizier import Vizier

        v = Vizier(
            columns=[
                "RAJ2000",
                "DEJ2000",
                "Bmag",
                "Vmag",
                "g'mag",
                "r'mag",
                "i'mag",
                "e_Bmag",
                "e_Vmag",
                "e_g'mag",
                "e_r'mag",
                "e_i'mag",
            ],
            row_limit=-1,
        )
        result = v.query_region(center, radius=radius, catalog="II/336/apass9")

        if not result:
            warnings.warn(
                "No APASS data found for region.",
                category=OstPhotometryAnalyzeWarning,
            )
            return Table()

        apass = result[0]
        if "Vmag" in apass.colnames:
            apass = apass[apass["Vmag"] < mag_limit]

        output = Table()
        output["ra"] = apass["RAJ2000"]
        output["dec"] = apass["DEJ2000"]
        output["apass_id"] = [f"APASS_{i:06d}" for i in range(len(apass))]

        if "Bmag" in apass.colnames:
            output["mag_std_B"] = apass["Bmag"]
            output["err_std_B"] = (
                apass["e_Bmag"] if "e_Bmag" in apass.colnames else 0.02
            )
        if "Vmag" in apass.colnames:
            output["mag_std_V"] = apass["Vmag"]
            output["err_std_V"] = (
                apass["e_Vmag"] if "e_Vmag" in apass.colnames else 0.02
            )

        if "r'mag" in apass.colnames and "i'mag" in apass.colnames:
            r_sloan = np.array(apass["r'mag"])
            i_sloan = np.array(apass["i'mag"])
            # Error in r' and i' magnitudes
            # Fallback to 0.02 mag if not available
            err_r = (
                np.array(apass["e_r'mag"])
                if "e_r'mag" in apass.colnames
                else np.full(len(apass), 0.02)
            )
            err_i = (
                np.array(apass["e_i'mag"])
                if "e_i'mag" in apass.colnames
                else np.full(len(apass), 0.02)
            )
            valid = np.isfinite(r_sloan) & np.isfinite(i_sloan)
            R_johnson = np.full(len(apass), np.nan)
            I_johnson = np.full(len(apass), np.nan)
            err_R = np.full(len(apass), np.nan)
            err_I = np.full(len(apass), np.nan)
            R_johnson[valid] = self.APASS_TRANSFORMS["R_from_ri"](
                r_sloan[valid], i_sloan[valid]
            )
            I_johnson[valid] = self.APASS_TRANSFORMS["I_from_ri"](
                r_sloan[valid], i_sloan[valid]
            )
            # Error propagation: sigma_f^2 = (df/dr)^2 * sigma_r^2 + (df/di)^2 * sigma_i^2
            dR_dr, dR_di = self.APASS_TRANSFORM_GRADIENTS["R_from_ri"]
            dI_dr, dI_di = self.APASS_TRANSFORM_GRADIENTS["I_from_ri"]
            err_R[valid] = np.sqrt(
                (dR_dr * err_r[valid]) ** 2 + (dR_di * err_i[valid]) ** 2
            )
            err_I[valid] = np.sqrt(
                (dI_dr * err_r[valid]) ** 2 + (dI_di * err_i[valid]) ** 2
            )
            output["mag_std_R"] = R_johnson
            output["mag_std_I"] = I_johnson
            output["err_std_R"] = err_R
            output["err_std_I"] = err_I

        for band in ["g", "r", "i"]:
            col = f"{band}'mag"
            if col in apass.colnames:
                output[f"mag_std_{band}"] = apass[col]

        return output

    def crossmatch(
        self,
        sources: Table,
        apass_data: Table,
        ra_col: str = "ra",
        dec_col: str = "dec",
        match_radius: u.Quantity = 2.0 * u.arcsec,
    ) -> Table:
        """Crossmatch sources with APASS. One-to-one matching by quality (separation).

        Each source gets at most one APASS star and vice versa. When multiple
        candidates exist, the pair with smallest separation is kept.
        """
        from astropy.coordinates import matching

        from . import utilities

        source_coords = SkyCoord(sources[ra_col], sources[dec_col], unit="deg")
        apass_coords = SkyCoord(apass_data["ra"], apass_data["dec"], unit="deg")

        idx_src, idx_apass, sep, _ = matching.search_around_sky(
            source_coords, apass_coords, match_radius
        )
        sep_arcsec = np.asarray(sep.arcsec, dtype=float)

        if len(idx_src) == 0:
            result = sources.copy()
            result["match_sep_arcsec"] = np.full(len(result), np.nan)
            for col in apass_data.colnames:
                if col in ["ra", "dec"] or not np.issubdtype(
                    apass_data[col].dtype, np.number
                ):
                    continue
                result[col] = np.full(len(result), np.nan, dtype=float)
            return result

        # One-to-one: remove duplicate sources (keep closest APASS per source)
        idx_src, sep_arcsec, idx_apass = utilities.clear_duplicates(
            idx_src, sep_arcsec, idx_apass
        )
        # One-to-one: remove duplicate APASS (keep closest source per APASS)
        idx_apass, sep_arcsec, idx_src = utilities.clear_duplicates(
            idx_apass, sep_arcsec, idx_src
        )

        # Map source index -> APASS index and separation for matched pairs
        apass_idx_for_source = np.full(len(sources), -1, dtype=np.intp)
        sep_for_source = np.full(len(sources), np.nan)
        apass_idx_for_source[idx_src] = idx_apass
        sep_for_source[idx_src] = sep_arcsec
        good_match = apass_idx_for_source >= 0

        result = sources.copy()
        result["match_sep_arcsec"] = sep_for_source

        for col in apass_data.colnames:
            if col in ["ra", "dec"]:
                continue
            if not np.issubdtype(apass_data[col].dtype, np.number):
                continue
            matched_vals = apass_data[col][apass_idx_for_source[good_match]]
            if hasattr(matched_vals, "value"):
                matched_vals = np.asarray(matched_vals.value, dtype=float)
            else:
                matched_vals = np.asarray(matched_vals, dtype=float)
            new_col = np.full(len(result), np.nan, dtype=float)
            new_col[good_match] = matched_vals
            result[col] = new_col
        return result


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
        color_indices: Optional[Dict] = None,
        extinction_corrector: Optional[ExtinctionCorrector] = None,
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
        self.calibrations: Dict[str, CalibrationResult] = {}

    def calibrate_epoch(
        self,
        data: Table,
        epoch_id: str,
        filters: List[str],
        comparison_mask: np.ndarray,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        airmass_col: str = "airmass",
        sigma_clip: float = 2.5,
        min_comparisons: int = 3,
        determine_color_terms: bool = True,
        output_dir: Optional[str] = None,
        file_type: str = "pdf",
    ) -> CalibrationResult:
        """
        Calibrate a single calibration epoch with ensemble photometry.

        For each filter, fits m_std - m_inst = T * color + ZP using comparison stars.
        T (color term) corrects for passband mismatch; ZP (zero point) sets the scale.
        Applies extinction correction first, then iterative sigma-clip to reject outliers.
        Requires color spread > 0.1 mag for reliable T; otherwise uses ZP-only.

        Parameters
        ----------
        output_dir : str, optional
            If provided, save transformation fit plots to output_dir/calibration/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        """
        result = CalibrationResult(identifier=epoch_id)
        plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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
            data, filters_use, airmass_col
        ):
            data = self.extinction.correct(
                data,
                airmass_col=airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,
                filters=filters_use,
                inplace=False,
                catalog_color_prefix=std_col_prefix,
            )

        comps = data[comparison_mask]

        for filt in filters_use:
            # --- Column names for instrumental and standard magnitudes ---
            inst_col = f"{mag_col_prefix}{filt}"
            std_col = f"{std_col_prefix}{filt}"

            if inst_col not in comps.colnames or std_col not in comps.colnames:
                missing = [c for c in (inst_col, std_col) if c not in comps.colnames]
                warnings.warn(
                    f"[{epoch_id}] Filter {filt} skipped: missing columns {missing}",
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
                    f"[{epoch_id}] Filter {filt} skipped: only {np.sum(valid)} valid "
                    f"comparisons (need >= {min_comparisons})",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )
                continue

            # --- Color index for transformation: m_std - m_inst = T * (CI1 - CI2) + ZP ---
            ci_filters = self.color_indices.get(filt, ("B", "V"))
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
                    f"[{epoch_id}] Filter {filt} skipped: after color-index cut "
                    f"only {np.sum(valid)} valid (need >= {min_comparisons})",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=1,
                )
                continue

            # --- Iterative sigma-clip fit: T (color term), ZP (zero point) ---
            mask = valid.copy()
            T, ZP, T_err, ZP_err = 0.0, 0.0, 0.0, 0.0
            rms = 0.0
            warned_color_spread = False

            for _ in range(5):
                c = color_std[mask]
                if determine_color_terms and has_color and np.std(c) > 0.1:
                    T, ZP, T_err, ZP_err = self._weighted_linear_fit(
                        c, m_std[mask] - m_inst[mask], np.ones(np.sum(mask))
                    )
                else:
                    if determine_color_terms and has_color and not warned_color_spread:
                        warnings.warn(
                            f"[{epoch_id}] Filter {filt}: color spread std={np.std(c):.3f} mag "
                            f"<= 0.1, using ZP-only fit (no color term)",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=1,
                        )
                        warned_color_spread = True
                    ZP = np.median((m_std - m_inst)[mask])
                    ZP_err = np.std((m_std - m_inst)[mask]) / np.sqrt(np.sum(mask))

                all_residuals = (m_std - m_inst) - (T * color_std + ZP)
                rms = np.nanstd(all_residuals[mask])
                new_mask = valid & (np.abs(all_residuals) < sigma_clip * rms)
                if np.sum(new_mask) == np.sum(mask) or np.sum(new_mask) < min_comparisons:
                    break
                mask = new_mask

            result.transformation[filt] = TransformationCoefficients(
                filter_name=filt,
                color_term=T,
                color_term_err=T_err,
                zero_point=ZP,
                zero_point_err=ZP_err,
                color_index_filters=ci_filters,
                n_stars_used=int(np.sum(mask)),
                rms_residual=rms,
            )
            if output_dir:
                plot_data[filt] = (color_std, m_std - m_inst, mask)

        if output_dir and plot_data:
            from . import plots
            plots.plot_calibration_transformation(
                output_dir,
                epoch_id,
                plot_data,
                result.transformation,
                file_type=file_type,
            )

        result.n_comparison_stars = int(np.sum(comparison_mask))
        self.calibrations[epoch_id] = result
        return result

    def calibrate_frame(
        self,
        data: Table,
        frame_id: str,
        filters: List[str],
        comparison_mask: np.ndarray,
        **kwargs,
    ) -> CalibrationResult:
        """Deprecated: use :meth:`calibrate_epoch` (same API, ``frame_id`` = ``epoch_id``)."""
        return self.calibrate_epoch(
            data, frame_id, filters, comparison_mask, **kwargs
        )

    def calibrate_night(
        self,
        epochs: Dict[str, Table],
        filters: List[str],
        comparison_mask_func,
        night_id: str = "night",
        output_dir: Optional[str] = None,
        file_type: str = "pdf",
        inverse_variance_min_error: float = 1e-10,
        **kwargs,
    ) -> CalibrationResult:
        """
        Calibrate a full night by averaging coefficients over all calibration epochs.

        Calibrates each epoch individually, then combines T and ZP across epochs
        with inverse-variance weights (w_i = 1/σ_i²). The reported uncertainty
        on the combined value is 1/sqrt(Σ w_i). Per-epoch σ are
        ``color_term_err`` and ``zero_point_err``. Very small σ are floored to
        ``min_error`` to avoid division by zero.

        Assumes stable atmospheric conditions and instrument response over the night.

        Parameters
        ----------
        inverse_variance_min_error : float
            Minimum σ used as weight floor (mag). Avoids infinite weights when
            reported errors are zero or tiny.
        """
        epoch_results = []
        for epoch_id, data in epochs.items():
            mask = comparison_mask_func(data)
            try:
                result = self.calibrate_epoch(
                    data, epoch_id, filters, mask,
                    output_dir=output_dir,
                    file_type=file_type,
                    **kwargs,
                )
                epoch_results.append(result)
            except Exception as e:
                warnings.warn(
                    f"Epoch {epoch_id} failed: {e}",
                    category=OstPhotometryAnalyzeWarning,
                )

        if not epoch_results:
            raise ValueError("No epochs calibrated successfully!")

        if output_dir and len(epoch_results) > 1:
            from . import plots
            plots.plot_calibration_night_summary(
                output_dir,
                [fr.identifier for fr in epoch_results],
                [fr.transformation for fr in epoch_results],
                filters,
                file_type=file_type,
            )

        night_result = CalibrationResult(identifier=night_id)
        for filt in filters:
            fr_with_filt = [fr for fr in epoch_results if filt in fr.transformation]
            if not fr_with_filt:
                continue
            T_vals = np.array(
                [fr.transformation[filt].color_term for fr in fr_with_filt],
                dtype=float,
            )
            ZP_vals = np.array(
                [fr.transformation[filt].zero_point for fr in fr_with_filt],
                dtype=float,
            )
            T_errs = np.array(
                [fr.transformation[filt].color_term_err for fr in fr_with_filt],
                dtype=float,
            )
            ZP_errs = np.array(
                [fr.transformation[filt].zero_point_err for fr in fr_with_filt],
                dtype=float,
            )
            T_mean, T_err = self._inverse_variance_weighted_mean(
                T_vals, T_errs, min_error=inverse_variance_min_error
            )
            ZP_mean, ZP_err = self._inverse_variance_weighted_mean(
                ZP_vals, ZP_errs, min_error=inverse_variance_min_error
            )
            night_result.transformation[filt] = TransformationCoefficients(
                filter_name=filt,
                color_term=T_mean,
                color_term_err=T_err,
                zero_point=ZP_mean,
                zero_point_err=ZP_err,
                color_index_filters=fr_with_filt[0].transformation[filt].color_index_filters,
                n_stars_used=sum(fr.transformation[filt].n_stars_used for fr in fr_with_filt),
                rms_residual=np.mean(
                    [fr.transformation[filt].rms_residual for fr in fr_with_filt]
                ),
            )
        self.calibrations[night_id] = night_result
        return night_result

    def apply_calibration(
        self,
        data: Table,
        calibration: Union[str, CalibrationResult],
        filters: Optional[List[str]] = None,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        output_prefix: str = "mag_cal_",
        err_col_prefix: str = "err_",
        output_err_prefix: str = "err_cal_",
        airmass_col: str = "airmass",
        max_iterations: int = 10,
        inplace: bool = False,
    ) -> Table:
        """
        Apply calibration: m_cal = m_inst + T * color + ZP.

        First corrects extinction, then applies transformation. Iterative when
        color index uses magnitudes from other filters (e.g. B-V needs calibrated
        B and V); converges when changes < 0.0001 mag.

        Notes
        -----
        Reference magnitudes (e.g. APASS ``mag_std_*``) are used only when *fitting*
        T and ZP in ``calibrate_epoch`` / ``calibrate_night``. Here, the same
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
            data, filters, airmass_col
        ):
            data = self.extinction.correct(
                data,
                airmass_col=airmass_col,
                mag_col_prefix=mag_col_prefix,
                output_prefix=mag_col_prefix,
                filters=filters,
                inplace=True,
                catalog_color_prefix=std_col_prefix,
            )
        # Running calibrated mags per filter (starts as instrumental, not APASS)
        iter_cal_mags: Dict[str, np.ndarray] = {}
        for filt in filters:
            col = f"{mag_col_prefix}{filt}"
            if col in data.colnames:
                iter_cal_mags[filt] = np.array(data[col], dtype=float)

        warned_no_transform: set[str] = set()
        warned_no_inst: set[str] = set()
        warned_no_color: set[str] = set()
        converged = False
        last_max_change = 0.0

        for _ in range(max_iterations):
            max_change = 0.0
            for filt in filters:
                if filt not in cal.transformation:
                    if filt not in warned_no_transform:
                        warnings.warn(
                            f"apply_calibration: filter {filt} has no transformation "
                            f"in calibration '{getattr(cal, 'identifier', calibration)}'; skipping.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_transform.add(filt)
                    continue
                tc = cal.transformation[filt]
                inst_col = f"{mag_col_prefix}{filt}"
                if inst_col not in data.colnames:
                    if filt not in warned_no_inst:
                        warnings.warn(
                            f"apply_calibration: missing instrumental column '{inst_col}' "
                            f"for filter {filt}; skipping.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_inst.add(filt)
                    continue
                m_inst = np.array(data[inst_col], dtype=float)
                ci_f1, ci_f2 = tc.color_index_filters
                if ci_f1 in iter_cal_mags and ci_f2 in iter_cal_mags:
                    color = iter_cal_mags[ci_f1] - iter_cal_mags[ci_f2]
                else:
                    c1, c2 = f"{mag_col_prefix}{ci_f1}", f"{mag_col_prefix}{ci_f2}"
                    if c1 in data.colnames and c2 in data.colnames:
                        color = np.array(data[c1]) - np.array(data[c2])
                    else:
                        s1, s2 = f"{std_col_prefix}{ci_f1}", f"{std_col_prefix}{ci_f2}"
                        if s1 in data.colnames and s2 in data.colnames:
                            color = np.asarray(data[s1], dtype=float) - np.asarray(
                                data[s2], dtype=float
                            )
                            color = np.where(np.isfinite(color), color, 0.0)
                        else:
                            color = np.zeros(len(data))
                            if abs(tc.color_term) > 1e-12 and filt not in warned_no_color:
                                missing = [c for c in (c1, c2) if c not in data.colnames]
                                warnings.warn(
                                    f"apply_calibration: filter {filt} needs color "
                                    f"({ci_f1}-{ci_f2}) but columns {missing} missing, "
                                    f"no catalog columns {s1}/{s2}, and calibrated mags for "
                                    f"{ci_f1}/{ci_f2} not yet available; using color=0.",
                                    category=OstPhotometryAnalyzeWarning,
                                    stacklevel=2,
                                )
                                warned_no_color.add(filt)
                m_cal_new = m_inst + tc.color_term * color + tc.zero_point
                if filt in iter_cal_mags:
                    max_change = max(max_change, np.nanmax(np.abs(m_cal_new - iter_cal_mags[filt])))
                iter_cal_mags[filt] = m_cal_new
            last_max_change = max_change
            if max_change < 0.0001:
                converged = True
                break

        if not converged and max_iterations > 0:
            cal_id = getattr(cal, "identifier", calibration)
            warnings.warn(
                f"apply_calibration: calibration '{cal_id}' did not converge after "
                f"{max_iterations} iterations (last max_change={last_max_change:.6f} mag).",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )

        warned_no_err: set[str] = set()
        for filt in filters:
            if filt in iter_cal_mags:
                data[f"{output_prefix}{filt}"] = iter_cal_mags[filt]
                if filt in cal.transformation:
                    err_col = f"{err_col_prefix}{filt}"
                    if err_col in data.colnames:
                        tc = cal.transformation[filt]
                        inst_err = np.asarray(data[err_col], dtype=float)
                        n = len(inst_err)
                        ci_f1, ci_f2 = tc.color_index_filters
                        # Final color (same logic as calibration iteration)
                        if ci_f1 in iter_cal_mags and ci_f2 in iter_cal_mags:
                            color = iter_cal_mags[ci_f1] - iter_cal_mags[ci_f2]
                        else:
                            c1 = f"{mag_col_prefix}{ci_f1}"
                            c2 = f"{mag_col_prefix}{ci_f2}"
                            if c1 in data.colnames and c2 in data.colnames:
                                color = (
                                    np.asarray(data[c1], dtype=float)
                                    - np.asarray(data[c2], dtype=float)
                                )
                            else:
                                s1 = f"{std_col_prefix}{ci_f1}"
                                s2 = f"{std_col_prefix}{ci_f2}"
                                if s1 in data.colnames and s2 in data.colnames:
                                    color = np.asarray(data[s1], dtype=float) - np.asarray(
                                        data[s2], dtype=float
                                    )
                                    color = np.where(np.isfinite(color), color, 0.0)
                                else:
                                    color = np.zeros(n, dtype=float)
                        # σ_color from instrumental errors on the two color filters (approx.)
                        e1c = f"{err_col_prefix}{ci_f1}"
                        e2c = f"{err_col_prefix}{ci_f2}"
                        if e1c in data.colnames and e2c in data.colnames:
                            sigma_color_sq = (
                                np.asarray(data[e1c], dtype=float) ** 2
                                + np.asarray(data[e2c], dtype=float) ** 2
                            )
                        else:
                            sigma_color_sq = np.zeros(n, dtype=float)
                        T, sT, sZP = (
                            tc.color_term,
                            tc.color_term_err,
                            tc.zero_point_err,
                        )
                        # m_cal = m_inst + T*color + ZP  →  first-order, uncorrelated:
                        # σ² ≈ σ_inst² + σ_ZP² + (color·σ_T)² + T²·σ_color²
                        # (cov(T,ZP) from the same fit is neglected)
                        var = (
                            inst_err**2
                            + sZP**2
                            + (color * sT) ** 2
                            + (T**2) * sigma_color_sq
                        )
                        data[f"{output_err_prefix}{filt}"] = np.sqrt(np.maximum(var, 0.0))
                    elif filt not in warned_no_err:
                        warnings.warn(
                            f"apply_calibration: no column '{err_col}' for filter {filt}; "
                            f"calibrated magnitude errors not written.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                        warned_no_err.add(filt)
        return data

    @staticmethod
    def _inverse_variance_weighted_mean(
        values: np.ndarray,
        errors: np.ndarray,
        min_error: float = 1e-10,
    ) -> Tuple[float, float]:
        """
        Combine measurements with inverse-variance weights.

        w_i = 1/σ_i² (σ_i floored to ``min_error``), weighted mean
        ``Σ w_i x_i / Σ w_i``, formal uncertainty ``1/√(Σ w_i)``.
        Reduces to the single-measurement case for one epoch.
        """
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        if values.size == 0:
            raise ValueError("_inverse_variance_weighted_mean: empty input")
        sigma = np.maximum(errors, min_error)
        w = 1.0 / (sigma**2)
        wsum = np.sum(w)
        mean = float(np.sum(w * values) / wsum)
        mean_err = float(1.0 / np.sqrt(wsum))
        return mean, mean_err

    @staticmethod
    def _weighted_linear_fit(x, y, weights):
        """
        Weighted least-squares fit y = a*x + b.

        Returns (a, b, a_err, b_err). Used for m_std - m_inst vs color to get
        T (slope) and ZP (intercept). Errors from residual variance.
        """
        W, Wx, Wy = np.sum(weights), np.sum(weights * x), np.sum(weights * y)
        Wxx, Wxy = np.sum(weights * x**2), np.sum(weights * x * y)
        denom = W * Wxx - Wx**2
        if abs(denom) < 1e-10:
            return 0.0, np.mean(y), 0.0, np.std(y)
        a = (W * Wxy - Wx * Wy) / denom
        b = (Wxx * Wy - Wx * Wxy) / denom
        residuals = y - (a * x + b)
        n = len(x)
        var = np.sum(weights * residuals**2) / (n - 2) if n > 2 else 0
        a_err = np.sqrt(var * W / denom) if denom > 0 else 0
        b_err = np.sqrt(var * Wxx / denom) if denom > 0 else 0
        return a, b, a_err, b_err


class PhotometryCalibrator:
    """
    Main class for differential photometry calibration.

    Combines APASS query, extinction correction, differential photometry.
    """

    def __init__(
        self,
        mode: CoefficientMode = CoefficientMode.PER_NIGHT,
        extinction_order: ExtinctionOrder = ExtinctionOrder.FIRST,
        extinction_coefficients: Optional[Dict[str, ExtinctionCoefficients]] = None,
        observatory_location: Optional[object] = None,
        color_indices: Optional[Dict] = None,
    ):
        self.mode = mode
        self.location = observatory_location
        self.extinction = ExtinctionCorrector(
            coefficients=extinction_coefficients, order=extinction_order
        )
        self.apass = APASSCatalog()
        self.apass_data: Optional[Table] = None
        self.photometer = DifferentialPhotometer(
            color_indices=color_indices,
            extinction_corrector=self.extinction,
        )
        self.epochs: Dict[str, Table] = {}
        self.epoch_metadata: Dict[str, dict] = {}
        self.fixed_calibration: Optional[CalibrationResult] = None

    def setup_apass(
        self,
        center: SkyCoord,
        radius: u.Quantity = 15 * u.arcmin,
        mag_limit: float = 16.0,
    ):
        """Load APASS catalog for field."""
        self.apass_data = self.apass.query_region(center, radius, mag_limit)

    def add_epoch(
        self,
        epoch_id: str,
        data: Table,
        obstime: Optional[Time] = None,
        airmass: Optional[float] = None,
        filter_obstimes: Optional[Dict[str, Time]] = None,
        ra_col: str = "ra",
        dec_col: str = "dec",
    ):
        """
        Add a calibration epoch table (multi-band rows, aligned ``id``).

        Expects ``mag_<filter>`` / ``err_<filter>``. If ``airmass_<filter>`` columns
        are missing, they are filled from ``filter_obstimes[f]``, then ``obstime``,
        then scalar ``airmass``, else 1.0. A mean ``airmass`` column is added when
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
            t_use: Optional[Time] = None
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
                data[col_a] = 1.0

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
                data["airmass"] = 1.0

        if self.apass_data is not None and len(self.apass_data) > 0:
            data = self.apass.crossmatch(data, self.apass_data, ra_col, dec_col)

        self.epochs[epoch_id] = data
        self.epoch_metadata[epoch_id] = {
            "obstime": obstime,
            "filter_obstimes": filter_obstimes,
            "airmass_mean": float(np.nanmean(data["airmass"]))
            if "airmass" in data.colnames
            else None,
        }

    def add_frame(
        self,
        frame_id: str,
        data: Table,
        **kwargs,
    ) -> None:
        """Deprecated: use :meth:`add_epoch` (same API, ``frame_id`` = ``epoch_id``)."""
        self.add_epoch(frame_id, data, **kwargs)

    def fit_extinction_from_epochs(
        self,
        mag_col_prefix: str = "mag_",
        std_col_prefix: str = "mag_std_",
        airmass_col: str = "airmass",
        output_dir: Optional[str] = None,
        file_type: str = "pdf",
    ) -> Dict[str, ExtinctionCoefficients]:
        """
        Fit extinction coefficients from APASS-matched comparison stars in epochs.

        Call after add_epoch() for all epochs. Updates internal ExtinctionCorrector.
        Requires epoch tables with mag_std_* from APASS crossmatch.

        Parameters
        ----------
        output_dir : str, optional
            If provided, save diagnostic plots to output_dir/extinction_fit/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        """
        fitted = fit_extinction_from_comparison_stars(
            self.epochs,
            mag_col_prefix=mag_col_prefix,
            std_col_prefix=std_col_prefix,
            airmass_col=airmass_col,
            output_dir=output_dir,
            file_type=file_type,
        )
        if fitted:
            self.extinction.coefficients.update(fitted)
        return fitted

    def set_fixed_coefficients(
        self, coefficients: Dict[str, TransformationCoefficients]
    ):
        """Set fixed coefficients for FIXED mode."""
        self.fixed_calibration = CalibrationResult(
            identifier="fixed", transformation=coefficients
        )

    def calibrate(
        self,
        filters: List[str],
        comparison_selector=None,
        determine_color_terms: bool = True,
        min_comparisons: int = 5,
        sigma_clip: float = 2.5,
        output_dir: Optional[str] = None,
        file_type: str = "pdf",
        inverse_variance_min_error: float = 1e-10,
    ) -> Dict[str, CalibrationResult]:
        """
        Run calibration.

        Parameters
        ----------
        output_dir : str, optional
            If provided, save calibration diagnostic plots to output_dir/calibration/.
        file_type : str
            Plot file format when output_dir is set. Default is ``pdf``.
        inverse_variance_min_error : float
            For ``PER_NIGHT`` mode: floor on per-epoch σ when combining T and ZP
            with inverse-variance weights (see :meth:`calibrate_night`).
        """
        if not self.epochs:
            raise ValueError("No epochs added!")

        if comparison_selector is None:
            def comparison_selector(table):
                mask = np.ones(len(table), dtype=bool)
                for filt in filters:
                    std_col = f"mag_std_{filt}"
                    if std_col in table.colnames:
                        mask &= np.isfinite(table[std_col])
                return mask

        results = {}
        if self.mode == CoefficientMode.FIXED:
            if self.fixed_calibration is None:
                raise ValueError("FIXED mode but no coefficients set!")
            for epoch_id in self.epochs:
                results[epoch_id] = self.fixed_calibration
        elif self.mode == CoefficientMode.PER_IMAGE:
            for epoch_id, data in self.epochs.items():
                mask = comparison_selector(data)
                result = self.photometer.calibrate_epoch(
                    data, epoch_id, filters, mask,
                    determine_color_terms=determine_color_terms,
                    min_comparisons=min_comparisons,
                    sigma_clip=sigma_clip,
                    output_dir=output_dir,
                    file_type=file_type,
                )
                results[epoch_id] = result
        elif self.mode == CoefficientMode.PER_NIGHT:
            result = self.photometer.calibrate_night(
                self.epochs, filters, comparison_selector,
                night_id="night_combined",
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip,
                output_dir=output_dir,
                file_type=file_type,
                inverse_variance_min_error=inverse_variance_min_error,
            )
            for epoch_id in self.epochs:
                results[epoch_id] = result
        elif self.mode == CoefficientMode.ENSEMBLE:
            combined = vstack(list(self.epochs.values()))
            combined_mask = comparison_selector(combined)
            result = self.photometer.calibrate_epoch(
                combined, "ensemble", filters, combined_mask,
                determine_color_terms=determine_color_terms,
                min_comparisons=min_comparisons,
                sigma_clip=sigma_clip,
            )
            for epoch_id in self.epochs:
                results[epoch_id] = result

        self._calibration_results = results
        return results

    def get_calibrated_photometry(
        self,
        output_prefix: str = "mag_cal_",
        target_selector=None,
    ) -> Table:
        """Apply calibration and return calibrated table."""
        if not hasattr(self, "_calibration_results"):
            raise ValueError("Call calibrate() first!")

        all_results = []
        for epoch_id, data in self.epochs.items():
            cal = self._calibration_results[epoch_id]
            calibrated = self.photometer.apply_calibration(
                data, cal, output_prefix=output_prefix, inplace=False
            )
            calibrated["epoch_id"] = epoch_id
            if target_selector is not None:
                calibrated = calibrated[target_selector(calibrated)]
            all_results.append(calibrated)
        return vstack(all_results) if all_results else Table()

    @property
    def frames(self) -> Dict[str, Table]:
        """Deprecated alias for :attr:`epochs`."""
        return self.epochs

    @property
    def frame_metadata(self) -> Dict[str, dict]:
        """Deprecated alias for :attr:`epoch_metadata`."""
        return self.epoch_metadata

    fit_extinction_from_frames = fit_extinction_from_epochs


__all__ = [
    "APASSCatalog",
    "CalibrationResult",
    "DifferentialPhotometer",
    "PhotometryCalibrator",
    "TransformationCoefficients",
]
