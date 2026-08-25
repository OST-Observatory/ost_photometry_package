"""Pipeline configuration with step-specific sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

import astropy.units as u
from astropy.coordinates import EarthLocation

WcsMethod = Literal["astrometry", "astap", "twirl"]
CorrelationMethod = Literal["astropy", "own"]
PhotometryExtractionMethod = Literal["PSF", "APER"]
CalibrationStrategy = Literal["median_zp", "linear_fit"]
CalibrationGrouping = Literal["per_image", "per_night", "ensemble", "fixed"]
ExtinctionMode = Literal[
    "none",
    "tabulated",
    "from_comparison_stars",
    "from_value_airmass",
]
ExtinctionOrderName = Literal["first", "second"]
ColorTermFit = Literal["always", "auto", "never"]
UncertaintyMode = Literal["fit_errors", "flux_monte_carlo", "both"]

CALIBRATION_PRESETS: dict[str, dict[str, Any]] = {
    # Median zero-point per exposure (e.g. stacked multi-filter fields).
    "median_zp_per_image": {
        "calibration_strategy": "median_zp",
        "calibration_grouping": "per_image",
        "extinction_mode": "none",
        "color_term_fit": "never",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": True,
    },
    # Nightly linear color term + ZP (e.g. multi-epoch light curves).
    "linear_fit_per_night": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_night",
        "extinction_mode": "none",
        "color_term_fit": "auto",
        "derive_transform_from_data": True,
        "zp_subsample_statistic": False,
    },
    # Same as linear_fit_per_night, plus extinction from comparison stars.
    "linear_fit_per_night_extinction": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_night",
        "extinction_mode": "from_comparison_stars",
        "color_term_fit": "auto",
    },
    # WCS + extraction + intra-filter correlation; protect calibrators; no apply.
    "extract_protect_calibrators": {
        "protect_calibration_objects": True,
        "skip_calibration": True,
        "skip_correlation_inter": True,
        "skip_light_curve": True,
        "extinction_mode": "none",
    },
    # Ensemble linear transform / derive-transform over all epochs.
    "linear_fit_ensemble": {
        "calibration_strategy": "linear_fit",
        "derive_transform_from_data": True,
        "calibration_grouping": "ensemble",
        "extinction_mode": "none",
        "color_term_fit": "never",
    },
    # Use tabulated site extinction (bundled table when path is None).
    "tabulated_extinction": {
        "extinction_mode": "tabulated",
        "path_extinction_coefficients": None,
    },
}

# Deprecated aliases → canonical preset names.
CALIBRATION_PRESET_ALIASES: dict[str, str] = {
    "n2_stack": "median_zp_per_image",
    "c7_variable": "linear_fit_per_night",
    "c7_variable_extinction": "linear_fit_per_night_extinction",
    "mk_calib_trans": "extract_protect_calibrators",
    "mk_calib_calibrate": "linear_fit_ensemble",
    "ost_site": "tabulated_extinction",
}

@dataclass
class DiagnosticPlots:
    """QC figures under ``<output_dir>/diagnostics/``.

    Most checks are on by default so a normal pipeline run leaves enough plots
    to judge photometry, cross-match, and calibration quality. Disable
    individually via ``diagnostic_plots__<name>=False`` overrides. Growth curves
    stay off (pick a star / more specialized).
    """

    calibration_crossmatch_separation_histogram: bool = True
    photometry_mag_vs_error_scatter: bool = True
    #: Also write per-filter overview across all images when series length > 1.
    photometry_mag_vs_error_overview: bool = True
    photometry_radial_growth_curve: bool = False
    calibration_instrumental_vs_catalog: bool = True
    calibration_zeropoint_residual_histogram: bool = True
    calibration_zeropoint_residual_vs_color: bool = True
    calibration_color_check_cal_stars: bool = True
    correlation_inter_filter_separation_plot: bool = True
    #: Max individual inter-filter pair PDFs / geometry figures
    #: (``None`` = all; ``0`` = overview only). Geometry quiver plots share this cap.
    correlation_inter_filter_max_pair_plots: int | None = 25
    #: ΔJD / image-id pairing table + plot (same groups as ``exposure_pairing``).
    exposure_pairing_overview: bool = True
    combined_separation_histograms: bool = True


@dataclass
class WcsConfig:
    wcs_method: WcsMethod = "astrometry"
    force_wcs_determination: bool = False
    skip_wcs: bool = False


@dataclass
class ExtractionConfig:
    extraction_mode: Literal["single", "multi", "auto"] = "auto"
    n_cores_multiprocessing: int = 6
    reference_image_index: int = 0
    fwhm_object_psf: dict[str, float] | None = None
    #: Accepted FWHM range in pixels for automatic estimation (per-star filter).
    fwhm_estimate_min: float = 2.0
    fwhm_estimate_max: float = 15.0
    cosmic_ray_removal: bool = False
    limiting_contrast_rm_cosmics: float = 5.0
    read_noise: float = 8.0
    sigma_clipping_value: float = 4.5
    saturation_level: float = 65535.0
    plots_for_all_images: bool = False
    file_type_plots: str = "pdf"
    use_wcs_projection_for_star_maps: bool = True
    #: Simbad overlay on the reference image (``SimbadAnnotateStep``, not extraction).
    annotate_image: bool = True
    #: Same overlay in multi-image extraction mode (``SimbadAnnotateStep``).
    annotate_reference_image: bool = True
    magnitude_limit_image_annotation: float | None = None
    filter_magnitude_limit_image_annotation: str | None = None
    sigma_value_background_clipping: float = 5.0
    multiplier_background_rms: float = 5.0
    size_epsf_region: int = 25
    size_extraction_region_epsf: int = 11
    epsf_fitter: str = "TRFLSQFitter"
    n_iterations_eps_extraction: int = 1
    fraction_epsf_stars: float = 0.2
    maximum_n_eps_stars: int | None = 100
    oversampling_factor_epsf: int = 4
    max_n_iterations_epsf_determination: int = 7
    use_initial_positions_epsf: bool = True
    object_finder_method: str = "IRAF"
    multiplier_background_rms_epsf: float = 5.0
    multiplier_grouper_epsf: float = 2.0
    strict_cleaning_epsf_results: bool = True
    minimum_n_eps_stars: int = 15
    photometry_extraction_method: PhotometryExtractionMethod = "PSF"
    radius_aperture: float = 5.0
    inner_annulus_radius: float = 7.0
    outer_annulus_radius: float = 10.0
    radii_unit: str = "arcsec"
    strict_epsf_checks: bool = True
    transform_object_positions_to_reference: bool = False
    skip_extraction: bool = False

    def main_extract_kwargs(self, *, fwhm: float | None = None) -> dict[str, Any]:
        """Keyword arguments for ``main_extract`` (single-image path)."""
        return {
            "fwhm_object_psf": fwhm,
            "fwhm_estimate_min": self.fwhm_estimate_min,
            "fwhm_estimate_max": self.fwhm_estimate_max,
            "sigma_value_background_clipping": self.sigma_value_background_clipping,
            "multiplier_background_rms": self.multiplier_background_rms,
            "size_epsf_region": self.size_epsf_region,
            "size_extraction_region_epsf": self.size_extraction_region_epsf,
            "epsf_fitter": self.epsf_fitter,
            "n_iterations_eps_extraction": self.n_iterations_eps_extraction,
            "fraction_epsf_stars": self.fraction_epsf_stars,
            "maximum_n_eps_stars": self.maximum_n_eps_stars,
            "oversampling_factor_epsf": self.oversampling_factor_epsf,
            "max_n_iterations_epsf_determination": self.max_n_iterations_epsf_determination,
            "use_initial_positions_epsf": self.use_initial_positions_epsf,
            "object_finder_method": self.object_finder_method,
            "multiplier_background_rms_epsf": self.multiplier_background_rms_epsf,
            "multiplier_grouper_epsf": self.multiplier_grouper_epsf,
            "strict_cleaning_epsf_results": self.strict_cleaning_epsf_results,
            "minimum_n_eps_stars": self.minimum_n_eps_stars,
            "strict_epsf_checks": self.strict_epsf_checks,
            "photometry_extraction_method": self.photometry_extraction_method,
            "radius_aperture": self.radius_aperture,
            "inner_annulus_radius": self.inner_annulus_radius,
            "outer_annulus_radius": self.outer_annulus_radius,
            "radii_unit": self.radii_unit,
            "cosmic_ray_removal": self.cosmic_ray_removal,
            "limiting_contrast_rm_cosmics": self.limiting_contrast_rm_cosmics,
            "read_noise": self.read_noise,
            "sigma_clipping_value": self.sigma_clipping_value,
            "saturation_level": self.saturation_level,
            "plots_for_all_images": self.plots_for_all_images,
            "file_type_plots": self.file_type_plots,
            "use_wcs_projection_for_star_maps": self.use_wcs_projection_for_star_maps,
        }

    def extract_multiprocessing_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``extract_multiprocessing`` (multi-image path)."""
        kw = self.main_extract_kwargs()
        kw.pop("fwhm_object_psf", None)
        kw.pop("cosmic_ray_removal", None)
        kw.pop("limiting_contrast_rm_cosmics", None)
        kw.pop("read_noise", None)
        kw.pop("sigma_clipping_value", None)
        kw.pop("saturation_level", None)
        return kw


@dataclass
class CorrelationConfig:
    max_pixel_between_objects: int = 3
    ooi_correlation_strategy: int = 1
    cross_identification_limit: int = 1
    n_allowed_non_detections_object: int = 1
    expected_bad_image_fraction: float = 1.0
    protect_ooi: bool = True
    protect_calibration_objects: bool = False
    protected_object_ids: list[int] | None = None
    correlation_method: CorrelationMethod = "astropy"
    separation_limit: u.Quantity = 2.0 * u.arcsec
    duplicate_handling_object_identification: dict[str, str] | None = None
    verbose: bool = False
    skip_correlation_intra: bool = False
    skip_correlation_inter: bool = False


@dataclass
class CalibrationConfig:
    calibration_strategy: CalibrationStrategy = "median_zp"
    calibration_grouping: CalibrationGrouping = "per_image"
    extinction_mode: ExtinctionMode = "none"
    extinction_order: ExtinctionOrderName = "first"
    k_second: dict[str, float] | None = None
    color_term_fit: ColorTermFit = "auto"
    fit_sigma_clip: float = 2.5
    per_image_rolling_median_color_term: bool = False
    per_image_rolling_median_zero_point: bool = False
    per_image_rolling_mean_color_term: bool = False
    per_image_rolling_mean_zero_point: bool = False
    per_image_rolling_window: int = 3
    color_indices: dict[str, tuple[str, str]] | None = None
    exposure_pairing: Literal["jd_nearest", "index"] = "jd_nearest"
    exposure_jd_tolerance: float = 0.02
    debug_exposure_pairing: bool = False
    reference_filter: str | None = None
    calibration_summary_use_jd_x: bool = False
    zp_subsample_statistic: bool = True
    derive_transform_from_data: bool = False
    uncertainty_mode: UncertaintyMode = "fit_errors"
    calibration_catalog_radius_arcmin: float = 15.0
    calibration_catalog_mag_range: tuple[float, float] = (2.0, 18.5)
    debug_verify_ooi_global_ids: bool = False
    observatory_location: EarthLocation = field(
        default_factory=lambda: EarthLocation(
            lat=52.409184 * u.deg,
            lon=12.973185 * u.deg,
            height=39 * u.m,
        )
    )
    calibration_source: str = "APASS"
    path_calibration_file: str | None = None
    vizier_dict: dict[str, str] | None = None
    region_to_select_calibration_stars: object = None
    transformation_coefficients_dict: dict[str, float | str] | None = None
    distribution_samples: int = 1000
    aperture_radius: float = 4.0
    skip_calibration: bool = False


@dataclass
class PostProcessConfig:
    object_id: int | None = None
    extract_only_circular_region: bool = False
    region_radius: float = 600.0
    identify_cluster_gaia_data: bool = False
    clean_objs_using_pm: bool = False
    max_distance_cluster: float = 6.0
    find_cluster_para_set: int = 1
    cluster_selection_id: int | None = None
    convert_magnitudes: bool = False
    #: Preferred output filter family after calibration (``auto`` = calibrated set).
    output_filter_set: Literal["auto", "bessell", "sdss"] = "auto"
    #: Preferred Vega/AB system (``auto`` = catalog system; SDSS forces AB).
    output_magnitude_system: Literal["auto", "vega", "ab"] = "auto"
    #: Deprecated alias for output preferences (``SDSS`` / ``AB`` / ``BESSELL``).
    target_filter_system: str | None = None
    skip_cluster_region_step: bool = False
    skip_cluster_gaia_step: bool = False
    skip_cluster_pm_step: bool = False
    skip_magnitude_convert_step: bool = False
    skip_save_post_processed_magnitudes: bool = False
    skip_derive_limiting_magnitude: bool = False


@dataclass
class LightCurveConfig:
    skip_light_curve: bool = True
    plot_light_curve_objects_of_interest: bool = True
    plot_light_curve_calibration_objects: bool = False
    plot_light_curve_all_objects: bool = False
    light_curve_binning_factor: float | None = None
    light_curve_quantity: Literal["magnitude", "flux"] = "magnitude"
    light_curve_calibration_rows: Literal["auto", "transformed", "simple"] = "auto"


@dataclass
class HipsConfig:
    skip_hips_reference_subtraction: bool = True
    hips_reference_subtraction_filter: str | None = None
    hips_reference_subtraction_image_index: int = 0
    hips_reference_subtraction_wcs_method: WcsMethod | None = None
    hips_reference_subtraction_plot_comp: bool = True
    hips_reference_subtraction_hips_source: str = "CDS/P/DSS2/blue"
    hips_reference_subtraction_trim: tuple[int, int, int, int] | None = None
    hips_reference_subtraction_reuse_pipeline_wcs: bool = True
    hips_reference_subtraction_timeout_ms: int = 120_000
    hips_reference_subtraction_server: str = (
        "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    )
    hips_reference_subtraction_verbose: bool = False
    hips_reference_subtraction_hotpants_executable: str | None = None
    hips_reference_subtraction_hotpants_extra_args: tuple[str, ...] = field(
        default_factory=tuple
    )
    hips_reference_subtraction_output_filename: str = "hotpants_diff.fits"


@dataclass
class ExtinctionConfig:
    """Settings for :class:`~ost_photometry.analyze.pipeline.steps.extinction_fit.ExtinctionFitStep`."""

    path_extinction_coefficients: str | None = None
    extinction_night_id: str | None = None
    extinction_fit_mag_col: str = "mags_fit"
    extinction_fit_use_flux: bool = False
    extinction_coefficients_filename: str = "extinction_coefficients.json"


_SECTION_NAMES = (
    "wcs",
    "extraction",
    "correlation",
    "calibration",
    "post_process",
    "light_curve",
    "hips",
    "extinction",
)


_CONFIG_ALIASES = {"protect_reference_obj": "protect_ooi"}


class PipelineConfig:
    """
    Central pipeline configuration composed of step-specific sub-configs.

    Flat attribute access (``config.skip_extraction``) is supported for backward
    compatibility with existing scripts and ``run_pipeline(**kwargs)``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.wcs = WcsConfig()
        self.extraction = ExtractionConfig()
        self.correlation = CorrelationConfig()
        self.calibration = CalibrationConfig()
        self.post_process = PostProcessConfig()
        self.light_curve = LightCurveConfig()
        self.hips = HipsConfig()
        self.extinction = ExtinctionConfig()
        self.diagnostic_plots = DiagnosticPlots()
        self.apply_overrides(**kwargs)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineConfig:
        """Build config from a named calibration preset.

        Canonical names describe the mode, e.g. ``median_zp_per_image``,
        ``linear_fit_per_night``, ``extract_protect_calibrators``,
        ``linear_fit_ensemble``, ``tabulated_extinction``. Deprecated aliases
        (``n2_stack``, ``c7_variable``, ``mk_calib_trans``, ``ost_site``, …)
        still work and emit ``DeprecationWarning``.
        """
        import warnings

        resolved = CALIBRATION_PRESET_ALIASES.get(preset, preset)
        if resolved != preset:
            warnings.warn(
                f"Calibration preset {preset!r} is deprecated; use {resolved!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
        if resolved not in CALIBRATION_PRESETS:
            known = ", ".join(sorted(CALIBRATION_PRESETS))
            raise ValueError(f"Unknown calibration preset {preset!r}; known: {known}")
        kw = dict(CALIBRATION_PRESETS[resolved])
        if overrides:
            kw.update(overrides)
        return cls(**kw)

    def _sections(self) -> tuple[Any, ...]:
        return (
            self.wcs,
            self.extraction,
            self.correlation,
            self.calibration,
            self.post_process,
            self.light_curve,
            self.hips,
            self.extinction,
            self.diagnostic_plots,
        )

    def _find_section_for(self, name: str) -> Any | None:
        if name == "diagnostic_plots":
            return self.diagnostic_plots
        for section_name in _SECTION_NAMES:
            section = getattr(self, section_name)
            if hasattr(section, name):
                return section
        return None

    def __getattr__(self, name: str) -> Any:
        if name in _SECTION_NAMES or name == "diagnostic_plots":
            raise AttributeError(name)
        if name in _CONFIG_ALIASES:
            name = _CONFIG_ALIASES[name]
        section = self._find_section_for(name)
        if section is not None:
            return getattr(section, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SECTION_NAMES or name == "diagnostic_plots":
            super().__setattr__(name, value)
            return
        if name in _CONFIG_ALIASES:
            name = _CONFIG_ALIASES[name]
        section = None
        if hasattr(self, "wcs"):
            section = self._find_section_for(name)
        if section is not None:
            setattr(section, name, value)
            return
        super().__setattr__(name, value)

    def apply_overrides(self, **kwargs: Any) -> None:
        """Apply flat keyword overrides (used by ``run_pipeline``)."""
        diag_prefix = "diagnostic_plots__"
        for key, val in kwargs.items():
            if key.startswith(diag_prefix):
                sub = key[len(diag_prefix) :]
                if hasattr(self.diagnostic_plots, sub):
                    setattr(self.diagnostic_plots, sub, val)
                continue
            section = self._find_section_for(key)
            if section is None and key in _CONFIG_ALIASES:
                key = _CONFIG_ALIASES[key]
                section = self._find_section_for(key)
            if section is not None:
                setattr(section, key, val)

    def as_flat_dict(self) -> dict[str, Any]:
        """Serialize all sub-config fields to a flat dictionary."""
        out: dict[str, Any] = {}
        for section in self._sections():
            for f in fields(section):
                out[f.name] = getattr(section, f.name)
        return out
