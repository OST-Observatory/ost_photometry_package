"""Pipeline configuration with step-specific sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Tuple

import astropy.units as u
from astropy.coordinates import EarthLocation

WcsMethod = Literal["astrometry", "astap", "twirl"]
CorrelationMethod = Literal["astropy", "own"]
PhotometryExtractionMethod = Literal["PSF", "APER"]
DifferentialCoefficientMode = Literal["per_image", "per_night", "fixed", "ensemble"]
DifferentialExtinctionOrder = Literal["none", "first", "second"]


@dataclass
class DiagnosticPlots:
    """Optional QC figures under ``<output_dir>/diagnostics/``."""

    calibration_crossmatch_separation_histogram: bool = False
    photometry_mag_vs_error_scatter: bool = False
    photometry_radial_growth_curve: bool = False
    calibration_instrumental_vs_catalog: bool = False
    calibration_zeropoint_residual_histogram: bool = False
    calibration_zeropoint_residual_vs_color: bool = False
    calibration_color_check_cal_stars: bool = False
    correlation_inter_filter_separation_plot: bool = False
    combined_separation_histograms: bool = False


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
    cosmic_ray_removal: bool = False
    limiting_contrast_rm_cosmics: float = 5.0
    read_noise: float = 8.0
    sigma_clipping_value: float = 4.5
    saturation_level: float = 65535.0
    plots_for_all_images: bool = False
    file_type_plots: str = "pdf"
    use_wcs_projection_for_star_maps: bool = True
    annotate_image: bool = False
    annotate_reference_image: bool = False
    magnitude_limit_image_annotation: float | None = None
    filter_magnitude_limit_image_annotation: str | None = None
    sigma_value_background_clipping: float = 5.0
    multiplier_background_rms: float = 5.0
    size_epsf_region: int = 25
    size_extraction_region_epsf: int = 11
    epsf_fitter: str = "TRFLSQFitter"
    n_iterations_eps_extraction: int = 1
    fraction_epsf_stars: float = 0.2
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
            "sigma_value_background_clipping": self.sigma_value_background_clipping,
            "multiplier_background_rms": self.multiplier_background_rms,
            "size_epsf_region": self.size_epsf_region,
            "size_extraction_region_epsf": self.size_extraction_region_epsf,
            "epsf_fitter": self.epsf_fitter,
            "n_iterations_eps_extraction": self.n_iterations_eps_extraction,
            "fraction_epsf_stars": self.fraction_epsf_stars,
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
            "annotate_image": self.annotate_image,
            "magnitude_limit_image_annotation": self.magnitude_limit_image_annotation,
            "filter_magnitude_limit_image_annotation": (
                self.filter_magnitude_limit_image_annotation
            ),
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
        kw.pop("annotate_image", None)
        kw["annotate_reference_image"] = self.annotate_reference_image
        return kw


@dataclass
class CorrelationConfig:
    max_pixel_between_objects: int = 3
    ooi_correlation_strategy: int = 1
    cross_identification_limit: int = 1
    n_allowed_non_detections_object: int = 1
    expected_bad_image_fraction: float = 1.0
    protect_reference_obj: bool = True
    correlation_method: CorrelationMethod = "astropy"
    separation_limit: u.Quantity = 2.0 * u.arcsec
    duplicate_handling_object_identification: dict[str, str] | None = None
    verbose: bool = False
    skip_correlation_intra: bool = False
    skip_correlation_inter: bool = False


@dataclass
class CalibrationConfig:
    calibration_module: Literal["legacy", "differential"] = "legacy"
    differential_coefficient_mode: DifferentialCoefficientMode = "per_night"
    differential_fit_sigma_clip: float = 2.5
    differential_per_image_rolling_median_color_term: bool = False
    differential_per_image_rolling_median_zero_point: bool = False
    differential_per_image_rolling_mean_color_term: bool = False
    differential_per_image_rolling_mean_zero_point: bool = False
    differential_per_image_rolling_window: int = 3
    differential_extinction_order: DifferentialExtinctionOrder = "first"
    differential_fit_extinction_from_data: bool = False
    calibration_catalog_radius_arcmin: float = 15.0
    calibration_catalog_mag_range: tuple[float, float] = (2.0, 18.5)
    differential_color_indices: dict[str, tuple[str, str]] | None = None
    differential_exposure_pairing: Literal["jd_nearest", "index"] = "jd_nearest"
    differential_exposure_jd_tolerance: float = 0.02
    differential_debug_exposure_pairing: bool = False
    debug_verify_ooi_global_ids: bool = False
    differential_reference_filter: Optional[str] = None
    differential_calibration_summary_use_jd_x: bool = False
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
    apply_transformation: bool = True
    transformation_coefficients_dict: dict[str, float | str] | None = None
    derive_transformation_coefficients: bool = False
    calculate_zero_point_statistic: bool = True
    distribution_samples: int = 1000
    aperture_radius: float = 4.0
    write_differential_legacy_magnitudes_dat: bool = False
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
    convert_magnitudes: bool = False
    target_filter_system: str = "SDSS"
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
    hips_reference_subtraction_filter: Optional[str] = None
    hips_reference_subtraction_image_index: int = 0
    hips_reference_subtraction_wcs_method: Optional[WcsMethod] = None
    hips_reference_subtraction_plot_comp: bool = True
    hips_reference_subtraction_hips_source: str = "CDS/P/DSS2/blue"
    hips_reference_subtraction_trim: Optional[Tuple[int, int, int, int]] = None
    hips_reference_subtraction_reuse_pipeline_wcs: bool = True
    hips_reference_subtraction_timeout_ms: int = 120_000
    hips_reference_subtraction_server: str = (
        "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    )
    hips_reference_subtraction_verbose: bool = False
    hips_reference_subtraction_hotpants_executable: Optional[str] = None
    hips_reference_subtraction_hotpants_extra_args: Tuple[str, ...] = field(
        default_factory=tuple
    )
    hips_reference_subtraction_output_filename: str = "hotpants_diff.fits"


@dataclass
class ExtinctionConfig:
    skip_extinction_fit: bool = True
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
        section = self._find_section_for(name)
        if section is not None:
            return getattr(section, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SECTION_NAMES or name == "diagnostic_plots":
            super().__setattr__(name, value)
            return
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
            if section is not None:
                setattr(section, key, val)

    def as_flat_dict(self) -> dict[str, Any]:
        """Serialize all sub-config fields to a flat dictionary."""
        out: dict[str, Any] = {}
        for section in self._sections():
            for f in fields(section):
                out[f.name] = getattr(section, f.name)
        return out
