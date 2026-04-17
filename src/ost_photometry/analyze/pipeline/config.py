"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from astropy.coordinates import EarthLocation
import astropy.units as u

WcsMethod = Literal["astrometry", "astap", "twirl"]
CorrelationMethod = Literal["astropy", "own"]
PhotometryExtractionMethod = Literal["PSF", "APER"]
DifferentialCoefficientMode = Literal["per_image", "per_night", "fixed", "ensemble"]
DifferentialExtinctionOrder = Literal["none", "first", "second"]


@dataclass
class DiagnosticPlots:
    """
    Toggle optional quality-control / debugging figures written under
    ``<output_dir>/diagnostics/`` (format from ``PipelineConfig.file_type_plots``).

    All default to ``False`` so normal runs stay quiet. Enable individual flags or
    pass overrides via ``run_pipeline(..., diagnostic_plots__<field>=True)``.
    """

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
class PipelineConfig:
    """Central configuration for the analysis pipeline."""

    # Extraction
    extraction_mode: Literal["single", "multi", "auto"] = "auto"
    n_cores_multiprocessing: int = 6
    reference_image_index: int = 0
    fwhm_object_psf: dict[str, float] | None = None

    # Extraction plot/output options
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

    # WCS
    wcs_method: WcsMethod = "astrometry"
    force_wcs_determination: bool = False

    # Extraction parameters (passed to main_extract/extract_multiprocessing)
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

    # Transform (multi mode)
    transform_object_positions_to_reference: bool = False

    # Correlation
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

    # Calibration
    calibration_module: Literal["legacy", "differential"] = "legacy"
    differential_coefficient_mode: DifferentialCoefficientMode = "per_night"
    # T/ZP fit: reject comparison stars with |residual| > sigma_clip * rms (iterative)
    differential_fit_sigma_clip: float = 2.5
    # PER_IMAGE: optional centered rolling median / mean over the epoch-ordered run (pandas rolling)
    differential_per_image_rolling_median_color_term: bool = False
    differential_per_image_rolling_median_zero_point: bool = False
    differential_per_image_rolling_mean_color_term: bool = False
    differential_per_image_rolling_mean_zero_point: bool = False
    differential_per_image_rolling_window: int = 3  # odd >= 1; even values are bumped to odd; median and mean
    differential_extinction_order: DifferentialExtinctionOrder = "first"
    differential_fit_extinction_from_data: bool = False  # fit k from catalog comparison stars
    calibration_catalog_radius_arcmin: float = 15.0
    calibration_catalog_mag_range: tuple[float, float] = (2.0, 18.5)
    # Color indices for transformation: {filter: (filter1, filter2)} e.g. {"V": ("B", "V")}
    differential_color_indices: dict[str, tuple[str, str]] | None = None
    # Multi-band calibration epochs (bridge): pair exposures across filters
    differential_exposure_pairing: Literal["jd_nearest", "index"] = "jd_nearest"
    differential_exposure_jd_tolerance: float = 0.02  # days (~29 min)
    # Log paired exposure filenames (see bridge._pairing_index / _pairing_jd_nearest)
    differential_debug_exposure_pairing: bool = False
    # After inter-filter correlation: re-match OOI sky coords vs id_in_image_series
    debug_verify_ooi_global_ids: bool = False
    differential_reference_filter: Optional[str] = None  # None -> filter_list[0]
    # Calibration summary plot (T/ZP vs epochs): use JD from calibration_epoch_meta when True
    differential_calibration_summary_use_jd_x: bool = False
    # EarthLocation for airmass
    observatory_location: EarthLocation = EarthLocation(
        lat=52.409184 * u.deg,
        lon=12.973185 * u.deg,
        height=39 * u.m,
    )
    # Legacy + differential: calibration_source key (APASS, simbad, vsp, simbad_vot, or vizier_dict key)
    calibration_source: str = "APASS"
    path_calibration_file: str | None = None
    vizier_dict: dict[str, str] | None = None
    region_to_select_calibration_stars: object = None  # regions.RectanglePixelRegion
    apply_transformation: bool = True
    transformation_coefficients_dict: dict[str, float | str] | None = None
    derive_transformation_coefficients: bool = False
    calculate_zero_point_statistic: bool = True
    distribution_samples: int = 1000
    aperture_radius: float = 4.0

    # Post-process
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
    # Differential calibration: also write legacy wide ``.dat`` (``save_magnitudes_ascii``)
    write_differential_legacy_magnitudes_dat: bool = False

    # Light curves (``LightCurveStep``; disabled by default)
    skip_light_curve: bool = True
    plot_light_curve_objects_of_interest: bool = True
    plot_light_curve_calibration_objects: bool = False
    plot_light_curve_all_objects: bool = False
    light_curve_binning_factor: float | None = None
    light_curve_quantity: Literal["magnitude", "flux"] = "magnitude"
    light_curve_calibration_rows: Literal["auto", "transformed", "simple"] = "auto"

    # HiPS archival template subtraction (HOTPANTS; disabled by default)
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

    # Extinction fit workflow (cat-star.org method)
    skip_extinction_fit: bool = True  # Set False to run extinction determination
    extinction_fit_mag_col: str = "mags_fit"
    extinction_fit_use_flux: bool = False
    extinction_coefficients_filename: str = "extinction_coefficients.json"

    # Optional diagnostic figures (see ``DiagnosticPlots``)
    diagnostic_plots: DiagnosticPlots = field(default_factory=DiagnosticPlots)

    # Skip flags
    skip_wcs: bool = False
    skip_extraction: bool = False
    skip_correlation_intra: bool = False
    skip_correlation_inter: bool = False
    skip_calibration: bool = False
