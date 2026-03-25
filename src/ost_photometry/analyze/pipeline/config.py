"""Pipeline configuration."""

from dataclasses import dataclass
from typing import Literal, Optional

from astropy.coordinates import EarthLocation
import astropy.units as u


@dataclass
class PipelineConfig:
    """Central configuration for the analysis pipeline."""

    # Extraction
    extraction_mode: Literal["single", "multi", "auto"] = "auto"
    n_cores_multiprocessing: int = 6
    reference_image_id: int = 0
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
    wcs_method: str = "astrometry"
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
    photometry_extraction_method: str = "PSF"
    radius_aperture: float = 5.0
    inner_annulus_radius: float = 7.0
    outer_annulus_radius: float = 10.0
    radii_unit: str = "arcsec"
    strict_epsf_checks: bool = True

    # Transform (multi mode)
    transform_object_positions_to_reference: bool = False

    # Correlation
    max_pixel_between_objects: int = 3
    own_correlation_option: int = 1
    cross_identification_limit: int = 1
    n_allowed_non_detections_object: int = 1
    expected_bad_image_fraction: float = 1.0
    protect_reference_obj: bool = True
    correlation_method: str = "astropy"
    separation_limit: u.Quantity = 2.0 * u.arcsec
    duplicate_handling_object_identification: dict[str, str] | None = None
    verbose: bool = False

    # Calibration
    calibration_module: Literal["legacy", "differential"] = "legacy"
    differential_coefficient_mode: str = "per_night"  # per_image, per_night, fixed, ensemble
    differential_extinction_order: str = "first"  # none, first, second
    differential_fit_extinction_from_data: bool = False  # fit k from APASS comparison stars
    differential_apass_radius: float = 15.0  # arcmin
    differential_apass_mag_limit: float = 16.0
    # Color indices for transformation: {filter: (filter1, filter2)} e.g. {"V": ("B", "V")}
    differential_color_indices: dict[str, tuple[str, str]] | None = None
    # Multi-band calibration epochs (bridge): pair exposures across filters
    differential_exposure_pairing: Literal["jd_nearest", "index"] = "jd_nearest"
    differential_exposure_jd_tolerance: float = 0.02  # days (~29 min)
    differential_reference_filter: Optional[str] = None  # None -> filter_list[0]
    # EarthLocation for airmass
    observatory_location: object = EarthLocation(
        lat=52.409184 * u.deg,
        lon=12.973185 * u.deg,
        height=39 * u.m,
    )
    calibration_method: str = "APASS"
    magnitude_range: tuple[float, float] = (0.0, 18.5)
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

    # Extinction fit workflow (cat-star.org method)
    skip_extinction_fit: bool = True  # Set False to run extinction determination
    extinction_fit_mag_col: str = "mags_fit"
    extinction_fit_use_flux: bool = False
    extinction_coefficients_filename: str = "extinction_coefficients.json"

    # Skip flags
    skip_wcs: bool = False
    skip_extraction: bool = False
    skip_correlation_intra: bool = False
    skip_correlation_inter: bool = False
    skip_calibration: bool = False
    skip_post_process: bool = False
