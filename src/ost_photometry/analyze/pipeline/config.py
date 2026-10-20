"""Pipeline configuration with step-specific sub-configs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_args, get_origin, get_type_hints

import astropy.units as u
from astropy.coordinates import EarthLocation

from ...fits_headers import CosmicRayRemovalMode, normalize_cosmic_ray_removal

WcsMethod = Literal["astrometry", "astap", "twirl"]
CorrelationMethod = Literal["astropy", "own"]
CorrelationLinkMode = Literal["to_reference", "sequential"]
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
UncertaintyMode = Literal["fit_errors", "monte_carlo"]

CALIBRATION_PRESETS: dict[str, dict[str, Any]] = {
    # Median zero-point per exposure (stacked fields without a color term).
    "median_zp_per_image": {
        "calibration_strategy": "median_zp",
        "calibration_grouping": "per_image",
        "extinction_mode": "none",
        "color_term_fit": "never",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": True,
        # Stacked B/V (N2) are sequential, not simultaneous — pair by file order.
        "exposure_pairing": "index",
    },
    # Linear T+ZP per epoch (no derive-transform). Pairing is not part of this
    # preset: N2 stacks set ``exposure_pairing="index"`` in the script; C7
    # visits keep the package default ``jd_nearest``.
    "linear_fit_per_image": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_image",
        "extinction_mode": "none",
        "color_term_fit": "auto",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": False,
    },
    # Same as linear_fit_per_image, plus extinction from comparison stars.
    "linear_fit_per_image_extinction": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_image",
        "extinction_mode": "from_comparison_stars",
        "color_term_fit": "auto",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": False,
    },
    # Nightly linear color term + ZP: per-epoch fits combined (GLS; catalog
    # part kept, epoch scatter as zero_point_sys).
    "linear_fit_per_night": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_night",
        "extinction_mode": "none",
        "color_term_fit": "auto",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": False,
    },
    # Same as linear_fit_per_night, plus extinction from comparison stars.
    "linear_fit_per_night_extinction": {
        "calibration_strategy": "linear_fit",
        "calibration_grouping": "per_night",
        "extinction_mode": "from_comparison_stars",
        "color_term_fit": "auto",
        "derive_transform_from_data": False,
        "zp_subsample_statistic": False,
    },
    # WCS + extraction + intra-filter correlation; protect calibrators; no apply.
    "extract_protect_calibrators": {
        "protect_calibration_objects": True,
        "skip_calibration": True,
        "skip_correlation_inter": True,
        "skip_light_curve": True,
        "extinction_mode": "none",
    },
    # One ZP over all epochs: per-star means of the extinction-corrected
    # epochs (each star once).
    "linear_fit_ensemble": {
        "calibration_strategy": "linear_fit",
        "derive_transform_from_data": False,
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


@dataclass
class DiagnosticPlots:
    """QC figures under ``<output_dir>/diagnostics/<step>/``.

    Most checks are on by default so a normal pipeline run leaves enough plots
    to judge photometry, cross-match, and calibration quality. Disable
    individually via ``diagnostic_plots__<name>=False`` overrides. Growth curves
    stay off (pick a star / more specialized). Layout:
    ``diagnostics/extraction|correlation|calibration|extinction|cluster|lightcurves``.
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
    #: Per-filter track QC after intra-filter correlation
    #: (``diagnostics/correlation/track_qc_<filter>``): matched tracks and
    #: pixel shift per frame, completeness, position and magnitude scatter per
    #: track, field map with residual vectors. Plus ``track_qc_<filter>.ecsv``.
    correlation_track_qc: bool = True
    #: Max individual inter-filter pair PDFs / geometry figures
    #: (``None`` = all; ``0`` = overview only). Geometry quiver plots share this cap.
    correlation_inter_filter_max_pair_plots: int | None = 25
    #: Max per-epoch calibration QC PDFs (cross-match, mag–error, catalog checks).
    #: ``None`` = all epochs; ``0`` = none (combined histogram still uses epoch 0).
    calibration_max_epoch_plots: int | None = 25
    #: ΔJD / image-id pairing table + plot (same groups as ``exposure_pairing``).
    exposure_pairing_overview: bool = True
    combined_separation_histograms: bool = True


@dataclass
class WcsConfig:
    wcs_method: WcsMethod = "astap"
    force_wcs_determination: bool = False
    skip_wcs: bool = False
    #: Solve a WCS for every frame instead of broadcasting the reference solution.
    wcs_solve_all_images: bool = False


@dataclass
class ExtractionConfig:
    extraction_mode: Literal["single", "multi", "auto"] = "auto"
    #: Worker processes for multi-image extraction. ``None`` → half the CPUs.
    n_cores_multiprocessing: int | None = None
    reference_image_index: int | Literal["auto"] = 0
    fwhm_object_psf: dict[str, float] | None = None
    #: Accepted FWHM range in pixels for automatic estimation (per-star filter).
    fwhm_estimate_min: float = 2.0
    fwhm_estimate_max: float = 15.0
    #: ``auto`` skips lacosmic when reduction already set CRIDENT / cosmics_*.
    cosmic_ray_removal: bool | CosmicRayRemovalMode = "auto"
    limiting_contrast_rm_cosmics: float = 5.0
    #: L.A.Cosmic read noise (e-) and saturation for the image in electrons;
    #: ``None`` uses ``RDNOISE`` / ``SATLEVEL`` from the reduction header
    #: (fallback 8 e- / 65535).
    read_noise: float | None = None
    sigma_clipping_value: float = 4.5
    saturation_level: float | None = None
    plots_for_all_images: bool = False
    file_type_plots: str = "pdf"
    use_wcs_projection_for_star_maps: bool = True
    #: Simbad overlay on the reference image (``SimbadAnnotateStep``, not extraction).
    annotate_image: bool = True
    #: Same overlay in multi-image extraction mode (``SimbadAnnotateStep``).
    annotate_reference_image: bool = True
    #: Brightest Simbad magnitude on the overlay (TAP cut + plot). ``None`` = no mag cut.
    magnitude_limit_image_annotation: float | None = None
    #: Band for that limit (``V``, ``B``, …). Defaults to ``V`` when the limit is set.
    filter_magnitude_limit_image_annotation: str | None = None
    #: Hierarchical SIMBAD types, e.g. ``["Star", "Galaxy", "Cl*"]``. ``None`` = all types.
    simbad_annotate_otypes: list[str] | None = None
    #: If True, only objects with a SIMBAD common name (``NAME …`` identifier).
    simbad_annotate_require_common_name: bool = False
    #: Extra TAP/ADQL fragment ANDed into the cone query.
    simbad_annotate_criteria: str | None = None
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
    finder_sharpness_range: tuple[float, float] = (0.2, 1.0)
    finder_roundness_range: tuple[float, float] = (-1.0, 1.0)
    #: ``min_separation = max(2, int(fwhm * factor))`` for IRAF/DAO finders.
    finder_min_separation_fwhm: float = 1.0
    #: Keep detections with finder FWHM in this factor times the image FWHM.
    #: ``None`` disables. IRAF provides a ``fwhm`` column; DAO typically does not.
    finder_fwhm_scale_range: tuple[float, float] | None = (0.5, 2.0)
    multiplier_background_rms_epsf: float = 5.0
    multiplier_grouper_epsf: float = 2.0
    #: PSF: stars within the fit half-width + this many FWHM are fitted
    #: together (neighbours' wings in the fit box are modelled). ``None``:
    #: only ``multiplier_grouper_epsf`` × FWHM, much faster in crowded
    #: fields with poor seeing, but neighbours bias faint stars.
    psf_group_wing_fwhm: float | None = 2.5
    #: PSF: second pass with the local sky from the residual image (all
    #: fitted stars subtracted). ``False`` halves the time.
    psf_residual_sky: bool = True
    minimum_n_eps_stars: int = 15
    photometry_extraction_method: PhotometryExtractionMethod = "PSF"
    #: If False, fit finder positions only (no residual re-find).
    psf_find_in_residuals: bool = False
    #: APER radii (aperture, annulus inner / outer) in ``radii_unit``;
    #: ``None`` takes the unit's default (``fwhm``: 1.0 / 2.5 / 4.0,
    #: ``arcsec``: 5 / 7 / 10; ``pixel`` needs all three).
    radius_aperture: float | None = None
    inner_annulus_radius: float | None = None
    outer_annulus_radius: float | None = None
    #: ``fwhm`` (multiples of each image's FWHM: the same fraction of the
    #: star in every frame and stack), ``arcsec`` or ``pixel``.
    radii_unit: Literal["fwhm", "arcsec", "pixel"] = "fwhm"
    #: APER: model the detected stars (Moffat) and measure sky and aperture
    #: with the neighbours removed (crowded fields, poor seeing).
    aperture_neighbour_subtraction: bool = True
    strict_epsf_checks: bool = True
    transform_object_positions_to_reference: bool = False
    #: Add atmospheric scintillation (Osborn et al. 2015) to ``flux_err`` /
    #: ``mags_unc`` of every source; the photon / sky error stays in
    #: ``flux_err_phot``. Needs the aperture (telescope catalog or
    #: ``telescope_aperture_m``); site altitude from ``observatory_location``.
    #: A star is ``saturated`` when a pixel within the measurement radius
    #: reaches this fraction of ``SATLEVEL`` (non-linearity starts below
    #: full well); saturated stars are no calibrators or ensemble members.
    saturation_fraction: float = 0.9
    scintillation_noise: bool = True
    #: Correction factor to Young's coefficient (Osborn et al.: median ≈ 1.5).
    scintillation_coefficient: float = 1.5
    #: Aperture diameter in m; ``None`` takes it from ``TELESCOP`` /
    #: ``FOCALLEN`` and ``data/telescopes.json``.
    telescope_aperture_m: float | None = None
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
            "finder_sharpness_range": self.finder_sharpness_range,
            "finder_roundness_range": self.finder_roundness_range,
            "finder_min_separation_fwhm": self.finder_min_separation_fwhm,
            "finder_fwhm_scale_range": self.finder_fwhm_scale_range,
            "multiplier_background_rms_epsf": self.multiplier_background_rms_epsf,
            "multiplier_grouper_epsf": self.multiplier_grouper_epsf,
            "psf_group_wing_fwhm": self.psf_group_wing_fwhm,
            "psf_residual_sky": self.psf_residual_sky,
            "minimum_n_eps_stars": self.minimum_n_eps_stars,
            "saturation_fraction": self.saturation_fraction,
            "strict_epsf_checks": self.strict_epsf_checks,
            "photometry_extraction_method": self.photometry_extraction_method,
            "psf_find_in_residuals": self.psf_find_in_residuals,
            "radius_aperture": self.radius_aperture,
            "inner_annulus_radius": self.inner_annulus_radius,
            "outer_annulus_radius": self.outer_annulus_radius,
            "radii_unit": self.radii_unit,
            "aperture_neighbour_subtraction": self.aperture_neighbour_subtraction,
            "cosmic_ray_removal": normalize_cosmic_ray_removal(self.cosmic_ray_removal),
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
    #: Dense mode (``require_complete_intersection=True``): objects missing on
    #: this many frames are removed before bad frames are rejected. Ignored
    #: when ``min_detection_fraction`` is set.
    n_allowed_non_detections_object: int = 1
    expected_bad_image_fraction: float = 1.0
    #: If True, drop tracks that miss any remaining frame (mk_calib / N2).
    require_complete_intersection: bool = True
    #: Sparse tracks: keep objects detected on at least this fraction of frames.
    #: This is the only completeness criterion when set.
    min_detection_fraction: float | None = None
    correlation_link_mode: CorrelationLinkMode = "to_reference"
    #: ``auto`` uses pixel matching when frames look registered (``aa_true``).
    correlation_coordinates: Literal["auto", "sky", "pixel"] = "auto"
    protect_ooi: bool = True
    protect_calibration_objects: bool = False
    protected_object_ids: list[int] | None = None
    correlation_method: CorrelationMethod = "astropy"
    separation_limit: u.Quantity = 2.0 * u.arcsec
    #: OOI sky match. ``None`` → at least 5 arcsec (WCS residuals).
    ooi_separation_limit: u.Quantity | None = None
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
    #: ``jd_nearest`` (C7 visits) or ``index`` (N2 stacks). One image per filter
    #: is always paired by index, so stacked B/V may be many minutes apart.
    exposure_pairing: Literal["jd_nearest", "index"] = "jd_nearest"
    #: Max |ΔJD| (days) when ``exposure_pairing="jd_nearest"``. Ignored for
    #: stacked cluster fields with one image per filter (paired by index).
    exposure_jd_tolerance: float = 0.02
    debug_exposure_pairing: bool = False
    reference_filter: str | None = None
    calibration_summary_use_jd_x: bool = False
    zp_subsample_statistic: bool = True
    derive_transform_from_data: bool = False
    uncertainty_mode: UncertaintyMode = "fit_errors"
    calibration_catalog_radius_arcmin: float = 15.0
    calibration_catalog_mag_range: tuple[float, float] = (2.0, 18.5)
    #: On-sky radius for matching extracted stars to the calibration catalog.
    #: Independent of ``CorrelationConfig.separation_limit`` (intra/inter-filter).
    calibration_match_radius: u.Quantity = 2.0 * u.arcsec
    #: Drop catalog stars that coincide with VSX (known variables) before the fit.
    exclude_known_variables: bool = True
    #: Match radius catalog star ↔ VSX: VSX positions of older variables
    #: are often off by 1-2" (3", was 1").
    exclude_known_variables_radius: u.Quantity = 3.0 * u.arcsec
    #: Reject catalog stars whose ``err_<filter>`` sits above the binned p84
    #: ridge of all detections at that magnitude.
    calibrator_error_p84_clip: bool = True
    #: Reject if ``σ`` exceeds this factor times the photon+sky envelope.
    #: ``None`` disables the envelope cut.
    calibrator_photon_factor: float | None = 2.0
    calibrator_qfit_max: float | None = 0.2
    calibrator_cfit_max: float | None = 0.2
    calibrator_qfit_bright_percentile: float | None = 90.0
    calibrator_cfit_bright_percentile: float | None = 90.0
    calibrator_bright_frac: float = 0.4
    #: Re-apply ``finder_sharpness_range`` / ``finder_roundness_range`` to
    #: catalog stars before the fit. Missing/NaN shape columns are kept.
    calibrator_apply_finder_shape_cuts: bool = True
    #: Skip a quality cut if it would leave fewer than this many candidates.
    calibrator_min_keep: int = 3
    #: Calibrators whose aperture is masked by more than this fraction (bad
    #: pixels, frame edge) are excluded in every case.
    calibrator_max_masked_fraction: float = 0.05
    #: Calibrators must be isolated: no own detection within this radius
    #: (arcsec) less than ``calibrator_isolation_dmag`` fainter (catalog
    #: positions of APASS-like surveys merge such pairs), and no more than
    #: ``calibrator_max_neighbour_fraction`` neighbour light in the aperture
    #: (``blended``). ``None`` disables a test; skipped when too few remain.
    calibrator_isolation_arcsec: float | None = 10.0
    calibrator_isolation_dmag: float = 5.0
    calibrator_max_neighbour_fraction: float | None = 0.01
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
    #: Blank-sky aperture of the limiting magnitude in ``radii_unit``;
    #: ``None`` uses the APER measurement aperture.
    aperture_radius: float | None = None
    skip_calibration: bool = False


@dataclass
class PostProcessConfig:
    object_id: int | None = None
    extract_only_circular_region: bool = False
    region_radius: float = 600.0
    identify_cluster_gaia_data: bool = False
    clean_objs_using_pm: bool = False
    max_distance_cluster: float = 6.0
    #: Deprecated SpectralClustering preset; ignored by GMM membership.
    find_cluster_para_set: int | None = None
    #: Deprecated alias for ``cluster_component_id``.
    cluster_selection_id: int | None = None
    cluster_ruwe_max: float = 1.4
    cluster_plx_snr_min: float | None = None
    cluster_pmem_min: float = 0.5
    cluster_membership_method: Literal["gmm", "xd", "hdbscan"] = "gmm"
    cluster_component_id: int | None = None
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
    #: Check-star QC (OOI vs most variable catalog calibrators; OOI excluded
    #: from the calibrator pool) and ensemble variability.
    plot_light_curve_calibration_objects: bool = False
    #: Overview raster (OOI + N random), not one PDF per star.
    plot_light_curve_all_objects: bool = False
    #: Phase-bin width (0–1) or number of phase bins (≥ 1) for folded curves.
    light_curve_binning_factor: float | None = None
    light_curve_quantity: Literal["magnitude", "flux"] = "magnitude"
    light_curve_calibration_rows: Literal["auto", "transformed", "simple"] = "auto"
    light_curve_overview_n: int = 8
    light_curve_calibrator_qc_n: int = 3
    #: MAD clip: short faint runs (1–2 epochs) and bright spikes; eclipse dips stay.
    light_curve_outlier_sigma: float | None = 5.0
    light_curve_color: str | None = None
    light_curve_time_scale: Literal["bjd_tdb", "jd"] = "bjd_tdb"
    light_curve_phase_cycles: int = 1
    light_curve_show_airmass: bool = True
    #: Differential product (OOI / quiet comparison ensemble, instrumental
    #: magnitudes only) and global ``flag_epoch`` on ``light_curves.ecsv``.
    light_curve_differential: bool = True
    #: Number of quiet comparison stars per filter.
    light_curve_ensemble_size: int = 10
    #: Half width (mag) of the brightness window around the OOI.
    light_curve_ensemble_mag_window: float = 1.0
    #: Minimum fraction of epochs an ensemble candidate is measured in.
    light_curve_ensemble_min_coverage: float = 0.8
    #: Epoch flags: robust sigma for transparency / ensemble scatter
    #: (``None``: only the ensemble-completeness test).
    light_curve_epoch_flag_sigma: float | None = 5.0
    #: Smallest common-mode deviation (mag) that flags an epoch.
    light_curve_epoch_flag_min_mag: float = 0.05
    #: Fraction of ensemble stars an epoch needs.
    light_curve_epoch_min_ensemble_fraction: float = 0.5


@dataclass
class HipsConfig:
    skip_hips_reference_subtraction: bool = True
    hips_reference_subtraction_filter: str | None = None
    hips_reference_subtraction_image_index: int = 0
    hips_reference_subtraction_wcs_method: WcsMethod | None = None
    hips_reference_subtraction_plot_comp: bool = True
    # None / "auto" → bandpass-matched PanSTARRS DR1 (linear CCD).
    hips_reference_subtraction_hips_source: str | None = None
    hips_reference_subtraction_trim: tuple[int, int, int, int] | None = None
    hips_reference_subtraction_reuse_pipeline_wcs: bool = True
    hips_reference_subtraction_timeout_ms: int = 120_000
    hips_reference_subtraction_server: str = (
        "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    )
    hips_reference_subtraction_fallback_servers: tuple[str, ...] = (
        "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
    )
    hips_reference_subtraction_retries: int = 3
    hips_reference_subtraction_retry_backoff_s: float = 1.5
    hips_reference_subtraction_use_cache: bool = True
    hips_reference_subtraction_verbose: bool = False
    hips_reference_subtraction_backend: Literal["auto", "hotpants", "alard_lupton"] = (
        "auto"
    )
    hips_reference_subtraction_hotpants_executable: str | None = None
    hips_reference_subtraction_hotpants_extra_args: tuple[str, ...] = field(
        default_factory=tuple
    )
    hips_reference_subtraction_output_filename: str = "diff.fits"


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


def _literal_values(hint) -> tuple | None:
    """Allowed values of a ``Literal`` (or ``Literal | None``) annotation."""
    if hint is None:
        return None
    if get_origin(hint) is Literal:
        return get_args(hint)
    args = get_args(hint)
    if args and type(None) in args:
        literals = [a for a in args if get_origin(a) is Literal]
        others = [a for a in args if a is not type(None) and get_origin(a) is not Literal]
        if len(literals) == 1 and not others:
            return (*get_args(literals[0]), None)
    return None


#: Options that no longer exist -> what to use instead (error, not warning).
_REMOVED_OPTIONS: dict[str, str] = {
    "aperture_scale_with_fwhm": (
        "APER radii scale with the FWHM by default now: radii_unit='fwhm' with "
        "radius_aperture / inner_annulus_radius / outer_annulus_radius as FWHM "
        "multiples (default 1.0 / 2.5 / 4.0)"
    ),
    "aperture_fwhm_factor": "use radius_aperture with radii_unit='fwhm'",
    "inner_annulus_fwhm_factor": "use inner_annulus_radius with radii_unit='fwhm'",
    "outer_annulus_fwhm_factor": "use outer_annulus_radius with radii_unit='fwhm'",
}


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

        Names describe the mode, e.g. ``linear_fit_per_image``,
        ``linear_fit_per_image_extinction``, ``median_zp_per_image``,
        ``linear_fit_per_night``, ``extract_protect_calibrators``,
        ``linear_fit_ensemble``, ``tabulated_extinction``.
        """
        if preset not in CALIBRATION_PRESETS:
            known = ", ".join(sorted(CALIBRATION_PRESETS))
            raise ValueError(f"Unknown calibration preset {preset!r}; known: {known}")
        kw = dict(CALIBRATION_PRESETS[preset])
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
        """Apply flat keyword overrides (used by ``run_pipeline``).

        Unknown keys warn (a typo would otherwise silently keep the default);
        values of ``Literal`` options are checked (:meth:`validate`).
        """
        removed = [k for k in kwargs if k in _REMOVED_OPTIONS]
        if removed:
            raise ValueError(
                "PipelineConfig: removed option(s): "
                + "; ".join(f"{k} ({_REMOVED_OPTIONS[k]})" for k in removed)
            )
        diag_prefix = "diagnostic_plots__"
        unknown: list[str] = []
        for key, val in kwargs.items():
            if key.startswith(diag_prefix):
                sub = key[len(diag_prefix) :]
                if hasattr(self.diagnostic_plots, sub):
                    setattr(self.diagnostic_plots, sub, val)
                else:
                    unknown.append(key)
                continue
            section = self._find_section_for(key)
            if section is None and key in _CONFIG_ALIASES:
                key = _CONFIG_ALIASES[key]
                section = self._find_section_for(key)
            if section is not None:
                setattr(section, key, val)
            else:
                unknown.append(key)
        if unknown:
            warnings.warn(
                f"PipelineConfig: unknown option(s) {sorted(unknown)} ignored.",
                stacklevel=3,
            )
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for values outside an option's ``Literal`` set."""
        problems: list[str] = []
        for section in self._sections():
            hints = get_type_hints(type(section), globalns=globals())
            for f in fields(section):
                allowed = _literal_values(hints.get(f.name))
                if allowed is None:
                    continue
                value = getattr(section, f.name)
                if value not in allowed:
                    problems.append(f"{f.name}={value!r} (allowed: {', '.join(map(repr, allowed))})")
        if self.derive_transform_from_data and self.calibration_grouping != "per_image":
            problems.append(
                "derive_transform_from_data=True fits each epoch on its own; use it with "
                f"calibration_grouping='per_image', not {self.calibration_grouping!r}"
            )
        if self.calibration_grouping == "fixed" and not self.transformation_coefficients_dict:
            problems.append(
                "calibration_grouping='fixed' needs transformation_coefficients_dict "
                "(colour terms per filter)"
            )
        if problems:
            raise ValueError("PipelineConfig: invalid option value(s): " + "; ".join(problems))
        if self.calibration_grouping in ("per_night", "ensemble") and self.extinction_mode == "none":
            warnings.warn(
                f"PipelineConfig: calibration_grouping={self.calibration_grouping!r} shares one "
                "zero point between epochs, but extinction_mode='none': magnitudes drift with the "
                "airmass by k·(X - X_mean) (k ≈ 0.2 mag/airmass in V, 0.33 in B). The drift is "
                "carried as a systematic error; use extinction_mode='from_comparison_stars' or "
                "'tabulated' when the night allows it.",
                stacklevel=3,
            )

    def as_flat_dict(self) -> dict[str, Any]:
        """Serialize all sub-config fields to a flat dictionary."""
        out: dict[str, Any] = {}
        for section in self._sections():
            for f in fields(section):
                out[f.name] = getattr(section, f.name)
        return out
