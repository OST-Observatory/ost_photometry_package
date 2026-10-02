"""Reduction workflow: config module."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReduceConfig:
    """Configuration for the data reduction pipeline."""

    image_path: Path
    output_dir: Path
    image_type_dir: dict[str, list[str]]
    gain: float | None = None
    read_noise: float | None = None
    dark_rate: float | None = None
    # Gain / read noise from the camera catalog or measured on bias and flat
    # pairs, see ost_photometry.reduce.detector_noise.NOISE_SOURCES
    camera_noise_source: str = "catalog"
    # How the camera bins (read noise per binned pixel): auto, digital, charge
    binning_mode: str = "auto"
    # True, False or "auto" (skip when the filter is stacked from enough
    # frames, whose sigma clipping removes cosmic rays)
    rm_cosmic_rays: bool | str = "auto"
    cosmic_ray_auto_min_frames: int = 7
    mask_cosmic_rays: bool = False
    saturation_level: float | None = None
    limiting_contrast_rm_cosmic_rays: float = 5.0
    sigma_clipping_value_rm_cosmic_rays: float = 4.0
    scale_image_with_exposure_time: bool = True
    # Explicit reference frame index (after sorting by time). ``None`` lets
    # ``reference_image_selection`` decide.
    reference_image_index: int | None = None
    # See ost_photometry.reduce.frame_selection.REFERENCE_SELECTION
    reference_image_selection: str = "best_fwhm"
    enforce_bias: bool = False
    add_hot_bad_pixel_mask: bool = True
    # See ost_photometry.reduce.registration.SHIFT_METHODS
    shift_method: str = "aa_true"
    n_cores_multiprocessing: int | None = None
    stack_images: bool = True
    # Deprecated alias for ``measure_frame_quality`` (kept for old scripts).
    estimate_fwhm: bool = False
    # Frame quality: per-frame FWHM / roundness / star count / background,
    # written to ``frame_quality.ecsv`` and the FITS headers.
    measure_frame_quality: bool = True
    # Mapping or ``FrameSelection``; ``None`` measures only, rejects nothing.
    frame_selection: object | None = None
    # See ost_photometry.reduce.frame_selection.STACK_WEIGHTING
    stack_weighting: str = "none"
    # Keep ``aligned_lights/`` after stacking (re-stacking, export).
    keep_aligned_lights: bool = False
    shift_all: bool = False
    exposure_time_tolerance: float = 0.5
    stack_method: str = "average"
    target_name: str | None = None
    find_wcs: bool = True
    wcs_method: str = "astap"
    find_wcs_of_all_images: bool = False
    force_wcs_determination: bool = False
    rm_outliers_image_shifts: bool = True
    filter_window_image_shifts: int = 25
    threshold_image_shifts: float = 10.0
    temperature_tolerance: float = 5.0
    plot_dark_statistic_plots: bool = False
    plot_flat_statistic_plots: bool = False
    ignore_readout_mode_mismatch: bool = False
    ignore_instrument_mismatch: bool = False
    trim_x_start: int = 0
    trim_x_end: int = 0
    trim_y_start: int = 0
    trim_y_end: int = 0
    dtype: str | np.dtype | None = None
    debug: bool = False
    save_only_transformation: bool = False
    validate_inputs: bool = True
    sanity_check_sample_size: int = 3
    fail_on_missing_flat: bool = True
    # False: never prompt about reusing masters / reduced frames.
    interactive: bool = True
    # Floating type of written images (ost_photometry.reduce.storage)
    storage_dtype: str = "float32"


