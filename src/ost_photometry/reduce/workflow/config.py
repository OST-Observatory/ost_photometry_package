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
    rm_cosmic_rays: bool = True
    mask_cosmic_rays: bool = False
    saturation_level: float | None = None
    limiting_contrast_rm_cosmic_rays: float = 5.0
    sigma_clipping_value_rm_cosmic_rays: float = 4.0
    scale_image_with_exposure_time: bool = True
    reference_image_index: int = 0
    enforce_bias: bool = False
    add_hot_bad_pixel_mask: bool = True
    # aa / aa_true / own / skimage / flow / wcs (reproject onto reference WCS)
    shift_method: str = "aa_true"
    n_cores_multiprocessing: int | None = None
    stack_images: bool = True
    estimate_fwhm: bool = False
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


