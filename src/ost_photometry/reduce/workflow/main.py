"""Reduction workflow: main module."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ... import calibration_parameters, checks, style, terminal_output
from ... import utilities as base_utilities
from .. import quality, registration, utilities, validation
from ..detector_noise import (
    NoiseMeasurement,
    binned_read_noise,
    check_binning_mode,
    check_noise_source,
    measure_detector_noise,
    resolve_binning_mode,
)
from ..frame_selection import (
    SUPPORTED_REFERENCE_SELECTION,
    SUPPORTED_STACK_WEIGHTING,
    FrameSelection,
    merge_alignment_result,
    write_quality_table,
)
from ..instrument import get_egain_from_collection, resolve_system_gain
from ..storage import check_storage_dtype
from .bias import master_bias
from .config import ReduceConfig
from .dark import master_dark, reduce_dark
from .flat import master_flat, reduce_flat
from .science import check_cosmic_ray_mode, reduce_light
from .stack import stack_image


def reduce_main(
    image_path: str,
    output_dir: str,
    image_type_dir: dict[str, list[str]] | None = None,
    gain: float | None = None,
    read_noise: float | None = None,
    dark_rate: float | None = None,
    camera_noise_source: str = "catalog",
    binning_mode: str = "auto",
    rm_cosmic_rays: bool | str = "auto",
    cosmic_ray_auto_min_frames: int = 7,
    mask_cosmic_rays: bool = False,
    saturation_level: float | None = None,
    limiting_contrast_rm_cosmic_rays: float = 5.0,
    sigma_clipping_value_rm_cosmic_rays: float = 4.0,
    scale_image_with_exposure_time: bool = True,
    reference_image_index: int | None = None,
    enforce_bias: bool = False,
    add_hot_bad_pixel_mask: bool = True,
    shift_method: str = "aa_true",
    n_cores_multiprocessing: int | None = None,
    stack_images: bool = True,
    estimate_fwhm: bool = False,
    measure_frame_quality: bool = True,
    frame_selection: FrameSelection | Mapping[str, object] | None = None,
    reference_image_selection: str = "best_fwhm",
    stack_weighting: str = "none",
    keep_aligned_lights: bool = False,
    shift_all: bool = False,
    exposure_time_tolerance: float = 0.5,
    stack_method: str = "average",
    target_name: str | None = None,
    find_wcs: bool = True,
    wcs_method: str = "astap",
    find_wcs_of_all_images: bool = False,
    force_wcs_determination: bool = False,
    rm_outliers_image_shifts: bool = True,
    filter_window_image_shifts: int = 25,
    threshold_image_shifts: float = 10.0,
    temperature_tolerance: float = 5.0,
    plot_dark_statistic_plots: bool = False,
    plot_flat_statistic_plots: bool = False,
    ignore_readout_mode_mismatch: bool = False,
    ignore_instrument_mismatch: bool = False,
    trim_x_start: int = 0,
    trim_x_end: int = 0,
    trim_y_start: int = 0,
    trim_y_end: int = 0,
    dtype: str | np.dtype | None = None,
    debug: bool = False,
    save_only_transformation: bool = False,
    validate_inputs: bool = True,
    sanity_check_sample_size: int = 3,
    fail_on_missing_flat: bool = True,
    interactive: bool = True,
    storage_dtype: str = "float32",
) -> None:
    """
    Main reduction routine: Creates master images for bias, darks,
                            flats, reduces the science images and trims
                            them to the same filed of view.

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be stored

    image_type_dir
        Image types of the images. Possibilities: bias, dark, flat,
        light
        Default is ``None``.

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) per pixel of the images (i.e. per binned pixel).
        ``None`` takes it from ``camera_noise_source``.
        Default is ``None``.

    dark_rate
        Dark rate in e-/pix/s:
        Default is ``None``.

    camera_noise_source
        Where gain and read noise come from when not given explicitly:
        ``catalog`` (header / camera catalog; the catalog read noise per
        native pixel is scaled to the binned pixel, see ``binning_mode``) or
        ``measured`` (read noise from bias pairs, or dark pairs without
        bias, gain from flat pairs of equal filter and exposure time; falls
        back to the catalog when the frames do not allow a measurement).
        See :mod:`ost_photometry.reduce.detector_noise`.
        Default is ``catalog``.

    binning_mode
        How the camera bins, for the catalog read noise: ``digital``
        (CMOS, pixels summed after the readout: read noise x
        sqrt(xbin * ybin)), ``charge`` (CCD on-chip binning: unchanged) or
        ``auto`` (camera catalog / camera name).
        Default is ``auto``.

    rm_cosmic_rays
        ``True`` removes cosmic rays with L.A.Cosmic, ``False`` never.
        ``auto`` removes them only where the stack cannot: if
        ``stack_images`` is False or a filter has fewer than
        ``cosmic_ray_auto_min_frames`` light frames. Otherwise the sigma
        clipping of the stack rejects cosmic rays without touching the
        sky noise.
        Default is ``auto``.

    cosmic_ray_auto_min_frames
        Minimum number of light frames per filter from which
        ``rm_cosmic_rays="auto"`` leaves the cosmic rays to the stack.
        Frames rejected later by ``frame_selection`` are counted.
        Default is ``7``.

    mask_cosmic_rays
        If True cosmics will ''only'' be masked. If False the
        cosmics will be removed from the input image and the mask will
        be added.
        Default is ``False``.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``None``.

    limiting_contrast_rm_cosmic_rays
        Parameter for the cosmic ray removal: Minimum contrast between
        Laplacian image and the fine structure image.
        Default is ``5``.

    sigma_clipping_value_rm_cosmic_rays
        Parameter for the cosmic ray removal: Fractional detection limit
        for neighboring pixels.
        Default is ``4.5``.

    scale_image_with_exposure_time
        If True the image will be scaled with the exposure time.
        Default is ``True``.

    reference_image_index
        ID of the image (after sorting by observation time) that should be
        used as the alignment reference. An explicit index always wins;
        ``None`` lets ``reference_image_selection`` decide.
        Default is ``None``.

    enforce_bias
        If True the usage of bias frames during the reduction is
        enforced if possible.
        Default is ``False``.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    shift_method
        Alignment backend. See
        :data:`~ost_photometry.reduce.registration.SHIFT_METHODS`.
        Default is ``aa_true``.

    n_cores_multiprocessing
        Worker processes for alignment, stacking, and calibration-frame
        combination. ``None`` or ``<= 0`` uses half the logical CPUs
        (``cpu_count() // 2``). Default is ``None``.

    stack_images
        If True the individual images of each filter will be stacked and
        those images will be aligned to each other.
        Default is ``True``.

    estimate_fwhm
        Deprecated alias for ``measure_frame_quality=True``.
        Default is ``False``.

    measure_frame_quality
        If True, FWHM, roundness, star count, sky background and masked
        fraction are measured for every reduced science frame before the
        alignment. Results go to ``<output>/frame_quality.ecsv`` and the
        frame headers (``FWHM``, ``NSTARS``, ...). Required for
        ``frame_selection``, ``stack_weighting`` and the ``best_fwhm``
        reference selection.
        Default is ``True``.

    frame_selection
        Frame rejection criteria as a mapping or
        :class:`~ost_photometry.reduce.frame_selection.FrameSelection`, e.g.
        ``{"fwhm_max": 4.0}`` (pixels), ``{"best_fraction": 0.8}`` (keep the
        sharpest 80 % per filter), ``{"fwhm_sigma_clip": 3.0}``,
        ``{"roundness_max": 0.3, "n_stars_min": 20, "min_frames": 5}``.
        Rejected frames are moved to ``<output>/rejected_lights/`` with the
        reason in their header (``QCREJ`` / ``QCREASON``). ``None`` measures
        only and rejects nothing.
        Default is ``None``.

    reference_image_selection
        How the alignment reference is chosen when ``reference_image_index``
        is ``None``: ``best_fwhm`` (sharpest kept frame per filter, or one
        global frame with ``shift_all``) or ``first``. See
        :data:`~ost_photometry.reduce.frame_selection.REFERENCE_SELECTION`.
        Default is ``best_fwhm``.

    stack_weighting
        Per-frame weights for ``stack_method="average"``: ``none``, ``fwhm``
        (``(median FWHM / FWHM)^2``), ``n_stars`` or ``noise``
        (``(median RMS / RMS)^2``), each normalised per filter. See
        :data:`~ost_photometry.reduce.frame_selection.STACK_WEIGHTING`.
        Default is ``none``.

    keep_aligned_lights
        Keep ``<output>/aligned_lights/`` after stacking, e.g. to re-stack
        with a different selection or to export the registered frames.
        Default is ``False``.

    shift_all
        If False shifts between images are only calculated for images of
        the same filter. If True shifts between all images are
        estimated.
        Default is ``False``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    stack_method
        Method used for combining the images.
        Possibilities: ``median`` or ``average`` or ``sum``
        Default is ``average`.

    target_name
        Name of the target. Used for file selection.
        Default is ``None``.

    find_wcs
        If `True` the WCS will be determined for the images.
        Default is ``True``.

    wcs_method
        Method to use for WCS determination.
        Possibilities are 'astrometry', 'astap', and 'twirl'
        Default is ``astap``.

    find_wcs_of_all_images
        If `True` the WCS will be calculated for each image
        individually.
        Default is ``False``.

    force_wcs_determination
        If ``True`` a new WCS determination will be calculated even if
        a WCS is already present in the FITS Header.
        Default is ``False``.

    rm_outliers_image_shifts
        If True outliers in the image shifts will be detected and removed.
        Default is ``True``.

    filter_window_image_shifts
        Width of the median filter window
        Default is ``25``.

    threshold_image_shifts
        Difference above the running median above an element is
        considered to be an outlier.
        Default is ``10.``.

    temperature_tolerance
        The images are required to have the temperature. This value
        specifies the temperature difference that is acceptable.
        Default is ``5.``.

    plot_dark_statistic_plots
        If True some plots showing some statistic on the dark frames are
        created.
        Default is ``False``

    plot_flat_statistic_plots
        If True some plots showing some statistic on the flat frames are
        created.
        Default is ``False``

    ignore_readout_mode_mismatch
        If set to `True` a mismatch of the detected readout modes will
        be ignored.
        Default is ``False``.

    ignore_instrument_mismatch
        If set to `True` a mismatch of the detected instruments will
        be ignored.
        Default is ``False``.

    trim_x_start
        Number of pixels to trim from the start of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_x_end
        Number of pixels to trim from the end of the X direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_start
        Number of pixels to trim from the start of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    trim_y_end
        Number of pixels to trim from the end of the Y direction,
        e.g. to remove an overscan region.
        Default is ``0``.

    dtype
        The dtype that should be used while combining the images.
        Default is ''None''. -> None is equivalent to float64

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    save_only_transformation
        If ``True'', only the transformation matrix is saved, not the transformed image itself.
        Default is ``False``.

    interactive
        If ``False``, never ask whether existing masters / reduced frames
        should be reused (they are recomputed). For unattended pipelines.
        Default is ``True``.

    storage_dtype
        Floating type of the written masters, reduced and aligned frames
        and stacks: ``float32`` (half the disk space; its rounding is far
        below the pixel noise) or ``float64``. Calculations run in float64
        regardless. See :mod:`ost_photometry.reduce.storage`.
        Default is ``float32``.
    """
    if estimate_fwhm:
        warnings.warn(
            "reduce_main(estimate_fwhm=True) is deprecated; use "
            "measure_frame_quality=True (now the default).",
            DeprecationWarning,
            stacklevel=2,
        )
        measure_frame_quality = True

    cfg = ReduceConfig(
        image_path=Path(image_path),
        output_dir=Path(output_dir),
        image_type_dir=image_type_dir or calibration_parameters.get_image_types(),
        gain=gain,
        read_noise=read_noise,
        dark_rate=dark_rate,
        camera_noise_source=camera_noise_source,
        binning_mode=binning_mode,
        rm_cosmic_rays=rm_cosmic_rays,
        cosmic_ray_auto_min_frames=cosmic_ray_auto_min_frames,
        mask_cosmic_rays=mask_cosmic_rays,
        saturation_level=saturation_level,
        limiting_contrast_rm_cosmic_rays=limiting_contrast_rm_cosmic_rays,
        sigma_clipping_value_rm_cosmic_rays=sigma_clipping_value_rm_cosmic_rays,
        scale_image_with_exposure_time=scale_image_with_exposure_time,
        reference_image_index=reference_image_index,
        enforce_bias=enforce_bias,
        add_hot_bad_pixel_mask=add_hot_bad_pixel_mask,
        shift_method=shift_method,
        n_cores_multiprocessing=n_cores_multiprocessing,
        stack_images=stack_images,
        estimate_fwhm=estimate_fwhm,
        measure_frame_quality=measure_frame_quality,
        frame_selection=frame_selection,
        reference_image_selection=reference_image_selection,
        stack_weighting=stack_weighting,
        keep_aligned_lights=keep_aligned_lights,
        shift_all=shift_all,
        exposure_time_tolerance=exposure_time_tolerance,
        stack_method=stack_method,
        target_name=target_name,
        find_wcs=find_wcs,
        wcs_method=wcs_method,
        find_wcs_of_all_images=find_wcs_of_all_images,
        force_wcs_determination=force_wcs_determination,
        rm_outliers_image_shifts=rm_outliers_image_shifts,
        filter_window_image_shifts=filter_window_image_shifts,
        threshold_image_shifts=threshold_image_shifts,
        temperature_tolerance=temperature_tolerance,
        plot_dark_statistic_plots=plot_dark_statistic_plots,
        plot_flat_statistic_plots=plot_flat_statistic_plots,
        ignore_readout_mode_mismatch=ignore_readout_mode_mismatch,
        ignore_instrument_mismatch=ignore_instrument_mismatch,
        trim_x_start=trim_x_start,
        trim_x_end=trim_x_end,
        trim_y_start=trim_y_start,
        trim_y_end=trim_y_end,
        dtype=dtype,
        debug=debug,
        save_only_transformation=save_only_transformation,
        validate_inputs=validate_inputs,
        sanity_check_sample_size=sanity_check_sample_size,
        fail_on_missing_flat=fail_on_missing_flat,
        interactive=interactive,
        storage_dtype=storage_dtype,
    )
    return _run_reduction(cfg)


@dataclass(frozen=True)
class CameraParameters:
    """Camera electronics resolved for one homogeneous set of frames.

    ``read_noise`` is per binned pixel (the pixels of the frames).
    ``noise_source`` says where gain and read noise come from: ``user``,
    ``catalog`` or ``measured``.
    """

    instrument: str
    readout_mode: str
    gain_setting: int | float | None
    pixel_bit_value: int
    temperature: float
    gain: float | None
    read_noise: float | None
    dark_rate: float | dict | None
    saturation_level: float | None
    binning: tuple[int, int] = (1, 1)
    binning_mode: str | None = None
    noise_source: str = "catalog"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _collection_binning(image_file_collection) -> tuple[int, int]:
    """Most common ``XBINNING`` / ``YBINNING`` of the collection (default 1)."""
    summary = image_file_collection.summary
    values = []
    for key in ("xbinning", "ybinning"):
        if summary is None or key not in summary.colnames:
            values.append(1)
            continue
        column = [v for v in summary[key] if v is not None and not np.ma.is_masked(v)]
        numbers = []
        for v in column:
            try:
                numbers.append(max(int(float(v)), 1))
            except (TypeError, ValueError):
                continue
        values.append(max(set(numbers), key=numbers.count) if numbers else 1)
    return values[0], values[1]


def _collection_files(image_file_collection, image_type_dir, image_class: str) -> list[str]:
    image_type = utilities.get_image_type(image_file_collection, image_type_dir,
                                          image_class=image_class)
    if not image_type:
        return []
    return list(image_file_collection.files_filtered(imagetyp=image_type, include_path=True))


def measure_collection_noise(
    image_file_collection,
    image_type_dir: Mapping[str, list[str]],
    saturation_level: float | None = None,
) -> NoiseMeasurement | None:
    """Read noise and gain from the bias, dark and flat frames of a collection."""
    return measure_detector_noise(
        _collection_files(image_file_collection, image_type_dir, "bias"),
        _collection_files(image_file_collection, image_type_dir, "flat"),
        _collection_files(image_file_collection, image_type_dir, "dark"),
        saturation_level=saturation_level,
    )


def resolve_camera_parameters(
    image_file_collection,
    cfg: ReduceConfig,
    *,
    measurement: NoiseMeasurement | None = None,
) -> CameraParameters:
    """Instrument, readout mode, gain, read noise, dark rate, saturation level.

    Header values (via :func:`~ost_photometry.reduce.instrument.get_instrument_info`)
    are combined with the camera catalog; explicit values in ``cfg`` win.
    The collection must hold one instrument / readout mode / gain setting.

    With ``cfg.camera_noise_source == "catalog"`` the catalog read noise
    (per native pixel) is scaled to the binned pixel according to
    ``cfg.binning_mode``. With ``"measured"`` gain and read noise are
    measured on the bias / dark / flat frames of the collection (or taken
    from ``measurement``); if that fails, the catalog values are used with
    a warning.
    """
    check_noise_source(cfg.camera_noise_source)
    check_binning_mode(cfg.binning_mode)
    instrument, readout_mode, gain_setting, pixel_bit_value, temperature = (
        utilities.get_instrument_info(
            image_file_collection,
            cfg.temperature_tolerance,
            ignore_readout_mode_mismatch=cfg.ignore_readout_mode_mismatch,
            ignore_instrument_mismatch=cfg.ignore_instrument_mismatch,
        )
    )

    gain = cfg.gain
    read_noise = cfg.read_noise
    dark_rate = cfg.dark_rate
    saturation_level = cfg.saturation_level
    egain = get_egain_from_collection(image_file_collection)
    calibration_gain = None
    if (
        read_noise is None
        or gain is None
        or dark_rate is None
        or saturation_level is None
    ):
        camera_info = calibration_parameters.camera_info(
            instrument,
            readout_mode,
            temperature,
            gain_setting=gain_setting,
        )
        if read_noise is None:
            read_noise = camera_info[0]
        calibration_gain = camera_info[1]
        if dark_rate is None:
            dark_rate = camera_info[2]
        if saturation_level is None:
            saturation_level = pow(2, pixel_bit_value) - 1

    gain = resolve_system_gain(
        instrument,
        gain_setting,
        egain,
        calibration_gain,
        user_gain=cfg.gain,
    )

    binning = _collection_binning(image_file_collection)
    binning_mode = resolve_binning_mode(instrument, cfg.binning_mode)
    noise_source = "user" if cfg.read_noise is not None else "catalog"
    if cfg.read_noise is None:
        catalog_read_noise = read_noise
        read_noise, binning_mode = binned_read_noise(
            read_noise, instrument, *binning, binning_mode=cfg.binning_mode
        )
        if binning != (1, 1) and binning_mode is None:
            terminal_output.print_to_terminal(
                f"Binning {binning[0]}x{binning[1]}, but it is unknown whether "
                f"{instrument!r} bins on the chip or digitally; the catalog read "
                "noise is used unscaled. Set binning_mode='digital' or 'charge'.",
                indent=1,
                style_name="WARNING",
            )
        elif read_noise != catalog_read_noise:
            terminal_output.print_to_terminal(
                f"Read noise {catalog_read_noise:.2f} e- per native pixel -> "
                f"{read_noise:.2f} e- per {binning[0]}x{binning[1]} binned pixel "
                f"({binning_mode} binning).",
                indent=1,
            )

    if cfg.camera_noise_source == "measured":
        if measurement is None:
            measurement = measure_collection_noise(
                image_file_collection, cfg.image_type_dir, saturation_level
            )
        if measurement is None:
            terminal_output.print_to_terminal(
                "camera_noise_source='measured': no bias or dark pairs to measure the "
                f"read noise; using the {noise_source} values.",
                indent=1,
                style_name="WARNING",
            )
        else:
            catalog_gain = gain
            if cfg.gain is None and measurement.gain is not None:
                gain = measurement.gain
            elif cfg.gain is None:
                terminal_output.print_to_terminal(
                    "camera_noise_source='measured': no usable flat pair for the "
                    "gain; using the header / catalog gain.",
                    indent=1,
                    style_name="WARNING",
                )
            if cfg.read_noise is None and gain is not None:
                read_noise = measurement.read_noise(gain)
                noise_source = "measured"
            terminal_output.print_to_terminal(
                f"Measured {measurement.describe()} -> gain {_fmt(gain)} e-/ADU "
                f"(header/catalog {_fmt(catalog_gain)}), read noise {_fmt(read_noise)} e-.",
                indent=1,
            )

    return CameraParameters(
        instrument=instrument,
        readout_mode=readout_mode,
        gain_setting=gain_setting,
        pixel_bit_value=pixel_bit_value,
        temperature=temperature,
        gain=gain,
        read_noise=read_noise,
        dark_rate=dark_rate,
        saturation_level=saturation_level,
        binning=binning,
        binning_mode=binning_mode,
        noise_source=noise_source,
    )


def _run_reduction(cfg: ReduceConfig) -> None:
    """Execute the reduction pipeline with the given configuration."""
    ###
    #   Parameter sanity checks
    #
    if cfg.stack_images and cfg.save_only_transformation:
        terminal_output.print_to_terminal(
            "WARNING: Both 'stack_images' and 'save_only_transformation' "
            "are set to ``True``. It makes no sense to keep only the "
            "transformation matrices if the images are to be stacked, "
            "because the images have to be there to be stacked. -> Set "
            "'save_only_transformation' to ``False``.",
            style_name="WARNING",
        )
        cfg.save_only_transformation = False

    if cfg.estimate_fwhm and not cfg.measure_frame_quality:
        #   Direct ReduceConfig users may still set the deprecated flag.
        cfg.measure_frame_quality = True
    check_noise_source(cfg.camera_noise_source)
    check_binning_mode(cfg.binning_mode)
    check_cosmic_ray_mode(cfg.rm_cosmic_rays)
    check_storage_dtype(cfg.storage_dtype)
    if cfg.stack_weighting not in SUPPORTED_STACK_WEIGHTING:
        raise ValueError(
            f"stack_weighting must be one of {SUPPORTED_STACK_WEIGHTING}, "
            f"got {cfg.stack_weighting!r}"
        )
    if cfg.reference_image_selection not in SUPPORTED_REFERENCE_SELECTION:
        raise ValueError(
            f"reference_image_selection must be one of {SUPPORTED_REFERENCE_SELECTION}, "
            f"got {cfg.reference_image_selection!r}"
        )
    #   Validate the selection early (unknown keys, ranges) and normalise it.
    frame_selection = FrameSelection.from_mapping(cfg.frame_selection)
    if not cfg.measure_frame_quality:
        if frame_selection.is_active():
            raise ValueError(
                "frame_selection requires measure_frame_quality=True."
            )
        if cfg.stack_weighting != "none":
            raise ValueError(
                "stack_weighting requires measure_frame_quality=True."
            )

    ###
    #   Prepare reduction
    #
    file_path = cfg.image_path
    output_path = cfg.output_dir
    image_type_dir = cfg.image_type_dir

    #   Get image file collection
    image_file_collection = utilities.image_file_collection(file_path)

    #   Except if image collection is empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No images found -> EXIT\n"
            f"\t=> Check paths to the images!{style.Bcolors.ENDC}"
        )

    #   Get image types
    ifc_image_types = set(image_file_collection.summary["imagetyp"])

    if cfg.validate_inputs:
        issues = validation.validate_raw_collection(
            image_file_collection,
            image_type_dir,
            sanity_check_sample_size=cfg.sanity_check_sample_size,
        )
        validation.emit_validation_warnings(issues)
        validation.raise_on_fatal_validation_issues(
            issues,
            fail_on_missing_flat=cfg.fail_on_missing_flat,
        )

    #   Check exposure times:   Successful if dark frames with ~ the same
    #                           exposure time are available all flat and
    #                           science
    #   Dark times
    dark_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir["dark"],
    )

    #   Flat times
    flat_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir["flat"],
    )

    #   Science times
    science_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir["light"],
    )

    #   Check if bias frames are available
    bias_true = np.any(
        [True if t in ifc_image_types else False for t in image_type_dir["bias"]]
    ).astype(bool)

    #   Check flats
    image_scaling_required = utilities.check_exposure_times(
        image_file_collection,
        image_type_dir["flat"],
        flat_times,
        dark_times,
        bias_true,
        exposure_time_tolerance=cfg.exposure_time_tolerance,
    )

    #   Check science exposures
    image_scaling_required = image_scaling_required | utilities.check_exposure_times(
        image_file_collection,
        image_type_dir["light"],
        science_times,
        dark_times,
        bias_true,
        exposure_time_tolerance=cfg.exposure_time_tolerance,
    )

    ###
    #   Get camera specific parameters
    #
    camera = resolve_camera_parameters(image_file_collection, cfg)
    instrument = camera.instrument
    gain = camera.gain
    read_noise = camera.read_noise
    dark_rate = camera.dark_rate
    saturation_level = camera.saturation_level

    ###
    #   Check master files on disk
    #
    science_filters = validation.collect_science_filters(
        image_file_collection, image_type_dir
    )
    required_dark_exposure_times = sorted(set(science_times) | set(flat_times))

    #   Check is master files already exist
    master_available = utilities.check_master_files_on_disk(
        output_path,
        image_type_dir,
        required_dark_exposure_times,
        science_filters,
        image_scaling_required,
        exposure_time_tolerance=cfg.exposure_time_tolerance,
    )

    mk_new_master_files = True
    if master_available and cfg.interactive:
        user_input, timed_out = base_utilities.get_input(
            f"{style.Bcolors.OKBLUE}   Master files are already calculated."
            f" Should these files be used? [yes/no] {style.Bcolors.ENDC}"
        )
        if user_input in ["y", "yes"]:
            mk_new_master_files = False

    #   Set master boolean for bias subtraction
    rm_bias = True if image_scaling_required or cfg.enforce_bias else False

    if mk_new_master_files:
        ###
        #   Reduce bias
        #
        if rm_bias:
            terminal_output.print_to_terminal(
                "Create master bias...",
                indent=1,
            )
            master_bias(
                file_path,
                output_path,
                image_type_dir,
                trim_x_start=cfg.trim_x_start,
                trim_x_end=cfg.trim_x_end,
                trim_y_start=cfg.trim_y_start,
                trim_y_end=cfg.trim_y_end,
                dtype=cfg.dtype,
                storage_dtype=cfg.storage_dtype,
            )

        ###
        #   Master dark and master flat darks
        #
        terminal_output.print_to_terminal("Create master darks...", indent=1)

        if rm_bias:
            #   Reduce dark frames and apply bias subtraction
            reduce_dark(
                file_path,
                output_path,
                image_type_dir,
                gain=gain,
                read_noise=read_noise,
                n_cores_multiprocessing=cfg.n_cores_multiprocessing,
                trim_x_start=cfg.trim_x_start,
                trim_x_end=cfg.trim_x_end,
                trim_y_start=cfg.trim_y_start,
                trim_y_end=cfg.trim_y_end,
                storage_dtype=cfg.storage_dtype,
            )

            #   Set dark path
            dark_path = Path(output_path / "dark")
        else:
            dark_path = file_path

        #   Create master dark
        master_dark(
            dark_path,
            output_path,
            image_type_dir,
            gain=gain,
            read_noise=read_noise,
            dark_rate=dark_rate,
            plot_plots=cfg.plot_dark_statistic_plots,
            debug=cfg.debug,
            n_cores_multiprocessing=cfg.n_cores_multiprocessing,
            rm_bias=rm_bias,
            trim_x_start=cfg.trim_x_start,
            trim_x_end=cfg.trim_x_end,
            trim_y_start=cfg.trim_y_start,
            trim_y_end=cfg.trim_y_end,
            dtype=cfg.dtype,
            storage_dtype=cfg.storage_dtype,
        )

        ###
        #   Master flat
        #
        terminal_output.print_to_terminal("Create master flat...", indent=1)

        #   Reduce flats
        reduce_flat(
            file_path,
            output_path,
            image_type_dir,
            gain=gain,
            read_noise=read_noise,
            rm_bias=rm_bias,
            exposure_time_tolerance=cfg.exposure_time_tolerance,
            debug=cfg.debug,
            n_cores_multiprocessing=cfg.n_cores_multiprocessing,
            trim_x_start=cfg.trim_x_start,
            trim_x_end=cfg.trim_x_end,
            trim_y_start=cfg.trim_y_start,
            trim_y_end=cfg.trim_y_end,
            storage_dtype=cfg.storage_dtype,
        )

        #   Create master flat
        master_flat(
            Path(output_path / "flat"),
            output_path,
            image_type_dir,
            plot_plots=cfg.plot_flat_statistic_plots,
            debug=cfg.debug,
            n_cores_multiprocessing=cfg.n_cores_multiprocessing,
            dtype=cfg.dtype,
            storage_dtype=cfg.storage_dtype,
        )

    ###
    #   Image reduction & stacking (calculation of image shifts, etc. )
    #
    terminal_output.print_to_terminal("Reduce science images...", indent=1)

    reduce_light(
        file_path,
        output_path,
        image_type_dir,
        rm_cosmic_rays=cfg.rm_cosmic_rays,
        cosmic_ray_auto_min_frames=cfg.cosmic_ray_auto_min_frames,
        frames_are_stacked=cfg.stack_images,
        storage_dtype=cfg.storage_dtype,
        mask_cosmics=cfg.mask_cosmic_rays,
        gain=gain,
        read_noise=read_noise,
        limiting_contrast_rm_cosmic_rays=cfg.limiting_contrast_rm_cosmic_rays,
        sigma_clipping_value_rm_cosmic_rays=cfg.sigma_clipping_value_rm_cosmic_rays,
        saturation_level=saturation_level,
        rm_bias=rm_bias,
        verbose=cfg.debug,
        add_hot_bad_pixel_mask=cfg.add_hot_bad_pixel_mask,
        exposure_time_tolerance=cfg.exposure_time_tolerance,
        target_name=cfg.target_name,
        scale_image_with_exposure_time=cfg.scale_image_with_exposure_time,
        n_cores_multiprocessing=cfg.n_cores_multiprocessing,
        trim_x_start=cfg.trim_x_start,
        trim_x_end=cfg.trim_x_end,
        trim_y_start=cfg.trim_y_start,
        trim_y_end=cfg.trim_y_end,
        fail_on_missing_flat=cfg.fail_on_missing_flat,
        interactive=cfg.interactive,
    )

    ###
    #   Frame quality: measure, select, weight, choose reference frames
    #
    quality_table = None
    reference_files: dict[str, str] | None = None
    quality_table_path = output_path / "frame_quality.ecsv"
    use_best_fwhm = (
        cfg.reference_image_index is None
        and cfg.reference_image_selection == "best_fwhm"
    )
    if cfg.measure_frame_quality:
        terminal_output.print_to_terminal("Assess frame quality...", indent=1)
        quality_table, references = quality.assess_frame_quality(
            output_path / "light",
            image_type_list=image_type_dir["light"],
            selection=frame_selection,
            stack_weighting=cfg.stack_weighting,
            per_filter_reference=not cfg.shift_all,
            rejected_dir=output_path / "rejected_lights",
            table_path=quality_table_path,
            n_cores_multiprocessing=cfg.n_cores_multiprocessing,
        )
        if use_best_fwhm:
            reference_files = references
    elif use_best_fwhm:
        terminal_output.print_to_terminal(
            "Frame quality measurement is disabled, so the 'best_fwhm' "
            "reference cannot be resolved; using the first frame (index 0).",
            indent=1,
            style_name="WARNING",
        )

    ###
    #   Calculate and apply image shifts
    #
    terminal_output.print_to_terminal(
        "Trim images to the same field of view...",
        indent=1,
    )

    alignment_result = registration.align_images(
        output_path / "light",
        output_path,
        image_type_dir["light"],
        reference_image_index=(
            cfg.reference_image_index if cfg.reference_image_index is not None else 0
        ),
        shift_method=cfg.shift_method,
        n_cores_multiprocessing=cfg.n_cores_multiprocessing,
        rm_outliers=cfg.rm_outliers_image_shifts,
        filter_window=cfg.filter_window_image_shifts,
        threshold=cfg.threshold_image_shifts,
        instrument=instrument,
        debug=cfg.debug,
        image_output_directory="aligned_lights",
        save_only_transformation=cfg.save_only_transformation,
        align_filter_wise=not cfg.shift_all,
        wcs_method=cfg.wcs_method,
        force_wcs_determination=cfg.force_wcs_determination,
        reference_file_names=reference_files,
    )

    if quality_table is not None and len(quality_table):
        merge_alignment_result(quality_table, alignment_result)
        write_quality_table(quality_table, quality_table_path)
        quality.plot_frame_quality(
            quality_table,
            output_path,
            selection=frame_selection,
        )

    #   Set the image directory depending on whether we have aligned images or
    #   just the image transformation matrices.
    if cfg.save_only_transformation:
        image_directory = "light"
    else:
        image_directory = "aligned_lights"

    if cfg.find_wcs and cfg.find_wcs_of_all_images:
        if cfg.shift_method == "wcs" and not cfg.force_wcs_determination:
            terminal_output.print_to_terminal(
                "WCS already solved during alignment (shift_method='wcs'); "
                "skipping a second per-image solve.",
                indent=1,
            )
        else:
            ###
            #   Determine WCS and add it to all reduced images
            #
            terminal_output.print_to_terminal("Determine WCS ...", indent=1)
            utilities.determine_wcs_all_images(
                output_path / image_directory,
                output_path / image_directory,
                wcs_method=cfg.wcs_method,
                force_wcs_determination=cfg.force_wcs_determination,
            )

    if cfg.stack_images:
        ###
        #   Stack images of the individual filters
        #
        terminal_output.print_to_terminal(
            "Combine the images of the individual filter...",
            indent=1,
        )
        stack_image(
            output_path / "aligned_lights",
            output_path,
            image_type_dir["light"],
            stacking_method=cfg.stack_method,
            dtype=cfg.dtype,
            debug=cfg.debug,
            n_cores_multiprocessing=cfg.n_cores_multiprocessing,
            stack_weighting=cfg.stack_weighting,
            quality_table=quality_table,
            keep_input_frames=cfg.keep_aligned_lights,
        )

        if cfg.find_wcs and not cfg.find_wcs_of_all_images:
            ###
            #   Determine WCS and add it to the stacked images
            #
            terminal_output.print_to_terminal("Determine WCS ...", indent=1)

            utilities.determine_wcs_all_images(
                output_path,
                output_path,
                force_wcs_determination=cfg.force_wcs_determination,
                wcs_method=cfg.wcs_method,
                only_combined_images=True,
                image_type=image_type_dir["light"],
            )

        if not cfg.shift_all:
            ###
            #   Make large images with the same dimensions to allow
            #   cross correlation
            #
            enlarged: bool = False
            if cfg.shift_method not in ("aa_true", "wcs"):
                registration.make_big_images(
                    output_path,
                    output_path,
                    image_type_dir["light"],
                )
                enlarged = True

            ###
            #   Calculate and apply image shifts between filters
            #
            terminal_output.print_to_terminal(
                "Trim stacked images of the filters to the same field of view...",
                indent=1,
            )

            registration.align_images(
                output_path,
                output_path,
                image_type_dir["light"],
                shift_method=cfg.shift_method,
                n_cores_multiprocessing=cfg.n_cores_multiprocessing,
                rm_outliers=cfg.rm_outliers_image_shifts,
                filter_window=cfg.filter_window_image_shifts,
                threshold=cfg.threshold_image_shifts,
                debug=cfg.debug,
                save_only_transformation=cfg.save_only_transformation,
                enlarged_only=enlarged,
                terminal_alignment_comment="\tDisplacement between the images of the different filters",
                modify_file_name=True,
                wcs_method=cfg.wcs_method,
                force_wcs_determination=cfg.force_wcs_determination,
            )

    else:
        ###
        #   Sort images according to filter into subdirectories
        #
        #   Select ``light`` frames from image file collection
        light_image_type = utilities.get_image_type(
            image_file_collection,
            image_type_dir,
            image_class="light",
        )
        ifc_filtered = image_file_collection.filter(imagetyp=light_image_type)

        #   Find used filters
        filters = set(
            ifc_filtered.summary["filter"][
                np.invert(ifc_filtered.summary["filter"].mask)
            ]
        )
        for filter_ in filters:
            ###
            #   The aligned images
            #
            if not cfg.save_only_transformation:
                #   Remove old files in the output directory
                checks.clear_directory(output_path / filter_)

                #   Set path to files
                file_path = checks.check_pathlib_path(output_path / "aligned_lights")

                #   New image collection for the images
                image_file_collection = utilities.image_file_collection(file_path)

                #   Restrict to current filter
                filtered_files = image_file_collection.files_filtered(
                    filter=filter_,
                    include_path=True,
                )

                #   Link files to corresponding directory
                base_utilities.link_files(output_path / filter_, filtered_files)

            if cfg.debug or cfg.save_only_transformation:
                ###
                #   The NOT shifted and/or trimmed images
                #
                #   Remove old files in the output directory
                checks.clear_directory(output_path / f"{filter_}_not_aligned")

                #   Set path to files
                file_path = checks.check_pathlib_path(output_path / "light")

                #   New image collection for the images
                image_file_collection = utilities.image_file_collection(file_path)

                #   Restrict to current filter
                filtered_files = image_file_collection.files_filtered(
                    filter=filter_,
                    include_path=True,
                )

                #   Link files to corresponding directory
                base_utilities.link_files(
                    output_path / f"{filter_}_not_aligned",
                    filtered_files,
                )


