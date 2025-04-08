############################################################################
#                               Libraries                                  #
############################################################################

import shutil

from pathlib import Path

import numpy as np

import ccdproc as ccdp

from astropy.stats import mad_std
from astropy.nddata import CCDData
import astropy.units as u

from . import utilities, plots, registration

from .. import checks, style, terminal_output, calibration_parameters

from .. import utilities as base_utilities

from ..analyze.utilities import Executor


############################################################################
#                           Routines & definitions                         #
############################################################################

def reduce_main(
        image_path: str, output_dir: str,
        image_type_dir: dict[str, list[str]] | None = None,
        gain: float | None = None, read_noise: float | None = None,
        dark_rate: float | None = None, rm_cosmic_rays: bool = True,
        mask_cosmic_rays: bool = False, saturation_level: float | None = None,
        limiting_contrast_rm_cosmic_rays: float = 5.,
        sigma_clipping_value_rm_cosmic_rays: float = 4.0,
        scale_image_with_exposure_time: bool = True,
        reference_image_id: int = 0, enforce_bias: bool = False,
        add_hot_bad_pixel_mask: bool = True, shift_method: str = 'skimage',
        n_cores_multiprocessing: int | None = None, stack_images: bool = True,
        estimate_fwhm: bool = False, shift_all: bool = False,
        exposure_time_tolerance: float = 0.5, stack_method: str = 'average',
        target_name: str | None = None, find_wcs: bool = True,
        wcs_method: str = 'astrometry', find_wcs_of_all_images: bool = False,
        force_wcs_determination: bool = False,
        rm_outliers_image_shifts: bool = True,
        filter_window_image_shifts: int = 25,
        threshold_image_shifts: float = 10., temperature_tolerance: float = 5.,
        plot_dark_statistic_plots: bool = False,
        plot_flat_statistic_plots: bool = False,
        ignore_readout_mode_mismatch: bool = False, trim_x_start: int = 0,
        trim_x_end: int = 0, trim_y_start: int = 0, trim_y_end: int = 0,
        dtype: str | np.dtype | None = None, debug: bool = False,
        save_only_transformation: bool = False,
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
        The read noise (e-) of the camera chip.
        Default is ``None``.

    dark_rate
        Dark rate in e-/pix/s:
        Default is ``None``.

    rm_cosmic_rays
        If True cosmics rays will be removed.
        Default is ``True``.

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

    reference_image_id
        ID of the image that should be used as a reference
        Default is ``0``.

    enforce_bias
        If True the usage of bias frames during the reduction is
        enforced if possible.
        Default is ``False``.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    shift_method
        Method to use for image alignment.
        Possibilities: 'aa'      = astroalign module only accounting for
                                   xy shifts
                       'aa_true' = astroalign module with corresponding
                                   transformation
                       'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation with skimage
        Default is ``skimage``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    stack_images
        If True the individual images of each filter will be stacked and
        those images will be aligned to each other.
        Default is ``True``.

    estimate_fwhm
        If True the FWHM of each image will be estimated.
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
        Default is ``astrometry``.

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
    """
    ###
    #   Parameter sanity checks (some parameter combination do not make sence)
    #
    #   It makes no sense to keep only the transformation matrices if the
    #   images are to be stacked, because the images have to be there to be
    #   stacked.
    if stack_images and save_only_transformation:
        terminal_output.print_to_terminal(
            "WARNING: Both 'stack_images' and 'save_only_transformation' "
            "are set to ``True``. It makes no sense to keep only the "
            "transformation matrices if the images are to be stacked, "
            "because the images have to be there to be stacked. -> Set "
            "'save_only_transformation' to ``False``.",
            style_name='WARNING',
        )
        save_only_transformation = False

    ###
    #   Prepare reduction
    #
    #   Sanitize the provided paths
    file_path = Path(image_path)
    output_path = Path(output_dir)

    #   Get image file collection
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Get image types
    if image_type_dir is None:
        image_type_dir = calibration_parameters.get_image_types()

    #   Except if image collection is empty
    if not image_file_collection.files:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}No images found -> EXIT\n'
            f'\t=> Check paths to the images!{style.Bcolors.ENDC}'
        )

    #   Get image types
    ifc_image_types = set(image_file_collection.summary['imagetyp'])

    #   TODO: Add a completeness check so that all science images have
    #         the necessary flats. Add here or in utilities.

    #   Check exposure times:   Successful if dark frames with ~ the same
    #                           exposure time are available all flat and
    #                           science
    #   Dark times
    dark_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir['dark'],
    )

    #   Flat times
    flat_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir['flat'],
    )

    #   Science times
    science_times = utilities.get_exposure_times(
        image_file_collection,
        image_type_dir['light'],
    )

    #   Check if bias frames are available
    bias_true = np.any(
        [True if t in ifc_image_types else False for t in image_type_dir['bias']]
    ).astype(bool)

    #   Check flats
    image_scaling_required = utilities.check_exposure_times(
        image_file_collection,
        image_type_dir['flat'],
        flat_times,
        dark_times,
        bias_true,
        exposure_time_tolerance=exposure_time_tolerance,
    )

    #   Check science exposures
    image_scaling_required = image_scaling_required | utilities.check_exposure_times(
        image_file_collection,
        image_type_dir['light'],
        science_times,
        dark_times,
        bias_true,
        exposure_time_tolerance=exposure_time_tolerance,
    )

    ###
    #   Get camera specific parameters
    #
    image_parameters = utilities.get_instrument_info(
        image_file_collection,
        temperature_tolerance,
        ignore_readout_mode_mismatch=ignore_readout_mode_mismatch,
    )
    instrument = image_parameters[0]
    readout_mode = image_parameters[1]
    gain_setting = image_parameters[2]
    pixel_bit_value = image_parameters[3]
    temperature = image_parameters[4]

    if (read_noise is None or gain is None or dark_rate is None
            or saturation_level is None):
        camera_info = calibration_parameters.camera_info(
            instrument,
            readout_mode,
            temperature,
            gain_setting=gain_setting,
        )
        if read_noise is None:
            read_noise = camera_info[0]
        if gain is None:
            gain = camera_info[1]
        if dark_rate is None:
            dark_rate = camera_info[2]
        if saturation_level is None:
            saturation_level = pow(2, pixel_bit_value) - 1

    ###
    #   Check master files on disk
    #
    #   Get all filter
    filters = set(
        image_file_collection.summary['filter'][
            np.invert(image_file_collection.summary['filter'].mask)
        ]
    )

    #   Check is master files already exist
    master_available = utilities.check_master_files_on_disk(
        output_path,
        image_type_dir,
        dark_times,
        filters,
        image_scaling_required,
    )

    mk_new_master_files = True
    if master_available:
        user_input, timed_out = base_utilities.get_input(
            f"{style.Bcolors.OKBLUE}   Master files are already calculated."
            f" Should these files be used? [yes/no] {style.Bcolors.ENDC}"
        )
        if timed_out:
            user_input = 'n'

        if user_input in ['y', 'yes']:
            mk_new_master_files = False

    #   Set master boolean for bias subtraction
    rm_bias = True if image_scaling_required or enforce_bias else False

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
                trim_x_start=trim_x_start,
                trim_x_end=trim_x_end,
                trim_y_start=trim_y_start,
                trim_y_end=trim_y_end,
                dtype=dtype,
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
                n_cores_multiprocessing=n_cores_multiprocessing,
                trim_x_start=trim_x_start,
                trim_x_end=trim_x_end,
                trim_y_start=trim_y_start,
                trim_y_end=trim_y_end,
            )

            #   Set dark path
            dark_path = Path(output_path / 'dark')
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
            plot_plots=plot_dark_statistic_plots,
            debug=debug,
            n_cores_multiprocessing=n_cores_multiprocessing,
            rm_bias=rm_bias,
            trim_x_start=trim_x_start,
            trim_x_end=trim_x_end,
            trim_y_start=trim_y_start,
            trim_y_end=trim_y_end,
            dtype=dtype,
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
            exposure_time_tolerance=exposure_time_tolerance,
            debug=debug,
            n_cores_multiprocessing=n_cores_multiprocessing,
            trim_x_start=trim_x_start,
            trim_x_end=trim_x_end,
            trim_y_start=trim_y_start,
            trim_y_end=trim_y_end,
        )

        #   Create master flat
        master_flat(
            Path(output_path / 'flat'),
            output_path,
            image_type_dir,
            plot_plots=plot_flat_statistic_plots,
            debug=debug,
            # n_cores_multiprocessing=n_cores_multiprocessing,
            n_cores_multiprocessing=1,
            dtype=dtype,
        )

    ###
    #   Image reduction & stacking (calculation of image shifts, etc. )
    #
    terminal_output.print_to_terminal("Reduce science images...", indent=1)

    reduce_light(
        file_path,
        output_path,
        image_type_dir,
        rm_cosmic_rays=rm_cosmic_rays,
        mask_cosmics=mask_cosmic_rays,
        gain=gain,
        read_noise=read_noise,
        limiting_contrast_rm_cosmic_rays=limiting_contrast_rm_cosmic_rays,
        sigma_clipping_value_rm_cosmic_rays=sigma_clipping_value_rm_cosmic_rays,
        saturation_level=saturation_level,
        rm_bias=rm_bias,
        verbose=debug,
        add_hot_bad_pixel_mask=add_hot_bad_pixel_mask,
        exposure_time_tolerance=exposure_time_tolerance,
        target_name=target_name,
        scale_image_with_exposure_time=scale_image_with_exposure_time,
        n_cores_multiprocessing=n_cores_multiprocessing,
        trim_x_start=trim_x_start,
        trim_x_end=trim_x_end,
        trim_y_start=trim_y_start,
        trim_y_end=trim_y_end,
    )

    ###
    #   Calculate and apply image shifts
    #
    terminal_output.print_to_terminal(
        "Trim images to the same field of view...",
        indent=1,
    )

    registration.align_images(
        output_path / 'light',
        output_path,
        image_type_dir['light'],
        reference_image_id=reference_image_id,
        shift_method=shift_method,
        n_cores_multiprocessing=n_cores_multiprocessing,
        rm_outliers=rm_outliers_image_shifts,
        filter_window=filter_window_image_shifts,
        threshold=threshold_image_shifts,
        instrument=instrument,
        debug=debug,
        image_output_directory='aligned_lights',
        save_only_transformation=save_only_transformation,
        align_filter_wise=not shift_all,
    )

    #   Set the image directory depending on whether we have aligned images or
    #   just the image transformation matrices.
    if save_only_transformation:
        image_directory = 'light'
    else:
        image_directory = 'aligned_lights'

    if find_wcs and find_wcs_of_all_images:
        ###
        #   Determine WCS and add it to all reduced images
        #
        terminal_output.print_to_terminal("Determine WCS ...", indent=1)
        utilities.determine_wcs_all_images(
            output_path / image_directory,
            output_path / image_directory,
            wcs_method=wcs_method,
            force_wcs_determination=force_wcs_determination,
        )

    if estimate_fwhm:
        ###
        #   Estimate FWHM
        #
        terminal_output.print_to_terminal("Estimate FWHM ...", indent=1)
        utilities.estimate_fwhm(
            # output_path / 'aligned_lights',
            output_path / image_directory,
            output_path / 'fwhm',
            image_type_dir['light'],
        )

    if stack_images:
        ###
        #   Stack images of the individual filters
        #
        terminal_output.print_to_terminal(
            "Combine the images of the individual filter...",
            indent=1,
        )
        stack_image(
            output_path / 'aligned_lights',
            output_path,
            image_type_dir['light'],
            stacking_method=stack_method,
            dtype=dtype,
            debug=debug,
        )

        if find_wcs and not find_wcs_of_all_images:
            ###
            #   Determine WCS and add it to the stacked images
            #
            terminal_output.print_to_terminal("Determine WCS ...", indent=1)

            utilities.determine_wcs_all_images(
                output_path,
                output_path,
                force_wcs_determination=force_wcs_determination,
                wcs_method=wcs_method,
                only_combined_images=True,
                image_type=image_type_dir['light'],
            )

        if not shift_all:
            ###
            #   Make large images with the same dimensions to allow
            #   cross correlation
            #
            enlarged: bool = False
            if shift_method != 'aa_true':
                registration.make_big_images(
                    output_path,
                    output_path,
                    image_type_dir['light'],
                )
                enlarged = True

            ###
            #   Calculate and apply image shifts between filters
            #
            terminal_output.print_to_terminal(
                "Trim stacked images of the filters to the same "
                "field of view...",
                indent=1,
            )

            registration.align_images(
                output_path,
                output_path,
                image_type_dir['light'],
                shift_method=shift_method,
                n_cores_multiprocessing=n_cores_multiprocessing,
                rm_outliers=rm_outliers_image_shifts,
                filter_window=filter_window_image_shifts,
                threshold=threshold_image_shifts,
                debug=debug,
                save_only_transformation=save_only_transformation,
                enlarged_only=enlarged,
                terminal_alignment_comment='\tDisplacement between the images of the different filters',
                modify_file_name=True,
            )

    else:
        ###
        #   Sort images according to filter into subdirectories
        #
        #   Select ``light`` frames from image file collection
        light_image_type = utilities.get_image_type(
            image_file_collection,
            image_type_dir,
            image_class='light',
        )
        ifc_filtered = image_file_collection.filter(imagetyp=light_image_type)

        #   Find used filters
        filters = set(
            ifc_filtered.summary['filter'][
                np.invert(ifc_filtered.summary['filter'].mask)
            ]
        )
        for filter_ in filters:
            ###
            #   The aligned images
            #
            if not save_only_transformation:
                #   Remove old files in the output directory
                checks.clear_directory(output_path / filter_)

                #   Set path to files
                file_path = checks.check_pathlib_path(
                    output_path / 'aligned_lights'
                )

                #   New image collection for the images
                image_file_collection = ccdp.ImageFileCollection(file_path)

                #   Restrict to current filter
                filtered_files = image_file_collection.files_filtered(
                    filter=filter_,
                    include_path=True,
                )

                #   Link files to corresponding directory
                base_utilities.link_files(output_path / filter_, filtered_files)

            if debug or save_only_transformation:
                ###
                #   The NOT shifted and/or trimmed images
                #
                #   Remove old files in the output directory
                checks.clear_directory(output_path / f'{filter_}_not_aligned')

                #   Set path to files
                file_path = checks.check_pathlib_path(output_path / 'light')

                #   New image collection for the images
                image_file_collection = ccdp.ImageFileCollection(file_path)

                #   Restrict to current filter
                filtered_files = image_file_collection.files_filtered(
                    filter=filter_,
                    include_path=True,
                )

                #   Link files to corresponding directory
                base_utilities.link_files(
                    output_path / f'{filter_}_not_aligned',
                    filtered_files,
                )


def master_bias(
        bias_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], trim_x_start: int = 0,
        trim_x_end: int = 0, trim_y_start: int = 0,
        trim_y_end: int = 0, dtype: str | np.dtype | None = None
    ) -> None:
    """
    This function calculates master biases from individual bias images
    located in one directory.

    Parameters
    ----------
    bias_path            : `string` or `pathlib.Path`
        Path to the images

    output_dir           : `string` or `pathlib.Path`
        Path to the directory where the master files should be saved to

    image_type           : `dictionary`
        Image types of the images. Possibilities: bias, dark, flat,
        light

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
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(bias_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    #   Get bias frames
    bias_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='bias',
    )
    bias_frames = image_file_collection.files_filtered(
        imagetyp=bias_image_type,
        include_path=True,
    )

    #   Combine biases: Average images + sigma clipping to remove outliers,
    #                   set memory limit to 15GB, set unit to 'adu' since
    #                   this is not set in our images -> find better
    #                   solution
    combined_bias = ccdp.combine(
        bias_frames,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        signma_clip_dev_func=mad_std,
        mem_limit=15e9,
        unit='adu',
        dtype=dtype,
    )

    #   Trimming the image, for example to remove an overscan region
    image_shape = combined_bias.data.shape
    combined_bias = combined_bias[
                        trim_y_start:image_shape[0]-trim_y_end,
                        trim_x_start:image_shape[1]-trim_x_end
                    ]

    #   Add header keyword to mark the file as a Master
    combined_bias.meta['combined'] = True

    #   Write file to disk
    combined_bias.write(out_path / 'combined_bias.fit', overwrite=True)


def master_image_list(*args, **kwargs):
    """
        Wrapper function to create a master calibration image for the files
        in the directories given in the path list 'paths'
    """
    if kwargs['calib_type'] == 'dark':
        master_dark(*args, **kwargs)
    elif kwargs['calib_type'] == 'flat':
        master_flat(*args, **kwargs)


def reduce_dark(
        image_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], gain: float | None = None,
        read_noise: float = 8., n_cores_multiprocessing: int | None = None,
        trim_x_start: int = 0, trim_x_end: int = 0, trim_y_start: int = 0,
        trim_y_end: int = 0
    ) -> None:
    """
    Reduce dark images: This function reduces the raw dark frames

    Parameters
    ----------
    image_path          : `string` or `pathlib.Path`
        Path to the images

    output_dir          : `string` or `pathlib.Path`
        Path to the directory where the master files should be saved to

    image_type          : `dictionary`
        Image types of the images. Possibilities: bias, dark, flat,
        light

    gain                : `float` or `None`, optional
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise          : `float`, optional
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

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
    """
    terminal_output.print_to_terminal("Reduce darks...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection for the raw data
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Create image collection for the reduced data
    image_file_collection_reduced = ccdp.ImageFileCollection(out_path)

    #   Get master bias
    bias_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class='bias',
    )
    stacked_bias = CCDData.read(
        image_file_collection_reduced.files_filtered(
            imagetyp=bias_image_type,
            combined=True,
            include_path=True,
        )[0]
    )

    #   Set new dark path
    dark_path = Path(out_path / 'dark')
    checks.clear_directory(dark_path)

    #   Determine possible image types
    dark_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='dark',
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=dark_image_type)),
        add_progress_bar=True,
    )

    #   Loop over darks and reduce darks
    for file_name in image_file_collection.files_filtered(
            include_path=True,
            imagetyp=dark_image_type,
    ):
        executor.schedule(
            reduce_dark_image,
            args=(
                file_name,
                stacked_bias,
                dark_path,
            ),
            kwargs={
                'gain': gain,
                'read_noise': read_noise,
                'trim_x_start': trim_x_start,
                'trim_x_end': trim_x_end,
                'trim_y_start': trim_y_start,
                'trim_y_end': trim_y_end,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Dark image reduction using multiprocessing'
            f' failed :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

def reduce_dark_image(
        dark_file_name: str, stacked_bias: CCDData, dark_path: Path,
        gain: float | None = None, read_noise: float = 8., trim_x_start: int = 0,
        trim_x_end: int = 0, trim_y_start: int = 0, trim_y_end: int = 0
    ) -> None:
    """
    This function reduces the individual raw dark frame images

    Parameters
    ----------
    dark_file_name
        The file name of the dark image that will be reduced

    stacked_bias
        Reduced and stacked Bias CCDData object

    dark_path
        Path where the reduced images should be saved

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

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
    """
    #   Read image file
    dark = CCDData.read(dark_file_name, unit='adu')

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = dark.header['EGAIN']

    #   Trimming the image, for example to remove an overscan region
    image_shape = dark.data.shape
    dark = dark[
                trim_y_start:image_shape[0]-trim_y_end,
                trim_x_start:image_shape[1]-trim_x_end
            ]

    #   Calculated uncertainty
    dark = ccdp.create_deviation(
        dark,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    # Subtract bias
    dark = ccdp.subtract_bias(dark, stacked_bias)

    #   Save the result
    file_name = dark_file_name.split('/')[-1]
    dark.write(dark_path / file_name, overwrite=True)


def master_dark(
        image_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], gain: float | None = None,
        read_noise: float = 8., dark_rate: float | None = None,
        mk_hot_pixel_mask: bool = True, plot_plots: bool = False,
        debug: bool = False, n_cores_multiprocessing: int | None = None,
        rm_bias: bool = False, trim_x_start: int = 0, trim_x_end: int = 0,
        trim_y_start: int = 0, trim_y_end: int = 0,
        dtype: str | np.dtype | None = None, **kwargs
    ) -> None:
    """
    This function calculates master darks from individual dark images
    located in one directory. The dark images are group according to
    their exposure time.

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    dark_rate
        Temperature dependent dark rate in e-/pix/s:
        Default is ``None``.

    mk_hot_pixel_mask
        If True a hot pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the dark frames are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    rm_bias
        If True the master bias image will be subtracted from the flats
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
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    terminal_output.print_to_terminal("Stack darks...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Sanitize dark rate
    if dark_rate is None:
        terminal_output.print_to_terminal(
            f"Dark current not specified. Assume 0.1 e-/pix/s.",
            indent=1,
            style_name='WARNING',
        )
        # dark_rate = {0: 0.1}
        dark_rate = 0.1

    #   Create image collection
    try:
        image_file_collection = ccdp.ImageFileCollection(out_path / 'dark')
    except FileNotFoundError:
        image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    #   Find darks
    dark_mask = [True if file in image_type['dark'] else False
                 for file in image_file_collection.summary['imagetyp']]

    #   Return if no darks are found in this directory
    if not dark_mask:
        return

    #   Get all available shapes with exposure times
    all_available_image_shapes_and_exposure_times: set[tuple[int, int, float]] = set(tuple(zip(
        image_file_collection.summary['naxis1'][dark_mask],
        image_file_collection.summary['naxis2'][dark_mask],
        image_file_collection.summary['exptime'][dark_mask]
    )))

    #   Get only the shapes
    all_available_image_shapes: set[tuple[int, int]] = set(tuple(zip(
        image_file_collection.summary['naxis1'][dark_mask],
        image_file_collection.summary['naxis2'][dark_mask]
    )))

    #   Get the maximum exposure time for each shape
    max_exposure_time_per_shape: list = []
    for shape in all_available_image_shapes:
        exposure_times: list = []
        for shape_expo_time in all_available_image_shapes_and_exposure_times:
            if shape[0] == shape_expo_time[0] and shape[1] == shape_expo_time[1]:
                exposure_times.append(shape_expo_time[2])
        max_exposure_time_per_shape.append((*shape, np.max(exposure_times)))

    #   Get exposure times (set allows to return only unique values)
    dark_exposure_times = set(
        image_file_collection.summary['exptime'][dark_mask]
    )

    #   Get dark image type
    dark_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='dark',
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(sorted(dark_exposure_times)),
        add_progress_bar=True,
    )
    # executor = Executor(n_cores_multiprocessing)

    #   Reduce science images and save to an extra directory
    for exposure_time in sorted(dark_exposure_times):
        executor.schedule(
            master_dark_stacking,
            args=(
                image_file_collection,
                exposure_time,
                dark_image_type,
                max_exposure_time_per_shape,
                out_path,
                dark_rate,
            ),
            kwargs={
                'gain': gain,
                'read_noise': read_noise,
                'mk_hot_pixel_mask': mk_hot_pixel_mask,
                'plot_plots': plot_plots,
                'rm_bias': rm_bias,
                'trim_x_start': trim_x_start,
                'trim_x_end': trim_x_end,
                'trim_y_start': trim_y_start,
                'trim_y_end': trim_y_end,
                'dtype': dtype,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Dark image stacking using multiprocessing'
            f' failed :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(out_path / 'dark', ignore_errors=True)

def master_dark_stacking(
        image_file_collection: ccdp.ImageFileCollection,
        exposure_time: float, dark_image_type: str | list[str] | None,
        max_exposure_time_per_shape: list[tuple[int, int, float]],
        out_path: Path, dark_rate: float, gain: int | None = None,
        read_noise: float = 8., mk_hot_pixel_mask: bool = True,
        plot_plots: bool = False, debug: bool = False, rm_bias: bool = False,
        trim_x_start: int = 0, trim_x_end: int = 0, trim_y_start: int = 0,
        trim_y_end: int = 0, dtype: str | np.dtype | None = None
    ) -> None:
    """
    This function stacks all dark images with the same exposure time.

    Parameters
    ----------
    image_file_collection
        Image file collection for referencing all dark files

    exposure_time
        Exposure time of the current set of dark images

    dark_image_type
        Image type designation used for dark files

    out_path
        Path to the directory where the master files should be saved to

    max_exposure_time_per_shape
        Maximum exposure time for each available image shape

    dark_rate
        Temperature dependent dark rate in e-/pix/s:

    gain
        The gain (e-/adu) of the camera. If set to `None` the gain will
        be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera.
        Default is 8 e-.

    mk_hot_pixel_mask
        If True a hot pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the dark frames are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    rm_bias
        If True the master bias image will be subtracted from the flats
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
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    #   Get only the darks with the correct exposure time
    calibrated_darks = image_file_collection.files_filtered(
        imagetyp=dark_image_type,
        exptime=exposure_time,
        include_path=True,
    )

    #   Combine darks: Average images + sigma clipping to remove
    #                  outliers, set memory limit to 15GB, set unit to
    #                  'adu' since this is not set in our images
    #                  -> find better solution
    combined_dark = ccdp.combine(
        calibrated_darks,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        mem_limit=15e9,
        unit='adu',
        dtype=dtype,
    )

    #   Trimming the image, for example to remove an overscan region
    if not rm_bias:
        image_shape = combined_dark.data.shape
        combined_dark = combined_dark[
                            trim_y_start:image_shape[0]-trim_y_end,
                            trim_x_start:image_shape[1]-trim_x_end
                        ]

    #   Add Header keyword to mark the file as a Master
    combined_dark.meta['combined'] = True

    #   Write file to disk
    dark_file_name = f'combined_dark_{exposure_time:4.2f}.fit'
    combined_dark.write(out_path / dark_file_name, overwrite=True)

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = int(combined_dark.header['EGAIN'])

    #   Plot histogram
    if plot_plots:
        plots.plot_histogram(
            combined_dark.data,
            out_path,
            gain,
            exposure_time,
        )
        plots.plot_dark_with_distributions(
            combined_dark.data,
            read_noise,
            dark_rate,
            out_path,
            exposure_time=exposure_time,
            gain=gain,
        )

    #   Create mask with hot pixels
    current_shape_x = combined_dark.meta['naxis1']
    current_shape_y = combined_dark.meta['naxis2']
    if ((current_shape_x, current_shape_y, exposure_time) in
            max_exposure_time_per_shape and mk_hot_pixel_mask):
        utilities.make_hot_pixel_mask(
            combined_dark,
            gain,
            out_path,
            verbose=debug,
        )


def reduce_flat(
        image_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], gain: float | None = None,
        read_noise: float = 8., rm_bias: bool = False,
        exposure_time_tolerance: float = 0.5,
        n_cores_multiprocessing: int | None = None, trim_x_start: int = 0,
        trim_x_end: int = 0, trim_y_start: int = 0, trim_y_end: int = 0,
        **kwargs) -> None:
    """
    Reduce flat images: This function reduces the raw flat frames,
                        subtracts master dark and if necessary also
                        master bias

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    gain
        The gain (e-/adu) of the camera. If set to `None` the gain will
        be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera.
        Default is 8 e-.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    exposure_time_tolerance
        Maximum difference, in seconds, between the image and the
        closest entry from the exposure time list. Set to ``None`` to
        skip the tolerance test.
        Default is ``0.5``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

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
    """
    terminal_output.print_to_terminal("Reduce flats...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create image collection for the flats
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        return

    #   Find flats
    #   TODO: Rewrite with image_file_collection from the loop below
    flats = [
        True if file in image_type['flat'] else False for file in
        image_file_collection.summary['imagetyp']
    ]

    #   Return if no flats are found in this directory
    if not flats:
        return

    #   Get image collection for the reduced files
    image_file_collection_reduced = ccdp.ImageFileCollection(out_path)

    #   Get master dark
    dark_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class='dark',
    )
    combined_darks = {
        ccd.header['exptime']: ccd for ccd in image_file_collection_reduced.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }

    #   Get master bias
    combined_bias = None
    if rm_bias:
        bias_image_type = utilities.get_image_type(
            image_file_collection_reduced,
            image_type,
            image_class='bias',
        )

        combined_bias = CCDData.read(
            image_file_collection_reduced.files_filtered(
                imagetyp=bias_image_type,
                combined=True,
                include_path=True,
            )[0]
        )

    #   Set new flat path
    flat_path = Path(out_path / 'flat')
    checks.clear_directory(flat_path)

    #   Get flat image types
    flat_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='flat',
    )

    #   Initialize multiprocessing object
    #   TODO: Replace with flats below if rewrite above has happened
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=flat_image_type)),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for file_name in image_file_collection.files_filtered(
            include_path=True,
            imagetyp=flat_image_type,
    ):
        executor.schedule(
            reduce_flat_image,
            args=(
                file_name,
                combined_bias,
                combined_darks,
                flat_path
            ),
            kwargs={
                'gain': gain,
                'read_noise': read_noise,
                'rm_bias': rm_bias,
                'exposure_time_tolerance': exposure_time_tolerance,
                'trim_x_start': trim_x_start,
                'trim_x_end': trim_x_end,
                'trim_y_start': trim_y_start,
                'trim_y_end': trim_y_end,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Flat image reduction using multiprocessing'
            f' failed :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()


def reduce_flat_image(
        flat_file_name: str, combined_bias: CCDData | None,
        combined_darks: dict[float, CCDData],
        flat_path: Path, gain: float | None = None, read_noise: float = 8.,
        rm_bias: bool = False, exposure_time_tolerance: float = 0.5,
        trim_x_start: int = 0, trim_x_end: int = 0, trim_y_start: int = 0,
        trim_y_end: int = 0
    ) -> None:
    """
    Reduce an individual image

    Parameters
    ----------
    flat_file_name
        The CCDData object of the flat that should be reduced.

    combined_bias
        Reduced and stacked Bias CCDData object

    combined_darks
        Combined darks in a dictionary with exposure times as keys and
        CCDData object as values.

    flat_path
        Path where the reduced flats should be saved

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

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
    """
    #   Read fla image
    flat = CCDData.read(flat_file_name, unit='adu')

    #   Trimming the image, for example to remove an overscan region
    image_shape = flat.data.shape
    flat = flat[
                trim_y_start:image_shape[0]-trim_y_end,
                trim_x_start:image_shape[1]-trim_x_end
            ]

    #   Set gain _> get it from Header if not provided
    if gain is None:
        gain = flat.header['EGAIN']

    #   Calculated uncertainty
    flat = ccdp.create_deviation(
        flat,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    # Subtract bias
    if rm_bias:
        flat = ccdp.subtract_bias(flat, combined_bias)

    #   Find the correct dark exposure
    valid_dark_available, closest_dark_exposure_time = utilities.find_nearest_exposure_time_to_reference_image(
        flat,
        list(combined_darks.keys()),
        time_tolerance=exposure_time_tolerance,
    )

    #   Exit if no dark with a similar exposure time have been found
    if not valid_dark_available and not rm_bias:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Closest dark exposure time is "
            f"{closest_dark_exposure_time} for flat of exposure time "
            f"{flat.header['exptime']}. {style.Bcolors.ENDC}"
        )

    #   Subtract the dark current
    flat = ccdp.subtract_dark(
        flat,
        combined_darks[closest_dark_exposure_time],
        exposure_time='exptime',
        exposure_unit=u.second,
        scale=rm_bias,
    )

    #   Save the result
    file_name = flat_file_name.split('/')[-1]
    flat.write(flat_path / file_name, overwrite=True)


def master_flat(
        image_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], mk_bad_pixel_mask: bool = True,
        plot_plots: bool = False, debug: bool = False,
        n_cores_multiprocessing: int | None = None,
        dtype: str | np.dtype | None = None, **kwargs
    ) -> None:
    """
    This function calculates master flats from individual flat field
    images located in one directory. The flat field images are group
    according to their exposure time.

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    mk_bad_pixel_mask
        If True a bad pixel mask is created.
        Default is ``True``.

    plot_plots
        If True some plots showing some statistic on the flat fields are
        created.
        Default is ``False``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64
    """
    terminal_output.print_to_terminal("Stack flats...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Create new image collection for the reduced flat images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Determine filter
    flat_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='flat',
    )
    filters = set(
        h['filter'] for h in image_file_collection.headers(imagetyp=flat_image_type)
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(filters),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for filter_ in filters:
        executor.schedule(
            stack_flat_images,
            args=(
                image_file_collection,
                flat_image_type,
                filter_,
                out_path,
            ),
            kwargs={
                'plot_plots': plot_plots,
                'dtype': dtype,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Stacking of flat images using multiprocessing'
            f' failed :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Collect multiprocessing results
    #
    #   Get bad pixel masks
    bad_pixel_mask_list: list[np.ndarray] = executor.res

    if mk_bad_pixel_mask:
        utilities.make_bad_pixel_mask(
            bad_pixel_mask_list,
            out_path,
            verbose=debug,
        )

    #   Remove reduced dark files if they exist
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)


def stack_flat_images(
        image_file_collection: ccdp.ImageFileCollection,
        flat_image_type: str | list[str] | None, filter_: str, out_path: Path,
        plot_plots: bool = False, dtype: str | np.dtype | None = None
    ) -> np.ndarray:
    """
    Stack flats for the individual filters

    Parameters
    ----------
    image_file_collection
        Image file collection for referencing all dark files

    flat_image_type
        Image type designation used for dark files

    filter_
        Current filter

    out_path
        Path to the directory where the master files should be saved to

    plot_plots
        If True some plots showing some statistic on the flat fields are
        created.
        Default is ``False``.

    dtype
        Data type used in the ccdproc calculations
        Default is ''None''. -> None is equivalent to float64

    Returns
    -------
    bad_pixel_mask_list
    """
    #   Select flats to combine
    flats_to_combine = image_file_collection.files_filtered(
        imagetyp=flat_image_type,
        filter=filter_,
        include_path=True,
    )

    #   Combine darks: Average images + sigma clipping to remove
    #                  outliers, set memory limit to 15GB, scale the
    #                  frames so that they have the same median value
    #                  ('inv_median')
    combined_flat = ccdp.combine(
        flats_to_combine,
        method='average',
        scale=utilities.inverse_median,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        signma_clip_dev_func=mad_std,
        mem_limit=15e9,
        dtype=dtype,
    )

    #   Add Header keyword to mark the file as a Master
    combined_flat.meta['combined'] = True

    #   Define name and write file to disk
    flat_file_name = 'combined_flat_filter_{}.fit'.format(
        filter_.replace("''", "p")
    )
    combined_flat.write(out_path / flat_file_name, overwrite=True)

    #   Plot flat medians and means
    if plot_plots:
        plots.plot_median_of_flat_fields(
            image_file_collection,
            flat_image_type,
            out_path,
            filter_,
        )

    return ccdp.ccdmask(combined_flat.data)


def reduce_master(paths, *args, **kwargs):
    """
    Wrapper function for reduction of the science images

    Parameters
    ----------
    paths           : `list of strings`
        List with paths to the images
    """
    if isinstance(paths, list):
        for path in paths:
            reduce_light(path, *args, **kwargs)
    elif isinstance(paths, str) or isinstance(paths, Path):
        reduce_light(paths, *args, **kwargs)
    else:
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Supplied path is neither str nor list'
            f'{style.Bcolors.ENDC}'
        )


def reduce_light(
        image_path: str | Path, output_dir: str | Path,
        image_type: dict[str, list[str]], rm_cosmic_rays: bool = True,
        mask_cosmics: bool = False, gain: float | None = None,
        read_noise: float = 8., saturation_level: float | None = 65535.,
        limiting_contrast_rm_cosmic_rays: float = 5.,
        sigma_clipping_value_rm_cosmic_rays: float = 4.5,
        scale_image_with_exposure_time: bool = True, rm_bias: bool = False,
        verbose: bool = False, add_hot_bad_pixel_mask: bool = True,
        exposure_time_tolerance: float = 0.5,
        target_name: str | None = None,
        n_cores_multiprocessing: int | None = None, trim_x_start: int = 0,
        trim_x_end: int = 0, trim_y_start: int = 0, trim_y_end: int = 0
    ) -> None:
    """
    Reduce the science images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be stored

    image_type
        Image types of the images. Possibilities: bias, dark, flat,
        light

    rm_cosmic_rays
        If True cosmic rays will be removed.
        Default is ``True``.

    mask_cosmics
        If True cosmics will ''only'' be masked. If False the
        cosmics will be removed from the input image and the mask will
        be added.
        Default is ``False``.

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``65535``.

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

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    target_name
        Name of the target. Used for file selection.
        Default is ``None``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

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
    """
    terminal_output.print_to_terminal("Reduce light images...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Get image collection for the science images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Return if image collection is empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tNo object image detected.\n\t"
            f"-> EXIT{style.Bcolors.ENDC}"
        )

    #   Limit images to those of the target. If a target is given.
    if target_name is not None:
        image_file_collection = image_file_collection.filter(
            object=target_name
        )

    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \tERROR: No image left after filtering by "
            f"object name.\n\t-> EXIT{style.Bcolors.ENDC}"
        )

    #   Find science images
    lights = [True if file in image_type['light'] else False for file in
              image_file_collection.summary['imagetyp']]

    #   Return if no science images are found in this directory
    if not lights:
        return

    #   Get image collection for the reduced files
    image_file_collection_reduced = ccdp.ImageFileCollection(out_path)

    #   Load combined darks and flats in dictionary for easy access
    dark_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class='dark',
    )
    combined_darks: dict[float, CCDData] = {
        ccd.header['exptime']: ccd for ccd in image_file_collection_reduced.ccds(
            imagetyp=dark_image_type,
            combined=True,
        )
    }
    flat_image_type = utilities.get_image_type(
        image_file_collection_reduced,
        image_type,
        image_class='flat',
    )
    combined_flats: dict[str, CCDData] = {
        ccd.header['filter']: ccd for ccd in image_file_collection_reduced.ccds(
            imagetyp=flat_image_type,
            combined=True,
        )
    }

    #   Get master bias
    combined_bias: CCDData | None = None
    if rm_bias:
        bias_image_type = utilities.get_image_type(
            image_file_collection_reduced,
            image_type,
            image_class='bias',
        )

        combined_bias = CCDData.read(
            image_file_collection_reduced.files_filtered(
                imagetyp=bias_image_type,
                combined=True,
                include_path=True,
            )[0]
        )

    #   Set science image path
    light_path = Path(out_path / 'light')

    dir_empty = checks.check_if_directory_is_empty(light_path)

    if not dir_empty:
        user_input, timed_out = base_utilities.get_input(
            f"{style.Bcolors.OKBLUE}   Reduced images from a previous run "
            f"found. Should these be used? [yes/no] {style.Bcolors.ENDC}"
        )
        if timed_out:
            user_input = 'n'

        if user_input in ['y', 'yes']:
            return

    checks.clear_directory(light_path)

    #   Get possible image types
    light_image_type = utilities.get_image_type(
        image_file_collection,
        image_type,
        image_class='light',
    )

    #   Initialize multiprocessing object
    executor = Executor(
        n_cores_multiprocessing,
        n_tasks=len(image_file_collection.files_filtered(imagetyp=light_image_type)),
        add_progress_bar=True,
    )

    #   Reduce science images and save to an extra directory
    for file_name in image_file_collection.files_filtered(
            include_path=True,
            imagetyp=light_image_type,
            # ccd_kwargs=dict(unit='adu'),
        ):
        executor.schedule(
            reduce_light_image,
            args=(
                file_name,
                combined_bias,
                combined_darks,
                combined_flats,
                out_path,
                light_path
            ),
            kwargs={
                'gain': gain,
                'read_noise': read_noise,
                'rm_bias': rm_bias,
                'exposure_time_tolerance': exposure_time_tolerance,
                'add_hot_bad_pixel_mask': add_hot_bad_pixel_mask,
                'rm_cosmic_rays': rm_cosmic_rays,
                'limiting_contrast_rm_cosmic_rays': limiting_contrast_rm_cosmic_rays,
                'sigma_clipping_value_rm_cosmic_rays': sigma_clipping_value_rm_cosmic_rays,
                'saturation_level': saturation_level,
                'mask_cosmics': mask_cosmics,
                'scale_image_with_exposure_time': scale_image_with_exposure_time,
                'verbose': verbose,
                'trim_x_start': trim_x_start,
                'trim_x_end': trim_x_end,
                'trim_y_start': trim_y_start,
                'trim_y_end': trim_y_end,
            }
        )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Light image reduction using multiprocessing'
            f' failed :({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()


def reduce_light_image(
        light_file_name: str, combined_bias: CCDData | None,
        combined_darks: dict[float, CCDData],
        combined_flats: dict[str, CCDData],
        out_path: Path, light_path: Path,
        gain: float | None = None, read_noise: float = 8.,
        rm_bias: bool = False, exposure_time_tolerance: float = 0.5,
        add_hot_bad_pixel_mask: bool = True, rm_cosmic_rays: bool = True,
        limiting_contrast_rm_cosmic_rays: float = 5.,
        sigma_clipping_value_rm_cosmic_rays: float = 4.5,
        saturation_level: float | None = 65535., mask_cosmics: bool = False,
        scale_image_with_exposure_time: bool = True, verbose: bool = False,
        trim_x_start: int = 0, trim_x_end: int = 0, trim_y_start: int = 0,
        trim_y_end: int = 0
    ) -> None:
    """
    Reduce an individual image

    Parameters
    ----------
    light_file_name
        The CCDData object that should be reduced.

    combined_bias
        Reduced and stacked Bias CCDData object

    combined_darks
        Combined darks in a dictionary with exposure times as keys and
        CCDData object as values.

    combined_flats
        Combined flats in a dictionary with exposure times as keys and
        CCDData object as values.

    out_path
        Path to the general output directory

    light_path
        Path where the reduced images should be saved

    gain
        The gain (e-/adu) of the camera chip. If set to `None` the gain
        will be extracted from the FITS header.
        Default is ``None``.

    read_noise
        The read noise (e-) of the camera chip.
        Default is ``8`` e-.

    rm_bias
        If True the master bias image will be subtracted from the flats
        Default is ``False``.

    exposure_time_tolerance
        Tolerance between science and dark exposure times in s.
        Default is ``0.5``s.

    add_hot_bad_pixel_mask
        If True add hot and bad pixel mask to the reduced science
        images.
        Default is ``True``.

    rm_cosmic_rays
        If True cosmic rays will be removed.
        Default is ``True``.

    limiting_contrast_rm_cosmic_rays
        Parameter for the cosmic ray removal: Minimum contrast between
        Laplacian image and the fine structure image.
        Default is ``5``.

    sigma_clipping_value_rm_cosmic_rays
        Parameter for the cosmic ray removal: Fractional detection limit
        for neighboring pixels.
        Default is ``4.5``.

    saturation_level
        Saturation limit of the camera chip.
        Default is ``65535``.

    mask_cosmics
        If True cosmics will ''only'' be masked. If False the
        cosmics will be removed from the input image and the mask will
        be added.
        Default is ``False``.

    scale_image_with_exposure_time
        If True the image will be scaled with the exposure time.
        Default is ``True``.

    verbose
        If True additional output will be printed to the command line.
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
    """
    #   Read light image
    light = CCDData.read(light_file_name, unit='adu')

    #   Trimming the image, for example to remove an overscan region
    image_shape = light.data.shape
    light = light[
                trim_y_start:image_shape[0]-trim_y_end,
                trim_x_start:image_shape[1]-trim_x_end
            ]

    #   Get base file name
    file_name = light_file_name.split('/')[-1]

    #   Set gain -> get it from Header if not provided
    if gain is None:
        try:
            gain = light.header['EGAIN']
        except KeyError:
            gain = 1.
            terminal_output.print_to_terminal(
                "WARNING: Gain could not de derived from the "
                "image header. Use 1.0 instead",
                style_name='WARNING',
                indent=2,
            )

    #   Calculated uncertainty
    light = ccdp.create_deviation(
        light,
        gain=gain * u.electron / u.adu,
        readnoise=read_noise * u.electron,
        disregard_nan=True,
    )

    #   Subtract bias
    if rm_bias:
        light = ccdp.subtract_bias(light, combined_bias)

    #   Find the correct dark exposure
    valid_dark_available, closest_dark_exposure_time = utilities.find_nearest_exposure_time_to_reference_image(
        light,
        list(combined_darks.keys()),
        time_tolerance=exposure_time_tolerance,
    )

    #   Exit if no dark with a similar exposure time have been found
    if not valid_dark_available and not rm_bias:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}Closest dark exposure time is "
            f"{closest_dark_exposure_time} for science image of exposure "
            f"time {light.header['exptime']}. {style.Bcolors.ENDC}"
        )

    #   Subtract dark
    reduced: CCDData = ccdp.subtract_dark(
        light,
        combined_darks[closest_dark_exposure_time],
        exposure_time='exptime',
        exposure_unit=u.second,
        scale=rm_bias,
    )

    #   Mask negative pixel
    mask = reduced.data < 0.
    reduced.mask = reduced.mask | mask

    #   Check if the "FILTER" keyword is set in Header
    #   TODO: Added ability to skip if filter not found. Add warning about which file will be skipped.
    #   TODO: Check if this works...
    if 'filter' not in reduced.header:
        terminal_output.print_to_terminal(
            f"WARNING: FILTER keyword not found in HEADER. \n Skip file: {file_name}.",
            style_name='WARNING',
            indent=2,
        )
        return

    #   Get master flat field
    flat_master = combined_flats[reduced.header['filter']]

    #   Divided science by the master flat
    reduced: CCDData = ccdp.flat_correct(reduced, flat_master)

    if add_hot_bad_pixel_mask:
        #   Get mask of bad and hot pixel
        mask_available, bad_hot_pixel_mask = utilities.get_pixel_mask(
            out_path,
            reduced.shape,
        )

        #   Add bad pixel mask: If there was already a mask, keep it
        if mask_available:
            if reduced.mask is not None:
                reduced.mask = reduced.mask | bad_hot_pixel_mask
            else:
                reduced.mask = bad_hot_pixel_mask

    #   Gain correct data
    reduced = ccdp.gain_correct(reduced, gain * u.electron / u.adu)

    #   Remove cosmic rays
    if rm_cosmic_rays:
        if verbose:
            terminal_output.print_to_terminal(
                f'Remove cosmic rays from image {file_name}'
            )

        #   Sanitize saturation level
        if saturation_level is None:
            terminal_output.print_to_terminal(
                f"Saturation level not specified. Assume 16bit == 65535",
                indent=1,
                style_name='WARNING',
            )
            saturation_level = 65535

        reduced_without_cosmics = ccdp.cosmicray_lacosmic(
            reduced,
            objlim=limiting_contrast_rm_cosmic_rays,
            readnoise=read_noise,
            sigclip=sigma_clipping_value_rm_cosmic_rays,
            satlevel=saturation_level,
            verbose=verbose,
        )

        if mask_cosmics:
            if add_hot_bad_pixel_mask:
                reduced.mask = reduced.mask | reduced_without_cosmics.mask

                #   Add a header keyword to indicate that the cosmics have been
                #   masked
                reduced.meta['cosmic_mas'] = True
        else:
            reduced = reduced_without_cosmics
            if not add_hot_bad_pixel_mask:
                reduced.mask = np.zeros(reduced.shape, dtype=bool)

            #   Add header keyword to indicate that cosmics have been removed
            reduced.meta['cosmics_rm'] = True

        if verbose:
            terminal_output.print_to_terminal('')

    #   Scale image with exposure time
    if scale_image_with_exposure_time:
        #   Get exposure time and all meta data
        exposure_time = reduced.header['exptime']
        reduced_meta = reduced.meta

        #   Scale image
        reduced = reduced.divide(exposure_time * u.second)

        #   Put metadata back on the image, because it is lost while
        #   dividing
        reduced.meta = reduced_meta
        reduced.meta['HIERARCH'] = 'Image scaled by exposure time:'
        reduced.meta['HIERARCH'] = 'Unit: e-/s/pixel'

        #   Set data units to electron / s
        reduced.unit = u.electron / u.s

    #   Write reduced science image to disk
    reduced.write(light_path / file_name, overwrite=True)


def stack_image(
        image_path: Path, output_dir: Path, image_type_list: list[str],
        stacking_method: str = 'average', dtype: str | np.dtype | None = None,
        new_target_name: str | None = None, debug: bool = False
    ) -> None:
    """
    Combine images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type_list
        Header keyword characterizing the image type for which the
        shifts shall be determined

    stacking_method
        Method used for combining the images.
        Possibilities: ``median`` or ``average`` or ``sum``
        Default is ``average`.

    dtype
        The dtype that should be used while combining the images.
        Default is ''None''. -> None is equivalent to float64

    new_target_name
        Name of the target. If not None, this target name will be written
        to the FITS header.
        Default is ``None``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.
    """
    terminal_output.print_to_terminal("Stack light images...", indent=2)

    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   New image collection for the images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Check if image_file_collection is not empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.Bcolors.ENDC}"
        )

    #   Determine filter
    image_type = utilities.get_image_type(
        image_file_collection,
        image_type_list,
    )
    filters: set[str] = set(h['filter'] for h in image_file_collection.headers(imagetyp=image_type))

    #   Combine images for the individual filters
    #   TODO: Add multiprocessing
    for filter_ in filters:
        #   Select images to combine
        images_to_combine = image_file_collection.files_filtered(
            imagetyp=image_type,
            filter=filter_,
            include_path=True,
        )

        #   Combine darks: Average images + sigma clipping to remove
        #                  outliers, set memory limit to 15GB
        combined_image = ccdp.combine(
            images_to_combine,
            method=stacking_method,
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            signma_clip_dev_func=mad_std,
            mem_limit=15e9,
            dtype=dtype,
        )

        #   Update Header keywords
        utilities.update_header_information(
            combined_image,
            len(images_to_combine),
            new_target_name,
        )

        #   Define name and write file to disk
        file_name = 'combined_filter_{}.fit'.format(
            filter_.replace("''", "p")
        )
        combined_image.write(out_path / file_name, overwrite=True)

    #   Remove individual reduced images
    if not debug:
        shutil.rmtree(file_path, ignore_errors=True)
