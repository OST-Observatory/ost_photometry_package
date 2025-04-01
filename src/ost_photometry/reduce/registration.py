############################################################################
#                               Libraries                                  #
############################################################################

import sys

import shutil

from pathlib import Path

import yaml

import numpy as np

import ccdproc as ccdp

import astroalign as aa

import math

from scipy.ndimage import shift as shift_scipy

from astropy.stats import mad_std
from astropy.nddata import CCDData, StdDevUncertainty

from skimage.registration import (
    phase_cross_correlation,
    optical_flow_tvl1,
    # optical_flow_ilk,
)
from skimage.transform import warp, SimilarityTransform

from . import utilities, plots
from .. import checks, style, terminal_output
from ..analyze.utilities import Executor
from .. import utilities as base_utilities
from ..terminal_output import print_to_terminal
from ..analyze import utilities as analysis_utilities

############################################################################
#                           Routines & definitions                         #
############################################################################

def align_images(
        image_path: str | Path, output_dir: str | Path,
        image_type_list: list[str], reference_image_id: int = 0,
        enlarged_only: bool = False, shift_method: str = 'skimage',
        n_cores_multiprocessing: int | None = None,
        rm_outliers: bool = True, filter_window: int = 25,
        threshold: int | float = 10., instrument: str | None = None,
        debug: bool = False, image_output_directory: str | None = None,
        transformation_output_directory: str = 'image_transformations',
        save_only_transformation: bool = False,
        terminal_alignment_comment: str | None = None,
        modify_file_name: bool = False,
        align_filter_wise: bool = False,
    ) -> None:
    """
    Calculate shift between images and trim those to the save field of
    view

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type_list
        Header keywords characterizing the image type for which the
        shifts shall be determined

    reference_image_id
        ID of the image that should be used as a reference
        Default is ``0``.

    enlarged_only
        It true the file selection will be restricted to images with a
        header keyword 'enlarged' that is set to True.
        Default is ``False``.

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

    rm_outliers
        If True outliers in the image shifts will be detected and removed.
        Default is ``True``.

    filter_window
        Width of the median filter window
        Default is ``25``.

    threshold
        Difference above the running median above an element is
        considered to be an outlier.
        Default is ``10.``.

    instrument
        The instrument used
        Default is ``None``.

    debug
        If `True` the intermediate files of the data reduction will not
        be removed.
        Default is ``False``.

    image_output_directory
        Directory to store the aligned images.
        Default is ``None``.

    transformation_output_directory
        Directory to store the image transformation matrices.
        Default is ``image_transformations``.

    save_only_transformation
        If ``True'', only the transformation matrix is saved, not the transformed image itself.
        Default is ``False``.

    terminal_alignment_comment
        Text string that is used to label the output.
        Default is ``None``.

    modify_file_name
        It ``True`` the trimmed image will be saved, using a modified file name.
        Default is ``False``.

    align_filter_wise
        If ``True'', only the images that belong to the same filter will be aligned.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   Set output paths
    if image_output_directory is not None:
        aligned_path = Path(out_path / image_output_directory)
        checks.clear_directory(aligned_path)
    else:
        aligned_path = out_path

    output_path_transformation = out_path / transformation_output_directory
    checks.clear_directory(output_path_transformation)

    #   New image collection for the images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Check if image_file_collection is not empty
    if not image_file_collection.files:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}No FITS files found in {file_path}. "
            f"=> EXIT {style.Bcolors.ENDC}"
        )

    #   Get image type
    image_type = utilities.get_image_type(
        image_file_collection,
        image_type_list,
    )

    #   Apply image_file_collection filter to the image collection
    #   -> This is necessary so that:
    #       1) the path to the image directory is
    #          added to the file names. This is required for
    #          `align_image_main`.
    #       2) Files like masks are excluded
    ifc_image_type_filtered = image_file_collection.filter(
        imagetyp=image_type,
    )

    #   Sort by time
    if 'jd' in ifc_image_type_filtered.summary.colnames:
        ifc_image_type_filtered.sort('jd')
    elif 'date-obs' in ifc_image_type_filtered.summary.colnames:
        ifc_image_type_filtered.sort('date-obs')

    if align_filter_wise:
        #   Determine filter
        filters = set(
            h['filter'] for h in ifc_image_type_filtered.headers()
        )

        for filter_ in filters:
            #   Restrict image collection to those images with the correct
            #   filter
            if enlarged_only:
                #   Select only enlarged images
                ifc_filtered = ifc_image_type_filtered.filter(
                    filter=filter_,
                    enlarged=enlarged_only,
                )
            else:
                ifc_filtered = ifc_image_type_filtered.filter(
                    filter=filter_,
                )

            #   Calculate image shifts and trim images accordingly
            align_image_main(
                ifc_filtered,
                aligned_path,
                output_path_transformation,
                shift_method=shift_method,
                n_cores_multiprocessing=n_cores_multiprocessing,
                reference_image_id=reference_image_id,
                terminal_alignment_comment=f'\tDisplacement for images in filter: {filter_}',
                rm_outliers=rm_outliers,
                filter_window=filter_window,
                instrument=instrument,
                threshold=threshold,
                verbose=debug,
                save_only_transformation=save_only_transformation,
            )
    else:
        if enlarged_only:
            #   Select only enlarged images
            ifc_filtered = ifc_image_type_filtered.filter(
                enlarged=enlarged_only,
            )
        else:
            ifc_filtered = ifc_image_type_filtered

        #   Calculate image shifts and trim images accordingly
        align_image_main(
            ifc_filtered,
            aligned_path,
            output_path_transformation,
            shift_method=shift_method,
            n_cores_multiprocessing=n_cores_multiprocessing,
            reference_image_id=reference_image_id,
            terminal_alignment_comment=terminal_alignment_comment,
            rm_enlarged_keyword=enlarged_only,
            modify_file_name=modify_file_name,
            rm_outliers=rm_outliers,
            filter_window=filter_window,
            instrument=instrument,
            threshold=threshold,
            verbose=debug,
            save_only_transformation=save_only_transformation,
        )

    #   Remove reduced files if they exist, but only if they are no longer
    #   needed. DO NOT remove files if only enlarged images are aligned, as
    #   this is currently done directly in the output directory, so this would
    #   remove all results. DO NOT remove files when running in debug mode.
    #   Do not remove images if only transformations are saved, so that there
    #   are still reduced images for checking. This will be simplified in a
    #   future release.
    if not debug and not save_only_transformation and not enlarged_only:
        shutil.rmtree(file_path, ignore_errors=True)


def align_image_main(
        image_file_collection: ccdp.ImageFileCollection, output_path: Path,
        output_path_transformation: Path,
        shift_method: str = 'skimage',
        n_cores_multiprocessing: int | None = None,
        reference_image_id: int = 0,
        terminal_alignment_comment: str | None = None,
        rm_enlarged_keyword: bool = False, modify_file_name: bool = False,
        rm_outliers: bool = True, filter_window: int = 25,
        threshold: int | float = 10., instrument: str | None = None,
        verbose: bool = False, save_only_transformation: bool = False,
    ) -> None:
    """
    Core steps of the image shift calculations and trimming to a
    common filed of view

    Parameters
    ----------
    image_file_collection
        Image file collection with all images

    output_path
        Path to the output directory

    output_path_transformation
        Path to save the image transformation matrices

    shift_method
        Method to use for image alignment.
        Possibilities: 'aa'      = astroalign module only accounting for
                                   xy shifts
                       'aa_true' = astroalign module with corresponding
                                   transformation
                       'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation implemented by
                                   skimage
                       'flow'    = image registration using optical flow
                                   implementation by skimage
        Default is ``skimage``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    reference_image_id
        ID of the image that should be used as a reference
        Default is ``0``.

    terminal_alignment_comment
        Text string that is used to label the output.
        Default is ``None``.

    rm_enlarged_keyword
        It True the header keyword 'enlarged' will be removed.
        Default is ``False``.

    modify_file_name
        It True the trimmed image will be saved, using a modified file
        name.
        Default is ``False``.

    rm_outliers
        If True outliers in the image shifts will be detected and removed.
        Default is ``True``.

    filter_window
        Width of the median filter window
        Default is ``25``.

    threshold
        Difference above the running median above an element is
        considered to be an outlier.
        Default is ``10.``.

    instrument
        The instrument used
        Default is ``None``.

    verbose
        If True additional output will be printed to the console
        Default is ``False``.

    save_only_transformation
        If ``True'', only the transformation matrix is saved, not the transformed image itself.
        Default is ``False``.
    """
    if terminal_alignment_comment is None:
        terminal_alignment_comment = '\tImage displacement:'
    elif not isinstance(terminal_alignment_comment, str):
        terminal_output.print_to_terminal(
            "The 'terminal_alignment_comment' is not a string as expected. "
            "Set it to the default.",
            indent=2,
            style_name='WARNING',
        )
        terminal_alignment_comment = '\tImage displacement:'

    #   Calculate image shifts
    if shift_method in ['own', 'skimage', 'aa']:
        image_shifts, image_flips = calculate_xy_image_shifts(
            image_file_collection,
            reference_image_id,
            terminal_alignment_comment,
            correlation_method=shift_method,
            n_cores_multiprocessing=n_cores_multiprocessing,
        )

        #   Find IDs of potential outlier
        if rm_outliers:
            outlier_ids = utilities.detect_outlier(
                image_shifts,
                filter_window=filter_window,
                threshold=threshold,
            )
            if outlier_ids.size:
                terminal_output.print_to_terminal(
                    "The images with the following IDs will be removed "
                    f"because of not reliable shifts:\n {outlier_ids.ravel()}.",
                    indent=2,
                    style_name='WARNING',
                )

                #   Set outlier image shifts to NANs
                image_shifts[:, outlier_ids] = np.nan

        terminal_output.print_to_terminal(
            'Apply image shifts and crop images accordingly',
            indent=2
        )

        #   Initialize multiprocessing object
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=np.invert(np.isnan(image_shifts[1, :])).sum(),
            add_progress_bar=True,
        )

        #   Trim all images
        for current_image_id, current_image_name in enumerate(image_file_collection.files):
            #   Check for outliers and those images where the shift determination failed
            if not np.isnan(image_shifts[1, current_image_id]):
                executor.schedule(
                    apply_xy_image_shift,
                    args=(
                        current_image_name,
                        image_shifts,
                        image_flips,
                        current_image_id,
                        output_path,
                    ),
                    kwargs={
                        'shift_method': shift_method,
                        'modify_file_name': modify_file_name,
                        'rm_enlarged_keyword': rm_enlarged_keyword,
                        'instrument': instrument,
                        'verbose': verbose,
                    }
                )

        #   Exit if exceptions occurred
        if executor.err is not None:
            raise RuntimeError(
                f'\n{style.Bcolors.FAIL}Image offset could not be applied.'
                f'It was not possible to recover from this error.'
                f':({style.Bcolors.ENDC}'
            )

        #   Close multiprocessing pool and wait until it finishes
        executor.wait()

    elif shift_method == 'flow':
        reference_file_name = image_file_collection.files[reference_image_id]

        #   Initialize multiprocessing object
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=len(image_file_collection.files),
            add_progress_bar=True,
        )

        #   Trim all images
        for current_image_id, current_image_name in enumerate(image_file_collection.files):
            executor.schedule(
                apply_optical_flow,
                args=(
                    current_image_name,
                    reference_file_name,
                    output_path,
                ),
                kwargs={
                    'modify_file_name': modify_file_name,
                    'rm_enlarged_keyword': rm_enlarged_keyword,
                    'instrument': instrument,
                }
            )

        #   Exit if exceptions occurred
        if executor.err is not None:
            raise RuntimeError(
                f'\n{style.Bcolors.FAIL}Image offset could not be determined or applied.'
                f'It was not possible to recover from this error.'
                f':({style.Bcolors.ENDC}'
            )

        #   Close multiprocessing pool and wait until it finishes
        executor.wait()

    elif shift_method == 'aa_true':
        reference_file_name = image_file_collection.files[reference_image_id]

        #   Initialize multiprocessing object
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=len(image_file_collection.files),
            add_progress_bar=True,
        )

        #   Trim all images
        for current_image_id, current_image_name in enumerate(image_file_collection.files):
            executor.schedule(
                apply_astro_align,
                args=(
                    current_image_name,
                    reference_file_name,
                    output_path,
                    output_path_transformation,
                ),
                kwargs={
                    'modify_file_name': modify_file_name,
                    'rm_enlarged_keyword': rm_enlarged_keyword,
                    'instrument': instrument,
                    'save_only_transformation': save_only_transformation,
                }
            )

        #   Exit if exceptions occurred
        if executor.err is not None:
            raise RuntimeError(
                f'\n{style.Bcolors.FAIL}Image offset could not be determined or applied.'
                f'It was not possible to recover from this error.'
                f':({style.Bcolors.ENDC}'
            )

        #   Close multiprocessing pool and wait until it finishes
        executor.wait()
    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Method {shift_method} not known '
            f'-> EXIT {style.Bcolors.ENDC}'
        )


#   TODO: Combine with image_shift_astroalign_method
#   TODO: Check if this can be removed
# def shift_stack_astroalign(
#         path: str | Path, output_dir: Path, image_type: list[str]) -> None:
#     """
#     Calculate shift between stacked images and trim those
#     to the save field of view

#     Parameters
#     ----------
#     path
#         The path to the images

#     output_dir
#         Path to the directory where the master files should be saved to

#     image_type
#         Header keyword characterizing the image type for which the
#         shifts shall be determined
#     """
#     #   New image collection for the images
#     image_file_collection = ccdp.ImageFileCollection(path)
#     img_type = utilities.get_image_type(image_file_collection, image_type)
#     ifc_filtered = image_file_collection.filter(
#         combined=True,
#         imagetyp=img_type,
#     )

#     for current_image_id, (current_image_ccd, file_name) in enumerate(ifc_filtered.ccds(return_fname=True)):
#         reference_image_ccd: ccdp.CCDData | None = None
#         if current_image_id == 0:
#             reference_image_ccd = current_image_ccd
#             image_out = reference_image_ccd
#         else:
#             #   Byte order of the system
#             sbo = sys.byteorder

#             #   Adjust endianness
#             current_image_ccd = utilities.adjust_edian_compatibility(current_image_ccd)
#             reference_image_ccd = utilities.adjust_edian_compatibility(reference_image_ccd)

#             #   Determine transformation between the images
#             transformation_parameter, (_, _) = aa.find_transform(
#                 current_image_ccd,
#                 reference_image_ccd,
#                 max_control_points=100,
#                 detection_sigma=3,
#             )

#             #   Transform image data
#             image_data, footprint = aa.apply_transform(
#                 transformation_parameter,
#                 current_image_ccd,
#                 reference_image_ccd,
#                 propagate_mask=True,
#             )

#             #   Transform uncertainty array
#             image_uncertainty, _ = aa.apply_transform(
#                 transformation_parameter,
#                 current_image_ccd.uncertainty.array,
#                 reference_image_ccd.uncertainty.array,
#             )

#             #   Build new CCDData object
#             image_out = CCDData(
#                 image_data,
#                 mask=footprint,
#                 meta=current_image_ccd.meta,
#                 unit=current_image_ccd.unit,
#                 wcs=current_image_ccd.wcs,
#                 uncertainty=StdDevUncertainty(image_uncertainty),
#             )

#         #   Get filter
#         filter_ = image_out.meta['filter']

#         image_out.meta['trimmed'] = True
#         image_out.meta.remove('combined')

#         #   Define name and write trimmed image to disk
#         file_name = 'combined_trimmed_filter_{}.fit'.format(
#             filter_.replace("''", "p")
#         )
#         image_out.write(output_dir / file_name, overwrite=True)


def make_big_images(
        image_path: str | Path, output_dir: str | Path,
        image_type_list: list[str], combined_only: bool = True,
        set_efault_file_name: bool = False,
    ) -> None:
    """
    Image size unification:
        Find the largest image and use this for all other images

    Parameters
    ----------
    image_path
        Path to the images

    output_dir
        Path to the directory where the master files should be saved to

    image_type_list
        Header keyword characterizing the image type for which the
        shifts shall be determined

    combined_only
        It true the file selection will be restricted to images with a
        header keyword 'combined' that is set to True.
        Default is ``True``.

    set_efault_file_name
        If ``True'', a new filename is created that marks the image as
        enlarged and contains the filter used.
        Default is ``False``.
    """
    #   Sanitize the provided paths
    file_path = checks.check_pathlib_path(image_path)
    out_path = checks.check_pathlib_path(output_dir)

    #   New image collection for the images
    image_file_collection = ccdp.ImageFileCollection(file_path)

    #   Image list
    image_type = utilities.get_image_type(
        image_file_collection,
        image_type_list,
    )
    img_dict: dict[str, CCDData] = {
        file_name: ccd for ccd, file_name in image_file_collection.ccds(
            imagetyp=image_type,
            return_fname=True,
            combined=combined_only,
        )
    }

    #   Image list
    image_list: list[CCDData] = list(img_dict.values())

    #   File name list
    file_names: list[str] = list(img_dict.keys())

    #   Number of images
    n_images = len(file_names)

    #   Get image dimensions
    image_shape_array_x = np.zeros(n_images, dtype='int')
    image_shape_array_y = np.zeros(n_images, dtype='int')
    for i, current_image in enumerate(image_list):
        #   Original image dimension
        image_shape_array_x[i] = current_image.shape[1]
        image_shape_array_y[i] = current_image.shape[0]

    #   Maximum size
    image_shape_x_max = np.max(image_shape_array_x)
    image_shape_y_max = np.max(image_shape_array_y)

    for i, current_image in enumerate(image_list):
        #   Make big image ans mask
        big_image = np.zeros((image_shape_y_max, image_shape_x_max))
        big_mask = np.ones((image_shape_y_max, image_shape_x_max), dtype=bool)
        big_uncertainty = np.zeros((image_shape_y_max, image_shape_x_max))

        #   Fill image and mask
        big_image[0:image_shape_array_y[i], 0:image_shape_array_x[i]] = current_image.data
        big_mask[0:image_shape_array_y[i], 0:image_shape_array_x[i]] = current_image.mask
        big_uncertainty[0:image_shape_array_y[i], 0:image_shape_array_x[i]] = current_image.uncertainty.array

        #   Replace
        current_image.data = big_image
        current_image.mask = big_mask
        current_image.uncertainty.array = big_uncertainty

        #   Add Header keyword to mark the file as a Master
        current_image.meta['enlarged'] = True
        current_image.meta.remove('combined')

        #   Get filter
        filter_ = current_image.meta['filter']

        #   Define name and write trimmed image to disk
        if set_efault_file_name:
            file_name = 'combined_enlarged_filter_{}.fit'.format(
                filter_.replace("''", "p")
            )
        else:
            file_name = file_names[i]
        current_image.write(out_path / file_name, overwrite=True)


def apply_xy_image_shift(
        current_image_name: str, image_shifts: np.ndarray,
        image_flips: np.ndarray, image_id: int, output_path: Path,
        shift_method: str = 'skimage', modify_file_name: bool = False,
        rm_enlarged_keyword: bool = False, instrument: str | None = None,
        verbose: bool = False) -> None:
    """
    Apply shift to an individual image

    Parameters
    ----------
    current_image_name
        Path to the current image

    image_shifts
        Shifts of the images in X and Y direction

    image_flips
        Flip necessary to account for pier flips

    image_id
        ID of the image

    output_path
        Path to the output directory

    shift_method
        Method to use for image alignment.
        Possibilities: 'aa'      = astroalign module only accounting for
                                   xy shifts
                       'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation implemented by
                                   skimage
        Default is ``skimage``.

    modify_file_name
        It true the trimmed image will be saved, using a modified file
        name.
        Default is ``False``.

    rm_enlarged_keyword
        It true the header keyword 'enlarged' will be removed.
        Default is ``False``.

    instrument
        The instrument used
        Default is ``None``.

    verbose
        If True additional output will be printed to the console
        Default is ``False``.
    """
    #   Get image data
    current_image_ccd = CCDData.read(current_image_name)

    #   Trim images
    if shift_method in ['own', 'skimage', 'aa']:
        #   Flip image if pier side changed
        if image_flips[image_id]:
            current_image_ccd = ccdp.transform_image(
                current_image_ccd,
                np.flip,
                axis=(0, 1),
            )

        output_image = trim_image(
            current_image_ccd,
            image_id,
            image_shifts,
            correlation_method=shift_method,
            verbose=verbose,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nThe provided method to determine the "
            f"shifts is not known. Got {shift_method}. Allowed: own, "
            f"skimage, aa, flow, aa_true {style.Bcolors.ENDC}"
        )

    #   Reset the device as it may have been updated
    if instrument is not None and instrument != '':
        output_image.meta['INSTRUME'] = instrument

    #   Add Header keyword to mark the file as trimmed
    output_image.meta['trimmed'] = True
    if rm_enlarged_keyword:
        output_image.meta.remove('enlarged')

    if modify_file_name:
        #   Get filter
        filter_ = output_image.meta['filter']

        #   Define name and write trimmed image to disk
        image_name = 'combined_trimmed_filter_{}.fit'.format(
            filter_.replace("''", "p")
        )

    #   Write trimmed image to disk
    file_name = current_image_name.split('/')[-1]
    output_image.write(output_path / file_name, overwrite=True)


def apply_optical_flow(
        current_image_name: str, reference_image_name: str,
        output_path: Path, modify_file_name: bool = False,
        rm_enlarged_keyword: bool = False, instrument: str | None = None,
    ) -> None:
    """
    Apply shift to an individual image

    Parameters
    ----------
    current_image_name
        Path to the current image

    reference_image_name
        Path to the reference image

    output_path
        Path to the output directory

    modify_file_name
        It true the trimmed image will be saved, using a modified file
        name.
        Default is ``False``.

    rm_enlarged_keyword
        It true the header keyword 'enlarged' will be removed.
        Default is ``False``.

    instrument
        The instrument used
        Default is ``None``.
    """
    #   Get image data
    current_image_ccd = CCDData.read(current_image_name)

    #   Trim images
    reference_image_ccd = CCDData.read(reference_image_name)
    try:
        output_image = optical_flow_align(
            reference_image_ccd,
            current_image_ccd,
        )
    except ValueError as e:
        terminal_output.print_to_terminal(
            f"WARNING: Failed to calculate image offset for image"
            f" {current_image_name} with ERROR code: \n\n {e} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )
        return

    #   Reset the device as it may have been updated
    if instrument is not None and instrument != '':
        output_image.meta['INSTRUME'] = instrument

    #   Add Header keyword to mark the file as trimmed
    output_image.meta['trimmed'] = True
    if rm_enlarged_keyword:
        output_image.meta.remove('enlarged')

    #   Get file name
    file_name = current_image_name.split('/')[-1]

    if modify_file_name:
        #   Get filter
        filter_ = output_image.meta['filter']

        #   Define name and write trimmed image to disk
        file_name = 'combined_trimmed_filter_{}.fit'.format(
            filter_.replace("''", "p")
        )

    #   Write trimmed image to disk
    output_image.write(output_path / file_name, overwrite=True)


def apply_astro_align(
        current_image_name: str, reference_image_name: str,
        output_path: Path, output_path_transformation: Path,
        modify_file_name: bool = False, rm_enlarged_keyword: bool = False,
        instrument: str | None = None, save_only_transformation: bool = False,
    ) -> None:
    """
    Apply shift to an individual image

    Parameters
    ----------
    current_image_name
        Path to the current image

    reference_image_name
        Path to the reference image

    output_path
        Path to the output directory

    output_path_transformation
        Path to save the image transformation matrices

    modify_file_name
        It true the trimmed image will be saved, using a modified file
        name.
        Default is ``False``.

    rm_enlarged_keyword
        It true the header keyword 'enlarged' will be removed.
        Default is ``False``.

    instrument
        The instrument used
        Default is ``None``.

    save_only_transformation
        If ``True'', only the transformation matrix is saved, not the transformed image itself.
        Default is ``False``.
    """
    #   Get image data
    current_image_ccd = CCDData.read(current_image_name)
    reference_image_ccd = CCDData.read(reference_image_name)

    #   Trim images
    try:
        output_image, similarity_transforma = astro_align(
            reference_image_ccd,
            current_image_ccd,
        )
    except (aa.MaxIterError, TypeError, ValueError) as e:
        terminal_output.print_to_terminal(
            f"WARNING: Failed to calculate image offset for image"
            f" {current_image_name} with ERROR code: \n\n {e} \n Skip file.",
            style_name='WARNING',
            indent=2,
        )
        return

    #   Get file name
    file_name = current_image_name.split('/')[-1]

    if modify_file_name:
        #   Get filter
        filter_ = output_image.meta['filter']

        #   Define name and write trimmed image to disk
        file_name = 'combined_trimmed_filter_{}.fit'.format(
            filter_.replace("''", "p")
        )

    if not save_only_transformation:
        #   Reset the instrument as it may have been updated
        if instrument is not None and instrument != '':
            output_image.meta['INSTRUME'] = instrument

        #   Add Header keyword to mark the file as trimmed
        output_image.meta['trimmed'] = True
        if rm_enlarged_keyword:
            output_image.meta.remove('enlarged')

        #   Write trimmed image to disk
        output_image.write(output_path / file_name, overwrite=True)

    #   Save similarity transformation matrix
    base_name = base_utilities.get_basename(file_name)
    with open(output_path_transformation / f'{base_name}.yaml', 'w') as file:
        yaml.dump(similarity_transforma.params.tolist(), file)


def own_image_cross_correlation(
        image_1: np.ndarray, image_2: np.ndarray, maximum_shift_x: int,
        maximum_shift_y: int, debug: bool) -> tuple[int, int]:
    """
    Cross correlation:

    Adapted from add_images written by Nadine Giese for use within the
    astrophysics lab course at Potsdam University.
    The source code may be modified, reused, and distributed as long as
    it retains a reference to the original author(s).

    Idea and further information:
    http://en.wikipedia.org/wiki/Phase_correlation

    Parameters
    ----------
    image_1
        Data of first image

    image_2
        Data of second image

    maximum_shift_x
        Maximal allowed shift between the images in Pixel - X axis

    maximum_shift_y
        Maximal allowed shift between the images in Pixel - Y axis

    debug
        If True additional plots will be created

    Returns
    -------
    index_1
        Shift of image_1 with respect to image_2 in the Y direction

    index_2
        Shift of image_1 with respect to image_2 in the X direction
    """

    image_dimension_x = image_1.shape[1]
    image_dimension_y = image_1.shape[0]

    #   Fast fourier transformation
    image_1_fft = np.fft.fft2(image_1)
    image_2_fft = np.fft.fft2(image_2)
    image_2_fft_cc = np.conj(image_2_fft)
    fft_cc = image_1_fft * image_2_fft_cc
    fft_cc = fft_cc / np.absolute(fft_cc)
    # cc = np.fft.ifft2(fft_cc)
    cc_matrix = np.fft.fft2(fft_cc)
    cc_matrix[0, 0] = 0.

    #   Limit to allowed shift range
    for i in range(maximum_shift_x, image_dimension_x - maximum_shift_x):
        for j in range(0, image_dimension_y):
            cc_matrix[j, i] = 0
    for i in range(0, image_dimension_x):
        for j in range(maximum_shift_y, image_dimension_y - maximum_shift_y):
            cc_matrix[j, i] = 0

    #   Debug plot showing the cc matrix
    if debug:
        plots.cross_correlation_matrix(image_2, cc_matrix)

    #   Find the maximum in cc to identify the shift
    index_1, index_2 = np.unravel_index(cc_matrix.argmax(), cc_matrix.shape)

    # if index_2 > image_dimension_x/2.:
    # index_2 = (index_2-1)-image_dimension_x+1
    # else:
    # index_2 = index_2 - 1
    # if index_1 > image_dimension_y/2.:
    # index_1 = (index_1-1)-image_dimension_y+1
    # else:
    # index_1 = index_1 - 1
    if index_2 > image_dimension_x / 2.:
        index_2 = index_2 - image_dimension_x - 2
    else:
        index_2 = index_2 + 2
    if index_1 > image_dimension_y / 2.:
        index_1 = index_1 - image_dimension_y - 2
    else:
        index_1 = index_1 + 2

    return -index_1, -index_2


def calculate_min_max_image_shifts(
        shifts: np.ndarray, python_format: bool = False
        ) -> tuple[float, float, float, float]:
    """
    Calculate shifts

    Parameters
    ----------
    shifts
        2D numpy array with the image shifts in X and Y direction

    python_format
        If True the python style of image ordering is used. If False the
        natural/fortran style of image ordering is use.
        Default is ``False``.

    Returns
    -------
    minimum_shift_x
        Minimum shift in X direction

    maximum_shift_x
        Maximum shift in X direction

    minimum_shift_y
        Minimum shift in Y direction

    maximum_shift_y
        Maximum shift in Y direction
    """
    #   Distinguish between python format and natural format
    if python_format:
        id_x = 1
        id_y = 0
    else:
        id_x = 0
        id_y = 1

    #   Maximum and minimum shifts
    minimum_shift_x = np.nanmin(shifts[id_x, :])
    maximum_shift_x = np.nanmax(shifts[id_x, :])

    minimum_shift_y = np.nanmin(shifts[id_y, :])
    maximum_shift_y = np.nanmax(shifts[id_y, :])

    return minimum_shift_x, maximum_shift_x, minimum_shift_y, maximum_shift_y


def calculate_xy_image_shifts_core(
        current_file_name: str, reference_file_name: str,
        image_id: int, correlation_method: str = 'skimage'
    ) -> tuple[int, tuple[float | int, float | int], bool]:
    """
    Calculate image shifts using different methods

    Parameters
    ----------
    current_file_name
        File name of the current image

    reference_file_name
        File name of the reference image

    image_id
        ID of the image

    correlation_method
        Method to use for image alignment.
        Possibilities: 'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation with skimage'
                       'aa'      = astroalign module
        Default is 'skimage'.

    Returns
    -------
    image_id
        ID of the image

    image_shift
        Shifts of the image in X and Y direction

    flip_necessary
        If `True` the image needs to be flipped
    """
    #   Read images
    image_ccd = CCDData.read(current_file_name)
    reference_ccd = CCDData.read(reference_file_name)

    #   Get reference image, reference mask, and corresponding file name
    reference_data = reference_ccd.data
    reference_mask = np.invert(reference_ccd.mask)

    #   Image pier side
    reference_pier = reference_ccd.meta.get('PIERSIDE', 'EAST')
    current_pier = image_ccd.meta.get('PIERSIDE', 'EAST')

    #   Flip if pier side changed
    if current_pier != reference_pier:
        image_ccd = ccdp.transform_image(
            image_ccd,
            np.flip,
            axis=(0, 1),
        )
        flip_necessary = True
    else:
        flip_necessary = False

    #   Image and mask to compare with
    current_data = image_ccd.data
    current_mask = np.invert(image_ccd.mask)

    #   Calculate shifts
    if correlation_method == 'skimage':
        try:
            image_shift = phase_cross_correlation(
                reference_data,
                current_data,
                reference_mask=reference_mask,
                moving_mask=current_mask,
            )
            image_shift = image_shift[0]
        except ValueError as e:
            image_shift = (np.nan, np.nan)
            terminal_output.print_to_terminal(
                f"Image offset determination failed for image: {current_file_name}",
                indent=2,
                style_name='WARNING',
            )
            terminal_output.print_to_terminal(
                f'The exception is: {e}',
                indent=2,
                style_name='WARNING',
            )

    elif correlation_method == 'own':
        try:
            image_shift = own_image_cross_correlation(
                reference_data,
                current_data,
                1000,
                1000,
                False,
            )
        except (IndexError, RuntimeError) as e:
            image_shift = (np.nan, np.nan)
            terminal_output.print_to_terminal(
                f"Image offset determination failed for image: {current_file_name}",
                indent=2,
                style_name='WARNING',
            )
            terminal_output.print_to_terminal(
                f'The exception is: {e}',
                indent=2,
                style_name='WARNING',
            )

    elif correlation_method == 'aa':
        if flip_necessary:
            print_to_terminal(
                'The current "aa" correlation method, combined with the '
                'meridian flips that occurred in this observation, usually '
                'gives rather poor results. It is better to use the "aa_true" '
                'correlation method in this case.',
                indent=2,
                style_name='WARNING',
            )


        #   Adjust endianness
        image_ccd = utilities.adjust_edian_compatibility(image_ccd)
        reference_ccd = utilities.adjust_edian_compatibility(reference_ccd)

        #   Determine transformation between the images
        try:
            transformation_coefficients, (_, _) = aa.find_transform(
                image_ccd,
                reference_ccd,
                detection_sigma=3,
            )

            image_shift = (
                transformation_coefficients.translation[1],
                transformation_coefficients.translation[0]
            )
        except (aa.MaxIterError, IndexError, TypeError, ValueError) as e:
            image_shift = (np.nan, np.nan)
            terminal_output.print_to_terminal(
                f"Image offset determination failed for image: {current_file_name}",
                indent=2,
                style_name='WARNING',
            )
            terminal_output.print_to_terminal(
                f'The exception is: {e}',
                indent=2,
                style_name='WARNING',
            )
    else:
        #   This should not happen...
        raise RuntimeError(
            f'{style.Bcolors.FAIL}Image correlation method '
            f'{correlation_method} not known\n {style.Bcolors.ENDC}'
        )
    file_name = current_file_name.split('/')[-1]
    terminal_output.print_to_terminal(
        f'\t{image_id}\t{image_shift[1]:+.1f}\t{image_shift[0]:+.1f}'
        f'\t{file_name}',
        indent=0,
    )

    return image_id, image_shift, flip_necessary


def calculate_xy_image_shifts(
        image_file_collection: ccdp.ImageFileCollection,
        id_reference_image: int, comment: str,
        correlation_method: str = 'skimage',
        n_cores_multiprocessing: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate image shifts

    Parameters
    ----------
    image_file_collection
        Image file collection

    id_reference_image
        Number of the reference image

    comment
        Information regarding for which images the shifts will be
        calculated

    correlation_method
        Method to use for image alignment.
        Possibilities: 'own'     = own correlation routine based on
                                   phase correlation, applying fft to
                                   the images
                       'skimage' = phase correlation with skimage'
                       'aa'      = astroalign module
        Default is 'skimage'.

    n_cores_multiprocessing
        Number of cores to use during multiprocessing.
        Default is ``None``.

    Returns
    -------
    image_shift
        Shifts of the images in X and Y direction

    flip_necessary
        Flip necessary to account for pier flips
    """
    #   Number of images
    n_files = len(image_file_collection.files)

    #   Get reference image file name
    reference_file_name = image_file_collection.files[id_reference_image]

    #   Prepare an array for the shifts
    image_shift = np.zeros((2, n_files))
    flip_necessary = np.zeros(n_files, dtype=bool)

    terminal_output.print_to_terminal(comment, indent=0)
    terminal_output.print_to_terminal('\tImage\tx\ty\tFilename', indent=0)
    terminal_output.print_to_terminal(
        '\t----------------------------------------',
        indent=0,
    )
    terminal_output.print_to_terminal(
        f'\t{id_reference_image}\t{0:+.1f}\t{0:+.1f}\t'
        f'{reference_file_name.split("/")[-1]}',
        indent=0,
    )

    #   Initialize multiprocessing object
    executor = analysis_utilities.Executor(n_cores_multiprocessing)

    #   Calculate image shifts
    for i, current_file_name in enumerate(image_file_collection.files):
        if i != id_reference_image:
            executor.schedule(
                calculate_xy_image_shifts_core,
                args=(
                    current_file_name,
                    reference_file_name,
                    i,
                ),
                kwargs={
                'correlation_method':correlation_method,
                }
            )

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.Bcolors.FAIL}Image offset could not be determined. '
            f'It was not possible to recover from this error.'
            f':({style.Bcolors.ENDC}'
        )

    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Extract results
    res = executor.res

    #   Sort multiprocessing results
    for ref_id, shift_i, flip_i in res:
        image_shift[:,ref_id] = shift_i
        flip_necessary[ref_id] = flip_i

    terminal_output.print_to_terminal('')

    return image_shift, flip_necessary


def astro_align(
        reference_ccd: CCDData, current_ccd:CCDData
    ) -> tuple[CCDData, SimilarityTransform]:
    """
    Calculate image shifts using the astroalign method

    Parameters
    ----------
    reference_ccd_object
        Reference image

    current_ccd_object
        Current image

    Returns
    -------

        Aligned image
    """
    #   Adjust endianness
    current_ccd = utilities.adjust_edian_compatibility(current_ccd)
    reference_ccd = utilities.adjust_edian_compatibility(reference_ccd)

    #   Determine transformation between the images
    transformation_coefficients, (_, _) = aa.find_transform(
        current_ccd,
        reference_ccd,
        detection_sigma=3,
    )

    #   Transform image data
    #   TODO: Check whether 'footprint' should be saved in an extra mask,
    #         so that it can be used as 'coverage_mask' in 2D background
    #         extraction, for example.
    image_data, footprint_mask = aa.apply_transform(
        transformation_coefficients,
        current_ccd,
        reference_ccd,
        propagate_mask=True,
        fill_value=0.,
    )

    #   Transform uncertainty array
    image_uncertainty, _ = aa.apply_transform(
        transformation_coefficients,
        current_ccd.uncertainty.array,
        reference_ccd.uncertainty.array,
        fill_value=0.,
    )

    #   Build new CCDData object
    new_ccd = CCDData(
        image_data,
        mask=footprint_mask,
        meta=current_ccd.meta,
        unit=current_ccd.unit,
        uncertainty=StdDevUncertainty(image_uncertainty),
    )
    return new_ccd, transformation_coefficients


def optical_flow_align(
        reference_ccd_object: CCDData, current_ccd_object: CCDData
    ) -> CCDData:
    """
    Calculate image shifts using the optical flow method

    Parameters
    ----------
    reference_ccd_object
        Reference image

    current_ccd_object
        Current image

    Returns
    -------

        Aligned image
    """
    #   Prepare data, mask, and uncertainty arrays
    current_data = current_ccd_object.data
    current_mask = current_ccd_object.mask
    current_uncertainty = current_ccd_object.uncertainty.array

    #   Compute optical flow
    flow_v, flow_u = optical_flow_tvl1(reference_ccd_object.data, current_data)

    #   Prepare grid for flow map
    image_dimension_x, image_dimension_y = reference_ccd_object.data.shape
    row_coordinates, column_coordinates = np.meshgrid(
        np.arange(image_dimension_x),
        np.arange(image_dimension_y),
        indexing='ij',
    )

    #   Registrate image data, mask, and uncertainty
    image_out_data = warp(
        current_data,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )
    image_out_mask = warp(
        current_mask,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )
    image_out_uncertainty = warp(
        current_uncertainty,
        np.array([row_coordinates + flow_v, column_coordinates + flow_u]),
        mode='edge',
    )

    #   Build new CCDData object
    return CCDData(
        image_out_data,
        mask=image_out_mask,
        meta=current_ccd_object.meta,
        unit=current_ccd_object.unit,
        uncertainty=StdDevUncertainty(image_out_uncertainty),
    )


def calculate_index_from_shifts(
        shifts: np.ndarray, id_current_image: int
        ) -> tuple[float, float, float, float]:
    """
    Calculate image index positions from image shifts

    Parameters
    ----------
    shifts
        The shifts of all images in X and Y direction

    id_current_image
        ID of the current image

    Returns
    -------
    x_start, x_end, y_start, y_end
        Start/End pixel index in X and Y direction.
    """
    #   Calculate maximum and minimum shifts
    min_shift_x, max_shift_x, min_shift_y, max_shift_y = (
        calculate_min_max_image_shifts(shifts, python_format=True)
    )

    #   Calculate indexes from image shifts
    if min_shift_x >= 0 and max_shift_x >= 0:
        x_start = max_shift_x - shifts[1, id_current_image]
        x_end = shifts[1, id_current_image] * -1
    elif min_shift_x < 0 and max_shift_x < 0:
        x_start = shifts[1, id_current_image] * -1
        x_end = max_shift_x - shifts[1, id_current_image]
    else:
        x_start = max_shift_x - shifts[1, id_current_image]
        x_end = min_shift_x - shifts[1, id_current_image]

    if min_shift_y >= 0 and max_shift_y >= 0:
        y_start = max_shift_y - shifts[0, id_current_image]
        y_end = shifts[0, id_current_image] * -1
    elif min_shift_y < 0 and max_shift_y < 0:
        y_start = shifts[0, id_current_image] * -1
        y_end = max_shift_y - shifts[0, id_current_image]
    else:
        y_start = max_shift_y - shifts[0, id_current_image]
        y_end = min_shift_y - shifts[0, id_current_image]

    return (int(np.around(x_start, decimals=0)),
            int(np.around(x_end, decimals=0)),
            int(np.around(y_start, decimals=0)),
            int(np.around(y_end, decimals=0)))


def trim_image(
        image: CCDData, image_id: int, image_shift: np.ndarray,
        correlation_method: str = 'skimage', verbose: bool = False) -> CCDData:
    """
    Trim image based on a shift compared to a reference image

    Parameters
    ----------
    image
        The image

    image_id
        Number of the image in the sequence

    image_shift
        Shift of this specific image in X and Y direction

    correlation_method
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

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    Returns
    -------
    trimmed_image
        The trimmed image
    """
    if verbose:
        #   Write status to console
        terminal_output.print_to_terminal(
            f"\r\tApply shift to image {image_id}",
        )

    if correlation_method in ['own', 'skimage']:
        #   Calculate indexes from image shifts
        x_start, x_end, y_start, y_end = calculate_index_from_shifts(
            image_shift,
            image_id,
        )
    elif correlation_method == 'aa':
        #   Shift image on sub pixel basis
        image = ccdp.transform_image(
            image,
            shift_scipy,
            shift=image_shift[:, image_id],
            order=1,
        )

        #   TODO: The following calculations do not need to be repeated for
        #    every image. Move it out of the loop.
        #   Calculate maximum and minimum shifts
        min_shift_x, max_shift_x, min_shift_y, max_shift_y = calculate_min_max_image_shifts(
            image_shift,
            python_format=True,
        )

        #   Set trim margins
        if min_shift_x > 0:
            x_start = int(math.ceil(max_shift_x))
            x_end = 0
        elif min_shift_x < 0 and max_shift_x < 0:
            x_start = 0
            x_end = int(math.ceil(np.abs(min_shift_x))) * -1
        else:
            x_start = int(math.ceil(max_shift_x))
            x_end = int(math.ceil(np.abs(min_shift_x))) * -1

        if min_shift_y > 0:
            y_start = int(math.ceil(max_shift_y))
            y_end = 0
        elif min_shift_y < 0 and max_shift_y < 0:
            y_start = 0
            y_end = int(math.ceil(np.abs(min_shift_y))) * -1
        else:
            y_start = int(math.ceil(max_shift_y))
            y_end = int(math.ceil(np.abs(min_shift_y))) * -1

    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Shift method not known. Expected: '
            f'"pixel" or "sub_pixel", but got '
            f'"{correlation_method}" {style.Bcolors.ENDC}'
        )

    #   Trim the image
    return ccdp.trim_image(
        image[y_start:image.shape[0] + y_end, x_start:image.shape[1] + x_end]
    )


#   TODO: Check if this function can be merged with `trim_image` -> Used by N1 script
def trim_image_simple(
        image_file_collection: ccdp.ImageFileCollection, output_path: Path,
        redundant_pixel_x_start: int = 100, redundant_pixel_x_end: int = 100,
        redundant_pixel_y_start: int = 100, redundant_pixel_y_end: int = 100
        ) -> ccdp.ImageFileCollection:
    """
    Trim images in X and Y direction

    Parameters
    ----------
    image_file_collection
        Image file collection

    output_path
        Path to save the individual images

    redundant_pixel_x_start
        Number of Pixel to be removed from the start of the image in
        X direction.

    redundant_pixel_x_end
        Number of Pixel to be removed from the end of the image in
        X direction.

    redundant_pixel_y_start
        Number of Pixel to be removed from the start of the image in
        Y direction.

    redundant_pixel_y_end
        Number of Pixel to be removed from the end of the image in
        Y direction.

    Returns
    -------
    trimmed_images_ifc
        Image file collection pointing to the trimmed images
    """
    terminal_output.print_to_terminal("Trim images", indent=2)

    #   Check directory
    checks.check_output_directories(output_path)
    output_path_trimmed = output_path / 'trimmed'
    checks.check_output_directories(output_path_trimmed)

    for image, file_name in image_file_collection.ccds(
            ccd_kwargs={'unit': 'adu'},
            return_fname=True,
    ):
        #   Trim image
        trimmed_image = ccdp.trim_image(image[
            redundant_pixel_y_start:-redundant_pixel_y_end,
            redundant_pixel_x_start:-redundant_pixel_x_end
            ])

        #   Save the result
        trimmed_image.write(output_path_trimmed / file_name, overwrite=True)

    #   Return new image file collection
    return ccdp.ImageFileCollection(output_path_trimmed)
