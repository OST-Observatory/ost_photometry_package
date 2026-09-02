"""Image alignment orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path

import ccdproc as ccdp
import numpy as np
from astropy.nddata import CCDData

from ... import checks, style, terminal_output
from ...core.parallel import Executor
from .. import utilities
from ..image_collection import image_file_collection as make_image_file_collection
from ..trim_slices import aa_common_trim_margins
from .shifts import (
    apply_astro_align,
    apply_optical_flow,
    apply_xy_image_shift,
    calculate_xy_image_shifts,
)


def align_images(
        image_path: str | Path, output_dir: str | Path,
        image_type_list: list[str], reference_image_index: int = 0,
        enlarged_only: bool = False, shift_method: str = 'aa_true',
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

    reference_image_index
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
        Default is ``aa_true``.

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
    image_file_collection = make_image_file_collection(file_path)

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
                reference_image_index=reference_image_index,
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
            reference_image_index=reference_image_index,
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
        shift_method: str = 'aa_true',
        n_cores_multiprocessing: int | None = None,
        reference_image_index: int = 0,
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
        Default is ``aa_true``.

    n_cores_multiprocessing
        Number of cores to use during calculation of the image shifts.
        Default is ``None``.

    reference_image_index
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
            reference_image_index,
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
        aa_trim_margins = (
            aa_common_trim_margins(image_shifts)
            if shift_method == 'aa'
            else None
        )
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
                        'aa_trim_margins': aa_trim_margins,
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
        reference_file_name = image_file_collection.files[reference_image_index]

        #   Initialize multiprocessing object
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=len(image_file_collection.files),
            add_progress_bar=True,
        )

        #   Trim all images
        for current_image_name in image_file_collection.files:
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
        reference_file_name = image_file_collection.files[reference_image_index]

        #   Initialize multiprocessing object
        executor = Executor(
            n_cores_multiprocessing,
            n_tasks=len(image_file_collection.files),
            add_progress_bar=True,
        )

        #   Trim all images
        for current_image_name in image_file_collection.files:
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
    image_file_collection = make_image_file_collection(file_path)

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
