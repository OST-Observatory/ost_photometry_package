"""Image-shift algorithms and apply helpers."""

from __future__ import annotations

from pathlib import Path

import astroalign as aa
import ccdproc as ccdp
import numpy as np
import yaml
from astropy.nddata import CCDData, StdDevUncertainty
from skimage.registration import optical_flow_tvl1, phase_cross_correlation
from skimage.transform import SimilarityTransform, warp

from ... import style, terminal_output
from ... import utilities as base_utilities
from ...core.parallel import Executor
from ...terminal_output import print_to_terminal
from .. import plots, utilities
from .trim import trim_image


def apply_xy_image_shift(
        current_image_name: str, image_shifts: np.ndarray,
        image_flips: np.ndarray, image_id: int, output_path: Path,
        shift_method: str = 'aa_true', modify_file_name: bool = False,
        rm_enlarged_keyword: bool = False, instrument: str | None = None,
        verbose: bool = False,
        aa_trim_margins: tuple[int, int, int, int] | None = None,
        ) -> None:
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
        Alignment backend for translation-only apply. Allowed: ``own``,
        ``skimage``, ``aa``. See
        :data:`~ost_photometry.reduce.registration.SHIFT_METHODS`.

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
            aa_trim_margins=aa_trim_margins,
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

    file_name = Path(current_image_name).name
    if modify_file_name:
        filter_ = output_image.meta['filter']
        file_name = 'combined_trimmed_filter_{}.fit'.format(
            filter_.replace("''", "p")
        )

    #   Write trimmed image to disk
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
    file_name = Path(current_image_name).name

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
    file_name = Path(current_image_name).name

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


def calculate_xy_image_shifts_core(
        current_file_name: str, reference_file_name: str,
        image_id: int, correlation_method: str = 'aa_true'
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
        Translation-only backend (``own``, ``skimage``, ``aa``). See
        :data:`~ost_photometry.reduce.registration.SHIFT_METHODS`.
        Default is ``aa_true``.

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
        image_ccd = utilities.adjust_endian_compatibility(image_ccd)
        reference_ccd = utilities.adjust_endian_compatibility(reference_ccd)

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
    file_name = Path(current_file_name).name
    terminal_output.print_to_terminal(
        f'\t{image_id}\t{image_shift[1]:+.1f}\t{image_shift[0]:+.1f}'
        f'\t{file_name}',
        indent=0,
    )

    return image_id, image_shift, flip_necessary


def calculate_xy_image_shifts(
        image_file_collection: ccdp.ImageFileCollection,
        id_reference_image: int, comment: str,
        correlation_method: str = 'aa_true',
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
        Translation-only backend (``own``, ``skimage``, ``aa``). See
        :data:`~ost_photometry.reduce.registration.SHIFT_METHODS`.
        Default is ``aa_true``.

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
    executor = Executor(n_cores_multiprocessing)

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
    current_ccd = utilities.adjust_endian_compatibility(current_ccd)
    reference_ccd = utilities.adjust_endian_compatibility(reference_ccd)

    #   Determine transformation between the images
    transformation_coefficients, (_, _) = aa.find_transform(
        current_ccd,
        reference_ccd,
        detection_sigma=3,
    )

    #   Transform image data. ``footprint_mask`` is True where the warped frame
    #   has no coverage; ``propagate_mask=True`` also warps ``current_ccd.mask``
    #   into that footprint so both stay excluded from later 2D background /
    #   photometry (no separate extra mask). Uncovered pixels are NaN, not 0,
    #   so they are not treated as fake sky.
    image_data, footprint_mask = aa.apply_transform(
        transformation_coefficients,
        current_ccd,
        reference_ccd,
        propagate_mask=True,
        fill_value=np.nan,
    )

    #   Transform uncertainty array
    image_uncertainty, _ = aa.apply_transform(
        transformation_coefficients,
        current_ccd.uncertainty.array,
        reference_ccd.uncertainty.array,
        fill_value=np.nan,
    )

    footprint_mask = np.asarray(footprint_mask, dtype=bool)
    footprint_mask |= ~np.isfinite(image_data)
    footprint_mask |= ~np.isfinite(image_uncertainty)

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
