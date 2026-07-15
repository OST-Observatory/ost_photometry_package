"""Low-level dataset correlation (astropy and srcor-style matchers)."""

from __future__ import annotations

import numpy as np
import typing

from .. import utilities
from ... import style, terminal_output

from astropy.coordinates import SkyCoord, matching
import astropy.units as u
from astropy import wcs


def _dataset_positions_identical(
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    x_current: np.ndarray,
    y_current: np.ndarray,
) -> bool:
    """Return True when two datasets list the same pixel positions."""
    return (
        x_reference.shape == x_current.shape
        and y_reference.shape == y_current.shape
        and np.array_equal(x_reference, x_current)
        and np.array_equal(y_reference, y_current)
    )


def _drop_protected_from_rejected_object_ids(
    rejected_object_ids: np.ndarray,
    special_object_ids: np.ndarray | list[int],
) -> np.ndarray:
    """Remove protected object column IDs from the rejection list."""
    rejected = np.asarray(rejected_object_ids, dtype=int).ravel()
    special = np.asarray(special_object_ids, dtype=int).ravel()
    if rejected.size == 0 or special.size == 0:
        return rejected
    return rejected[~np.isin(rejected, special)]


def correlate_datasets(
        x_pixel_positions: list[np.ndarray],
        y_pixel_positions: list[np.ndarray],
        current_wcs: wcs.WCS, n_objects: int, n_images: int,
        dataset_type: str = 'image', reference_dataset_id: int = 0,
        protected_object_ids: list[int] | None = None,
        reference_object_ids: list[int] | None = None,
        protect_ooi: bool = True,
        calibration_object_ids: list[int] | None = None,
        protect_calibration_objects: bool = False,
        n_allowed_non_detections_object: int = 1,
        separation_limit: u.Quantity = 2. * u.arcsec,
        advanced_cleanup: bool = True,
        max_pixel_between_objects: float = 3.,
        expected_bad_image_fraction: float = 1.0,
        ooi_correlation_strategy: int = 1, cross_identification_limit: int = 1,
        correlation_method: str = 'astropy'
        ) -> tuple[np.ndarray, int, np.ndarray, int]:
    """
    Correlate the pixel positions from different dataset such as
    images or image series.

    Parameters
    ----------
    x_pixel_positions
        Pixel positions in X direction

    y_pixel_positions
        Pixel positions in Y direction

    current_wcs
        WCS information

    n_objects
        Number of objects

    n_images
        Number of images

    dataset_type
        Characterizes the dataset.
        Default is ``image``.

    reference_dataset_id
        ID of the reference dataset
        Default is ``0``.

    reference_object_ids
        IDs of the reference objects (legacy; merged into ``protected_object_ids``).
        Default is ``None``.

    protect_ooi
        If ``True``, include ``reference_object_ids`` in the protected set.
        Default is ``True``.

    protected_object_ids
        Explicit row indices to protect, independent of object type.
        Default is ``None``.

    calibration_object_ids
        IDs of the calibration objects (legacy; merged into ``protected_object_ids``).
        Default is ``None``.

    protect_calibration_objects
        If ``True``, include ``calibration_object_ids`` in the protected set.
        Default is ``False``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    advanced_cleanup
        If ``True`` a multilevel cleanup of the results will be
        attempted. If ``False`` only the minimal necessary removal of
        objects that are not on all datasets will be performed.
        Default is ``True``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    Returns
    -------
    correlation_index
        IDs of the correlated objects

    new_reference_dataset_id
        New ID of the reference dataset
        Default is ``0``.

    rejected_datasets
        IDs of the datasets that were rejected because of insufficient quality

    n_common_objects
        Number of objects found on all datasets
    """
    #   Prepare variables necessary to protect selected objects
    from .protection import merge_protected_object_ids

    special_object_ids = merge_protected_object_ids(
        protected_object_ids=protected_object_ids,
        reference_object_ids=reference_object_ids,
        calibration_object_ids=calibration_object_ids,
        protect_ooi=protect_ooi,
        protect_calibration_objects=protect_calibration_objects,
    )
    protect_special_objects = bool(special_object_ids)

    if correlation_method == 'astropy':
        #   Astropy version: 2x faster than own
        correlation_index, rejected_datasets = correlation_astropy(
            x_pixel_positions,
            y_pixel_positions,
            current_wcs,
            reference_dataset_id=reference_dataset_id,
            special_object_ids=special_object_ids,
            expected_bad_image_fraction=n_allowed_non_detections_object,
            protect_special_objects=protect_special_objects,
            separation_limit=separation_limit,
            advanced_cleanup=advanced_cleanup,
        )
        n_common_objects = len(correlation_index[0])

    elif correlation_method == 'own':
        #   'Own' correlation method requires positions to be in a numpy array
        x_pixel_positions_all = np.zeros((n_objects, n_images))
        y_pixel_positions_all = np.zeros((n_objects, n_images))

        for i in range(0, n_images):
            x_pixel_positions_all[0:len(x_pixel_positions[i]), i] = x_pixel_positions[i]
            y_pixel_positions_all[0:len(y_pixel_positions[i]), i] = y_pixel_positions[i]

        #   Own version based on srcor from the IDL Astro Library
        correlation_index, rejected_datasets, n_common_objects, _ = correlation_own(
            x_pixel_positions_all,
            y_pixel_positions_all,
            max_pixel_between_objects=max_pixel_between_objects,
            expected_bad_image_fraction=expected_bad_image_fraction,
            ooi_correlation_strategy=ooi_correlation_strategy,
            cross_identification_limit=cross_identification_limit,
            reference_dataset_id=reference_dataset_id,
            special_object_ids=special_object_ids,
            n_allowed_non_detections_object=n_allowed_non_detections_object,
            protect_special_objects=protect_special_objects,
        )
    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Correlation method not known. Expected: '
            f'"own" or astropy, but got "{correlation_method}"{style.Bcolors.ENDC}'
        )

    #   Print correlation result or raise error if not enough common
    #   objects were detected
    if n_common_objects == 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nOnly one common object "
            f"found! {style.Bcolors.ENDC}"
        )
    elif n_common_objects == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo common objects "
            f"found!{style.Bcolors.ENDC}"
        )
    else:
        terminal_output.print_to_terminal(
            f"{n_common_objects} objects identified on all {dataset_type}s",
            style_name='GOOD',
            indent=2,
        )

    n_bad_images = len(rejected_datasets)
    if n_bad_images > 0:
        terminal_output.print_to_terminal(
            f"{n_bad_images} images do not meet the criteria -> removed",
            style_name='ITALIC',
            indent=2,
        )
    if n_bad_images > 1:
        terminal_output.print_to_terminal(
            f"Rejected {dataset_type} IDs: {rejected_datasets}",
            style_name='ITALIC',
            indent=2,
        )
    elif n_bad_images == 1:
        terminal_output.print_to_terminal(
            f"ID of the rejected {dataset_type}: {rejected_datasets}",
            style_name='ITALIC',
            indent=2,
        )

    #   Post process correlation results
    #
    #   Remove "bad" images from index array
    #   (only necessary for 'own' method)
    if correlation_method == 'own':
        correlation_index = np.delete(correlation_index, rejected_datasets, 0)

    #   Calculate new index of the reference dataset
    shift_id = np.argwhere(rejected_datasets < reference_dataset_id).ravel()
    new_reference_dataset_id = reference_dataset_id - len(shift_id)

    return correlation_index, new_reference_dataset_id, rejected_datasets, n_common_objects


def correlation_astropy(
        x_pixel_positions: list[np.ndarray],
        y_pixel_positions: list[np.ndarray], current_wcs: wcs.WCS,
        reference_dataset_id: int = 0,
        special_object_ids: list[int] | None = None,
        expected_bad_image_fraction: int = 1,
        protect_special_objects: bool = True,
        separation_limit: u.Quantity = 2. * u.arcsec,
        advanced_cleanup: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    The function correlates data sets based on astropy matching algorithm

    Parameters
    ----------
    x_pixel_positions
        Object positions in pixel coordinates. X direction.

    y_pixel_positions
        Object positions in pixel coordinates. Y direction.

    current_wcs
        WCS information

    reference_dataset_id
        ID of the reference dataset
        Default is ``0``.

    special_object_ids
        IDs of the special objects. The special objects will not be
        removed from the list of objects.
        Default is ``None``.

    expected_bad_image_fraction
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``1``.

    protect_special_objects
        If ``False`` also special objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    advanced_cleanup
        If ``True`` a multilevel cleanup of the results will be
        attempted. If ``False`` only the minimal necessary removal of
        objects that are not on all datasets will be performed.
        Default is ``True``.

    Returns
    -------
    index_array
        IDs of the correlated objects

    rejected_images
        IDs of the images that were rejected because of insufficient quality
    """
    #   Sanitize special object
    if special_object_ids is None or special_object_ids is [None]:
        special_object_ids = []

    #   Number of datasets/images
    n_datasets = len(x_pixel_positions)

    #   Create reference SkyCoord object
    x_pixel_positions_reference = x_pixel_positions[reference_dataset_id].value.ravel()
    y_pixel_positions_reference = y_pixel_positions[reference_dataset_id].value.ravel()
    reference_coordinates = SkyCoord.from_pixel(
        x_pixel_positions_reference,
        y_pixel_positions_reference,
        current_wcs,
    )

    #   Prepare index array and fill in values for the reference dataset
    index_array = np.ones(
        (n_datasets, len(x_pixel_positions[reference_dataset_id])),
        dtype=int
    )
    index_array *= -1
    index_array[reference_dataset_id, :] = np.arange(
        len(x_pixel_positions[reference_dataset_id])
    )

    #   Loop over datasets
    for i in range(0, n_datasets):
        #   Do nothing for the reference dataset
        if i != reference_dataset_id:
            x_pixel_positions_reference = x_pixel_positions[
                reference_dataset_id
            ].value.ravel()
            y_pixel_positions_reference = y_pixel_positions[
                reference_dataset_id
            ].value.ravel()
            x_pixel_positions_current = x_pixel_positions[i].value.ravel()
            y_pixel_positions_current = y_pixel_positions[i].value.ravel()

            try:
                current_coordinates = SkyCoord.from_pixel(
                    x_pixel_positions_current,
                    y_pixel_positions_current,
                    current_wcs,
                )
                index_reference, index_current, distance, _ = (
                    matching.search_around_sky(
                        reference_coordinates,
                        current_coordinates,
                        separation_limit,
                    )
                )
            except (ValueError, TypeError, IndexError):
                if _dataset_positions_identical(
                    x_pixel_positions_reference,
                    y_pixel_positions_reference,
                    x_pixel_positions_current,
                    y_pixel_positions_current,
                ):
                    index_array[i, :] = index_array[reference_dataset_id, :]
                    continue
                raise

            #   Identify and remove duplicate indexes
            index_reference, distance, index_current = utilities.clear_duplicates(
                index_reference,
                distance,
                index_current,
            )
            index_current, _, index_reference = utilities.clear_duplicates(
                index_current,
                distance,
                index_reference,
            )

            #   Fill ID array
            index_array[i, index_reference] = index_current

    #   Cleanup: Remove "bad" objects and datasets
    #
    #   1. Remove bad objects (pre burner) -> Useful to remove bad objects
    #                                         that may spoil the correct
    #                                        identification of bad datasets.
    if advanced_cleanup:
        #   Identify objects that were not identified in all datasets
        rows_to_rm = np.where(index_array == -1)

        #   Reduce to unique objects
        objects_to_rm, n_times_to_rm = np.unique(
            rows_to_rm[1],
            return_counts=True,
        )

        #   Identify objects that are not in >= "expected_bad_image_fraction"
        #   of all images
        ids_rejected_objects = np.argwhere(
            n_times_to_rm >= expected_bad_image_fraction
        ).ravel()
        rejected_object_ids = objects_to_rm[ids_rejected_objects]

        #   Check if special objects are within the "bad" objects
        if protect_special_objects and np.any(
            np.isin(rejected_object_ids, special_object_ids)
        ):
            rejected_object_ids = _drop_protected_from_rejected_object_ids(
                rejected_object_ids,
                special_object_ids,
            )

        #   Remove "bad" objects
        index_array = np.delete(index_array, rejected_object_ids, 1)

        #   Calculate new special object position
        if not isinstance(special_object_ids, np.ndarray):
            special_object_ids = np.array(special_object_ids)
        for index, special_object_id in np.ndenumerate(special_object_ids):
            object_shift = np.argwhere(rejected_object_ids < special_object_id).ravel()
            n_shift = len(object_shift)
            special_object_ids[index] = special_object_id - n_shift

        #   2. Remove bad images

        #   Identify objects that were not identified in all datasets
        rows_to_rm = np.where(index_array == -1)

        #   Reduce to unique objects
        images_to_rm, n_times_to_rm = np.unique(
            rows_to_rm[0],
            return_counts=True,
        )

        #   Create mask -> Identify all datasets as bad that contain less
        #                  than 98% of all objects from the reference dataset.
        mask = n_times_to_rm > 0.02 * len(x_pixel_positions[reference_dataset_id])
        rejected_images = images_to_rm[mask]

        #   Remove those datasets
        index_array = np.delete(index_array, rejected_images, 0)

    else:
        rejected_images = np.array([], dtype=int)

    #   3. Remove remaining objects that are not on all datasets
    #      (afterburner)
    #
    #   Identify objects that were not identified in all datasets
    rows_to_rm = np.where(index_array == -1)

    if protect_special_objects:
        #   Check if special objects are within the "bad" objects
        ref_is_in = np.isin(rows_to_rm[1], special_object_ids)

        #   If YES remove special objects from "bad" objects and remove
        #   the datasets on which they were not detected instead.
        if np.any(ref_is_in):
            if n_datasets <= 2:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nSpecial objects found on only"
                    f"one image or not at all. This is not sufficient. "
                    f"=> Exit {style.Bcolors.ENDC}"
                )
            rejected_object_ids = rows_to_rm[1]
            rejected_object_ids = np.unique(rejected_object_ids)
            rejected_object_ids = _drop_protected_from_rejected_object_ids(
                rejected_object_ids,
                special_object_ids,
            )

            #   Remove remaining bad objects
            index_array = np.delete(index_array, rejected_object_ids, 1)

            #   Remove datasets
            rows_to_rm = np.where(index_array == -1)
            rejected_images_two = np.unique(rows_to_rm[0])
            index_array = np.delete(index_array, rejected_images_two, 0)

            rejected_images_two_old = []
            for images_in_two in rejected_images_two:
                for images_in_one in rejected_images:
                    if images_in_one <= images_in_two:
                        images_in_two += 1
                rejected_images_two_old.append(images_in_two)

            rejected_images = np.concatenate(
                (rejected_images, np.array(rejected_images_two_old))
            )

            return index_array, rejected_images

    #   Remove bad objects
    index_array = np.delete(index_array, rows_to_rm[1], 1)

    return index_array, rejected_images


def correlation_own(
        x_pixel_positions: np.ndarray, y_pixel_positions: np.ndarray,
        max_pixel_between_objects: float = 3.,
        expected_bad_image_fraction: float = 1.0,
        cross_identification_limit: int = 1, reference_dataset_id: int = 0,
        special_object_ids: list[int] | None = None,
        n_allowed_non_detections_object: int = 1, indent: int = 1,
        ooi_correlation_strategy: int | None = None, magnitudes: np.ndarray | None = None,
        silent: bool = False, protect_special_objects: bool = True
        ) -> tuple[np.ndarray, np.ndarray | int, int, np.ndarray]:
    """
    Correlate source positions from several images (e.g., different images)

    Source matching is done by finding objects within a specified
    radius. The code is adapted from the standard srcor routine from
    the IDL Astronomy User's Library. The normal srcor routine was
    extended to fit the requirements of the C7 experiment within the
    astrophysics lab course at Potsdam University.

    SOURCE: Adapted from the IDL Astro Library

    Parameters
    ----------
    x_pixel_positions

    y_pixel_positions
        Arrays of x and y coordinates (several columns each). The
        following syntax is expected: x[array of source
        positions]. The program marches through the columns
        element by element, looking for the closest match.

    max_pixel_between_objects
        Critical radius outside which correlations are rejected,
        but see ``ooi_correlation_strategy`` below.
        Default is ````.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        positions.
        Default is ``1.0``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_dataset_id
        ID of the reference dataset (e.g., an image).
        Default is ``0``.

    special_object_ids
        Ids of the special objects. The special objects will not be
        removed from the list of objects.
        Default is ``None``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``1``.

    indent
        Indentation for the console output lines
        Default is ``1``.

    ooi_correlation_strategy
        Integer code controlling matching / deduplication (see table). If
        ``None``, treated as ``0``. When ``magnitudes`` is given, the code is
        forced to ``4`` (brightest match within radius).

        ===== =============================================
        Value Meaning
        ===== =============================================
        0     Closest match per reference object; drop if outside radius (many-to-one OK).
        1     One-to-one: keep only minimum-distance match per reference object; prune others.
        2     One-to-one: remove all objects that had multiple in-radius candidates (any distance).
        3     Return all in-radius matches (wider index arrays).
        4     Brightest in-radius match (requires ``magnitudes``); set automatically when magnitudes given.
        ===== =============================================

        Default is ``None`` (same as ``0``).

    magnitudes
        An array of stellar magnitudes corresponding to x and y.
        If magnitude is supplied, the brightest objects within
        'max_pixel_between_objects' is taken as a match. The ``ooi_correlation_strategy`` keyword
        is set to 4 internally when magnitudes are supplied.
        Default is ``None``.

    silent
        Suppresses output if True.
        Default is ``False``.

    protect_special_objects
        Also special objects will be rejected if Falls.
        Default is ``True``.

    Returns
    -------
    index_array
        Array of index positions of matched objects in the images,
        set to -1 if no matches are found.

    rejected_images
        Vector with indexes of all images which should be removed

    count
        Integer giving number of matches returned

    rejected_objects
        Vector with indexes of all objects which should be removed
    """
    #   Sanitize special object
    if special_object_ids is None:
        special_object_ids = []

    #   Keywords.
    if ooi_correlation_strategy is None:
        ooi_correlation_strategy = 0
    if magnitudes is not None:
        ooi_correlation_strategy = 4
    if ooi_correlation_strategy not in (0, 1, 2, 3, 4):
        terminal_output.print_to_terminal(
            "Invalid ooi_correlation_strategy code.",
            indent=indent,
        )

    #   Set up some variables.
    #
    #   Number of images
    n_images = len(x_pixel_positions[0, :])
    #   Max. number of objects in the images
    n_objects = len(x_pixel_positions[:, 0])
    #   Square of the required maximal distance
    dcr2 = max_pixel_between_objects ** 2.

    #   Debug output
    if not silent:
        terminal_output.print_to_terminal(
            f"   ooi_correlation_strategy = {ooi_correlation_strategy}",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            f"   {n_images} images (figures)",
            indent=indent,
        )
        terminal_output.print_to_terminal(
            f"   max. number of objects {n_objects}",
            indent=indent,
        )

    #   The main loop.  Step through each object of the reference dataset,
    #                   look for matches in all the other images.
    #
    #   Outer loop to allow for a pre burner to rejected_images objects that
    #   are not detected on enough images
    #
    #   Initialize counter of mutual sources and rejected objects
    count = 0
    rejected_objects = 0

    index_array: np.ndarray | None = None
    rejected_img = np.zeros(n_images, dtype=int)
    for z in range(0, 2):
        #    Prepare index and rejected_images arrays
        #       <- arbitrary * 10 to allow for multi identifications (strategy 3)
        index_array = np.zeros((n_images, n_objects * 10), dtype=int) - 1
        rejected_img = np.zeros(n_images, dtype=int)
        rejected_obj = np.zeros(n_objects, dtype=int)
        #   Reset counter of mutual sources
        count = 0

        #   Loop over the number of objects
        for i in range(0, n_objects):
            #   Check that objects exists in the reference dataset
            if x_pixel_positions[i, reference_dataset_id] != 0.:
                #   Prepare dummy arrays and counter for bad images
                _correlation_index = np.zeros(n_images, dtype=int) - 1
                _correlation_index[reference_dataset_id] = i
                _img_rejected = np.zeros(n_images, dtype=int)
                _obj_rejected = np.zeros(n_objects, dtype=int)
                _n_bad_images = 0

                #   Loop over all images
                for j in range(0, n_images):
                    #   Exclude reference dataset
                    if j != reference_dataset_id:
                        comparison_x_pixel_positions = np.copy(
                            x_pixel_positions[:, j]
                        )
                        comparison_y_pixel_positions = np.copy(
                            y_pixel_positions[:, j]
                        )
                        comparison_x_pixel_positions[comparison_x_pixel_positions == 0] = 9E13
                        comparison_y_pixel_positions[comparison_y_pixel_positions == 0] = 9E13

                        #   Calculate radii
                        d2 = ((x_pixel_positions[i, reference_dataset_id] - comparison_x_pixel_positions) ** 2
                              + (y_pixel_positions[i, reference_dataset_id] - comparison_y_pixel_positions) ** 2)

                        if ooi_correlation_strategy == 3:
                            #   Find objects with distances that are smaller
                            #   than the required dcr
                            possible_matches = np.argwhere(d2 <= dcr2).ravel()

                            #   Fill ind array
                            n_possible_matches = len(possible_matches)
                            if n_possible_matches:
                                index_array[j, count:count + n_possible_matches] = possible_matches
                                index_array[reference_dataset_id, count:count + n_possible_matches] = \
                                    _correlation_index[reference_dataset_id]
                                count += n_possible_matches
                        else:
                            #   Find the object with the smallest distance
                            smallest_distance_between_matches = np.amin(d2)
                            best_match = np.argmin(d2)

                            #   Check the critical radius criterion. If this
                            #   fails, the source will be marked as bad.
                            if smallest_distance_between_matches <= dcr2:
                                _correlation_index[j] = best_match
                            else:
                                #   Number of bad images for this source
                                #   -> counts up
                                _n_bad_images += 1

                                #   Fill the rejected_images vectors
                                #   Mark image as "problematic"
                                _img_rejected[j] = 1

                                #   Check that object is not a reference
                                if i not in special_object_ids or not protect_special_objects:
                                    #   Mark object as problematic
                                    #   -> counts up
                                    _obj_rejected[i] += 1

                if ooi_correlation_strategy != 3:
                    if (_n_bad_images > (1 - expected_bad_image_fraction) * n_images
                            and (i not in special_object_ids or not protect_special_objects)):
                        rejected_obj += _obj_rejected
                        continue
                    else:
                        rejected_img += _img_rejected

                        index_array[:, count] = _correlation_index
                        count += 1

        #   Prepare to discard objects that are not on
        #   `n_allowed_non_detections_object` images
        rejected_obj = np.argwhere(
            rejected_obj >= n_allowed_non_detections_object
        ).ravel()
        rej_obj_tup = tuple(rejected_obj)

        #   Exit loop if there are no objects to be removed
        #   or if it is the second iteration
        if len(rejected_obj) == 0 or z == 1:
            break

        rejected_objects = np.copy(rejected_obj)

        if not silent:
            terminal_output.print_to_terminal(
                f"   {len(rejected_objects)} objects removed because they "
                f"are not found on >={n_allowed_non_detections_object} images",
                indent=indent,
            )

        #   Discard objects that are on not enough datasets
        x_pixel_positions[rej_obj_tup, reference_dataset_id] = 0.
        y_pixel_positions[rej_obj_tup, reference_dataset_id] = 0.

    if not silent:
        terminal_output.print_to_terminal(
            f"   {count} matches found.",
            indent=indent,
        )

    if count > 0:
        index_array = index_array[:, 0:count]
        _correlation_index_2 = np.zeros(count, dtype=int) - 1
    else:
        rejected_images: int | np.ndarray = -1
        return index_array, rejected_images, count, rejected_objects

    #   Return in case of ooi_correlation_strategy 0 and 3
    if ooi_correlation_strategy == 0:
        return index_array, rejected_img, count, rejected_objects
    if ooi_correlation_strategy == 3:
        return index_array

    #   Modify the matches depending on input options.
    #
    if not silent:
        if ooi_correlation_strategy == 4:
            terminal_output.print_to_terminal(
                "   Cleaning up output array using magnitudes.",
                indent=indent,
            )
        else:
            if ooi_correlation_strategy == 1:
                terminal_output.print_to_terminal(
                    "   Cleaning up output array (ooi_correlation_strategy = 1).",
                    indent=indent,
                )
            else:
                terminal_output.print_to_terminal(
                    "   Cleaning up output array (ooi_correlation_strategy = 2).",
                    indent=indent,
                )

    #   Loop over the images
    for j in range(0, len(index_array[:, 0])):
        if j == reference_dataset_id:
            continue
        #   Loop over the indexes of the objects
        for i in range(0, np.max(index_array[j, :])):
            c_save = len(index_array[j, :])

            #   First find many-to-one identifications
            many_to_one_ids = np.argwhere(index_array[j, :] == i).ravel()
            n_multi = len(many_to_one_ids)
            #   All but one of the images in WW must eventually be removed.
            if n_multi > 1:
                #   Mark images that should be rejected.
                if n_multi >= cross_identification_limit and n_images > 2:
                    rejected_img[j] = 1

                if ooi_correlation_strategy == 4 and n_images == 2:
                    possible_matches = np.argmin(
                        magnitudes[
                            index_array[reference_dataset_id, many_to_one_ids]
                        ]
                    )
                else:
                    #   Calculate individual distances of the many-to-one
                    #   identifications
                    x_current = x_pixel_positions[i, j]
                    y_current = y_pixel_positions[i, j]
                    x_many = x_pixel_positions[
                        index_array[reference_dataset_id, many_to_one_ids],
                        reference_dataset_id
                    ]
                    y_many = y_pixel_positions[
                        index_array[reference_dataset_id, many_to_one_ids],
                        reference_dataset_id
                    ]
                    d2 = (x_current - x_many) ** 2 + (y_current - y_many) ** 2

                    #   Logical test
                    if len(d2) != n_multi:
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL}\nLogic error 1"
                            f"{style.Bcolors.ENDC}"
                        )

                    #   Find the element with the minimum distance
                    possible_matches = np.argmin(d2)

                #   Delete the minimum element from the
                #   deletion list itself.
                if ooi_correlation_strategy == 1:
                    many_to_one_ids = np.delete(
                        many_to_one_ids,
                        possible_matches
                    )

                #   Now delete the deletion list from the original index
                #   arrays.
                for t in range(0, len(index_array[:, 0])):
                    _correlation_index_2 = index_array[t, :]
                    _correlation_index_2 = np.delete(
                        _correlation_index_2,
                        many_to_one_ids
                    )
                    for o in range(0, len(_correlation_index_2)):
                        index_array[t, o] = _correlation_index_2[o]

                #   Cut arrays depending on the number of
                #   one-to-one matches found in all images
                index_array = index_array[:, 0:len(_correlation_index_2)]

                #   Logical tests
                if ooi_correlation_strategy == 2:
                    if len(index_array[j, :]) != (c_save - n_multi):
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(index_array[reference_dataset_id, :]) != (c_save - n_multi):
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                else:
                    if len(index_array[j, :]) != (c_save - n_multi + 1):
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(index_array[reference_dataset_id, :]) != (c_save - n_multi + 1):
                        raise RuntimeError(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                if len(index_array[j, :]) != len(index_array[reference_dataset_id, :]):
                    raise RuntimeError(
                        f"{style.Bcolors.FAIL}\nLogic error 4"
                        f"{style.Bcolors.ENDC}"
                    )

    #   Determine the indexes of the images to be discarded
    rejected_images = np.argwhere(rejected_img >= 1).ravel()

    #   Set count variable once more
    count = len(index_array[reference_dataset_id, :])

    if not silent:
        terminal_output.print_to_terminal(
            f"       {len(index_array[reference_dataset_id, :])} unique "
            f"matches found.",
            indent=indent,
            style_name='OKGREEN',
        )

    return index_array, rejected_images, count, rejected_objects
