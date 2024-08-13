############################################################################
#                               Libraries                                  #
############################################################################
import multiprocessing as mp

import numpy as np

import typing

if typing.TYPE_CHECKING:
    from . import analyze, plots

from . import calibration_data, utilities, plots
from .. import style, terminal_output
from .. import utilities as base_utilities

from astropy.coordinates import SkyCoord, matching
import astropy.units as u
from astropy.table import Table
from astropy import wcs


############################################################################
#                           Routines & definitions                         #
############################################################################


def find_objects_of_interest_astropy(
        x_pixel_position_dataset: np.ndarray,
        y_pixel_position_dataset: np.ndarray,
        objects_of_interest: list['analyze.ObjectOfInterest'], filter_: str,
        current_wcs: wcs.WCS, separation_limit: u.Quantity = 2. * u.arcsec
) -> None:
    """
    Find the image coordinates of a star based on the stellar
    coordinates and the WCS of the image, using astropy matching
    algorithms.

    Parameters
    ----------
    x_pixel_position_dataset
        Positions of the objects in Pixel in X direction

    y_pixel_position_dataset
        Positions of the objects in Pixel in Y direction

    objects_of_interest
        Object with 'object of interest' properties

    filter_
        Filter identifier

    current_wcs
        WCS info


    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.
    """
    #   Create SkyCoord object for dataset
    coordinates_dataset = SkyCoord.from_pixel(
        x_pixel_position_dataset,
        y_pixel_position_dataset,
        current_wcs,
    )

    for object_ in objects_of_interest:
        coordinates_object = object_.coordinates_object

        #   Find matches in the dataset
        separation = coordinates_dataset.separation(coordinates_object)
        mask = separation < separation_limit
        object_id = np.argwhere(mask).ravel()

        if len(object_id) > 1:
            #   message would be feasible
            terminal_output.print_to_terminal(
                f"More than one object detected within the separation limit to "
                f"{object_.name}. Use the object that is the closest.",
                style_name='WARNING',
            )
            object_id = np.argmin(separation)

        elif not object_id:
            terminal_output.print_to_terminal(
                f"No object detected within the separation limit to "
                f"{object_.name}. Set object ID to None",
                style_name='WARNING',
            )
            object_id = None

        else:
            object_id = object_id[0]

        #   Add ID to object of interest
        object_.id_in_image_series[filter_] = object_id


def find_objects_of_interest_srcor(
        x_pixel_position_dataset: np.ndarray,
        y_pixel_position_dataset: np.ndarray,
        objects_of_interest: list['analyze.ObjectOfInterest'], filter_: str,
        current_wcs: wcs.WCS, max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1, verbose: bool = False) -> None:
    """
    Find the image coordinates of a star based on the stellar
    coordinates and the WCS of the image

    Parameters
    ----------
    x_pixel_position_dataset
        Positions of the objects in Pixel in X direction

    y_pixel_position_dataset
        Positions of the objects in Pixel in Y direction

    objects_of_interest
        Object with 'object of interest' properties

    filter_
        Filter identifier

    current_wcs
        WCS info

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.
    """
    #   Number of objects
    n_obj_dataset = len(x_pixel_position_dataset)

    #   Define and fill new arrays to allow correlation
    pixel_position_all_x = np.zeros((n_obj_dataset, 2))
    pixel_position_all_y = np.zeros((n_obj_dataset, 2))
    pixel_position_all_x[0:n_obj_dataset, 1] = x_pixel_position_dataset
    pixel_position_all_y[0:n_obj_dataset, 1] = y_pixel_position_dataset

    #   Loop over all objects of interest
    for object_ in objects_of_interest:
        coordinates_object = object_.coordinates_object

        #   Convert ra & dec to pixel coordinates
        obj_pixel_position_x, obj_pixel_position_y = current_wcs.all_world2pix(
            coordinates_object.ra,
            coordinates_object.dec,
            0,
        )

        #   Add pixel position of object of interest to pixel position array
        pixel_position_all_x[0, 0] = obj_pixel_position_x
        pixel_position_all_y[0, 0] = obj_pixel_position_y

        #   Correlate calibration stars with stars on the image
        index_obj, reject, count, reject_obj = correlation_own(
            pixel_position_all_x,
            pixel_position_all_y,
            max_pixel_between_objects=max_pixel_between_objects,
            option=own_correlation_option,
            silent=not verbose,
        )

        #   Current object ID
        object_id = index_obj[1]

        if len(object_id) > 1:
            #   message would be feasible
            terminal_output.print_to_terminal(
                f"More than one object detected within the separation limit to "
                f"{object_.name}. Take the first one in the list.",
                style_name='WARNING',
            )
            object_id = object_id[0]

        elif not object_id:
            terminal_output.print_to_terminal(
                f"No object detected within the separation limit to "
                f"{object_.name}. Set object ID to None",
                style_name='WARNING',
            )
            object_id = None

        else:
            object_id = object_id[0]

        #   Add ID to object of interest
        object_.id_in_image_series[filter_] = object_id


def identify_object_of_interest_in_dataset(
        x_pixel_positions: np.ndarray, y_pixel_positions: np.ndarray,
        objects_of_interest: list['analyze.ObjectOfInterest'], filter_: str,
        current_wcs: wcs.WCS, separation_limit: u.Quantity = 2. * u.arcsec,
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        verbose: bool = False, correlation_method: str = 'astropy') -> None:
    """
    Identify a specific star based on its right ascension and declination
    in a dataset of pixel coordinates. Requires a valid WCS.

    Parameters
    ----------
    x_pixel_positions
        Object positions in pixel coordinates. X direction.

    y_pixel_positions
        Object positions in pixel coordinates. Y direction.

    objects_of_interest
        Object with 'object of interest' properties

    filter_
        Filter identifier

    current_wcs
        WCS information

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.
    """
    if correlation_method == 'astropy':
        find_objects_of_interest_astropy(
            x_pixel_positions,
            y_pixel_positions,
            objects_of_interest,
            filter_,
            current_wcs,
            separation_limit=separation_limit,
        )

    elif correlation_method == 'own':
        find_objects_of_interest_srcor(
            x_pixel_positions,
            y_pixel_positions,
            objects_of_interest,
            filter_,
            current_wcs,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            verbose=verbose,
        )

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own".'
            f'Got {correlation_method} instead.'
        )


def correlate_datasets(
        x_pixel_positions: list[np.ndarray],
        y_pixel_positions: list[np.ndarray],
        current_wcs: wcs.WCS, n_objects: int, n_images: int,
        dataset_type: str = 'image', reference_dataset_id: int = 0,
        reference_obj_ids: list[int] | None = None,
        protect_reference_obj: bool = True,
        n_allowed_non_detections_object: int = 1,
        separation_limit: u.Quantity = 2. * u.arcsec,
        advanced_cleanup: bool = True,
        max_pixel_between_objects: float = 3.,
        expected_bad_image_fraction: float = 1.0,
        own_correlation_option: int = 1, cross_identification_limit: int = 1,
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

    reference_obj_ids
        IDs of the reference objects. The reference objects will not be
        removed from the list of objects.
        Default is ``None``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

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

    own_correlation_option
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

    new_reference_image_id
        New ID of the reference image
        Default is ``0``.

    rejected_images
        IDs of the images that were rejected because of insufficient quality

    n_common_objects
        Number of objects found on all datasets
    """
    if correlation_method == 'astropy':
        #   Astropy version: 2x faster than own
        correlation_index, rejected_images = correlation_astropy(
            x_pixel_positions,
            y_pixel_positions,
            current_wcs,
            reference_dataset_id=reference_dataset_id,
            reference_object_ids=reference_obj_ids,
            expected_bad_image_fraction=n_allowed_non_detections_object,
            protect_reference_obj=protect_reference_obj,
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
        correlation_index, rejected_images, n_common_objects, _ = correlation_own(
            x_pixel_positions_all,
            y_pixel_positions_all,
            max_pixel_between_objects=max_pixel_between_objects,
            expected_bad_image_fraction=expected_bad_image_fraction,
            option=own_correlation_option,
            cross_identification_limit=cross_identification_limit,
            reference_dataset_id=reference_dataset_id,
            reference_obj_id=reference_obj_ids,
            n_allowed_non_detections_object=n_allowed_non_detections_object,
            protect_reference_obj=protect_reference_obj,
        )
    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Correlation method not known. Expected: '
            f'"own" or astropy, but got "{correlation_method}"{style.Bcolors.ENDC}'
        )

    ###
    #   Print correlation result or raise error if not enough common
    #   objects were detected
    #
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
            indent=2,
        )

    n_bad_images = len(rejected_images)
    if n_bad_images > 0:
        terminal_output.print_to_terminal(
            f"{n_bad_images} images do not meet the criteria -> removed",
            indent=2,
        )
    if n_bad_images > 1:
        terminal_output.print_to_terminal(
            f"Rejected {dataset_type} IDs: {rejected_images}",
            indent=2,
        )
    elif n_bad_images == 1:
        terminal_output.print_to_terminal(
            f"ID of the rejected {dataset_type}: {rejected_images}",
            indent=2,
        )
    terminal_output.print_to_terminal('')

    ###
    #   Post process correlation results
    #
    #   Remove "bad" images from index array
    #   (only necessary for 'own' method)
    if correlation_method == 'own':
        correlation_index = np.delete(correlation_index, rejected_images, 0)

    #   Calculate new index of the reference dataset
    #   TODO: Check if a bug hides here
    shift_id = np.argwhere(rejected_images < reference_dataset_id)
    new_reference_image_id = reference_dataset_id - len(shift_id)

    return correlation_index, new_reference_image_id, rejected_images, n_common_objects


def correlation_astropy(
        x_pixel_positions: list[np.ndarray],
        y_pixel_positions: list[np.ndarray], current_wcs: wcs.WCS,
        reference_dataset_id: int = 0,
        reference_object_ids: list[int] | None = None,
        expected_bad_image_fraction: int = 1,
        protect_reference_obj: bool = True,
        separation_limit: u.Quantity = 2. * u.arcsec,
        advanced_cleanup: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Correlation based on astropy matching algorithm

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

    reference_object_ids
        IDs of the reference objects. The reference objects will not be
        removed from the list of objects.
        Default is ``None``.

    expected_bad_image_fraction
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``1``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
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
    #   Sanitize reference object
    if reference_object_ids is None:
        reference_object_ids = []

    #   Number of datasets/images
    n_datasets = len(x_pixel_positions)

    #   Create reference SkyCoord object
    reference_coordinates = SkyCoord.from_pixel(
        x_pixel_positions[reference_dataset_id],
        y_pixel_positions[reference_dataset_id],
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
        #   Do nothing for the reference object
        if i != reference_dataset_id:
            #   Dirty fix: In case of identical positions between the
            #              reference and the current data set,
            #              matching.search_around_sky will fail.
            #              => set reference indexes
            if ((len(x_pixel_positions[i]) == len(x_pixel_positions[reference_dataset_id])) and
                    (np.all(x_pixel_positions[i] == x_pixel_positions[reference_dataset_id]) and
                     np.all(y_pixel_positions[i] == y_pixel_positions[reference_dataset_id]))):
                index_array[i, :] = index_array[reference_dataset_id, :]
            else:
                #   Create coordinates object
                current_coordinates = SkyCoord.from_pixel(
                    x_pixel_positions[i],
                    y_pixel_positions[i],
                    current_wcs,
                )

                #   Find matches between the datasets
                index_reference, index_current, _, _ = matching.search_around_sky(
                    reference_coordinates,
                    current_coordinates,
                    separation_limit,
                )

                #   Fill ID array
                index_array[i, index_reference] = index_current

    ###
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
        )
        rejected_object_ids = objects_to_rm[ids_rejected_objects].flatten()

        #   Check if reference objects are within the "bad" objects
        ref_is_in = np.isin(rejected_object_ids, reference_object_ids)

        #   If YES remove reference objects from the "bad" objects
        if protect_reference_obj and np.any(ref_is_in):
            id_difference = rejected_object_ids.reshape(rejected_object_ids.size, 1) - reference_object_ids
            id_reference_obj_in_rejected_objects = np.argwhere(
                id_difference == 0.
            )[:, 0]
            rejected_object_ids = np.delete(
                rejected_object_ids,
                id_reference_obj_in_rejected_objects
            )

        #   Remove "bad" objects
        index_array = np.delete(index_array, rejected_object_ids, 1)

        #   Calculate new reference object position
        if not isinstance(reference_object_ids, np.ndarray):
            reference_object_ids = np.array(reference_object_ids)
        for index, reference_obj_id in np.ndenumerate(reference_object_ids):
            object_shift = np.argwhere(rejected_object_ids < reference_obj_id)
            n_shift = len(object_shift)
            reference_object_ids[index] = reference_obj_id - n_shift

        #   2. Remove bad images

        #   Identify objects that were not identified in all datasets
        rows_to_rm = np.where(index_array == -1)

        #   Reduce to unique objects
        images_to_rm, n_times_to_rm = np.unique(
            rows_to_rm[0],
            return_counts=True,
        )

        #   Create mask -> Identify all datasets as bad that contain less
        #                  than 90% of all objects from the reference image.
        mask = n_times_to_rm > 0.02 * len(x_pixel_positions[reference_dataset_id])
        rejected_images = images_to_rm[mask]

        #   Remove those datasets
        index_array = np.delete(index_array, rejected_images, 0)

    else:
        rejected_images = np.array([], dtype=int)

    #   3. Remove remaining objects that are not on all datasets
    #      (afterburner)

    #   Identify objects that were not identified in all datasets
    rows_to_rm = np.where(index_array == -1)

    if protect_reference_obj:
        #   Check if reference objects are within the "bad" objects
        ref_is_in = np.isin(rows_to_rm[1], reference_object_ids)

        #   If YES remove reference objects from "bad" objects and remove
        #   the datasets on which they were not detected instead.
        if np.any(ref_is_in):
            if n_datasets <= 2:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nReference object only found on "
                    "one or on none image at all. This is not sufficient. "
                    f"=> Exit {style.Bcolors.ENDC}"
                )
            rejected_object_ids = rows_to_rm[1]
            rejected_object_ids = np.unique(rejected_object_ids)
            id_difference = rejected_object_ids.reshape(rejected_object_ids.size, 1) - reference_object_ids
            id_reference_obj_in_rejected_objects = np.argwhere(
                id_difference == 0.
            )[:, 0]
            rejected_object_ids = np.delete(
                rejected_object_ids,
                id_reference_obj_in_rejected_objects
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
        reference_obj_id: list[int] | None = None,
        n_allowed_non_detections_object: int = 1, indent: int = 1,
        option: int | None = None, magnitudes: np.ndarray | None = None,
        silent: bool = False, protect_reference_obj: bool = True
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
        but see 'option' below.
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

    reference_obj_id
        Ids of the reference objects. The reference objects will not be
        removed from the list of objects.
        Default is ``None``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``1``.

    indent
        Indentation for the console output lines
        Default is ``1``.

    option
        Changes behavior of the program & description of output
        lists slightly, as follows:
          OPTION=0 | left out
                For each object of the reference image the closest match
                from all other images is found, but if none is found within
                the distance of 'dcr', the match is thrown out. Thus, the
                index of that object will not appear in the 'ind' output
                array.
          OPTION=1
                Forces the output mapping to be one-to-one.  OPTION=0
                results, in general, in a many-to-one mapping from the
                reference image to the all other images. Under OPTION=1, a
                further processing step is performed to keep only the
                minimum-distance match, whenever an entry from the
                reference image appears more than once in the initial
                mapping.
                Caution: The entries that exceed the distance of the
                         minimum-distance match will be removed from all
                         images. Hence, selection of reference image
                         matters.
          OPTION=2
                Same as OPTION=1, except that all entries which appears
                more than once in the initial mapping will be removed from
                all images independent of distance.
          OPTION=3
                All matches that are within 'dcr' are returned
        Default is ``None``.

    magnitudes
        An array of stellar magnitudes corresponding to x and y.
        If magnitude is supplied, the brightest objects within
        'max_pixel_between_objects' is taken as a match. The option keyword
        is set to 4 internally.
        Default is ``None``.

    silent
        Suppresses output if True.
        Default is ``False``.

    protect_reference_obj
        Also reference objects will be rejected if Falls.
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
    #   Sanitize reference object
    if reference_obj_id is None:
        reference_obj_id = []

    ###
    #   Keywords.
    #
    if option is None:
        option = 0
    if magnitudes is not None:
        option = 4
    if option < 0 or option > 3:
        terminal_output.print_to_terminal(
            "Invalid option code.",
            indent=indent,
        )

    ###
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
            f"   Option code = {option}",
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

    ###
    #   The main loop.  Step through each object of the reference image,
    #                   look for matches in all the other images.
    #

    #   Outer loop to allow for a pre burner to rejected_images objects that
    #   are not detected on enough images
    #
    #   Initialize counter of mutual sources and rejected objects
    count = 0
    rejected_objects = 0
    #
    index_array: np.ndarray | None = None
    rejected_img = np.zeros(n_images, dtype=int)
    for z in range(0, 2):
        #    Prepare index and rejected_images arrays
        #       <- arbitrary * 10 to allow for multi identifications (option 3)
        index_array = np.zeros((n_images, n_objects * 10), dtype=int) - 1
        rejected_img = np.zeros(n_images, dtype=int)
        rejected_obj = np.zeros(n_objects, dtype=int)
        #   Reset counter of mutual sources
        count = 0

        #   Loop over the number of objects
        for i in range(0, n_objects):
            #   Check that objects exists in the reference image
            if x_pixel_positions[i, reference_dataset_id] != 0.:
                #   Prepare dummy arrays and counter for bad images
                _correlation_index = np.zeros(n_images, dtype=int) - 1
                _correlation_index[reference_dataset_id] = i
                _img_rejected = np.zeros(n_images, dtype=int)
                _obj_rejected = np.zeros(n_objects, dtype=int)
                _n_bad_images = 0

                #   Loop over all images
                for j in range(0, n_images):
                    #   Exclude reference image
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

                        if option == 3:
                            #   Find objects with distances that are smaller
                            #   than the required dcr
                            possible_matches = np.argwhere(d2 <= dcr2)
                            possible_matches = possible_matches.ravel()

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
                                if i not in reference_obj_id or not protect_reference_obj:
                                    #   Mark object as problematic
                                    #   -> counts up
                                    _obj_rejected[i] += 1

                if option != 3:
                    if (_n_bad_images > (1 - expected_bad_image_fraction) * n_images
                            and (i not in reference_obj_id or not protect_reference_obj)):
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

        #   Discard objects that are on not enough images
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

    #   Return in case of option 0 and 3
    if option == 0:
        return index_array, rejected_img, count, rejected_objects
    if option == 3:
        return index_array

    ###
    #   Modify the matches depending on input options.
    #
    if not silent:
        if option == 4:
            terminal_output.print_to_terminal(
                "   Cleaning up output array using magnitudes.",
                indent=indent,
            )
        else:
            if option == 1:
                terminal_output.print_to_terminal(
                    "   Cleaning up output array (option = 1).",
                    indent=indent,
                )
            else:
                terminal_output.print_to_terminal(
                    "   Cleaning up output array (option = 2).",
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
            many_to_one_ids = np.argwhere(index_array[j, :] == i)
            n_multi = len(many_to_one_ids)
            #   All but one of the images in WW must eventually be removed.
            if n_multi > 1:
                #   Mark images that should be rejected.
                if n_multi >= cross_identification_limit and n_images > 2:
                    rejected_img[j] = 1

                if option == 4 and n_images == 2:
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
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 1"
                            f"{style.Bcolors.ENDC}"
                        )

                    #   Find the element with the minimum distance
                    possible_matches = np.argmin(d2)

                #   Delete the minimum element from the
                #   deletion list itself.
                if option == 1:
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
                if option == 2:
                    if len(index_array[j, :]) != (c_save - n_multi):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(index_array[reference_dataset_id, :]) != (c_save - n_multi):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                else:
                    if len(index_array[j, :]) != (c_save - n_multi + 1):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(index_array[reference_dataset_id, :]) != (c_save - n_multi + 1):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                if len(index_array[j, :]) != len(index_array[reference_dataset_id, :]):
                    raise Exception(
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


def correlate_image_series_images(
        image_series: 'analyze.ImageSeries',
        max_pixel_between_objects: float = 3.,
        own_correlation_option: int = 1,
        cross_identification_limit: int = 1,
        reference_obj_ids: list[int] | None = None,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec) -> None:
    """
    Correlate object positions from all stars in an image series to
    identify those objects that are visible on all images

    Parameters
    ----------
    image_series
        Image series of images, e.g., taken in one filter

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_obj_ids
        IDs of the reference objects. The reference objects will not be
        removed from the list of objects.
        Default is ``None``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.
    """
    #   Number of images
    n_images = len(image_series.image_list)

    #   Set proxy image position IDs
    image_ids_arr = np.arange(n_images)

    terminal_output.print_to_terminal(
        f"Correlate results from the images ({image_ids_arr})",
        indent=1,
    )

    #   Get WCS
    current_wcs = image_series.wcs

    #   Extract pixel positions of the objects
    x, y, n_objects = image_series.get_object_positions_pixel()

    # #   Correlate the object positions from the images
    # #   -> find common objects
    correlation_index, new_reference_image_id, rejected_images, _ = correlate_datasets(
        x,
        y,
        current_wcs,
        n_objects,
        n_images,
        reference_dataset_id=image_series.reference_image_id,
        reference_obj_ids=reference_obj_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        protect_reference_obj=protect_reference_obj,
        separation_limit=separation_limit,
        max_pixel_between_objects=max_pixel_between_objects,
        expected_bad_image_fraction=expected_bad_image_fraction,
        own_correlation_option=own_correlation_option,
        cross_identification_limit=cross_identification_limit,
        correlation_method=correlation_method,
    )

    #   Remove "bad" images from image IDs
    image_ids_arr = np.delete(image_ids_arr, rejected_images, 0)

    #   Remove images that are rejected (bad images) during the correlation process.
    image_series.image_list = [image_series.image_list[i] for i in image_ids_arr]
    # image_series.image_list = np.delete(img_list, reject)
    # image_series.nfiles = len(image_ids_arr)
    image_series.reference_image_id = new_reference_image_id

    #   Limit the photometry tables to common objects.
    for j, image in enumerate(image_series.image_list):
        image.photometry = image.photometry[correlation_index[j, :]]


def correlate_image_series(
        observation: 'analyze.Observation', filter_list: list[str] | set[str],
        max_pixel_between_objects: int = 3,
        own_correlation_option: int = 1, cross_identification_limit: int = 1,
        reference_image_series_id: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        force_correlation_calibration_objects: bool = False,
        verbose: bool = False, file_type_plots: str = 'pdf',
        indent: int = 1) -> None:
    """
    Correlate star lists from the stacked images of all filters to find
    those stars that are visible on all images

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        List with filter identifiers.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_image_series_id
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    force_correlation_calibration_objects
        If ``True`` the correlation between the already correlated
        series and the calibration data will be enforced.
        Default is ``False``

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    terminal_output.print_to_terminal(
        "Correlate image series",
        indent=1,
    )

    #   Get image series
    image_series_dict = observation.get_image_series(filter_list)
    image_series_keys = list(image_series_dict.keys())

    #   Get Reference filter
    #   TODO: Check: Is there a better solution?
    reference_filter = list(filter_list)[reference_image_series_id]

    #   Define variables
    n_object_all_images_list = []
    x_pixel_positions_all_images = []
    y_pixel_positions_all_images = []
    wcs_list_image_series = []

    #   Number of objects in each table/image
    reference_obj_ids = []
    for id_series, series in enumerate(image_series_dict.values()):
        wcs_list_image_series.append(series.wcs)

        reference_image_id = series.reference_image_id
        _x = series.image_list[reference_image_id].photometry['x_fit']
        x_pixel_positions_all_images.append(_x)
        y_pixel_positions_all_images.append(
            series.image_list[reference_image_id].photometry['y_fit']
        )
        n_object_all_images_list.append(len(_x))

        #   Check if reference object is set
        if id_series == reference_image_series_id:
            reference_obj_ids = getattr(series, 'variable_id', [])

    #   Max. number of objects
    n_objects_max = np.max(n_object_all_images_list)

    #   Number of image series
    n_series = len(x_pixel_positions_all_images)

    #   Correlate the object positions from the images
    #   -> find common objects
    correlation_index, _, rejected_series, _ = correlate_datasets(
        x_pixel_positions_all_images,
        y_pixel_positions_all_images,
        wcs_list_image_series[reference_image_series_id],
        n_objects_max,
        n_series,
        dataset_type='series',
        reference_dataset_id=reference_image_series_id,
        reference_obj_ids=reference_obj_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        protect_reference_obj=protect_reference_obj,
        separation_limit=separation_limit,
        advanced_cleanup=False,
        max_pixel_between_objects=max_pixel_between_objects,
        expected_bad_image_fraction=expected_bad_image_fraction,
        own_correlation_option=own_correlation_option,
        cross_identification_limit=cross_identification_limit,
        correlation_method=correlation_method,
    )

    #   Remove "bad"/rejected image series
    for series_rejected in rejected_series:
        image_series_dict.pop(image_series_keys[series_rejected])

    #   Limit the photometry tables object_ids to common objects.
    for j, series in enumerate(image_series_dict.values()):
        for image in series.image_list:
            image.photometry = image.photometry[correlation_index[j, :]]

    #   Re-identify position of objects of interest
    objects_of_interest = observation.objects_of_interest
    if objects_of_interest:
        terminal_output.print_to_terminal(
            "Identify objects of interest",
        )

        series = image_series_dict[reference_filter]
        reference_image_id = series.reference_image_id
        identify_object_of_interest_in_dataset(
            series.image_list[reference_image_id].photometry['x_fit'],
            series.image_list[reference_image_id].photometry['y_fit'],
            objects_of_interest,
            reference_filter,
            series.wcs,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            verbose=verbose,
        )

        #   Replicate IDs for the objects of interest
        #   -> This is required, since the identification above is only for the
        #      reference filter / image series
        #   TODO: Is there a better solution?
        for object_ in objects_of_interest:
            id_object = object_.id_in_image_series[reference_filter]
            for filter_ in filter_list:
                if filter_ != list(filter_list)[reference_image_series_id]:
                    object_.id_in_image_series[filter_] = id_object

    #   Check if correlation with calibration data is necessary
    calibration_parameters = getattr(observation, 'calib_parameters', None)

    if calibration_parameters is not None and (calibration_parameters.ids_calibration_objects is None
                                               or force_correlation_calibration_objects):
        calibration_tbl = calibration_parameters.calib_tbl
        column_names = calibration_parameters.column_names
        ra_unit_calibration = calibration_parameters.ra_unit
        dec_unit_calibration = calibration_parameters.dec_unit

        #   Convert coordinates of the calibration stars to SkyCoord object
        calibration_object_coordinates = SkyCoord(
            calibration_tbl[column_names['ra']].data,
            calibration_tbl[column_names['dec']].data,
            unit=(ra_unit_calibration, dec_unit_calibration),
            frame="icrs"
        )

        #   Correlate with calibration stars
        #   -> assumes that calibration stars are already cleared of any reference objects
        #      or variable stars
        calibration_tbl, index_obj_instrument = correlate_with_calibration_objects(
            list(image_series_dict.values())[reference_image_series_id],
            calibration_object_coordinates,
            calibration_tbl,
            filter_list,
            column_names,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            file_type_plots=file_type_plots,
            indent=indent,
        )

        observation.calib_parameters.calib_tbl = calibration_tbl
        observation.calib_parameters.ids_calibration_objects = index_obj_instrument


def correlate_preserve_variable(
        observation: 'analyze.Observation', filter_: str,
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        cross_identification_limit: int = 1, reference_image_id: int = 0,
        n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec, verbose: bool = False,
        plot_reference_only: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Correlate results from all images, while preserving the variable
    star

    Parameters
    ----------
    observation
        Container object with image series and object of interest properties

    filter_
        Filter

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_image_id
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    plot_reference_only
        If True only the starmap for the reference image will
        be created.
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Get image series
    image_series = observation.image_series_dict[filter_]

    #   Get object of interests
    objects_of_interest = observation.objects_of_interest

    #   Find position of the variable star I
    terminal_output.print_to_terminal(
        "Identify the variable star",
        indent=1,
    )

    identify_object_of_interest_in_dataset(
        image_series.image_list[reference_image_id].photometry['x_fit'],
        image_series.image_list[reference_image_id].photometry['y_fit'],
        objects_of_interest,
        filter_,
        image_series.wcs,
        separation_limit=separation_limit,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        verbose=verbose,
    )

    ###
    #   Check if variable star was detected I
    #
    # if n_detections == 0:
    #     raise RuntimeError(
    #         f"{style.Bcolors.FAIL} \tERROR: The variable object was not "
    #         f"detected in the reference image.\n\t-> EXIT{style.Bcolors.ENDC}"
    #     )

    #   Get object of interests ID list
    object_of_interest_ids = observation.get_ids_object_of_interest(filter_=filter_)

    #   Correlate the stellar positions from the different filter
    correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        cross_identification_limit=cross_identification_limit,
        reference_obj_ids=object_of_interest_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        protect_reference_obj=protect_reference_obj,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    #   Find position of the variable star II
    terminal_output.print_to_terminal(
        "Re-identify the variable star",
        indent=1,
    )

    # object_of_interest_ids, n_detections =
    identify_object_of_interest_in_dataset(
        image_series.image_list[image_series.reference_image_id].photometry['x_fit'],
        image_series.image_list[image_series.reference_image_id].photometry['y_fit'],
        objects_of_interest,
        filter_,
        image_series.wcs,
        separation_limit=separation_limit,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        verbose=verbose,
    )

    #   Convert ra & dec to pixel coordinates
    coordinates_objects_of_interest = observation.objects_of_interest_coordinates
    x_position_object, y_position_object = image_series.wcs.all_world2pix(
        coordinates_objects_of_interest.ra,
        coordinates_objects_of_interest.dec,
        0,
    )

    # ###
    # #   Check if variable star was detected II
    # #
    # if n_detections == 0:
    #     raise RuntimeError(
    #         f"{style.Bcolors.FAIL} \tERROR: The variable was not detected "
    #         f"in the reference image.\n\t-> EXIT{style.Bcolors.ENDC}"
    #     )

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        # [x_position_object],
        # [y_position_object],
        x_position_object,
        y_position_object,
        plot_reference_only=plot_reference_only,
        file_type_plots=file_type_plots,
    )


def determine_object_position(
        image: base_utilities.Image, ra_obj: float, dec_obj: float, w: wcs.WCS,
        maximal_pixel_between_objects: float = 3.,
        own_correlation_option: int = 1,
        ra_unit: u.quantity.Quantity = u.hourangle,
        dec_unit: u.quantity.Quantity = u.deg, verbose: bool = False
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Find the image coordinates of a star based on the stellar
    coordinates and the WCS of the image

    Parameters
    ----------
    image
        Object with all image specific properties

    ra_obj
        Right ascension of the object

    dec_obj
        Declination of the object

    w
        WCS infos

    maximal_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    ra_unit
        Right ascension unit
        Default is ``u.hourangle``.

    dec_unit
        Declination unit
        Default is ``u.deg``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    Returns
    -------
    indexes
        Index positions of matched objects in the origins. Is -1 is no
        objects were found.

    count
        Number of times the object has been identified on the image

    x_position_object
        X coordinates of the objects in pixel

    y_position_object
        Y coordinates of the objects in pixel
    """
    #   Make coordinates object
    coord_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
    )

    #   Convert ra & dec to pixel coordinates
    x_position_object, y_position_object = w.all_world2pix(
        coord_obj.ra,
        coord_obj.dec,
        0
    )

    #   Get photometry tabel
    tbl = image.photometry

    #   Number of objects
    count = len(tbl['x_fit'])

    #   Define and fill new arrays to allow correlation
    x_position_all = np.zeros((count, 2))
    y_position_all = np.zeros((count, 2))
    x_position_all[0, 0] = x_position_object
    x_position_all[0:count, 1] = tbl['x_fit']
    y_position_all[0, 0] = y_position_object
    y_position_all[0:count, 1] = tbl['y_fit']

    #   Correlate calibration stars with stars on the image
    indexes, reject, count, reject_obj = correlation_own(
        x_position_all,
        y_position_all,
        maximal_pixel_between_objects,
        option=own_correlation_option,
        silent=not verbose,
    )

    return indexes, count, x_position_object, y_position_object


def correlate_preserve_calibration_objects(
        image_series: 'analyze.ImageSeries', filter_list: list[str],
        calib_method: str = 'APASS',
        magnitude_range: tuple[float, float] = (0., 18.5),
        vizier_dict: dict[str, str] | None = None, calib_file=None,
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        verbose: bool = False, cross_identification_limit: int = 1,
        reference_image_id: int = 0, n_allowed_non_detections_object: int = 1,
        expected_bad_image_fraction: float = 1.0,
        protect_reference_obj: bool = True,
        plot_only_reference_starmap: bool = True,
        correlation_method: str = 'astropy',
        separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        file_type_plots: str = 'pdf') -> None:
    """
    Correlate results from all images, while preserving the calibration
    stars

    Parameters
    ----------
    image_series
        Image series object with all image data taken in a specific
        filter

    filter_list
        Filter list

    calib_method
        Calibration method
        Default is ``APASS``.

    magnitude_range
        Magnitude range
        Default is ``(0.,18.5)``.

    vizier_dict
        Identifiers of catalogs, containing calibration data
        Default is ``None``.

    calib_file
        Path to the calibration file
        Default is ``None``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    cross_identification_limit
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_image_id
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    plot_only_reference_starmap
        If True only the starmap for the reference image will be created.
        Default is ``True``.

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Load calibration data
    calib_tbl, column_names, ra_unit = calibration_data.load_calibration_data_table(
        image_series.image_list[reference_image_id],
        filter_list,
        calibration_method=calib_method,
        magnitude_range=magnitude_range,
        vizier_dict=vizier_dict,
        path_calibration_file=calib_file,
    )

    #   Number of calibration stars
    n_calib_stars = len(calib_tbl)

    if n_calib_stars == 0:
        raise Exception(
            f"{style.Bcolors.FAIL} \nNo match between calibrations stars and "
            f"the\n extracted stars detected. -> EXIT {style.Bcolors.ENDC}"
        )

    #   Find IDs of calibration stars to ensure they are not deleted in
    #   the correlation process
    #
    #   Lists for IDs, and xy coordinates
    calib_stars_ids = []
    calib_x_pixel_positions = []
    calib_y_pixel_positions = []

    #   Loop over all calibration stars
    for k in range(0, n_calib_stars):
        #   Find the calibration star
        id_calib_star, ref_count, x_calib_star, y_calib_star = determine_object_position(
            image_series.image_list[reference_image_id],
            calib_tbl[column_names['ra']].data[k],
            calib_tbl[column_names['dec']].data[k],
            image_series.wcs,
            maximal_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            ra_unit=ra_unit,
            verbose=verbose,
        )
        if verbose:
            terminal_output.print_to_terminal('')

        #   Add ID and coordinates of the calibration star to the lists
        if ref_count != 0:
            calib_stars_ids.append(id_calib_star[1][0])
            calib_x_pixel_positions.append(x_calib_star)
            calib_y_pixel_positions.append(y_calib_star)
    terminal_output.print_to_terminal(
        f"{len(calib_stars_ids):d} matches",
        indent=3,
        style_name='OKBLUE',
    )
    terminal_output.print_to_terminal('')

    #   Correlate the results from all images
    correlate_image_series_images(
        image_series,
        max_pixel_between_objects=max_pixel_between_objects,
        own_correlation_option=own_correlation_option,
        cross_identification_limit=cross_identification_limit,
        reference_obj_ids=calib_stars_ids,
        n_allowed_non_detections_object=n_allowed_non_detections_object,
        expected_bad_image_fraction=expected_bad_image_fraction,
        protect_reference_obj=protect_reference_obj,
        correlation_method=correlation_method,
        separation_limit=separation_limit,
    )

    #   Plot image with the final positions overlaid (final version)
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        calib_x_pixel_positions,
        calib_y_pixel_positions,
        plot_reference_only=plot_only_reference_starmap,
        file_type_plots=file_type_plots,
    )


def correlate_with_calibration_objects(
        image_series: 'analyze.ImageSeries',
        calibration_object_coordinates: SkyCoord,
        calibration_tbl: Table, filter_list: list[str],
        column_names: dict[str, str], correlation_method: str = 'astropy',
        separation_limit: u.Quantity = 2. * u.arcsec,
        max_pixel_between_objects: int = 3, own_correlation_option: int = 1,
        id_object: int | None = None,
        indent: int = 1, file_type_plots: str = 'pdf'
        ) -> tuple[Table, np.ndarray]:
    """
    Correlate observed objects with calibration stars

    Parameters
    ----------
    image_series
        Class with all images of a specific image series

    calibration_object_coordinates
        Coordinates of the calibration objects

    calibration_tbl
        Table with calibration data

    filter_list
        Filter list

    column_names
        Actual names of the columns in calibration_tbl versus
        the internal default names

    correlation_method
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option
        Option for the srcor correlation function
        Default is ``1``.

    id_object
        ID of the object
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``1``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    Returns
    -------
    calibration_tbl_sort
        Sorted table with calibration data

    index_obj_instrument
        Index of the observed stars that correspond to the calibration stars
    """
    #   Pixel positions of the observed object
    reference_image_id = image_series.reference_image_id
    pixel_position_obj_x = image_series.image_list[reference_image_id].photometry['x_fit']
    pixel_position_obj_y = image_series.image_list[reference_image_id].photometry['y_fit']

    #   Pixel positions of calibration object
    pixel_position_cali_x, pixel_position_cali_y = calibration_object_coordinates.to_pixel(image_series.wcs)

    if correlation_method == 'astropy':
        #   Create coordinates object
        object_coordinates = SkyCoord.from_pixel(
            pixel_position_obj_x,
            pixel_position_obj_y,
            image_series.wcs,
        )

        #   Find matches between the datasets
        index_obj_instrument, index_obj_literature, _, _ = matching.search_around_sky(
            object_coordinates,
            calibration_object_coordinates,
            separation_limit,
        )

        n_identified_literature_objs = len(index_obj_literature)

    elif correlation_method == 'own':
        #   Max. number of objects
        n_obj_max = np.max(len(pixel_position_obj_x), len(pixel_position_cali_x))

        #   Define and fill new arrays
        pixel_position_all_x = np.zeros((n_obj_max, 2))
        pixel_position_all_y = np.zeros((n_obj_max, 2))
        pixel_position_all_x[0:len(pixel_position_obj_x), 0] = pixel_position_obj_x
        pixel_position_all_x[0:len(pixel_position_cali_x), 1] = pixel_position_cali_x
        pixel_position_all_y[0:len(pixel_position_obj_y), 0] = pixel_position_obj_y
        pixel_position_all_y[0:len(pixel_position_cali_y), 1] = pixel_position_cali_y

        #   Correlate calibration stars with stars on the image
        correlated_indexes, rejected_images, n_identified_literature_objs, rejected_obj = correlation_own(
            pixel_position_all_x,
            pixel_position_all_y,
            max_pixel_between_objects=max_pixel_between_objects,
            option=own_correlation_option,
        )
        index_obj_instrument = correlated_indexes[0]
        index_obj_literature = correlated_indexes[1]

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own". Got '
            f'{correlation_method} instead.'
        )

    if n_identified_literature_objs == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo calibration star was identified "
            f"-> EXIT {style.Bcolors.ENDC}"
        )
    if n_identified_literature_objs == 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL}\nOnly one calibration star was identified\n"
            "Unfortunately, that is not enough at the moment\n"
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    #   Limit calibration table to common objects
    calibration_tbl_sort = calibration_tbl[index_obj_literature]

    ###
    #   Plots
    #
    #   Make new arrays based on the correlation results
    pixel_position_common_objs_x = pixel_position_obj_x[list(index_obj_instrument)]
    pixel_position_common_objs_y = pixel_position_obj_y[list(index_obj_instrument)]
    index_common_new = np.arange(n_identified_literature_objs)

    #   Add pixel positions and object ids to the calibration table
    calibration_tbl_sort.add_columns(
        [np.intc(index_common_new), pixel_position_common_objs_x, pixel_position_common_objs_y],
        names=['id', 'xcentroid', 'ycentroid']
    )

    calibration_tbl.add_columns(
        [np.arange(0, len(pixel_position_cali_y)), pixel_position_cali_x, pixel_position_cali_y],
        names=['id', 'xcentroid', 'ycentroid']
    )

    #   Plot star map with calibration stars
    if id_object is not None:
        rts = f'calibration - object: {id_object}'
    else:
        rts = 'calibration'
    for filter_ in filter_list:
        if 'mag' + filter_ in column_names:
            p = mp.Process(
                target=plots.starmap,
                args=(
                    image_series.out_path.name,
                    #   Replace with reference image in the future
                    image_series.image_list[0].get_data(),
                    filter_,
                    calibration_tbl,
                ),
                kwargs={
                    'tbl_2': calibration_tbl_sort,
                    'label': 'downloaded calibration stars',
                    'label_2': 'matched calibration stars',
                    'rts': rts,
                    # 'name_object': image_series.object_name,
                    'wcs_image': image_series.wcs,
                    'file_type': file_type_plots,
                }
            )
            p.start()

    terminal_output.print_to_terminal(
        f"{len(calibration_tbl_sort)} calibration stars have been matched to"
        f" observed stars",
        indent=indent + 2,
        style_name='OKBLUE',
    )

    return calibration_tbl_sort, index_obj_instrument
