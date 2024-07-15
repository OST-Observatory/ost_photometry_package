############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

import typing

from astropy import units as u

if typing.TYPE_CHECKING:
    from . import analyze

from . import calibration_data, utilities
from .. import style, terminal_output

from astropy.coordinates import SkyCoord, matching
import astropy.units as u


############################################################################
#                           Routines & definitions                         #
############################################################################


# def determine_pixel_coordinates_obj_astropy(
#         x_pixel_position_dataset, y_pixel_position_dataset, ra_objects,
#         dec_objects, wcs, ra_unit=u.hourangle, dec_unit=u.deg,
#         separation_limit=2. * u.arcsec):
#     """
#         Find the image coordinates of a star based on the stellar
#         coordinates and the WCS of the image, using astropy matching
#         algorithms.
#
#         Parameters
#         ----------
#         x_pixel_position_dataset    : `numpy.ndarray`
#             Positions of the objects in Pixel in X direction
#
#         y_pixel_position_dataset    : `numpy.ndarray`
#             Positions of the objects in Pixel in Y direction
#
#         ra_objects                  : `string` or `list` of `string`
#             Right ascension of the object
#
#         dec_objects                 : `string` or `list` of `string`
#             Declination of the object
#
#         wcs                         : `astropy.wcs.WCS`
#             WCS info
#
#         ra_unit                     : `astropy.units`, optional
#             Right ascension unit
#             Default is ``u.hourangle``.
#
#         dec_unit                    : `astropy.units`, optional
#             Declination unit
#             Default is ``u.deg``.
#
#         separation_limit            : `astropy.units`, optional
#             Allowed separation between objects.
#             Default is ``2.*u.arcsec``.
#
#         Returns
#         -------
#         index_object                : `numpy.ndarray`
#             Index positions of matched objects in the images. Is -1 is no
#             objects were found.
#
#         count                       : `integer`
#             Number of times the object has been identified on the image
#
#         obj_pixel_position_x        : `float`
#             X coordinates of the objects in pixel
#
#         obj_pixel_position_y        : `float`
#             Y coordinates of the objects in pixel
#     """
#     #   Make coordinates object
#     #   TODO: Check - Replace with SkyCoord from Observation object?
#     coordinates_objects = SkyCoord(
#         ra_objects,
#         dec_objects,
#         unit=(ra_unit, dec_unit),
#         frame="icrs",
#     )
#
#     #   Convert ra & dec to pixel coordinates
#     obj_pixel_position_x, obj_pixel_position_y = wcs.all_world2pix(
#         coordinates_objects.ra,
#         coordinates_objects.dec,
#         0,
#     )
#
#     #   Create SkyCoord object for dataset
#     coordinates_dataset = SkyCoord.from_pixel(
#         x_pixel_position_dataset,
#         y_pixel_position_dataset,
#         wcs,
#     )
#
#     #   Find matches in the dataset
#     index_object_list = []
#     for coordinate_object in coordinates_objects:
#         separation = coordinates_dataset.separation(coordinate_object)
#         mask = separation < separation_limit
#         object_id = np.argwhere(mask).ravel()
#
#         if len(object_id) > 1:
#             #   Note: with an 'object of interest' object a better error
#             #   message would be feasible
#             terminal_output.print_to_terminal(
#                 f"More than one object detected within the separation limit to "
#                 f"the object of interest at the following coordinates "
#                 f"{coordinate_object.ra} {coordinate_object.dec}. Use the "
#                 f"object that is the closest.",
#                 style_name='WARNING',
#             )
#             object_id = np.argmin(separation)
#         if not object_id:
#             terminal_output.print_to_terminal(
#                 f"No object detected within the separation limit to "
#                 f"the object of interest at the following coordinates "
#                 f"{coordinate_object.ra} {coordinate_object.dec}. Set object "
#                 f"ID to None",
#                 style_name='WARNING',
#             )
#             object_id = None
#
#         index_object_list.append(object_id)
#
#     #   TODO: This does not work as intended
#     # if isinstance(ra_objects, list):
#     #     mask = np.zeros(len(coordinates_dataset), dtype=bool)
#     #     for coordinate_object in coordinates_objects:
#     #         separation = coordinates_dataset.separation(coordinate_object)
#     #
#     #         #   Calculate mask of all object closer than ``radius``
#     #         mask = mask | (separation < separation_limit)
#     # #   TODO: Check if this else is necessary
#     # else:
#     #     mask = coordinates_dataset.separation(coordinates_objects) < separation_limit
#     #
#     # index_object = np.argwhere(mask).ravel()
#
#     return index_object_list, len(index_object_list), obj_pixel_position_x, obj_pixel_position_y


def determine_pixel_coordinates_obj_astropy(
        x_pixel_position_dataset, y_pixel_position_dataset,
        objects_of_interest, filter_, wcs, separation_limit=2. * u.arcsec):
    """
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image, using astropy matching
        algorithms.

        Parameters
        ----------
        x_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in X direction

        y_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in Y direction

        objects_of_interest         : `observation.objects_of_interest`
            Object with 'object of interest' properties

        filter_                     : `string`
            Filter identifier

        wcs                         : `astropy.wcs.WCS`
            WCS info


        separation_limit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    """
    #   Create SkyCoord object for dataset
    coordinates_dataset = SkyCoord.from_pixel(
        x_pixel_position_dataset,
        y_pixel_position_dataset,
        wcs,
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


def determine_pixel_coordinates_obj_srcor(
        x_pixel_position_dataset, y_pixel_position_dataset,
        objects_of_interest, filter_, wcs, max_pixel_between_objects=3,
        own_correlation_option=1, verbose=False):
    """
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image

        Parameters
        ----------
        x_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in X direction

        y_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in Y direction

        objects_of_interest         : `observation.objects_of_interest`
            Object with 'object of interest' properties

        filter_                     : `string`
            Filter identifier

        wcs                         : `astropy.wcs.WCS`
            WCS info

        max_pixel_between_objects   : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option      : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        verbose                     : `boolean`, optional
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
        obj_pixel_position_x, obj_pixel_position_y = wcs.all_world2pix(
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


def identify_star_in_dataset(
        x_pixel_positions, y_pixel_positions, objects_of_interest, filter_,
        wcs, separation_limit=2. * u.arcsec, max_pixel_between_objects=3,
        own_correlation_option=1, verbose=False, correlation_method='astropy'):
    """
        Identify a specific star based on its right ascension and declination
        in a dataset of pixel coordinates. Requires a valid WCS.

        Parameters
        ----------
        x_pixel_positions           : `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y_pixel_positions           : `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        objects_of_interest         : `observation.objects_of_interest`
            Object with 'object of interest' properties

        filter_                     : `string`
            Filter identifier

        wcs                         : `astropy.wcs` object
            WCS information

        separation_limit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        max_pixel_between_objects   : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option      : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        verbose                     : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        correlation_method          : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.


        Returns
        -------
        index_obj                   : `numpy.ndarray`
            Index positions of the object.

        count                       : `integer`
            Number of times the object has been identified on the image

        obj_pixel_position_x        : `float`
            X coordinates of the objects in pixel

        obj_pixel_position_y        : `float`
            Y coordinates of the objects in pixel
    """
    if correlation_method == 'astropy':
        determine_pixel_coordinates_obj_astropy(
            x_pixel_positions,
            y_pixel_positions,
            objects_of_interest,
            filter_,
            wcs,
            separation_limit=separation_limit,
        )

    elif correlation_method == 'own':
        determine_pixel_coordinates_obj_srcor(
            x_pixel_positions,
            y_pixel_positions,
            objects_of_interest,
            filter_,
            wcs,
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
        x_pixel_positions, y_pixel_positions, wcs, n_objects, n_images,
        dataset_type='image', reference_dataset_id=0, reference_obj_ids=None,
        protect_reference_obj=True, n_allowed_non_detections_object=1,
        separation_limit=2. * u.arcsec, advanced_cleanup=True,
        max_pixel_between_objects=3., expected_bad_image_fraction=1.0,
        own_correlation_option=1, cross_identification_limit=1,
        correlation_method='astropy'):
    """
        Correlate the pixel positions from different dataset such as
        images or image series.

        Parameters
        ----------
        x_pixel_positions               : `list` or `list` of `lists` with `floats`
            Pixel positions in X direction

        y_pixel_positions               : `list` or `list` of `lists` with `floats`
            Pixel positions in Y direction

        wcs                             : `astropy.wcs.WCS`
            WCS information

        n_objects                       : `integer`
            Number of objects

        n_images                        : `integer`
            Number of images

        dataset_type                    : `string`
            Characterizes the dataset.
            Default is ``image``.

        reference_dataset_id            : `integer`, optional
            ID of the reference dataset
            Default is ``0``.

        reference_obj_ids               : `list` of `integer` or `numpy.ndarray` or `None`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        protect_reference_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        n_allowed_non_detections_object : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``i`.

        separation_limit                : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        advanced_cleanup                : `boolean`, optional
            If ``True`` a multilevel cleanup of the results will be
            attempted. If ``False`` only the minimal necessary removal of
            objects that are not on all datasets will be performed.
            Default is ``True``.

        max_pixel_between_objects       : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        expected_bad_image_fraction     : `float`, optional
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            Default is ``1.0``.

        own_correlation_option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        cross_identification_limit      : `integer`, optional
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        correlation_method              : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.


        Returns
        -------
        correlation_index               : `numpy.ndarray`
            IDs of the correlated objects

        new_reference_image_id          : `integer`, optional
            New ID of the reference image
            Default is ``0``.

        rejected_images                 : `numpy.ndarray`
            IDs of the images that were rejected because of insufficient quality

        n_common_objects                : `integer`
            Number of objects found on all datasets
    """
    if correlation_method == 'astropy':
        #   Astropy version: 2x faster than own
        correlation_index, rejected_images = correlation_astropy(
            x_pixel_positions,
            y_pixel_positions,
            wcs,
            reference_dataset_id=reference_dataset_id,
            reference_obj_ids=reference_obj_ids,
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
        x_pixel_positions, y_pixel_positions, wcs, reference_dataset_id=0,
        reference_obj_ids=None, expected_bad_image_fraction=1,
        protect_reference_obj=True, separation_limit=2. * u.arcsec,
        advanced_cleanup=True):
    """
        Correlation based on astropy matching algorithm

        Parameters
        ----------
        x_pixel_positions           : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y_pixel_positions           : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        wcs                         : `astropy.wcs ` object
            WCS information

        reference_dataset_id        : `integer`, optional
            ID of the reference dataset
            Default is ``0``.

        reference_obj_ids           : `list` of `integer` or `numpy.ndarray` or None, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        expected_bad_image_fraction : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``1``.

        protect_reference_obj       : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        separation_limit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        advanced_cleanup            : `boolean`, optional
            If ``True`` a multilevel cleanup of the results will be
            attempted. If ``False`` only the minimal necessary removal of
            objects that are not on all datasets will be performed.
            Default is ``True``.

        Returns
        -------
        index_array                     : `numpy.ndarray`
            IDs of the correlated objects

        rejected_images                 : `numpy.ndarray`
            IDs of the images that were rejected because of insufficient quality
    """
    #   Sanitize reference object
    if reference_obj_ids is None:
        reference_obj_ids = []

    #   Number of datasets/images
    n_datasets = len(x_pixel_positions)

    #   Create reference SkyCoord object
    reference_coordinates = SkyCoord.from_pixel(
        x_pixel_positions[reference_dataset_id],
        y_pixel_positions[reference_dataset_id],
        wcs,
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
                    wcs,
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
        ref_is_in = np.isin(rejected_object_ids, reference_obj_ids)

        #   If YES remove reference objects from the "bad" objects
        if protect_reference_obj and np.any(ref_is_in):
            id_difference = rejected_object_ids.reshape(rejected_object_ids.size, 1) - reference_obj_ids
            id_reference_obj_in_rejected_objects = np.argwhere(
                id_difference == 0.
            )[:,0]
            rejected_object_ids = np.delete(
                rejected_object_ids,
                id_reference_obj_in_rejected_objects
            )

        #   Remove "bad" objects
        index_array = np.delete(index_array, rejected_object_ids, 1)

        #   Calculate new reference object position
        #   TODO: Check if this needs to be adjusted to account for multiple reference objects -> Done?
        if not isinstance(reference_obj_ids, np.ndarray):
            reference_obj_ids = np.array(reference_obj_ids)
        for index, reference_obj_id in np.ndenumerate(reference_obj_ids):
            object_shift = np.argwhere(rejected_object_ids < reference_obj_id)
            n_shift = len(object_shift)
            reference_obj_ids[index] = reference_obj_id - n_shift

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
        ref_is_in = np.isin(rows_to_rm[1], reference_obj_ids)

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
            id_difference = rejected_object_ids.reshape(rejected_object_ids.size, 1) - reference_obj_ids
            id_reference_obj_in_rejected_objects = np.argwhere(
                id_difference == 0.
            )[:,0]
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


def correlation_own(x_pixel_positions, y_pixel_positions,
                    max_pixel_between_objects=3.,
                    expected_bad_image_fraction=1.0,
                    cross_identification_limit=1, reference_dataset_id=0,
                    reference_obj_id=None,
                    n_allowed_non_detections_object=1, indent=1, option=None,
                    magnitudes=None, silent=False,
                    protect_reference_obj=True):
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
        x_pixel_positions               : `numpy.ndarray`

        y_pixel_positions               : `numpy.ndarray`
            Arrays of x and y coordinates (several columns each). The
            following syntax is expected: x[array of source
            positions]. The program marches through the columns
            element by element, looking for the closest match.

        max_pixel_between_objects       : `float`, optional
            Critical radius outside which correlations are rejected,
            but see 'option' below.
            Default is ````.

        expected_bad_image_fraction     : `float`, optional
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            positions.
            Default is ``1.0``.

        cross_identification_limit      : `integer`, optional
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        reference_dataset_id            : `integer`, optional
            ID of the reference dataset (e.g., an image).
            Default is ``0``.

        reference_obj_id                : `integer`, optional
            Ids of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        n_allowed_non_detections_object : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``1``.

        indent                          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.

        option                          : `integer`, optional
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

        magnitudes                      : `numpy.ndarray`, optional
            An array of stellar magnitudes corresponding to x and y.
            If magnitude is supplied, the brightest objects within
            'max_pixel_between_objects' is taken as a match. The option keyword
            is set to 4 internally.
            Default is ``None``.

        silent                          : `boolean`, optional
            Suppresses output if True.
            Default is ``False``.

        protect_reference_obj           : `boolean`, optional
            Also reference objects will be rejected if Falls.
            Default is ``True``.

        Returns
        -------
        index_array                     : `numpy.ndarray`
            Array of index positions of matched objects in the images,
            set to -1 if no matches are found.

        rejected_images                 : `numpy.ndarray`
            Vector with indexes of all images which should be removed

        count                           : `integer`
            Integer giving number of matches returned

        rejected_objects                : `numpy.ndarray`
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
    #   are on not enough images
    rejected_objects = 0
    for z in range(0, 2):
        #    Prepare index and rejected_images arrays
        #       <- arbitrary * 10 to allow for multi identifications (option 3)
        index_array = np.zeros((n_images, n_objects * 10), dtype=int) - 1
        rejected_img = np.zeros(n_images, dtype=int)
        rejected_obj = np.zeros(n_objects, dtype=int)
        #   Initialize counter of mutual sources
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
                        d2 = (x_pixel_positions[i, reference_dataset_id] - comparison_x_pixel_positions) ** 2 \
                             + (y_pixel_positions[i, reference_dataset_id] - comparison_y_pixel_positions) ** 2

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
        rejected_images = -1
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
        image_series: 'analyze.ImageSeries', max_pixel_between_objects=3., own_correlation_option=1,
        cross_identification_limit=1, reference_obj_ids=None,
        n_allowed_non_detections_object=1, expected_bad_image_fraction=1.0,
        protect_reference_obj=True, correlation_method='astropy',
        separation_limit=2. * u.arcsec):
    """
        Correlate object positions from all stars in an image series to
        identify those objects that are visible on all images

        Parameters
        ----------
        image_series
            Image series of images, e.g., taken in one filter

        max_pixel_between_objects       : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        cross_identification_limit      : `integer`, optional
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        reference_obj_ids               : `list` of `integer` or `numpy.ndarray` or None, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        n_allowed_non_detections_object : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``i`.

        expected_bad_image_fraction     : `float`, optional
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            Default is ``1.0``.

        protect_reference_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correlation_method              : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        separation_limit                : `astropy.units`, optional
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
    wcs = image_series.wcs

    #   Extract pixel positions of the objects
    x, y, n_objects = image_series.get_object_positions_pixel()

    # #   Correlate the object positions from the images
    # #   -> find common objects
    correlation_index, new_reference_image_id, rejected_images, _ = correlate_datasets(
        x,
        y,
        wcs,
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
        observation: 'analyze.Observation', filter_list, max_pixel_between_objects=3.,
        own_correlation_option=1, cross_identification_limit=1,
        reference_image_series_id=0, n_allowed_non_detections_object=1,
        expected_bad_image_fraction=1.0, protect_reference_obj=True,
        correlation_method='astropy', separation_limit: u.quantity.Quantity = 2. * u.arcsec,
        force_correlation_calibration_objects=False, reference_image_id=0,
        verbose=False, indent=1):
    """
    Correlate star lists from the stacked images of all filters to find
    those stars that are visible on all images

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list                             : `list` or `set` of `string`
        List with filter identifiers.

    max_pixel_between_objects               : `float`, optional
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option                  : `integer`, optional
        Option for the srcor correlation function
        Default is ``1``.

    cross_identification_limit              : `integer`, optional
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_image_series_id                   : `integer`, optional
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object         : `integer`, optional
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction             : `float`, optional
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj                   : `boolean`, optional
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    correlation_method                      : `string`, optional
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    force_correlation_calibration_objects   : `boolean`, optional
        If ``True`` the correlation between the already correlated
        series and the calibration data will be enforced.
        Default is ``False``

    reference_image_id                      : `integer`, optional
        ID of the reference image
        Default is ``0``.

    verbose                                 : `boolean`, optional
        If True additional output will be printed to the command line.
        Default is ``False``.

    indent                              : `integer`, optional
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

        #   TODO: Check if this is necessary
        #   TODO: Remove loop -> replace with image_series_dict[filter_list[reference_image_series_id]]
        # for id_series, series in enumerate(image_series_dict.values()):
        #     if id_series == reference_image_series_id:
        series = image_series_dict[reference_filter]
        identify_star_in_dataset(
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
        #   TODO: Check if there is a better solution
        for object_ in objects_of_interest:
            id_object = object_.id_in_image_series[reference_filter]
            for filter_ in filter_list:
                if filter_ != list(filter_list)[reference_image_series_id]:
                    object_.id_in_image_series[filter_] = id_object

    #   Check if correlation with calibration data is necessary
    calibration_parameters = getattr(observation, 'calib_parameters', None)

    if calibration_parameters is not None and (calibration_parameters.inds is None
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
        calibration_tbl, index_obj_instrument = calibration_data.correlate_with_calibration_objects(
            list(image_series_dict.values())[reference_image_series_id],
            calibration_object_coordinates,
            calibration_tbl,
            filter_list,
            column_names,
            correlation_method=correlation_method,
            separation_limit=separation_limit,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            reference_image_id=reference_image_id,
            indent=indent,
        )

        observation.calib_parameters.calib_tbl = calibration_tbl
        observation.calib_parameters.inds = index_obj_instrument


def correlate_preserve_variable(
        observation: 'analyze.Observation', filter_,
        max_pixel_between_objects=3., own_correlation_option=1,
        cross_identification_limit=1, reference_image_id=0,
        n_allowed_non_detections_object=1, expected_bad_image_fraction=1.0,
        protect_reference_obj=True, correlation_method='astropy',
        separation_limit=2. * u.arcsec, verbose=False,
        plot_reference_only=True) -> None:
    """
        Correlate results from all images, while preserving the variable
        star

        Parameters
        ----------
        observation
            Container object with image series and object of interest properties

        filter_                         : `string`
            Filter

        max_pixel_between_objects       : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        cross_identification_limit      : `integer`, optional
            Cross-identification limit between multiple objects in the current
            image and one object in the reference image. The current image is
            rejected when this limit is reached.
            Default is ``1``.

        reference_image_id              : `integer`, optional
            ID of the reference image
            Default is ``0``.

        n_allowed_non_detections_object : `integer`, optional
            Maximum number of times an object may not be detected in an image.
            When this limit is reached, the object will be removed.
            Default is ``i`.

        expected_bad_image_fraction     : `float`, optional
            Fraction of low quality images, i.e. those images for which a
            reduced number of objects with valid source positions are expected.
            Default is ``1.0``.

        protect_reference_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correlation_method              : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        separation_limit                : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose                         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        plot_reference_only             : `boolean`, optional
            If True only the starmap for the reference image will
            be created.
            Default is ``True``.
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

    # object_of_interest_ids, n_detections, _, _ =
    identify_star_in_dataset(
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
    identify_star_in_dataset(
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
    )


#   TODO: Check were this is used and if it is still functional, rename
def correlate_preserve_calibration_objects(
        image_series: 'analyze.ImageSeries', filter_list: list[str, str],
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
        separation_limit: u.quantity.Quantity = 2. * u.arcsec) -> None:
    """
    Correlate results from all images, while preserving the calibration
    stars

    Parameters
    ----------
    image_series
        Image series object with all image data taken in a specific
        filter

    filter_list                     : `list` with `strings`
        Filter list

    calib_method                    : `string`, optional
        Calibration method
        Default is ``APASS``.

    magnitude_range                 : `tuple` or `float`, optional
        Magnitude range
        Default is ``(0.,18.5)``.

    vizier_dict
        Identifiers of catalogs, containing calibration data
        Default is ``None``.

    calib_file                      : `string`, optional
        Path to the calibration file
        Default is ``None``.

    max_pixel_between_objects       : `float`, optional
        Maximal distance between two objects in Pixel
        Default is ``3``.

    own_correlation_option          : `integer`, optional
        Option for the srcor correlation function
        Default is ``1``.

    verbose                         : `boolean`, optional
        If True additional output will be printed to the command line.
        Default is ``False``.

    cross_identification_limit      : `integer`, optional
        Cross-identification limit between multiple objects in the current
        image and one object in the reference image. The current image is
        rejected when this limit is reached.
        Default is ``1``.

    reference_image_id              : `integer`, optional
        ID of the reference image
        Default is ``0``.

    n_allowed_non_detections_object : `integer`, optional
        Maximum number of times an object may not be detected in an image.
        When this limit is reached, the object will be removed.
        Default is ``i`.

    expected_bad_image_fraction     : `float`, optional
        Fraction of low quality images, i.e. those images for which a
        reduced number of objects with valid source positions are expected.
        Default is ``1.0``.

    protect_reference_obj           : `boolean`, optional
        If ``False`` also reference objects will be rejected, if they do
        not fulfill all criteria.
        Default is ``True``.

    plot_only_reference_starmap     : `boolean`, optional
        If True only the starmap for the reference image will be created.
        Default is ``True``.

    correlation_method              : `string`, optional
        Correlation method to be used to find the common objects on
        the images.
        Possibilities: ``astropy``, ``own``
        Default is ``astropy``.

    separation_limit                : `astropy.units`, optional
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.
    """
    ###
    #   Load calibration data
    #
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

    ###
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
        #   TODO: Fix this
        id_calib_star, ref_count, x_calib_star, y_calib_star = posi_obj_srcor_img(
            image_series.image_list[reference_image_id],
            calib_tbl[column_names['ra']].data[k],
            calib_tbl[column_names['dec']].data[k],
            image_series.wcs,
            dcr=max_pixel_between_objects,
            option=own_correlation_option,
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
    terminal_output.print_to_terminal()

    ###
    #   Correlate the results from all images
    #
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

    ###
    #   Plot image with the final positions overlaid (final version)
    #
    utilities.prepare_and_plot_starmap_from_image_series(
        image_series,
        calib_x_pixel_positions,
        calib_y_pixel_positions,
        plot_reference_only=plot_only_reference_starmap,
    )
