############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

from .. import style, terminal_output

from astropy.coordinates import SkyCoord, matching
import astropy.units as u


############################################################################
#                           Routines & definitions                         #
############################################################################


def determine_pixel_coordinates_obj_astropy(x_pixel_position_dataset,
                                            y_pixel_position_dataset,
                                            ra_obj, dec_obj, wcs,
                                            ra_unit=u.hourangle,
                                            dec_unit=u.deg,
                                            separation_limit=2. * u.arcsec):
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

        ra_obj                      : `float`
            Right ascension of the object

        dec_obj                     : `float`
            Declination of the object

        wcs                         : `astropy.wcs.WCS`
            WCS info

        ra_unit                     : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit                    : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

        separation_limit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        Returns
        -------
        index_obj                   : `numpy.ndarray`
            Index positions of matched objects in the images. Is -1 is no
            objects were found.

        count                       : `integer`
            Number of times the object has been identified on the image

        obj_pixel_position_x        : `float`
            X coordinates of the objects in pixel

        obj_pixel_position_y        : `float`
            Y coordinates of the objects in pixel
    """
    #   Make coordinates object
    coordinates_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
    )

    #   Convert ra & dec to pixel coordinates
    obj_pixel_position_x, obj_pixel_position_y = wcs.all_world2pix(
        coordinates_obj.ra,
        coordinates_obj.dec,
        0,
    )

    #   Create SkyCoord object for dataset
    coordinates_dataset = SkyCoord.from_pixel(
        x_pixel_position_dataset,
        y_pixel_position_dataset,
        wcs,
    )

    #   Find matches in the dataset
    #   TODO: Check - This seems to fail if more than
    #    one object is within the separation limit.
    mask = coordinates_dataset.separation(coordinates_obj) < separation_limit
    index_obj = np.argwhere(mask).ravel()

    return index_obj, len(index_obj), obj_pixel_position_x, obj_pixel_position_y


def determine_pixel_coordinates_obj_srcor(x_pixel_position_dataset,
                                          y_pixel_position_dataset,
                                          ra_obj, dec_obj, wcs,
                                          max_pixel_between_objects=3,
                                          own_correlation_option=1,
                                          ra_unit=u.hourangle,
                                          dec_unit=u.deg, verbose=False):
    """
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image

        Parameters
        ----------
        x_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in X direction

        y_pixel_position_dataset    : `numpy.ndarray`
            Positions of the objects in Pixel in Y direction

        ra_obj                      : `float`
            Right ascension of the object

        dec_obj                     : `float`
            Declination of the object

        wcs                         : `astropy.wcs.WCS`
            WCS info

        max_pixel_between_objects   : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        own_correlation_option      : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        ra_unit                     : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit                    : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

        verbose                     : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        Returns
        -------
        index_obj            : `numpy.ndarray`
            Index positions of matched objects in the images. Is -1 is no
            objects were found.

        count           : `integer`
            Number of times the object has been identified on the image

        obj_pixel_position_x           : `float`
            X coordinates of the objects in pixel

        obj_pixel_position_x
            Y coordinates of the objects in pixel
    """
    #   Make coordinates object
    coordinates_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
    )

    #   Convert ra & dec to pixel coordinates
    obj_pixel_position_x, obj_pixel_position_x = wcs.all_world2pix(
        coordinates_obj.ra,
        coordinates_obj.dec,
        0,
    )

    #   Number of objects
    n_obj_dataset = len(x_pixel_position_dataset)

    #   Define and fill new arrays to allow correlation
    pixel_position_all_x = np.zeros((n_obj_dataset, 2))
    pixel_position_all_y = np.zeros((n_obj_dataset, 2))
    pixel_position_all_x[0, 0] = obj_pixel_position_x
    pixel_position_all_x[0:n_obj_dataset, 1] = x_pixel_position_dataset
    pixel_position_all_y[0, 0] = obj_pixel_position_x
    pixel_position_all_y[0:n_obj_dataset, 1] = y_pixel_position_dataset

    #   Correlate calibration stars with stars on the image
    index_obj, reject, count, reject_obj = correlation_own(
        pixel_position_all_x,
        pixel_position_all_y,
        max_pixel_between_objects=max_pixel_between_objects,
        option=own_correlation_option,
        silent=not verbose,
    )

    return index_obj, count, obj_pixel_position_x, obj_pixel_position_x


def identify_star_in_dataset(x_pixel_positions, y_pixel_positions, ra_obj,
                             dec_obj, wcs, ra_unit=u.hourangle,
                             dec_unit=u.deg, separation_limit=2. * u.arcsec,
                             max_pixel_between_objects=3,
                             own_correlation_option=1, verbose=False,
                             correlation_method='astropy'):
    """
        Identify a specific star based on its right ascension and declination
         in a dataset of pixel coordinates. Requires a valid WCS.

        Parameters
        ----------
        x_pixel_positions           : `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y_pixel_positions           : `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        ra_obj                      : `float`
            Right ascension of the object

        dec_obj                     : `float`
            Declination of the object

        wcs                         : `astropy.wcs` object
            WCS information

        ra_unit                     : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit                    : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

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
        index_obj                   : `integer`
            Index positions of the object.

        count                       : `integer`
            Number of times the object has been identified on the image

        obj_pixel_position_x        : `float`
            X coordinates of the objects in pixel

        obj_pixel_position_y        : `float`
            Y coordinates of the objects in pixel
    """
    if correlation_method == 'astropy':
        index_obj, count, obj_pixel_position_x, obj_pixel_position_y = determine_pixel_coordinates_obj_astropy(
            x_pixel_positions,
            y_pixel_positions,
            ra_obj,
            dec_obj,
            wcs,
            ra_unit=ra_unit,
            dec_unit=dec_unit,
            separation_limit=separation_limit,
        )

    elif correlation_method == 'own':
        index_obj, count, obj_pixel_position_x, obj_pixel_position_y = determine_pixel_coordinates_obj_srcor(
            x_pixel_positions,
            y_pixel_positions,
            ra_obj,
            dec_obj,
            wcs,
            max_pixel_between_objects=max_pixel_between_objects,
            own_correlation_option=own_correlation_option,
            verbose=verbose,
            ra_unit=ra_unit,
            dec_unit=dec_unit,
        )

        # if verbose:
        #     terminal_output.print_terminal()

        #   Current object ID
        index_obj = index_obj[1]

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own".'
            f'Got {correlation_method} instead.'
        )

    return index_obj, count, obj_pixel_position_x, obj_pixel_position_y


def correlate_datasets(x_pixel_positions, y_pixel_positions, wcs, n_objects,
                       n_images, dataset_type='image', reference_image_id=0,
                       reference_obj_ids=None, protect_reference_obj=True,
                       n_allowed_non_detections_object=1,
                       separation_limit=2. * u.arcsec, advanced_cleanup=True,
                       max_pixel_between_objects=3.,
                       expected_bad_image_fraction=1.0,
                       own_correlation_option=1, cross_identification_limit=1,
                       correlation_method='astropy'):
    """
        Correlate the pixel positions from different dataset such as
        images or image ensembles.

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

        reference_image_id              : `integer`, optional
            ID of the reference image
            Default is ``0``.

        reference_obj_ids               : `list` of `integer` or `None`, optional
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
            reference_image_id=reference_image_id,
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
            reference_image_id=reference_image_id,
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

    #   Calculate new index of the reference image
    shift_id = np.argwhere(rejected_images < reference_image_id)
    new_reference_image_id = reference_image_id - len(shift_id)

    return correlation_index, new_reference_image_id, rejected_images, n_common_objects


def correlation_astropy(x_pixel_positions, y_pixel_positions, wcs,
                        reference_image_id=0, reference_obj_ids=None,
                        expected_bad_image_fraction=1,
                        protect_reference_obj=True,
                        separation_limit=2. * u.arcsec, advanced_cleanup=True):
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

        reference_image_id          : `integer`, optional
            ID of the reference image
            Default is ``0``.

        reference_obj_ids           : `list` of `integer` or None, optional
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
    n_images = len(x_pixel_positions)

    #   Create reference SkyCoord object
    reference_coordinates = SkyCoord.from_pixel(
        x_pixel_positions[reference_image_id],
        y_pixel_positions[reference_image_id],
        wcs,
    )

    #   Prepare index array and fill in values for the reference dataset
    index_array = np.ones(
        (n_images, len(x_pixel_positions[reference_image_id])),
        dtype=int
    )
    index_array *= -1
    index_array[reference_image_id, :] = np.arange(
        len(x_pixel_positions[reference_image_id])
    )

    #   Loop over datasets
    for i in range(0, n_images):
        #   Do nothing for the reference object
        if i != reference_image_id:
            #   Dirty fix: In case of identical positions between the
            #              reference and the current data set,
            #              matching.search_around_sky will fail.
            #              => set reference indexes
            if ((len(x_pixel_positions[i]) == len(x_pixel_positions[reference_image_id])) and
                    (np.all(x_pixel_positions[i] == x_pixel_positions[reference_image_id]) and
                     np.all(y_pixel_positions[i] == y_pixel_positions[reference_image_id]))):
                index_array[i, :] = index_array[reference_image_id, :]
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
            id_reference_obj_in_rejected_objects = np.argwhere(
                rejected_object_ids == reference_obj_ids
            )
            rejected_object_ids = np.delete(
                rejected_object_ids,
                id_reference_obj_in_rejected_objects
            )

        #   Remove "bad" objects
        index_array = np.delete(index_array, rejected_object_ids, 1)

        #   Calculate new reference object position
        #   TODO: Check if this needs to adjusted to account for multiple reference objects
        shift_obj = np.argwhere(rejected_object_ids < reference_obj_ids)
        n_shift = len(shift_obj)
        reference_obj_ids = np.array(reference_obj_ids) - n_shift

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
        mask = n_times_to_rm > 0.02 * len(x_pixel_positions[reference_image_id])
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
            if n_images <= 2:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nReference object only found on "
                    "one or on none image at all. This is not sufficient. "
                    f"=> Exit {style.Bcolors.ENDC}"
                )
            rejected_object_ids = rows_to_rm[1]
            rejected_object_ids = np.unique(rejected_object_ids)
            id_reference_obj_in_rejected_objects = np.argwhere(
                rejected_object_ids == reference_obj_ids
            )
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
                    cross_identification_limit=1, reference_image_id=0,
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

        reference_image_id              : `integer`, optional
            ID of the reference image (e.g., an image).
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
            if x_pixel_positions[i, reference_image_id] != 0.:
                #   Prepare dummy arrays and counter for bad images
                _correlation_index = np.zeros(n_images, dtype=int) - 1
                _correlation_index[reference_image_id] = i
                _img_rejected = np.zeros(n_images, dtype=int)
                _obj_rejected = np.zeros(n_objects, dtype=int)
                _n_bad_images = 0

                #   Loop over all images
                for j in range(0, n_images):
                    #   Exclude reference image
                    if j != reference_image_id:
                        comparison_x_pixel_positions = np.copy(
                            x_pixel_positions[:, j]
                        )
                        comparison_y_pixel_positions = np.copy(
                            y_pixel_positions[:, j]
                        )
                        comparison_x_pixel_positions[comparison_x_pixel_positions == 0] = 9E13
                        comparison_y_pixel_positions[comparison_y_pixel_positions == 0] = 9E13

                        #   Calculate radii
                        d2 = (x_pixel_positions[i, reference_image_id] - comparison_x_pixel_positions) ** 2 \
                            + (y_pixel_positions[i, reference_image_id] - comparison_y_pixel_positions) ** 2

                        if option == 3:
                            #   Find objects with distances that are smaller
                            #   than the required dcr
                            possible_matches = np.argwhere(d2 <= dcr2)
                            possible_matches = possible_matches.ravel()

                            #   Fill ind array
                            n_possible_matches = len(possible_matches)
                            if n_possible_matches:
                                index_array[j, count:count + n_possible_matches] = possible_matches
                                index_array[reference_image_id, count:count + n_possible_matches] = \
                                    _correlation_index[reference_image_id]
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
        x_pixel_positions[rej_obj_tup, reference_image_id] = 0.
        y_pixel_positions[rej_obj_tup, reference_image_id] = 0.

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
        if j == reference_image_id:
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
                            index_array[reference_image_id, many_to_one_ids]
                        ]
                    )
                else:
                    #   Calculate individual distances of the many-to-one
                    #   identifications
                    x_current = x_pixel_positions[i, j]
                    y_current = y_pixel_positions[i, j]
                    x_many = x_pixel_positions[
                        index_array[reference_image_id, many_to_one_ids],
                        reference_image_id
                    ]
                    y_many = y_pixel_positions[
                        index_array[reference_image_id, many_to_one_ids],
                        reference_image_id
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
                    if len(index_array[reference_image_id, :]) != (c_save - n_multi):
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
                    if len(index_array[reference_image_id, :]) != (c_save - n_multi + 1):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                if len(index_array[j, :]) != len(index_array[reference_image_id, :]):
                    raise Exception(
                        f"{style.Bcolors.FAIL}\nLogic error 4"
                        f"{style.Bcolors.ENDC}"
                    )

    #   Determine the indexes of the images to be discarded
    rejected_images = np.argwhere(rejected_img >= 1).ravel()

    #   Set count variable once more
    count = len(index_array[reference_image_id, :])

    if not silent:
        terminal_output.print_to_terminal(
            f"       {len(index_array[reference_image_id, :])} unique "
            f"matches found.",
            indent=indent,
            style_name='OKGREEN',
        )

    return index_array, rejected_images, count, rejected_objects
