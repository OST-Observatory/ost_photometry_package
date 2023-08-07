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


#   TODO: Unify the code in the following 4 functions.
def posi_obj_astropy(xs, ys, ra_obj, dec_obj, w, ra_unit=u.hourangle,
                     dec_unit=u.deg, seplimit=2. * u.arcsec):
    """
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image, using astropy matching
        algorithms.

        Parameters
        ----------
        xs              : `numpy.ndarray`
            Positions of the objects in Pixel in X direction

        ys              : `numpy.ndarray`
            Positions of the objects in Pixel in Y direction

        ra_obj          : `float`
            Right ascension of the object

        dec_obj         : `float`
            Declination of the object

        w               : `astropy.wcs.WCS`
            WCS infos

        ra_unit         : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit        : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        Returns
        -------
        inds            : `numpy.ndarray`
            Index positions of matched objects in the origins. Is -1 is no
            objects were found.

        count           : `integer`
            Number of times the object has been identified on the image

        x_obj           : `float`
            X coordinates of the objects in pixel

        y_obj
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
    x_obj, y_obj = w.all_world2pix(coord_obj.ra, coord_obj.dec, 0)

    #   Create SkyCoord object for dataset
    coords_ds = SkyCoord.from_pixel(xs, ys, w)

    #   Find matches in the dataset
    dist_mask = coords_ds.separation(coord_obj) < seplimit
    id_ds = np.argwhere(dist_mask).ravel()

    return id_ds, len(id_ds), x_obj, y_obj


def posi_obj_srcor(xs, ys, ra_obj, dec_obj, w, dcr=3, option=1,
                   ra_unit=u.hourangle, dec_unit=u.deg, verbose=False):
    """
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image

        Parameters
        ----------
        xs              : `numpy.ndarray`
            Positions of the objects in Pixel in X direction

        ys              : `numpy.ndarray`
            Positions of the objects in Pixel in Y direction

        ra_obj          : `float`
            Right ascension of the object

        dec_obj         : `float`
            Declination of the object

        w               : `astropy.wcs.WCS`
            WCS infos

        dcr             : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        ra_unit         : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit        : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        Returns
        -------
        inds            : `numpy.ndarray`
            Index positions of matched objects in the origins. Is -1 is no
            objects were found.

        count           : `integer`
            Number of times the object has been identified on the image

        x_obj           : `float`
            X coordinates of the objects in pixel

        y_obj
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
    x_obj, y_obj = w.all_world2pix(coord_obj.ra, coord_obj.dec, 0)

    #   Number of objects
    count = len(xs)

    #   Define and fill new arrays to allow correlation
    xall = np.zeros((count, 2))
    yall = np.zeros((count, 2))
    xall[0, 0] = x_obj
    xall[0:count, 1] = xs
    yall[0, 0] = y_obj
    yall[0:count, 1] = ys

    #   Correlate calibration stars with stars on the image
    inds, reject, count, reject_obj = newsrcor(
        xall,
        yall,
        dcr=dcr,
        option=option,
        silent=not verbose,
    )

    return inds, count, x_obj, y_obj


def identify_star_in_dataset(x, y, ra_obj, dec_obj, w, ra_unit=u.hourangle,
                             dec_unit=u.deg, seplimit=2. * u.arcsec, dcr=3, option=1,
                             verbose=False, correl_method='astropy'):
    """
        Identify a specific star based on its right ascension and declination
         in a dataset of pixel coordinates. Requires a valid WCS.

        Parameters
        ----------
        x                   : `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y                   : `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        ra_obj              : `float`
            Right ascension of the object

        dec_obj             : `float`
            Declination of the object

        w                   : `astropy.wcs` object
            WCS information

        ra_unit             : `astropy.units`, optional
            Right ascension unit
            Default is ``u.hourangle``.

        dec_unit            : `astropy.units`, optional
            Declination unit
            Default is ``u.deg``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.


        Returns
        -------
        variable_id     : `integer`
            Index positions of the object.

        count           : `integer`
            Number of times the object has been identified on the image

        x_obj           : `float`
            X coordinates of the objects in pixel

        y_obj           : `float`
            Y coordinates of the objects in pixel
    """
    if correl_method == 'astropy':
        variable_id, count, x_obj, y_obj = posi_obj_astropy(
            x,
            y,
            ra_obj,
            dec_obj,
            w,
            ra_unit=ra_unit,
            dec_unit=dec_unit,
            seplimit=seplimit,
        )

    elif correl_method == 'own':
        inds_obj, count, x_obj, y_obj = posi_obj_srcor(
            x,
            y,
            ra_obj,
            dec_obj,
            w,
            dcr=dcr,
            option=option,
            verbose=verbose,
            ra_unit=ra_unit,
            dec_unit=dec_unit,
        )

        # if verbose:
        #     terminal_output.print_terminal()

        #   Current object ID
        variable_id = inds_obj[1]

    else:
        raise ValueError(f'The correlation method needs to either "astropy" or "own". Got {correl_method} instead.')

    return variable_id, count, x_obj, y_obj


def correlate_datasets(x, y, w, n_objects, n_images, dataset_type='image',
                       ref_ori=0, reference_obj=None, nmissed=1, s_ref_obj=True,
                       seplimit=2. * u.arcsec, cleanup_advanced=True,
                       dcr=3., bfrac=1.0, option=1, maxid=1,
                       correl_method='astropy'):
    """
        Correlate the pixel positions from different dataset such as
        images or image ensembles.

        Parameters
        ----------
        x                   : `list` or `list` of `lists` with `floats`
            Pixel positions in X direction

        y                   : `list` or `list` of `lists` with `floats`
            Pixel positions in Y direction

        w                   : `astropy.wcs.WCS`
            WCS information

        n_objects           : `integer`
            Number of objects

        n_images           : `integer`
            Number of images

        dataset_type        : `string`
            Characterizes the dataset.
            Default is ``image``.

        ref_ori             : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        reference_obj             : `list` of `integer` or `None`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        s_ref_obj           : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        cleanup_advanced    : `boolean`, optional
            If ``True`` a multilevel cleanup of the results will be
            attempted. If ``False`` only the minimal necessary removal of
            objects that are not on all datasets will be performed.
            Default is ``True``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.


        Returns
        -------
        ind_sr             : `numpy.ndarray`
            IDs of the correlated objects

        ref_ori_new        : `integer`, optional
            New ID of the reference origin
            Default is ``0``.

        reject              : `numpy.ndarray`
            IDs of the images that were rejected because of insufficient quality

        count               : `integer`
            Number of objects found on all datasets
    """
    if correl_method == 'astropy':
        #   Astropy version: 2x faster than own
        ind_sr, reject = astropycor(
            x,
            y,
            w,
            reference_image_id=ref_ori,
            reference_obj=reference_obj,
            nmissed=nmissed,
            protect_reference_obj=s_ref_obj,
            seplimit=seplimit,
            cleanup_advanced=cleanup_advanced,
        )
        count = len(ind_sr[0])

    elif correl_method == 'own':
        #   'Own' correlation method requires positions to be in a numpy array
        xall = np.zeros((n_objects, n_images))
        yall = np.zeros((n_objects, n_images))

        for i in range(0, n_images):
            xall[0:len(x[i]), i] = x[i]
            yall[0:len(y[i]), i] = y[i]

        #   Own version based on srcor from the IDL Astro Library
        ind_sr, reject, count, rej_obj = newsrcor(
            xall,
            yall,
            dcr=dcr,
            bfrac=bfrac,
            option=option,
            maxid=maxid,
            reference_image_id=ref_ori,
            reference_obj=reference_obj,
            nmissed=nmissed,
            protect_reference_obj=s_ref_obj,
        )
    else:
        raise ValueError(
            f'{style.Bcolors.FAIL}Correlation method not known. Expected: '
            f'"own" or astropy, but got "{correl_method}"{style.Bcolors.ENDC}'
        )

    ###
    #   Print correlation result or raise error if not enough common
    #   objects were detected
    #
    if count == 1:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nOnly one common object "
            f"found! {style.Bcolors.ENDC}"
        )
    elif count == 0:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo common objects "
            f"found!{style.Bcolors.ENDC}"
        )
    else:
        terminal_output.print_to_terminal(
            f"{count} objects identified on all {dataset_type}s",
            indent=2,
        )

    nbad = len(reject)
    if nbad > 0:
        terminal_output.print_to_terminal(
            f"{nbad} images do not meet the criteria -> removed",
            indent=2,
        )
    if nbad > 1:
        terminal_output.print_to_terminal(
            f"Rejected {dataset_type} IDs: {reject}",
            indent=2,
        )
    elif nbad == 1:
        terminal_output.print_to_terminal(
            f"ID of the rejected {dataset_type}: {reject}",
            indent=2,
        )
    terminal_output.print_to_terminal('')

    ###
    #   Post process correlation results
    #

    #   Remove "bad" images from index array
    #   (only necessary for 'own' method)
    if correl_method == 'own':
        ind_sr = np.delete(ind_sr, reject, 0)

    #   Calculate new index of the reference origin
    shift_id = np.argwhere(reject < ref_ori)
    ref_ori_new = ref_ori - len(shift_id)

    return ind_sr, ref_ori_new, reject, count


def astropycor(x, y, w, reference_image_id=0, reference_obj=None, nmissed=1,
               protect_reference_obj=True, seplimit=2. * u.arcsec,
               cleanup_advanced=True):
    """
        Correlation based on astropy matching algorithm

        Parameters
        ----------
        x                       : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y                       : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        w                       : `astropy.wcs ` object
            WCS information

        reference_image_id      : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        reference_obj           : `list` of `integer` or None, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        nmissed                 : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``1``.

        protect_reference_obj   : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        seplimit                : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        cleanup_advanced        : `boolean`, optional
            If ``True`` a multilevel cleanup of the results will be
            attempted. If ``False`` only the minimal necessary removal of
            objects that are not on all datasets will be performed.
            Default is ``True``.
    """
    #   Sanitize reference object
    if reference_obj is None:
        reference_obj = []

    #   Number of datasets/images
    n = len(x)

    #   Create reference SkyCoord object
    coords_ref = SkyCoord.from_pixel(
        x[reference_image_id],
        y[reference_image_id],
        w,
    )

    #   Prepare index array and fill in values for the reference dataset
    idarray = np.ones((n, len(x[reference_image_id])), dtype=int)
    idarray *= -1
    idarray[reference_image_id, :] = np.arange(len(x[reference_image_id]))

    #   Loop over datasets
    for i in range(0, n):
        #   Do nothing for the reference object
        if i != reference_image_id:
            #   Dirty fix: In case of identical positions between the
            #              reference and the current data set,
            #              matching.search_around_sky will fail.
            #              => set reference indexes
            if ((len(x[i]) == len(x[reference_image_id])) and
                    (np.all(x[i] == x[reference_image_id]) and np.all(y[i] == y[reference_image_id]))):
                idarray[i, :] = idarray[reference_image_id, :]
            else:
                #   Create coordinates object
                coords = SkyCoord.from_pixel(
                    x[i],
                    y[i],
                    w,
                )

                #   Find matches between the datasets
                id_ref, id_current, d2ds, d3ds = matching.search_around_sky(
                    coords_ref,
                    coords,
                    seplimit,
                )

                #   Fill ID array
                idarray[i, id_ref] = id_current

    ###
    #   Cleanup: Remove "bad" objects and datasets
    #

    #   1. Remove bad objects (preburner) -> Useful to remove bad objects
    #                                        that may spoil the correct
    #                                        identification of bad datasets.
    if cleanup_advanced:
        #   Identify objects that were not identified in all datasets
        rowsrm = np.where(idarray == -1)

        #   Reduce to unique objects
        unique_obj, count_obj = np.unique(rowsrm[1], return_counts=True)

        #   Identify objects that are not in >= "nmissed" datasets
        rej_obj_id = np.argwhere(count_obj >= nmissed)
        rej_obj = unique_obj[rej_obj_id].flatten()

        #   Check if reference objects are within the "bad" objects
        ref_isin = np.isin(rej_obj, reference_obj)

        #   If YES remove reference objects from the "bad" objects
        if protect_reference_obj and np.any(ref_isin):
            reference_obj_id = np.argwhere(rej_obj == reference_obj)
            rej_obj = np.delete(rej_obj, reference_obj_id)

        #   Remove "bad" objects
        idarray = np.delete(idarray, rej_obj, 1)

        #   Calculate new reference object position
        shift_obj = np.argwhere(rej_obj < reference_obj)
        n_shift = len(shift_obj)
        reference_obj = np.array(reference_obj) - n_shift

        #   2. Remove bad images

        #   Identify objects that were not identified in all datasets
        rowsrm = np.where(idarray == -1)

        #   Reduce to unique objects
        unique_ori, count_ori = np.unique(rowsrm[0], return_counts=True)

        #   Create mask -> Identify all datasets as bad that contain less
        #                  than 90% of all objects from the reference image.
        mask = count_ori > 0.02 * len(x[reference_image_id])
        rej_ori = unique_ori[mask]

        #   Remove those datasets
        idarray = np.delete(idarray, rej_ori, 0)

    else:
        rej_ori = np.array([], dtype=int)

    #   3. Remove remaining objects that are not on all datasets
    #      (afterburner)

    #   Identify objects that were not identified in all datasets
    rowsrm = np.where(idarray == -1)

    if protect_reference_obj:
        #   Check if reference objects are within the "bad" objects
        ref_isin = np.isin(rowsrm[1], reference_obj)

        #   If YES remove reference objects from "bad" objects and remove
        #   the datasets on which they were not detected instead.
        if np.any(ref_isin):
            if n <= 2:
                raise RuntimeError(
                    f"{style.Bcolors.FAIL} \nReference object only found one "
                    "or on no image at all. This is not sufficient. "
                    f"=> Exit {style.Bcolors.ENDC}"
                )
            rej_obj = rowsrm[1]
            rej_obj = np.unique(rej_obj)
            reference_obj_id = np.argwhere(rej_obj == reference_obj)
            rej_obj = np.delete(rej_obj, reference_obj_id)

            #   Remove remaining bad objects
            idarray = np.delete(idarray, rej_obj, 1)

            #   Remove datasets
            rowsrm = np.where(idarray == -1)
            rej_ori_two = np.unique(rowsrm[0])
            idarray = np.delete(idarray, rej_ori_two, 0)

            rej_ori_two_old = []
            for el_two in rej_ori_two:
                for el_one in rej_ori:
                    if el_one <= el_two:
                        el_two += 1
                rej_ori_two_old.append(el_two)

            rej_ori = np.concatenate((rej_ori, np.array(rej_ori_two_old)))

            return idarray, rej_ori

    #   Remove bad objects
    idarray = np.delete(idarray, rowsrm[1], 1)

    return idarray, rej_ori


def newsrcor(x, y, dcr=3., bfrac=1.0, maxid=1, reference_image_id=0, reference_obj=None,
             nmissed=1, indent=1, option=None, magnitude=None,
             silent=False, protect_reference_obj=True):
    """
        Correlate source positions from several origins (e.g., different images)

        Source matching is done by finding objects within a specified
        radius. The code is adapted from the standard srcor routine from
        the IDL Astronomy User's Library. The normal srcor routine was
        extended to fit the requirements of the C7 experiment within the
        astrophysics lab course at Potsdam University.

        SOURCE: Adapted from the IDL Astro Libary

        Parameters
        ----------
        x                       : `numpy.ndarray`

        y                       : `numpy.ndarray`
            Arrays of x and y coordinates (several columns each). The
            following syntax is expected: x[array of source
            positions]. The program marches through the columns
            element by element, looking for the closest match.

        dcr                     : `float`, optional
            Critical radius outside which correlations are rejected,
            but see 'option' below.
            Default is ````.

        bfrac                   : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins (columns in x and y), for which it is expected to
            find a reduced number of objects with valid source
            positions.
            Default is ``1.0``.

        maxid                   : `integer`, optional
            Max. number of allowed identical cross identifications
            between objects from a specific origin (columns in x and y)
            and objects from the origin with the id 'refORI'. The origin
            will be rejected, if this limit is reached.
            Default is ``1``.

        reference_image_id      : `integer`, optional
            ID of the reference origin (e.g., an image).
            Default is ``0``.

        reference_obj           : `integer`, optional
            Ids of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``None``.

        nmissed                 : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``1``.

        indent                  : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.

        option                  : `integer`, optional
            Changes behavior of the program & description of output
            lists slightly, as follows:
              OPTION=0 | left out
                    For each object of the origin 'refORI' the closest match
                    from all other origins is found, but if none is found within
                    the distance of 'dcr', the match is thrown out. Thus, the
                    index of that object will not appear in the 'ind' output
                    array.
              OPTION=1
                    Forces the output mapping to be one-to-one.  OPTION=0
                    results, in general, in a many-to-one mapping from the
                    origin 'refORI' to the all other origins. Under OPTION=1, a
                    further processing step is performed to keep only the
                    minimum-distance match, whenever an entry from the origin
                    'refORI' appears more than once in the initial mapping.
                    Caution: The entries that exceed the distance of the
                             minimum-distance match will be removed from all
                             origins. Hence, selection of 'refORI' matters.
              OPTION=2
                    Same as OPTION=1, except that all entries which appears more
                    than once in the initial mapping will be removed from all
                    origins independent of distance.
              OPTION=3
                    All matches that are within 'dcr' are returned
            Default is ``None``.

        magnitude               : `nump.ndarray`, optional
            An array of stellar magnitudes corresponding to x and y.
            If magnitude is supplied, the brightest objects within 'dcr'
            is taken as a match. The option keyword is set to 4
            internally.
            Default is ``None``.

        silent                  : `boolean`, optional
            Suppresses output if True.
            Default is ``False``.

        protect_reference_obj   : `boolean`, optional
            Also reference objects will be rejected if Falls.
            Default is ``True``.

        Returns
        -------
        ind                     : `numpy.ndarray`
            Array of index positions of matched objects in the origins,
            set to -1 if no matches are found.

        reject                  : `numpy.ndarray`
            Vector with indexes of all origins which should be removed

        count                   : `integer`
            Integer giving number of matches returned

        reject_obj              : `numpy.ndarray`
            Vector with indexes of all objects which should be removed
    """
    # print(Bcolors.WARNING+indent+"Remove me if possible."+Bcolors.ENDC)

    #   Sanitize reference object
    if reference_obj is None:
        reference_obj = []

    ###
    #   Keywords.
    #
    if option is None:
        option = 0
    if magnitude is not None:
        option = 4
    if option < 0 or option > 3:
        terminal_output.print_to_terminal("Invalid option code.", indent=indent)

    ###
    #   Set up some variables.
    #
    #   Number of origins
    k = len(x[0, :])
    #   Max. number of objects in the origins
    n = len(x[:, 0])
    #   Square of the required minimal distance
    dcr2 = dcr ** 2.

    #   Debug output
    if not silent:
        terminal_output.print_to_terminal(f"   Option code = {option}", indent=indent)
        terminal_output.print_to_terminal(f"   {k} origins (figures)", indent=indent)
        terminal_output.print_to_terminal(f"   max. number of objects {n}", indent=indent)

    ###
    #   The main loop.  Step through each index of origin with 'refORI',
    #                   look for matches in all the other origins.
    #

    #   Outer loop to allow for a pre burner to reject objects that are on
    #   not enough images
    reject_obj = 0
    for z in range(0, 2):
        #    Prepare index and reject arrays
        ind = np.zeros((k, n * 10), dtype=int) - 1  # arbitrary 10 to
        #   allow for multi
        #   identifications
        #   (option 3)
        rej_ori = np.zeros(k, dtype=int)
        rej_obj = np.zeros(n, dtype=int)
        #   Initialize counter of mutual sources
        count = 0

        #   Loop over the number of objects
        for i in range(0, n):
            #   Check that objects exists in origin with 'refORI'
            if x[i, reference_image_id] != 0.:
                #   Prepare dummy arrays and counter for bad origins
                _ind = np.zeros(k, dtype=int) - 1
                _ind[reference_image_id] = i
                _ori_rej = np.zeros(k, dtype=int)
                _obj_rej = np.zeros(n, dtype=int)
                _bad_ori = 0

                #   Loop over all origins
                for j in range(0, k):
                    #   Exclude origin with id 'refORI'
                    if j != reference_image_id:
                        xcomp = np.copy(x[:, j])
                        ycomp = np.copy(y[:, j])
                        xcomp[xcomp == 0] = 9E13
                        ycomp[ycomp == 0] = 9E13

                        #   Calculate radii
                        d2 = (x[i, reference_image_id] - xcomp) ** 2 + (y[i, reference_image_id] - ycomp) ** 2

                        if option == 3:
                            #   Find objects with distances that are smaller
                            #   than the required dcr
                            m = np.argwhere(d2 <= dcr2)
                            m = m.ravel()

                            #   Fill ind array
                            ml = len(m)
                            if ml != 0:
                                ind[j, count:count + ml] = m
                                ind[reference_image_id, count:count + ml] = _ind[reference_image_id]
                                count += ml
                        else:
                            #   Find object with minimum distance
                            dmch = np.amin(d2)
                            m = np.argmin(d2)

                            #   Check the critical radius criterion. If this
                            #   fails, the source will be marked as bad.
                            if dmch <= dcr2:
                                _ind[j] = m
                            else:
                                #   Number of bad origins for this source
                                #   -> counts up
                                _bad_ori += 1

                                #   Fill the reject vectors
                                #   Mark origin as "problematic"
                                _ori_rej[j] = 1

                                #   Check that object is not a reference
                                if i not in reference_obj or not protect_reference_obj:
                                    #   Mark object as problematic
                                    #   -> counts up
                                    _obj_rej[i] += 1

                if option != 3:
                    if (_bad_ori > (1 - bfrac) * k
                            and (i not in reference_obj or not protect_reference_obj)):
                        rej_obj += _obj_rej
                        continue
                    else:
                        rej_ori += _ori_rej

                        ind[:, count] = _ind
                        count += 1

        #   Prepare to discard objects that are not on N-nmissed origins
        rej_obj = np.argwhere(rej_obj >= nmissed).ravel()
        rej_obj_tup = tuple(rej_obj)

        #   Exit loop if there are no objects to be removed
        #   or if it is the second iteration
        if len(rej_obj) == 0 or z == 1:
            break

        reject_obj = np.copy(rej_obj)

        if not silent:
            terminal_output.print_to_terminal(
                f"   {len(reject_obj)} objects removed because "
                f"they are not found on >={nmissed} images",
                indent=indent,
            )

        #   Discard objects that are on not enough images
        x[rej_obj_tup, reference_image_id] = 0.
        y[rej_obj_tup, reference_image_id] = 0.

    if not silent:
        terminal_output.print_to_terminal(f"   {count} matches found.", indent=indent)

    if count > 0:
        ind = ind[:, 0:count]
        _ind2 = np.zeros(count, dtype=int) - 1
    else:
        reject = -1
        return ind, reject, count, reject_obj

    #   Return in case of option 0 and 3
    if option == 0:
        return ind, rej_ori, count, reject_obj
    if option == 3:
        return ind

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

    #   Loop over the origins
    for j in range(0, len(ind[:, 0])):
        if j == reference_image_id:
            continue
        #   Loop over the indexes of the objects
        # for i in range(0, np.max(ind[j,:])+1):
        for i in range(0, np.max(ind[j, :])):
            csave = len(ind[j, :])

            #   First find many-to-one identifications and saves the
            #   corresponding indexes in the ww array.
            ww = np.argwhere(ind[j, :] == i)
            ncount = len(ww)
            #   All but one of the origins in WW must eventually be removed.
            if ncount > 1:
                #   Mark origins that should be rejected.
                if ncount >= maxid and k > 2:
                    rej_ori[j] = 1

                if option == 4 and k == 2:
                    m = np.argmin(magnitude[ind[reference_image_id, ww]])
                else:
                    xx = x[i, j]
                    yy = y[i, j]
                    #   Calculate individual distances of the many-to-one
                    #   identifications
                    d2 = ((xx - x[ind[reference_image_id, ww], reference_image_id]) ** 2 +
                          (yy - y[ind[reference_image_id, ww], reference_image_id]) ** 2)

                    #   Logical test
                    if len(d2) != ncount:
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 1"
                            f"{style.Bcolors.ENDC}"
                        )

                    #   Find the element with the minimum distance
                    m = np.argmin(d2)

                #   Delete the minimum element from the
                #   deletion list itself.
                if option == 1:
                    ww = np.delete(ww, m)

                #   Now delete the deletion list from the original index
                #   arrays.
                for t in range(0, len(ind[:, 0])):
                    _ind2 = ind[t, :]
                    _ind2 = np.delete(_ind2, ww)
                    for l in range(0, len(_ind2)):
                        ind[t, l] = _ind2[l]

                #   Cut arrays depending on the number of
                #   one-to-one matches found in all origins
                ind = ind[:, 0:len(_ind2)]

                #   Logical tests
                if option == 2:
                    if len(ind[j, :]) != (csave - ncount):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(ind[reference_image_id, :]) != (csave - ncount):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                else:
                    if len(ind[j, :]) != (csave - ncount + 1):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 2"
                            f"{style.Bcolors.ENDC}"
                        )
                    if len(ind[reference_image_id, :]) != (csave - ncount + 1):
                        raise Exception(
                            f"{style.Bcolors.FAIL}\nLogic error 3"
                            f"{style.Bcolors.ENDC}"
                        )
                if len(ind[j, :]) != len(ind[reference_image_id, :]):
                    raise Exception(
                        f"{style.Bcolors.FAIL}\nLogic error 4"
                        f"{style.Bcolors.ENDC}"
                    )

    #   Determine the indexes of the images to be discarded
    reject = np.argwhere(rej_ori >= 1).ravel()

    #   Set count variable once more
    count = len(ind[reference_image_id, :])

    if not silent:
        terminal_output.print_to_terminal(
            f"       {len(ind[reference_image_id, :])} unique matches found.",
            indent=indent,
            style_name='OKGREEN',
        )

    return ind, reject, count, reject_obj
