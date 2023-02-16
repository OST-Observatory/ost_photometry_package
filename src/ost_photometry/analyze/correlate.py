############################################################################
####                            Libraries                               ####
############################################################################

import numpy as np

from .. import style
from ..style import bcolors

from astropy.coordinates import SkyCoord, matching
import astropy.units as u

############################################################################
####                        Routines & definitions                      ####
############################################################################


def posi_obj_astropy(xs, ys, ra_obj, dec_obj, w, ra_unit=u.hourangle,
                     dec_unit=u.deg, seplimit=2.*u.arcsec):
    '''
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image, using astropy matching
        algorisms.

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
    '''
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


def posi_obj_astropy_img(image, ra_obj, dec_obj, w, ra_unit=u.hourangle,
                         dec_unit=u.deg, seplimit=2.*u.arcsec):
    '''
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image, using astropy matching
        algorisms.

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

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
    '''
    #   Make coordinates object
    coord_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
        )

    #   Convert ra & dec to pixel coordinates
    x_obj, y_obj = w.all_world2pix(coord_obj.ra, coord_obj.dec, 0)

    #   Get photometry tabel
    tbl = image.photometry

    #   Create SkyCoord object for dataset
    coords_ds = SkyCoord.from_pixel(
        tbl['x_fit'],
        tbl['y_fit'],
        w,
        )

    #   Find matches in the dataset
    dist_mask = coords_ds.separation(coord_obj) < seplimit
    id_ds = np.argwhere(dist_mask).ravel()

    return id_ds, len(id_ds), x_obj, y_obj


def posi_obj_srcor_img(image, ra_obj, dec_obj, w, dcr=3, option=1,
                       ra_unit=u.hourangle, dec_unit=u.deg, verbose=False):
    '''
        Find the image coordinates of a star based on the stellar
        coordinates and the WCS of the image

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

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
    '''
    #   Make coordinates object
    coord_obj = SkyCoord(
        ra_obj,
        dec_obj,
        unit=(ra_unit, dec_unit),
        frame="icrs",
        )

    #   Convert ra & dec to pixel coordinates
    x_obj, y_obj = w.all_world2pix(coord_obj.ra, coord_obj.dec, 0)

    #   Get photometry tabel
    tbl = image.photometry

    #   Number of objects
    count = len(tbl['x_fit'])

    #   Define and fill new arrays to allow correlation
    xall = np.zeros((count,2))
    yall = np.zeros((count,2))
    xall[0,0]       = x_obj
    xall[0:count,1] = tbl['x_fit']
    yall[0,0]       = y_obj
    yall[0:count,1] = tbl['y_fit']

    #   Correlate calibration stars with stars on the image
    inds, reject, count, reject_obj = newsrcor(
        xall,
        yall,
        dcr,
        option=option,
        silent=not verbose,
        )

    return inds, count, x_obj, y_obj


def posi_obj_srcor(xs, ys, ra_obj, dec_obj, w, dcr=3, option=1,
                   ra_unit=u.hourangle, dec_unit=u.deg, verbose=False):
    '''
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
    '''
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
    xall = np.zeros((count,2))
    yall = np.zeros((count,2))
    xall[0,0]       = x_obj
    xall[0:count,1] = xs
    yall[0,0]       = y_obj
    yall[0:count,1] = ys

    #   Correlate calibration stars with stars on the image
    inds, reject, count, reject_obj = newsrcor(
        xall,
        yall,
        dcr,
        option=option,
        silent=not verbose,
        )

    return inds, count, x_obj, y_obj


def astropycor(x, y, w, refORI=0, refOBJ=[], nmissed=1, s_refOBJ=True,
               seplimit=2.*u.arcsec, cleanup_advanced=True):
    '''
        Correlation based on astropy matching algorithm

        Parameters
        ----------
        x                   : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. X direction.

        y                   : `list` of `numpy.ndarray`
            Object positions in pixel coordinates. Y direction.

        w                   : `astropy.wcs ` object
            WCS information

        refORI              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        refOBJ              : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``1``.

        s_refOBJ            : `boolean`, optional
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
    '''
    #   Number of datasets/images
    n = len(x)

    #   Create reference SkyCoord object
    coords_ref = SkyCoord.from_pixel(
        x[refORI],
        y[refORI],
        w,
        )

    #   Prepare index array and fill in values for the reference dataset
    idarray = np.ones((n, len(x[refORI])), dtype=int)
    idarray *= -1
    idarray[refORI,:] = np.arange(len(x[refORI]))

    #   Loop over datasets
    for i in range(0,n):
        #   Do nothing for the reference object
        if i != refORI:
            #   Dirty fix: In case of identical positions between the
            #              reference and the current data set,
            #              matching.search_around_sky will fail.
            #              => set reference indexes
            if np.all(x[i] == x[refORI]) and np.all(y[i] == y[refORI]):
                idarray[i,:] = idarray[refORI,:]
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
                idarray[i,id_ref] = id_current


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
        ref_isin = np.isin(rej_obj, refOBJ)

        #   If YES remove reference objects from the "bad" objects
        if s_refOBJ and np.any(ref_isin):
            refOBJ_id = np.argwhere(rej_obj==refOBJ)
            rej_obj = np.delete(rej_obj, refOBJ_id)

        #   Remove "bad" objects
        idarray = np.delete(idarray, rej_obj, 1)

        #   Calculate new reference object position
        shiftOBJ = np.argwhere(rej_obj < refOBJ)
        Nshift  = len(shiftOBJ)
        refOBJ = np.array(refOBJ) - Nshift

        #   2. Remove bad images

        #   Identify objects that were not identified in all datasets
        rowsrm = np.where(idarray == -1)

        #   Reduce to unique objects
        unique_ori, count_ori = np.unique(rowsrm[0], return_counts=True)

        #   Create mask -> Identify all datasets as bad that contain less
        #                  than 90% of all objects from the reference image.
        mask = count_ori > 0.02*len(x[refORI])
        rej_ori = unique_ori[mask]

        #   Remove those datasets
        idarray = np.delete(idarray, rej_ori, 0)

    else:
        rej_ori = np.array([], dtype=int)


    #   3. Remove remaining objects that are not on all datasets
    #      (afterburner)

    #   Identify objects that were not identified in all datasets
    rowsrm = np.where(idarray == -1)

    if s_refOBJ:
        #   Check if reference objects are within the "bad" objects
        ref_isin = np.isin(rowsrm[1], refOBJ)

        #   If YES remove reference objects from "bad" objects and remove
        #   the datasets on which they were not detected instead.
        if np.any(ref_isin):
            if n <= 2:
                raise RuntimeError(
                    f"{style.bcolors.FAIL} \nReference object only found one "
                    "or on no image at all. This is not sufficient. "
                    f"=> Exit {style.bcolors.ENDC}"
                    )
            rej_obj = rowsrm[1]
            rej_obj = np.unique(rej_obj)
            refOBJ_id = np.argwhere(rej_obj==refOBJ)
            rej_obj = np.delete(rej_obj, refOBJ_id)

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


def newsrcor(x, y, dcr=3, bfrac=1.0, maxid=1, refORI=0, refOBJ=[],
             nmissed=1, indent='   ', option=None, magnitude=None,
             silent=False, s_refOBJ=True):
    """
    NAME:
    ----
          NEWSRCOR

    PURPOSE:
    -------
          Correlate source positions from several origins (e.g., different
          images)

    SOURCE:
    ------
          Adapted from the IDL Astro Libary

    EXPLANATION:
    -----------
          Source matching is done by finding objects within a specified
          radius. The code is adapted from the standard srcor routine from
          the IDL Astronomy User's Library. The normal srcor routine was
          extended to fit the requirements of the C7 experiment within the
          astrophysics lab course at Potsdam University.

    CALLING SEQUENCE:
    ----------------
          srcor(x,y,dcr,bfrac,maxid,refORI,[option=,magnitude=,silent=]

    PARAMETERS:
    ----------
      x,y     - Arrays of x and y coordinates (several columns each). The
                following syntax is expected: x[array of source
                positions]. The program marches through the columns
                element by element, looking for the closest match.
      dcr     - Critical radius outside which correlations are rejected,
                but see 'option' below.
      bfrac   - Fraction of low quality source position origins, i.e., those
                origins (columns in x and y), for which it is expected to
                find a reduced number of objects with valid source
                positions.
      maxid   - Max. number of allowed identical cross identifications
                between objects from a specific origin (columns in x and y)
                and objects from the origin with the id 'refORI'. The origin
                will be rejected, if this limit is reached.
      refORI  - Id of the reference origin (e.g., an image).
      refOBJ  - Ids of the reference objects. The reference objects will not be
                removed from the list of objects.
      nmissed - Maximum number an object is allowed to be not detected in an
                origin. If this limit is reached the object will be removed.

    OPTIONAL KEYWORD PARAMETERS:
    ---------------------------
       option - Changes behavior of the program & description of output
                lists slightly, as follows:
          OPTION=0 | left out
                For each object of the origin 'refORI' the closest match
                from all other origins is found, but if none is found within
                the distance of 'dcr', the match is thrown out. Thus the
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
    magnitude - An array of stellar magnitudes corresponding to x and y.
                If magnitude is supplied, the brightest objects within 'dcr'
                is taken as a match. The option keyword is set to 4
                internally.
    silent    - Suppresses output if True.
    s_refOBJ  - Also reference objects will be rejected if Falls.

    Returns:
    -------
      ind     - Array of index positions of matched objects in the origins,
                set to -1 if no matches are found
      reject  - Vector with indexes of all origins which should be removed
      count   - Integer giving number of matches returned
    """
    #print(bcolors.WARNING+indent+"Remove me if possible."+bcolors.ENDC)

    ###
    #   Keywords.
    #
    if option is None: option=0
    if magnitude is not None: option=4
    if option < 0 or option > 3:
        print(bcolors.BOLD+indent+"Invalid option code."+bcolors.ENDC)


    ###
    #   Set up some variables.
    #
    #   Number of origins
    k = len(x[0,:])
    #   Max. number of objects in the origins
    n = len(x[:,0])
    #   Square of the required minimal distance
    dcr2 = dcr**2.

    #   Debug output
    if not silent:
        print(
            bcolors.BOLD
            +indent+"   Option code = "+str(option).strip()
            +bcolors.ENDC
            )
        print(
            bcolors.BOLD
            +indent+"   "+str(k).strip()+" origins (figures)"
            +bcolors.ENDC
            )
        print(
            bcolors.BOLD
            +indent+"   max. number of objects "+str(n).strip()
            +bcolors.ENDC
            )


    ###
    #   The main loop.  Step through each index of origin with 'refORI',
    #                   look for matches in all the other origins.
    #

    #   Outer loop to allow for a pre burner to reject objects that are on
    #   not enough images
    reject_obj = 0
    for z in range(0,2):
        #    Prepare index and reject arrays
        ind     = np.zeros((k,n*10), dtype=int) - 1     #   arbitrary 10 to
                                                        #   allow for multi
                                                        #   identifications
                                                        #   (option 3)
        rej_ori = np.zeros((k), dtype=int)
        rej_obj = np.zeros((n), dtype=int)
        #   Initialize counter of mutual sources
        count   = 0

        #   Loop over the number of objects
        for i in range(0, n):
            #   Check that objects exists in origin with 'refORI'
            if x[i,refORI] != 0.:
                #   Prepare dummy arrays and counter for bad origins
                _ind         = np.zeros((k), dtype=int) -1
                _ind[refORI] = i
                _ori_rej     = np.zeros((k), dtype=int)
                _obj_rej     = np.zeros((n), dtype=int)
                _bad_ori             = 0

                #   Loop over all origins
                for j in range(0, k):
                    #   Exclude origin with id 'refORI'
                    if j != refORI:
                        xcomp = np.copy(x[:,j])
                        ycomp = np.copy(y[:,j])
                        xcomp[xcomp == 0] = 9E13
                        ycomp[ycomp == 0] = 9E13

                        #   Calculate radii
                        d2=(x[i,refORI]-xcomp)**2+(y[i,refORI]-ycomp)**2

                        if option == 3:
                            #   Find objects with distances that are smaller
                            #   than the required dcr
                            m=np.argwhere(d2 <= dcr2)
                            m=m.ravel()

                            #   Fill ind array
                            ml = len(m)
                            if ml != 0:
                                ind[j,count:count+ml] = m
                                ind[refORI,count:count+ml] = _ind[refORI]
                                count += ml
                        else:
                            #   Find object with minimum distance
                            dmch=np.amin(d2)
                            m=np.argmin(d2)

                            #   Check the critical radius criterion. If this
                            #   fails, the source will be marked as bad.
                            if dmch <= dcr2:
                                _ind[j] = m
                            else:
                                #   Number of bad origins for this source
                                #   -> counts up
                                _bad_ori += 1

                                ##  Fill the reject vectors
                                #   Mark origin as "problematic"
                                _ori_rej[j] = 1

                                #   Check that object is not a reference
                                if i not in refOBJ or not s_refOBJ:
                                    #   Mark object as problematic
                                    #   -> counts up
                                    _obj_rej[i] += 1

                if option != 3:
                    if (_bad_ori > (1-bfrac)*k
                        and (i not in refOBJ or not s_refOBJ)):
                        rej_obj += _obj_rej
                        continue
                    else:
                        rej_ori += _ori_rej

                        ind[:,count] = _ind
                        count += 1

        #   Prepare to discard objects that are not on N-nmissed origins
        rej_obj     = np.argwhere(rej_obj >= nmissed).ravel()
        rej_obj_tup = tuple(rej_obj)

        #   Exit loop if there are no objects to be removed
        #   or it is the second iteration
        if len(rej_obj) == 0 or z == 1:
            break

        reject_obj = np.copy(rej_obj)


        if not silent:
            print(
                bcolors.BOLD
                +indent+"   "+str(len(reject_obj))
                +" objects removed because they are not found on >="
                +str(nmissed)+" images"
                +bcolors.ENDC
                )

        #   Discard objects that are on not enough images
        x[rej_obj_tup,refORI] = 0.
        y[rej_obj_tup,refORI] = 0.

    if not silent:
        print(
            bcolors.BOLD
            +indent+"   "+str(count).strip()+" matches found."
            +bcolors.ENDC
            )

    if count > 0:
        ind = ind[:,0:count]
        _ind2 = np.zeros((count), dtype=int) -1
    else:
        reject = np.copy(rej_ori)
        reject = -1
        return ind, reject, count, reject_obj

    #   Return in case of option 0 and 3
    if option == 0: return ind, reject, count, reject_obj
    if option == 3: return ind

    ###
    #   Modify the matches depending on input options.
    #
    if not silent:
        if option == 4:
            print(
                bcolors.BOLD
                +indent+"   Cleaning up output array using magnitudes."
                +bcolors.ENDC
                )
        else:
            if option == 1:
                print(
                    bcolors.BOLD
                    +indent+"   Cleaning up output array (option = 1)."
                    +bcolors.ENDC
                    )
            else:
                print(
                    bcolors.BOLD
                    +indent+"   Cleaning up output array (option = 2)."
                    +bcolors.ENDC
                    )

    #   Loop over the origins
    for j in range(0, len(ind[:,0])):
        if j == refORI:
            continue
        #   Loop over the indexes of the objects
        #for i in range(0, np.max(ind[j,:])+1):
        for i in range(0, np.max(ind[j,:])):
            csave = len(ind[j,:])

            #   First find many-to-one identifications and saves the
            #   corresponding indexes in the ww array.
            ww = np.argwhere(ind[j,:] == i)
            ncount = len(ww)
            #   All but one of the origins in WW must eventually be removed.
            if ncount > 1:
                #   Mark origins that should be rejected.
                if ncount >= maxid and k>2:
                    rej_ori[j] = 1

                if option == 4 and k==2:
                    m=np.argmin(magnitude[ind[refORI,ww]])
                else:
                    xx=x[i,j]
                    yy=y[i,j]
                    #   Calculate individual distances of the many-to-one
                    #   identifications
                    d2=((xx-x[ind[refORI,ww],refORI])**2 +
                        (yy-y[ind[refORI,ww],refORI])**2)

                    #   Logical test
                    if len(d2) != ncount:
                        raise Exception(
                            f"{style.bcolors.FAIL}\nLogic error 1"
                            f"{style.bcolors.ENDC}"
                            )

                    #   Find the element with the minimum distance
                    m = np.argmin(d2)

                #   Delete the minimum element from the
                #   deletion list itself.
                if option == 1:
                    ww = np.delete(ww, m)

                #   Now delete the deletion list from the original index
                #   arrays.
                for t in range(0, len(ind[:,0])):
                    _ind2 = ind[t,:]
                    _ind2 = np.delete(_ind2, ww)
                    for l in range(0, len(_ind2)):
                        ind[t,l] = _ind2[l]

                #   Cut arrays depending on the number of
                #   one-to-one matches found in all origins
                ind = ind[:,0:len(_ind2)]

                #   Logical tests
                if option == 2:
                    if len(ind[j,:]) != (csave-ncount):
                        raise Exception(
                            f"{style.bcolors.FAIL}\nLogic error 2"
                            f"{style.bcolors.ENDC}"
                            )
                    if len(ind[refORI,:]) != (csave-ncount):
                        raise Exception(
                            f"{style.bcolors.FAIL}\nLogic error 3"
                            f"{style.bcolors.ENDC}"
                            )
                else:
                    if len(ind[j,:]) != (csave-ncount+1):
                        raise Exception(
                            f"{style.bcolors.FAIL}\nLogic error 2"
                            f"{style.bcolors.ENDC}"
                            )
                    if len(ind[refORI,:]) != (csave-ncount+1):
                        raise Exception(
                            f"{style.bcolors.FAIL}\nLogic error 3"
                            f"{style.bcolors.ENDC}"
                            )
                if len(ind[j,:]) != len(ind[refORI,:]):
                    raise Exception(
                        f"{style.bcolors.FAIL}\nLogic error 4"
                        f"{style.bcolors.ENDC}"
                        )

    #   Determine the indexes of the images to be discarded
    reject = np.argwhere(rej_ori >= 1).ravel()

    #   Set count variable once more
    count = len(ind[refORI,:])

    if not silent:
        print(
            bcolors.OKGREEN
            +indent+"       "+str(len(ind[refORI,:])).strip()
            +" unique matches found."
            +bcolors.ENDC
            )

    return ind, reject, count, reject_obj


