"""Objects-of-interest identification and global-id verification."""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np

if typing.TYPE_CHECKING:
    from .. import analyze

import astropy.units as u
from astropy import wcs
from astropy.coordinates import SkyCoord

from ... import terminal_output
from ..ooi_ids import ooi_photometry_id
from .core import correlation_own


def find_objects_of_interest_astropy(
        x_pixel_position_dataset: np.ndarray,
        y_pixel_position_dataset: np.ndarray, flux: np.ndarray,
        objects_of_interest: list[analyze.ObjectOfInterest], filter_: str,
        current_wcs: wcs.WCS, duplicate_handling: str = 'distance',
        separation_limit: u.Quantity = 2. * u.arcsec,
        indent: int = 1
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

    flux
        Object flux

    objects_of_interest
        Object with 'object of interest' properties

    filter_
        Filter identifier

    current_wcs
        WCS info

    duplicate_handling
        Specifies how to handle multiple object identification filtering.
        There are two options: 'distance' and 'flux'. The 'distance' filtering
        is based on the distance between the correlated objects. In this case,
        the one with the smallest distance is used. The second option is
        based on the measure flux values. In this case the highest one is used.
        Default is ``distance``.

    separation_limit
        Allowed separation between objects.
        Default is ``2.*u.arcsec``.

    indent
        Indentation for the console output lines
        Default is ``1``.
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
        candidate_rows = np.argwhere(mask).ravel()

        if len(candidate_rows) > 1:
            terminal_output.print_to_terminal(
                f"More than one object detected within the separation limit to "
                f"{object_.name}.",
                style_name='WARNING',
                indent=indent,
            )

            if duplicate_handling not in ['distance', 'flux']:
                terminal_output.print_to_terminal(
                    f"Option ({duplicate_handling}) for filtering multiple "
                    f"object identifications are not recognized. Available "
                    f"options are 'distance' and 'flux'. Use the "
                    f"'first_in_list' option.",
                    style_name='WARNING',
                    indent=indent,
                )
                duplicate_handling = 'distance'

            if duplicate_handling == 'distance':
                photometry_row_index = int(np.argmin(separation))
                terminal_output.print_to_terminal(
                    "Use the object that is the closest.",
                    style_name='WARNING',
                    indent=indent,
                )

            elif duplicate_handling == 'flux':
                #   Calculate nd filter object ids based on observed flux.
                #   Use the one with the highes flux.
                photometry_row_index = int(
                    candidate_rows[np.argmax(flux[candidate_rows])]
                )
                terminal_output.print_to_terminal(
                    "Use the object that is the brightest.",
                    style_name='WARNING',
                    indent=indent,
                )

        elif candidate_rows.size == 0:
            terminal_output.print_to_terminal(
                f"No object detected within the separation limit to "
                f"{object_.name}. Set object ID to None",
                style_name='WARNING',
                indent=indent,
            )
            photometry_row_index = None

        else:
            photometry_row_index = int(candidate_rows[0])

        #   Add ID to object of interest
        object_.id_in_image_series[filter_] = photometry_row_index


def find_objects_of_interest_srcor(
        x_pixel_position_dataset: np.ndarray,
        y_pixel_position_dataset: np.ndarray, flux: np.ndarray,
        objects_of_interest: list[analyze.ObjectOfInterest], filter_: str,
        current_wcs: wcs.WCS, max_pixel_between_objects: int = 3,
        ooi_correlation_strategy: int = 1, duplicate_handling: str = 'first_in_list',
        verbose: bool = False, indent: int = 1) -> None:
    """
    Find the image coordinates of a star based on the stellar
    coordinates and the WCS of the image

    Parameters
    ----------
    x_pixel_position_dataset
        Positions of the objects in Pixel in X direction

    y_pixel_position_dataset
        Positions of the objects in Pixel in Y direction

    flux
        Object flux

    objects_of_interest
        Object with 'object of interest' properties

    filter_
        Filter identifier

    current_wcs
        WCS info

    max_pixel_between_objects
        Maximal distance between two objects in Pixel
        Default is ``3``.

    ooi_correlation_strategy
        Option for the srcor correlation function
        Default is ``1``.

    duplicate_handling
        Specifies how to handle multiple object identification filtering.
        There are two options: 'first_in_list' and 'flux'. The 'distance'
        filtering is based on the distance between the correlated objects.
        In this case, the one with the smallest distance is used. The
        second option is based on the measure flux values. In this case
        the highest one is used.
        Default is ``first_in_list``.

    verbose
        If True additional output will be printed to the command line.
        Default is ``False``.

    indent
        Indentation for the console output lines
        Default is ``1``.
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

        #   Correlate object of interest with stars on the image
        index_obj, reject, count, reject_obj = correlation_own(
            pixel_position_all_x,
            pixel_position_all_y,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            silent=not verbose,
        )

        #   correlation_own column 1: matched photometry row index(es)
        srcor_matches = index_obj[1]

        if len(srcor_matches) > 1:
            if duplicate_handling not in ['first_in_list', 'flux']:
                terminal_output.print_to_terminal(
                    f"Option ({duplicate_handling}) for filtering multiple "
                    f"object identifications are not recognized. Available "
                    f"options are 'first_in_list' and 'flux'. Use the "
                    f"'first_in_list' option.",
                    style_name='WARNING',
                    indent=indent,
                )
                duplicate_handling = 'first_in_list'

            if duplicate_handling == 'first_in_list':
                #   message would be feasible
                terminal_output.print_to_terminal(
                    "Take the first one in the list.",
                    style_name='WARNING',
                    indent=indent,
                )
                photometry_row_index = srcor_matches[0]

            elif duplicate_handling == 'flux':
                #   Calculate nd filter object ids based on observed flux.
                #   Use the one with the highes flux.
                photometry_row_index = srcor_matches[
                    np.argmax(flux[srcor_matches])
                ]
                terminal_output.print_to_terminal(
                    "Use the object that is the brightest.",
                    style_name='WARNING',
                    indent=indent,
                )

        elif not srcor_matches:
            terminal_output.print_to_terminal(
                f"No object detected within the separation limit to "
                f"{object_.name}. Set object ID to None",
                style_name='WARNING',
                indent=indent,
            )
            photometry_row_index = None

        else:
            photometry_row_index = srcor_matches[0]

        #   Add ID to object of interest
        object_.id_in_image_series[filter_] = photometry_row_index


def identify_object_of_interest_in_dataset(
        x_pixel_positions: np.ndarray, y_pixel_positions: np.ndarray,
        flux: np.ndarray,
        objects_of_interest: list[analyze.ObjectOfInterest], filter_: str,
        current_wcs: wcs.WCS, separation_limit: u.Quantity = 2. * u.arcsec,
        max_pixel_between_objects: int = 3, ooi_correlation_strategy: int = 1,
        verbose: bool = False, correlation_method: str = 'astropy',
        duplicate_handling: dict[str, str] | None = None,
        indent: int = 1) -> None:
    """
    Identify a specific star based on its right ascension and declination
    in a dataset of pixel coordinates. Requires a valid WCS.

    Parameters
    ----------
    x_pixel_positions
        Object positions in pixel coordinates. X direction.

    y_pixel_positions
        Object positions in pixel coordinates. Y direction.

    flux
        Object flux

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

    ooi_correlation_strategy
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

    duplicate_handling
        Specifies how to handle multiple object identification filtering.
        There are two options for each 'correlation_method':
            'own':     'first_in_list' and 'flux'.  The 'first_in_list'
                        filtering just takes the first obtained result.
            'astropy': 'distance' and 'flux'. The 'distance' filtering is
                        based on the distance between the correlated objects.
                        In this case, the one with the smallest distance is
                        used.
        The second option for both correlation method is based on the measure
        flux values. In this case the largest one is used.
        Default is ``None``.

    indent
        Indentation for the console output lines
        Default is ``1``.
    """
    if duplicate_handling is None:
        duplicate_handling = {'own': 'first_in_list', 'astropy': 'distance'}

    if correlation_method == 'astropy':
        find_objects_of_interest_astropy(
            x_pixel_positions,
            y_pixel_positions,
            flux,
            objects_of_interest,
            filter_,
            current_wcs,
            separation_limit=separation_limit,
            duplicate_handling=duplicate_handling['astropy'],
            indent=indent,
        )

    elif correlation_method == 'own':
        find_objects_of_interest_srcor(
            x_pixel_positions,
            y_pixel_positions,
            flux,
            objects_of_interest,
            filter_,
            current_wcs,
            max_pixel_between_objects=max_pixel_between_objects,
            ooi_correlation_strategy=ooi_correlation_strategy,
            duplicate_handling=duplicate_handling['own'],
            verbose=verbose,
            indent=indent,
        )

    else:
        raise ValueError(
            f'The correlation method needs to either "astropy" or "own".'
            f'Got {correlation_method} instead.'
        )

def _debug_image_label_for_verify(image) -> str:
    fn = getattr(image, "filename", None)
    if isinstance(fn, str) and fn:
        return fn
    p = getattr(image, "path", None)
    if p is not None:
        return Path(p).name
    return f"image_id={getattr(image, 'image_id', '?')}"


def _reidentify_ooi_row_index_astropy(
    x_pixel_position_dataset: np.ndarray,
    y_pixel_position_dataset: np.ndarray,
    flux: np.ndarray,
    coordinates_object: SkyCoord,
    current_wcs: wcs.WCS,
    separation_limit: u.Quantity,
    duplicate_handling: str,
) -> int | None:
    """Sky-match one position; return row index or None (no terminal output)."""
    coordinates_dataset = SkyCoord.from_pixel(
        np.asarray(x_pixel_position_dataset, dtype=float),
        np.asarray(y_pixel_position_dataset, dtype=float),
        current_wcs,
    )
    separation = coordinates_dataset.separation(coordinates_object)
    mask = separation < separation_limit
    matched_rows = np.argwhere(mask).ravel()
    if matched_rows.size == 0:
        return None
    if matched_rows.size > 1:
        if duplicate_handling not in ("distance", "flux"):
            duplicate_handling = "distance"
        if duplicate_handling == "distance":
            at = matched_rows.astype(int)
            j = int(np.argmin(separation[at].to_value(u.arcsec)))
            return int(at[j])
        at = matched_rows.astype(int)
        j = int(np.argmax(np.asarray(flux, dtype=float)[at]))
        return int(at[j])
    return int(matched_rows[0])


def _reidentify_ooi_row_index_srcor(
    x_pixel_position_dataset: np.ndarray,
    y_pixel_position_dataset: np.ndarray,
    flux: np.ndarray,
    coordinates_object: SkyCoord,
    current_wcs: wcs.WCS,
    max_pixel_between_objects: float,
    ooi_correlation_strategy: int,
    duplicate_handling: str,
) -> int | None:
    """Same recipe as :func:`find_objects_of_interest_srcor` for one position."""
    n_obj_dataset = len(x_pixel_position_dataset)
    if n_obj_dataset == 0:
        return None
    pixel_position_all_x = np.zeros((n_obj_dataset, 2))
    pixel_position_all_y = np.zeros((n_obj_dataset, 2))
    pixel_position_all_x[0:n_obj_dataset, 1] = x_pixel_position_dataset
    pixel_position_all_y[0:n_obj_dataset, 1] = y_pixel_position_dataset

    obj_pixel_position_x, obj_pixel_position_y = current_wcs.all_world2pix(
        coordinates_object.ra,
        coordinates_object.dec,
        0,
    )
    pixel_position_all_x[0, 0] = obj_pixel_position_x
    pixel_position_all_y[0, 0] = obj_pixel_position_y

    index_obj, _reject, _count, _reject_obj = correlation_own(
        pixel_position_all_x,
        pixel_position_all_y,
        max_pixel_between_objects=max_pixel_between_objects,
        ooi_correlation_strategy=ooi_correlation_strategy,
        silent=True,
    )
    oid = np.atleast_1d(np.asarray(index_obj[1])).ravel()
    if oid.size == 0:
        return None
    if oid.size > 1:
        if duplicate_handling not in ("first_in_list", "flux"):
            duplicate_handling = "first_in_list"
        if duplicate_handling == "first_in_list":
            return int(oid[0])
        ii = oid.astype(int)
        return int(ii[np.argmax(np.asarray(flux, dtype=float)[ii])])
    return int(oid[0])


def verify_objects_of_interest_global_correlated_ids(
    observation: analyze.Observation,
    filter_list: list[str],
    *,
    separation_limit: u.Quantity = 2.0 * u.arcsec,
    correlation_method: str = "astropy",
    duplicate_handling_object_identification: dict[str, str] | None = None,
    max_pixel_between_objects: float = 3.0,
    ooi_correlation_strategy: int = 1,
    indent: int = 1,
) -> None:
    """
    After :func:`assign_global_correlated_object_ids`, re-run sky / srcor matching
    on every image and compare to ``ObjectOfInterest.correlated_id`` (falling
    back to ``id_in_image_series``).

    Prints mismatches and a short summary to ``terminal_output``.
    """
    ooi_list = getattr(observation, "objects_of_interest", None) or []
    if not ooi_list:
        return

    dup = duplicate_handling_object_identification or {
        "own": "first_in_list",
        "astropy": "distance",
    }

    terminal_output.print_to_terminal(
        "Debug: verify objects_of_interest vs. global correlated row ids "
        "(all filters, all images)",
        style_name="HEADER",
        indent=indent,
    )

    n_ok = 0
    n_mismatch = 0
    n_no_sky = 0
    n_bad_id_col = 0

    for filter_ in filter_list:
        series = observation.image_series_dict.get(filter_)
        if series is None or not series.image_list:
            continue
        wcs_obj = getattr(series, "wcs", None)
        if wcs_obj is None:
            terminal_output.print_to_terminal(
                f"  skip filter {filter_!r}: no WCS on series",
                style_name="WARNING",
                indent=indent,
            )
            continue

        for image in series.image_list:
            phot = image.photometry
            if phot is None or len(phot) == 0:
                continue
            x = np.asarray(phot["x_fit"], dtype=float)
            y = np.asarray(phot["y_fit"], dtype=float)
            fv = phot["flux_fit"]
            if hasattr(fv, "value"):
                fv = fv.value
            flux = np.asarray(fv, dtype=float)
            im_label = _debug_image_label_for_verify(image)

            for object_ in ooi_list:
                stored = ooi_photometry_id(object_, filter_=filter_)
                if stored is None:
                    continue
                try:
                    stored_i = int(stored)
                except (TypeError, ValueError):
                    terminal_output.print_to_terminal(
                        f"  {object_.name!r} {filter_} {im_label}: "
                        f"non-integer stored id {stored!r}",
                        style_name="WARNING",
                        indent=indent,
                    )
                    n_mismatch += 1
                    continue

                if correlation_method == "astropy":
                    k = _reidentify_ooi_row_index_astropy(
                        x,
                        y,
                        flux,
                        object_.coordinates_object,
                        wcs_obj,
                        separation_limit,
                        dup.get("astropy", "distance"),
                    )
                elif correlation_method == "own":
                    k = _reidentify_ooi_row_index_srcor(
                        x,
                        y,
                        flux,
                        object_.coordinates_object,
                        wcs_obj,
                        max_pixel_between_objects,
                        ooi_correlation_strategy,
                        dup.get("own", "first_in_list"),
                    )
                else:
                    terminal_output.print_to_terminal(
                        f"  skip verify: unknown correlation_method={correlation_method!r}",
                        style_name="WARNING",
                        indent=indent,
                    )
                    return

                if k is None:
                    n_no_sky += 1
                    terminal_output.print_to_terminal(
                        f"  {object_.name!r} {filter_} {im_label}: "
                        f"no sky match (stored id was {stored_i})",
                        style_name="WARNING",
                        indent=indent,
                    )
                    continue

                if k != stored_i:
                    n_mismatch += 1
                    terminal_output.print_to_terminal(
                        f"  MISMATCH {object_.name!r} {filter_} {im_label}: "
                        f"stored id={stored_i}, reidentified row={k}",
                        style_name="WARNING",
                        indent=indent,
                    )
                    continue

                n_ok += 1
                if "id" in phot.colnames:
                    id_at = int(np.asarray(phot["id"][k]))
                    if id_at != stored_i:
                        n_bad_id_col += 1
                        terminal_output.print_to_terminal(
                            f"  id column != stored at row {k} "
                            f"{object_.name!r} {filter_} {im_label}: "
                            f"photometry['id']={id_at}, stored={stored_i}",
                            style_name="WARNING",
                            indent=indent,
                        )

    terminal_output.print_to_terminal(
        f"OOI id verify summary: {n_ok} row matches, {n_mismatch} mismatches, "
        f"{n_no_sky} no re-match, {n_bad_id_col} id-column mismatches",
        style_name="OKGREEN" if n_mismatch == 0 and n_no_sky == 0 else "WARNING",
        indent=indent,
    )
    terminal_output.print_to_terminal("", indent=indent)

