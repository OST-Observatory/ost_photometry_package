"""Cluster / proper-motion / region selection (Gaia Vizier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, matching
from astropy.stats import sigma_clip
from astropy.table import Column, Table
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from sklearn.cluster import SpectralClustering

from ... import style, terminal_output
from ... import utilities as base_utilities
from .. import plots
from ..post_processing.coords import (
    plot_starmap_from_imaging_context,
    table_object_sky_coords,
)
from ..post_processing.imaging import ImagingPlotContext, imaging_context_from_image_series
from .duplicates import clear_duplicates
from .starmaps import prepare_and_plot_starmap

if TYPE_CHECKING:
    from ..models import ImageSeries

def _resolve_imaging_plot_context(
    *,
    image_series: "analyze.ImageSeries | None" = None,
    plot_context: ImagingPlotContext | None = None,
) -> ImagingPlotContext:
    if plot_context is not None:
        return plot_context
    if image_series is not None:
        return imaging_context_from_image_series(image_series)
    raise TypeError(
        "Provide plot_context=... or image_series=... (for example "
        "plot_context=imaging_context_from_image_series(series))."
    )


def _vizier_field_cone(
    ctx: ImagingPlotContext,
    image_series: "analyze.ImageSeries | None",
) -> tuple[SkyCoord, u.Quantity]:
    """Center and radius for ``Vizier.query_region`` (Gaia cone)."""
    if ctx.field_center_icrs is not None and ctx.field_radius_arcmin is not None:
        return ctx.field_center_icrs, ctx.field_radius_arcmin * u.arcmin
    if image_series is not None:
        return (
            image_series.coordinates_image_center,
            image_series.field_of_view_x * u.arcmin,
        )
    raise TypeError(
        "ImagingPlotContext.field_center_icrs and field_radius_arcmin must be set, "
        "or pass image_series=..., for Gaia / Vizier cone queries."
    )


def proper_motion_selection(
        tbl: Table,
        *,
        image_series: 'analyze.ImageSeries | None' = None,
        plot_context: ImagingPlotContext | None = None,
        catalog: str = "I/355/gaiadr3", g_mag_limit: int = 20,
        separation_limit: float = 1., sigma: float = 3.,
        max_n_iterations_sigma_clipping: int = 3,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf',
    ) -> Column:
    """
    Select a subset of objects based on their proper motion

    Parameters
    ----------
    tbl
        Table with position information

    image_series
        Optional :class:`~ost_photometry.analyze.models.ImageSeries`; used with
        ``plot_context is None`` to build an :class:`~ost_photometry.analyze.post_processing.imaging.ImagingPlotContext`.

    plot_context
        :class:`~ost_photometry.analyze.post_processing.imaging.ImagingPlotContext`
        with WCS, filter name, and (for Gaia) field center / radius. Provide this **or**
        ``image_series``.

    catalog
        Identifier for the catalog to download.
        Default is ``I/350/gaiaedr3``.

    g_mag_limit
        Limiting magnitude in the G band. Fainter objects will not be
        downloaded.

    separation_limit
        Maximal allowed separation between objects in arcsec.
        Default is ``1``.

    sigma
        The sigma value used in the sigma clipping of the proper motion
        values.
        Default is ``3``.

    max_n_iterations_sigma_clipping
        Maximal number of iteration of the sigma clipping.
        Default is ``3``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    ctx = _resolve_imaging_plot_context(
        image_series=image_series, plot_context=plot_context
    )
    w = ctx.wcs
    plot_stub = str(ctx.out_path_stub)
    v_center, v_radius = _vizier_field_cone(ctx, image_series)

    obj_coordinates = table_object_sky_coords(tbl, w)

    #   Get Gaia data from Vizier
    #
    #   Columns to download
    columns = [
        'RA_ICRS',
        'DE_ICRS',
        'Gmag',
        'Plx',
        'e_Plx',
        'pmRA',
        'e_pmRA',
        'pmDE',
        'e_pmDE',
        'RUWE',
    ]

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog,
        column_filters={'Gmag': '<' + str(g_mag_limit)},
    )

    #   Get data from the corresponding catalog for the objects in
    #   the field of view
    result = v.query_region(v_center, radius=v_radius)

    #   Create SkyCoord object with coordinates of all Gaia objects
    calib_coordinates = SkyCoord(
        result[0]['RA_ICRS'],
        result[0]['DE_ICRS'],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    separation_limit = separation_limit * u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        obj_coordinates,
        calib_coordinates,
        separation_limit,
    )

    #   Identify and remove duplicate indexes
    id_img, d2ds, id_calib = clear_duplicates(
        id_img,
        d2ds,
        id_calib,
    )
    id_calib, d2ds, id_img = clear_duplicates(
        id_calib,
        d2ds,
        id_img,
    )

    #   Sigma clipping of the proper motion values
    #
    #   Proper motion of the common objects
    pm_de = result[0]['pmDE'][id_calib]
    pm_ra = result[0]['pmRA'][id_calib]

    #   Parallax
    parallax = result[0]['Plx'][id_calib].data / 1000 * u.arcsec

    #   Distance
    distance = parallax.to_value(u.kpc, equivalencies=u.parallax())

    #   Sigma clipping
    sigma_clip_de = sigma_clip(
        pm_de,
        sigma=sigma,
        maxiters=max_n_iterations_sigma_clipping,
    )
    sigma_clip_ra = sigma_clip(
        pm_ra,
        sigma=sigma,
        maxiters=max_n_iterations_sigma_clipping,
    )

    #   Create mask from sigma clipping
    mask = sigma_clip_ra.mask | sigma_clip_de.mask

    #   Make plots
    #
    #   Restrict Gaia table to the common objects
    result_cut = result[0][id_calib][mask]

    #   Convert ra & dec to pixel coordinates
    x_obj, y_obj = w.all_world2pix(
        result_cut['RA_ICRS'],
        result_cut['DE_ICRS'],
        0,
    )

    tbl_pm_plot = Table(names=["x_fit", "y_fit"], data=[x_obj, y_obj])
    plot_starmap_from_imaging_context(
        ctx,
        tbl_pm_plot,
        filter_=ctx.filter_name,
        x_name="x_fit",
        y_name="y_fit",
        rts_pre="proper motion [Gaia]",
        label="Objects selected based on proper motion",
        add_image_id=True,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

    #   2D and 3D plot of the proper motion and the distance
    plots.scatter(
        [pm_ra],
        'pm_RA * cos(DEC) (mas/yr)',
        [pm_de],
        'pm_DEC (mas/yr)',
        'compare_pm_',
        plot_stub,
        file_type=file_type_plots,
    )
    plots.d3_scatter(
        [pm_ra],
        [pm_de],
        [distance],
        plot_stub,
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        file_type=file_type_plots,
    )

    #   Apply mask
    return tbl[id_img][mask]


def region_selection(
        coordinates_target: SkyCoord | list[SkyCoord], tbl: Table,
        *,
        image_series: 'analyze.ImageSeries | None' = None,
        plot_context: ImagingPlotContext | None = None,
        radius: float = 600., file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True,
    ) -> tuple[Table, np.ndarray]:
    """
    Select a subset of objects based on a target coordinate and a radius

    Parameters
    ----------
    coordinates_target
        Coordinates of the observed object such as a star cluster

    tbl
        Table with object position information

    image_series
        Optional series used to build ``ImagingPlotContext`` when ``plot_context``
        is omitted.

    plot_context
        Context for WCS and starmaps; provide this or ``image_series``.

    radius
        Selection radius around the object in arcsec
        Default is ``600``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    Returns
    -------
    tbl
        Table with object position information

    mask
        Boolean mask applied to the table
    """
    ctx = _resolve_imaging_plot_context(
        image_series=image_series, plot_context=plot_context
    )
    obj_coordinates = table_object_sky_coords(tbl, ctx.wcs)

    #   Calculate separation between the coordinates defined in ``coord``
    #   the objects in ``tbl``
    if isinstance(coordinates_target, list):
        mask = np.zeros(len(obj_coordinates), dtype=bool)
        for target_coordinates in coordinates_target:
            sep = obj_coordinates.separation(target_coordinates)

            #   Calculate mask of all object closer than ``radius``
            mask = mask | (sep.arcsec <= radius)
    else:
        sep = obj_coordinates.separation(coordinates_target)

        #   Calculate mask of all object closer than ``radius``
        mask = sep.arcsec <= radius

    #   Limit objects to those within radius
    tbl = tbl[mask]

    #   Plot starmap
    plot_starmap_from_imaging_context(
        ctx,
        tbl,
        filter_=ctx.filter_name,
        x_name="x",
        y_name="y",
        rts_pre="radius selection, image",
        label=f"Objects selected within {radius}'' of the target",
        add_image_id=True,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

    return tbl, mask


def find_cluster(
        tbl: Table, object_names: list[str],
        *,
        image_series: 'analyze.ImageSeries | None' = None,
        plot_context: ImagingPlotContext | None = None,
        catalog: str = "I/355/gaiadr3", g_mag_limit: float = 20.,
        separation_limit: float = 1., max_distance: float = 6.,
        parameter_set: int = 1, file_type_plots: str = 'pdf',
        use_wcs_projection_for_star_maps: bool = True,
        cluster_selection_id: int | None = None,
    ) -> tuple[Table, int, np.ndarray, np.ndarray]:
    """
    Identify cluster in data

    Parameters
    ----------
    tbl
        Table with position information

    object_names
        Names of the objects. This first entry in the list is assumed to
        be the custer of interest.

    image_series
        Optional series; used to build context when ``plot_context`` is omitted.

    plot_context
        Imaging context (WCS, filter, Vizier cone). Provide this or ``image_series``.

    catalog
        Identifier for the catalog to download.
        Default is ``I/350/gaiaedr3``.

    g_mag_limit
        Limiting magnitude in the G band. Fainter objects will not be
        downloaded.

    separation_limit
        Maximal allowed separation between objects in arcsec.
        Default is ``1``.

    max_distance
        Maximal distance of the star cluster.
        Default is ``6.``.

    parameter_set
        Predefined parameter sets can be used.
        Possibilities: ``1``, ``2``, ``3``
        Default is ``1``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    Returns
    -------
    tbl
        Table with object position information

    id_img

    mask
        The mask that needs to be applied to the table.

    cluster_mask
        Mask that identifies cluster members according to the user
        input.
    """
    ctx = _resolve_imaging_plot_context(
        image_series=image_series, plot_context=plot_context
    )
    obj_coordinates = table_object_sky_coords(tbl, ctx.wcs)
    plot_stub = str(ctx.out_path_stub)
    v_center, v_radius = _vizier_field_cone(ctx, image_series)

    #   Get Gaia data from Vizier
    #
    #   Columns to download
    columns = [
        'RA_ICRS',
        'DE_ICRS',
        'Gmag',
        'Plx',
        'e_Plx',
        'pmRA',
        'e_pmRA',
        'pmDE',
        'e_pmDE',
        'RUWE',
    ]

    #   Define astroquery instance
    v = Vizier(
        columns=columns,
        row_limit=1e6,
        catalog=catalog,
        column_filters={'Gmag': '<' + str(g_mag_limit)},
    )

    #   Get data from the corresponding catalog for the objects in
    #   the field of view
    result = v.query_region(v_center, radius=v_radius)[0]

    #   Multiple objects can be specified. The first object is assumed to
    #   be the cluster of interest.
    object_name = object_names[0]

    #   Restrict proper motion to Simbad value plus some margin
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('pmra', 'pmdec')

    result_simbad = custom_simbad.query_object(object_name)
    pm_ra_object = result_simbad['pmra'].value[0]
    pm_de_object = result_simbad['pmdec'].value[0]
    if pm_ra_object != '--' and pm_de_object != '--':
        pm_m = 3.
        mask_de = ((result['pmDE'] <= pm_de_object - pm_m) |
                   (result['pmDE'] >= pm_de_object + pm_m))
        mask_ra = ((result['pmRA'] <= pm_ra_object - pm_m) |
                   (result['pmRA'] >= pm_ra_object + pm_m))
        mask = np.invert(mask_de | mask_ra)
        result = result[mask]

    #   Create SkyCoord object with coordinates of all Gaia objects
    calib_coordinates = SkyCoord(
        result['RA_ICRS'],
        result['DE_ICRS'],
        unit=(u.degree, u.degree),
        frame="icrs"
    )

    #   Correlate own objects with Gaia objects
    #
    #   Set maximal separation between objects
    separation_limit = separation_limit * u.arcsec

    #   Correlate data
    id_img, id_calib, d2ds, d3ds = matching.search_around_sky(
        obj_coordinates,
        calib_coordinates,
        separation_limit,
    )

    #   Identify and remove duplicate indexes
    id_img, d2ds, id_calib = clear_duplicates(
        id_img,
        d2ds,
        id_calib,
    )
    id_calib, d2ds, id_img = clear_duplicates(
        id_calib,
        d2ds,
        id_img,
    )

    #   Find cluster in proper motion and distance data
    #

    #   Proper motion of the common objects
    pm_de_common_objects = result['pmDE'][id_calib]
    pm_ra_common_objects = result['pmRA'][id_calib]

    #   Parallax
    parallax = result['Plx'][id_calib].data / 1000 * u.arcsec

    #   Distance
    distance = parallax.to_value(u.kpc, equivalencies=u.parallax())

    #   Restrict sample to objects closer than 'max_distance'
    #   and remove nans and infs
    if max_distance is not None:
        max_mask = np.invert(distance <= max_distance)
        distance_mask = np.isnan(distance) | np.isinf(distance) | max_mask
    else:
        distance_mask = np.isnan(distance) | np.isinf(distance)

    #   Calculate a mask accounting for NaNs in proper motion and the
    #   distance estimates
    mask = np.invert(pm_de_common_objects.mask | pm_ra_common_objects.mask
                     | distance_mask)

    #   Convert astropy table to pandas data frame and add distance
    pd_result = result[id_calib].to_pandas()
    pd_result['distance'] = distance
    pd_result = pd_result[mask]

    #   Prepare SpectralClustering object to identify the "cluster" in the
    #   proper motion and distance data sets
    if parameter_set == 1:
        n_clusters = 2
        random_state = 25
        n_neighbors = 20
        affinity = 'nearest_neighbors'
    elif parameter_set == 2:
        n_clusters = 10
        random_state = 2
        n_neighbors = 4
        affinity = 'nearest_neighbors'
    elif parameter_set == 3:
        n_clusters = 2
        random_state = 25
        n_neighbors = 20
        affinity = 'rbf'
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo valid parameter set defined: "
            f"Possibilities are 1, 2, or 3. {style.Bcolors.ENDC}"
        )
    spectral_cluster_model = SpectralClustering(
        # eigen_solver='lobpcg',
        n_clusters=n_clusters,
        random_state=random_state,
        # gamma=2.,
        # gamma=5.,
        n_neighbors=n_neighbors,
        affinity=affinity,
    )

    #   Find "cluster" in the data
    #   SpectralClustering is O(n³) and hangs on datasets >~1500 objects.
    #   Subsample when necessary and assign rest via nearest centroid.
    max_cluster_sample = 100
    cluster_features = ['pmDE', 'pmRA', 'distance']
    n_objects = len(pd_result)
    if n_objects > max_cluster_sample:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(
            len(pd_result), size=max_cluster_sample, replace=False
        )
        pd_sample = pd_result.iloc[sample_idx]
        sample_labels = spectral_cluster_model.fit_predict(
            pd_sample[cluster_features]
        )
        # Compute cluster centroids and assign all points to nearest
        centroids = []
        for c in range(n_clusters):
            mask_c = sample_labels == c
            centroids.append(
                pd_sample.loc[mask_c, cluster_features].mean().values
            )
        from scipy.spatial.distance import cdist
        centroid_arr = np.array(centroids)
        dists = cdist(
            pd_result[cluster_features].values,
            centroid_arr,
        )
        pd_result['cluster'] = np.argmin(dists, axis=1)
    else:
        pd_result['cluster'] = spectral_cluster_model.fit_predict(
            pd_result[cluster_features],
        )

    #   3D plot of the proper motion and the distance
    #   -> select the star cluster by eye
    groups = pd_result.groupby('cluster')
    pm_ra_group = []
    pm_de_group = []
    distance_group = []
    for name, group in groups:
        pm_ra_group.append(group.pmRA.values)
        pm_de_group.append(group.pmDE.values)
        distance_group.append(group.distance.values)
    plots.d3_scatter(
        pm_ra_group,
        pm_de_group,
        distance_group,
        plot_stub,
        # color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        # string='_3D_cluster_',
        pm_ra=pm_ra_object,
        pm_dec=pm_de_object,
        file_type=file_type_plots,
    )
    plots.d3_scatter(
        pm_ra_group,
        pm_de_group,
        distance_group,
        plot_stub,
        # color=np.unique(pd_result['cluster']),
        name_x='pm_RA * cos(DEC) (mas/yr)',
        name_y='pm_DEC (mas/yr)',
        name_z='d (kpc)',
        # string='_3D_cluster_',
        pm_ra=pm_ra_object,
        pm_dec=pm_de_object,
        display=True,
        file_type=file_type_plots,
    )

    #   Get user input
    # plots.D3_scatter(
    # [pd_result['pmRA']],
    # [pd_result['pmDE']],
    # [pd_result['distance']],
    # image.outpath.name,
    # color=[pd_result['cluster']],
    # name_x='pm_RA * cos(DEC) (mas/yr)',
    # name_y='pm_DEC (mas/yr)',
    # name_z='d (kpc)',
    # string='_3D_cluster_',
    # )

    #   Get user input
    if cluster_selection_id is not None:
        cluster_id = int(cluster_selection_id)
        terminal_output.print_to_terminal(
            f"Using configured cluster id: {cluster_id}",
            indent=2,
            style_name="INFO",
        )
    else:
        cluster_id_raw, timed_out = base_utilities.get_input(
            style.Bcolors.OKBLUE +
            "\n   Which one is the correct cluster (id)? \n"
            + style.Bcolors.ENDC,
            timeout=300,
        )
        if timed_out or cluster_id_raw is None or str(cluster_id_raw).strip() == "":
            cluster_id = 0
        else:
            parsed = base_utilities.parse_cluster_selection_id(cluster_id_raw)
            if parsed is None:
                terminal_output.print_to_terminal(
                    f"Could not parse cluster id from {cluster_id_raw!r}; using 0.",
                    indent=2,
                    style_name="WARNING",
                )
                cluster_id = 0
            else:
                cluster_id = parsed
                available = np.unique(pd_result["cluster"])
                if cluster_id not in available:
                    terminal_output.print_to_terminal(
                        f"Cluster id {cluster_id} not in {sorted(available.tolist())}; "
                        "using 0.",
                        indent=2,
                        style_name="WARNING",
                    )
                    cluster_id = 0

    #   Calculated mask according to user input
    cluster_mask = pd_result['cluster'] == cluster_id

    #   Apply correlation results and masks to the input table
    tbl = tbl[id_img][mask][cluster_mask.values]

    #   Make star map
    #
    plot_starmap_from_imaging_context(
        ctx,
        tbl,
        filter_=ctx.filter_name,
        x_name="x",
        y_name="y",
        rts_pre="selected cluster members",
        label="Cluster members based on proper motion and distance evaluation",
        add_image_id=False,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )

    #   Return table
    return tbl, id_img, mask, cluster_mask.values


__all__ = [
    "find_cluster",
    "proper_motion_selection",
    "region_selection",
]
