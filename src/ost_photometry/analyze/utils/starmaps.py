"""Starmap plotting helpers for extraction and correlation."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table

from ... import utilities as base_utilities
from ... import terminal_output
from .. import plots

if TYPE_CHECKING:
    from ..models import ImageSeries
    from ..observation import Observation

def prepare_and_plot_starmap(
        image: base_utilities.Image,
        terminal_logger: terminal_output.TerminalLog | None = None,
        tbl: Table | None = None, x_name: str = 'x_fit', y_name: str = 'y_fit',
        rts_pre: str = 'image',
        label: str = 'Stars with photometric extractions',
        add_image_id: bool = True,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Creates a star map using information from an Image object

    Parameters
    ----------
    image
        Object with all image specific properties

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    tbl
        Table with position information.
        Default is ``None``.

    x_name
        Name of the X column in ``tbl``.
        Default is ``x_fit``.

    y_name
        Name of the Y column in ``tbl``.
        Default is ``y_fit``.

    rts_pre
        Expression used in the plot title / filename stem (image basename may
        be added to the title only when ``add_image_id`` is True).

    label
        String that characterizes the star map.
        Default is ``Stars with photometric extractions``.

    add_image_id
        If ``True`` the image ID (and file name) are added to the plot title;
        only the image ID is used in the output filename stem.
        Default is ``True``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Get table, data, filter, & object name
    if tbl is None:
        tbl = image.photometry
    data = image.get_data()
    filter_ = image.filter_

    #   Prepare table
    n_stars = len(tbl)
    tbl_xy = Table(
        names=['id', 'x_centroid', 'y_centroid'],
        data=[np.arange(0, n_stars), tbl[x_name], tbl[y_name]],
    )

    #   Title may include image basename; filename stem must not.
    title_rts = rts_pre
    filename_suffix = rts_pre
    if add_image_id:
        name = getattr(image, "filename", None) or Path(getattr(image, "path", "")).name
        filename_suffix = f"{rts_pre}: {image.image_id}"
        if name:
            title_rts = f"{rts_pre}: {image.image_id} ({name})"
        else:
            title_rts = filename_suffix

    #   Plot star map
    plots.starmap(
        image.out_path.name,
        data,
        filter_,
        tbl_xy,
        label=label,
        rts=title_rts,
        filename_suffix=filename_suffix,
        wcs_image=image.wcs,
        use_wcs_projection=use_wcs_projection_for_star_maps,
        terminal_logger=terminal_logger,
        file_type=file_type_plots,
    )


def prepare_and_plot_starmap_from_observation(
        observation: 'analyze.Observation', filter_list: list[str],
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Creates a star map using information from an observation container

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        List with filter names

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    terminal_output.print_to_terminal(
        "Plot star maps with positions from the final correlation",
        indent=1,
        style_name='NORMAL',
    )

    for filter_ in filter_list:
        rts = 'final version'

        #   Get reference image
        image = observation.image_series_dict[filter_].reference_image

        #   Using multiprocessing to create the plot
        p = mp.Process(
            target=plots.starmap,
            args=(
                image.out_path.name,
                image.get_data(),
                filter_,
                image.photometry,
            ),
            kwargs={
                'rts': rts,
                'label': f'Stars identified in {filter_list[0]} and '
                         f'{filter_list[1]} filter',
                'wcs_image': image.wcs,
                'use_wcs_projection': use_wcs_projection_for_star_maps,
                'file_type': file_type_plots,
            }
        )
        p.start()
    terminal_output.print_to_terminal('')


def prepare_and_plot_starmap_from_image_series(
        image_series: 'analyze.ImageSeries',
        calib_xs: np.ndarray | list[float], calib_ys: np.ndarray | list[float],
        plots_for_all_images: bool = False,
        use_wcs_projection_for_star_maps: bool = True,
        file_type_plots: str = 'pdf') -> None:
    """
    Creates a star map using information from an image series

    Parameters
    ----------
    image_series
        Image image_series class object

    calib_xs
        Position of the calibration objects on the image in pixel
        in X direction

    calib_ys
        Position of the calibration objects on the image in pixel
        in Y direction

    plots_for_all_images
        If True star map plots for all stars are created
        Default is ``False``.

    use_wcs_projection_for_star_maps
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    file_type_plots
        Type of plot file to be created
        Default is ``pdf``.
    """
    terminal_output.print_to_terminal(
        "Plot star map with objects identified on all images",
        style_name='NORMAL',
        indent=2,
    )

    #   Get image IDs, IDs of the objects, and pixel coordinates
    img_ids = image_series.get_image_ids()

    #   Make new table with the position of the calibration stars
    tbl_xy_calib = Table(
        names=['x_centroid', 'y_centroid'],
        data=[[calib_xs], [calib_ys]]
    )

    #   Make the plot using multiprocessing
    for j, image_id in enumerate(img_ids):
        if not plots_for_all_images and j != image_series.reference_image_index:
            continue
        img = image_series.image_list[j]
        fname = getattr(img, "filename", None) or ""
        filename_suffix = f"image: {image_id}, final version"
        rts = (
            f"image: {image_id} ({fname}), final version"
            if fname
            else filename_suffix
        )
        p = mp.Process(
            target=plots.starmap,
            args=(
                image_series.out_path.name,
                img.get_data(),
                image_series.filter_,
                img.photometry,
            ),
            kwargs={
                'tbl_2': tbl_xy_calib,
                'rts': rts,
                'filename_suffix': filename_suffix,
                'label': 'Stars identified in all images',
                # 'label_2': 'Calibration stars',
                'label_2': 'Objects of interest',
                'wcs_image': image_series.wcs,
                'use_wcs_projection': use_wcs_projection_for_star_maps,
                'file_type': file_type_plots,
            }
        )
        p.start()
        terminal_output.print_to_terminal('')


__all__ = [
    "prepare_and_plot_starmap",
    "prepare_and_plot_starmap_from_observation",
    "prepare_and_plot_starmap_from_image_series",
]
