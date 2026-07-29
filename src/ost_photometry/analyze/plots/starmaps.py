"""Starmap, annotation, and image-comparison plots."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from photutils.utils import ImageDepth

from ... import checks, style, terminal_output

plt.switch_backend("Agg")

def compare_images(
        output_dir: str, original_image: np.ndarray,
        comparison_image: np.ndarray, file_type: str = 'pdf') -> None:
    """
    Plot two images for comparison

    Parameters
    ----------
    output_dir
        Output directory

    original_image
        Original image data

    comparison_image
        Comparison image data

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Prepare plot
    plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    #   Original image: normalize and plot
    norm = simple_norm(original_image.data, 'log', percent=99.)
    ax1.imshow(original_image.data, norm=norm, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Original image')

    #   Comparison image: normalize and plot
    norm = simple_norm(comparison_image, 'log', percent=99.)
    ax2.imshow(comparison_image, norm=norm, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Downloaded image')

    #   Save the plot
    plt.savefig(
        f'{output_dir}/img_comparison.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()

def _sanitize_starmap_filename_part(text: str) -> str:
    replace_dict = {
        ',': '', '.': '', '\\': '', '[': '', '&': '', ' ': '_',
        ':': '', ']': '', '{': '', '}': '', '(': '', ')': '',
    }
    for key, value in replace_dict.items():
        text = text.replace(key, value)
    return text.lower()


def starmap(
        output_dir: str, image: np.ndarray, filter_: str, tbl: Table,
        tbl_2: Table = None, label: str = 'Identified stars',
        label_2: str = 'Identified stars (set 2)', rts: str | None = None,
        mode: str | None = None, magnitude_column: str | None = None,
        name_object: str | None = None,
        wcs_image: wcs.WCS = None, use_wcs_projection: bool = True,
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type: str = 'pdf', indent: int = 2,
        filename_suffix: str | None = None) -> None:
    """
    Plot star maps  -> overlays of the determined star positions on FITS
                    -> supports different versions

    Parameters
    ----------
    output_dir
        Output directory

    image
        The image data

    filter_
        Filter identifier

    tbl
        Astropy table with data of the objects

    tbl_2
        Second astropy table with data of special objects
        Default is ``None``

    label
        Identifier for the objects in `tbl`
        Default is ``Identified stars``

    label_2
        Identifier for the objects in `tbl_2`
        Default is ``Identified stars (set 2)``

    rts
        Expression characterizing the plot (used in the title / log message).
        Default is ``None``

    filename_suffix
        Optional stem used in the output filename instead of ``rts``.
        Prefer this when the title should include an image basename that must
        not appear in the path. Default is ``None`` (fall back to ``rts``).

    mode
        String used to switch between different plot modes
        Default is ``None``

    magnitude_column
        When ``mode='mags'``, use this column for labels (e.g. ``mag_cal_V``).
        If ``None``, uses ``mag_cali_trans`` or ``mag_cali_no-trans``.
        Default is ``None``.

    name_object
        Name of the object
        Default is ``None``

    wcs_image
        WCS information
        Default is ``None``

    use_wcs_projection
        If ``True`` the starmap will be plotted with sky coordinates instead
        of pixel coordinates
        Default is ``True``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    indent
        Indentation for the console output lines
        Default is ``2``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'starmaps'),
    )

    if rts is not None:
        if terminal_logger is not None:
            terminal_logger.add_to_cache(
                f"Plot {filter_} band image with stars overlaid ({rts})",
                style_name='NORMAL',
                indent=indent,
            )
        else:
            terminal_output.print_to_terminal(
                f"Plot {filter_} band image with stars overlaid ({rts})",
                style_name='NORMAL',
                indent=indent,
            )

    #   Check if column with X and Y coordinates are available for table 1
    if 'x' in tbl.colnames:
        x_column = 'x'
        y_column = 'y'
    elif 'x_centroid' in tbl.colnames:
        x_column = 'x_centroid'
        y_column = 'y_centroid'
    elif 'xfit' in tbl.colnames:
        x_column = 'xfit'
        y_column = 'yfit'
    elif 'x_fit' in tbl.colnames:
        x_column = 'x_fit'
        y_column = 'y_fit'
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nNo valid X and Y column found for "
            f"table 1. {style.Bcolors.ENDC}"
        )
    #   Check if column with X and Y coordinates are available for table 2
    if tbl_2 is not None:
        if 'x' in tbl_2.colnames:
            x_column_2 = 'x'
            y_column_2 = 'y'
        elif 'x_centroid' in tbl_2.colnames:
            x_column_2 = 'x_centroid'
            y_column_2 = 'y_centroid'
        elif 'xfit' in tbl_2.colnames:
            x_column_2 = 'xfit'
            y_column_2 = 'yfit'
        else:
            raise RuntimeError(
                f"{style.Bcolors.FAIL} \nNo valid X and Y column found for "
                f"table 2. {style.Bcolors.ENDC}"
            )

    #   Set layout
    fig = plt.figure(figsize=(20, 9))

    if not use_wcs_projection:
        ax = fig.add_subplot()
    else:
        if wcs_image is not None:
            ax = plt.subplot(projection=wcs_image)
        else:
            terminal_output.print_to_terminal(
                f"Sky projection for master plot not possible, since no WCS "
                f"was provided. Use Pixel coordinates instead.",
                style_name='WARNING',
                indent=indent,
            )
            ax = fig.add_subplot()


    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title of the complete plot
    if rts is None and name_object is None:
        sub_title = f'Star map ({filter_} filter)'
    elif rts is None:
        sub_title = f'{name_object} - {filter_} filter'
    elif name_object is None:
        sub_title = f'{filter_} filter, {rts}'
    else:
        sub_title = f'{name_object} - {filter_} filter, {rts}'

    fig.suptitle(sub_title, fontsize=17)

    #   Set up normalization for the image
    norm = ImageNormalize(image, interval=ZScaleInterval(contrast=0.15, ))

    #   Display the actual image
    ax.imshow(
        image,
        cmap='PuBu',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )

    #   Plot apertures
    ax.scatter(
        tbl[x_column],
        tbl[y_column],
        s=40,
        facecolors=(0.5, 0., 0.5, 0.2),
        edgecolors=(0.5, 0., 0.5, 0.7),
        lw=0.9,
        label=label,
    )
    if tbl_2 is not None:
        ax.scatter(
            tbl_2[x_column_2],
            tbl_2[y_column_2],
            s=40,
            facecolors=(0., 0.7, 0.35, 0.2),
            edgecolors=(0., 0.7, 0.35, 0.7),
            lw=0.9,
            label=label_2,
        )

    #   Set plot limits
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(0, image.shape[0] - 1)

    # Plot labels next to the apertures
    # if isinstance(tbl[x_column], u.quantity.Quantity):
    if hasattr(tbl[x_column], "value"):
        x = tbl[x_column].value.ravel()
        y = tbl[y_column].value.ravel()
    else:
        x = tbl[x_column]
        y = tbl[y_column]

    if mode == 'mags':
        if magnitude_column is not None:
            magnitudes = tbl[magnitude_column]
        else:
            try:
                magnitudes = tbl['mag_cali_trans']
            except KeyError:
                magnitudes = tbl['mag_cali_no-trans']
        for i in range(0, len(x)):
            mag_i = magnitudes[i]
            if hasattr(mag_i, 'value'):
                mag_i = mag_i.value
            ax.text(
                x[i] + 11,
                y[i] + 8,
                f" {float(mag_i):.1f}",
                fontdict=style.font,
                color='purple',
            )
    elif mode == 'list':
        for i in range(0, len(x)):
            ax.text(
                x[i],
                y[i],
                f" {i}",
                fontdict=style.font,
                color='purple',
            )
    else:
        for i in range(0, len(x)):
            ax.text(
                x[i] + 11,
                y[i] + 8,
                f" {tbl['id'][i]}",
                fontdict=style.font,
                color='purple',
            )

    #   Define the ticks
    ax.tick_params(
        axis='both',
        which='both',
        # top=True,
        # right=True,
        direction='in',
    )
    ax.minorticks_on()

    #   Set labels
    if wcs_image is not None:
        ax.set_xlabel("Right ascension", fontsize=16)
        ax.set_ylabel("Declination", fontsize=16)
    else:
        ax.set_xlabel("[pixel]", fontsize=16)
        ax.set_ylabel("[pixel]", fontsize=16)

    #   Enable grid for WCS
    # if wcs is not None:
    ax.grid(True, color='white', linestyle='--')

    #   Plot legend
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1.0, 0.102),
        loc=3,
        ncol=2,
        mode='expand',
        borderaxespad=0.,
    )

    #   Write the plot to disk (title may include image basename; filename must not)
    stem_source = filename_suffix if filename_suffix is not None else rts
    if stem_source is None:
        plt.savefig(
            f'{output_dir}/starmaps/starmap_{filter_}.{file_type}',
            bbox_inches='tight',
            format=file_type,
        )
    else:
        stem = _sanitize_starmap_filename_part(stem_source)
        plt.savefig(
            f"{output_dir}/starmaps/starmap_{filter_}_{stem}.{file_type}",
            bbox_inches='tight',
            format=file_type,
        )
    # plt.show()
    plt.close()



def plot_limiting_mag_sky_apertures(
        output_dir: str, img_data: np.ndarray, mask: np.ndarray,
        image_depth: ImageDepth, file_type: str = 'pdf') -> None:
    """
    Plot the sky apertures that are used to estimate the limiting magnitude

    Parameters
    ----------
    output_dir
        Output directory

    img_data
        Image data

    mask
        Indicating the position of detected objects

    image_depth
        Object used to derive the limiting magnitude

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'limiting_mag'),
    )

    #   Plot magnitudes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

    #   Set title
    ax[0].set_title('Data with blank apertures')
    ax[1].set_title('Mask with blank apertures')

    #   Normalize the image data and plot
    norm = ImageNormalize(img_data, interval=ZScaleInterval(contrast=0.15, ))
    ax[0].imshow(
        img_data,
        norm=norm,
        cmap='PuBu',
        interpolation='nearest',
        origin='lower',
    )

    #   Plot mask with object positions
    ax[1].imshow(
        mask,
        interpolation='none',
        origin='lower',
    )

    #   Plot apertures used to derive limiting magnitude
    image_depth.apertures[0].plot(ax[0], color='purple', lw=0.2)
    image_depth.apertures[0].plot(ax[1], color='orange', lw=0.2)

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.05,
        top=0.95,
        wspace=0.2,
    )

    #   Set labels
    label_font_size = 10
    ax[0].set_xlabel("[pixel]", fontsize=label_font_size)
    ax[0].set_ylabel("[pixel]", fontsize=label_font_size)
    ax[1].set_xlabel("[pixel]", fontsize=label_font_size)
    ax[1].set_ylabel("[pixel]", fontsize=label_font_size)

    #   Save plot
    plt.savefig(
        f'{output_dir}/limiting_mag/limiting_mag_sky_regions.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()



def plot_annotated_image(
        image_data: np.ndarray, wcs_image: wcs.WCS, simbad_objects: Table,
        output_dir: Path, filter_: str, file_type: str = 'pdf',
        filter_mag: str | None = None, mag_limit: float | None = None,
    ) -> None :
    """
    Visualises the image and marks objects from the Simbad database.

    Parameters
    ----------
    image_data
        2D image data

    wcs_image
        WCS object

    simbad_objects
        Table with Simbad objects

    output_dir
        Output directory

    filter_
        Filter identifier

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    filter_mag
        Name of the filter (e.g. 'V')
        Default is ``None``.

    mag_limit
        Limiting magnitude, only objects brighter as this limit will be shown
        Default is ``None``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'starmaps'),
    )

    #   Setup figure
    fig, ax = plt.subplots(figsize=(20, 9), subplot_kw={'projection': wcs_image})

    #   Set up normalization for the image
    norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.1, ))

    #   Display the actual image
    ax.imshow(
        image_data,
        cmap='gray',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )

    #   Define the ticks
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
    )
    ax.minorticks_on()

    #   Set labels
    ax.set_xlabel("Right ascension", fontsize=16)
    ax.set_ylabel("Declination", fontsize=16)

    #   Enable grid for WCS
    # if wcs is not None:
    ax.grid(True, color='white', linestyle='--')

    #   Setup list for legend
    legend_elements = []

    for obj in simbad_objects:
        if 'ra' in obj.colnames:
           simbad_objects.rename_column('ra', 'RA')
        if 'dec' in obj.colnames:
            simbad_objects.rename_column('dec', 'DEC')
        if 'otype' in obj.colnames:
            simbad_objects.rename_column('otype', 'OTYPE')
        if 'main_id' in obj.colnames:
            simbad_objects.rename_column('main_id', 'MAIN_ID')

        ra, dec = obj['RA'], obj['DEC']
        obj_type = obj['OTYPE']
        name = obj['MAIN_ID']

        #   Check that the magnitude is available and meets the filter
        #   and magnitude limit conditions
        if filter_mag and mag_limit is not None:
            mag_col = f'FLUX_{filter_mag.upper()}'
            if (mag_col not in obj.colnames or obj[mag_col] is None or
                    isinstance(obj[mag_col], np.ma.core.MaskedConstant) or obj[mag_col] > mag_limit):
                continue

        #   Conversion of world coordinates to image coordinates
        coord = SkyCoord(ra=ra, dec=dec, unit=("hourangle", "deg"))
        x, y = wcs_image.world_to_pixel(coord)

        #   Check if the objects are actually within the image boundaries
        if 0 <= x < image_data.shape[1] and 0 <= y < image_data.shape[0]:
            # print(obj_type)
            #   Select icon and colour based on the object type
            plot_marker = False
            if 'Star' in obj_type:
                color, marker = 'lightblue', '*'
                plot_marker = True

                if not any(e.get_label() == 'Star' for e in legend_elements):
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            marker=marker,
                            markerfacecolor='none',
                            markersize=8,
                            linestyle='None',
                            label='Star',
                        )
                    )


            elif obj_type in ['Galaxy', 'Seyfert1', 'Seyfert2', 'AGN_Candidate', 'QSO']:
                color = 'lightsalmon'
                #   Test if object dimension is available
                if 'DIMENSIONS' in obj.colnames and obj['DIMENSIONS'] is not None:
                    dimensions = obj['DIMENSIONS']
                    # print(dimensions)
                    try:
                        major_axis, minor_axis = [float(dim) for dim in dimensions.split('x')]
                        #   TODO: Check if rotation information is available
                        angle = 0

                        #   Convert arc minute to pixel
                        major_axis_px = (major_axis / 60.0) / wcs.wcs.cdelt[0]
                        minor_axis_px = (minor_axis / 60.0) / wcs.wcs.cdelt[1]

                        #   Draw ellipse
                        ellipse = Ellipse(
                            (x, y),
                            width=major_axis_px,
                            height=minor_axis_px,
                            angle=angle,
                            edgecolor=color,
                            facecolor='none',
                            lw=1.5,
                            alpha=0.7,
                        )
                        ax.add_patch(ellipse)
                    except ValueError:
                        pass
                else:
                    #   No dimension tag -> set default marker
                    marker = 's'
                    plot_marker = True

                if not any(e.get_label() == 'Galaxy' for e in legend_elements):
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            marker=marker,
                            markerfacecolor='none',
                            markersize=8,
                            linestyle='None',
                            label='Galaxy',
                        )
                    )

            elif 'Nebula' in obj_type:
                color, marker = 'lightpink', 'o'
                plot_marker = True

                if not any(e.get_label() == 'Nebula' for e in legend_elements):
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            marker=marker,
                            markerfacecolor='none',
                            markersize=8,
                            linestyle='None',
                            label='Nebula',
                        )
                    )

            else:
                color, marker = 'lightgreen', 'H'
                plot_marker = True
                name = f'{name} ({obj_type})'

                if not any(e.get_label() == 'Other' for e in legend_elements):
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color,
                            marker=marker,
                            markerfacecolor='none',
                            markersize=8,
                            linestyle='None',
                            label='Other',
                        )
                    )

            #   Mark objects
            if plot_marker:
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    markersize=11,
                    alpha=0.8,
                )
            ax.text(
                x + 70,
                y,
                name,
                color=color,
                fontsize=8,
                alpha=0.9,
                verticalalignment='center',
                weight="bold",
            )


    #   Add legend
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1.0, 0.102),
        loc=3,
        handles=legend_elements,
        ncol=5,
        fontsize=8,
        frameon=True,
        mode='expand',
        borderaxespad=0.,
    )

    #   Save plot
    plt.savefig(
        output_dir / f'starmaps/annotated_starmap_{filter_}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()
    # plt.show()


