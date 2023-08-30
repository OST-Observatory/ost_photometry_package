############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

import os

from pathlib import Path

import itertools

from astropy.visualization import (
    ImageNormalize,
    ZScaleInterval,
    simple_norm,
)

from astropy.stats import sigma_clip as sigma_clipping
from astropy.time import Time
from astropy.timeseries import aggregate_downsample
import astropy.units as u

from itertools import cycle

from .. import checks, style, terminal_output

import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

# plt.switch_backend('TkAgg')

############################################################################
#                           Routines & definitions                         #
############################################################################


def compare_images(output_dir, original_image, comparison_image):
    """
        Plot two images for comparison

        Parameters
        ----------
        output_dir          : `string`
            Output directory

        original_image      : `numpy.ndarray`
            Original image data

        comparison_image    : `numpy.ndarray`
            Comparison image data
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
        f'{output_dir}/img_comparison.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def starmap(output_dir, image, filter_, tbl, tbl_2=None,
            label='Identified stars', label_2='Identified stars (set 2)',
            rts=None, mode=None, name_obj=None, terminal_logger=None,
            indent=2):
    """
        Plot star maps  -> overlays of the determined star positions on FITS
                        -> supports different versions

        Parameters
        ----------
        output_dir      : `string`
            Output directory

        image           : `numpy.ndarray`
            Image data

        filter_         : `string`
            Filter identifier

        tbl             : `astropy.table.Table`
            Astropy table with data of the objects

        tbl_2           : `astropy.table.Table`, optional
            Second astropy table with data of special objects
            Default is ``None``

        label           : `string`, optional
            Identifier for the objects in `tbl`
            Default is ``Identified stars``

        label_2         : `string`, optional
            Identifier for the objects in `tbl_2`
            Default is ``Identified stars (set 2)``

        rts             : `string`, optional
            Expression characterizing the plot
            Default is ``None``

        mode            : `string`, optional
            String used to switch between different plot modes
            Default is ``None``

        name_obj        : `string`, optional
            Name of the object
            Default is ``None``

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
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
                indent=indent,
            )
        else:
            terminal_output.print_to_terminal(
                f"Plot {filter_} band image with stars overlaid ({rts})",
                indent=indent,
            )

    #   Check if column with X and Y coordinates are available for table 1
    if 'x' in tbl.colnames:
        x_column = 'x'
        y_column = 'y'
    elif 'xcentroid' in tbl.colnames:
        x_column = 'xcentroid'
        y_column = 'ycentroid'
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
        elif 'xcentroid' in tbl_2.colnames:
            x_column_2 = 'xcentroid'
            y_column_2 = 'ycentroid'
        elif 'xfit' in tbl_2.colnames:
            x_column_2 = 'xfit'
            y_column_2 = 'yfit'
        else:
            raise RuntimeError(
                f"{style.Bcolors.FAIL} \nNo valid X and Y column found for "
                f"table 2. {style.Bcolors.ENDC}"
            )

    #   Set layout of image
    fig = plt.figure(figsize=(20, 9))

    #   Set title of the complete plot
    if rts is None and name_obj is None:
        sub_title = f'Star map ({filter_} band)'
    elif rts is None:
        sub_title = f'Star map ({filter_} band) - {name_obj}'
    elif name_obj is None:
        sub_title = f'Star map ({filter_} band, {rts})'
    else:
        sub_title = f'Star map ({filter_} band, {rts}) - {name_obj}'

    fig.suptitle(sub_title, fontsize=17)

    #   Set up normalization for the image
    norm = ImageNormalize(image, interval=ZScaleInterval(contrast=0.15,))

    #   Display the actual image
    plt.imshow(
        image,
        cmap='PuBu',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )

    #   Plot apertures
    plt.scatter(
        tbl[x_column],
        tbl[y_column],
        s=40,
        facecolors='none',
        edgecolors='purple',
        alpha=0.7,
        lw=0.9,
        label=label,
    )
    if tbl_2 is not None:
        plt.scatter(
            tbl_2[x_column_2],
            tbl_2[y_column_2],
            s=40,
            facecolors='none',
            edgecolors='#02c14d',
            # alpha=0.7,
            lw=0.9,
            label=label_2,
        )

    #   Set plot limits
    plt.xlim(0, image.shape[1] - 1)
    plt.ylim(0, image.shape[0] - 1)

    # Plot labels next to the apertures
    if isinstance(tbl[x_column], u.quantity.Quantity):
        x = tbl[x_column].value
        y = tbl[y_column].value
    else:
        x = tbl[x_column]
        y = tbl[y_column]
    if mode == 'mags':
        for i in range(0, len(x)):
            plt.text(
                x[i],
                y[i],
                f" {tbl['mags_fit'][i]:.1f}",
                fontdict=style.font,
                color='purple',
            )
    elif mode == 'list':
        for i in range(0, len(x)):
            plt.text(
                x[i],
                y[i],
                f" {i}",
                fontdict=style.font,
                color='purple',
            )
    else:
        for i in range(0, len(x)):
            plt.text(
                x[i] + 11,
                y[i] + 7,
                f" {tbl['id'][i]}",
                fontdict=style.font,
                color='purple',
            )

    #   Define the ticks
    plt.tick_params(axis='both', which='both', top=True, right=True,
                    direction='in')
    plt.minorticks_on()

    #   Set labels
    plt.xlabel("[pixel]", fontsize=16)
    plt.ylabel("[pixel]", fontsize=16)

    #   Plot legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1.0, 0.102), loc=3, ncol=2,
               mode='expand', borderaxespad=0.)

    #   Write the plot to disk
    if rts is None:
        plt.savefig(
            f'{output_dir}/starmaps/starmap_{filter_}.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    else:
        plt.savefig(
            f'{output_dir}/starmaps/starmap_{filter_}_{rts}.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    # plt.show()
    plt.close()


def plot_apertures(output_dir, image, aperture, annulus_aperture,
                   filename_string):
    """
        Plot the apertures used for extracting the stellar fluxes
               (star map plot for aperture photometry)

        Parameters
        ----------
        output_dir          : `string`
            Output directory

        image               : `numpy.ndarray`
            Image data (2D)

        aperture            : `photutils.aperture.CircularAperture`
            Apertures used to extract the stellar flux

        annulus_aperture    : `photutils.aperture.CircularAnnulus`
            Apertures used to extract the background flux

        filename_string     : `string`
            String characterizing the output file
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'aperture'),
    )

    ###
    #   Make plot
    #
    plt.figure(figsize=(20, 9))

    #   Normalize the image
    norm = ImageNormalize(image, interval=ZScaleInterval())

    #   Plot the image
    plt.imshow(
        image,
        cmap='viridis',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )

    #   Plot stellar apertures
    ap_patches = aperture.plot(
        color='lightcyan',
        lw=0.2,
        label='Object aperture',
    )

    #   Plot background apertures
    ann_patches = annulus_aperture.plot(
        color='darkred',
        lw=0.2,
        label='Background annulus',
    )

    #
    handles = (ap_patches[0], ann_patches[0])

    #   Plot legend
    plt.legend(
        loc=(0.17, 0.05),
        facecolor='#458989',
        labelcolor='white',
        handles=handles,
        prop={'weight': 'bold', 'size': 9},
    )

    #   Save figure
    plt.savefig(
        f'{output_dir}/aperture/aperture_{filename_string}.pdf',
        bbox_inches='tight',
        format='pdf',
    )

    #   Set labels
    plt.xlabel("[pixel]", fontsize=16)
    plt.ylabel("[pixel]", fontsize=16)

    plt.close()


def plot_cutouts(output_dir, stars, identifier, terminal_logger=None,
                 max_plot_stars=25, name_obj=None, indent=2):
    """
        Plot the cutouts of the stars used to estimate the ePSF

        Parameters
        ----------
        output_dir      : `string`
            Output directory

        stars           : `numpy.ndarray`
            Numpy array with cutouts of the ePSF stars

        identifier      : `string`
            String characterizing the plot

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        max_plot_stars  : `integer`, optional
            Maximum number of cutouts to plot
            Default is ``25``.

        name_obj        : `string`, optional
            Name of the object
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines.
            Default is ``2``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'cutouts'),
    )

    #   Set number of cutouts
    if len(stars) > max_plot_stars:
        n_cutouts = max_plot_stars
    else:
        n_cutouts = len(stars)

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            "Plot ePSF cutouts ({string})",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            "Plot ePSF cutouts ({string})",
            indent=indent,
        )

    #   Plot the first cutouts (default: 25)
    #   Set number of rows and columns
    n_rows = 5
    n_columns = 5

    #   Prepare plot
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(20, 15),
                           squeeze=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.25)

    #   Set title of the complete plot
    if name_obj is None:
        sub_title = f'Cutouts of the {n_cutouts} faintest stars ({identifier})'
    else:
        sub_title = f'Cutouts of the {n_cutouts} faintest stars ({identifier}) - {name_obj}'
    fig.suptitle(sub_title, fontsize=17)

    ax = ax.ravel()  # flatten the image?

    #   Loop over the cutouts (default: 25)
    for i in range(n_cutouts):
        # Remove bad pixels that would spoil the image normalization
        data_image = np.where(stars[i].data <= 0, 1E-7, stars[i].data)
        # Set up normalization for the image
        norm = simple_norm(data_image, 'log', percent=99.)
        # Plot individual cutouts
        ax[i].set_xlabel("Pixel")
        ax[i].set_ylabel("Pixel")
        ax[i].imshow(data_image, norm=norm, origin='lower', cmap='viridis')
    plt.savefig(
        f'{output_dir}/cutouts/cutouts_{identifier}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    # plt.show()
    plt.close()


def plot_epsf(output_dir, epsf, name_obj=None, terminal_logger=None, indent=1):
    """
        Plot the ePSF image of all filters

        Parameters
        ----------
        output_dir      : `string`
            Output directory

        epsf            : `epsf.object` ???
            ePSF object, usually constructed by epsf_builder

        name_obj         : `string`, optional
            Name of the object
            Default is ``None``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'epsfs'),
    )

    if terminal_logger is not None:
        terminal_logger.add_to_cache("Plot ePSF image", indent=indent)
    else:
        terminal_output.print_to_terminal("Plot ePSF image", indent=indent)

    #   Set font size
    rcParams['font.size'] = 13

    #   Set up plot
    n_plots = len(epsf)
    if n_plots == 1:
        fig = plt.figure(figsize=(6, 5))
    elif n_plots == 2:
        fig = plt.figure(figsize=(13, 5))
    else:
        fig = plt.figure(figsize=(20, 15))

    #   Set title of the complete plot
    if name_obj is None:
        fig.suptitle('ePSF', fontsize=17)
    else:
        fig.suptitle(f'ePSF ({name_obj})', fontsize=17)

    #   Plot individual subplots
    for i, (band, eps) in enumerate(epsf.items()):
        #   Remove bad pixels that would spoil the image normalization
        epsf_clean = np.where(eps.data <= 0, 1E-7, eps.data)
        #   Set up normalization for the image
        norm = simple_norm(epsf_clean, 'log', percent=99.)

        #   Make the subplots
        if n_plots == 1:
            ax = fig.add_subplot(1, 1, i + 1)
        elif n_plots == 2:
            ax = fig.add_subplot(1, 2, i + 1)
        else:
            ax = fig.add_subplot(n_plots, n_plots, i + 1)

        #   Plot the image
        im1 = ax.imshow(epsf_clean, norm=norm, origin='lower',
                        cmap='viridis')

        #   Set title of subplot
        ax.set_title(band)

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im1, ax=ax)

    if n_plots >= 2:
        plt.savefig(
            f'{output_dir}/epsfs/epsfs_multiple_filter.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    else:
        plt.savefig(
            f'{output_dir}/epsfs/epsf.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    # plt.show()
    plt.close()


def plot_residual(name, image_orig, residual_image, output_dir,
                  name_obj=None, terminal_logger=None, indent=1):
    """
        Plot the original and the residual image

        Parameters
        ----------
        name            : `string`
            Name of the plot

        image_orig      : `dictionary` {`string`: `numpy.ndarray`}
            Original image data

        residual_image  : `dictionary` {`string`: `numpy.ndarray`}
            Residual image data

        output_dir          : `string`
            Output directory

        name_obj         : `string`, optional
            Name of the object
            Default is ``None``.

        terminal_logger : `terminal_output.TerminalLog` or None, optional
            Logger object. If provided, the terminal output will be directed
            to this object.
            Default is ``None``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'residual'),
    )

    if terminal_logger is not None:
        terminal_logger.add_to_cache(
            "Plot original and the residual image",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            "Plot original and the residual image",
            indent=indent,
        )

    #   Set font size
    rcParams['font.size'] = 13

    #   Set up plot
    n_plots = len(image_orig)
    if n_plots == 1:
        fig = plt.figure(figsize=(20, 5))
    elif n_plots == 2:
        fig = plt.figure(figsize=(20, 10))
    else:
        fig = plt.figure(figsize=(20, 20))

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=None,
        hspace=0.25,
    )

    #   Set title of the complete plot
    if name_obj is None:
        fig.suptitle(name, fontsize=17)
    else:
        fig.suptitle(f'{name} ({name_obj})', fontsize=17)

    i = 1
    for band, image in image_orig.items():
        #   Plot original image
        #   Set up normalization for the image
        norm = ImageNormalize(image, interval=ZScaleInterval())

        if n_plots == 1:
            ax = fig.add_subplot(1, 2, i)
        elif n_plots == 2:
            ax = fig.add_subplot(2, 2, i)
        else:
            ax = fig.add_subplot(n_plots, 2, i)

        #   Plot image
        im1 = ax.imshow(
            image,
            norm=norm,
            cmap='viridis',
            aspect=1,
            interpolation='nearest',
            origin='lower',
        )

        #   Set title of subplot
        ax.set_title(f'Original Image ({band})')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im1, ax=ax)

        i += 1

        #   Plot residual image
        #   Set up normalization for the image
        norm = ImageNormalize(residual_image[band],
                              interval=ZScaleInterval())

        if n_plots == 1:
            ax = fig.add_subplot(1, 2, i)
        elif n_plots == 2:
            ax = fig.add_subplot(2, 2, i)
        else:
            ax = fig.add_subplot(n_plots, 2, i)

        #   Plot image
        im2 = ax.imshow(
            residual_image[band],
            norm=norm,
            cmap='viridis',
            aspect=1,
            interpolation='nearest',
            origin='lower',
        )

        #   Set title of subplot
        ax.set_title(f'Residual Image ({band})')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im2, ax=ax)

        i += 1

    #   Write the plot to disk
    if n_plots == 1:
        plt.savefig(
            f'{output_dir}/residual/residual_images_multiple_filter.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    else:
        plt.savefig(
            f'{output_dir}/residual/residual_images.pdf',
            bbox_inches='tight',
            format='pdf',
        )
    # plt.show()
    plt.close()


#   TODO: Check if the following plot is used or not
# def sigma_plot(bv, mags, bands, band, nr, outdir, nameobj=None, fit=None):
#     """
#         Illustrate sigma clipping of magnitudes
#
#         Parameters
#         ----------
#         bv          : `numpy.ndarray`
#             Delta color - (mag_2-mag_1)_observed - (mag_2-mag_1)_literature
#
#         mags        : `numpy.ndarray`
#             Magnitudes
#
#         bands       : `list` of `string`
#             Filter list
#
#         band        : `list` of `string`
#             Filter name
#
#         nr          : `integer`
#             Number of the star to plot
#
#         outdir      : `string`
#             Output directory
#
#         nameobj     : `string`, optional
#             Name of the object
#             Default is ``None``.
#
#         fit             : ` astropy.modeling.fitting` instance, optional
#             Fit to plot
#             Default is ``None``.
#     """
#     #   Check output directories
#     checks.check_out(
#         outdir,
#         os.path.join(outdir, 'sigmag'),
#     )
#
#     #   Sigma clip magnitudes
#     clip = sigma_clipping(mags, sigma=1.5)
#     mask = np.invert(clip.recordmask)
#     clip_bv = bv[mask]
#     mag_clip = mags[mask]
#
#     #   Plot sigma clipped magnitudes
#     fig = plt.figure(figsize=(8, 8))
#
#     #   Set title
#     if nameobj is None:
#         sub_titel = f'Sigma clipped magnitudes -- star: {nr}'
#     else:
#         sub_titel = f'Sigma clipped magnitudes -- star: {nr} ({nameobj})'
#     fig.suptitle(sub_titel, fontsize=17)
#
#     #   Plot data
#     plt.plot(mags, bv, color='blue', marker='.', linestyle='none')
#     plt.plot(mag_clip, clip_bv, color='red', marker='.', linestyle='none')
#
#     #   Plot fit
#     if fit is not None:
#         mags_sort = np.sort(mags)
#         plt.plot(
#             mags_sort,
#             fit(mags_sort),
#             color='r',
#             linewidth=3,
#             label='Polynomial fit',
#         )
#
#     #   Set x and y axis label
#     plt.xlabel(f"{band} [mag]")
#     plt.ylabel(f"Delta {bands[0]}-{bands[1]}")
#
#     #   Save plot
#     plt.savefig(
#         f'{outdir}/sigmag/{nr}_{band}.png',
#         bbox_inches='tight',
#         format='png',
#     )
#     plt.close()
#     # plt.show()


def light_curve_jd(ts, data_column, err_column, output_dir, error_bars=True,
                   name_obj=None):
    """
        Plot the light curve over Julian Date

        Parameters
        ----------
        ts          : `astropy.timeseries.TimeSeries`
            Time series

        data_column : `string`
            Filter

        err_column  : `string`
            Name of the error column

        output_dir  : `string`
            Output directory

        error_bars  : `boolean`, optional
            If True error bars will be plotted.
            Default is ``False``.

        name_obj    : `string`, optional
            Name of the object
            Default is ``None``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'lightcurve'),
    )

    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20, 9))

    #   Plot grid
    plt.grid(True, color='lightgray', linestyle='--')

    #   Set tick size
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    #   Set title
    if name_obj is None:
        fig.suptitle(f'Light curve', fontsize=30)
    else:
        fig.suptitle(f'Light curve - {name_obj}', fontsize=30)

    #   Plot data with or without error bars
    if not error_bars:
        plt.plot(ts.time.jd, ts[data_column], 'k.', markersize=3)
    else:
        plt.errorbar(
            ts.time.jd,
            np.array(ts[data_column]),
            yerr=np.array(ts[err_column]),
            fmt='k.',
            markersize=4,
            capsize=2,
            ecolor='dodgerblue',
            color='darkred',
        )

    #   Get median of the data
    median_data = np.median(ts[data_column].value)
    min_data = np.min(ts[data_column].value)
    max_data = np.max(ts[data_column].value)

    #   Invert y-axis
    if median_data > 1.5 or median_data < 0.5:
        plt.gca().invert_yaxis()

    #   Set plot limits
    y_err = ts[err_column].value
    y_err_sigma = sigma_clipping(y_err, sigma=1.5)
    max_err = np.max(y_err_sigma)

    if median_data > 1.1 or median_data < 0.9:
        y_lim = np.max([max_err * 1.5, 0.1])
        # y_lim = np.max([max_err*2.0, 0.1])
        plt.ylim([median_data + y_lim, median_data - y_lim])
        # plt.y_lim([max_data+y_lim, min_data-y_lim])
        y_label_text = ' [mag] (Vega)'
    else:
        y_lim = max_err * 1.2
        # plt.y_lim([median_data+y_lim,median_data-y_lim])
        plt.ylim([min_data - y_lim, max_data + y_lim])
        y_label_text = ' [flux] (normalized)'

    #   Set x and y axis label
    plt.xlabel('Julian Date', fontsize=15)
    plt.ylabel(data_column + y_label_text, fontsize=15)

    #   Save plot
    plt.savefig(
        f'{output_dir}/lightcurve/lightcurve_jd_{data_column}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def light_curve_fold(time_series, data_column, err_column, output_dir,
                     transit_time, period, binning_factor=None,
                     error_bars=True, name_obj=None):
    """
        Plot a folded light curve

        Parameters
        ----------
        time_series              : `astropy.timeseries.TimeSeries`
            Time series

        data_column     : `string`
            Filter

        err_column      : `string`
            Name of the error column

        output_dir      : `string`
            Output directory

        transit_time    : `string`
            Time of the transit - Format example: "2020-09-18T01:00:00"

        period          : `float`
            Period in days

        binning_factor  : `float`, optional
            Light-curve binning-factor in days
            Default is ``None``.

        error_bars      : `boolean`, optional
            If True error bars will be plotted.
            Default is ``False``.

        name_obj        : `string`, optional
            Name of the object
            Default is ``None``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'lightcurve'),
    )

    #   Make a time object for the  transit times
    transit_time = Time(transit_time, format='isot', scale='utc')

    #   Fold lightcurve
    ts_folded = time_series.fold(
        period=period * u.day,
        epoch_time=transit_time,
    )

    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20, 9))

    #   Plot grid
    plt.grid(True, color='lightgray', linestyle='--')

    #   Set tick size
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    #   Set title
    if name_obj is None:
        fig.suptitle('Folded light curve', fontsize=30)
    else:
        fig.suptitle(f'Folded light curve - {name_obj}', fontsize=30)

    #   Calculate binned lightcurve => plot
    if binning_factor is not None:
        ts_binned = aggregate_downsample(
            ts_folded,
            time_bin_size=binning_factor * u.day,
        )

        #   Remove zero entries in case the binning time is smaller than the
        #   time between the data points
        mask = np.array(ts_binned[data_column]) == 0.
        mask = np.invert(mask)

        if error_bars:
            plt.errorbar(
                ts_binned.time_bin_start.jd[mask],
                np.array(ts_binned[data_column][mask]),
                yerr=np.array(ts_binned[err_column][mask]),
                # fmt='k.',
                marker='o',
                ls='none',
                elinewidth=1,
                markersize=3,
                capsize=2,
                ecolor='dodgerblue',
                color='darkred',
            )
        else:
            plt.plot(
                ts_binned.time_bin_start.jd[mask],
                ts_binned[data_column][mask],
                'k.',
                markersize=3,
            )
    else:
        if error_bars:
            plt.errorbar(
                ts_folded.time.jd,
                np.array(ts_folded[data_column]),
                yerr=np.array(ts_folded[err_column]),
                # fmt='k.',
                marker='o',
                ls='none',
                elinewidth=1,
                markersize=3,
                capsize=2,
                ecolor='dodgerblue',
                color='darkred',
            )
        else:
            plt.plot(
                ts_folded.time.jd,
                ts_folded[data_column],
                'k.',
                markersize=3,
            )

    #   Get median of the data
    median_data = np.median(ts_folded[data_column].value)
    min_data = np.min(ts_folded[data_column].value)
    max_data = np.max(ts_folded[data_column].value)

    #   Invert y-axis
    if median_data > 1.5 or median_data < 0.5:
        plt.gca().invert_yaxis()

    # plt.y_lim([0.97,1.03])

    #   Set plot limits
    y_err = time_series[err_column].value
    y_err_sigma = sigma_clipping(y_err, sigma=1.5)
    max_err = np.max(y_err_sigma)

    if median_data > 1.1 or median_data < 0.9:
        y_lim = np.max([max_err * 1.5, 0.1])
        plt.ylim([median_data + y_lim, median_data - y_lim])
        y_label_text = ' [mag] (Vega)'
    else:
        y_lim = max_err * 1.3
        # plt.y_lim([median_data - y_lim, median_data + y_lim])
        plt.ylim([min_data - y_lim, max_data + y_lim])
        y_label_text = ' [flux] (normalized)'

    #   Set x and y axis label
    plt.xlabel('Time (days)', fontsize=16)
    plt.ylabel(data_column + y_label_text, fontsize=16)

    #   Save plot
    plt.savefig(
        f'{output_dir}/lightcurve/lightcurve_folded_{data_column}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_transform(output_dir, filter_1, filter_2, color_lit, fit_variable,
                   a_fit, b_fit, b_err_fit, fit_function, air_mass,
                   filter_=None, color_lit_err=None, fit_var_err=None,
                   name_obj=None):
    """
        Plots illustrating magnitude transformation results

        Parameters
        ----------
        output_dir          : `string`
            Output directory

        filter_1            : `string`
            Filter 1

        filter_2            : `string`
            Filter 2

        color_lit           : `numpy.ndarray`
            Colors of the calibration stars

        fit_variable        : `numpy.ndarray`
            Fit variable

        a_fit               : `float`
            First parameter of the fit

        b_fit               : `float`
            Second parameter of the fit
            Currently only two fit parameters are supported
            TODO: -> Needs to generalized

        b_err_fit           : `float`
            Error of `b`

        fit_function        : `fit.function`
            Fit function, used for determining the fit

        air_mass            : `float`
            Air mass

        filter_             : `string`, optional
            Filter, used to distinguish between the different plot options
            Default is ``None``

        color_lit_err       : `numpy.ndarray`, optional
            Color errors of the calibration stars
            Default is ``None``.

        fit_var_err         : `numpy.ndarray`, optional
            Fit variable errors
            Default is ``None``.

        name_obj            : `string`
            Name of the object
            Default is ``None``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'trans_plots'),
    )

    #   Fit data
    x_lin = np.sort(color_lit)
    y_lin = fit_function(x_lin, a_fit, b_fit)

    #   Set labels etc.
    air_mass = round(air_mass, 2)
    if filter_ is None:
        #   coeff  = 1./b
        if name_obj is None:
            title = f'Color transform ({filter_1.lower()}-{filter_2.lower()}' \
                    + f' vs. {filter_1}-{filter_2}) (X = {air_mass})'
        else:
            title = f'Color transform ({filter_1.lower()}-{filter_2.lower()}' \
                    + f' vs. {filter_1}-{filter_2}) - {name_obj} (X = {air_mass})'
        y_label = f'{filter_1.lower()}-{filter_2.lower()} [mag]'
        path = f'{output_dir}/trans_plots/{filter_1.lower() + filter_2.lower()}' \
               + f'_{filter_1 + filter_2}.pdf'
        p_label = f'slope = {b_fit}, T{filter_1.lower()}{filter_2.lower()}' \
                 + f' = {1. / b_fit} +/- {b_err_fit}'
    else:
        #   coeff  = b
        if name_obj is None:
            title = f'{filter_}{filter_1.lower()}{filter_2.lower()}' \
                    + f'-mag transform ({filter_}-{filter_.lower()}' \
                    + f' vs. {filter_1}-{filter_2}) (X = {air_mass})'
        else:
            title = f'{filter_}{filter_1.lower()}{filter_2.lower()}' \
                    + f'-mag transform ({filter_}-{filter_.lower()}' \
                    + f' vs. {filter_1}-{filter_2}) - {name_obj}' \
                    + f' (X = {air_mass})'
        y_label = f'{filter_}-{filter_.lower()} [mag]'
        path = f'{output_dir}/trans_plots/{filter_}{filter_.lower()}' \
               + f'_{filter_1}{filter_2}.pdf'
        p_label = f'slope = {b_fit}, C{filter_.lower()}_' \
                 + f'{filter_1.lower()}{filter_2.lower()} = {b_fit} +/- {b_err_fit}'
    x_label = f'{filter_1}-{filter_2} [mag]'

    #   Make plot
    fig = plt.figure(figsize=(15, 8))

    #   Set title
    fig.suptitle(title, fontsize=20)

    #   Plot data
    plt.errorbar(
        color_lit,
        fit_variable,
        xerr=color_lit_err,
        yerr=fit_var_err,
        marker='o',
        markersize=3,
        capsize=2,
        color='darkgreen',
        ecolor='wheat',
        elinewidth=1,
        linestyle='none',
    )

    #   Plot fit
    plt.plot(
        x_lin,
        y_lin,
        linestyle='-',
        color='maroon',
        linewidth=1.,
        label=p_label,
    )

    #   Set legend
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1.0, 0.102),
        loc=3,
        ncol=4,
        mode='expand',
        borderaxespad=0.,
    )

    #   Set x and y axis label
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    #   Add grid
    plt.grid(True, color='lightgray', linestyle='--', alpha=0.3)
    # plt.grid(color='0.95')

    #   Get median of the data
    y_min = np.min(fit_variable)
    y_max = np.max(fit_variable)

    #   Set plot limits
    if fit_var_err is not None:
        y_err = fit_var_err
        y_err_sigma = sigma_clipping(y_err, sigma=1.5)
        max_err = np.max(y_err_sigma)

        y_lim = np.max([max_err * 1.5, 0.1])
        plt.ylim([y_max + y_lim, y_min - y_lim])

    #   Save plot
    plt.savefig(path, bbox_inches='tight', format='pdf')
    plt.close()


def check_plot(size_x, size_y, magnitudes, color_magnitudes, y_range_max,
               y_range_min, x_range_max, x_range_min):
    """
        Check the plot dimensions and set defaults

        Parameters
        ----------
        size_x              : `float`
            Figure size in cm (x direction)

        size_y              : `float`
            Figure size in cm (y direction)

        magnitudes          : `numpy.ndarray`
            Filter magnitude - 1D

        color_magnitudes    : `numpy.ndarray`
            Color - 1D

        y_range_max         : `float`
            The maximum of the plot range in Y direction

        y_range_min         : `float`
            The minimum of the plot range in Y direction

        x_range_max         : `float`
            The maximum of the plot range in X direction

        x_range_min         : `float`
            The minimum of the plot range in X direction
    """
    #   Set figure size
    if size_x == "" or size_x == "?" or size_y == "" or size_y == "?":
        terminal_output.print_terminal(
            string="[Info] No Plot figure size given, use default: 8cm x 8cm",
            style_name='WARNING',
        )
        plt.figure(figsize=(8, 8))
    else:
        plt.figure(figsize=(int(size_x), int(size_y)))

    #   Set plot range -> automatic adjustment
    #   Y range
    try:
        float(y_range_max)
    except ValueError:
        plt.ylim([
            float(np.max(magnitudes)) + 0.5,
            float(np.min(magnitudes)) - 0.5
        ])
        terminal_output.print_terminal(
            string="[Info] Use automatic plot range for Y",
            style_name='WARNING',
        )
    else:
        try:
            float(y_range_min)
        except ValueError:
            plt.ylim([
                float(np.max(magnitudes)) + 0.5,
                float(np.min(magnitudes)) - 0.5
            ])
            terminal_output.print_terminal(
                string="[Info] Use automatic plot range for Y",
                style_name='WARNING',
            )
        else:
            plt.ylim([float(y_range_min), float(y_range_max)])

    #   X range
    try:
        float(x_range_max)
    except ValueError:
        plt.xlim([
            float(np.min(color_magnitudes)) - 0.5,
            float(np.max(color_magnitudes)) + 0.5
        ])
        terminal_output.print_terminal(
            string="[Info] Use automatic plot range for Y",
            style_name='WARNING',
        )
    else:
        try:
            float(x_range_min)
        except ValueError:
            plt.xlim([
                float(np.min(color_magnitudes)) - 0.5,
                float(np.max(color_magnitudes)) + 0.5
            ])
            terminal_output.print_terminal(
                string="[Info] Use automatic plot range for Y",
                style_name='WARNING',
            )
        else:
            plt.xlim([float(x_range_min), float(x_range_max)])


def mk_ticks_labels(filter_, color):
    """
        Set default ticks and labels

        Parameters
        ----------
        filter_ : `string`
            Filter

        color   : `string`
            Color
    """
    #   Set ticks
    plt.tick_params(
        axis='both',
        which='both',
        top=True,
        right=True,
        direction='in',
    )
    plt.minorticks_on()

    #   Set labels
    plt.xlabel(r'${}$'.format(color))
    plt.ylabel(r'${}$'.format(filter_))


def fill_lists(list_, iso_column, iso_column_type, filter_1, filter_2,
               iso_mag1, iso_mag2, iso_color):
    """
        Sort magnitudes into lists and calculate the color if necessary

        Parameters
        ----------
        list_           : `list` of `string`
            List of strings

        iso_column      : `dictionary`
            Columns to use from the ISO file.
            Keys = filter           : `string`
            Values = column numbers : `integer`

        iso_column_type : `dictionary`
            Type of the columns from the ISO file
            Keys = filter : `string`
            Values = type : `string`

        filter_1        : `string`
            First filter

        filter_2        : `string`
            Second filter

        iso_mag1        : `list` of `float`
            Magnitude list (first filter)

        iso_mag2        : `list` of `float`
            Magnitude list (second filter)

        iso_color       : `list` of `float`
            Color list

        Returns
        -------
        iso_mag1        : `list` of `float`
            Magnitude list (first filter)

        iso_mag2        : `list` of `float`
            Magnitude list (second filter)

        iso_color       : `list` of `float`
            Color list
    """
    mag1 = float(list_[iso_column[filter_1] - 1])
    iso_mag1.append(mag1)
    if iso_column_type[filter_2] == 'color':
        color = float(list_[iso_column[filter_2] - 1])
        iso_color.append(color)
    elif iso_column_type[filter_2] == 'single':
        mag2 = float(list_[iso_column[filter_2] - 1])
        iso_mag2.append(mag2)
        iso_color.append(mag2 - mag1)

    return iso_mag1, iso_mag2, iso_color


def mk_colormap(n_iso):
    """
        Make a color map e.g. for isochrones

        Parameters
        ----------
        n_iso    : `integer`
            Number of isochrone files
    """
    #   Prepare colors for the isochrones
    #   Self defined colormap
    cm1 = mcol.LinearSegmentedColormap.from_list(
        "MyCmapName",
        ['orchid',
         'blue',
         'cyan',
         'forestgreen',
         'limegreen',
         'gold',
         'orange',
         "red",
         'saddlebrown',
         ]
    )
    cnorm = mcol.Normalize(vmin=0, vmax=n_iso)
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    return cpick


def mk_line_cycler():
    """
        Make a line cycler
    """
    lines = ["-", "--", "-.", ":"]
    return cycle(lines)


def mk_color_cycler_symbols():
    """
        Make a color cycler
    """
    colors = ['darkgreen', 'darkred', 'mediumblue', 'yellowgreen']
    return cycle(colors)


def mk_color_cycler_error_bars():
    """
        Make a color cycler
    """
    colors = ['wheat', 'dodgerblue', 'violet', 'gold']
    return cycle(colors)


def write_cmd(name_of_star_cluster, filename, filter_, color, file_type,
              plot_type, output_dir='output'):
    """
        Write plot to disk

        Parameters
        ----------
        name_of_star_cluster    : `string`
            Name of cluster

        filename                : `string`
            Base name of the file to write

        filter_                 : `string`
            Filter

        color                   : `string`
            Color

        file_type               : `string`
            File type

        plot_type                   : `string`
            Plot type

        output_dir              : `string`, optional
            Output directory
            Default is ``output``.
    """
    if name_of_star_cluster == "" or name_of_star_cluster == "?":
        terminal_output.print_terminal(
            output_dir,
            filename,
            plot_type,
            filter_,
            color,
            file_type,
            string="Write plotfile: ./{}/{}_{}_{}_{}.{}",
        )
        plt.savefig(
            f'./{output_dir}/{filename}_{plot_type}_{filter_}_{color}.{file_type}',
            format=file_type,
            bbox_inches="tight",
        )
    else:
        name_of_star_cluster = name_of_star_cluster.replace(' ', '_')
        terminal_output.print_terminal(
            output_dir,
            filename,
            name_of_star_cluster,
            plot_type,
            filter_,
            color,
            file_type,
            string="Write plotfile: ./{}/{}_{}_{}_{}_{}.{}",
        )
        plt.savefig(
            f'./{output_dir}/{filename}_{name_of_star_cluster}_{plot_type}_{filter_}'
            f'_{color}.{file_type}',
            format=file_type,
            bbox_inches="tight",
        )


#   TODO: Make class for CMD plots and add associated functions
def plot_apparent_cmd(magnitude_color, magnitude_filter_1,
                      name_of_star_cluster, file_name, file_type, filter_1,
                      filter_2, figure_size_x='', figure_size_y='',
                      y_plot_range_max='', y_plot_range_min='',
                      x_plot_range_max='', x_plot_range_min='',
                      output_dir='output', magnitude_filter_1_err=None,
                      color_err=None):
    """
        Plot calibrated cmd with apparent magnitudes

        Parameters
        ----------
        magnitude_color             : `numpy.ndarray`
            Color - 1D

        magnitude_filter_1          : `numpy.ndarray`
            Filter magnitude - 1D

        name_of_star_cluster        : `string`
            Name of cluster

        file_name                   : `string`
            Base name of the file to write

        file_type                   : `string`
            File type

        filter_1                    : `string`
            First filter

        filter_2                    : `string`
            Second filter

        figure_size_x               : `float`
            Figure size in cm (x direction)

        figure_size_y               : `float`
            Figure size in cm (y direction)

        y_plot_range_max            : `float`
            The maximum of the plot range in Y direction

        y_plot_range_min            : `float`
            The minimum of the plot range in Y direction

        x_plot_range_max            : `float`
            The maximum of the plot range in X direction

        x_plot_range_min            : `float`
            The minimum of the plot range in X direction

        output_dir                  : `string`, optional
            Output directory
            Default is ``output``.

        magnitude_filter_1_err      : `numpy.ndarray' or ``None``, optional
            Error for ``magnitude_filter_1``
            Default is ``None``.

        color_err                   : `numpy.ndarray' or ``None``, optional
            Error for ``mag_color``
            Default is ``None``.

    """
    #   Check plot dimensions and set defaults
    check_plot(
        figure_size_x,
        figure_size_y,
        magnitude_filter_1,
        magnitude_color,
        y_plot_range_max,
        y_plot_range_min,
        x_plot_range_max,
        x_plot_range_min,
    )

    #   Plot the stars
    terminal_output.print_to_terminal("Add stars", indent=1)
    plt.errorbar(
        magnitude_color,
        magnitude_filter_1,
        xerr=magnitude_filter_1_err,
        yerr=color_err,
        mew=0.0,
        fmt='b.',
        markersize=3,
        capsize=2,
        ecolor='lightgray',
    )

    #   Set ticks and labels
    color = f'{filter_2}-{filter_1}'
    mk_ticks_labels(filter_1, color)

    #   Write plot to disk
    write_cmd(
        name_of_star_cluster,
        file_name,
        filter_1,
        color,
        file_type,
        'apparent',
        output_dir=output_dir,
    )
    plt.close()


def plot_absolute_cmd(magnitude_color, magnitude_filter_1,
                      name_of_star_cluster, file_name, file_type, filter_1,
                      filter_2, isochrones, isochrone_type,
                      isochrone_column_type, isochrone_column,
                      isochrone_log_age, isochrone_keyword, isochrone_legend,
                      figure_size_x='', figure_size_y='', y_plot_range_max='',
                      y_plot_range_min='', x_plot_range_max='',
                      x_plot_range_min='', output_dir='output',
                      magnitude_filter_1_err=None, color_err=None):
    """
        Plot calibrated CMD with
            * magnitudes corrected for reddening and distance
            * isochrones

        Parameters
        ----------
        magnitude_color             : `numpy.ndarray`
            Color - 1D

        magnitude_filter_1          : `numpy.ndarray`
            Filter magnitude - 1D

        name_of_star_cluster        : `string`
            Name of cluster

        file_name                   : `string`
            Base name of the file to write

        file_type                   : `string`
            File type

        filter_1                    : `string`
            First filter

        filter_2                    : `string`
            Second filter

        isochrones                  : `string`
            Path to the isochrone directory or the isochrone file

        isochrone_type              : `string`
            Type of 'isochrones'
            Possibilities: 'directory' or 'file'

        isochrone_column_type       : `dictionary`
            Keys = filter : `string`
            Values = type : `string`

        isochrone_column            : `dictionary`
            Keys = filter           : `string`
            Values = column numbers : `integer`

        isochrone_log_age           : `boolean`
            Logarithmic age

        isochrone_keyword           : `string`
            Keyword to identify a new isochrone

        isochrone_legend            : `boolean`
            If True plot legend for isochrones.

        figure_size_x               : `float`, optional
            Figure size in cm (x direction)
            Default is ````.

        figure_size_y               : `float`, optional
            Figure size in cm (y direction)
            Default is ````.

        y_plot_range_max            : `float`, optional
            The maximum of the plot range in Y
                                direction
            Default is ````.

        y_plot_range_min            : `float`, optional
            The minimum of the plot range in Y
                                direction
            Default is ````.

        x_plot_range_max            : `float`, optional
            The maximum of the plot range in X
                                direction
            Default is ````.

        x_plot_range_min            : `float`, optional
            The minimum of the plot range in X direction

        output_dir                  : `string`, optional
            Output directory
            Default is ``output``.

        magnitude_filter_1_err      : `numpy.ndarray' or ``None``, optional
            Error for ``magnitude_filter_1``
            Default is ``None``.

        color_err                   : `numpy.ndarray' or ``None``, optional
            Error for ``mag_color``
            Default is ``None``.
    """
    #   Check plot dimensions and set defaults
    check_plot(
        figure_size_x,
        figure_size_y,
        magnitude_filter_1,
        magnitude_color,
        y_plot_range_max,
        y_plot_range_min,
        x_plot_range_max,
        x_plot_range_min,
    )

    #   Plot the stars
    terminal_output.print_to_terminal("Add stars")
    plt.errorbar(
        magnitude_color,
        magnitude_filter_1,
        xerr=magnitude_filter_1_err,
        yerr=color_err,
        mew=0.0,
        fmt='b.',
        markersize=3,
        capsize=2,
        ecolor='lightgray',
    )

    ###
    #   Plot isochrones
    #

    #   Check if isochrones are specified
    if isochrones != '' and isochrones != '?':
        #   OPTION I: Individual isochrone files in a specific directory
        if isochrone_type == 'directory':
            #   Resolve iso path
            isochrones = Path(isochrones).expanduser()

            #   Make list of isochrone files
            file_list = os.listdir(isochrones)

            #   Number of isochrones
            n_isochrones = len(file_list)
            terminal_output.print_to_terminal(
                f"Plot {n_isochrones} isochrone(s)",
                style_name='OKGREEN',
            )

            #   Make color map
            color_pick = mk_colormap(n_isochrones)

            #   Prepare cycler for the line styles
            line_cycler = mk_line_cycler()

            #   Cycle through iso files
            for i in range(0, n_isochrones):
                #   Load file
                isochrone_data = open(isochrones / file_list[i])

                #   Prepare variables for the isochrone data
                isochrone_magnitude_1 = []
                isochrone_magnitude_2 = []
                isochrone_color = []
                age_value = ''
                age_unit = ''

                #   Extract B and V values & make lists
                #   Loop over all lines in the file
                for line in isochrone_data:
                    line_elements = line.split()

                    #   Check that the entries are not HEADER keywords
                    try:
                        float(line_elements[0])
                    except:
                        #   Try to find and extract age information
                        if 'Age' in line_elements or 'age' in line_elements:
                            try:
                                age_index = line_elements.index('age')
                            except:
                                age_index = line_elements.index('Age')

                            for string in line_elements[age_index + 1:]:
                                #   Find age unit
                                if string.rfind("yr") != -1:
                                    age_unit = string
                                #   Find age value
                                try:
                                    if isinstance(age_value, str):
                                        age_value = int(float(string))
                                except:
                                    pass
                        continue

                    #   Fill lists
                    isochrone_magnitude_1, isochrone_magnitude_2, isochrone_color = fill_lists(
                        line_elements,
                        isochrone_column,
                        isochrone_column_type,
                        filter_1,
                        filter_2,
                        isochrone_magnitude_1,
                        isochrone_magnitude_2,
                        isochrone_color,
                    )

                #   Construct label
                if not isinstance(age_value, str):
                    label = str(age_value)
                    if age_unit != '':
                        label += f' {age_unit}'
                else:
                    label = os.path.splitext(file_list[i])[0]

                #   Plot iso lines
                plt.plot(
                    isochrone_color,
                    isochrone_magnitude_1,
                    linestyle=next(line_cycler),
                    color=color_pick.to_rgba(i),
                    linewidth=0.8,
                    label=label,
                )

                #   Close file with the iso data
                isochrone_data.close()

        #   OPTION II: Isochrone file containing many individual isochrones
        if isochrone_type == 'file':
            #   Resolve iso path
            isochrones = Path(isochrones).expanduser()

            #   Load file
            isochrone_data = open(isochrones)

            #   Overall lists for the isochrones
            age_list = []
            isochrone_magnitude_1_list = []
            isochrone_magnitude_2_list = []
            isochrone_color_list = []

            #   Number of detected isochrones
            n_isochrones = 0

            #   Loop over all lines in the file
            for line in isochrone_data:
                line_elements = line.split()

                #   Check for a key word to distinguish the isochrones
                try:
                    if line[0:len(isochrone_keyword)] == isochrone_keyword:
                        #   Add data from the last isochrone to the overall lists
                        #   for the isochrones.
                        if n_isochrones:
                            #   This part is only active after an isochrone has been detected.
                            #   The variables are then assigned.
                            age_list.append(age)
                            isochrone_magnitude_1_list.append(isochrone_magnitude_1)
                            isochrone_magnitude_2_list.append(isochrone_magnitude_2)
                            isochrone_color_list.append(isochrone_color)

                        #   Save age for the case where age is given as a
                        #   keyword and not as a column
                        if isochrone_column['AGE'] == 0:
                            age = line.split('=')[1].split()[0]

                        #   Prepare/reset lists for the single isochrones
                        isochrone_magnitude_1 = []
                        isochrone_magnitude_2 = []
                        isochrone_color = []

                        n_isochrones += 1
                        continue
                except RuntimeError:
                    continue

                #   Check that the entries are not HEADER keywords
                try:
                    float(line_elements[0])
                except ValueError:
                    continue

                #   Fill lists
                if isochrone_column['AGE'] != 0:
                    age = float(line_elements[isochrone_column['AGE'] - 1])

                isochrone_magnitude_1, isochrone_magnitude_2, isochrone_color = fill_lists(
                    line_elements,
                    isochrone_column,
                    isochrone_column_type,
                    filter_1,
                    filter_2,
                    isochrone_magnitude_1,
                    isochrone_magnitude_2,
                    isochrone_color,
                )

            #   Add last isochrone to overall lists
            age_list.append(age)
            isochrone_magnitude_1_list.append(isochrone_magnitude_1)
            isochrone_magnitude_2_list.append(isochrone_magnitude_2)
            isochrone_color_list.append(isochrone_color)

            #   Close isochrone file
            isochrone_data.close()

            #   Number of isochrones
            n_isochrones = len(isochrone_magnitude_1_list)
            terminal_output.print_to_terminal(
                f"Plot {n_isochrones} isochrone(s)",
                style_name='OKGREEN',
            )

            #   Make color map
            color_pick = mk_colormap(n_isochrones)

            #   Prepare cycler for the line styles
            line_cycler = mk_line_cycler()

            #   Cycle through iso lines
            for i in range(0, n_isochrones):
                if isochrone_log_age:
                    age_value = float(age_list[i])
                    age_value = 10 ** age_value / 10 ** 9
                    age_value = round(age_value, 2)
                else:
                    age_value = round(float(age_list[i]), 2)
                age_string = f'{age_value} Gyr'

                #   Plot iso lines
                plt.plot(
                    isochrone_color_list[i],
                    isochrone_magnitude_1_list[i],
                    linestyle=next(line_cycler),
                    color=color_pick.to_rgba(i),
                    linewidth=0.8,
                    label=age_string,
                )
                isochrone_data.close()

        #   Plot legend
        if isochrone_legend:
            plt.legend(
                bbox_to_anchor=(0., 1.02, 1.0, 0.102),
                loc=3,
                ncol=4,
                mode='expand',
                borderaxespad=0.,
            )

    #   Set ticks and labels
    color = f'{filter_2}-{filter_1}'
    mk_ticks_labels(filter_1, color)

    #   Write plot to disk
    write_cmd(
        name_of_star_cluster,
        file_name,
        filter_1,
        color,
        file_type,
        'absolut',
        output_dir=output_dir,
    )
    plt.close()

def onpick3(event):
    print('---------------------')
    print(dir(event))
    ind = event.ind
    # print('onpick3 scatter:', ind, np.take(x, ind))
    print('onpick3 scatter:', ind)
    print(event.artist)
    print(dir(event.artist))
    print(event.artist.get_label())
    print(event.artist.get_gid())
    # print(event.mouseevent)
    # print(dir(event.mouseevent))
    # print(event.mouseevent.inaxes)
    # print(dir(event.mouseevent.inaxes))
    # print(event.name)
    print('+++++++++++++++++++++')


def click_point(event):
    print('---------------------')
    print(dir(event))
    print(event.button)
    print(event.guiEvent)
    print(event.key)
    print(event.lastevent)
    print(event.name)
    print(event.step)
    print('+++++++++++++++++++++')


def d3_scatter(xs, ys, zs, output_dir, color=None, name_x='', name_y='',
               name_z='', string='_3D_', pm_ra=None, pm_dec=None,
               display=False):
    """
        Make a 2D scatter plot

        Parameters
        ----------
        xs           : `list` of `numpy.ndarray`s
            X values

        ys          : `list` of `numpy.ndarray`s
            Y values

        zs          : `list` of `numpy.ndarray`s
            Z values

        color       : `list` of `string`
            Color definitions

        output_dir  : `string`
            Output directory

        name_x      : `string`, optional
            Label for the X axis
            Default is ````.

        name_y      : `string`, optional
            Label for the Y axis
            Default is ````.

        name_z      : `string`, optional
            Label for the Z axis
            Default is ````.

        string      : `string`, optional
            String characterizing the output file
            Default is ``_3D_``.

        pm_ra       : `float`, optional
            Literature proper motion in right ascension.
            If not ``None`` the value will be printed to the plot.
            Default is ``None``.

        pm_dec      : `float`, optional
            Literature proper motion in declination.
            If not ``None`` the value will be printed to the plot.
            Default is ``None``.

        display     : `boolean`, optional
            If ``True`` the 3D plot will be displayed in an interactive
            window. If ``False`` four views of the 3D plot will be saved to
            a file.
            Default is ``False``.
    """
    #   Switch backend to allow direct display of the plot
    if display:
        plt.switch_backend('TkAgg')

    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'compare'),
    )

    #   Prepare plot
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)

    #   Set title
    if display:
        if pm_ra is not None and pm_dec is not None:
            fig.suptitle(
                f'Proper motion vs. distance: Literature proper motion: '
                f'{pm_ra:.1f}, {pm_dec:.1f} - Choose a cluster then close the '
                f'plot',
                fontsize=17,
            )
        else:
            fig.suptitle(
                'Proper motion vs. distance: Literature proper motion: '
                '- Choose a cluster then close the plot',
                fontsize=17,
            )
    else:
        if pm_ra is not None and pm_dec is not None:
            fig.suptitle(
                f'Proper motion vs. distance: Literature proper motion: '
                f'{pm_ra:.1f}, {pm_dec:.1f} ',
                fontsize=17,
            )
        else:
            fig.suptitle(
                'Proper motion vs. distance',
                fontsize=17,
            )

    #   Switch to one subplot for direct display
    if display:
        n_subplots = 1
    else:
        n_subplots = 4

    #   Loop over all subplots
    for i in range(0, n_subplots):
        if display:
            ax = fig.add_subplot(1, 1, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        #   Change view angle
        ax.view_init(25, 45 + i * 90)

        #   Labelling X-Axis
        ax.set_xlabel(name_x)

        #   Labelling Y-Axis
        ax.set_ylabel(name_y)

        #   Labelling Z-Axis
        ax.set_zlabel(name_z)

        #   Set default plot ranges/limits
        default_pm_range = [-20, 20]
        default_dist_range = [0, 10]

        #   Find suitable plot ranges
        xs_list = list(itertools.chain.from_iterable(xs))
        max_xs = np.max(xs_list)
        min_xs = np.min(xs_list)

        ys_list = list(itertools.chain.from_iterable(ys))
        max_ys = np.max(ys_list)
        min_ys = np.min(ys_list)

        dist_list = list(itertools.chain.from_iterable(zs))
        max_zs = np.max(dist_list)
        min_zs = np.min(dist_list)

        #   Set range: defaults or values from above
        if default_pm_range[0] < min_xs:
            x_min = min_xs
        else:
            x_min = default_pm_range[0]
        if default_pm_range[1] > min_xs:
            x_max = max_xs
        else:
            x_max = default_pm_range[1]
        if default_pm_range[0] < min_ys:
            y_min = min_ys
        else:
            y_min = default_pm_range[0]
        if default_pm_range[1] > min_ys:
            y_max = max_ys
        else:
            y_max = default_pm_range[1]
        if default_dist_range[0] < min_zs:
            z_min = min_zs
        else:
            z_min = default_dist_range[0]
        if default_dist_range[1] > min_zs:
            z_max = max_zs
        else:
            z_max = default_dist_range[1]

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        #   Plot data
        if color is None:
            for j, x in enumerate(xs):
                ax.scatter3D(
                    x,
                    ys[j],
                    zs[j],
                    # c=zs[i],
                    cmap='cividis',
                    # cmap='tab20',
                    label=f'Cluster {j}',
                    # picker=True,
                    picker=5,
                )
                ax.legend()
        else:
            for j, x in enumerate(xs):
                ax.scatter3D(
                    x,
                    ys[j],
                    zs[j],
                    c=color[j],
                    cmap='cividis',
                    # cmap='tab20',
                    label=f'Cluster {j}',
                )
                ax.legend()

    # fig.canvas.mpl_connect('pick_event', onpick3)
    # fig.canvas.mpl_connect('button_press_event',click_point)

    #   Display plot and switch backend back to default
    if display:
        plt.show()
        # plt.show(block=False)
        # time.sleep(300)
        # print('after sleep')
        plt.close()
        plt.switch_backend('Agg')
    else:
        #   Save image if it is not displayed directly
        plt.savefig(
            f'{output_dir}/compare/pm_vs_distance.pdf',
            bbox_inches='tight',
            format='pdf',
        )
        plt.close()


def scatter(x_values, name_x, y_values, name_y, rts, output_dir, x_errors=None,
            y_errors=None, dataset_label=None, name_obj=None, fits=None,
            one_to_one=False):
    """
        Plot magnitudes

        Parameters
        ----------
        x_values        : `list` of `numpy.ndarray`
            List of arrays with X values

        name_x          : `string`
            Name of quantity 1

        y_values        : `list` of `numpy.ndarray`
            List of arrays with Y values

        name_y          : `string`
            Name of quantity 2

        rts             : `string`
            Expression characterizing the plot

        output_dir      : `string`
            Output directory

        x_errors        : `list` of `numpy.ndarray' or ``None``, optional
            Errors for the X values
            Default is ``None``.

        y_errors        : `list` of `numpy.ndarray' or ``None``, optional
            Errors for the Y values
            Default is ``None``.

        dataset_label   : 'list` of 'string` or `None`, optional
            Label for the datasets
            Default is ``None``.

        name_obj        : `string`, optional
            Name of the object
            Default is ``None``

        fits            : `list` of `astropy.modeling.fitting` instance, optional
            Fits to the data
            Default is ``None``.

        one_to_one      : `boolean`, optional
            If True a 1:1 line will be plotted.
            Default is ``False``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'scatter'),
    )

    #   Plot magnitudes
    fig = plt.figure(figsize=(8, 8))

    #   Set title
    if name_obj is None:
        sub_title = f'{name_x} vs. {name_y}'
    else:
        sub_title = f'{name_x} vs. {name_y} ({name_obj})'
    fig.suptitle(
        sub_title,
        fontsize=17,
    )

    #   Initialize color cyclers
    color_cycler_symbols = mk_color_cycler_symbols()
    color_cycler_error_bars = mk_color_cycler_error_bars()

    #   Plot data
    for i, x in enumerate(x_values):
        if dataset_label is None:
            dataset_label_i = ''
        else:
            dataset_label_i = dataset_label[i]
        plt.errorbar(
            x,
            y_values[i],
            xerr=x_errors[i],
            yerr=y_errors[i],
            marker='o',
            ls='none',
            markersize=3,
            capsize=2,
            color=next(color_cycler_symbols),
            ecolor=next(color_cycler_error_bars),
            elinewidth=1,
            label=f'{dataset_label_i}'
        )

        #   Plot fit
        if fits is not None:
            if fits[i] is not None:
                x_sort = np.sort(x)
                plt.plot(
                    x_sort,
                    fits[i](x_sort),
                    color='darkorange',
                    linewidth=1,
                    label=f'Fit to dataset {dataset_label_i}',
                )

    #   Add legend
    if dataset_label is not None:
        plt.legend()

    #   Add grid
    plt.grid(True, color='lightgray', linestyle='--', alpha=0.3)

    #   Plot the 1:1 line
    if one_to_one:
        x_min = np.amin(x_values)
        x_max = np.amax(x_values)
        y_min = np.amin(y_values)
        y_max = np.amax(y_values)
        max_plot = np.max([x_max, y_max])
        min_plot = np.min([x_min, y_min])

        plt.plot(
            [min_plot, max_plot],
            [min_plot, max_plot],
            color='black',
            lw=2,
        )

    #   Set x and y axis label
    plt.ylabel(name_y)
    plt.xlabel(name_x)

    #   Save plot
    plt.savefig(
        f'{output_dir}/scatter/{rts}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_limiting_mag_sky_apertures(output_dir, img_data, mask, image_depth):
    """
        Plot the sky apertures that are used to estimate the limiting magnitude

        Parameters
        ----------
        output_dir          : `string`
            Output directory

        img_data            : `numpy.ndarray`
            Image data

        mask                : `numpy.ndarray`
            Mask showing the position of detected objects

        image_depth         : `photutils.utils.ImageDepth`
            Object used to derive the limiting magnitude
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
    norm = ImageNormalize(img_data, interval=ZScaleInterval(contrast=0.15,))
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
        f'{output_dir}/limiting_mag/limiting_mag_sky_regions.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()
