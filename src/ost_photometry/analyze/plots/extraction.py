"""Extraction / ePSF quality-control plots."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
from matplotlib import rcParams
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.psf import EPSFStars, ImagePSF

from ... import checks, terminal_output
from ... import utilities as base_utilities

plt.switch_backend("Agg")

def plot_apertures(
        output_dir: str, image: base_utilities.Image,
        aperture: CircularAperture, annulus_aperture: CircularAnnulus,
        filename_string: str, file_type: str = 'pdf',
        pixel_scale: float | None = None) -> None:
    """
    Plot the apertures used for extracting the stellar fluxes
           (star map plot for aperture photometry)

    Parameters
    ----------
    output_dir
        Output directory

    image
        2D Image data

    aperture
        Apertures used to extract the stellar flux

    annulus_aperture
        Apertures used to extract the background flux

    filename_string
        String characterizing the output file

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    pixel_scale
        Arcsec per pixel; if set, legend also shows radii in arcsec.
        Default is ``None``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'aperture'),
    )

    def _radius_label(r_pix: float) -> str:
        r_pix = float(r_pix)
        if pixel_scale is not None and np.isfinite(pixel_scale) and pixel_scale > 0:
            return f"{r_pix:.2f} px ({r_pix * float(pixel_scale):.2f}\")"
        return f"{r_pix:.2f} px"

    r_ap = float(np.atleast_1d(aperture.r)[0])
    r_in = float(np.atleast_1d(annulus_aperture.r_in)[0])
    r_out = float(np.atleast_1d(annulus_aperture.r_out)[0])

    #   Make plot
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
        label=f'Object aperture (r={_radius_label(r_ap)})',
    )

    #   Plot background apertures
    ann_patches = annulus_aperture.plot(
        color='darkred',
        lw=0.2,
        label=(
            f'Background annulus '
            f'(r_in={_radius_label(r_in)}, r_out={_radius_label(r_out)})'
        ),
    )

    #
    handles = (ap_patches[0], ann_patches[0])

    #   Set labels
    plt.xlabel("[pixel]", fontsize=16)
    plt.ylabel("[pixel]", fontsize=16)
    plt.title(
        f"Aperture photometry: r={_radius_label(r_ap)}, "
        f"annulus {_radius_label(r_in)}–{_radius_label(r_out)}",
        fontsize=14,
    )

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
        f'{output_dir}/aperture/aperture_{filename_string}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )

    plt.close()


def plot_cutouts(output_dir: str, stars: EPSFStars, identifier: str,
                 terminal_logger: terminal_output.TerminalLog | None = None,
                 max_plot_stars: int = 25, name_object: str | None = None,
                 file_type: str = 'pdf', indent: int = 2) -> None:
    """
    Plot the cutouts of the stars used to estimate the ePSF

    Parameters
    ----------
    output_dir
        Output directory

    stars
        Numpy array with cutouts of the ePSF stars

    identifier
        String characterizing the plot

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    max_plot_stars
        Maximum number of cutouts to plot
        Default is ``25``.

    name_object
        Name of the object
        Default is ``None``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    indent
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
            f"Plot ePSF cutouts ({identifier})",
            indent=indent,
        )
    else:
        terminal_output.print_to_terminal(
            f"Plot ePSF cutouts ({identifier})",
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

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title of the complete plot
    if name_object is None:
        sub_title = f'Cutouts of the {n_cutouts} faintest stars ({identifier})'
    else:
        sub_title = f'Cutouts of the {n_cutouts} faintest stars ({identifier}) - {name_object}'
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
        f'{output_dir}/cutouts/cutouts_{identifier}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    # plt.show()
    plt.close()


def plot_epsf(
        output_dir: str, epsf: dict[str, list[ImagePSF]],
        name_object: str | None = None, id_image: str = '',
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type: str = 'pdf', indent: int = 1) -> None:
    """

    Plot the ePSF image of all filters

    Parameters
    ----------
    output_dir
        Output directory

    epsf
        PSF object, usually constructed by epsf_builder

    name_object
        Name of the object
        Default is ``None``.

    id_image
        ID of the image that should be added to the file name.
        Default is ````.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.


    indent
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

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title of the complete plot
    if name_object is None:
        fig.suptitle('ePSF', fontsize=17)
    else:
        fig.suptitle(f'ePSF ({name_object})', fontsize=17)

    #   Plot individual subplots
    for i, (filter_, eps_s) in enumerate(epsf.items()):
        for eps in eps_s:
            if eps is not None:
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
                ax.set_title(filter_)

                #   Set labels
                ax.set_xlabel("Pixel")
                ax.set_ylabel("Pixel")

                #   Set color bar
                fig.colorbar(im1, ax=ax)

    if n_plots >= 2:
        plt.savefig(
            f'{output_dir}/epsfs/epsfs_multiple_filter{id_image}.{file_type}',
            bbox_inches='tight',
            format=file_type,
        )
    else:
        plt.savefig(
            f'{output_dir}/epsfs/epsf{id_image}.{file_type}',
            bbox_inches='tight',
            format=file_type,
        )
    # plt.show()
    plt.close()


def plot_residual(
        image_orig: dict[str, np.ndarray],
        residual_image: dict[str, np.ndarray],
        output_dir: str, name_object: str | None = None,
        terminal_logger: terminal_output.TerminalLog | None = None,
        file_type: str = 'pdf', indent: int = 1) -> None:
    """
    Plot the original and the residual ePSF image

    Parameters
    ----------
    image_orig
        Original image data

    residual_image
        Residual image data

    output_dir
        Output directory

    name_object
        Name of the object
        Default is ``None``.

    terminal_logger
        Logger object. If provided, the terminal output will be directed
        to this object.
        Default is ``None``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    indent
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
        fig = plt.figure(figsize=(10, 10))
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

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title of the complete plot
    if name_object is not None:
        fig.suptitle(f'{name_object}', fontsize=17)

    i = 1
    filter_ = None
    for filter_, image in image_orig.items():
        #   Plot original image
        #   Set up normalization for the image
        norm = ImageNormalize(image, interval=ZScaleInterval())

        if n_plots == 1:
            ax = fig.add_subplot(2, 1, i)
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
        ax.set_title(f'Original Image ({filter_})')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im1, ax=ax)

        i += 1

        #   Plot residual image
        #   Set up normalization for the image
        norm = ImageNormalize(residual_image[filter_],
                              interval=ZScaleInterval())

        if n_plots == 1:
            ax = fig.add_subplot(2, 1, i)
        elif n_plots == 2:
            ax = fig.add_subplot(2, 2, i)
        else:
            ax = fig.add_subplot(n_plots, 2, i)

        #   Plot image
        im2 = ax.imshow(
            residual_image[filter_],
            norm=norm,
            cmap='viridis',
            aspect=1,
            interpolation='nearest',
            origin='lower',
        )

        #   Set title of subplot
        ax.set_title(f'Residual Image ({filter_})')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im2, ax=ax)

        i += 1

    #   Write the plot to disk
    if n_plots == 1:
        plt.savefig(
            f'{output_dir}/residual/residual_images_{filter_}.{file_type}'.replace(":", "")
            .replace(",", "").replace(" ", "_"),
            bbox_inches='tight',
            format=file_type,
        )
    else:
        plt.savefig(
            f'{output_dir}/residual/residual_images.{file_type}',
            bbox_inches='tight',
            format=file_type
        )
    # plt.show()
    plt.close()


