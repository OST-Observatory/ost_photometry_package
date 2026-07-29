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
from astropy.table import Table
from astropy.stats import sigma_clip as sigma_clipping
from astropy.stats import sigma_clipped_stats
from astropy.modeling import fitting
from astropy.time import Time
from astropy.timeseries import aggregate_downsample
import astropy.units as u
from astropy.timeseries import TimeSeries
from astropy import wcs
from astropy.coordinates import SkyCoord

from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.psf import EPSFStars, ImagePSF
from photutils.utils import ImageDepth

from scipy.spatial import KDTree

from itertools import cycle

from ... import checks, style, terminal_output, calibration_parameters
from ... import utilities as base_utilities

import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib import rcParams, gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator


def _nonnegative_errorbar_yerr(values) -> np.ndarray:
    """``matplotlib.errorbar`` rejects negative ``yerr``; use absolute uncertainty."""
    return np.abs(np.asarray(values, dtype=float))


def _safe_ylim_from_series(
    y_values,
    err_values,
    *,
    own_scaling: bool,
    invert_axis: bool,
    mag_like_threshold: tuple[float, float] = (0.9, 1.1),
) -> str:
    """
    Choose y-axis label suffix; set ``plt.ylim`` only when limits are finite.

    Skips ``plt.ylim`` when stats are non-finite (e.g. NaN magnitudes left in series).
    """
    y = np.asarray(y_values, dtype=float)
    fin = np.isfinite(y)
    if not np.any(fin):
        return " [mag] (Vega)"

    median_data = float(np.nanmedian(y))
    min_data = float(np.nanmin(y))
    max_data = float(np.nanmax(y))

    if invert_axis and (median_data > 1.5 or median_data < 0.5):
        plt.gca().invert_yaxis()

    y_err = _nonnegative_errorbar_yerr(err_values)
    y_err_sigma = sigma_clipping(y_err, sigma=1.5)
    max_err = float(np.nanmax(y_err_sigma))
    if not np.isfinite(max_err) or max_err <= 0:
        max_err = 0.1

    lo_mag, hi_mag = mag_like_threshold
    if median_data > hi_mag or median_data < lo_mag:
        y_lim = max(float(np.nanmax(np.array([max_err * 1.5, 0.1]))), 0.05)
        if own_scaling and np.isfinite(median_data):
            plt.ylim([median_data + y_lim, median_data - y_lim])
        return " [mag] (Vega)"

    y_lim = max(max_err * 1.2, 0.05)
    if own_scaling and np.isfinite(min_data) and np.isfinite(max_data):
        plt.ylim([min_data - y_lim, max_data + y_lim])
    return " [flux] (normalized)"

############################################################################
#                           Routines & definitions                         #
############################################################################



#   TODO: Fix type hints for fit_function
def plot_transform(
        output_dir: str, filter_1: str, filter_2: str, current_filter: str,
        target_filter: str, color_literature: np.ndarray,
        fit_variable: np.ndarray, a_fit: float, b_fit: float,
        b_err_fit: float, fit_function: any, air_mass: float,
        color_literature_err: np.ndarray | None = None,
        fit_variable_err: np.ndarray | None = None,
        name_object: list[str] | str | None = None,
        image_id: int | None = None, x_data_original: np.ndarray | None = None,
        y_data_original: np.ndarray | None = None,
        file_type: str = 'pdf') -> None:
    """
    Plots illustrating magnitude transformation results

    Parameters
    ----------
    output_dir
        Output directory

    filter_1
        Filter 1

    filter_2
        Filter 2

    current_filter
        Current filter

    target_filter
        Filter for which the derived parameters will be used

    color_literature
        Colors of the calibration stars

    fit_variable
        Fit variable

    a_fit
        First parameter of the fit

    b_fit
        Second parameter of the fit
        Currently only two fit parameters are supported
        TODO: -> Needs to generalized

    b_err_fit
        Error of `b`

    fit_function
        Fit function, used for determining the fit

    air_mass
        Air mass

    color_literature_err
        Color errors of the calibration stars
        Default is ``None``.

    fit_variable_err
        Fit variable errors
        Default is ``None``.

    name_object
        Name of the object
        Default is ``None``.

    image_id
        ID of the image

    x_data_original
        Original abscissa data with out any modification, which might
        have been applied to data

    y_data_original
        Original ordinate data with out any modification, which might
        have been applied to data

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'trans_plots'),
    )

    #   Add image ID to file name, if available
    if image_id is not None:
        id_image_str = f'_{image_id}'
    else:
        id_image_str = ''

    #   Fit data
    x_lin = np.sort(color_literature)
    y_lin = fit_function(x_lin, a_fit, b_fit)

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set labels etc.
    air_mass = round(air_mass, 2)
    #   coeff  = b
    if name_object is None:
        title = f'{current_filter}{filter_1.lower()}{filter_2.lower()}' \
                f'-mag transform ({current_filter}-{current_filter.lower()}' \
                f' vs. {filter_1}-{filter_2}) (X = {air_mass}, ' \
                f'target filter: {target_filter})'
    else:
        title = f'{current_filter}{filter_1.lower()}{filter_2.lower()}' \
                f'-mag transform ({current_filter}-{current_filter.lower()}' \
                f' vs. {filter_1}-{filter_2}) - {name_object}' \
                f' (X = {air_mass})'
    y_label = f'{current_filter}-{current_filter.lower()} [mag]'
    path = f'{output_dir}/trans_plots/{target_filter}_{current_filter}' \
           f'{current_filter.lower()}_{filter_1}{filter_2}{id_image_str}.{file_type}'
    p_label = (f'slope = {b_fit:.5f}, C{current_filter.lower()}_{filter_1.lower()}'
               f'{filter_2.lower()} = {b_fit:.5f} +/- {b_err_fit:.5f}')
    x_label = f'{filter_1}-{filter_2} [mag]'

    #   Make plot
    fig = plt.figure(figsize=(15, 8))

    #   Set title
    fig.suptitle(title, fontsize=20)

    if x_data_original is not None and y_data_original is not None:
        plt.errorbar(
            x_data_original,
            y_data_original,
            marker='o',
            markersize=3,
            capsize=2,
            color='darkred',
            ecolor='wheat',
            elinewidth=1,
            linestyle='none',
        )

    #   Plot data
    plt.errorbar(
        color_literature,
        fit_variable,
        xerr=color_literature_err,
        yerr=fit_variable_err,
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

    #   Get median of the data
    y_min = np.min(fit_variable)
    y_max = np.max(fit_variable)

    #   Set plot limits
    if fit_variable_err is not None:
        y_err = fit_variable_err
        y_err_sigma = sigma_clipping(y_err, sigma=1.5)
        max_err = np.max(y_err_sigma)

        y_lim = np.max([max_err * 1.5, 0.1])
        plt.ylim([y_max + y_lim, y_min - y_lim])

    #   Save plot
    plt.savefig(path, bbox_inches='tight', format=file_type)
    plt.close()



def initialize_plot(size_x: str, size_y: str) -> plt.figure:
    """
    Check the plot dimensions and set defaults

    Parameters
    ----------
    size_x
        Figure size in cm (x direction)

    size_y
        Figure size in cm (y direction)

    Returns
    -------
    fig
        Figure object
    """
    #   Set figure size
    if size_x == "" or size_x == "?" or size_y == "" or size_y == "?":
        terminal_output.print_to_terminal(
            "[Info] No Plot figure size given, use default: 8cm x 8cm",
            style_name='WARNING',
        )
        fig = plt.figure(figsize=(8, 8))
    else:
        fig = plt.figure(figsize=(int(size_x), int(size_y)))

    return fig


def mk_ticks_labels(
        y_axis_label: str, x_axis_label: str, ax: plt.subplot) -> None:
    """
    Set default ticks and labels

    Parameters
    ----------
    y_axis_label
        Filter

    x_axis_label
        Color

    ax
        Subplot
    """
    #   Set ticks
    ax.tick_params(
        axis='both',
        which='both',
        top=True,
        right=True,
        direction='in',
    )
    ax.minorticks_on()
    ax.grid(True, color='lightgray', linestyle='--')

    #   Set labels
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)


class MaxRecursionError(Exception):
    pass


def mk_colormap(n_color_steps):
    """
        Make a color map e.g. for isochrones

        Parameters
        ----------
        n_color_steps    : `integer`
            Number of things to color
    """
    #   Prepare colormap
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
    cnorm = mcol.Normalize(vmin=0, vmax=n_color_steps)
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    return cpick


def mk_line_cycler() -> cycle:
    """
        Make a line cycler
    """
    lines: list[str] = ["-", "--", "-.", ":"]
    return cycle(lines)


def mk_color_cycler_symbols() -> cycle:
    """
        Make a color cycler
    """
    colors: list[str] = ['darkgreen', 'darkred', 'mediumblue', 'yellowgreen']
    return cycle(colors)


def mk_color_cycler_error_bars() -> cycle:
    """
        Make a color cycler
    """
    colors: list[str] = ['wheat', 'dodgerblue', 'violet', 'gold']
    return cycle(colors)


def onpick3(event):
    """Matplotlib pick callback (debug output removed)."""
    _ = event.ind


def click_point(event):
    """Matplotlib click callback (debug output removed)."""
    _ = event.button


def d3_scatter(
        xs: list[np.ndarray], ys: list[np.ndarray], zs: list[np.ndarray],
        output_dir: str, color: list[str] | None = None, name_x: str = '',
        name_y: str = '', name_z: str = '', pm_ra: float | None = None,
        pm_dec: float | None = None, display: bool = False,
        file_type: str = 'pdf') -> None:
    """
    Make a 3D scatter plot

    Parameters
    ----------
    xs
        X values

    ys
        Y values

    zs
        Z values

    color
        Specifiers for the color

    output_dir
        Output directory

    name_x
        Label for the X axis
        Default is ````.

    name_y
        Label for the Y axis
        Default is ````.

    name_z
        Label for the Z axis
        Default is ````.

    pm_ra
        Literature proper motion in right ascension.
        If not ``None`` the value will be printed to the plot.
        Default is ``None``.

    pm_dec
        Literature proper motion in declination.
        If not ``None`` the value will be printed to the plot.
        Default is ``None``.

    display
        If ``True`` the 3D plot will be displayed in an interactive
        window. If ``False`` four views of the 3D plot will be saved to
        a file.
        Default is ``False``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.
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
                    # cmap='cividis',
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
            f'{output_dir}/compare/pm_vs_distance.{file_type}',
            bbox_inches='tight',
            format=file_type,
        )
        plt.close()




def extinction_curves(rv: float) -> None:
    """
    Plots extinction curves
    Currently only Fitzpatrick (without most of the UV range) is supported

    Parameters
    ----------
    rv
    Ration of absolute to relative extinction: AV/E(B-V)
    """
    #   Get Fitzpatrick law
    fitzpatrick_extinction_curve = calibration_parameters.fitzpatrick_extinction_curve(rv)

    #   Get x (1/lambda) range
    x = np.arange(0, 4, 0.1)

    #   Plot dimension
    fig = plt.figure(figsize=(8, 8))

    #   Set title
    fig.suptitle(
        "Extinction curves",
        fontsize=17,
    )

    plt.plot(
        x,
        fitzpatrick_extinction_curve(x),
        color='darkorange',
        linewidth=1,
        label=r'Fitzpatrick: $R_\mathrm{V} = $' + f'{rv}',
    )

    #   Set x and y-axis label and legend
    plt.xlabel(r'1/$\lambda$ ($\mu\mathrm{m}^{-1}$)', fontsize=16)
    plt.ylabel(r'A($\lambda$)/E(B-V)', fontsize=16)
    plt.legend()

    #   Add grid
    plt.grid(True, color='lightgray', linestyle='--', alpha=0.3)

    plt.show()
    plt.close()


def filled_iso_contours(
        object_table: Table, shape_image: tuple[int, int], filter_: str,
        output_dir: str = './', fraction_bright_objects_to_use: float = 0.2,
        spacing_grid_positions: int = 20, object_property: str = 'fwhm',
        file_type: str = 'pdf') -> None:
    """
    Filled iso contour surfaces

    Parameter
    ---------
    object_table
        Table with object positions (XY) in Pixel

    shape_image
        Dimension of the input image

    filter_
        Filter name

    output_dir
        Path to the directory where the master files should be saved to
        Default is ``.``.

    fraction_bright_objects_to_use
        Fraction of bright objects to use for iso contour determination
        Default is ``0.2``

    spacing_grid_positions
        Spacing between grid positions, usually in Pixel.
        Default is ``20``

    object_property
        Property of the objects used to derive the iso contour levels
        Default is ``fwhm``

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Limit object table to the most
    n_sources = len(object_table)
    object_table.sort('flux', reverse=True)
    object_table = object_table[0:int(n_sources * fraction_bright_objects_to_use)]

    #   Define positions and apertures
    xy_object_position = np.transpose(
        (object_table['y_centroid'], object_table['x_centroid'])
    )

    #   Set up mesh and define grid positions
    x, y = np.meshgrid(
        np.arange(0, shape_image[1], spacing_grid_positions),
        np.arange(0, shape_image[0], spacing_grid_positions)
    )
    xy_grid_shape = x.shape
    xy_grid_positions = np.array([y.ravel(), x.ravel()]).T

    #   Find matches between object and grid positions and assign z values
    object_tree = KDTree(xy_object_position, leafsize=100)
    _, nearst_neighbour_indexes = object_tree.query(xy_grid_positions, k=1)

    if object_property in object_table.colnames:
        z = object_table[object_property].value[nearst_neighbour_indexes]
    else:
        terminal_output.print_to_terminal(
            f'{object_property} is not available. Try roundness instead.',
        )
        if 'roundness' in object_table.colnames:
            z = object_table['roundness'].value[nearst_neighbour_indexes]
            object_property = 'roundness'
        elif 'roundness1' in object_table.colnames:
            z = object_table['roundness1'].value[nearst_neighbour_indexes]
            object_property = 'roundness1'
        else:
            raise RuntimeError('Roundness is also not available.')
    z = z.reshape(xy_grid_shape)

    #   Setup plot
    fig, ax = plt.subplots(figsize=(20, 20))

    #   Plot contours
    cs = ax.contourf(x, y, z)
    ax.contour(
        cs,
        colors='k',
        origin='lower',
    )
    ax.set_title(object_property.upper())

    #   Add color bar
    fig.colorbar(cs)

    # Plot grid
    ax.grid(c='k', ls='-', alpha=0.3)

    #   Save plot
    plt.savefig(
        f'{output_dir}/aberration/aberration_iso_contours_{filter_}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()
    # plt.show()


def histogram_statistic(
        parameter_list_0: list[np.ndarray], name_x: str, name_y: str, rts: str,
        output_dir: str, dataset_label: list[list[str]] | None = None,
        name_object: str = None, parameter_list_1: list[np.ndarray] = None,
        file_type: str = 'pdf',
    ) -> None:
    """
    Plots histogram statistics on properties such as the zero point

    Parameters
    ----------
    parameter_list_0
        List of arrays with parameters to plot

    name_x
        Name of quantity 1

    name_y
        Name of quantity 2

    rts
        Expression characterizing the plot

    output_dir
        Output directory

    dataset_label
        Label for the datasets
        Default is ``None``.

    name_object
        Name of the object
        Default is ``None``

    parameter_list_1
        Second list of arrays with parameters to plot such as sigma
        clipped values of parameter_list_0
        Default is ``None``

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'calibration'),
    )

    #   Plot magnitudes
    fig = plt.figure(figsize=(8, 8))

    #   Limit the space for the object names in case several are given
    if isinstance(name_object, list):
        name_object = ', '.join(name_object)
        if len(name_object) > 20:
            name_object = name_object[0:16] + ' ...'

    #   Set title
    if name_object is None:
        sub_title = f'{name_x} histogram'
    else:
        sub_title = f'{name_x} histogram ({name_object})'
    fig.suptitle(
        sub_title,
        fontsize=17,
    )

    #   Make color map
    color_pick = mk_colormap(len(parameter_list_0))

    for i, parameter in enumerate(parameter_list_0):
        plt.hist(
            parameter,
            bins=40,
            alpha=0.25,
            color=color_pick.to_rgba(i),
            label=f'{dataset_label[0][i]}',
        )
        median_parameter = np.ma.median(parameter)
        if isinstance(median_parameter, u.quantity.Quantity):
            median_parameter = median_parameter.value
        plt.axvline(
            median_parameter,
            # color='g',
            color=color_pick.to_rgba(i),
        )

    if parameter_list_1 is not None:
        for i, parameter in enumerate(parameter_list_1):
            plt.hist(
                parameter,
                bins=10,
                alpha=0.5,
                color=color_pick.to_rgba(i),
                label=f'{dataset_label[1][i]}',
            )

            median_parameter = np.ma.median(parameter)
            if isinstance(median_parameter, u.quantity.Quantity):
                median_parameter = median_parameter.value
            plt.axvline(
                median_parameter,
                color=color_pick.to_rgba(i),
            )

    #   Add legend
    if dataset_label is not None:
        plt.legend()

    #   Set x and y axis label
    plt.ylabel(name_y)
    plt.xlabel(name_x)

    #   Save plot
    plt.savefig(
        f'{output_dir}/calibration/{rts}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()




def plot_extinction_fit_value_airmass(
    output_dir: str | Path,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    use_magnitude: bool = True,
    y_label: str | None = None,
    file_type: str = "pdf",
) -> None:
    """
    Plot extinction fit from flux/magnitude vs airmass (cat-star.org method).

    For each filter: scatter of y vs airmass with regression line and k ± err in title.
    Supports per-star fits (multiple series) or single overall fit.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/extinction_fit/.
    data_by_filter : dict
        {filter_name: (airmass_arr, y_arr)} with airmass and ln(flux) or magnitude.
    coefficients : dict
        {filter_name: ExtinctionCoefficients} from fit_extinction_from_value_airmass.
    use_magnitude : bool
        If True, y is magnitude (slope = k). If False, y is ln(flux) (slope = -k).
    y_label : str, optional
        Override y-axis label (e.g. "m [mag]" or "ln(flux)").
    file_type : str
        Plot file format (pdf, png, etc.). Default is ``pdf``.
    """
    from ... import checks

    out = Path(output_dir) / "extinction_fit"
    checks.check_output_directories(out)

    for filter_, (airmass, y) in data_by_filter.items():
        ec = coefficients.get(filter_)
        if ec is None:
            continue
        k = ec.k_prime
        k_err = ec.k_prime_err

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(airmass, y, alpha=0.7, s=20, color="C0", edgecolors="none")

        # Regression line
        slope = k if use_magnitude else -k
        intercept = float(np.nanmean(y) - slope * np.nanmean(airmass))
        airmass_line = np.linspace(airmass.min(), airmass.max(), 50)
        ax.plot(airmass_line, slope * airmass_line + intercept, "C1-", lw=2, label="Fit")

        ax.set_xlabel("Airmass X")
        if y_label is not None:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel("m [mag]" if use_magnitude else "ln(flux)")
        ax.set_title(f"Filter {filter_}: k' = {k:.4f} ± {k_err:.4f} mag/airmass")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.savefig(
            out / f"extinction_value_airmass_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


def plot_extinction_fit_comparison_stars(
    output_dir: str | Path,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    file_type: str = "pdf",
) -> None:
    """
    Plot extinction fit from comparison stars (mean(m_obs - m_std) vs airmass).

    For each filter: scatter of delta vs X with regression line and k ± err in title.
    One point per frame.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/extinction_fit/.
    data_by_filter : dict
        {filter_name: (X_arr, delta_arr)} with airmass and mean(m_obs - m_std).
    coefficients : dict
        {filter_name: ExtinctionCoefficients} from fit_extinction_from_comparison_stars.
    file_type : str
        Plot file format (pdf, png, etc.). Default is ``pdf``.
    """
    from ... import checks

    out = Path(output_dir) / "extinction_fit"
    checks.check_output_directories(out)

    for filter_, (airmass, delta) in data_by_filter.items():
        ec = coefficients.get(filter_)
        if ec is None:
            continue
        k = ec.k_prime
        k_err = ec.k_prime_err

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(airmass, delta, alpha=0.7, s=40, color="C0", edgecolors="none")

        # Regression line
        slope = k
        intercept = float(np.nanmean(delta) - slope * np.nanmean(airmass))
        airmass_line = np.linspace(airmass.min(), airmass.max(), 50)
        ax.plot(airmass_line, slope * airmass_line + intercept, "C1-", lw=2, label="Fit")

        ax.set_xlabel("Airmass X")
        ax.set_ylabel(r"$\langle m_{\mathrm{obs}} - m_{\mathrm{std}} \rangle$ [mag]")
        ax.set_title(f"Filter {filter_}: k' = {k:.4f} ± {k_err:.4f} mag/airmass")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.savefig(
            out / f"extinction_comparison_stars_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()


def plot_calibration_transformation(
    output_dir: str | Path,
    epoch_id: str,
    data_by_filter: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    coefficients: dict[str, object],
    file_type: str = "pdf",
) -> None:
    """
    Plot calibration transformation fit: m_std - m_inst vs color.

    For each filter: scatter of (m_std - m_inst) vs color index with fit line
    T*color + ZP, and residuals panel. Allows checking fit quality and outliers.

    Parameters
    ----------
    output_dir : str or Path
        Base output directory. Plots saved to output_dir/calibration/.
    epoch_id : str
        Identifier for the calibration epoch (e.g. ``epoch_000``).
    data_by_filter : dict
        {filter: (color_arr, delta_arr, mask)} with color index, m_std - m_inst,
        and boolean mask of stars used in fit.
    coefficients : dict
        {filter: TransformationCoefficients} with T, ZP, color_index_filters.
    file_type : str
        Plot file format. Default is ``pdf``.
    """
    from ... import checks

    out = Path(output_dir) / "calibration"
    checks.check_output_directories(out)

    for filter_, (color, delta, mask) in data_by_filter.items():
        tc = coefficients.get(filter_)
        if tc is None:
            continue
        T, ZP = tc.color_term, tc.zero_point
        ci = f"({tc.color_index_filters[0]}-{tc.color_index_filters[1]})"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Left: m_std - m_inst vs color with fit
        n_excl = np.sum(~mask)
        if n_excl > 0:
            ax1.scatter(color[~mask], delta[~mask], alpha=0.4, s=15, c="gray", label="excluded")
        ax1.scatter(color[mask], delta[mask], alpha=0.7, s=25, c="C0", label="used")
        c_used = np.asarray(color, dtype=float)[mask]
        c_finite = c_used[np.isfinite(c_used)]
        if len(c_finite) > 0 and np.nanmax(c_finite) - np.nanmin(c_finite) > 0.01:
            c_min, c_max = float(np.nanmin(c_finite)), float(np.nanmax(c_finite))
            c_line = np.linspace(c_min, c_max, 50)
            ax1.plot(c_line, T * c_line + ZP, "C1-", lw=2, label="Fit")
        else:
            ax1.axhline(ZP, color="C1", ls="-", lw=2, label="Fit (ZP only)")
        ax1.set_xlabel(f"Color {ci} [mag]")
        ax1.set_ylabel(r"$m_{\mathrm{std}} - m_{\mathrm{inst}}$ [mag]")
        ax1.set_title(f"{epoch_id} {filter_}: T={T:.4f}, ZP={ZP:.4f}")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right: residuals
        residuals = delta - (T * color + ZP)
        if n_excl > 0:
            ax2.scatter(color[~mask], residuals[~mask], alpha=0.4, s=15, c="gray")
        ax2.scatter(color[mask], residuals[mask], alpha=0.7, s=25, c="C0")
        ax2.axhline(0, color="C1", ls="--", lw=1)
        ax2.set_xlabel(f"Color {ci} [mag]")
        ax2.set_ylabel("Residual [mag]")
        rms_val = np.nanstd(residuals[mask]) if np.sum(mask) > 0 else 0.0
        ax2.set_title(f"RMS = {rms_val:.4f} mag")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_id = str(epoch_id).replace("/", "_").replace(":", "_")
        plt.savefig(
            out / f"calibration_{safe_id}_{filter_}.{file_type}",
            bbox_inches="tight",
            format=file_type,
        )
        plt.close()

