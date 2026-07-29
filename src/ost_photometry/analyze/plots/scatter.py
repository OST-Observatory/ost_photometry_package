"""2D / 3D scatter plotting helpers."""
from __future__ import annotations

import os

import numpy as np
from astropy.modeling import fitting
import matplotlib.pyplot as plt

from ... import checks

from .style import (
    mk_color_cycler_error_bars,
    mk_color_cycler_symbols,
    mk_line_cycler,
)

plt.switch_backend("Agg")

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


def scatter(
        x_values: list[np.ndarray], name_x: str, y_values: list[np.ndarray],
        name_y: str, rts: str, output_dir: str,
        x_errors: list[np.ndarray | None] = [None],
        y_errors: list[np.ndarray | None] = [None],
        dataset_label: list[str] | None = None, name_object: str | None = None,
        fits: list[fitting] | None = None, one_to_one: bool = False,
        file_type: str = 'pdf') -> None:
    """
    Plot magnitudes

    Parameters
    ----------
    x_values
        List of arrays with X values

    name_x
        Name of quantity 1

    y_values
        List of arrays with Y values

    name_y
        Name of quantity 2

    rts
        Expression characterizing the plot

    output_dir
        Output directory

    x_errors
        Errors for the X values
        Default is ``None``.

    y_errors
        Errors for the Y values
        Default is ``None``.

    dataset_label
        Label for the datasets
        Default is ``None``.

    name_object
        Name of the object
        Default is ``None``

    fits
        List of objects, representing fits to the data
        Default is ``None``.

    one_to_one
        If True a 1:1 line will be plotted.
        Default is ``False``.

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        os.path.join(output_dir, 'scatter'),
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
        sub_title = f'{name_x} vs. {name_y}'
    else:
        sub_title = f'{name_x} vs. {name_y} ({name_object})'
    fig.suptitle(
        sub_title,
        fontsize=17,
    )

    #   Initialize color cyclers
    color_cycler_symbols = mk_color_cycler_symbols()
    color_cycler_error_bars = mk_color_cycler_error_bars()

    #   Prepare cycler for the line styles
    line_cycler = mk_line_cycler()

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
                    linestyle=next(line_cycler),
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
        f'{output_dir}/scatter/{rts}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()


