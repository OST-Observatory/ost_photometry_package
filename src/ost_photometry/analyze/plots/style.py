"""Matplotlib style helpers for analysis plots."""
from __future__ import annotations

from itertools import cycle

import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

from ... import terminal_output


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


