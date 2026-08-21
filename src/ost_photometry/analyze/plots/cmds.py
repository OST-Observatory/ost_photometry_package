"""Color-magnitude diagram plotting (MakeCMDs)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from matplotlib import gridspec
from scipy.spatial import KDTree

from ... import checks, terminal_output
from .cmd_reddening import combine_cmd_error_bars, reddening_for_absolute_cmd
from .style import (
    MaxRecursionError,
    initialize_plot,
    mk_colormap,
    mk_line_cycler,
    mk_ticks_labels,
)

plt.switch_backend("Agg")

class MakeCMDs:
    """
    This class contains the necessary functionality for color magnitude plots.
    It allows:
        * to create apparent CMDs
        * to create absolute CMDs
        * to plot isochrones
        * to fit isochrone to the absolute CMD
    """

    def __init__(
            self, name_of_star_cluster: str, file_name: str, file_type: str,
            filter_2: str, filter_1: str, magnitude_color: np.ndarray,
            magnitude_filter_2: np.ndarray,
            color_err: np.ndarray | None = None,
            magnitude_filter_2_err: np.ndarray | None = None,
            output_dir: str = 'output') -> None:
        """
        Parameters
        ----------
        name_of_star_cluster
            Name of cluster

        file_name
            Base name of the file to write

        file_type
            File type

        filter_2
            First filter

        filter_1
            Second filter

        magnitude_color
            Color - 1D

        magnitude_filter_2
            Filter magnitude - 1D

        color_err
            Error for ``mag_color``
            Default is ``None``.

        magnitude_filter_2_err
            Error for ``magnitude_filter_2``
            Default is ``None``.

        output_dir
            Output directory
            Default is ``output``.
        """
        self.name_of_star_cluster = name_of_star_cluster
        self.file_name = file_name
        self.file_type = file_type
        self.filter_2 = filter_2
        self.filter_1 = filter_1
        self.color = f'{filter_1}-{filter_2}'
        self.magnitude_color = magnitude_color
        self.magnitude_filter_2 = magnitude_filter_2
        self.magnitude_color_err = color_err
        self.magnitude_filter_2_err = magnitude_filter_2_err
        self.output_dir = output_dir

        #   Additional attributes filled later
        self.magnitude_filter_2_absolute: np.ndarray | None = None
        self.magnitude_color_absolute: np.ndarray | None = None

    def set_cmd_plot_details(
            self, y_range_max: str | float, y_range_min: str | float,
            x_range_max: str | float, x_range_min: str | float,
            ax: plt.subplot) -> None:
        """
        Check the CMD plot dimensions and set defaults

        Parameters
        ----------
        y_range_max
            The maximum of the plot range in Y direction

        y_range_min
            The minimum of the plot range in Y direction

        x_range_max
            The maximum of the plot range in X direction

        x_range_min
            The minimum of the plot range in X direction

        ax
            Subplot
        """
        #   Check for absolute vs. apparent CMD
        try:
            magnitude_2 = self.magnitude_filter_2_absolute
            color = self.magnitude_color_absolute
        except AttributeError:
            magnitude_2 = self.magnitude_filter_2
            color = self.magnitude_color

        if magnitude_2 is None:
            magnitude_2 = self.magnitude_filter_2
            color = self.magnitude_color

        #   Set plot range -> automatic adjustment
        #   Y range
        try:
            float(y_range_max)
        except ValueError:
            ax.set_ylim([
                float(np.max(magnitude_2)) + 0.5,
                float(np.min(magnitude_2)) - 0.5
            ])
            terminal_output.print_to_terminal(
                "[Info] Use automatic plot range for Y",
                style_name='WARNING',
            )
        else:
            try:
                float(y_range_min)
            except ValueError:
                ax.set_ylim([
                    float(np.max(magnitude_2)) + 0.5,
                    float(np.min(magnitude_2)) - 0.5
                ])
                terminal_output.print_to_terminal(
                    "[Info] Use automatic plot range for Y",
                    style_name='WARNING',
                )
            else:
                ax.set_ylim([float(y_range_min), float(y_range_max)])

        #   X range
        try:
            float(x_range_max)
        except ValueError:
            ax.set_xlim([
                float(np.min(color)) - 0.5,
                float(np.max(color)) + 0.5
            ])
            terminal_output.print_to_terminal(
                "[Info] Use automatic plot range for X",
                style_name='WARNING',
            )
        else:
            try:
                float(x_range_min)
            except ValueError:
                ax.set_xlim([
                    float(np.min(color)) - 0.5,
                    float(np.max(color)) + 0.5
                ])
                terminal_output.print_to_terminal(
                    "[Info] Use automatic plot range for X",
                    style_name='WARNING',
                )
            else:
                ax.set_xlim([float(x_range_min), float(x_range_max)])

    def write_cmd(self, plot_type: str):
        """
        Write plot to disk

        Parameters
        ----------
        plot_type
            Plot type
        """
        cmd_dir = f'{self.output_dir}/cmds'
        checks.check_output_directories(cmd_dir)

        if self.name_of_star_cluster == "" or self.name_of_star_cluster == "?":
            path = (f'{cmd_dir}/{self.file_name}_{plot_type}'
                    f'_{self.filter_2}_{self.color}.{self.file_type}')
            terminal_output.print_to_terminal(
                f"Save CMD plot ({self.file_type}): {path}",
            )
            plt.savefig(
                path,
                format=self.file_type,
                bbox_inches="tight",
            )
        else:
            name_of_star_cluster = self.name_of_star_cluster.replace(
                ' ',
                '_',
            )
            path = (f'{cmd_dir}/{self.file_name}_{name_of_star_cluster}'
                    f'_{plot_type}_{self.filter_2}_{self.color}'
                    f'.{self.file_type}')
            terminal_output.print_to_terminal(
                f"Save CMD plot ({self.file_type}): {path}\n",
            )
            plt.savefig(
                path,
                format=self.file_type,
                bbox_inches="tight",
            )

    def decode_isochrone_filter_relation(
            self, isochrone_column_type: dict[str, str],
            isochrone_column: dict[str, int], current_filter: str,
            relation_list: list[tuple[int, int]], recursion_number: int
            ) -> list[tuple[int, int]]:
        """
        Decodes relationship between isochrone entries. It fills a list with
        tuples of two in integer each. The first integer gives the ID of the filter
        and the second one specifies how the magnitude is derived from the
        relationships. The second integer can be 1 or -1 and determines whether the
        isochrone magnitude of this particular relationship must be added or
        subtracted.

        Parameter
        ---------
        isochrone_column_type
            Type of the columns from the ISO file
            Keys = filter : `string`
            Values = type : `string`

        isochrone_column
            Columns to use from the ISO file.
            Keys = filter           : `string`
            Values = column numbers : `integer`

        current_filter
            Current filter

        relation_list
            List with relations. Each tuple is one relationship. In each tuple the
            first integer gives the ID of the filter and the second one determines
            how the magnitude is derived from the relationships. The second integer
            can be 1 or -1 and determines whether the isochrone magnitude of this
            particular relationship must be added or subtracted.

        recursion_number

        Returns
        -------
        relation_list
            See above
        """
        #   Exit if recursion is two high
        if recursion_number > 10:
            raise MaxRecursionError(
                'Could not decode magnitudes from isochrone file '
                'because maximum number of recursions reached during '
                'color calculation'
            )

        #   Distinguish between color and 'single' magnitude entries
        if isochrone_column_type[current_filter][0] == 'single':
            relation_list.append(
                (isochrone_column[current_filter], 1)
            )
            return relation_list
        else:
            #   Set filter from color
            next_filter = isochrone_column_type[current_filter][2]

            #   Repeat until a single magnitude is found
            relation_list = self.decode_isochrone_filter_relation(
                isochrone_column_type,
                isochrone_column,
                next_filter,
                relation_list,
                recursion_number + 1,
            )

            #   Now we have to distinguish between, e.g., B-V vs. V-B
            if isochrone_column_type[current_filter][1] == 0:
                relation_list.append(
                    (isochrone_column[current_filter], 1)
                )

            else:
                relation_list.append(
                    (isochrone_column[current_filter], -1)
                )

            return relation_list

    @staticmethod
    def apply_isochrone_filter_relation(
            relation_list: list[tuple[int, int]], iso_data_line: list[str]
            ) -> float:
        """
        Uses isochrone filter relation such as color to derive individual
        magnitudes

        Parameter
        ---------
        relation_list
            List with relations. Each tuple is one relationship. In each tuple the
            first integer gives the ID of the filter and the second one determines
            how the magnitude is derived from the relationships. The second integer
            can be 1 or -1 and determines whether the isochrone magnitude of this
            particular relationship must be added or subtracted.

        iso_data_line
            Line with iso data - list of strings

        Returns
        -------
        target_magnitude
            Calculated magnitude
        """
        target_magnitude = 0.

        #   Calculate magnitude
        for relation in relation_list:
            relation_magnitude = float(iso_data_line[relation[0] - 1]) * relation[1]
            target_magnitude = target_magnitude + relation_magnitude

        return target_magnitude

    def fill_lists_with_isochrone_magnitudes(
            self, isochrone_data_line: list[str],
            isochrone_relation_filter_1: list[tuple[int, int]],
            isochrone_relation_filter_2: list[tuple[int, int]],
            isochrone_magnitude_2: list[float], isochrone_color: list[float]
            ) -> tuple[list[float], list[float]]:
        """
        Sort magnitudes and colors from isochrone files into lists and calculate
        the required color if necessary

        Parameter
        ---------
        isochrone_data_line
            Line with iso data - list of strings

        isochrone_relation_filter_1
            List with relation for filter 1. Each tuple is one relationship. In
            each tuple the first integer gives the ID of the filter and the second
            one determines how the magnitude is derived from the relationships. The
            second integer can be 1 or -1 and determines whether the isochrone
            magnitude of this particular relationship must be added or subtracted.

        isochrone_relation_filter_2
            List with relation for filter 2. Each tuple is one relationship. In
            each tuple the first integer gives the ID of the filter and the second
            one determines how the magnitude is derived from the relationships. The
            second integer can be 1 or -1 and determines whether the isochrone
            magnitude of this particular relationship must be added or subtracted.

        isochrone_magnitude_2
            List to fill with magnitudes (second filter)

        isochrone_color
            List to fill with color values

        Returns
        -------
        isochrone_magnitude_2
            Magnitude list (second filter)

        isochrone_color
            Color list
        """
        #   Calculate magnitudes and color
        magnitude_1 = self.apply_isochrone_filter_relation(
            isochrone_relation_filter_1,
            isochrone_data_line,
        )
        magnitude_2 = self.apply_isochrone_filter_relation(
            isochrone_relation_filter_2,
            isochrone_data_line,
        )
        color = magnitude_1 - magnitude_2

        isochrone_magnitude_2.append(magnitude_2)
        isochrone_color.append(color)

        return isochrone_magnitude_2, isochrone_color

    @staticmethod
    def calculate_chi_square(
            magnitude_filter_2: np.ndarray, magnitude_color: np.ndarray,
            isochrone_array: np.ndarray, nearst_neighbour_indexes: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Parameters
        ----------
        magnitude_filter_2
            Object magnitudes of filter 2

        magnitude_color
            Object colors

        isochrone_array
            Array with isochrone data

        nearst_neighbour_indexes
            Indexes of the nearest isochrone points to the reference points
            of the observed objects.

        Returns
        -------
        chi_square_magnitude_2
            Chi square based on object magnitudes

        chi_square_color
            Chi square based on object color

        chi_square_list
            See above
        """
        #   Calculate chi square
        chi_square_magnitude_2 = np.square(
            magnitude_filter_2[:, 1] - isochrone_array[:, 0][nearst_neighbour_indexes]
        ).sum()
        chi_square_color = np.square(
            magnitude_color[:, 1] - isochrone_array[:, 1][nearst_neighbour_indexes]
        ).sum()
        chi_square_total = chi_square_magnitude_2 + chi_square_color

        return chi_square_magnitude_2, chi_square_color, chi_square_total

    def plot_apparent_cmd(
            self, figure_size_x: str = '', figure_size_y: str = '',
            y_plot_range_max: str = '', y_plot_range_min: str = '',
            x_plot_range_max: str = '', x_plot_range_min: str = '') -> None:
        """
        Plot calibrated cmd with apparent magnitudes

        Parameters
        ----------
        figure_size_x
            Figure size in cm (x direction)

        figure_size_y
            Figure size in cm (y direction)

        y_plot_range_max
            The maximum of the plot range in Y direction

        y_plot_range_min
            The minimum of the plot range in Y direction

        x_plot_range_max
            The maximum of the plot range in X direction

        x_plot_range_min
            The minimum of the plot range in X direction
        """
        #   Initialize, set defaults and check plot dimensions
        initialize_plot(
            figure_size_x,
            figure_size_y,
        )

        ax0 = plt.subplot(1, 1, 1)

        self.set_cmd_plot_details(
            y_plot_range_max,
            y_plot_range_min,
            x_plot_range_max,
            x_plot_range_min,
            ax0,
        )

        #   Plot the stars
        terminal_output.print_to_terminal("Add stars", indent=1)
        ax0.errorbar(
            self.magnitude_color,
            self.magnitude_filter_2,
            yerr=self.magnitude_filter_2_err,
            xerr=self.magnitude_color_err,
            marker='o',
            ls='none',
            elinewidth=0.5,
            markersize=2,
            capsize=2,
            ecolor='#ccdbfd',
            color='darkred',
            alpha=0.4,
        )

        #   Set ticks and labels
        mk_ticks_labels(
            rf'${self.filter_2}$ [mag]',
            rf'${self.color}$ [mag]',
            ax0,
        )

        #   Write plot to disk
        self.write_cmd('apparent')
        plt.close()

    def plot_absolute_cmd(
            self, e_b_v: float, m_m: float, isochrones: str,
            isochrone_type: str, isochrone_column_type: dict[str, str],
            isochrone_column: dict[str, int], isochrone_log_age: bool,
            isochrone_keyword: str, isochrone_legend: bool,
            figure_size_x: str = '', figure_size_y: str = '',
            y_plot_range_max: str = '', y_plot_range_min: str = '',
            x_plot_range_max: str = '', x_plot_range_min: str = '',
            rv: float = 3.1, e_b_v_err: float | None = None,
            rv_err: float | None = None, fit_isochrone: bool = False,
            magnitude_fit_range: tuple[float | None, float | None] = (None, None),
            n_bin_observation: int = 40,
            fiduciary_points_observation: bool | None = None,
            fiduciary_points_isochrones: bool = False,
            chi_square_plot_mode: str | None = None) -> None:
        """
        Plot calibrated CMD with
            * magnitudes corrected for reddening and distance
            * isochrones

        Parameters
        ----------
        e_b_v                       : `float`
            Relative extinction between B and V band.

        e_b_v_err                   : `float` or `None`, optional
            1-sigma uncertainty on ``e_b_v``. Propagated into both plotted
            axes and combined in quadrature with the photometric errors
            (independent of ``rv_err``; no covariance).
            Default is ``None`` (photometric errors only).

        m_m                         : `float`
            Distance modulus

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

        rv                          : `float`, optional
            Ratio between absolute and relative extinction.
            Default is ``3.1``.

        rv_err                      : `float` or `None`, optional
            1-sigma uncertainty on ``rv``. Propagated into the extinction
            correction (and, except for B-V, into the colour excess) and
            combined in quadrature with the photometric errors.
            Default is ``None`` (photometric errors only).

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

        fit_isochrone               : `bool`, optional
            If `True`, the best fitting isochrone will be determined.
            Default is ``False``.

        magnitude_fit_range         : `tuple` of `float` or `None`
            Magnitude range to be used for the isochrone fitting and binning
            of the observations. If set to None, the minimum and maximum
            value are used.
            Default is ``(None, None)``,

        n_bin_observation           : `integer`, optional
            Number of bins into which the observation data will be combined.
            Default is ``40``.

        fiduciary_points_observation : `bool` or `None`, optional
            Determined if the binned observation will be plotted. Is set to
            `True` if fit_isochrone is `True` with the exception that
            fiduciary_points_observation is explicitly set to `False`.
            Default is ``None``.

        fiduciary_points_isochrones  : `bool`, optional
            If 'True', the isochrone points closest to the fiduciary observation
            points will be plotted.
            Default is ``False``.

        chi_square_plot_mode        : `string` or None, optional
            Mode to plot the chi square values from the isochrone fits.
            Possibilities: 1. simple   -> Combined chi square values shown on
                                          the right hand side.
                           2. detailed -> Chi square values split according
                                          to X and Y contributions. Plots are
                                          on top and on the right hand side of
                                          the CMD
            If `None` and fit_isochrone is `True` chi_square_plot_mode is set
            to `simple`.
            Default is ``None``.
        """
        #   Correct for reddening and distance
        a_filter_2, relative_extinction, a_filter_2_err, relative_extinction_err = (
            reddening_for_absolute_cmd(
                self.filter_1,
                self.filter_2,
                rv,
                e_b_v,
                e_b_v_err=e_b_v_err,
                rv_err=rv_err,
            )
        )
        magnitude_filter_2 = self.magnitude_filter_2 - a_filter_2 - m_m
        magnitude_color = self.magnitude_color - relative_extinction
        self.magnitude_filter_2_absolute = magnitude_filter_2
        self.magnitude_color_absolute = magnitude_color
        magnitude_filter_2_err = combine_cmd_error_bars(
            self.magnitude_filter_2_err,
            a_filter_2_err,
        )
        magnitude_color_err = combine_cmd_error_bars(
            self.magnitude_color_err,
            relative_extinction_err,
        )

        #   Plot fiduciary points if isochrone fit is performed
        if fiduciary_points_observation is None and fit_isochrone:
            fiduciary_points_observation = True
        #   Plot chi square deviation of isochrones from fiduciary points if fit is
        #   performed
        if fit_isochrone and chi_square_plot_mode is None:
            chi_square_plot_mode = 'simple'

        #   Initialize plot and check plot dimensions
        fig = initialize_plot(
            figure_size_x,
            figure_size_y,
        )

        #   Create grid for different subplots
        spec = gridspec.GridSpec(
            ncols=2,
            nrows=2,
            width_ratios=[4, 1],
            wspace=0.3,
            hspace=0.2,
            height_ratios=[4, 1],
        )

        #   Add main plot to plot grid
        ax0 = fig.add_subplot(spec[0])

        #   Set plot details
        self.set_cmd_plot_details(
            y_plot_range_max,
            y_plot_range_min,
            x_plot_range_max,
            x_plot_range_min,
            ax0,
        )

        #   Plot the stars
        terminal_output.print_to_terminal("Add stars")
        ax0.errorbar(
            magnitude_color,
            magnitude_filter_2,
            yerr=magnitude_filter_2_err,
            xerr=magnitude_color_err,
            marker='o',
            ls='none',
            elinewidth=0.5,
            markersize=2,
            capsize=2,
            ecolor='#ccdbfd',
            color='darkred',
            alpha=0.3,
        )

        #   Bin observation
        if fiduciary_points_observation or fit_isochrone:
            #   Check if fit range is defined. If not, minimum and maximum
            #   values of the data are used.
            if magnitude_fit_range[0] is None:
                min_magnitude_filter_2 = np.min(magnitude_filter_2)
            else:
                min_magnitude_filter_2 = magnitude_fit_range[0]
            if magnitude_fit_range[1] is None:
                max_magnitude_filter_2 = np.max(magnitude_filter_2)
            else:
                max_magnitude_filter_2 = magnitude_fit_range[1]

            #   Define bins
            bins = np.linspace(
                min_magnitude_filter_2,
                max_magnitude_filter_2,
                n_bin_observation,
            )

            #   Perform binning
            digitized = np.digitize(magnitude_filter_2, bins)
            magnitude_filter_2_binned = np.array([
                sigma_clipped_stats(magnitude_filter_2[digitized == i])
                for i in range(1, len(bins))
                if np.any(digitized == i)
            ])
            magnitude_color_binned = np.array([
                sigma_clipped_stats(magnitude_color[digitized == i])
                for i in range(1, len(bins))
                if np.any(digitized == i)
            ])
            magnitude_binned_array = np.column_stack(
                (magnitude_filter_2_binned[:, 1], magnitude_color_binned[:, 1])
            )

            if fiduciary_points_observation:
                ax0.errorbar(
                    magnitude_color_binned[:, 1],
                    magnitude_filter_2_binned[:, 1],
                    xerr=magnitude_color_binned[:, 2],
                    yerr=magnitude_filter_2_binned[:, 2],
                    marker='o',
                    ls='none',
                    elinewidth=1.0,
                    markersize=5,
                    capsize=3,
                    ecolor='#338af7',
                    color='#F8B195',
                    alpha=0.9,
                    zorder=99.,
                )
        else:
            magnitude_binned_array = None
            magnitude_filter_2_binned = None
            magnitude_color_binned = None

        #   Plot isochrones
        #
        #   Check if isochrones are specified
        if isochrones != '' and isochrones != '?':
            #   Decode relationships between isochrone magnitudes such as color
            #   relationships
            isochrone_magnitude_relation_filter_1 = self.decode_isochrone_filter_relation(
                isochrone_column_type,
                isochrone_column,
                self.filter_1,
                [],
                0,
            )
            isochrone_magnitude_relation_filter_2 = self.decode_isochrone_filter_relation(
                isochrone_column_type,
                isochrone_column,
                self.filter_2,
                [],
                0,
            )

            #   Initialize chi square subplots
            if chi_square_plot_mode == 'detailed' and fit_isochrone:
                ax1 = fig.add_subplot(spec[1])
                ax2 = fig.add_subplot(spec[2])
            elif chi_square_plot_mode == 'simple' and fit_isochrone:
                ax2 = fig.add_subplot(spec[2])
            else:
                ax1 = None
                ax2 = None

            #   Prepare list for chi square values
            age_list = []
            chi_square_list = []
            chi_square_magnitude_2_list = []
            chi_square_color_list = []
            isochrones_list = []

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
                        except (ValueError, IndexError):
                            #   Try to find and extract age information
                            if 'Age' in line_elements or 'age' in line_elements:
                                try:
                                    age_index = line_elements.index('age')
                                except ValueError:
                                    age_index = line_elements.index('Age')

                                for string in line_elements[age_index + 1:]:
                                    #   Find age unit
                                    if string.rfind("yr") != -1:
                                        age_unit = string
                                    #   Find age value
                                    try:
                                        if isinstance(age_value, str):
                                            age_value = float(string)
                                            if age_value >= 1000. and age_unit.rfind('Myr') != -1:
                                                age_value /= 1000.
                                                age_unit = 'Gyr'
                                            if age_unit.rfind('Myr') != -1:
                                                age_unit = 'Myr'
                                            age_list.append(age_value)
                                    except (TypeError, ValueError):
                                        pass
                            continue

                        #   Fill lists
                        isochrone_magnitude_2, isochrone_color = self.fill_lists_with_isochrone_magnitudes(
                            line_elements,
                            isochrone_magnitude_relation_filter_1,
                            isochrone_magnitude_relation_filter_2,
                            isochrone_magnitude_2,
                            isochrone_color,
                        )

                    #   Close file with the iso data
                    isochrone_data.close()

                    #   Construct label
                    if not isinstance(age_value, str):
                        label = str(age_value)
                        if age_unit != '':
                            label += f' {age_unit}'
                    else:
                        label = os.path.splitext(file_list[i])[0]

                    if fit_isochrone:
                        #   Find points to compare with binned observations
                        isochrone_array = np.array(
                            [isochrone_magnitude_2, isochrone_color]
                        ).T
                        isochrones_list.append(isochrone_array)
                        isochrone_tree = KDTree(isochrone_array, leafsize=100)
                        _, nearst_neighbour_indexes = isochrone_tree.query(
                            magnitude_binned_array,
                            k=1,
                        )
                    else:
                        nearst_neighbour_indexes = None
                        isochrone_array = None

                    #   Plot iso lines
                    if fiduciary_points_isochrones:
                        ax0.plot(
                            isochrone_array[:, 1][nearst_neighbour_indexes],
                            isochrone_array[:, 0][nearst_neighbour_indexes],
                            marker='o',
                            ls='none',
                            color=color_pick.to_rgba(i),
                            alpha=0.5,
                        )
                    if fit_isochrone:
                        alpha_isochrone = 0.2
                    else:
                        alpha_isochrone = 0.5
                    ax0.plot(
                        isochrone_color,
                        isochrone_magnitude_2,
                        linestyle=next(line_cycler),
                        color=color_pick.to_rgba(i),
                        linewidth=1.2,
                        label=label,
                        alpha=alpha_isochrone,
                    )

                    if fit_isochrone:
                        #   Calculate chi square
                        chi_square_magnitude_2, chi_square_color, chi_square_total = self.calculate_chi_square(
                            magnitude_filter_2_binned,
                            magnitude_color_binned,
                            isochrone_array,
                            nearst_neighbour_indexes,
                        )
                        chi_square_magnitude_2_list.append(
                            chi_square_magnitude_2
                        )
                        chi_square_color_list.append(chi_square_color)
                        chi_square_list.append(chi_square_total)

                        #   Plot chi square values
                        if chi_square_plot_mode == 'detailed':
                            ax1.scatter(
                                chi_square_magnitude_2,
                                age_value,
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )
                            ax2.scatter(
                                age_value,
                                chi_square_color,
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )
                        elif chi_square_plot_mode == 'simple':
                            ax2.scatter(
                                age_value,
                                chi_square_color + chi_square_magnitude_2,
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )

            #   OPTION II: Isochrone file containing many individual isochrones
            if isochrone_type == 'file':
                #   Resolve iso path
                isochrones = Path(isochrones).expanduser()

                #   Load file
                isochrone_data = open(isochrones)

                #   Overall lists for the isochrones
                nearst_neighbour_indexes_list = []
                isochrone_magnitude_2: list[float] = []
                isochrone_color: list[float] = []
                age: float | str | None = None

                def _flush_current_isochrone() -> None:
                    if not isochrone_magnitude_2:
                        return
                    age_list.append(float(age))
                    isochrone_array = np.column_stack(
                        (isochrone_magnitude_2, isochrone_color)
                    )
                    isochrones_list.append(isochrone_array)
                    if fit_isochrone:
                        isochrone_tree = KDTree(isochrone_array, leafsize=100)
                        _, nearst_neighbour_indexes = isochrone_tree.query(
                            magnitude_binned_array,
                            k=1,
                        )
                        nearst_neighbour_indexes_list.append(
                            nearst_neighbour_indexes
                        )

                #   Loop over all lines in the file
                for line in isochrone_data:
                    line_elements = line.split()

                    #   Check for a key word to distinguish the isochrones
                    try:
                        if line[0:len(isochrone_keyword)] == isochrone_keyword:
                            _flush_current_isochrone()

                            #   Save age for the case where age is given as a
                            #   keyword and not as a column
                            if isochrone_column['AGE'] == 0:
                                age = line.split('=')[1].split()[0]

                            isochrone_magnitude_2 = []
                            isochrone_color = []
                            continue
                    except RuntimeError:
                        continue

                    #   Check that the entries are not HEADER keywords
                    try:
                        float(line_elements[0])
                    except (ValueError, IndexError):
                        continue

                    if isochrone_column['AGE'] != 0:
                        age = float(line_elements[isochrone_column['AGE'] - 1])

                    isochrone_magnitude_2, isochrone_color = self.fill_lists_with_isochrone_magnitudes(
                        line_elements,
                        isochrone_magnitude_relation_filter_1,
                        isochrone_magnitude_relation_filter_2,
                        isochrone_magnitude_2,
                        isochrone_color,
                    )

                _flush_current_isochrone()

                #   Close isochrone file
                isochrone_data.close()

                #   Number of isochrones
                n_isochrones = len(isochrones_list)
                terminal_output.print_to_terminal(
                    f"Plot {n_isochrones} isochrone(s)",
                    style_name='OKGREEN',
                )

                #   Make color map
                color_pick = mk_colormap(n_isochrones)

                #   Prepare cycler for the line styles
                line_cycler = mk_line_cycler()

                #   Cycle through iso lines
                age_list_new = []
                for i in range(0, n_isochrones):
                    if isochrone_log_age:
                        age_value = 10 ** age_list[i] / 10 ** 9
                        age_value = round(age_value, 3)
                    else:
                        age_value = round(age_list[i], 3)
                    age_unit = 'Gyr'
                    age_string = f'{age_value} {age_unit}'
                    age_list_new.append(age_value)

                    #   Plot iso lines
                    if fiduciary_points_isochrones:
                        ax0.plot(
                            isochrones_list[i][:, 1][nearst_neighbour_indexes_list[i]],
                            isochrones_list[i][:, 0][nearst_neighbour_indexes_list[i]],
                            marker='o',
                            ls='none',
                            color=color_pick.to_rgba(i),
                            alpha=0.5,
                        )
                    if fit_isochrone:
                        alpha_isochrone = 0.2
                    else:
                        alpha_isochrone = 0.5
                    ax0.plot(
                        isochrones_list[i][:, 1],
                        isochrones_list[i][:, 0],
                        linestyle=next(line_cycler),
                        color=color_pick.to_rgba(i),
                        linewidth=1.2,
                        label=age_string,
                        alpha=alpha_isochrone,
                    )

                    if fit_isochrone:
                        #   Calculate chi square
                        chi_square_magnitude_2, chi_square_color, chi_square_total = self.calculate_chi_square(
                            magnitude_filter_2_binned,
                            magnitude_color_binned,
                            isochrones_list[i],
                            nearst_neighbour_indexes_list[i],
                        )
                        chi_square_magnitude_2_list.append(
                            chi_square_magnitude_2
                        )
                        chi_square_color_list.append(chi_square_color)
                        chi_square_list.append(chi_square_total)

                        #   Plot chi square values
                        if chi_square_plot_mode == 'detailed':
                            ax1.plot(
                                chi_square_magnitude_2,
                                age_value,
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )
                            ax2.plot(
                                age_value,
                                chi_square_color,
                                ls='none',
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )
                        elif chi_square_plot_mode == 'simple':
                            ax2.scatter(
                                age_value,
                                chi_square_color + chi_square_magnitude_2,
                                color=color_pick.to_rgba(i),
                                marker='o',
                                alpha=0.2,
                            )
                age_list = age_list_new

            #   Plot legend
            if isochrone_legend:
                legend_ = ax0.legend(
                    bbox_to_anchor=(0., 1.02, 1.0, 0.102),
                    loc=3,
                    ncol=4,
                    mode='expand',
                    borderaxespad=0.,
                )
                for element in legend_.legend_handles:
                    element.set_alpha(0.6)

        if fit_isochrone:
            #   Evaluate chi square
            min_chi_square_id = np.argmin(chi_square_list)

            terminal_output.print_to_terminal(
                f'Best fitting isochrone: {age_list[min_chi_square_id]:.1f} '
                f'{age_unit} with chi^2 = {chi_square_list[min_chi_square_id]:.3f}',
                style_name="GOOD",
            )

            #   Plot best isochrone
            ax0.plot(
                isochrones_list[min_chi_square_id][:, 1],
                isochrones_list[min_chi_square_id][:, 0],
                linestyle='-',
                color=color_pick.to_rgba(min_chi_square_id),
                linewidth=2,
            )

            #   Finish chi square plots
            if chi_square_plot_mode == 'detailed':
                ax1.scatter(
                    chi_square_magnitude_2_list[min_chi_square_id],
                    age_list[min_chi_square_id],
                    color=color_pick.to_rgba(min_chi_square_id),
                    marker='o',
                    alpha=1.0,
                )
                ax2.scatter(
                    age_list[min_chi_square_id],
                    chi_square_color_list[min_chi_square_id],
                    color=color_pick.to_rgba(min_chi_square_id),
                    marker='o',
                    alpha=1.0,
                )
                mk_ticks_labels(
                    f'Age [{age_unit}]',
                    r'$\chi^2$ ',
                    ax1,
                )
                mk_ticks_labels(
                    r'$\chi^2$ ',
                    f'Age [{age_unit}]',
                    ax2,
                )
            elif chi_square_plot_mode == 'simple':
                ax2.scatter(
                    age_list[min_chi_square_id],
                    chi_square_magnitude_2_list[min_chi_square_id] + chi_square_color_list[min_chi_square_id],
                    color=color_pick.to_rgba(min_chi_square_id),
                    marker='o',
                    alpha=1.0,
                )
                mk_ticks_labels(
                    r'$\chi^2$ ',
                    f'Age [{age_unit}]',
                    ax2,
                )

        #   Set ticks and labels for CMD
        mk_ticks_labels(
            rf'${self.filter_2}$ [mag]',
            rf'${self.color}$ [mag]',
            ax0,
        )

        #   Write plot to disk
        self.write_cmd('absolut')
        plt.close()


