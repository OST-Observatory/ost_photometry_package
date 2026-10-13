"""Color-magnitude diagram plotting (MakeCMDs)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.spatial import KDTree

from ... import checks, terminal_output
from ...output_layout import diagnostics_dir, results_dir, tables_dir
from ..cmd_prepare import format_isochrone_annotation
from ..isochrone_fit import age_interval, bin_fiducials, profile_chi_square
from .cmd_reddening import (
    cmd_correction_offsets,
    reddening_for_absolute_cmd,
)
from .style import (
    MaxRecursionError,
    initialize_plot,
    mk_colormap,
    mk_line_cycler,
    mk_ticks_labels,
)

plt.switch_backend("Agg")


def _overlay_cmd_axes(
    fig,
    *,
    chi_square_plot_mode: str | None,
    draw_chi2: bool,
):
    """CMD axes plus χ² panels only when those panels will be filled."""
    ax1 = None
    ax2 = None
    if draw_chi2 and chi_square_plot_mode == "detailed":
        spec = fig.add_gridspec(
            2,
            2,
            width_ratios=[4, 1],
            height_ratios=[4, 1],
            wspace=0.3,
            hspace=0.25,
        )
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[1, 0])
    elif draw_chi2 and chi_square_plot_mode == "simple":
        spec = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.28)
        ax0 = fig.add_subplot(spec[0])
        ax2 = fig.add_subplot(spec[1])
    else:
        ax0 = fig.add_subplot(1, 1, 1)
    return ax0, ax1, ax2


def _draw_reddening_vector(
    ax,
    *,
    e_color: float,
    a_mag: float,
    color_name: str,
    filter_2: str,
) -> None:
    """Arrow in data coordinates: dust makes stars redder and fainter."""
    if not (np.isfinite(e_color) and np.isfinite(a_mag)):
        return
    if abs(e_color) < 1e-4 and abs(a_mag) < 1e-4:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_start = x0 + 0.10 * (x1 - x0)
    y_start = y1 + 0.14 * (y0 - y1)
    x_end = x_start + e_color
    y_end = y_start + a_mag
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops={
            "arrowstyle": "->",
            "color": "0.2",
            "lw": 1.3,
            "mutation_scale": 12,
        },
        zorder=101,
        clip_on=True,
    )
    e_txt = f"{e_color:.3g}"
    a_txt = f"{a_mag:.3g}"
    ax.annotate(
        rf"$E({color_name})={e_txt}$" "\n" rf"$A_{{{filter_2}}}={a_txt}$",
        xy=(x_start, y_start),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8,
        color="0.2",
        ha="left",
        va="top",
        zorder=101,
    )


def _scatter_cmd_stars(
    ax,
    color: np.ndarray,
    mag: np.ndarray,
    *,
    members: np.ndarray | None,
    mag_err: np.ndarray | float | None,
    color_err: np.ndarray | float | None,
    color_by_error: bool,
    marker_alpha: float = 0.45,
    draw_error_bars: bool = True,
):
    """Field (grey) behind members; optional colour-coding by photometric σ."""
    n = len(color)
    if members is None or members.size != n:
        member_mask = np.ones(n, dtype=bool)
        has_field = False
    else:
        member_mask = np.asarray(members, dtype=bool)
        has_field = bool(np.any(~member_mask))
    if has_field:
        field = ~member_mask
        ax.scatter(
            color[field],
            mag[field],
            s=6,
            c="0.75",
            zorder=1,
            alpha=0.45,
            label="Field",
            rasterized=True,
        )
    m_color = color[member_mask]
    m_mag = mag[member_mask]
    member_label = "Members" if has_field else None
    mag_err_arr = None if mag_err is None else np.broadcast_to(
        np.asarray(mag_err, dtype=float), color.shape
    )
    color_err_arr = None if color_err is None else np.broadcast_to(
        np.asarray(color_err, dtype=float), color.shape
    )
    if color_by_error and mag_err_arr is not None and m_mag.size:
        sc = ax.scatter(
            m_color,
            m_mag,
            s=10,
            c=mag_err_arr[member_mask],
            cmap="viridis_r",
            zorder=3,
            alpha=0.75,
            label=member_label,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(r"$\sigma$ [mag]")
    else:
        ax.scatter(
            m_color,
            m_mag,
            s=10,
            c="darkred",
            zorder=3,
            alpha=marker_alpha,
            label=member_label,
            rasterized=True,
        )
    if draw_error_bars and (mag_err_arr is not None or color_err_arr is not None):
        ax.errorbar(
            color,
            mag,
            yerr=mag_err_arr,
            xerr=color_err_arr,
            fmt="none",
            elinewidth=0.5,
            capsize=2,
            ecolor="#ccdbfd",
            alpha=0.25,
            zorder=2,
        )
    if has_field:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


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
            output_dir: str = 'output',
            is_cluster_member: np.ndarray | None = None,
            color_by_error: bool = False,
            cmd_diagnostics: bool = False,
            color_mag_cov: np.ndarray | None = None,
            draw_error_bars: bool = True) -> None:
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

        color_mag_cov
            ``cov(colour, magnitude)`` per star for the isochrone fit.

        draw_error_bars
            Draw the per-star error bars. The errors are used by the
            isochrone fit either way.
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
        self.is_cluster_member = is_cluster_member
        self.color_by_error = color_by_error
        self.cmd_diagnostics = cmd_diagnostics
        self.color_mag_cov = color_mag_cov
        self.draw_error_bars = draw_error_bars
        self._fit_context: dict = {}
        self._fit_result: dict | None = None

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
        cmd_dir = str(results_dir(self.output_dir, "cmds"))
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

    def _cmd_file_stub(self) -> str:
        cluster = ""
        if self.name_of_star_cluster not in ("", "?"):
            cluster = "_" + str(self.name_of_star_cluster).replace(" ", "_")
        return f"{self.file_name}{cluster}_{self.filter_2}_{self.color}"

    def write_isochrone_fit_diagnostics(
        self,
        *,
        mag: np.ndarray,
        color: np.ndarray,
        iso_array: np.ndarray,
        best_age: float,
        best_age_unit: str | None,
        chi_square: float,
        n_fiducials: int,
        n_isochrones: int,
        e_b_v: float,
        rv: float,
        m_m: float,
        apply_corrections_to: str,
        isochrone_set: str | None,
        isochrones_path: str,
        n_members: int | None,
        n_field: int | None,
        plot_tag: str,
        fit_result: dict | None = None,
        residual_plot: bool = True,
    ) -> None:
        """Write the fit summary ECSV; with ``residual_plot`` also a residual CMD.

        ``fit_result`` (from :meth:`plot_absolute_cmd`) adds the age interval
        (``age_err_lo`` / ``age_err_hi``, from ``Δχ² = s``), ``chi_square_nu``,
        ``dof``, the Birge scale ``s`` and the fitted offsets ``delta_m`` /
        ``delta_e`` with the resulting ``m_m_fit`` / ``e_b_v_fit``.
        """
        fit_result = fit_result or {}
        interval = fit_result.get("interval")
        dm = float(fit_result.get("delta_m", 0.0))
        de = float(fit_result.get("delta_e", 0.0))
        ctx = getattr(self, "_fit_context", None) or {}
        iso_array = np.asarray(iso_array, dtype=float).copy()
        iso_array[:, 0] += dm + de * float(ctx.get("a_per_e", 0.0))
        iso_array[:, 1] += de * float(ctx.get("c_per_e", 0.0))
        tree = KDTree(iso_array, leafsize=100)
        points = np.column_stack(
            (np.asarray(mag, dtype=float), np.asarray(color, dtype=float))
        )
        _, idx = tree.query(points, k=1)
        idx = np.asarray(idx, dtype=int)
        dmag = points[:, 0] - iso_array[idx, 0]
        dcolor = points[:, 1] - iso_array[idx, 1]
        stub = self._cmd_file_stub()
        unit = "" if not best_age_unit else str(best_age_unit)
        row = {
            "color": self.color,
            "filter_2": self.filter_2,
            "plot_tag": plot_tag,
            "isochrone_set": "" if isochrone_set is None else str(isochrone_set),
            "isochrones": str(isochrones_path),
            "best_age": float(best_age),
            "age_unit": unit,
            "chi_square": float(chi_square),
            "n_fiducials": int(n_fiducials),
            "n_isochrones": int(n_isochrones),
            "n_members": -1 if n_members is None else int(n_members),
            "n_field": -1 if n_field is None else int(n_field),
            "e_b_v": float(e_b_v),
            "rv": float(rv),
            "m_m": float(m_m),
            "apply_corrections_to": str(apply_corrections_to),
            "median_d_color": float(np.nanmedian(dcolor)),
            "median_d_mag": float(np.nanmedian(dmag)),
            "age_err_lo": np.nan if interval is None else float(interval.age_err_lo),
            "age_err_hi": np.nan if interval is None else float(interval.age_err_hi),
            "chi_square_nu": np.nan if interval is None else float(interval.chi_square_nu),
            "dof": -1 if interval is None else int(interval.dof),
            "chi_square_scale": np.nan if interval is None else float(interval.scale),
            "delta_m": dm,
            "delta_e": de,
            "sigma_m_prior": float(ctx.get("sigma_m", 0.0)),
            "sigma_e_prior": float(ctx.get("sigma_e", 0.0)),
            "m_m_fit": float(m_m) + dm,
            "e_b_v_fit": float(e_b_v) + de,
        }
        fit_path = tables_dir(self.output_dir) / f"cmd_isochrone_fit_{stub}.ecsv"
        Table(rows=[row]).write(str(fit_path), format="ascii.ecsv", overwrite=True)
        terminal_output.print_to_terminal(f"Save isochrone fit table: {fit_path}")
        if not residual_plot:
            return

        fig = initialize_plot("", "")
        ax = fig.add_subplot(1, 1, 1)
        ax.axvline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.scatter(dcolor, mag, s=10, c="darkred", alpha=0.45, rasterized=True)
        mag_lo = float(np.nanmin(mag))
        mag_hi = float(np.nanmax(mag))
        ax.set_ylim(mag_hi + 0.3, mag_lo - 0.3)
        mk_ticks_labels(
            rf"${self.filter_2}$ [mag]",
            rf"$\Delta({self.color})$ [mag]",
            ax,
        )
        ax.set_title(
            rf"Residual vs best isochrone ({best_age:g} {unit})",
            fontsize=9,
        )
        diag = diagnostics_dir(self.output_dir, "cmds")
        res_path = diag / f"{stub}_residual.{self.file_type}"
        fig.savefig(str(res_path), format=self.file_type, bbox_inches="tight")
        plt.close(fig)
        terminal_output.print_to_terminal(f"Save CMD residual plot: {res_path}")

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

    def _annotation_fit_terms(self, m_m: float, e_b_v: float) -> dict:
        result = getattr(self, "_fit_result", None) or {}
        interval = result.get("interval")
        if interval is None:
            return {}
        terms = {
            "age_err_lo": interval.age_err_lo,
            "age_err_hi": interval.age_err_hi,
            "chi_square_nu": interval.chi_square_nu,
        }
        if result.get("n_free"):
            terms["m_m_fit"] = float(m_m) + float(result.get("delta_m", 0.0))
            terms["e_b_v_fit"] = float(e_b_v) + float(result.get("delta_e", 0.0))
        return terms

    def _score_isochrone(
        self,
        isochrone_array: np.ndarray,
        *,
        free: bool = False,
        data_scale: float = 1.0,
    ) -> tuple[float, float, float, float, float]:
        """χ² of one isochrone against the fiducials set by :meth:`plot_absolute_cmd`.

        ``free=False``: offsets fixed at the input distance / reddening (first
        pass, gives the χ²_ν scale). ``free=True``: δM / δE profiled with
        their priors and the data χ² divided by ``data_scale``
        (:func:`~ost_photometry.analyze.isochrone_fit.profile_chi_square`).
        Returns ``(χ²_mag, χ²_colour, χ², δM, δE)``.
        """
        ctx = self._fit_context
        res = profile_chi_square(
            ctx["fiducials"],
            isochrone_array[:, 0],
            isochrone_array[:, 1],
            a_per_e=ctx["a_per_e"],
            c_per_e=ctx["c_per_e"],
            sigma_m=ctx["sigma_m"] if free else None,
            sigma_e=ctx["sigma_e"] if free else None,
            data_scale=data_scale,
        )
        return (res.chi_square_mag, res.chi_square_color, res.chi_square,
                res.delta_m, res.delta_e)

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
        _scatter_cmd_stars(
            ax0,
            self.magnitude_color,
            self.magnitude_filter_2,
            members=self.is_cluster_member,
            mag_err=self.magnitude_filter_2_err,
            color_err=self.magnitude_color_err,
            color_by_error=self.color_by_error,
            draw_error_bars=self.draw_error_bars,
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
            rv_err: float | None = None,
            m_m_err: float | None = None,
            fit_isochrone: bool = False,
            magnitude_fit_range: tuple[float | None, float | None] = (None, None),
            n_bin_observation: int = 40,
            fiduciary_points_observation: bool | None = None,
            fiduciary_points_isochrones: bool = False,
            chi_square_plot_mode: str | None = None,
            apply_corrections_to: str = "observation",
            isochrone_set: str | None = None,
            feh: float | None = None,
            z: float | None = None,
            y: float | None = None,
            alpha_fe: float | None = None) -> None:
        """
        Plot a CMD with reddening and distance applied either to the stars
        (absolute magnitudes) or to the isochrones (apparent CMD overlay).

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

        m_m_err                     : `float` or `None`, optional
            1-sigma uncertainty on the distance modulus. Propagated into the
            magnitude axis only (distance does not change colour) and combined
            in quadrature with the reddening and photometric errors.
            Default is ``None``.

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
            Possibilities: 1. simple   -> Combined chi square values shown
                                          below the CMD.
                           2. detailed -> Chi square values split according
                                          to X and Y contributions. Plots are
                                          on top and on the right hand side of
                                          the CMD
            If `None` and fit_isochrone is `True` chi_square_plot_mode is set
            to `simple`.
            Default is ``None``.

        apply_corrections_to        : `string`, optional
            ``observation`` (default): subtract reddening and distance from
            the stars (absolute CMD). ``isochrone``: add the same terms to
            the theoretical isochrones so they overlay the apparent data.
            Default is ``observation``.

        isochrone_set               : `string` or `None`, optional
            Model / grid name for the CMD info box. Omitted if unset.

        feh, z, y, alpha_fe         : `float` or `None`, optional
            Optional composition values for the CMD info box. Missing
            values are omitted.
        """
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
        (dmag_obs, dcolor_obs, dmag_iso, dcolor_iso,
         _correction_mag_err, _correction_color_err) = cmd_correction_offsets(
            a_filter_2,
            relative_extinction,
            m_m,
            apply_to=apply_corrections_to,
            a_filter_2_err=a_filter_2_err,
            relative_extinction_err=relative_extinction_err,
            m_m_err=m_m_err,
        )
        apply_to_iso = str(apply_corrections_to).strip().lower() in (
            "isochrone",
            "isochrones",
        )
        magnitude_filter_2 = self.magnitude_filter_2 + dmag_obs
        magnitude_color = self.magnitude_color + dcolor_obs
        self.magnitude_filter_2_absolute = magnitude_filter_2
        self.magnitude_color_absolute = magnitude_color
        # Per-star errors stay photometric: distance and reddening are one
        # offset for all stars and enter the fit as nuisance parameters.
        magnitude_filter_2_err = self.magnitude_filter_2_err
        magnitude_color_err = self.magnitude_color_err
        # nuisance offsets: δE (colour excess) and δM (distance modulus and
        # the R_V part of the extinction), Gaussian priors from the inputs
        a_per_e, c_per_e, _, _ = reddening_for_absolute_cmd(
            self.filter_1, self.filter_2, rv, 1.0
        )
        a_rv_err, c_rv_err = 0.0, 0.0
        if rv_err:
            _, _, a_rv_err, c_rv_err = reddening_for_absolute_cmd(
                self.filter_1, self.filter_2, rv, e_b_v, rv_err=rv_err
            )
        sigma_e = float(np.hypot(e_b_v_err or 0.0, c_rv_err / c_per_e if c_per_e else 0.0))
        sigma_m = float(np.hypot(m_m_err or 0.0, a_rv_err))

        def _shift_isochrone(mag, color):
            return (
                np.asarray(mag, dtype=float) + dmag_iso,
                np.asarray(color, dtype=float) + dcolor_iso,
            )

        #   Plot fiduciary points if isochrone fit is performed
        if fiduciary_points_observation is None and fit_isochrone:
            fiduciary_points_observation = True
        #   Plot chi square deviation of isochrones from fiduciary points if fit is
        #   performed
        if fit_isochrone and chi_square_plot_mode is None:
            chi_square_plot_mode = 'simple'

        have_isochrones = isochrones not in ("", "?", None)
        if fit_isochrone and not have_isochrones:
            terminal_output.print_to_terminal(
                "fit_isochrone=True, but no isochrone file or directory is set "
                "(YAML missing, unreadable, or isochrones: empty). "
                "Plotting the CMD without a χ² fit.",
                style_name="WARNING",
            )
            fit_isochrone = False
        draw_chi2 = bool(
            have_isochrones
            and fit_isochrone
            and chi_square_plot_mode in ("simple", "detailed")
        )

        #   Initialize plot and check plot dimensions
        fig = initialize_plot(
            figure_size_x,
            figure_size_y,
        )
        ax0, ax1, ax2 = _overlay_cmd_axes(
            fig,
            chi_square_plot_mode=chi_square_plot_mode,
            draw_chi2=draw_chi2,
        )

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
        _scatter_cmd_stars(
            ax0,
            magnitude_color,
            magnitude_filter_2,
            members=self.is_cluster_member,
            mag_err=magnitude_filter_2_err,
            color_err=magnitude_color_err,
            color_by_error=self.color_by_error,
            draw_error_bars=self.draw_error_bars,
            marker_alpha=0.35,
        )

        #   Bin observation (cluster members only when a membership flag exists)
        fit_members = None
        if (
            self.is_cluster_member is not None
            and np.asarray(self.is_cluster_member).size == magnitude_filter_2.size
        ):
            fit_members = np.asarray(self.is_cluster_member, dtype=bool)
            if not np.any(fit_members):
                terminal_output.print_to_terminal(
                    "is_cluster_member is all False; fitting all stars",
                    style_name="WARNING",
                )
                fit_members = None
            else:
                terminal_output.print_to_terminal(
                    f"Isochrone fit uses {int(np.count_nonzero(fit_members))} "
                    "cluster member(s); field stars are plotted only",
                )
        mag_fit = (
            magnitude_filter_2 if fit_members is None
            else magnitude_filter_2[fit_members]
        )
        color_fit = (
            magnitude_color if fit_members is None
            else magnitude_color[fit_members]
        )
        mag_err_fit = magnitude_filter_2_err
        color_err_fit = magnitude_color_err
        if fit_members is not None:
            if mag_err_fit is not None:
                mag_err_fit = np.broadcast_to(
                    np.asarray(mag_err_fit, dtype=float),
                    magnitude_filter_2.shape,
                )[fit_members]
            if color_err_fit is not None:
                color_err_fit = np.broadcast_to(
                    np.asarray(color_err_fit, dtype=float),
                    magnitude_color.shape,
                )[fit_members]

        if fiduciary_points_observation or fit_isochrone:
            #   Check if fit range is defined. If not, minimum and maximum
            #   values of the data are used.
            if magnitude_fit_range[0] is None:
                min_magnitude_filter_2 = np.min(mag_fit)
            else:
                min_magnitude_filter_2 = magnitude_fit_range[0]
            if magnitude_fit_range[1] is None:
                max_magnitude_filter_2 = np.max(mag_fit)
            else:
                max_magnitude_filter_2 = magnitude_fit_range[1]

            #   Define bins
            bins = np.linspace(
                min_magnitude_filter_2,
                max_magnitude_filter_2,
                n_bin_observation,
            )
            cov_fit = self.color_mag_cov
            if cov_fit is not None and fit_members is not None:
                cov_fit = np.asarray(cov_fit, dtype=float)[fit_members]
            fiducials = bin_fiducials(
                mag_fit,
                color_fit,
                mag_err=mag_err_fit,
                color_err=color_err_fit,
                color_mag_cov=cov_fit,
                bins=bins,
            )
            if len(fiducials) == 0:
                raise ValueError(
                    "No CMD points left in the magnitude fit range."
                )
            self._fit_context = {
                "fiducials": fiducials,
                "a_per_e": float(a_per_e),
                "c_per_e": float(c_per_e),
                "sigma_m": sigma_m,
                "sigma_e": sigma_e,
            }
            magnitude_binned_array = np.column_stack(
                (fiducials.mag, fiducials.color)
            )

            if fiduciary_points_observation:
                ax0.errorbar(
                    fiducials.color,
                    fiducials.mag,
                    xerr=fiducials.color_err,
                    yerr=fiducials.mag_err,
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
            fiducials = None

        #   Plot isochrones
        #
        age_list: list = []
        chi_square_list: list = []
        chi_square_magnitude_2_list: list = []
        chi_square_color_list: list = []
        isochrones_list: list = []
        age_unit = "Gyr"
        color_pick = None

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
                if not isochrones.is_dir():
                    raise FileNotFoundError(
                        f"Isochrone directory not found: {isochrones}. "
                        "The YAML loaded, but the grid path under 'isochrones' "
                        "is missing (typical: ~/isochrone_database/...)."
                    )

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

                    isochrone_magnitude_2, isochrone_color = _shift_isochrone(
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
                        chi_square_magnitude_2, chi_square_color, chi_square_total, _dm, _de = self._score_isochrone(isochrone_array)
                        chi_square_magnitude_2_list.append(
                            chi_square_magnitude_2
                        )
                        chi_square_color_list.append(chi_square_color)
                        chi_square_list.append(chi_square_total)

            #   OPTION II: Isochrone file containing many individual isochrones
            if isochrone_type == 'file':
                #   Resolve iso path
                isochrones = Path(isochrones).expanduser()
                if not isochrones.is_file():
                    raise FileNotFoundError(
                        f"Isochrone data file not found: {isochrones}. "
                        "The YAML loaded, but the grid file under 'isochrones' "
                        "is missing (typical: ~/isochrone_database/parsec_iso/...)."
                    )

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
                    mag_iso, color_iso = _shift_isochrone(
                        isochrone_magnitude_2,
                        isochrone_color,
                    )
                    isochrone_array = np.column_stack((mag_iso, color_iso))
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
                        chi_square_magnitude_2, chi_square_color, chi_square_total, _dm, _de = self._score_isochrone(isochrones_list[i])
                        chi_square_magnitude_2_list.append(
                            chi_square_magnitude_2
                        )
                        chi_square_color_list.append(chi_square_color)
                        chi_square_list.append(chi_square_total)

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

        best_age = None
        best_age_unit = None
        best_chi_square = None
        min_chi_square_id = None
        if fit_isochrone:
            if not chi_square_list:
                terminal_output.print_to_terminal(
                    "Isochrone fit requested, but no isochrones were scored "
                    "(missing grid files, empty catalog, or no overlapping CMD points).",
                    style_name="WARNING",
                )
            else:
                #   Evaluate chi square. Pass 1 (in the loop): offsets fixed
                #   at the inputs -> χ²_ν scale s. Pass 2: δM / δE profiled
                #   with their priors on the data χ² divided by s.
                ctx = self._fit_context
                n_fid = len(ctx["fiducials"])
                n_free = int(bool(ctx["sigma_m"])) + int(bool(ctx["sigma_e"]))
                chi_nu_fixed = float(np.nanmin(chi_square_list)) / max(n_fid - 1, 1)
                scale = max(1.0, chi_nu_fixed)
                offsets = [(0.0, 0.0)] * len(chi_square_list)
                if n_free:
                    rescored = [
                        self._score_isochrone(iso, free=True, data_scale=scale)
                        for iso in isochrones_list[: len(chi_square_list)]
                    ]
                    chi_square_magnitude_2_list = [r[0] for r in rescored]
                    chi_square_color_list = [r[1] for r in rescored]
                    chi_square_list = [r[2] for r in rescored]
                    offsets = [(r[3], r[4]) for r in rescored]
                numeric_ages = len(age_list) == len(chi_square_list) and all(
                    isinstance(a, (int, float)) for a in age_list
                )
                if numeric_ages:
                    interval = age_interval(
                        np.asarray(age_list, dtype=float),
                        np.asarray(chi_square_list, dtype=float),
                        n_fiducials=n_fid,
                        n_free=n_free,
                        scale=1.0 if n_free else None,
                    )
                    if n_free:
                        interval.scale = scale
                        interval.chi_square_nu = chi_nu_fixed
                        interval.dof = max(n_fid - 1, 1)
                    min_chi_square_id = interval.index
                    best_age = interval.best_age
                else:
                    interval = None
                    min_chi_square_id = int(np.argmin(chi_square_list))
                    best_age = age_list[min_chi_square_id]
                best_age_unit = age_unit
                best_chi_square = chi_square_list[min_chi_square_id]
                best_offsets = offsets[min_chi_square_id]
                self._fit_result = {
                    "interval": interval,
                    "delta_m": best_offsets[0],
                    "delta_e": best_offsets[1],
                    "n_free": n_free,
                }
                age_note = ""
                if interval is not None:
                    age_note = (
                        f" (+{interval.age_err_hi:.2f} / -{interval.age_err_lo:.2f}), "
                        f"χ²_ν = {interval.chi_square_nu:.2f} (ν = {interval.dof})"
                    )
                terminal_output.print_to_terminal(
                    f'Best fitting isochrone: {best_age:.2f} {age_unit}{age_note}; '
                    f'offsets δM = {best_offsets[0]:+.3f}, '
                    f'δE = {best_offsets[1]:+.3f} mag',
                    style_name="GOOD",
                )

                #   Plot best isochrone with the fitted offsets
                best_iso = isochrones_list[min_chi_square_id]
                ax0.plot(
                    best_iso[:, 1] + best_offsets[1] * ctx["c_per_e"],
                    best_iso[:, 0] + best_offsets[0] + best_offsets[1] * ctx["a_per_e"],
                    linestyle='-',
                    color=color_pick.to_rgba(min_chi_square_id),
                    linewidth=2,
                )

                #   χ² panels: every isochrone, the best one highlighted
                ages_plot = age_list if numeric_ages else list(range(len(chi_square_list)))
                for i, age_i in enumerate(ages_plot):
                    alpha_i = 1.0 if i == min_chi_square_id else 0.2
                    if chi_square_plot_mode == 'detailed':
                        ax1.scatter(chi_square_magnitude_2_list[i], age_i,
                                    color=color_pick.to_rgba(i), marker='o', alpha=alpha_i)
                        ax2.scatter(age_i, chi_square_color_list[i],
                                    color=color_pick.to_rgba(i), marker='o', alpha=alpha_i)
                    elif chi_square_plot_mode == 'simple':
                        ax2.scatter(age_i, chi_square_list[i],
                                    color=color_pick.to_rgba(i), marker='o', alpha=alpha_i)
                if chi_square_plot_mode == 'detailed':
                    mk_ticks_labels(f'Age [{age_unit}]', r'$\chi^2$ ', ax1)
                    mk_ticks_labels(r'$\chi^2$ ', f'Age [{age_unit}]', ax2)
                elif chi_square_plot_mode == 'simple':
                    mk_ticks_labels(r'$\chi^2$ ', f'Age [{age_unit}]', ax2)

        #   Set ticks and labels for CMD
        mk_ticks_labels(
            rf'${self.filter_2}$ [mag]',
            rf'${self.color}$ [mag]',
            ax0,
        )
        _draw_reddening_vector(
            ax0,
            e_color=relative_extinction,
            a_mag=a_filter_2,
            color_name=self.color,
            filter_2=self.filter_2,
        )

        annotation = format_isochrone_annotation(
            isochrone_set=isochrone_set,
            feh=feh,
            z=z,
            y=y,
            alpha_fe=alpha_fe,
            e_b_v=e_b_v,
            rv=rv,
            m_m=m_m,
            m_m_err=m_m_err,
            apply_corrections_to=apply_corrections_to,
            best_age=best_age,
            best_age_unit=best_age_unit,
            chi_square=best_chi_square,
            **self._annotation_fit_terms(m_m, e_b_v),
        )
        if annotation:
            ax0.text(
                0.03,
                0.03,
                annotation,
                transform=ax0.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                linespacing=1.35,
                zorder=100,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.85,
                },
            )

        #   Write plot to disk
        plot_tag = "apparent_iso" if apply_to_iso else "absolut"
        self.write_cmd(plot_tag)
        if fit_isochrone and best_age is not None:
            iso_best = (
                isochrones_list[min_chi_square_id]
                if min_chi_square_id is not None and isochrones_list
                else None
            )
            if iso_best is not None and mag_fit.size:
                n_fid = (
                    0
                    if magnitude_binned_array is None
                    else int(np.shape(magnitude_binned_array)[0])
                )
                n_iso = len(isochrones_list)
                n_mem = (
                    None
                    if fit_members is None
                    else int(np.count_nonzero(fit_members))
                )
                n_fld = (
                    None
                    if fit_members is None
                    else int(np.count_nonzero(~fit_members))
                )
                self.write_isochrone_fit_diagnostics(
                    mag=mag_fit,
                    color=color_fit,
                    iso_array=iso_best,
                    best_age=float(best_age),
                    best_age_unit=best_age_unit,
                    chi_square=float(best_chi_square),
                    n_fiducials=n_fid,
                    n_isochrones=n_iso,
                    e_b_v=e_b_v,
                    rv=rv,
                    m_m=m_m,
                    apply_corrections_to=apply_corrections_to,
                    isochrone_set=isochrone_set,
                    isochrones_path=str(isochrones),
                    n_members=n_mem,
                    n_field=n_fld,
                    plot_tag=plot_tag,
                    fit_result=getattr(self, "_fit_result", None),
                    residual_plot=self.cmd_diagnostics,
                )
        plt.close()


