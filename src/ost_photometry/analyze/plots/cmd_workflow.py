"""Run apparent/absolute CMDs from a photometry table (course-script entry point)."""

from __future__ import annotations

from collections.abc import Sequence

from astropy.table import Table

from ... import terminal_output
from ..cmd_prepare import (
    cmd_series_from_table,
    distance_modulus,
    load_isochrone_config,
    mask_cmd_series,
)
from ..utils.cmd_defaults import check_variable_apparent_cmd
from .cmds import MakeCMDs

_RangePair = tuple[object, object]


def _range_at(ranges: Sequence[_RangePair] | None, index: int) -> _RangePair:
    if not ranges or index >= len(ranges):
        return ("", "")
    pair = ranges[index]
    if len(pair) < 2:
        return ("", "")
    return pair[0], pair[1]


def plot_cmds_from_table(
    tbl: Table,
    filter_color_combinations: Sequence[str],
    *,
    name_of_star_cluster: str,
    file_name: str,
    file_type: str,
    output_dir: str,
    e_b_v: float,
    rv: float = 3.1,
    m_m: str | float | None = None,
    distance: str | float | None = None,
    cali: dict[str, float] | None = None,
    do_error_bars: bool = False,
    magnitude_transformation: bool | None = None,
    e_b_v_err: float | None = None,
    rv_err: float | None = None,
    m_m_err: float | None = None,
    figure_size_x: str | float = "?",
    figure_size_y: str | float = "?",
    x_plot_range_apparent: Sequence[_RangePair] | None = None,
    y_plot_range_apparent: Sequence[_RangePair] | None = None,
    x_plot_range_absolute: Sequence[_RangePair] | None = None,
    y_plot_range_absolute: Sequence[_RangePair] | None = None,
    isochrone_configuration_file: str = "",
    fit_isochrone: bool = False,
    magnitude_fit_range: tuple[float | None, float | None] = (None, None),
    chi_square_plot_mode: str | None = None,
    n_bin_observation: int = 40,
    apply_corrections_to: str = "observation",
    max_photometric_err: float | None = None,
    color_by_error: bool = False,
    cmd_diagnostics: bool = False,
) -> None:
    """Plot apparent CMDs and, when a distance modulus is set, a second CMD
    with reddening/distance on the stars or on the isochrones.

    With ``apply_corrections_to="isochrone"`` the overlay uses the apparent
    plot ranges; otherwise the absolute ranges. Non-finite magnitudes are
    dropped; ``max_photometric_err`` optionally rejects large photometric σ.
    ``is_cluster_member`` in the table (if present) greys out field stars and
    restricts the isochrone fit to members. ``color_by_error`` colours members
    by photometric σ. ``cmd_diagnostics`` writes a fit ECSV and residual CMD.
    """
    if len(tbl) == 0:
        raise ValueError("CMD table is empty.")

    file_name, file_type = check_variable_apparent_cmd(file_name, file_type)
    mu = distance_modulus(m_m, distance)

    for filter_id, color_name in enumerate(filter_color_combinations):
        parts = color_name.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Colour combination must be 'F1-F2', got {color_name!r}"
            )
        filter_1, filter_2 = parts[0], parts[1]
        series = cmd_series_from_table(
            tbl,
            filter_1,
            filter_2,
            do_error_bars=do_error_bars,
            magnitude_transformation=magnitude_transformation,
            cali=cali,
        )
        n_in = series.color.size
        series = mask_cmd_series(
            series, max_photometric_err=max_photometric_err
        )
        n_dropped = n_in - series.color.size
        if n_dropped:
            terminal_output.print_to_terminal(
                f"Dropped {n_dropped} CMD point(s) (non-finite or error cut)",
            )
        if series.color.size == 0:
            raise ValueError(
                "No finite CMD points left after quality cuts "
                f"for {color_name}."
            )

        terminal_output.print_to_terminal(
            f"Create apparent CMD: {filter_2} vs. {filter_1}-{filter_2}",
        )
        cmds = MakeCMDs(
            name_of_star_cluster,
            file_name,
            file_type,
            filter_2,
            filter_1,
            series.color,
            series.magnitude_filter_2,
            color_err=series.color_err,
            magnitude_filter_2_err=series.magnitude_filter_2_err,
            output_dir=output_dir,
            is_cluster_member=series.is_cluster_member,
            color_by_error=color_by_error,
            cmd_diagnostics=cmd_diagnostics,
        )
        x_app = _range_at(x_plot_range_apparent, filter_id)
        y_app = _range_at(y_plot_range_apparent, filter_id)
        cmds.plot_apparent_cmd(
            figure_size_x=figure_size_x,
            figure_size_y=figure_size_y,
            y_plot_range_max=y_app[1],
            y_plot_range_min=y_app[0],
            x_plot_range_max=x_app[1],
            x_plot_range_min=x_app[0],
        )

        if mu == 0.0:
            continue

        iso_on_data = str(apply_corrections_to).strip().lower() in (
            "isochrone",
            "isochrones",
        )
        terminal_output.print_to_terminal(
            f"Create {'apparent (reddened isochrones)' if iso_on_data else 'absolute'}"
            f" CMD: {filter_2} vs. {filter_1}-{filter_2}",
        )
        iso = load_isochrone_config(isochrone_configuration_file, [filter_1, filter_2])
        if iso_on_data:
            x_overlay, y_overlay = x_app, y_app
        else:
            x_overlay = _range_at(x_plot_range_absolute, filter_id)
            y_overlay = _range_at(y_plot_range_absolute, filter_id)
        cmds.plot_absolute_cmd(
            e_b_v,
            mu,
            iso.isochrones,
            iso.isochrone_type,
            iso.isochrone_column_type,
            iso.isochrone_column,
            iso.isochrone_log_age,
            iso.isochrone_keyword,
            iso.isochrone_legend,
            rv=rv,
            e_b_v_err=e_b_v_err if do_error_bars else None,
            rv_err=rv_err if do_error_bars else None,
            m_m_err=m_m_err if do_error_bars else None,
            figure_size_x=figure_size_x,
            figure_size_y=figure_size_y,
            y_plot_range_max=y_overlay[1],
            y_plot_range_min=y_overlay[0],
            x_plot_range_max=x_overlay[1],
            x_plot_range_min=x_overlay[0],
            fit_isochrone=fit_isochrone,
            n_bin_observation=n_bin_observation,
            magnitude_fit_range=magnitude_fit_range,
            chi_square_plot_mode=chi_square_plot_mode,
            apply_corrections_to=apply_corrections_to,
            isochrone_set=iso.isochrone_set,
            feh=iso.feh,
            z=iso.z,
            y=iso.y,
            alpha_fe=iso.alpha_fe,
        )


__all__ = ["plot_cmds_from_table"]
