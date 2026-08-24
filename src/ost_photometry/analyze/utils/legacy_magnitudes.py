"""Helpers for magnitude-column names (epoch-native and legacy wide input)."""

from __future__ import annotations

from astropy.table import Table

from ... import calibration_parameters, terminal_output


def transformation_keys_for_table_magnitudes(
    tbl: Table, filter_list: list[str],
) -> dict[str, str]:
    """
    Build ``{ 'magB': column_name, ... }`` for
    :func:`find_filter_for_magnitude_transformation`.

    Recognizes leftover legacy wide columns ``{filter} (transformed, image=...)``
    and, as fallback, epoch-native ``mag_cal_<filter>`` or instrumental
    ``mag_inst_<filter>``.
    """
    out: dict[str, str] = {}
    for f in filter_list:
        prefix = f"{f} (transformed,"
        for name in tbl.colnames:
            if name.startswith(prefix) and not name.startswith(f"{f}_err"):
                out[f"mag{f}"] = name
                break
        else:
            if f"mag_cal_{f}" in tbl.colnames:
                out[f"mag{f}"] = f"mag_cal_{f}"
            elif f"mag_inst_{f}" in tbl.colnames:
                out[f"mag{f}"] = f"mag_inst_{f}"
    return out


def find_filter_for_magnitude_transformation(
        filter_list: list[str], calibration_filters: dict[str, str],
        valid_filter_combinations: list[list[str]] | None = None
        ) -> tuple[set[str], list[list[str]]]:
    """
    Identifies filter that can be used for magnitude transformation

    Parameters
    ----------
    filter_list
        List with observed filter names

    calibration_filters
        Names of the available filter with calibration data

    valid_filter_combinations
        Valid filter combinations to calculate magnitude transformation
        Default is ``None``.

    Returns
    -------
    valid_filter
        Filter for which magnitude transformation is possible

    usable_filter_combinations
        Filter combinations for which magnitude transformation
        can be applied
    """
    #   Load valid filter combinations, if none are supplied
    if valid_filter_combinations is None:
        valid_filter_combinations = calibration_parameters.valid_filter_combinations_for_transformation

    #   Setup list for valid filter etc.
    valid_filter = []
    usable_filter_combinations = []

    #   Determine usable filter combinations -> Filters must be in a valid
    #   filter combination for the magnitude transformation and calibration
    #   data must be available for the filter.
    for filter_combination in valid_filter_combinations:
        if filter_combination[0] in filter_list and filter_combination[1] in filter_list:
            faulty_filter = None
            if f'mag{filter_combination[0]}' not in calibration_filters:
                faulty_filter = filter_combination[0]
            if f'mag{filter_combination[1]}' not in calibration_filters:
                faulty_filter = filter_combination[1]
            if faulty_filter is not None:
                terminal_output.print_to_terminal(
                    "Magnitude transformation not possible because "
                    "no calibration data available for filter "
                    f"{faulty_filter}",
                    indent=2,
                    style_name='WARNING',
                )
                continue

            valid_filter.append(filter_combination[0])
            valid_filter.append(filter_combination[1])
            usable_filter_combinations.append(filter_combination)
    valid_filter = set(valid_filter)

    return valid_filter, usable_filter_combinations


__all__ = [
    "find_filter_for_magnitude_transformation",
    "transformation_keys_for_table_magnitudes",
]
