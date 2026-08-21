"""Defaults for variable-star CMD plotting scripts."""

from __future__ import annotations

import sys

from ... import terminal_output


def check_variable_apparent_cmd(
        filename: str, filetype: str) -> tuple[str, str]:
    """
    Check variables and set defaults for CMDs and isochrone plots

    Parameters
    ----------
    filename
        Specified file name - can also be empty -> set default

    filetype
        Specified file type - can also be empty -> set default

    Returns
    -------
    filename
        See above

    filetype
        See above
    """
    #   Set figure type
    if filename == "?" or filename == "":
        terminal_output.print_to_terminal(
            '[Warning] No filename given, us default (cmd)',
            indent=1,
            style_name='WARNING',
        )
        filename = 'cmd'

    if filetype == '?' or filetype == '':
        terminal_output.print_to_terminal(
            '[Warning] No filetype given, use default (pdf)',
            indent=1,
            style_name='WARNING',
        )
        filetype = 'pdf'

    #   Check if file type is valid and set default
    filetype_list = ['pdf', 'png', 'eps', 'ps', 'svg']
    if filetype not in filetype_list:
        terminal_output.print_to_terminal(
            '[Warning] Unknown filetype given, use default instead (pdf)',
            indent=1,
            style_name='WARNING',
        )
        filetype = 'pdf'

    return filename, filetype


def check_variable_absolute_cmd(
        filter_list: list[str], iso_column_type: dict[str, str],
        iso_column: dict[str, str]) -> None:
    """
    Check variables and set defaults for CMDs and isochrone plots

    Parameters
    ----------
    filter_list
        Filter list

    iso_column_type
        Keys = filter - Values = type

    iso_column
        Keys = filter - Values = column
    """
    #   Check if the column declaration for the isochrones fits to the
    #   specified filter
    for filter_ in filter_list:
        if filter_ not in iso_column_type.keys():
            terminal_output.print_to_terminal(
                f"[Error] No entry for filter {filter_} specified in "
                f"'ISOcolumntype'",
                indent=1,
                style_name='FAIL',
            )
            sys.exit()
        if filter_ not in iso_column.keys():
            terminal_output.print_to_terminal(
                f"[Error] No entry for filter {filter_} specified in"
                " 'ISOcolumn'",
                indent=1,
                style_name='FAIL',
            )
            sys.exit()


__all__ = [
    "check_variable_absolute_cmd",
    "check_variable_apparent_cmd",
]
