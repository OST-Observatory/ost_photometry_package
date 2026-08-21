"""
Vizier catalog download for photometric calibration.

Column sets and per-catalog quirks (renames, derived mags from color indices) are
driven by :obj:`ost_photometry.calibration_parameters.catalog_properties_dict`.
This module returns **Vizier-native** column names plus a ``column_dict`` that maps
logical keys (``magB``, ``errB``, …) to those names — :mod:`fetch` then converts
to the standard ``mag_std_*`` schema via :func:`vizier_result_to_standard_table`.

The query uses a circular region: ``center`` and radius ``field_of_view_arcmin``.
"""

from __future__ import annotations

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier

from ... import calibration_parameters, terminal_output


def get_vizier_catalog(
    filter_list: list[str],
    center: SkyCoord,
    field_of_view_arcmin: float,
    catalog_identifier: str,
    magnitude_range: tuple[float, float] = (0.0, 18.5),
    cleanup_magnitudes: bool = True,
    print_infos: bool = True,
    indent: int = 2,
) -> tuple[Table, dict[str, str], u.Unit]:
    """
    Download catalog with calibration info from Vizier.

    Parameters
    ----------
    filter_list
        Filters for which ``mag{F}`` / ``err{F}`` entries are added to ``column_dict``.
        May be empty (e.g. VSX query for variable stars only needs positions).
    center
        Field center (ICRS).
    field_of_view_arcmin
        Search radius in arcminutes.
    catalog_identifier
        Vizier catalog ID, key in ``catalog_properties_dict``.
    magnitude_range
        Inclusive range on a **single preferred band** chosen by priority order below
        (V, R, B, …). Used to drop very bright/faint stars (saturation vs. noise).
    cleanup_magnitudes
        If True: apply renames, arithmetic magnitudes, range filter, fill ``column_dict``.
        If False: return first table with only ``ra``/``dec`` keys in ``column_dict``
        (used e.g. for VSX without magnitude filtering).
    print_infos
        Terminal logging.

    Returns
    -------
    table, column_dict, ra_unit
        ``ra_unit`` is taken from catalog metadata (usually degrees).
    """
    if print_infos:
        terminal_output.print_to_terminal(
            f"Downloading calibration data from Vizier: {catalog_identifier}",
            indent=indent,
        )

    catalog_properties_dict = calibration_parameters.catalog_properties_dict[
        catalog_identifier
    ]

    columns = (
        catalog_properties_dict["ra_dec_columns"]
        + catalog_properties_dict["columns"]
        + catalog_properties_dict["err_columns"]
    )

    v = Vizier(
        columns=columns,
        row_limit=int(1e6),
        catalog=catalog_identifier,
    )

    table_list = v.query_region(
        center,
        radius=field_of_view_arcmin * u.arcmin,
    )

    if not table_list:
        if print_infos:
            terminal_output.print_to_terminal(
                "No calibration data available",
                indent=indent + 1,
                style_name="WARNING",
            )
        return Table(), {}, ""

    result = table_list[0]

    column_dict = {
        "ra": catalog_properties_dict["ra_dec_columns"][0],
        "dec": catalog_properties_dict["ra_dec_columns"][1],
    }

    if cleanup_magnitudes:
        # Catalog-specific renames (e.g. SDSS rpmag -> rmag)
        if "column_rename" in catalog_properties_dict:
            for element in catalog_properties_dict["column_rename"]:
                result.rename_column(element[0], element[1])

        # e.g. UBV catalogs: reconstruct Bmag from B-V + Vmag
        if "magnitude_arithmetic" in catalog_properties_dict:
            for element in catalog_properties_dict["magnitude_arithmetic"]:
                result[element[0]] = result[element[1]] + result[element[2]]

        # One reference band for magnitude_range cut (first match wins)
        if "Vmag" in result.keys():
            preferred_filter = "Vmag"
        elif "Rmag" in result.keys():
            preferred_filter = "Rmag"
        elif "Bmag" in result.keys():
            preferred_filter = "Bmag"
        elif "Imag" in result.keys():
            preferred_filter = "Imag"
        elif "Umag" in result.keys():
            preferred_filter = "Umag"
        elif "gmag" in result.keys():
            preferred_filter = "gmag"
        elif "rmag" in result.keys():
            preferred_filter = "rmag"
        elif "imag" in result.keys():
            preferred_filter = "imag"
        elif "zmag" in result.keys():
            preferred_filter = "zmag"
        elif "umag" in result.keys():
            preferred_filter = "umag"
        else:
            if print_infos:
                terminal_output.print_to_terminal(
                    "Calibration issue: Threshold magnitude not recognized",
                    indent=indent + 1,
                    style_name="ERROR",
                )
            raise RuntimeError

        mask = (result[preferred_filter] <= magnitude_range[1]) & (
            result[preferred_filter] >= magnitude_range[0]
        )
        result = result[mask]

        if print_infos:
            terminal_output.print_to_terminal(
                f"{len(result)} calibration objects remaining after magnitude "
                "filtering",
                indent=indent,
            )

        # Map requested filters to {F}mag / e_{F}mag column names if present
        for filter_ in filter_list:
            if f"{filter_}mag" in result.colnames:
                column_dict[f"mag{filter_}"] = f"{filter_}mag"
                if f"e_{filter_}mag" in result.colnames:
                    column_dict[f"err{filter_}"] = f"e_{filter_}mag"
            else:
                if print_infos:
                    terminal_output.print_to_terminal(
                        f"No calibration data for {filter_} band",
                        indent=indent + 1,
                        style_name="WARNING",
                    )

    return result, column_dict, catalog_properties_dict["ra_unit"]
