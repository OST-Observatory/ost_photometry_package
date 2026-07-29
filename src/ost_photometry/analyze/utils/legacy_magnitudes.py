"""Legacy wide magnitude tables and ASCII export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table

from ... import calibration_parameters, checks, terminal_output
from ..post_processing.adapters import ensure_epoch_native_photometry_table
from ..post_processing.io import write_epoch_native_magnitudes
from ..post_processing.light_curve import attach_observation_jd_column
from ..post_processing.schema import ascii_write_formats_for_columns

if TYPE_CHECKING:
    from ..observation import Observation

def mk_magnitudes_table(
        observation: 'analyze.Observation', filter_list: list[str]
        ) -> Table:
    """
    Create and export astropy table with object positions and magnitudes

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        Filter

    Returns
    -------
    tbl
        Table with CMD data
    """
    #   Get object indices, X & Y pixel positions and wcs
    #   Assumes that the image series are already correlated
    image_wcs = observation.image_series_dict[filter_list[0]].wcs
    index_objects = observation.image_series_dict[filter_list[0]].image_list[0].photometry['id']
    x_positions = observation.image_series_dict[filter_list[0]].image_list[0].photometry['x_fit']
    y_positions = observation.image_series_dict[filter_list[0]].image_list[0].photometry['y_fit']

    # Make CMD table
    tbl = Table(
        names=['i', 'x', 'y', ],
        data=[
            np.intc(index_objects),
            x_positions,
            y_positions,
        ]
    )

    #   Convert Pixel to sky coordinates
    sky = image_wcs.pixel_to_world(x_positions, y_positions)

    #   Add sky coordinates to table
    tbl['ra (deg)'] = sky.ra
    tbl['dec (deg)'] = sky.dec

    #   Add magnitude columns to table
    for filter_ in filter_list:
        #   Get image list
        image_series = observation.image_series_dict[filter_]
        image_list = image_series.image_list

        for image_id, image in enumerate(image_list):
            photometry_table = image.photometry
            for photometry_column_keyword in ['mag_cali_trans', 'mag_cali_no-trans']:
                try:
                    magnitudes = photometry_table[photometry_column_keyword]
                    magnitude_errors = photometry_table[
                        f'{photometry_column_keyword}_unc'
                    ]
                except KeyError:
                    magnitudes = np.ones((len(index_objects))) * 999.
                    magnitude_errors = magnitudes

                if photometry_column_keyword == 'mag_cali_no-trans':
                    column_name = f'{filter_} (simple, image={image_id})'
                    column_name_err = f'{filter_}_err (simple, image={image_id})'
                else:
                    column_name = f'{filter_} (transformed, image={image_id})'
                    column_name_err = f'{filter_}_err (transformed, image={image_id})'

                #   Add to table
                tbl.add_columns(
                    [magnitudes, magnitude_errors],
                    names=[column_name, column_name_err]
                )

            try:
                flux_fit = np.asarray(photometry_table["flux_fit"], dtype=float)
                flux_err = np.asarray(photometry_table["flux_err"], dtype=float)
            except KeyError:
                flux_fit = np.full(len(index_objects), np.nan, dtype=float)
                flux_err = np.full(len(index_objects), np.nan, dtype=float)
            tbl.add_columns(
                [
                    flux_fit,
                    flux_err,
                ],
                names=[
                    f"{filter_} (flux, image={image_id})",
                    f"{filter_}_err (flux, image={image_id})",
                ],
            )

    return tbl


def calibrated_epochs_to_legacy_wide_table(
    calibrated: Table,
    filter_list: list[str],
) -> Table:
    """
    Convert differential calibration output (vstacked by calibration epoch) to legacy format.

    Legacy format: one row per star, columns i, x, y, ra (deg), dec (deg),
    and per filter/image: {filter} (transformed, image={id}), {filter}_err (transformed, image={id}).

    Parameters
    ----------
    calibrated : Table
        Output from PhotometryCalibrator.get_calibrated_photometry (vstacked),
        after fit_transformation_parameters().
        Must have columns: id, ra, dec, x, y, epoch_id (or legacy frame_id),
        mag_cal_<filter>, err_cal_<filter>.

    filter_list : list[str]
        Filter names in order (e.g. ['B', 'V']).

    Returns
    -------
    Table
        Table in legacy format for save_magnitudes_ascii.
    """
    id_col = "epoch_id" if "epoch_id" in calibrated.colnames else "frame_id"
    unique_epochs = np.unique(calibrated[id_col])
    ref_epoch_id = unique_epochs[0]
    ref_mask = calibrated[id_col] == ref_epoch_id
    ref = calibrated[ref_mask]
    n_obj = len(ref)
    index_objects = np.asarray(ref["id"], dtype=int)
    x_positions = np.asarray(ref["x"], dtype=float)
    y_positions = np.asarray(ref["y"], dtype=float)
    ra_deg = np.asarray(ref["ra"], dtype=float)
    dec_deg = np.asarray(ref["dec"], dtype=float)

    tbl = Table(
        names=["i", "x", "y"],
        data=[index_objects, x_positions, y_positions],
    )
    tbl["ra (deg)"] = ra_deg
    tbl["dec (deg)"] = dec_deg

    for epoch_key in np.unique(calibrated[id_col]):
        sk = str(epoch_key)
        if sk.startswith("epoch_"):
            image_label = sk[len("epoch_") :]
        else:
            parts = sk.rsplit("_", 1)
            image_label = parts[1] if len(parts) > 1 else "0"
        epoch_mask = calibrated[id_col] == epoch_key
        epoch_data = calibrated[epoch_mask]
        id_to_idx = {int(ref["id"][i]): i for i in range(n_obj)}
        for filter_ in filter_list:
            mag_col = f"mag_cal_{filter_}"
            err_col = f"err_cal_{filter_}"
            if mag_col not in calibrated.colnames or err_col not in calibrated.colnames:
                continue
            mag_arr = np.full(n_obj, 999.0)
            err_arr = np.full(n_obj, 999.0)
            for row in epoch_data:
                idx = id_to_idx.get(int(row["id"]))
                if idx is not None:
                    mag_arr[idx] = row[mag_col]
                    err_arr[idx] = row[err_col]
            tbl[f"{filter_} (transformed, image={image_label})"] = mag_arr
            tbl[f"{filter_}_err (transformed, image={image_label})"] = err_arr

    return tbl


def transformation_keys_for_table_magnitudes(
    tbl: Table, filter_list: list[str],
) -> dict[str, str]:
    """
    Build ``{ 'magB': column_name, ... }`` for
    :func:`find_filter_for_magnitude_transformation`.

    Recognizes legacy wide columns ``{filter} (transformed, image=...)`` (as produced
    by :func:`calibrated_epochs_to_legacy_wide_table`) and, as fallback, raw
    differential columns ``mag_cal_<filter>`` or instrumental ``mag_inst_<filter>``.
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


def save_magnitudes_ascii(
        observation: 'analyze.Observation', tbl: Table,
        object_id: int | None = None,
        rts: str = '', photometry_extraction_method: str = '') -> None:
    """
    Save magnitudes as ASCII files

    Parameters
    ----------
    observation
        Image container object with image series objects for each
        filter

    tbl
        Table with magnitudes

    object_id
        Photometry ``id`` (row index) for filename suffix; optional.
        Default is ``None``.

    rts
        Additional string characterizing that should be included in the
        file name.
        Default is ``''``.

    photometry_extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.
    """
    #   Check output directories
    output_dir = list(observation.image_series_dict.values())[0].out_path
    checks.check_output_directories(
        output_dir,
        output_dir / 'tables',
    )

    #   Define file name specifier
    if object_id is not None:
        object_id_suffix = f'_img_{object_id}'
    else:
        object_id_suffix = ''
    if photometry_extraction_method != '':
        photometry_extraction_method = f'_{photometry_extraction_method}'

    #   Set file name
    filename = (
        f'calibrated_magnitudes{photometry_extraction_method}{object_id_suffix}{rts}.dat'
    )

    #   Combine to a path
    out_path = output_dir / 'tables' / filename

    #   Define output formats for the table columns
    #
    #   Get column names
    column_names = tbl.colnames

    #   Set default float format only for numeric columns (skip ra/dec and
    #   string columns such as epoch_id from differential calibration).
    for column_name in column_names:
        if column_name in ('ra (deg)', 'dec (deg)'):
            continue
        col = tbl[column_name]
        if not np.issubdtype(col.dtype, np.number):
            continue
        col.info.format = '{:12.3f}'

    #   Reset for id/i and x/y columns (only keys present on the table)
    formats = ascii_write_formats_for_columns(column_names)

    #   Write file
    tbl.write(
        str(out_path),
        format='ascii',
        overwrite=True,
        formats=formats,
    )






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


def save_calibration(
        observation: 'analyze.Observation', filter_list: list[str],
        object_id: int | None = None,
        photometry_extraction_method: str = '', rts: str = ''
        ) -> None:
    """
    Save calibrated magnitudes: legacy wide ``.dat`` under ``tables/`` and the same
    data as epoch-native ``.ecsv`` (``calibrated_magnitudes_<method>_<filters>.ecsv``).

    Parameters
    ----------
    observation
        Container object with image series objects for each filter

    filter_list
        Filter

    object_id
        Photometry ``id`` (row index) for filename suffix ``_img_<id>_``; ``None`` omits
        that suffix (same behaviour as :func:`save_magnitudes_ascii`).

    photometry_extraction_method
        Applied extraction method. Possibilities: ePSF or APER`
        Default is ``''``.

    rts
        Additional string characterizing that should be included in the
        file name.
        Default is ``''``.
    """
    #   Make astropy table
    table_magnitudes = mk_magnitudes_table(
        observation,
        filter_list,
    )

    #   Add table to observation container
    observation.table_magnitudes = table_magnitudes

    #   Save to file
    save_magnitudes_ascii(
        observation,
        table_magnitudes,
        object_id=object_id,
        photometry_extraction_method=photometry_extraction_method,
        rts=rts,
    )

    table_epoch_native = ensure_epoch_native_photometry_table(table_magnitudes)

    # Local import: avoid importing pipeline (orchestrator/steps) during
    # analyze.utilities module load — that cycle broke calibration import.
    from ..pipeline.bridge import build_legacy_calibration_epoch_meta

    _meta = build_legacy_calibration_epoch_meta(
        observation,
        filter_list,
        table_epoch_native,
    )
    _ref_f = filter_list[0] if filter_list else "V"
    table_epoch_native = attach_observation_jd_column(
        table_epoch_native, _meta, _ref_f
    )
    write_epoch_native_magnitudes(
        observation,
        table_epoch_native,
        object_id=object_id,
        photometry_extraction_method=photometry_extraction_method,
        rts=rts,
        file_stem="calibrated_magnitudes",
    )


__all__ = [
    "calibrated_epochs_to_legacy_wide_table",
    "find_filter_for_magnitude_transformation",
    "mk_magnitudes_table",
    "save_calibration",
    "save_magnitudes_ascii",
    "transformation_keys_for_table_magnitudes",
]
