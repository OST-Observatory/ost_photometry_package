"""Convert calibrated magnitudes to another filter system."""

from __future__ import annotations

import numpy as np
from astropy import uncertainty as unc
import astropy.units as u
from astropy.table import Table

from ... import calibration_parameters, terminal_output


def _build_epoch_native_magnitude_distributions(
    tbl: Table,
    distribution_samples: int,
) -> dict[str, unc.core.NdarrayDistribution]:
    """
    Build one normal distribution per filter from ``mag_cal_<F>`` / ``err_cal_<F>``,
    or (if no calibrated column for that filter) ``mag_inst_<F>`` / ``err_inst_<F>``.

    Filter names ``<F>`` are the suffixes after the prefix (e.g. ``V``, ``Clear``).
    """
    data_dict: dict[str, unc.core.NdarrayDistribution] = {}
    for mag_prefix, err_prefix in (
        ("mag_cal_", "err_cal_"),
        ("mag_inst_", "err_inst_"),
    ):
        for name in tbl.colnames:
            if not name.startswith(mag_prefix):
                continue
            filter_ = name[len(mag_prefix) :]
            if not filter_ or filter_ in data_dict:
                continue
            mag_col = name
            err_col = f"{err_prefix}{filter_}"
            m = np.asarray(tbl[mag_col], dtype=float)
            if err_col in tbl.colnames:
                e = np.abs(np.asarray(tbl[err_col], dtype=float))
            else:
                e = np.zeros_like(m, dtype=float)
            data_dict[filter_] = unc.normal(
                m * u.mag,
                std=e * u.mag,
                n_samples=distribution_samples,
            )
    return data_dict


def _convert_sdss_jordi(
    tbl: Table,
    data_dict: dict[str, unc.core.NdarrayDistribution],
    distribution_samples: int,
) -> None:
    """Apply Jordi et al. (2005) transformations; add ``mag_sdss_*`` / ``err_sdss_*`` columns."""
    calib_functions = calibration_parameters.filter_system_conversions["SDSS"][
        "Jordi_et_al_2005"
    ]
    dd: dict = {**data_dict, "distribution_samples": distribution_samples}
    for band in ("g", "u", "r", "i", "z"):
        result = calib_functions[band](**dd)
        if result is None:
            continue
        dd[band] = result
        med = result.pdf_median()
        std = result.pdf_std()
        tbl[f"mag_sdss_{band}"] = np.asarray(
            u.Quantity(med).to_value(u.mag), dtype=float
        )
        tbl[f"err_sdss_{band}"] = np.asarray(
            u.Quantity(std).to_value(u.mag), dtype=float
        )


def convert_magnitudes_to_other_system(
        tbl: Table, target_filter_system: str, distribution_samples=1000
        ) -> Table:
    """
    Convert epoch-native calibrated magnitudes to another photometric system.

    Expects ``mag_cal_<filter>`` / ``err_cal_<filter>``, or ``mag_inst_*`` if no calibrated
    columns exist for that filter (one magnitude per filter per row; multi-epoch supported).
    Output for SDSS is written as
    ``mag_sdss_<band>`` and ``err_sdss_<band>`` for bands ``u``, ``g``, ``r``, ``i``, ``z``
    when the Jordi et al. (2005) formulas apply given the available input filters.

    Parameters
    ----------
    tbl
        Table with ``mag_cal_*`` and/or ``mag_inst_*`` columns.

    target_filter_system
        Target system: ``SDSS``, ``AB``, or ``BESSELL`` (only ``SDSS`` is implemented).

    distribution_samples
        Number of samples used for distributions. Default is ``1000``.
    """
    if target_filter_system not in ['SDSS', 'AB', 'BESSELL']:
        terminal_output.print_to_terminal(
            f'Magnitude conversion not possible. Unfortunately, '
            f'there is currently no conversion formula for this '
            f'photometric system: {target_filter_system}.',
            style_name='WARNING',
        )
        return tbl

    data_dict = _build_epoch_native_magnitude_distributions(
        tbl, distribution_samples=distribution_samples
    )
    if not data_dict:
        terminal_output.print_to_terminal(
            "Magnitude conversion skipped: no ``mag_cal_*`` or ``mag_inst_*`` columns.",
            style_name="WARNING",
        )
        return tbl

    out = tbl.copy()

    if target_filter_system == 'AB':
        terminal_output.print_to_terminal(
            "AB system conversion is not implemented yet.",
            style_name="WARNING",
        )

    elif target_filter_system == 'SDSS':
        _convert_sdss_jordi(out, data_dict, distribution_samples)

    elif target_filter_system == 'BESSELL':
        terminal_output.print_to_terminal(
            "BESSELL system conversion is not implemented yet.",
            style_name="WARNING",
        )

    return out
