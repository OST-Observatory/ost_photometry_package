"""
Download calibration catalogs and normalize to the **standard column schema**.

Design
------
* All network / file IO for calibration lives here (and in :mod:`vizier_query`).
* Callers pass ``center`` + ``field_of_view_arcmin``; legacy code derives those from
  ``Image`` / ``ImageSeries`` (e.g. VSP uses ``1.5 * fov_x``, Vizier uses ``fov_x``).
* Output is always an :class:`~astropy.table.Table` with:

  * ``ra``, ``dec`` — ICRS, **decimal degrees**
  * ``mag_std_{f}``, ``err_std_{f}`` — per filter ``f`` (Johnson letters and/or Sloan g,r,i)
  * optional ``id``, ``apass_id``, etc.

* **APASS** is implemented entirely here (Vizier ``II/336/apass9``), not a separate module.
* **Lupton** R/I from Sloan r′, i′: :func:`transforms.add_johnson_ri_to_standard_table`
  for APASS; optional heuristic for other Vizier tables via
  :func:`_maybe_add_johnson_ri_standard`.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from ... import calibration_parameters, style, terminal_output
from .transforms import add_johnson_ri_to_standard_table
from .vizier_query import get_vizier_catalog

# Vizier catalog id for APASS DR9 — also matched when ``calibration_source`` maps here
APASS_VIZIER_ID = "II/336/apass9"


def _print(msg: str, indent: int, **kwargs) -> None:
    """Thin wrapper for consistent pipeline indentation."""
    terminal_output.print_to_terminal(msg, indent=indent, **kwargs)


def vizier_result_to_standard_table(
    raw: Table,
    column_dict: dict[str, str],
    filter_list: list[str],
    ra_unit: u.Unit,
) -> Table:
    """
    Convert :func:`get_vizier_catalog` output to the standard schema.

    Steps:
    1. Build ``SkyCoord`` from catalog RA/Dec columns using ``ra_unit`` for RA
       (degrees or hourangle per ``catalog_properties_dict``).
    2. Copy requested Johnson bands from ``column_dict`` (``magB`` → ``mag_std_B``, …).
    3. Also copy Sloan-style ``gmag``/``rmag``/``imag`` if present (lowercase ``g,r,i`` keys).
    4. Map APASS-style primed names (``g'mag``, …) if still present in ``raw``.

    Empty input returns an empty table with ``ra``, ``dec`` float columns (shape only).
    """
    if len(raw) == 0:
        return Table(names=["ra", "dec"], dtype=[float, float])

    ra_col, dec_col = column_dict["ra"], column_dict["dec"]
    sc = SkyCoord(
        ra=raw[ra_col],
        dec=raw[dec_col],
        unit=(ra_unit, u.deg),
        frame="icrs",
    )
    out = Table()
    out["ra"] = sc.ra.deg
    out["dec"] = sc.dec.deg
    for f in filter_list:
        mk, ek = f"mag{f}", f"err{f}"
        if mk in column_dict:
            out[f"mag_std_{f}"] = raw[column_dict[mk]]
        if ek in column_dict:
            out[f"err_std_{f}"] = raw[column_dict[ek]]
    # SDSS-style names after Vizier renames (gmag, rmag, …)
    for sloan, std_name in (("g", "mag_std_g"), ("r", "mag_std_r"), ("i", "mag_std_i")):
        col_mag = f"{sloan}mag"
        col_err = f"e_{sloan}mag"
        if col_mag in raw.colnames and std_name not in out.colnames:
            out[std_name] = raw[col_mag]
        if col_err in raw.colnames:
            err_std = f"err_std_{sloan}"
            if err_std not in out.colnames:
                out[err_std] = raw[col_err]
    # APASS / some tables keep primed Sloan columns in the raw table
    for ap_col, std_mag, std_err in (
        ("g'mag", "mag_std_g", "err_std_g"),
        ("r'mag", "mag_std_r", "err_std_r"),
        ("i'mag", "mag_std_i", "err_std_i"),
    ):
        if ap_col in raw.colnames and std_mag not in out.colnames:
            out[std_mag] = raw[ap_col]
        eap = f"e_{ap_col}" if f"e_{ap_col}" in raw.colnames else None
        if eap and std_err not in out.colnames:
            out[std_err] = raw[eap]
    return out


def _maybe_add_johnson_ri_standard(
    table: Table,
    *,
    catalog_identifier: str | None,
    apply_sloan_to_johnson_ri: bool,
) -> Table:
    """
    If enabled and Sloan r/i exist but Johnson R/I do not, apply Lupton transform.

    ``catalog_identifier`` is reserved for future per-catalog policy; currently unused.
    """
    if not apply_sloan_to_johnson_ri:
        return table
    if "mag_std_R" in table.colnames and "mag_std_I" in table.colnames:
        return table
    if "mag_std_r" not in table.colnames or "mag_std_i" not in table.colnames:
        return table
    return add_johnson_ri_to_standard_table(table, inplace=False)


def _fetch_vsp_standard(
    center: SkyCoord,
    field_of_view_arcmin: float,
    filter_list: list[str],
    calibration_catalog_mag_range: tuple[float, float],
    indent: int,
) -> Table:
    """
    AAVSO VSP JSON API → standard table.

    * Upper magnitude cut is sent to the API as ``maglimit`` (``range[1]``).
    * Lower cut: legacy used ``magV >= range[0]`` when V band exists.
    * RA from API is treated as **hours** and converted to degrees (×15) to match
      historical ``hourangle`` behavior; Dec assumed degrees.
    """
    _print("Downloading calibration data from www.aavso.org", indent=indent)
    filters = filter_list if filter_list else ["B", "V"]
    ra = center.ra.degree
    dec = center.dec.degree
    vsp_template = (
        'https://www.aavso.org/apps/vsp/api/chart/"'
        "?format=json&fov={}&maglimit={}&ra={}&dec={}&special=std_field"
    )
    r = requests.get(
        vsp_template.format(
            field_of_view_arcmin, calibration_catalog_mag_range[1], ra, dec
        )
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nThe request of the AAVSO website was not "
            "successful.\nProbably no calibration stars found.\n -> EXIT"
            f"{style.Bcolors.ENDC}"
        )

    obj_id: list = []
    obj_ra: list = []
    obj_dec: list = []
    n_obj = len(r.json()["photometry"])
    n_filter = len(filters)
    mags = np.zeros((n_obj, n_filter))
    errs = np.zeros((n_obj, n_filter))

    for i, star in enumerate(r.json()["photometry"]):
        obj_id.append(star["auid"])
        obj_ra.append(star["ra"])
        obj_dec.append(star["dec"])
        for j, filter_ in enumerate(filters):
            for band in star["bands"]:
                if band["band"][0] == filter_:
                    mags[i, j] = band["mag"]
                    errs[i, j] = band["error"]

    out = Table()
    out["id"] = obj_id
    # Hours → degrees for internal standard schema (ICRS deg)
    out["ra"] = np.asarray(obj_ra, dtype=float) * 15.0
    out["dec"] = np.asarray(obj_dec, dtype=float)
    for j, filter_ in enumerate(filters):
        out[f"mag_std_{filter_}"] = mags[:, j]
        out[f"err_std_{filter_}"] = errs[:, j]

    # Legacy lower bound only on V when present (typical variable-star charts)
    if "mag_std_V" in out.colnames:
        mask = out["mag_std_V"] >= calibration_catalog_mag_range[0]
        out = out[mask]
    _print(
        f"{len(out)} calibration objects remaining after magnitude filtering",
        indent=indent,
    )
    return out


def _fetch_simbad_standard(
    center: SkyCoord,
    field_of_view_arcmin: float,
    filter_list: list[str],
    calibration_catalog_mag_range: tuple[float, float],
    indent: int,
) -> Table:
    """
    Live Simbad cone search → standard table.

    * Callers pass ``field_of_view_arcmin`` as **1.5 × image FOV** (legacy); Simbad
      query radius is further scaled by **0.66** (historical behavior).
    * RA/Dec from Simbad are hourangle + deg → converted to degrees.
    * Magnitude range applied on a **preferred** band (V, R, B, …) same priority as Vizier.
    """
    _print("Downloading calibration data from Simbad", indent=indent)
    filters = filter_list if filter_list else ["B", "V"]
    my_simbad = Simbad()
    for filter_ in filters:
        my_simbad.add_votable_fields(f"flux({filter_})")
        my_simbad.add_votable_fields(f"flux_error({filter_})")

    simbad_table = my_simbad.query_region(
        center,
        radius=field_of_view_arcmin * 0.66 * u.arcmin,
    )
    _print(f"Found {len(simbad_table)} with the SIMBAD query", indent=indent)
    if not simbad_table:
        _print("No calibration data available", indent=indent + 1, style_name="WARNING")
        return Table(names=["ra", "dec"], dtype=[float, float])

    for filter_ in filters:
        simbad_table.rename_column(f"FLUX_{filter_}", f"{filter_}mag")
        simbad_table.rename_column(f"FLUX_ERROR_{filter_}", f"e_{filter_}mag")

    preferred = _preferred_threshold_column_simbad_style(simbad_table.colnames)
    if preferred is None:
        _print(
            "Calibration issue: Threshold magnitude not recognized",
            indent=indent + 1,
            style_name="ERROR",
        )
        raise RuntimeError
    mag = np.asarray(simbad_table[preferred], dtype=float)
    mask = (mag <= calibration_catalog_mag_range[1]) & (
        mag >= calibration_catalog_mag_range[0]
    )
    simbad_table = simbad_table[mask]
    _print(
        f"{len(simbad_table)} calibration objects remaining after magnitude filtering",
        indent=indent,
    )

    sc = SkyCoord(
        ra=simbad_table["RA"],
        dec=simbad_table["DEC"],
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )
    out = Table()
    out["ra"] = sc.ra.deg
    out["dec"] = sc.dec.deg
    for filter_ in filters:
        if f"{filter_}mag" in simbad_table.colnames:
            out[f"mag_std_{filter_}"] = simbad_table[f"{filter_}mag"]
            if f"e_{filter_}mag" in simbad_table.colnames:
                out[f"err_std_{filter_}"] = simbad_table[f"e_{filter_}mag"]
        else:
            _print(
                f"No calibration data for {filter_} band",
                indent=indent + 1,
                style_name="WARNING",
            )
    return out


def _preferred_threshold_column_simbad_style(colnames) -> str | None:
    """First available column name used for Simbad/Vizier-style magnitude cuts."""
    for name in (
        "Vmag",
        "Rmag",
        "Bmag",
        "Imag",
        "Umag",
        "gmag",
        "rmag",
        "imag",
        "zmag",
        "umag",
    ):
        if name in colnames:
            return name
    return None


def _read_votable_simbad_standard(
    path_calibration_file: str,
    filter_list: list[str],
    calibration_catalog_mag_range: tuple[float, float],
    indent: int,
) -> Table:
    """
    Pre-downloaded Simbad VO table → standard table.

    * Initial cut on ``FLUX_V`` with full ``calibration_catalog_mag_range``.
    * Per filter: remove rows flagged for multiplicity/variability (legacy logic).
    * Positions: ``RA_d``, ``DEC_d`` assumed degrees.
    """
    _print(
        f"Read calibration data from a VO table: {path_calibration_file}",
        indent=indent,
    )
    calib_tbl = Table.read(path_calibration_file, format="votable")
    mask = calib_tbl["FLUX_V"] >= calibration_catalog_mag_range[0]
    mask = mask & (calib_tbl["FLUX_V"] <= calibration_catalog_mag_range[1])
    calib_tbl = calib_tbl[mask]
    _print(
        f"{len(calib_tbl)} calibration objects remaining after magnitude filtering",
        indent=indent,
    )

    for filter_ in filter_list:
        flux = "FLUX_" + filter_
        if flux not in calib_tbl.colnames:
            _print(
                f"No calibration data for {filter_} band",
                indent=indent + 1,
                style_name="WARNING",
            )
            continue
        index_bad = np.where(calib_tbl["FLUX_MULT_" + filter_].mask)
        calib_tbl.remove_rows(index_bad)
        index_bad = np.nonzero(calib_tbl["FLUX_MULT_" + filter_])
        calib_tbl.remove_rows(index_bad)
        index_bad = np.where(calib_tbl["FLUX_VAR_" + filter_].mask)
        calib_tbl.remove_rows(index_bad)
        index_bad = np.nonzero(calib_tbl["FLUX_VAR_" + filter_])
        calib_tbl.remove_rows(index_bad)
        if not calib_tbl:
            raise Exception(
                f"{style.Bcolors.FAIL}\nAll calibration stars in the "
                f"{filter_} removed because of variability and multiplicity "
                f"citeria. -> EXIT {style.Bcolors.ENDC}"
            )

    sc = SkyCoord(
        ra=calib_tbl["RA_d"],
        dec=calib_tbl["DEC_d"],
        unit=u.deg,
        frame="icrs",
    )
    out = Table()
    out["ra"] = sc.ra.deg
    out["dec"] = sc.dec.deg
    for filter_ in filter_list:
        flux = "FLUX_" + filter_
        if flux in calib_tbl.colnames:
            out[f"mag_std_{filter_}"] = calib_tbl[flux]
            out[f"err_std_{filter_}"] = calib_tbl["FLUX_ERROR_" + filter_]
    return out


def _fetch_apass9_standard(
    center: SkyCoord,
    field_of_view_arcmin: float,
    filter_list: list[str],
    calibration_catalog_mag_range: tuple[float, float],
    indent: int,
) -> Table:
    """
    APASS via Vizier — full path in one place (no separate APASS module).

    * Query fixed column list (B, V, Sloan g′r′i′ + errors).
    * Magnitude range on **V** when available (inclusive tuple).
    * Map to ``mag_std_*``; default 0.02 mag if an error column is missing.
    * Lupton Johnson R/I from ``mag_std_r``, ``mag_std_i`` via
      :func:`transforms.add_johnson_ri_to_standard_table`.

    ``filter_list`` is accepted for API symmetry; APASS columns are fixed.
    """
    _print("Downloading APASS (Vizier II/336/apass9)", indent=indent)
    v = Vizier(
        columns=[
            "RAJ2000",
            "DEJ2000",
            "Bmag",
            "Vmag",
            "g'mag",
            "r'mag",
            "i'mag",
            "e_Bmag",
            "e_Vmag",
            "e_g'mag",
            "e_r'mag",
            "e_i'mag",
        ],
        row_limit=-1,
    )
    result = v.query_region(
        center,
        radius=field_of_view_arcmin * u.arcmin,
        catalog=APASS_VIZIER_ID,
    )
    if not result:
        _print("No calibration data available", indent=indent + 1, style_name="WARNING")
        return Table(names=["ra", "dec"], dtype=[float, float])

    apass = result[0]
    if "Vmag" in apass.colnames:
        vmag = np.asarray(apass["Vmag"], dtype=float)
        mask = (vmag <= calibration_catalog_mag_range[1]) & (
            vmag >= calibration_catalog_mag_range[0]
        )
        apass = apass[mask]

    _print(
        f"{len(apass)} calibration objects remaining after magnitude filtering",
        indent=indent,
    )

    out = Table()
    out["ra"] = apass["RAJ2000"]
    out["dec"] = apass["DEJ2000"]
    out["apass_id"] = [f"APASS_{i:06d}" for i in range(len(apass))]

    if "Bmag" in apass.colnames:
        out["mag_std_B"] = apass["Bmag"]
        out["err_std_B"] = (
            apass["e_Bmag"] if "e_Bmag" in apass.colnames else np.full(len(apass), 0.02)
        )
    if "Vmag" in apass.colnames:
        out["mag_std_V"] = apass["Vmag"]
        out["err_std_V"] = (
            apass["e_Vmag"] if "e_Vmag" in apass.colnames else np.full(len(apass), 0.02)
        )

    if "r'mag" in apass.colnames and "i'mag" in apass.colnames:
        out["mag_std_r"] = apass["r'mag"]
        out["mag_std_i"] = apass["i'mag"]
        out["err_std_r"] = (
            apass["e_r'mag"]
            if "e_r'mag" in apass.colnames
            else np.full(len(apass), 0.02)
        )
        out["err_std_i"] = (
            apass["e_i'mag"]
            if "e_i'mag" in apass.colnames
            else np.full(len(apass), 0.02)
        )

    if "g'mag" in apass.colnames:
        out["mag_std_g"] = apass["g'mag"]
        if "e_g'mag" in apass.colnames:
            out["err_std_g"] = apass["e_g'mag"]

    out = add_johnson_ri_to_standard_table(out, inplace=False)
    return out


def fetch_standard_calibration_catalog(
    filter_list: list[str],
    center: SkyCoord,
    *,
    calibration_source: str,
    field_of_view_arcmin: float,
    calibration_catalog_mag_range: tuple[float, float] = (0.0, 18.5),
    vizier_dict: dict[str, str] | None = None,
    path_calibration_file: str | None = None,
    apply_sloan_to_johnson_ri: bool = True,
    indent: int = 1,
) -> Table:
    """
    Fetch a calibration catalog and return the standard-schema ``Table``.

    ``calibration_source`` may be:

    * ``"vsp"``, ``"simbad"``, ``"simbad_vot"`` — special branches above.
    * ``"APASS"`` or any key whose ``vizier_dict`` value is ``II/336/apass9`` —
      :func:`_fetch_apass9_standard`.
    * Any other **key** in ``vizier_dict`` — generic :func:`get_vizier_catalog` +
      :func:`vizier_result_to_standard_table`, then optional Lupton.

    Parameters
    ----------
    field_of_view_arcmin
        Search radius in arcminutes. **Legacy convention:** image FOV for Vizier/APASS;
        ``1.5 * fov_x`` for VSP/Simbad outer argument (Simbad still applies 0.66 inside).
    calibration_catalog_mag_range
        Inclusive (bright, faint) limits on the catalog’s reference band(s); also
        excludes saturated/unreliable bright stars and too-faint objects.
    vizier_dict
        If None, uses :obj:`ost_photometry.calibration_parameters.vizier_dict`.
    path_calibration_file
        Required for ``simbad_vot``.
    apply_sloan_to_johnson_ri
        For generic Vizier: add Johnson R/I from ``mag_std_r``/``mag_std_i`` if missing.
    """
    if vizier_dict is None:
        vizier_dict = calibration_parameters.vizier_dict

    catalog_id: str | None = None
    if calibration_source in vizier_dict:
        catalog_id = vizier_dict[calibration_source]

    if calibration_source == "vsp":
        std = _fetch_vsp_standard(
            center,
            field_of_view_arcmin,
            filter_list,
            calibration_catalog_mag_range,
            indent=indent + 1,
        )
    elif calibration_source == "simbad_vot":
        if path_calibration_file is None:
            raise RuntimeError(
                f"{style.Bcolors.FAIL}simbad_vot requires path_calibration_file"
                f"{style.Bcolors.ENDC}"
            )
        std = _read_votable_simbad_standard(
            path_calibration_file,
            filter_list,
            calibration_catalog_mag_range,
            indent=indent + 1,
        )
    elif calibration_source == "simbad":
        std = _fetch_simbad_standard(
            center,
            field_of_view_arcmin,
            filter_list,
            calibration_catalog_mag_range,
            indent=indent + 1,
        )
    # APASS by name or any alias in vizier_dict pointing at the same Vizier id
    elif calibration_source == "APASS" or catalog_id == APASS_VIZIER_ID:
        std = _fetch_apass9_standard(
            center,
            field_of_view_arcmin,
            filter_list,
            calibration_catalog_mag_range,
            indent=indent + 1,
        )
    elif calibration_source in vizier_dict:
        raw, column_dict, ra_unit = get_vizier_catalog(
            filter_list,
            center,
            field_of_view_arcmin,
            catalog_id,
            magnitude_range=calibration_catalog_mag_range,
            cleanup_magnitudes=True,
            print_infos=True,
            indent=indent + 1,
        )
        std = vizier_result_to_standard_table(
            raw, column_dict, filter_list, ra_unit
        )
        std = _maybe_add_johnson_ri_standard(
            std,
            catalog_identifier=catalog_id,
            apply_sloan_to_johnson_ri=apply_sloan_to_johnson_ri,
        )
    else:
        raise RuntimeError(
            f"{style.Bcolors.FAIL} \nCalibration source not recognized\n"
            "Check ``calibration_source`` and ``vizier_dict`` "
            f"-> EXIT {style.Bcolors.ENDC}"
        )

    return std
