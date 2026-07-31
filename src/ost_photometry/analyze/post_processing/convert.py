"""Convert calibrated magnitudes between filter sets and Vega/AB systems."""

from __future__ import annotations

import numpy as np
from astropy import uncertainty as unc
import astropy.units as u
from astropy.table import Table

from ... import calibration_parameters, terminal_output
from ..calibration_sources.transforms import johnson_ubvri_from_sloan_arrays
from .magnitude_systems import (
    annotate_table_magnitude_meta,
    infer_filter_set,
    resolve_catalog_magnitude_system,
    resolve_effective_output,
    vega_to_ab_offset,
)


def _build_epoch_native_magnitude_distributions(
    tbl: Table,
    distribution_samples: int,
) -> dict[str, unc.core.NdarrayDistribution]:
    """
    Build one normal distribution per filter from ``mag_cal_<F>`` / ``err_cal_<F>``,
    or (if no calibrated column for that filter) ``mag_inst_<F>`` / ``err_inst_<F>``.
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


def _list_cal_filters(tbl: Table) -> list[str]:
    filters: list[str] = []
    for name in tbl.colnames:
        if name.startswith("mag_cal_"):
            f = name[len("mag_cal_") :]
            if f and f not in filters:
                filters.append(f)
    if filters:
        return filters
    for name in tbl.colnames:
        if name.startswith("mag_inst_"):
            f = name[len("mag_inst_") :]
            if f and f not in filters:
                filters.append(f)
    return filters


def _get_mag_err(
    tbl: Table, filter_: str
) -> tuple[np.ndarray, np.ndarray] | None:
    for mag_p, err_p in (("mag_cal_", "err_cal_"), ("mag_inst_", "err_inst_")):
        mc, ec = f"{mag_p}{filter_}", f"{err_p}{filter_}"
        if mc in tbl.colnames:
            m = np.asarray(tbl[mc], dtype=float)
            e = (
                np.abs(np.asarray(tbl[ec], dtype=float))
                if ec in tbl.colnames
                else np.zeros_like(m)
            )
            return m, e
    return None


def _set_cal_mag_err(tbl: Table, filter_: str, mag: np.ndarray, err: np.ndarray) -> None:
    tbl[f"mag_cal_{filter_}"] = np.asarray(mag, dtype=float)
    tbl[f"err_cal_{filter_}"] = np.asarray(err, dtype=float)


def _convert_sdss_jordi(
    tbl: Table,
    data_dict: dict[str, unc.core.NdarrayDistribution],
    distribution_samples: int,
    *,
    promote_to_mag_cal: bool = True,
) -> list[str]:
    """Apply Jordi et al. (2005); write ``mag_sdss_*`` and optionally promote to ``mag_cal_*``."""
    calib_functions = calibration_parameters.filter_system_conversions["SDSS"][
        "Jordi_et_al_2005"
    ]
    dd: dict = {**data_dict, "distribution_samples": distribution_samples}
    produced: list[str] = []
    for band in ("g", "u", "r", "i", "z"):
        result = calib_functions[band](**dd)
        if result is None:
            continue
        dd[band] = result
        med = result.pdf_median()
        std = result.pdf_std()
        mag = np.asarray(u.Quantity(med).to_value(u.mag), dtype=float)
        err = np.asarray(u.Quantity(std).to_value(u.mag), dtype=float)
        tbl[f"mag_sdss_{band}"] = mag
        tbl[f"err_sdss_{band}"] = err
        if promote_to_mag_cal:
            _set_cal_mag_err(tbl, band, mag, err)
        produced.append(band)
    return produced


def _apply_zp_flip(
    tbl: Table,
    *,
    from_system: str,
    to_system: str,
    filters: list[str] | None = None,
) -> list[str]:
    """Apply Vega↔AB constant offsets in place on ``mag_cal_*``."""
    if from_system == to_system:
        return []
    if from_system not in ("vega", "ab") or to_system not in ("vega", "ab"):
        raise ValueError(
            f"ZP flip requires vega/ab systems, got {from_system!r} → {to_system!r}"
        )
    sign = 1.0 if (from_system == "vega" and to_system == "ab") else -1.0
    changed: list[str] = []
    use = filters if filters is not None else _list_cal_filters(tbl)
    for f in use:
        off = vega_to_ab_offset(f)
        if off is None:
            terminal_output.print_to_terminal(
                f"No Vega↔AB offset for filter {f!r}; leaving unchanged.",
                style_name="WARNING",
            )
            continue
        pair = _get_mag_err(tbl, f)
        if pair is None:
            continue
        mag, err = pair
        _set_cal_mag_err(tbl, f, mag + sign * off, err)
        changed.append(f)
    return changed


def _convert_sdss_to_bessell(tbl: Table) -> list[str]:
    """Lupton/Jester SDSS→Johnson; write ``mag_cal_U/B/V/R/I``."""
    def _arr(f: str) -> np.ndarray | None:
        pair = _get_mag_err(tbl, f)
        return None if pair is None else pair[0]

    def _err(f: str) -> np.ndarray | None:
        pair = _get_mag_err(tbl, f)
        return None if pair is None else pair[1]

    bands = johnson_ubvri_from_sloan_arrays(
        u=_arr("u"),
        g=_arr("g"),
        r=_arr("r"),
        i=_arr("i"),
        err_u=_err("u"),
        err_g=_err("g"),
        err_r=_err("r"),
        err_i=_err("i"),
    )
    if not bands:
        raise ValueError(
            "SDSS→Bessell conversion requires Sloan g,r (for B,V) and/or r,i (for R,I) "
            "as mag_cal_* / mag_inst_* columns."
        )
    produced: list[str] = []
    for f, (mag, err) in bands.items():
        _set_cal_mag_err(tbl, f, mag, err)
        produced.append(f)
    return produced


def convert_magnitudes_to_other_system(
    tbl: Table,
    target_filter_system: str | None = None,
    distribution_samples: int = 1000,
    *,
    output_filter_set: str = "auto",
    output_magnitude_system: str = "auto",
    calibration_source: str | None = None,
    source_magnitude_system: str | None = None,
    source_filter_set: str | None = None,
) -> Table:
    """
    Convert epoch-native calibrated magnitudes toward a requested filter set / ZP.

    Prefer ``output_filter_set`` / ``output_magnitude_system``. Legacy
    ``target_filter_system`` (``SDSS`` / ``AB`` / ``BESSELL``) is still accepted.

    Parameters
    ----------
    tbl
        Table with ``mag_cal_*`` and/or ``mag_inst_*`` columns.
    target_filter_system
        Deprecated alias: ``SDSS``, ``AB``, or ``BESSELL``.
    distribution_samples
        Monte-Carlo samples for Jordi distribution propagation.
    output_filter_set, output_magnitude_system
        ``auto`` / ``bessell`` / ``sdss`` and ``auto`` / ``vega`` / ``ab``.
    calibration_source
        Catalog name for resolving the source magnitude system.
    source_magnitude_system, source_filter_set
        Overrides when known from prior calibration meta.
    """
    from .magnitude_systems import apply_target_filter_system_alias

    ofs = output_filter_set
    oms = output_magnitude_system
    if target_filter_system:
        a_fs, a_ms = apply_target_filter_system_alias(target_filter_system)
        if a_fs is not None:
            ofs = a_fs
        if a_ms is not None:
            oms = a_ms

    filters = _list_cal_filters(tbl)
    if not filters:
        terminal_output.print_to_terminal(
            "Magnitude conversion skipped: no ``mag_cal_*`` or ``mag_inst_*`` columns.",
            style_name="WARNING",
        )
        return tbl

    cal_fs = source_filter_set or infer_filter_set(filters)
    cat_ms = source_magnitude_system or resolve_catalog_magnitude_system(
        calibration_source
    )
    if cat_ms == "unknown" and source_magnitude_system is None:
        # Heuristic: SDSS filter set → AB; Bessell → Vega
        if cal_fs == "sdss":
            cat_ms = "ab"
        elif cal_fs == "bessell":
            cat_ms = "vega"

    effective = resolve_effective_output(
        output_filter_set=ofs,  # type: ignore[arg-type]
        output_magnitude_system=oms,  # type: ignore[arg-type]
        calibrated_filter_set=cal_fs,  # type: ignore[arg-type]
        catalog_magnitude_system=cat_ms,  # type: ignore[arg-type]
        convert_magnitudes=True,
    )

    out = tbl.copy()
    notes: list[str] = []

    # Filter-set change
    if effective.filter_set == "sdss" and cal_fs != "sdss":
        if cal_fs not in ("bessell", "mixed", "unknown"):
            raise ValueError(
                f"Cannot convert filter set {cal_fs!r} → SDSS; need Bessell UBVRI inputs."
            )
        # Ensure Vega-like Bessell before Jordi if currently AB
        if cat_ms == "ab" and cal_fs == "bessell":
            flipped = _apply_zp_flip(out, from_system="ab", to_system="vega", filters=filters)
            if flipped:
                notes.append("AB→Vega before Jordi")
                cat_ms = "vega"
        data_dict = _build_epoch_native_magnitude_distributions(
            out, distribution_samples=distribution_samples
        )
        produced = _convert_sdss_jordi(
            out, data_dict, distribution_samples, promote_to_mag_cal=True
        )
        if not produced:
            raise ValueError(
                "Bessell→SDSS (Jordi) produced no bands; need suitable UBVRI combinations."
            )
        notes.append(f"Jordi→SDSS bands {produced}")
        cal_fs = "sdss"
        cat_ms = "ab"
        filters = produced

    elif effective.filter_set == "bessell" and cal_fs == "sdss":
        produced = _convert_sdss_to_bessell(out)
        notes.append(f"Lupton→Bessell bands {produced}")
        cal_fs = "bessell"
        # Lupton relations are on AB SDSS → Johnson (typically treated as Vega-like)
        cat_ms = "ab"  # still AB until ZP flip; Johnson mags from AB SDSS are AB-ish
        # Actually Lupton maps AB SDSS fluxes to Johnson numbers that are used as
        # Vega-system Johnson magnitudes in practice. Treat result as Vega for ZP axis.
        cat_ms = "vega"
        filters = produced

    elif effective.filter_set not in ("auto", cal_fs, "unknown") and cal_fs not in (
        "unknown",
        effective.filter_set,
    ):
        raise ValueError(
            f"Unsupported filter-set conversion {cal_fs!r} → {effective.filter_set!r}."
        )

    # ZP-only flip on current filter set
    if effective.magnitude_system in ("vega", "ab") and cat_ms in ("vega", "ab"):
        if effective.magnitude_system != cat_ms:
            flipped = _apply_zp_flip(
                out,
                from_system=cat_ms,
                to_system=effective.magnitude_system,
                filters=filters,
            )
            if flipped:
                notes.append(
                    f"{cat_ms}→{effective.magnitude_system} offsets on {flipped}"
                )
                cat_ms = effective.magnitude_system

    annotate_table_magnitude_meta(
        out,
        filter_set=effective.filter_set if effective.filter_set != "unknown" else cal_fs,
        magnitude_system=effective.magnitude_system
        if effective.magnitude_system != "unknown"
        else cat_ms,
        catalog_magnitude_system=resolve_catalog_magnitude_system(calibration_source)
        if calibration_source
        else cat_ms,
        calibration_source=calibration_source,
        conversion_note="; ".join(notes) if notes else "identity",
    )
    return out


# Back-compat name used in older call sites
def convert_magnitudes_for_output(
    tbl: Table,
    *,
    output_filter_set: str = "auto",
    output_magnitude_system: str = "auto",
    calibration_source: str | None = None,
    distribution_samples: int = 1000,
    source_magnitude_system: str | None = None,
    source_filter_set: str | None = None,
) -> Table:
    """Convert using explicit output_* preferences (no legacy target string)."""
    return convert_magnitudes_to_other_system(
        tbl,
        distribution_samples=distribution_samples,
        output_filter_set=output_filter_set,
        output_magnitude_system=output_magnitude_system,
        calibration_source=calibration_source,
        source_magnitude_system=source_magnitude_system,
        source_filter_set=source_filter_set,
    )
