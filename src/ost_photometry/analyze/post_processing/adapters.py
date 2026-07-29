"""Legacy wide tables -> epoch-native long layout (bridge)."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from astropy.table import Table

from . import schema

_MAG_COL_RE = re.compile(r"^(.+) \((transformed|simple), image=([^)]+)\)$")
_ERR_COL_RE = re.compile(r"^(.+)_err \((transformed|simple), image=([^)]+)\)$")
_FLUX_COL_RE = re.compile(r"^(.+) \(flux, image=([^)]+)\)$")

# Legacy wide tables fill missing ``mag_cali_trans`` with this in :func:`mk_magnitudes_table`.
_LEGACY_WIDE_MISSING_MAG: float = 999.0


def _attach_schema_meta(out: Table) -> Table:
    meta = dict(out.meta) if out.meta else {}
    meta.setdefault("photometry_schema", schema.PHOTOMETRY_TABLE_SCHEMA_ID)
    out.meta = meta
    return out


def _has_long_format_magnitude_columns(tbl: Table) -> bool:
    """
    Magnitude columns for epoch-long differential / bridge tables.

    Includes calibrated ``mag_cal_*``, dumped instrumental ``mag_inst_*``, and
    per-filter instrumental ``mag_<filter>`` from :func:`observation_to_calibration_epochs`
    (excludes catalog ``mag_std_*`` and legacy ``mag_cali_*`` embedded names).
    """
    for n in tbl.colnames:
        if n.startswith("mag_cal_") or n.startswith("mag_inst_"):
            return True
        if n.startswith("mag_std_") or n.startswith("mag_cali"):
            continue
        if n.startswith("mag_"):
            return True
    return False


def _is_epoch_native_vstack(tbl: Table) -> bool:
    """True if table looks like vstacked epoch-native calibration output (long form)."""
    if len(tbl) == 0:
        return True
    time_col = "epoch_id" in tbl.colnames or "frame_id" in tbl.colnames
    has_star_id = "id" in tbl.colnames or "i" in tbl.colnames
    has_positions = ("x" in tbl.colnames and "y" in tbl.colnames) or (
        "ra" in tbl.colnames and "dec" in tbl.colnames
    )
    return bool(time_col and has_star_id and has_positions and _has_long_format_magnitude_columns(tbl))


def _wide_legacy_to_long_table(tbl: Table) -> Table:
    """
    One row per star -> one row per (star, epoch) with ``mag_cal_*`` / ``err_cal_*``
    and extracted ``flux_inst_*`` / ``flux_err_inst_*`` when wide columns
    ``{F} (flux, image=TAG)`` exist.

    Expects wide-table columns ``i`` or ``id``, ``x``, ``y``, ``ra (deg)``/``ra``,
    ``dec (deg)``/``dec``, and magnitude columns
    ``{F} (transformed|simple, image=TAG)`` plus matching ``{F}_err (...)``.

    For a given ``image=TAG``, if **usable** transformed columns exist for that tag
    (at least one finite magnitude not equal to the legacy 999 placeholder),
    **both** transformed and simple rows are emitted (``epoch_TAG`` and
    ``epoch_TAG_simple``) so both calibrations are available in epoch-native files.

    If transformed columns are present but only placeholders (e.g. no
    ``mag_cali_trans`` when ``apply_transformation=False``), only **simple**
    rows are kept so zero-point / simple calibration fills ``mag_cal_*``.

    JD-based light curves should exclude redundant ``*_simple`` rows when a
    transformed row exists for the same epoch (see
    :func:`prepare_time_series_epoch_native`).
    """
    mag_pairs: list[tuple[str, str, str, str, str]] = []
    for name in tbl.colnames:
        m = _MAG_COL_RE.match(name)
        if not m:
            continue
        filter_, kind, tag = m.group(1), m.group(2), m.group(3)
        err_name = f"{filter_}_err ({kind}, image={tag})"
        if err_name not in tbl.colnames:
            continue
        mag_pairs.append((filter_, kind, tag, name, err_name))

    if not mag_pairs:
        raise ValueError(
            "legacy_wide_table_to_epoch_native: no columns matching "
            "'{filter} (transformed|simple, image=...)' with matching _err columns."
        )

    by_epoch: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for filter_, kind, tag, mcol, ecol in mag_pairs:
        by_epoch[(kind, tag)].append((filter_, mcol, ecol))

    def _usable_transformed_for_tag(tag: str) -> bool:
        flist = by_epoch.get(("transformed", tag), [])
        if not flist:
            return False
        for _, mcol, _ecol in flist:
            arr = np.asarray(tbl[mcol], dtype=float)
            for v in arr.ravel():
                if np.isfinite(v) and abs(float(v) - _LEGACY_WIDE_MISSING_MAG) > 1e-6:
                    return True
        return False

    tags_with_usable_transformed = {
        tag
        for (kind, tag) in by_epoch
        if kind == "transformed" and _usable_transformed_for_tag(tag)
    }

    flux_by_tag: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for name in tbl.colnames:
        mf = _FLUX_COL_RE.match(name)
        if not mf:
            continue
        filter_f, tag_f = mf.group(1), mf.group(2)
        err_f = f"{filter_f}_err (flux, image={tag_f})"
        if err_f not in tbl.colnames:
            continue
        flux_by_tag[str(tag_f)].append((filter_f, name, err_f))
    flux_filters = sorted({p[0] for fl in flux_by_tag.values() for p in fl})

    id_key = "i" if "i" in tbl.colnames else "id"
    ra_key = "ra (deg)" if "ra (deg)" in tbl.colnames else "ra"
    dec_key = "dec (deg)" if "dec (deg)" in tbl.colnames else "dec"
    n = len(tbl)

    all_filters = sorted({p[0] for p in mag_pairs})
    rows: list[dict] = []

    for row_idx in range(n):
        sid = int(tbl[id_key][row_idx])
        x = float(tbl["x"][row_idx])
        y = float(tbl["y"][row_idx])
        ra = float(tbl[ra_key][row_idx])
        dec = float(tbl[dec_key][row_idx])

        for (kind, tag), flist in sorted(by_epoch.items()):
            if kind == "transformed" and tag not in tags_with_usable_transformed:
                continue
            if kind == "transformed":
                eid = f"epoch_{tag}"
            else:
                eid = f"epoch_{tag}_simple"

            row: dict = {
                "id": sid,
                "x": x,
                "y": y,
                "ra": ra,
                "dec": dec,
                "epoch_id": eid,
            }
            for filter_ in all_filters:
                row[f"mag_cal_{filter_}"] = np.nan
                row[f"err_cal_{filter_}"] = np.nan
            for filter_, mcol, ecol in flist:
                row[f"mag_cal_{filter_}"] = float(tbl[mcol][row_idx])
                row[f"err_cal_{filter_}"] = float(tbl[ecol][row_idx])
            for filter_ in flux_filters:
                row[f"flux_inst_{filter_}"] = np.nan
                row[f"flux_err_inst_{filter_}"] = np.nan
            for filter_, mcol, ecol in flux_by_tag.get(str(tag), []):
                row[f"flux_inst_{filter_}"] = float(tbl[mcol][row_idx])
                row[f"flux_err_inst_{filter_}"] = float(tbl[ecol][row_idx])
            rows.append(row)

    out = Table(rows)
    return _attach_schema_meta(out)


def legacy_wide_table_to_epoch_native(tbl: Table) -> Table:
    """
    Normalize a photometry table to epoch-native long form and attach schema metadata.

    - If the table already looks like long-form epoch data (``epoch_id``/``frame_id``,
      star id, positions, and ``mag_cal_*``, ``mag_inst_*``, or instrumental ``mag_<filter>``
      from the differential / bridge pipeline), returns a copy with ``photometry_schema`` meta.
    - If it looks like a legacy **wide** table (one row per star, magnitudes in
      ``{filter} (transformed|simple, image=...)`` columns), expands to one row per
      star and epoch.
    """
    if len(tbl) == 0:
        return _attach_schema_meta(tbl.copy())
    if _is_epoch_native_vstack(tbl):
        return _attach_schema_meta(tbl.copy())
    return _wide_legacy_to_long_table(tbl)


def ensure_epoch_native_photometry_table(tbl: Table) -> Table:
    """Same as :func:`legacy_wide_table_to_epoch_native` (clearer name for pipeline callers)."""
    return legacy_wide_table_to_epoch_native(tbl)
