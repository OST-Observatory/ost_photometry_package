"""Manifest of the frames of a data set (archive download or local directory).

One row per frame. Archive metadata and FITS header values live side by
side because the archive does not serialize FILTER, PIERSIDE, GAIN, OFFSET
or READOUTM, and header image types are often wrong. The manifest is the
input of the calibration grouping (:mod:`ost_photometry.reduce.grouping`).

Only ``numpy`` / ``astropy`` are imported, so the module stays usable
without the reduction stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

#: Frame roles in a data set.
ROLE_TARGET = "target"  # requested science frames (downloaded)
ROLE_CALIBRATION = "calibration"  # bias / dark / flat candidates (downloaded)
ROLE_CONTEXT = "context"  # other lights in the time window, metadata only
ROLES = (ROLE_TARGET, ROLE_CALIBRATION, ROLE_CONTEXT)

#: ``(name, kind, default)``; ``kind`` is ``str`` / ``float`` / ``int`` / ``bool``.
MANIFEST_COLUMNS: tuple[tuple[str, type, object], ...] = (
    # identity and storage
    ("frame_id", str, ""),
    ("pk", int, -1),
    ("run", str, ""),
    ("run_pk", int, -1),
    ("file_name", str, ""),
    ("sha256", str, ""),
    ("size", int, 0),
    ("local_path", str, ""),
    ("source_path", str, ""),
    ("role", str, ROLE_TARGET),
    ("downloaded", bool, False),
    # time and exposure
    ("obs_date", str, ""),
    ("jd", float, np.nan),
    ("exptime", float, np.nan),
    # archive classification
    ("instrument_archive", str, ""),
    ("telescope", str, ""),
    ("exposure_type", str, ""),
    ("exposure_type_ml", str, ""),
    ("ml_confidence", float, np.nan),
    ("exposure_type_user", str, ""),
    ("spectrograph", str, ""),
    ("main_target", str, ""),
    ("main_object_id", int, -1),
    ("main_object_name", str, ""),
    # pointing (degrees) and server-side plate solution
    ("ra", float, np.nan),
    ("dec", float, np.nan),
    ("plate_solved", bool, False),
    ("wcs_ra", float, np.nan),
    ("wcs_dec", float, np.nan),
    ("wcs_cd1_1", float, np.nan),
    ("wcs_cd1_2", float, np.nan),
    ("wcs_cd2_1", float, np.nan),
    ("wcs_cd2_2", float, np.nan),
    # FITS header
    ("imagetyp", str, ""),
    ("instrume", str, ""),
    ("telescop", str, ""),
    ("filter", str, ""),
    ("pierside", str, ""),
    ("readoutm", str, ""),
    ("object", str, ""),
    ("objctra", str, ""),
    ("objctdec", str, ""),
    ("gain", float, np.nan),
    ("offset", float, np.nan),
    ("egain", float, np.nan),
    ("set_temp", float, np.nan),
    ("ccd_temp", float, np.nan),
    ("xbinning", int, 1),
    ("ybinning", int, 1),
    ("naxis1", int, 0),
    ("naxis2", int, 0),
    ("bitpix", int, 0),
    ("focallen", float, np.nan),
    ("xpixsz", float, np.nan),
)

_NUMPY_KIND = {str: str, float: np.float64, int: np.int64, bool: np.bool_}
_COLUMN_NAMES = tuple(name for name, _kind, _default in MANIFEST_COLUMNS)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def _to_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _to_int(value: object, default: int) -> int:
    number = _to_float(value)
    return int(number) if math.isfinite(number) else default


def _to_str(value: object) -> str:
    if value is None or value is np.ma.masked:
        return ""
    return str(value).strip()


def parse_ra_dec(ra: object, dec: object) -> tuple[float, float]:
    """RA / Dec in degrees from sexagesimal strings or numbers; ``nan`` if invalid.

    Strings with spaces or colons are read as ``hh mm ss`` / ``dd mm ss``;
    plain numbers as degrees.
    """
    ra_s, dec_s = _to_str(ra), _to_str(dec)
    if not ra_s or not dec_s:
        return float("nan"), float("nan")
    sexagesimal = any(c in ra_s for c in " :hms") or any(c in dec_s for c in " :dms")
    if not sexagesimal:
        ra_deg, dec_deg = _to_float(ra_s), _to_float(dec_s)
    else:
        try:
            from astropy import units as u
            from astropy.coordinates import SkyCoord

            coord = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg))
        except (ValueError, TypeError):
            return float("nan"), float("nan")
        ra_deg, dec_deg = float(coord.ra.deg), float(coord.dec.deg)
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        return float("nan"), float("nan")
    if not (0.0 <= ra_deg < 360.0 and -90.0 <= dec_deg <= 90.0):
        return float("nan"), float("nan")
    return ra_deg, dec_deg


def jd_from_date_obs(date_obs: object) -> float:
    """Julian date from a FITS ``DATE-OBS`` string; ``nan`` if unparsable."""
    text = _to_str(date_obs).replace(" ", "T")
    if not text:
        return float("nan")
    try:
        from astropy.time import Time

        return float(Time(text, format="isot", scale="utc").jd)
    except (ValueError, TypeError):
        try:
            from astropy.time import Time

            return float(Time(text).jd)
        except (ValueError, TypeError):
            return float("nan")


def _first(header: Mapping[str, object], keys: Iterable[str]) -> object:
    for key in keys:
        if key in header and header.get(key) not in (None, ""):
            return header.get(key)
    return None


def header_fields(header: Mapping[str, object] | fits.Header) -> dict[str, object]:
    """Manifest fields from a FITS header (keyword variants are tolerated)."""
    xbin = _first(header, ("XBINNING", "XBIN", "BINX"))
    ybin = _first(header, ("YBINNING", "YBIN", "BINY"))
    if xbin is None and header.get("BINNING"):
        parts = [p for p in str(header.get("BINNING")).lower().split("x") if p.strip()]
        if len(parts) == 2:
            xbin, ybin = parts[0], parts[1]
    date_obs = _to_str(header.get("DATE-OBS"))
    jd = _to_float(header.get("JD"))
    if not math.isfinite(jd):
        jd = jd_from_date_obs(date_obs)
    objctra = _to_str(_first(header, ("OBJCTRA", "OBJRA")))
    objctdec = _to_str(_first(header, ("OBJCTDEC", "OBJDEC")))
    ra, dec = parse_ra_dec(objctra, objctdec)
    if not math.isfinite(ra):
        ra, dec = parse_ra_dec(header.get("RA"), header.get("DEC"))
    return {
        "imagetyp": _to_str(header.get("IMAGETYP")),
        "instrume": _to_str(header.get("INSTRUME")),
        "telescop": _to_str(header.get("TELESCOP")),
        "filter": _to_str(header.get("FILTER")),
        "pierside": _to_str(header.get("PIERSIDE")).upper(),
        "readoutm": _to_str(_first(header, ("READOUTM", "READMODE", "RDMODE"))),
        "object": _to_str(header.get("OBJECT")),
        "objctra": objctra,
        "objctdec": objctdec,
        "gain": _to_float(_first(header, ("GAIN", "ISO"))),
        "offset": _to_float(_first(header, ("OFFSET", "PEDESTAL", "BLKLEVEL"))),
        "egain": _to_float(header.get("EGAIN")),
        "set_temp": _to_float(header.get("SET-TEMP")),
        "ccd_temp": _to_float(_first(header, ("CCD-TEMP", "CCDTEMP", "TEMPERAT"))),
        "xbinning": _to_int(xbin, 1),
        "ybinning": _to_int(ybin, _to_int(xbin, 1)),
        "naxis1": _to_int(header.get("NAXIS1"), 0),
        "naxis2": _to_int(header.get("NAXIS2"), 0),
        "bitpix": _to_int(header.get("BITPIX"), 0),
        "focallen": _to_float(_first(header, ("FOCALLEN", "FOCAL", "FOCLEN"))),
        "xpixsz": _to_float(_first(header, ("XPIXSZ", "PIXSIZE1", "XPIXSIZE", "PIXSIZE"))),
        "obs_date": date_obs,
        "jd": jd,
        "exptime": _to_float(_first(header, ("EXPTIME", "EXPOSURE", "EXPOSUR"))),
        "header_ra": ra,
        "header_dec": dec,
    }


def apply_header(row: dict[str, object], header: Mapping[str, object] | fits.Header) -> None:
    """Merge header fields into a manifest row (archive values are kept)."""
    fields = header_fields(header)
    header_ra = fields.pop("header_ra")
    header_dec = fields.pop("header_dec")
    for key, value in fields.items():
        current = row.get(key)
        if key in ("obs_date", "jd", "exptime"):
            missing = current in (None, "") or (
                isinstance(current, float) and not math.isfinite(current)
            )
            if not missing:
                continue
        row[key] = value
    if not math.isfinite(_to_float(row.get("ra"))):
        row["ra"], row["dec"] = header_ra, header_dec


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def empty_manifest() -> Table:
    table = Table()
    for name, kind, _default in MANIFEST_COLUMNS:
        table[name] = np.zeros(0, dtype=_NUMPY_KIND[kind])
    return table


def manifest_from_rows(rows: Iterable[Mapping[str, object]]) -> Table:
    """Build the manifest; unknown keys are dropped, missing ones get defaults.

    Rows are sorted by ``jd`` (unknown times last).
    """
    rows = list(rows)
    if not rows:
        return empty_manifest()
    table = Table()
    for name, kind, default in MANIFEST_COLUMNS:
        values = [row.get(name, default) for row in rows]
        if kind is str:
            table[name] = np.array([_to_str(v) for v in values], dtype=str)
        elif kind is bool:
            table[name] = np.array([bool(v) for v in values], dtype=np.bool_)
        elif kind is int:
            table[name] = np.array([_to_int(v, int(default)) for v in values], dtype=np.int64)
        else:
            table[name] = np.array([_to_float(v) for v in values], dtype=np.float64)
    order = np.argsort(np.nan_to_num(np.asarray(table["jd"], dtype=float), nan=np.inf), kind="stable")
    return table[order]


def manifest_rows(table: Table) -> list[dict[str, object]]:
    """Rows of a manifest as plain dicts (column order of the table)."""
    names = table.colnames
    return [{name: row[name] for name in names} for row in table]


def write_manifest(table: Table, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return path


def read_manifest(path: str | Path) -> Table:
    """Read a manifest; missing canonical columns are added with defaults."""
    table = Table.read(Path(path), format="ascii.ecsv")
    for name, kind, default in MANIFEST_COLUMNS:
        if name in table.colnames:
            column = table[name]
            if hasattr(column, "filled"):
                fill = "" if kind is str else (False if kind is bool else default)
                column = column.filled(fill)
            if kind is str:
                table[name] = np.array([_to_str(v) for v in column], dtype=str)
            elif kind is bool:
                table[name] = np.asarray(column, dtype=bool)
            continue
        if kind is str:
            table[name] = np.array([str(default)] * len(table), dtype=str)
        else:
            table[name] = np.full(len(table), default, dtype=_NUMPY_KIND[kind])
    return table


__all__ = [
    "MANIFEST_COLUMNS",
    "ROLES",
    "ROLE_CALIBRATION",
    "ROLE_CONTEXT",
    "ROLE_TARGET",
    "apply_header",
    "empty_manifest",
    "header_fields",
    "jd_from_date_obs",
    "manifest_from_rows",
    "manifest_rows",
    "parse_ra_dec",
    "read_manifest",
    "write_manifest",
]
