"""Filter-set and magnitude-system (Vega/AB) model for calibrated photometry.

Two independent axes:

* **Filter set** — ``bessell`` (U,B,V,R,I) vs ``sdss`` (u,g,r,i,z)
* **Magnitude system** — ``vega`` vs ``ab`` (SDSS magnitudes are always AB)

Same-bandpass Vega↔AB uses constant per-filter offsets
(``m_AB = m_Vega + Δ``); ``Δ`` is a property of the bandpass and reference
spectra, not of stellar colour (Blanton & Roweis 2007 and standard Bessell
zeropoint tables). Filter-set changes use Jordi / Lupton colour terms.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from astropy.table import Table

from ... import calibration_parameters, terminal_output
from ..warnings_types import OstPhotometryAnalyzeWarning

FilterSet = Literal["bessell", "sdss", "mixed", "unknown"]
MagnitudeSystem = Literal["vega", "ab", "mixed", "unknown"]
OutputFilterSet = Literal["auto", "bessell", "sdss"]
OutputMagnitudeSystem = Literal["auto", "vega", "ab"]

META_MAGNITUDE_SYSTEM = "ost_photometry.magnitude_system"
META_FILTER_SET = "ost_photometry.filter_set"
META_CATALOG_SYSTEM = "ost_photometry.calibration_catalog_system"
META_CATALOG_SOURCE = "ost_photometry.calibration_source"
META_CONVERSION_NOTE = "ost_photometry.magnitude_conversion"

BESSELL_FILTERS = frozenset({"U", "B", "V", "R", "I"})
SDSS_FILTERS = frozenset({"u", "g", "r", "i", "z", "u`", "g`", "r`", "i`", "z`", "z-s`"})

# Unfiltered / luminance / white glass: no catalog ZP. Case-insensitive aliases.
UNCALIBRATED_BROADBAND_ALIASES = frozenset({
    "c",
    "clear",
    "l",
    "lum",
    "luminance",
    "nofilter",
    "no_filter",
    "none",
    "open",
    "unfiltered",
    "w",
    "white",
})

# m_AB = m_Vega + offset  (Bessell / Johnson–Cousins; Blanton & Roweis 2007 style)
# SDSS ugriz are defined on (nearly) the AB system → offset 0 when already AB.
VEGA_TO_AB_OFFSET: dict[str, float] = {
    "U": 0.79,
    "B": -0.09,
    "V": 0.02,
    "R": 0.21,
    "I": 0.45,
    "u": 0.91,
    "g": -0.08,
    "r": 0.16,
    "i": 0.37,
    "z": 0.54,
    "u`": 0.91,
    "g`": -0.08,
    "r`": 0.16,
    "i`": 0.37,
    "z`": 0.54,
    "z-s`": 0.54,
}

# Friendly catalog name and Vizier id → primary magnitude system of standards.
CATALOG_MAGNITUDE_SYSTEMS: dict[str, MagnitudeSystem] = {
    "APASS": "vega",
    "II/336/apass9": "vega",
    "SDSS_Release_16": "ab",
    "V/147/sdss12": "ab",
    "Stetson_2019": "vega",
    "J/MNRAS/485/3042/table4": "vega",
    "Pancino_2022": "vega",
    "J/A+A/664/A109/table5": "vega",
    "Swift/UVOT": "ab",
    "II/339/uvotssc1": "ab",
    "XMM-OM": "ab",
    "II/370/xmmom5s": "ab",
    "UCAC4": "vega",
    "I/322A": "vega",
    "GSC2.3": "vega",
    "I/305": "vega",
    "NOMAD": "vega",
    "I/297": "vega",
    "HMUBV": "vega",
    "II/168/ubvmeans": "vega",
    "GSPC2.4": "vega",
    "II/272/gspc24": "vega",
    "VRI-NCC": "vega",
    "J/MNRAS/443/725/catalog": "vega",
    "USNO-B1.0": "vega",
    "I/284/out": "vega",
    "URAT1": "vega",
    "I/329": "vega",
}


@dataclass(frozen=True)
class EffectiveMagnitudeOutput:
    """Resolved output preference after applying ``auto`` and SDSS→AB rules."""

    filter_set: FilterSet
    magnitude_system: MagnitudeSystem
    catalog_magnitude_system: MagnitudeSystem
    calibrated_filter_set: FilterSet
    needs_convert: bool
    conversion_note: str = ""


def normalize_filter_name(filter_: str) -> str:
    """Strip common prime suffixes for set membership checks."""
    f = str(filter_).strip()
    if f.endswith("`"):
        return f[:-1] if f[:-1] else f
    return f


def infer_filter_set(filters: Sequence[str] | Iterable[str]) -> FilterSet:
    """Infer ``bessell`` / ``sdss`` / ``mixed`` / ``unknown`` from filter names.

    Bessell letters are uppercase (``U,B,V,R,I``); SDSS are lowercase
    (``u,g,r,i,z``). Case is significant.
    """
    names = [str(f).strip() for f in filters if f]
    if not names:
        return "unknown"
    is_b: list[bool] = []
    is_s: list[bool] = []
    for f in names:
        n = normalize_filter_name(f)
        if n in BESSELL_FILTERS:
            is_b.append(True)
            is_s.append(False)
        elif n in {"u", "g", "r", "i", "z"} or f in SDSS_FILTERS:
            is_b.append(False)
            is_s.append(True)
        else:
            is_b.append(False)
            is_s.append(False)
    nb, ns = sum(is_b), sum(is_s)
    if nb and ns:
        return "mixed"
    if nb:
        return "bessell"
    if ns:
        return "sdss"
    return "unknown"


def resolve_catalog_magnitude_system(calibration_source: str | None) -> MagnitudeSystem:
    """Map ``calibration_source`` (friendly name or Vizier id) to Vega/AB/mixed/unknown."""
    if not calibration_source:
        return "unknown"
    key = str(calibration_source).strip()
    if key in CATALOG_MAGNITUDE_SYSTEMS:
        return CATALOG_MAGNITUDE_SYSTEMS[key]
    # Resolve friendly → Vizier id
    vizier = calibration_parameters.vizier_dict.get(key)
    if vizier and vizier in CATALOG_MAGNITUDE_SYSTEMS:
        return CATALOG_MAGNITUDE_SYSTEMS[vizier]
    if key in calibration_parameters.vizier_dict.values():
        return CATALOG_MAGNITUDE_SYSTEMS.get(key, "unknown")
    return "unknown"


def vega_to_ab_offset(filter_: str) -> float | None:
    """Return ``Δ`` for ``m_AB = m_Vega + Δ``, or ``None`` if unknown."""
    if filter_ in VEGA_TO_AB_OFFSET:
        return VEGA_TO_AB_OFFSET[filter_]
    n = normalize_filter_name(filter_)
    if n in VEGA_TO_AB_OFFSET:
        return VEGA_TO_AB_OFFSET[n]
    if n.upper() in VEGA_TO_AB_OFFSET:
        return VEGA_TO_AB_OFFSET[n.upper()]
    if n.lower() in VEGA_TO_AB_OFFSET:
        return VEGA_TO_AB_OFFSET[n.lower()]
    return None


def magnitude_system_axis_suffix(magnitude_system: str | None) -> str:
    """Y-axis suffix for magnitude light curves."""
    sys_ = (magnitude_system or "unknown").lower()
    if sys_ == "vega":
        return " [mag] (Vega)"
    if sys_ == "ab":
        return " [mag] (AB)"
    if sys_ == "mixed":
        return " [mag] (mixed)"
    return " [mag]"


def format_magnitude_output_label(
    filter_set: str | None,
    magnitude_system: str | None,
    *,
    catalog_source: str | None = None,
) -> str:
    """Human-readable output summary for logs."""
    fs = filter_set or "unknown"
    ms = magnitude_system or "unknown"
    base = f"{fs.capitalize() if fs != 'sdss' else 'SDSS'} / {ms.upper() if ms in ('ab',) else ms.capitalize()}"
    if catalog_source:
        return f"{base} (catalog {catalog_source})"
    return base


def validate_magnitude_output_request(
    *,
    output_filter_set: OutputFilterSet = "auto",
    output_magnitude_system: OutputMagnitudeSystem = "auto",
) -> None:
    """
    Raise ``ValueError`` for illegal combinations (e.g. SDSS + Vega).

    ``auto`` is always allowed; SDSS forces AB when resolved later.
    """
    ofs = str(output_filter_set).lower()
    oms = str(output_magnitude_system).lower()
    if ofs not in ("auto", "bessell", "sdss"):
        raise ValueError(
            f"output_filter_set={output_filter_set!r} invalid; "
            "use 'auto', 'bessell', or 'sdss'."
        )
    if oms not in ("auto", "vega", "ab"):
        raise ValueError(
            f"output_magnitude_system={output_magnitude_system!r} invalid; "
            "use 'auto', 'vega', or 'ab'."
        )
    if ofs == "sdss" and oms == "vega":
        raise ValueError(
            "Inconsistent magnitude output request: SDSS magnitudes are defined "
            "on the AB system; output_magnitude_system='vega' with "
            "output_filter_set='sdss' is not allowed."
        )


def resolve_effective_output(
    *,
    output_filter_set: OutputFilterSet = "auto",
    output_magnitude_system: OutputMagnitudeSystem = "auto",
    calibrated_filter_set: FilterSet | str = "unknown",
    catalog_magnitude_system: MagnitudeSystem | str = "unknown",
    convert_magnitudes: bool = False,
) -> EffectiveMagnitudeOutput:
    """
    Resolve ``auto`` preferences and SDSS→AB; report whether conversion is needed.
    """
    validate_magnitude_output_request(
        output_filter_set=output_filter_set,
        output_magnitude_system=output_magnitude_system,
    )
    cal_fs: FilterSet = calibrated_filter_set if calibrated_filter_set in (
        "bessell",
        "sdss",
        "mixed",
        "unknown",
    ) else infer_filter_set([])  # type: ignore[assignment]
    if calibrated_filter_set in ("bessell", "sdss", "mixed", "unknown"):
        cal_fs = calibrated_filter_set  # type: ignore[assignment]
    else:
        cal_fs = "unknown"

    cat_ms: MagnitudeSystem
    if catalog_magnitude_system in ("vega", "ab", "mixed", "unknown"):
        cat_ms = catalog_magnitude_system  # type: ignore[assignment]
    else:
        cat_ms = "unknown"

    ofs = str(output_filter_set).lower()
    oms = str(output_magnitude_system).lower()

    eff_fs: FilterSet
    if ofs == "auto":
        eff_fs = cal_fs if cal_fs in ("bessell", "sdss") else (
            "bessell" if cal_fs == "mixed" else cal_fs
        )
    else:
        eff_fs = ofs  # type: ignore[assignment]

    if oms == "auto":
        if eff_fs == "sdss":
            eff_ms: MagnitudeSystem = "ab"
        elif cat_ms in ("vega", "ab"):
            eff_ms = cat_ms
        else:
            eff_ms = "vega" if eff_fs == "bessell" else "ab"
    else:
        eff_ms = oms  # type: ignore[assignment]

    if eff_fs == "sdss":
        eff_ms = "ab"

    needs = False
    note = ""
    if convert_magnitudes:
        fs_change = (
            eff_fs in ("bessell", "sdss")
            and cal_fs in ("bessell", "sdss", "mixed")
            and eff_fs != cal_fs
            and not (cal_fs == "mixed")
        )
        # mixed → concrete set counts as needing convert
        if cal_fs == "mixed" and eff_fs in ("bessell", "sdss"):
            fs_change = True
        if cal_fs == "unknown" and ofs != "auto":
            fs_change = True
        ms_change = (
            eff_ms in ("vega", "ab")
            and cat_ms in ("vega", "ab", "unknown", "mixed")
            and eff_ms != cat_ms
        )
        if fs_change:
            needs = True
            note = f"filter_set {cal_fs}→{eff_fs}"
        if ms_change and not (eff_fs == "sdss" and eff_ms == "ab" and fs_change):
            # ZP change in addition to filter-set (SDSS path forces AB)
            needs = True
            note = (note + "; " if note else "") + f"magnitude_system {cat_ms}→{eff_ms}"
        if not needs and (ofs != "auto" or oms != "auto"):
            # User asked for explicit output matching current → still "convert" only if differs
            if eff_fs == cal_fs and eff_ms == cat_ms:
                needs = False
            elif cat_ms == "unknown" and oms != "auto":
                needs = True
                note = f"magnitude_system unknown→{eff_ms}"
    else:
        if ofs != "auto" or oms != "auto":
            if (ofs != "auto" and ofs != cal_fs) or (
                oms != "auto" and oms != cat_ms and cat_ms != "unknown"
            ):
                warnings.warn(
                    "output_filter_set / output_magnitude_system differ from the "
                    "calibrated catalog system, but convert_magnitudes=False; "
                    "outputs stay in the catalog/calibrated system. Set "
                    "convert_magnitudes=True to convert.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )

    return EffectiveMagnitudeOutput(
        filter_set=eff_fs,
        magnitude_system=eff_ms,
        catalog_magnitude_system=cat_ms,
        calibrated_filter_set=cal_fs,
        needs_convert=bool(convert_magnitudes and needs),
        conversion_note=note,
    )


def annotate_table_magnitude_meta(
    tbl: Table,
    *,
    filter_set: str,
    magnitude_system: str,
    catalog_magnitude_system: str | None = None,
    calibration_source: str | None = None,
    conversion_note: str | None = None,
) -> Table:
    """Write ost_photometry magnitude-system keys into ``tbl.meta``."""
    tbl.meta[META_FILTER_SET] = filter_set
    tbl.meta[META_MAGNITUDE_SYSTEM] = magnitude_system
    if catalog_magnitude_system is not None:
        tbl.meta[META_CATALOG_SYSTEM] = catalog_magnitude_system
    if calibration_source is not None:
        tbl.meta[META_CATALOG_SOURCE] = calibration_source
    if conversion_note:
        tbl.meta[META_CONVERSION_NOTE] = conversion_note
    return tbl


def table_magnitude_system(tbl: Table | None) -> str:
    """Read magnitude system from table meta (default ``vega`` for labels)."""
    if tbl is None:
        return "vega"
    return str(tbl.meta.get(META_MAGNITUDE_SYSTEM, "vega"))


def filter_expects_catalog_standards(filter_: str) -> bool:
    """True for Bessell UBVRI / SDSS ugriz; False for Clear, luminance, white, …"""
    key = normalize_filter_name(filter_).strip()
    if key.lower() in UNCALIBRATED_BROADBAND_ALIASES:
        return False
    return key in BESSELL_FILTERS or key in SDSS_FILTERS or str(filter_).strip() in SDSS_FILTERS


def _catalog_std_column_is_usable(
    standard_table: Table,
    filter_: str,
    *,
    std_prefix: str = "mag_std_",
) -> bool:
    import numpy as np

    col = f"{std_prefix}{filter_}"
    if col not in standard_table.colnames:
        return False
    vals = np.asarray(standard_table[col], dtype=float)
    return bool(np.any(np.isfinite(vals)))


def partition_catalog_fit_filters(
    filters: Sequence[str],
    standard_table: Table | None,
    *,
    std_prefix: str = "mag_std_",
) -> tuple[list[str], list[str]]:
    """
    Split ``filters`` into those with a usable ``mag_std_*`` column and the rest.

    A custom catalog that *does* provide ``mag_std_Clear`` still counts as covered.
    """
    covered: list[str] = []
    missing: list[str] = []
    if standard_table is None or len(standard_table) == 0:
        return [], [str(f) for f in filters]
    for f in filters:
        name = str(f)
        if _catalog_std_column_is_usable(
            standard_table, name, std_prefix=std_prefix
        ):
            covered.append(name)
        else:
            missing.append(name)
    return covered, missing


def require_catalog_bands_for_filters(
    standard_table: Table,
    filters: Sequence[str],
    *,
    std_prefix: str = "mag_std_",
) -> None:
    """
    Abort if any calibration filter lacks a matching ``mag_std_<F>`` column
    with at least one finite value.
    """
    missing: list[str] = []
    empty: list[str] = []
    import numpy as np

    for f in filters:
        col = f"{std_prefix}{f}"
        if col not in standard_table.colnames:
            missing.append(f)
            continue
        vals = np.asarray(standard_table[col], dtype=float)
        if not np.any(np.isfinite(vals)):
            empty.append(f)
    if missing or empty:
        parts = []
        if missing:
            parts.append(f"missing columns for filters {missing}")
        if empty:
            parts.append(f"no finite standards for filters {empty}")
        raise ValueError(
            "Calibration catalog does not cover all instrumental filters "
            f"({'; '.join(parts)}). Use a catalog with matching bands, or enable "
            "Sloan→Johnson R/I synthesis when appropriate."
        )


def apply_target_filter_system_alias(
    target_filter_system: str | None,
) -> tuple[OutputFilterSet | None, OutputMagnitudeSystem | None]:
    """
    Map deprecated ``target_filter_system`` onto new output_* fields.

    Returns ``(output_filter_set, output_magnitude_system)`` overrides (``None`` =
    leave unchanged).
    """
    if not target_filter_system:
        return None, None
    t = str(target_filter_system).strip().upper()
    if t == "SDSS":
        return "sdss", "ab"
    if t == "AB":
        return None, "ab"
    if t in ("BESSELL", "VEGA"):
        return "bessell", "vega"
    raise ValueError(
        f"target_filter_system={target_filter_system!r} is not supported; "
        "use output_filter_set ('auto'|'bessell'|'sdss') and "
        "output_magnitude_system ('auto'|'vega'|'ab'), or legacy "
        "target_filter_system in {'SDSS','AB','BESSELL'}."
    )


def log_magnitude_output(effective: EffectiveMagnitudeOutput, calibration_source: str | None) -> None:
    """Print a short terminal note about calibrated/output systems."""
    label = format_magnitude_output_label(
        effective.filter_set,
        effective.magnitude_system,
        catalog_source=calibration_source,
    )
    extra = ""
    if effective.needs_convert and effective.conversion_note:
        extra = f" [converting: {effective.conversion_note}]"
    elif not effective.needs_convert:
        extra = " [no conversion]"
    terminal_output.print_to_terminal(
        f"Magnitude output: {label}{extra}",
        style_name="INFO",
    )


__all__ = [
    "BESSELL_FILTERS",
    "CATALOG_MAGNITUDE_SYSTEMS",
    "EffectiveMagnitudeOutput",
    "FilterSet",
    "META_CATALOG_SOURCE",
    "META_CATALOG_SYSTEM",
    "META_CONVERSION_NOTE",
    "META_FILTER_SET",
    "META_MAGNITUDE_SYSTEM",
    "MagnitudeSystem",
    "OutputFilterSet",
    "OutputMagnitudeSystem",
    "SDSS_FILTERS",
    "UNCALIBRATED_BROADBAND_ALIASES",
    "VEGA_TO_AB_OFFSET",
    "annotate_table_magnitude_meta",
    "apply_target_filter_system_alias",
    "filter_expects_catalog_standards",
    "format_magnitude_output_label",
    "infer_filter_set",
    "log_magnitude_output",
    "magnitude_system_axis_suffix",
    "normalize_filter_name",
    "partition_catalog_fit_filters",
    "require_catalog_bands_for_filters",
    "resolve_catalog_magnitude_system",
    "resolve_effective_output",
    "table_magnitude_system",
    "validate_magnitude_output_request",
    "vega_to_ab_offset",
]
