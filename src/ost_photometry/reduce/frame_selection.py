"""Frame-quality selection, reference-frame choice, and stack weights.

This module works on the per-frame quality table produced by
:mod:`ost_photometry.reduce.quality` and stays free of ``ccdproc`` /
``photutils`` imports so it can be unit-tested (and used by scripts) without
the heavy reduction stack. Rows are frames, grouped by ``filter``; every
selection criterion is evaluated per filter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from astropy.table import Table

from .. import terminal_output

STACK_WEIGHTING: dict[str, str] = {
    "none": "equal weights",
    "fwhm": "w = (median(FWHM) / FWHM)^2 per filter",
    "n_stars": "w = N_stars / median(N_stars) per filter",
    "noise": "w = (median(BKGRMS) / BKGRMS)^2 per filter",
}
SUPPORTED_STACK_WEIGHTING = tuple(STACK_WEIGHTING)

REFERENCE_SELECTION: dict[str, str] = {
    "best_fwhm": "sharpest kept frame (per filter; one global frame with shift_all)",
    "first": "first frame after sorting by observation time (index 0)",
}
SUPPORTED_REFERENCE_SELECTION = tuple(REFERENCE_SELECTION)

FRAME_QUALITY_STATUS = ("ok", "no_stars", "fwhm_failed")
RANK_KEYS = ("fwhm_px", "fwhm_weighted")
FWHM_UNITS = ("px", "arcsec")

#: Global reference key used by :func:`resolve_reference_frames` when one
#: frame serves every filter (``shift_all=True``).
GLOBAL_REFERENCE_KEY = "__all__"

#: Weight bounds after normalisation; keeps one exceptional frame from
#: dominating (or vanishing from) the stack.
WEIGHT_CLIP = (0.1, 10.0)

#: Canonical quality-table columns: ``(name, kind, default)``. ``kind`` is
#: ``str`` / ``float`` / ``int`` / ``bool``. String columns are rebuilt on
#: every assignment (see :func:`set_string_column`) so fixed-width numpy
#: unicode dtypes never truncate a reason string.
QUALITY_COLUMNS: tuple[tuple[str, type, object], ...] = (
    ("file", str, ""),
    ("filter", str, ""),
    ("imagetyp", str, ""),
    ("date_obs", str, ""),
    ("jd", float, np.nan),
    ("exptime", float, np.nan),
    ("airmass", float, np.nan),
    ("fwhm_px", float, np.nan),
    ("fwhm_err", float, np.nan),
    ("fwhm_arcsec", float, np.nan),
    ("pixel_scale", float, np.nan),
    ("fwhm_weighted", float, np.nan),
    ("fwhm_source", str, ""),
    ("n_fwhm_stars", int, 0),
    ("n_stars", int, 0),
    ("roundness", float, np.nan),
    ("sharpness", float, np.nan),
    ("background", float, np.nan),
    ("background_rms", float, np.nan),
    ("masked_fraction", float, np.nan),
    ("status", str, "ok"),
    ("rejected", bool, False),
    ("reject_reason", str, ""),
    ("is_reference", bool, False),
    ("aligned", bool, False),
    ("align_note", str, ""),
    ("stack_weight", float, 1.0),
)

_NUMPY_KIND = {str: str, float: np.float64, int: np.int64, bool: np.bool_}


@dataclass(frozen=True)
class FrameSelection:
    """Frame rejection criteria; every set criterion must pass (logical AND).

    ``None`` disables a criterion. Criteria are evaluated per filter.
    ``best_fraction`` is applied after the threshold cuts to the surviving
    frames, ranked by ``rank_by``. ``min_frames`` is a floor: if fewer frames
    would remain in a filter, the best-ranked rejected frames are restored.
    """

    fwhm_max: float | None = None
    fwhm_unit: str = "px"
    best_fraction: float | None = None
    fwhm_sigma_clip: float | None = None
    rank_by: str = "fwhm_px"
    roundness_max: float | None = None
    n_stars_min: int | None = None
    background_max: float | None = None
    masked_fraction_max: float | None = None
    min_frames: int = 1
    reject_no_stars: bool = True

    def __post_init__(self) -> None:
        if self.fwhm_unit not in FWHM_UNITS:
            raise ValueError(f"fwhm_unit must be one of {FWHM_UNITS}, got {self.fwhm_unit!r}")
        if self.rank_by not in RANK_KEYS:
            raise ValueError(f"rank_by must be one of {RANK_KEYS}, got {self.rank_by!r}")
        if self.best_fraction is not None and not (0.0 < float(self.best_fraction) <= 1.0):
            raise ValueError(f"best_fraction must be in (0, 1], got {self.best_fraction!r}")
        if self.fwhm_sigma_clip is not None and float(self.fwhm_sigma_clip) <= 0.0:
            raise ValueError("fwhm_sigma_clip must be > 0")
        if int(self.min_frames) < 1:
            raise ValueError("min_frames must be >= 1")
        for name in ("fwhm_max", "roundness_max", "background_max", "masked_fraction_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.n_stars_min is not None and int(self.n_stars_min) < 0:
            raise ValueError("n_stars_min must be >= 0")

    @classmethod
    def from_mapping(cls, value: FrameSelection | Mapping[str, object] | None) -> FrameSelection:
        """Build from ``None`` (inactive), a mapping, or an existing instance."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"frame_selection must be a mapping or FrameSelection, got {type(value).__name__}"
            )
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown frame_selection keys {unknown}; allowed: {sorted(allowed)}"
            )
        return cls(**dict(value))

    def is_active(self) -> bool:
        """True if any rejection criterion is set (``reject_no_stars`` alone is not)."""
        return any(
            getattr(self, name) is not None
            for name in (
                "fwhm_max",
                "best_fraction",
                "fwhm_sigma_clip",
                "roundness_max",
                "n_stars_min",
                "background_max",
                "masked_fraction_max",
            )
        )

    def describe(self) -> str:
        """Short human-readable summary of the active criteria."""
        parts: list[str] = []
        if self.fwhm_max is not None:
            parts.append(f"fwhm <= {float(self.fwhm_max):g} {self.fwhm_unit}")
        if self.fwhm_sigma_clip is not None:
            parts.append(f"fwhm <= median + {float(self.fwhm_sigma_clip):g} MAD")
        if self.roundness_max is not None:
            parts.append(f"roundness <= {float(self.roundness_max):g}")
        if self.n_stars_min is not None:
            parts.append(f"n_stars >= {int(self.n_stars_min)}")
        if self.background_max is not None:
            parts.append(f"background <= {float(self.background_max):g}")
        if self.masked_fraction_max is not None:
            parts.append(f"masked_fraction <= {float(self.masked_fraction_max):g}")
        if self.best_fraction is not None:
            parts.append(f"best {float(self.best_fraction):.0%} by {self.rank_by}")
        if not parts:
            return "no frame selection"
        if self.min_frames > 1:
            parts.append(f"min {int(self.min_frames)} frames per filter")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Table construction helpers
# ---------------------------------------------------------------------------


def empty_quality_table() -> Table:
    """Zero-row table with the canonical quality columns and dtypes."""
    table = Table()
    for name, kind, _default in QUALITY_COLUMNS:
        table[name] = np.zeros(0, dtype=_NUMPY_KIND[kind])
    return table


def set_string_column(table: Table, name: str, values: Iterable[object]) -> None:
    """Replace a string column so the unicode width fits the new values."""
    table[name] = np.array([str(v) if v is not None else "" for v in values], dtype=str)


def quality_table_from_rows(rows: Iterable[Mapping[str, object]]) -> Table:
    """Build the quality table from per-frame dicts; missing keys get defaults.

    Rows are sorted by ``(filter, jd)`` and ``fwhm_weighted`` is filled.
    """
    rows = list(rows)
    if not rows:
        return empty_quality_table()
    table = Table()
    for name, kind, default in QUALITY_COLUMNS:
        values = [row.get(name, default) for row in rows]
        if kind is str:
            set_string_column(table, name, values)
        elif kind is bool:
            table[name] = np.array([bool(v) for v in values], dtype=np.bool_)
        elif kind is int:
            table[name] = np.array(
                [int(v) if v is not None else default for v in values], dtype=np.int64
            )
        else:
            table[name] = np.array(
                [float(v) if v is not None else np.nan for v in values], dtype=np.float64
            )
    order = np.lexsort((np.nan_to_num(table["jd"], nan=np.inf), table["filter"]))
    table = table[order]
    compute_fwhm_weighted(table)
    return table


def compute_fwhm_weighted(table: Table) -> None:
    """Siril-like wFWHM: ``fwhm_px * max(n_stars in filter) / n_stars``.

    Penalises frames that show fewer stars than the best frame of the same
    filter (thin clouds, poor transparency). ``nan`` where FWHM or the star
    count is unusable.
    """
    if len(table) == 0:
        return
    fwhm = np.asarray(table["fwhm_px"], dtype=float)
    n_stars = np.asarray(table["n_stars"], dtype=float)
    weighted = np.full(len(table), np.nan)
    for _filt, idx in group_indices_by_filter(table).items():
        n_max = np.max(n_stars[idx]) if idx.size else 0.0
        if n_max <= 0:
            continue
        ok = np.isfinite(fwhm[idx]) & (n_stars[idx] > 0)
        sub = idx[ok]
        weighted[sub] = fwhm[sub] * n_max / n_stars[sub]
    table["fwhm_weighted"] = weighted


def group_indices_by_filter(table: Table) -> dict[str, np.ndarray]:
    """Row indices per filter, in order of first appearance."""
    filters = [str(f) for f in table["filter"]] if len(table) else []
    groups: dict[str, list[int]] = {}
    for i, filt in enumerate(filters):
        groups.setdefault(filt, []).append(i)
    return {filt: np.asarray(idx, dtype=int) for filt, idx in groups.items()}


def rank_frames(table: Table, idx: np.ndarray, key: str = "fwhm_px") -> np.ndarray:
    """Order ``idx`` by ``key`` ascending (nan last), ties by ``n_stars`` descending."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return idx
    if key not in RANK_KEYS:
        raise ValueError(f"rank key must be one of {RANK_KEYS}, got {key!r}")
    values = np.asarray(table[key], dtype=float)[idx]
    values = np.where(np.isfinite(values), values, np.inf)
    n_stars = np.asarray(table["n_stars"], dtype=float)[idx]
    order = np.lexsort((-n_stars, values))
    return idx[order]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _fwhm_for_unit(table: Table, unit: str) -> np.ndarray:
    column = "fwhm_px" if unit == "px" else "fwhm_arcsec"
    return np.asarray(table[column], dtype=float)


def select_frames(
    table: Table,
    selection: FrameSelection,
    *,
    indent: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate ``selection`` per filter.

    Returns ``(keep_mask, reasons)``; ``reasons[i]`` is ``""`` for kept frames
    and a ``;``-joined list of failed criteria otherwise. The table itself is
    not modified (see :func:`mark_selection`).
    """
    n = len(table)
    keep = np.ones(n, dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(n)]
    if n == 0 or not selection.is_active():
        return keep, [""] * n

    status = np.asarray([str(s) for s in table["status"]])
    fwhm = _fwhm_for_unit(table, selection.fwhm_unit)
    roundness = np.asarray(table["roundness"], dtype=float)
    n_stars = np.asarray(table["n_stars"], dtype=float)
    background = np.asarray(table["background"], dtype=float)
    masked = np.asarray(table["masked_fraction"], dtype=float)

    for filt, idx in group_indices_by_filter(table).items():
        if selection.reject_no_stars:
            for i in idx[status[idx] != "ok"]:
                reasons[i].append(str(status[i]))

        if selection.fwhm_max is not None:
            limit = float(selection.fwhm_max)
            finite = np.isfinite(fwhm[idx])
            if not np.any(finite):
                terminal_output.print_to_terminal(
                    f"Filter {filt}: no usable FWHM in {selection.fwhm_unit}; "
                    "fwhm_max not applied.",
                    indent=indent,
                    style_name="WARNING",
                )
            for i in idx[finite & (fwhm[idx] > limit)]:
                reasons[i].append(f"fwhm {fwhm[i]:.2f} > {limit:.2f} {selection.fwhm_unit}")

        if selection.fwhm_sigma_clip is not None:
            finite = np.isfinite(fwhm[idx])
            vals = fwhm[idx][finite]
            if vals.size >= 3:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                limit = med + float(selection.fwhm_sigma_clip) * mad
                if mad > 0:
                    for i in idx[finite & (fwhm[idx] > limit)]:
                        reasons[i].append(
                            f"fwhm {fwhm[i]:.2f} > median+{selection.fwhm_sigma_clip:g}MAD "
                            f"({limit:.2f} {selection.fwhm_unit})"
                        )

        if selection.roundness_max is not None:
            limit = float(selection.roundness_max)
            for i in idx[np.isfinite(roundness[idx]) & (np.abs(roundness[idx]) > limit)]:
                reasons[i].append(f"roundness {roundness[i]:.2f} > {limit:.2f}")

        if selection.n_stars_min is not None:
            limit = int(selection.n_stars_min)
            for i in idx[n_stars[idx] < limit]:
                reasons[i].append(f"n_stars {int(n_stars[i])} < {limit}")

        if selection.background_max is not None:
            limit = float(selection.background_max)
            for i in idx[np.isfinite(background[idx]) & (background[idx] > limit)]:
                reasons[i].append(f"background {background[i]:.3g} > {limit:.3g}")

        if selection.masked_fraction_max is not None:
            limit = float(selection.masked_fraction_max)
            for i in idx[np.isfinite(masked[idx]) & (masked[idx] > limit)]:
                reasons[i].append(f"masked_fraction {masked[i]:.3f} > {limit:.3f}")

        for i in idx:
            if reasons[i]:
                keep[i] = False

        if selection.best_fraction is not None:
            survivors = idx[keep[idx]]
            if survivors.size:
                n_keep = int(math.ceil(float(selection.best_fraction) * survivors.size))
                ranked = rank_frames(table, survivors, key=selection.rank_by)
                for i in ranked[n_keep:]:
                    keep[i] = False
                    reasons[i].append(
                        f"not in best {float(selection.best_fraction):.0%} by {selection.rank_by}"
                    )

        floor = min(int(selection.min_frames), int(idx.size))
        n_kept = int(np.count_nonzero(keep[idx]))
        if n_kept < floor:
            rejected = idx[~keep[idx]]
            ranked = rank_frames(table, rejected, key=selection.rank_by)
            restore = ranked[: floor - n_kept]
            for i in restore:
                keep[i] = True
                reasons[i] = []
            terminal_output.print_to_terminal(
                f"Filter {filt}: only {n_kept} of {idx.size} frames pass the selection; "
                f"restored the best {restore.size} rejected frame(s) to reach "
                f"min_frames={int(selection.min_frames)}.",
                indent=indent,
                style_name="WARNING",
            )

    return keep, ["; ".join(r) for r in reasons]


def mark_selection(table: Table, selection: FrameSelection, *, indent: int = 2) -> Table:
    """Apply :func:`select_frames` in place: sets ``rejected`` / ``reject_reason``."""
    keep, reasons = select_frames(table, selection, indent=indent)
    table["rejected"] = ~keep
    set_string_column(table, "reject_reason", reasons)
    return table


def resolve_reference_frames(
    table: Table,
    *,
    per_filter: bool = True,
    indent: int = 2,
) -> dict[str, str]:
    """Sharpest kept frame per filter (or one global frame).

    Returns ``{filter: file}`` or ``{GLOBAL_REFERENCE_KEY: file}``. Filters
    without a finite FWHM are omitted; the caller falls back to index 0.
    """
    if len(table) == 0:
        return {}
    kept = ~np.asarray(table["rejected"], dtype=bool)
    fwhm = np.asarray(table["fwhm_px"], dtype=float)
    files = [str(f) for f in table["file"]]
    groups = group_indices_by_filter(table) if per_filter else {
        GLOBAL_REFERENCE_KEY: np.arange(len(table))
    }
    result: dict[str, str] = {}
    for key, idx in groups.items():
        candidates = idx[kept[idx] & np.isfinite(fwhm[idx])]
        if candidates.size == 0:
            label = "all frames" if key == GLOBAL_REFERENCE_KEY else f"filter {key}"
            terminal_output.print_to_terminal(
                f"No frame with a usable FWHM among {label}; "
                "reference falls back to the first frame.",
                indent=indent,
                style_name="WARNING",
            )
            continue
        best = int(rank_frames(table, candidates, key="fwhm_px")[0])
        result[key] = files[best]
    return result


def mark_reference_frames(table: Table, references: Mapping[str, str]) -> None:
    """Set ``is_reference`` for the files named in ``references``."""
    names = set(references.values())
    table["is_reference"] = np.array([str(f) in names for f in table["file"]], dtype=bool)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def stack_weights(table: Table, method: str, *, indent: int = 2) -> np.ndarray:
    """Per-frame stack weights for the kept frames of every filter.

    Weights are relative within a filter, clipped to :data:`WEIGHT_CLIP`, and
    normalised to a mean of one over the kept frames. Frames whose metric is
    not finite get the median weight. Rejected frames receive ``nan``.
    """
    if method not in STACK_WEIGHTING:
        raise ValueError(
            f"stack_weighting must be one of {SUPPORTED_STACK_WEIGHTING}, got {method!r}"
        )
    n = len(table)
    weights = np.ones(n, dtype=float)
    if n == 0:
        return weights
    rejected = np.asarray(table["rejected"], dtype=bool)
    weights[rejected] = np.nan
    if method == "none":
        return weights

    column = {"fwhm": "fwhm_px", "n_stars": "n_stars", "noise": "background_rms"}[method]
    metric = np.asarray(table[column], dtype=float)

    for filt, idx in group_indices_by_filter(table).items():
        kept = idx[~rejected[idx]]
        if kept.size == 0:
            continue
        vals = metric[kept]
        finite = np.isfinite(vals) & (vals > 0)
        if not np.any(finite):
            terminal_output.print_to_terminal(
                f"Filter {filt}: no usable '{column}' values; equal weights used.",
                indent=indent,
                style_name="WARNING",
            )
            weights[kept] = 1.0
            continue
        med = float(np.median(vals[finite]))
        if method == "n_stars":
            raw = vals / med
        else:
            raw = (med / vals) ** 2
        raw = np.where(finite, raw, 1.0)
        raw = np.clip(raw, WEIGHT_CLIP[0], WEIGHT_CLIP[1])
        weights[kept] = raw / float(np.mean(raw))
        if np.count_nonzero(~finite):
            terminal_output.print_to_terminal(
                f"Filter {filt}: {int(np.count_nonzero(~finite))} frame(s) without "
                f"'{column}' got the median weight.",
                indent=indent,
                style_name="WARNING",
            )
    return weights


# ---------------------------------------------------------------------------
# Alignment bookkeeping and summaries
# ---------------------------------------------------------------------------


def merge_alignment_result(table: Table, result: object) -> None:
    """Fill ``aligned`` / ``align_note`` from an alignment result object.

    ``result`` must provide ``aligned_files()`` (iterable of basenames) and
    ``skipped_files()`` (iterable of ``(basename, note)``); see
    :class:`ost_photometry.reduce.registration.accounting.AlignmentResult`.
    """
    if len(table) == 0:
        return
    aligned_names = {Path(str(f)).name for f in result.aligned_files()}
    skipped = {Path(str(f)).name: str(note) for f, note in result.skipped_files()}
    files = [Path(str(f)).name for f in table["file"]]
    rejected = np.asarray(table["rejected"], dtype=bool)
    aligned = np.array([f in aligned_names for f in files], dtype=bool)
    notes: list[str] = []
    for f, rej in zip(files, rejected, strict=True):
        if f in skipped:
            notes.append(skipped[f])
        elif f in aligned_names or rej:
            notes.append("")
        else:
            notes.append("not processed")
    table["aligned"] = aligned
    set_string_column(table, "align_note", notes)


def _fmt_fwhm(value: float, arcsec: float | None = None) -> str:
    if not np.isfinite(value):
        return "n/a"
    text = f"{value:.2f} px"
    if arcsec is not None and np.isfinite(arcsec):
        text += f' ({arcsec:.2f}")'
    return text


def summarize_selection(table: Table, *, indent: int = 2, max_listed: int = 8) -> None:
    """Print the per-filter frame-quality summary."""
    if len(table) == 0:
        terminal_output.print_to_terminal("Frame quality: no frames.", indent=indent)
        return
    fwhm = np.asarray(table["fwhm_px"], dtype=float)
    fwhm_as = np.asarray(table["fwhm_arcsec"], dtype=float)
    rejected = np.asarray(table["rejected"], dtype=bool)
    is_ref = np.asarray(table["is_reference"], dtype=bool)
    files = [str(f) for f in table["file"]]
    reasons = [str(r) for r in table["reject_reason"]]

    for filt, idx in group_indices_by_filter(table).items():
        finite = idx[np.isfinite(fwhm[idx])]
        if finite.size:
            med = float(np.median(fwhm[finite]))
            med_as = float(np.median(fwhm_as[finite])) if np.any(np.isfinite(fwhm_as[finite])) else None
            rng = f", range {np.min(fwhm[finite]):.1f}-{np.max(fwhm[finite]):.1f} px"
            fwhm_text = f"FWHM median {_fmt_fwhm(med, med_as)}{rng}"
        else:
            fwhm_text = "FWHM unavailable"
        terminal_output.print_to_terminal(
            f"Frame quality (filter {filt}): {idx.size} frames, {fwhm_text}",
            indent=indent,
        )
        rej = idx[rejected[idx]]
        kept = idx.size - rej.size
        if rej.size:
            listed = ", ".join(f"{files[i]} ({reasons[i]})" for i in rej[:max_listed])
            more = f", … +{rej.size - max_listed}" if rej.size > max_listed else ""
            terminal_output.print_to_terminal(
                f"kept {kept} of {idx.size}; rejected: {listed}{more}",
                indent=indent + 1,
                style_name="WARNING",
            )
        else:
            terminal_output.print_to_terminal(
                f"kept {kept} of {idx.size}; nothing rejected",
                indent=indent + 1,
            )
        refs = idx[is_ref[idx]]
        for i in refs:
            terminal_output.print_to_terminal(
                f"reference: {files[i]} (FWHM {_fmt_fwhm(fwhm[i], fwhm_as[i])})",
                indent=indent + 1,
            )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_quality_table(table: Table, path: str | Path) -> Path:
    """Write the quality table as ECSV (parent directory is created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return path


def read_quality_table(path: str | Path) -> Table:
    """Read a quality table written by :func:`write_quality_table`.

    Missing canonical columns are added with their defaults so tables from
    older runs stay usable.
    """
    table = Table.read(Path(path), format="ascii.ecsv")
    for name, kind, default in QUALITY_COLUMNS:
        if name in table.colnames:
            if kind is str:
                values = [
                    "" if (v is np.ma.masked or v is None) else str(v) for v in table[name]
                ]
                set_string_column(table, name, values)
            elif kind is bool:
                column = table[name]
                if hasattr(column, "filled"):
                    column = column.filled(False)
                table[name] = np.asarray(column, dtype=bool)
            continue
        if kind is str:
            set_string_column(table, name, [default] * len(table))
        else:
            table[name] = np.full(len(table), default, dtype=_NUMPY_KIND[kind])
    return table


__all__ = [
    "FRAME_QUALITY_STATUS",
    "FWHM_UNITS",
    "GLOBAL_REFERENCE_KEY",
    "QUALITY_COLUMNS",
    "RANK_KEYS",
    "REFERENCE_SELECTION",
    "STACK_WEIGHTING",
    "SUPPORTED_REFERENCE_SELECTION",
    "SUPPORTED_STACK_WEIGHTING",
    "WEIGHT_CLIP",
    "FrameSelection",
    "compute_fwhm_weighted",
    "empty_quality_table",
    "group_indices_by_filter",
    "mark_reference_frames",
    "mark_selection",
    "merge_alignment_result",
    "quality_table_from_rows",
    "rank_frames",
    "read_quality_table",
    "resolve_reference_frames",
    "select_frames",
    "set_string_column",
    "stack_weights",
    "summarize_selection",
    "write_quality_table",
]
