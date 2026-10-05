"""Calibration plan: from a manifest to reduction units, masters and targets.

:func:`build_calibration_plan` runs the whole grouping:

1. classify frame types (image statistics, header, archive),
2. electronic setups (bias / dark key) and nights,
3. preliminary targets from header pointings,
4. camera orientation of sampled lights (server WCS or local ASTAP),
5. final targets from solved field centres,
6. mount sessions,
7. bias / dark choice per setup and night,
8. flat sets and their probability per session,
9. masters and reduction units (session x electronic setup).

The result is written as ``calibration_groups.ecsv`` (one row per frame)
and ``calibration_plan.yaml`` (human-editable). Manual corrections go into
the ``overrides`` block of the YAML; they are read back when the grouping is
rebuilt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from ...wcs import solve_astap_copy
from .classify import BIAS, DARK, FLAT, LIGHT, SPECTROSCOPY, UNKNOWN, classify_frames
from .darks import (
    DarkAssignment,
    add_electronic_ids,
    assign_bias_dark,
    bias_levels,
    night_of,
)
from .flats import (
    CERTAIN,
    DEFAULT_FLAT_SETTINGS,
    LIKELY,
    REJECTED,
    UNCERTAIN,
    FlatAssignment,
    FlatSet,
    assign_flats,
    flat_sets,
)
from .orientation import (
    OrientationResult,
    orientation_table,
    refine_orientation_changes,
    sample_frames_for_solving,
    solve_orientations,
)
from .sessions import UNKNOWN_SESSION, Session, segment_sessions, sessions_table
from .setup_keys import binning, camera_id, filter_name, telescope_id
from .targets import UNKNOWN_TARGET, assign_targets, target_summary

PLAN_VERSION = 1


@dataclass
class PlanSettings:
    temp_tolerance: float = 2.0
    pa_tolerance: float = 0.5
    solve_interval_minutes: float = 30.0
    solve_gap_hours: float = 1.0
    solve_timeout: float = 120.0
    solve_radius_deg: float = 15.0
    solve_max_extra: int = 200
    target_overlap_fraction: float = 0.5
    target_grouping: str = "position"
    dark_exptime_tolerance: float = 0.5
    #: Relative part of the dark tolerance: max(absolute, fraction * exptime)
    dark_exptime_tolerance_fraction: float = 0.05
    calibration_window_days: float = 30.0
    no_flat_policy: str = "best_available"
    #: Lights are reduced only with complete calibration: darks for their
    #: exposure time and a flat of one of these categories (else blocked;
    #: release with the override ``force_units``)
    require_complete: bool = True
    accepted_flat_categories: list[str] = field(default_factory=lambda: ["certain", "likely"])
    n_cores_multiprocessing: int | None = None
    flat: dict = field(default_factory=dict)
    classify: dict = field(default_factory=dict)


@dataclass
class Overrides:
    exclude_frames: list[str] = field(default_factory=list)
    frame_types: dict[str, str] = field(default_factory=dict)
    session_breaks: list[str] = field(default_factory=list)
    merge_sessions: list[list[str]] = field(default_factory=list)
    force_flats: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    merge_targets: list[list[str]] = field(default_factory=list)
    rename_targets: dict[str, str] = field(default_factory=dict)
    no_stack_targets: list[str] = field(default_factory=list)
    #: Units reduced although their calibration is incomplete
    force_units: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping | None) -> Overrides:
        data = dict(data or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown override keys {unknown}; allowed: {sorted(known)}")
        out = cls()
        for key in known:
            if data.get(key) is not None:
                setattr(out, key, data[key])
        out.exclude_frames = [str(v) for v in out.exclude_frames]
        out.session_breaks = [str(v) for v in out.session_breaks]
        out.frame_types = {str(k): str(v) for k, v in dict(out.frame_types).items()}
        out.no_stack_targets = [str(v) for v in out.no_stack_targets]
        out.force_units = [str(v) for v in out.force_units]
        return out


@dataclass
class MasterSpec:
    master_id: str
    kind: str
    electronic_id: str = ""
    night: str = ""
    frame_ids: list[str] = field(default_factory=list)
    exptimes: list[float] = field(default_factory=list)
    filter: str = ""
    camera: str = ""
    binning: str = ""
    flat_set_ids: list[str] = field(default_factory=list)
    bias_id: str = ""
    dark_id: str = ""
    probability: float = float("nan")
    category: str = ""
    note: str = ""
    #: Calibration the master itself lacks (e.g. darks for the flat exposures)
    missing: list[str] = field(default_factory=list)
    #: Exposure times (s) of the master's frames without matching darks
    missing_exptimes: list[float] = field(default_factory=list)


@dataclass
class ReductionUnit:
    unit_id: str
    session_id: str
    electronic_id: str
    camera: str
    telescope: str
    bias_id: str = ""
    dark_id: str = ""
    flats: dict[str, str] = field(default_factory=dict)
    light_ids: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Incomplete calibration: lights of these filters / exposure times
    #: (``"300"``) are not reduced unless the unit is forced
    blocked_filters: dict[str, str] = field(default_factory=dict)
    blocked_exptimes: dict[str, str] = field(default_factory=dict)
    forced: bool = False
    #: Flat assignment of this unit's session per filter (a flat master can
    #: serve several sessions with different probabilities)
    flat_categories: dict[str, str] = field(default_factory=dict)
    flat_probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not (self.blocked_filters or self.blocked_exptimes):
            return "ready"
        if self.forced:
            return "forced"
        return "incomplete"


@dataclass
class CalibrationPlan:
    settings: PlanSettings
    overrides: Overrides
    frames: Table
    sessions: list[Session]
    flat_sets: list[FlatSet]
    flat_assignments: list[FlatAssignment]
    dark_assignments: list[DarkAssignment]
    masters: dict[str, MasterSpec]
    units: list[ReductionUnit]
    targets: Table
    orientations: dict[str, OrientationResult]
    report: list[str] = field(default_factory=list)


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _merge_sessions(table: Table, sessions: list[Session], groups: Sequence[Sequence[str]]
                    ) -> tuple[Table, list[Session]]:
    by_id = {s.session_id: s for s in sessions}
    for group in groups:
        members = [by_id[g] for g in group if g in by_id]
        if len(members) < 2:
            continue
        keep = min(members, key=lambda s: s.start_jd)
        for other in members:
            if other is keep:
                continue
            keep.frame_ids.extend(other.frame_ids)
            keep.start_jd = min(keep.start_jd, other.start_jd)
            keep.end_jd = max(keep.end_jd, other.end_jd)
            keep.n_solved += other.n_solved
            keep.evidence.append(f"merged with {other.session_id} (manual)")
            table["session_id"][np.asarray(table["session_id"]) == other.session_id] = keep.session_id
            sessions.remove(other)
            by_id.pop(other.session_id)
    return table, sessions


def build_calibration_plan(
    manifest: Table,
    *,
    work_dir: str | Path,
    settings: PlanSettings | None = None,
    overrides: Overrides | None = None,
    solver: Callable[..., object] = solve_astap_copy,
    log: Callable[[str], None] = print,
    progress: Callable[[int, int, str], None] | None = None,
) -> CalibrationPlan:
    """Run the complete grouping on ``manifest`` (see module docstring)."""
    settings = settings or PlanSettings()
    overrides = overrides or Overrides()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    # 1. Frame types
    frames = manifest.copy()
    excluded = set(overrides.exclude_frames)
    if "frame_type" not in frames.colnames:
        log("Classifying frames (image statistics)...")
        frames = classify_frames(frames, thresholds=settings.classify or None,
                                 n_cores_multiprocessing=settings.n_cores_multiprocessing)
    types = [str(t) for t in frames["frame_type"]]
    notes = [str(n) for n in frames["type_note"]]
    for i, fid in enumerate(str(f) for f in frames["frame_id"]):
        if fid in overrides.frame_types:
            types[i] = overrides.frame_types[fid]
            notes[i] = "manual override"
    frames["frame_type"] = np.array(types, dtype=str)
    frames["type_note"] = np.array(notes, dtype=str)
    frames["excluded"] = np.array([str(f) in excluded for f in frames["frame_id"]], dtype=bool)

    # 2. Electronic setups and nights
    frames = add_electronic_ids(frames, temp_tolerance=settings.temp_tolerance)
    frames["camera"] = np.array([camera_id(dict(zip(frames.colnames, r, strict=True)))
                                 for r in frames], dtype=str)
    roles = np.asarray(frames["role"]).astype(str)
    downloaded = np.asarray(frames["downloaded"], dtype=bool)
    kinds = np.asarray(frames["frame_type"]).astype(str)
    is_light = (kinds == LIGHT) & (roles != "context") & downloaded & ~np.asarray(frames["excluded"])
    lights = frames[is_light]
    context_mask = ~is_light & np.isin(kinds, [LIGHT, FLAT, SPECTROSCOPY])
    context = frames[context_mask]
    calibration = frames[downloaded & ~np.asarray(frames["excluded"]) & np.isin(kinds, [BIAS, DARK, FLAT])]
    log(f"{len(lights)} light frames, {len(calibration)} calibration frames, "
        f"{int(np.count_nonzero(kinds == SPECTROSCOPY))} spectroscopy frames (excluded)")

    # 3. Preliminary targets from header pointings
    prelim = assign_targets(lights, overlap_fraction=settings.target_overlap_fraction,
                            group_by=settings.target_grouping)
    labels = [str(t) for t in prelim["target_id"]]

    # 4. Orientation
    sample = sample_frames_for_solving(
        lights, interval_minutes=settings.solve_interval_minutes,
        gap_hours=settings.solve_gap_hours, target_labels=labels,
    )
    log(f"Plate-solving {len(sample)} sampled light frames (server WCS when available)...")
    cache_path = work_dir / "orientation_cache.ecsv"
    solve_dir = work_dir / "astap"
    orientations = solve_orientations(
        lights, sample, work_dir=solve_dir, cache_path=cache_path,
        timeout=settings.solve_timeout, solver=solver, progress=progress,
        radius_deg=settings.solve_radius_deg,
    )
    orientations = refine_orientation_changes(
        lights, orientations, work_dir=solve_dir, cache_path=cache_path,
        pa_tolerance=settings.pa_tolerance, timeout=settings.solve_timeout,
        solver=solver, max_extra=settings.solve_max_extra,
        radius_deg=settings.solve_radius_deg,
    )
    n_solved = sum(1 for r in orientations.values() if r.solved)
    report.append(f"Plate solutions: {n_solved} of {len(orientations)} solved frames")

    # 5. Final targets with solved centres
    centers = {k: (r.center_ra, r.center_dec) for k, r in orientations.items()
               if r.solved and math.isfinite(r.center_ra)}
    scales = {k: r.scale_arcsec for k, r in orientations.items() if r.solved}
    lights = assign_targets(
        lights, centers=centers, scales=scales,
        overlap_fraction=settings.target_overlap_fraction, group_by=settings.target_grouping,
        merge_targets=overrides.merge_targets, rename=overrides.rename_targets,
    )

    # 6. Sessions
    lights, sessions = segment_sessions(
        lights, orientations, context=context, pa_tolerance=settings.pa_tolerance,
        forced_breaks=set(overrides.session_breaks),
    )
    if overrides.merge_sessions:
        lights, sessions = _merge_sessions(lights, sessions, overrides.merge_sessions)

    # 7. Bias / darks for lights
    light_darks = assign_bias_dark(
        lights, calibration, exptime_tolerance=settings.dark_exptime_tolerance,
        exptime_tolerance_fraction=settings.dark_exptime_tolerance_fraction,
        window_days=settings.calibration_window_days,
    )

    # 8. Flats
    flat_settings = dict(DEFAULT_FLAT_SETTINGS)
    flat_settings.update(settings.flat or {})
    sets = flat_sets(calibration, block_gap_hours=flat_settings["block_gap_hours"])
    levels = bias_levels(calibration)
    flat_choice = assign_flats(
        lights, sessions, sets, context=context, bias_levels=levels,
        settings=flat_settings, no_flat_policy=settings.no_flat_policy,
    )
    set_by_id = {s.set_id: s for s in sets}
    for assignment in flat_choice:
        forced = overrides.force_flats.get(assignment.session_id, {}).get(assignment.filter)
        if forced:
            forced = [forced] if isinstance(forced, str) else list(forced)
            unknown = [f for f in forced if f not in set_by_id]
            if unknown:
                raise ValueError(f"force_flats: unknown flat set(s) {unknown}")
            assignment.flat_set_ids = forced
            assignment.probability = 1.0
            assignment.category = CERTAIN
            assignment.note = "manual override"

    # 9. Masters and units
    masters: dict[str, MasterSpec] = {}

    def master_for_set(entry, kind: str, bias_id: str = "") -> str:
        """Master for a bias / dark set. A dark master records the bias it is
        built with; light / flat reduction subtracts that same bias, so bias
        and dark are never applied inconsistently."""
        if entry is None:
            return ""
        mid = f"M{entry.set_id}_{entry.electronic_id.split('|')[0]}"
        if mid not in masters:
            masters[mid] = MasterSpec(mid, kind, entry.electronic_id, entry.night,
                                      list(entry.frame_ids), list(entry.exptimes),
                                      bias_id=bias_id)
        return mid

    def bias_and_darks(choice: DarkAssignment | None) -> tuple[str, str]:
        if choice is None:
            return "", ""
        bias_id = master_for_set(choice.bias, BIAS)
        dark_id = master_for_set(choice.darks, DARK, bias_id=bias_id)
        if dark_id:
            # A dark master shared by several consumers keeps the bias it was
            # first planned with; use exactly that bias for this consumer too.
            bias_id = masters[dark_id].bias_id
        return bias_id, dark_id

    flat_frames_rows = calibration[np.asarray(calibration["frame_type"]).astype(str) == FLAT]
    flat_darks = {(a.electronic_id, a.night): a for a in assign_bias_dark(
        flat_frames_rows, calibration, exptime_tolerance=settings.dark_exptime_tolerance,
        exptime_tolerance_fraction=settings.dark_exptime_tolerance_fraction,
        window_days=settings.calibration_window_days,
    )} if len(flat_frames_rows) else {}
    flat_exptime = {str(r["frame_id"]): _float(r["exptime"]) for r in flat_frames_rows}

    flat_master_ids: dict[tuple[str, ...], str] = {}

    def flat_master(assignment: FlatAssignment) -> str:
        key = tuple(sorted(assignment.flat_set_ids))
        if not key:
            return ""
        if key in flat_master_ids:
            return flat_master_ids[key]
        members = [set_by_id[k] for k in key]
        first = members[0]
        night = min(m.set_id.split("_")[1] for m in members)
        mid = f"MF_{night}_{first.camera}_{first.binning}_{first.filter}"
        suffix = 2
        while mid in masters:
            mid = f"MF_{night}_{first.camera}_{first.binning}_{first.filter}_{suffix}"
            suffix += 1
        dark_choice = flat_darks.get((first.electronic_id, night_of(first.start_jd)))
        frame_ids = [f for m in members for f in m.frame_ids]
        exptimes = sorted({round(flat_exptime[f], 3) for f in frame_ids
                           if math.isfinite(flat_exptime.get(f, math.nan))})
        missing: list[str] = []
        uncovered: list[float] = []
        if dark_choice is None or dark_choice.darks is None:
            missing.append("no darks for the flat exposures")
            uncovered = list(exptimes)
        else:
            uncovered = [e for e in dark_choice.missing_exptimes if round(e, 3) in exptimes]
            if uncovered:
                missing.append("no darks for flat exposure(s) "
                               + ", ".join(f"{e:g} s" for e in uncovered)
                               + ("" if dark_choice.bias else " and no bias for dark scaling"))
        masters[mid] = MasterSpec(
            mid, FLAT, first.electronic_id, night,
            frame_ids, exptimes, first.filter, first.camera,
            first.binning, list(key),
            *bias_and_darks(dark_choice),
            assignment.probability, assignment.category, assignment.note, missing, uncovered,
        )
        flat_master_ids[key] = mid
        return mid

    darks_by_key = {(a.electronic_id, a.night): a for a in light_darks}
    flats_by_key = {(a.session_id, a.filter, a.binning): a for a in flat_choice}
    units: dict[tuple[str, str], ReductionUnit] = {}
    unit_darks: dict[tuple[str, str], DarkAssignment | None] = {}
    unit_exptimes: dict[tuple[str, str], set[float]] = {}
    for row in lights:
        row = dict(zip(lights.colnames, row, strict=True))
        sid = str(row["session_id"])
        if sid == UNKNOWN_SESSION:
            continue
        key = (sid, str(row["electronic_id"]))
        unit = units.get(key)
        if unit is None:
            unit = ReductionUnit("", sid, key[1], camera_id(row), telescope_id(row))
            units[key] = unit
            dark_choice = darks_by_key.get((key[1], str(row["night"])))
            unit_darks[key] = dark_choice
            if dark_choice is not None:
                unit.bias_id, unit.dark_id = bias_and_darks(dark_choice)
                unit.notes.extend(dark_choice.notes)
        filt = filter_name(row)
        if filt not in unit.flats and filt not in unit.excluded:
            assignment = flats_by_key.get((sid, filt, binning(row)))
            mid = flat_master(assignment) if assignment is not None else ""
            if mid:
                unit.flats[filt] = mid
                unit.flat_categories[filt] = assignment.category
                unit.flat_probabilities[filt] = float(assignment.probability)
                if assignment.category in (UNCERTAIN, REJECTED):
                    unit.notes.append(f"filter {filt}: flat {assignment.category} "
                                      f"(p={assignment.probability:.2f}): {assignment.note}")
            elif settings.no_flat_policy == "exclude_lights":
                unit.excluded[filt] = assignment.note if assignment else "no flat"
            else:
                unit.flats[filt] = ""
                unit.notes.append(f"filter {filt}: no flat ("
                                  f"{assignment.note if assignment else 'no candidates'})")
        if filt not in unit.excluded:
            unit.light_ids.append(str(row["frame_id"]))
            if math.isfinite(_float(row["exptime"])):
                unit_exptimes.setdefault(key, set()).add(round(_float(row["exptime"]), 3))
    ordered = sorted(units.values(), key=lambda u: (u.session_id, u.electronic_id))
    for k, unit in enumerate(ordered, start=1):
        unit.unit_id = f"U{k:02d}_{unit.session_id}"
        key = (unit.session_id, unit.electronic_id)
        if settings.require_complete:
            _block_incomplete(unit, unit_darks.get(key), unit_exptimes.get(key, set()), masters,
                              set(settings.accepted_flat_categories),
                              allow_no_flat=settings.no_flat_policy == "skip_flat")
            unit.forced = unit.unit_id in set(overrides.force_units)

    # Frame table: grouping columns for every frame
    _annotate_frames(frames, lights, ordered, masters, sets, orientations)

    targets = target_summary(lights) if len(lights) else Table()
    if len(targets):
        no_stack = set(overrides.no_stack_targets)
        targets["stack"] = np.array(
            [not ({str(t), str(n)} & no_stack) for t, n in
             zip(targets["target_id"], targets["target_name"], strict=True)], dtype=bool)

    plan = CalibrationPlan(settings, overrides, frames, sessions, sets, flat_choice,
                           light_darks, masters, ordered, targets, orientations, report)
    plan.report.extend(_summary_lines(plan))
    return plan


def _annotate_frames(frames, lights, units, masters, sets, orientations) -> None:
    fid = [str(f) for f in frames["frame_id"]]
    index = {f: i for i, f in enumerate(fid)}
    n = len(frames)
    cols = {name: [""] * n for name in ("target_id", "target_name", "session_id",
                                        "unit_id", "flat_set_id", "master_ids")}
    pa = np.full(n, np.nan)
    for row in lights:
        i = index[str(row["frame_id"])]
        cols["target_id"][i] = str(row["target_id"])
        cols["target_name"][i] = str(row["target_name"])
        cols["session_id"][i] = str(row["session_id"])
    for unit in units:
        for f in unit.light_ids:
            cols["unit_id"][index[f]] = unit.unit_id
    for s in sets:
        for f in s.frame_ids:
            cols["flat_set_id"][index[f]] = s.set_id
    for master in masters.values():
        for f in master.frame_ids:
            i = index.get(f)
            if i is not None:
                cols["master_ids"][i] = ";".join(x for x in (cols["master_ids"][i],
                                                             master.master_id) if x)
    for key, result in orientations.items():
        if key in index and result.solved:
            pa[index[key]] = result.pa_mod180
    for name, values in cols.items():
        frames[name] = np.array(values, dtype=str)
    frames["pa_mod180"] = pa


def exptime_key(exptime: float) -> str:
    """Key of an exposure time in ``blocked_exptimes`` (``300``, ``1.5``)."""
    return f"{round(float(exptime), 3):g}"


def _block_incomplete(
    unit: ReductionUnit,
    darks: DarkAssignment | None,
    exptimes: set[float],
    masters: Mapping[str, MasterSpec],
    accepted_flat_categories: set[str],
    *,
    allow_no_flat: bool = False,
) -> None:
    """Record which lights of ``unit`` lack complete calibration.

    ``allow_no_flat`` (``no_flat_policy="skip_flat"``): lights without any
    flat are reduced without flat field instead of being blocked.
    """
    if not unit.dark_id:
        for e in sorted(exptimes):
            unit.blocked_exptimes[exptime_key(e)] = "no darks for this electronic setup"
    elif darks is not None:
        for e in darks.missing_exptimes:
            reason = "no darks with this exposure time"
            if darks.bias is None:
                reason += " and no bias for dark scaling"
            unit.blocked_exptimes[exptime_key(e)] = reason
    for filt, mid in sorted(unit.flats.items()):
        if not mid:
            if not allow_no_flat:
                unit.blocked_filters[filt] = "no flat"
            continue
        spec = masters[mid]
        category = unit.flat_categories.get(filt, spec.category)
        probability = unit.flat_probabilities.get(filt, spec.probability)
        if category not in accepted_flat_categories:
            unit.blocked_filters[filt] = (f"flat {mid} only {category} for this session "
                                          f"(p={probability:.2f})")
        elif spec.missing:
            unit.blocked_filters[filt] = f"flat {mid}: " + "; ".join(spec.missing)


def light_block_reason(
    row: Mapping[str, object],
    blocked_filters: Mapping[str, str],
    blocked_exptimes: Mapping[str, str],
) -> str:
    """Why a light is not reduced (incomplete calibration), or ``""``."""
    filt = filter_name(row)
    if filt in blocked_filters:
        return f"filter {filt}: {blocked_filters[filt]}"
    exptime = _float(row.get("exptime"))
    if math.isfinite(exptime) and exptime_key(exptime) in blocked_exptimes:
        return f"{exptime_key(exptime)} s: {blocked_exptimes[exptime_key(exptime)]}"
    return ""


def _summary_lines(plan: CalibrationPlan) -> list[str]:
    lines = []
    frame_rows = {str(r["frame_id"]): dict(zip(plan.frames.colnames, r, strict=True))
                  for r in plan.frames} if len(plan.frames) else {}
    lines.append(f"Sessions: {len(plan.sessions)}")
    for s in plan.sessions:
        pa = f"{s.pa_mod180:.2f} deg" if math.isfinite(s.pa_mod180) else "unverified"
        lines.append(f"  {s.session_id}: {len(s.frame_ids)} lights, orientation {pa}"
                     f" ({'; '.join(s.evidence)})")
    lines.append(f"Reduction units: {len(plan.units)}")
    for unit in plan.units:
        flats = ", ".join(f"{k}={v or '-'}" for k, v in sorted(unit.flats.items()))
        lines.append(f"  {unit.unit_id}: {len(unit.light_ids)} lights, bias {unit.bias_id or '-'}"
                     f", darks {unit.dark_id or '-'}, flats [{flats}]")
        for note in unit.notes:
            lines.append(f"    note: {note}")
        if unit.status != "ready":
            n_blocked = sum(
                1 for f in unit.light_ids
                if light_block_reason(frame_rows.get(f, {}), unit.blocked_filters,
                                      unit.blocked_exptimes)
            )
            verb = ("reduced anyway (force_units)" if unit.forced
                    else "not reduced until calibration is complete or the unit is in "
                         "overrides.force_units")
            lines.append(f"    INCOMPLETE: {n_blocked} of {len(unit.light_ids)} lights {verb}")
            for filt, reason in unit.blocked_filters.items():
                lines.append(f"      filter {filt}: {reason}")
            for exptime, reason in unit.blocked_exptimes.items():
                lines.append(f"      {exptime} s: {reason}")
        for filt, reason in unit.excluded.items():
            lines.append(f"    excluded filter {filt}: {reason}")
    counts = {CERTAIN: 0, LIKELY: 0, UNCERTAIN: 0, REJECTED: 0}
    for a in plan.flat_assignments:
        counts[a.category] = counts.get(a.category, 0) + 1
    line = "Flat assignments: " + ", ".join(f"{v} {k}" for k, v in counts.items())
    used_anyway = sum(1 for a in plan.flat_assignments
                      if a.category == REJECTED and a.flat_set_ids)
    if used_anyway:
        line += f" ({used_anyway} rejected used anyway as best available, see unit notes)"
    lines.append(line)
    if len(plan.targets):
        lines.append("Targets:")
        for row in plan.targets:
            flag = "" if row["stack"] else " (not stacked)"
            lines.append(f"  {row['target_id']} {row['target_name']}: {row['n_frames']} frames, "
                         f"{row['exposure_s'] / 60:.0f} min [{row['frames']}]{flag}")
    unknown = int(np.count_nonzero(np.asarray(plan.frames["session_id"]) == UNKNOWN_SESSION))
    if unknown:
        lines.append(f"{unknown} light frame(s) between two sessions without a plate solution "
                     "were left out (session unknown).")
    n_unknown_type = int(np.count_nonzero(np.asarray(plan.frames["frame_type"]) == UNKNOWN))
    if n_unknown_type:
        lines.append(f"{n_unknown_type} frame(s) of unknown type were ignored.")
    n_unknown_target = int(np.count_nonzero(np.asarray(plan.frames["target_id"]) == UNKNOWN_TARGET))
    if n_unknown_target:
        lines.append(f"{n_unknown_target} light frame(s) without a target were ignored.")
    return lines


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def plan_to_dict(plan: CalibrationPlan) -> dict:
    targets = {}
    for row in plan.targets:
        targets[str(row["target_id"])] = {
            "name": str(row["target_name"]),
            "ra": float(row["target_ra"]),
            "dec": float(row["target_dec"]),
            "n_frames": int(row["n_frames"]),
            "frames": str(row["frames"]),
            "exposure_s": float(row["exposure_s"]),
            "stack": bool(row["stack"]),
        }
    data = {
        "version": PLAN_VERSION,
        "settings": asdict(plan.settings),
        "sessions": {s.session_id: {
            "camera": s.camera, "telescope": s.telescope, "start_jd": s.start_jd,
            "end_jd": s.end_jd, "pa_mod180": s.pa_mod180, "pa_spread": s.pa_spread,
            "n_frames": len(s.frame_ids), "n_solved": s.n_solved, "evidence": s.evidence,
        } for s in plan.sessions},
        "flat_sets": {s.set_id: {
            "camera": s.camera, "binning": s.binning, "filter": s.filter,
            "start_jd": s.start_jd, "end_jd": s.end_jd, "n_frames": len(s.frame_ids),
        } for s in plan.flat_sets},
        "flat_assignments": [{
            "session_id": a.session_id, "filter": a.filter, "binning": a.binning,
            "flat_sets": a.flat_set_ids, "probability": a.probability,
            "category": a.category, "note": a.note,
            "candidates": [{"set": c.set_id, "p": c.probability, "category": c.category,
                            "reason": c.reason} for c in a.candidates],
        } for a in plan.flat_assignments],
        "masters": {mid: {k: v for k, v in asdict(m).items() if v not in ("", [], None)}
                    for mid, m in plan.masters.items()},
        "units": {u.unit_id: {
            "session_id": u.session_id, "electronic_id": u.electronic_id, "camera": u.camera,
            "telescope": u.telescope, "bias": u.bias_id, "darks": u.dark_id,
            "flats": u.flats, "lights": u.light_ids, "excluded_filters": u.excluded,
            "flat_categories": u.flat_categories,
            "notes": u.notes, "status": u.status, "blocked_filters": u.blocked_filters,
            "blocked_exptimes": u.blocked_exptimes, "forced": u.forced,
        } for u in plan.units},
        "targets": targets,
        "overrides": asdict(plan.overrides),
    }
    return _clean(data)


_HEADER = """\
# Calibration plan written by ost_photometry (archive pipeline).
#
# Edit only the 'overrides' block, then rebuild the plan
# (2_classify_and_group.py). Override keys:
#   exclude_frames:   [frame_id, ...]            ignore frames completely
#   frame_types:      {frame_id: bias|dark|flat|light}
#   session_breaks:   [frame_id, ...]            start a new mount session here
#   merge_sessions:   [[session_id, session_id], ...]
#   force_flats:      {session_id: {filter: [flat_set_id, ...]}}
#   merge_targets:    [[target name or id, ...], ...]   e.g. mosaic panels
#   rename_targets:   {target name or id: new name}
#   no_stack_targets: [target name or id, ...]
#   force_units:      [unit_id, ...]   reduce although calibration is incomplete
#
# Units with status 'incomplete' lack darks for some exposure times or a
# flat of an accepted category (settings.accepted_flat_categories); those
# lights are not reduced. missing_calibrations.ecsv lists what to take.
"""


#: Columns of :func:`missing_calibrations`.
MISSING_COLUMNS = ("kind", "camera", "instrume", "telescope", "binning", "gain", "offset",
                   "readout_mode", "set_temp", "exptime", "n_frames", "required",
                   "needed_for", "reason", "electronic_id")


def missing_calibrations(
    plan: CalibrationPlan,
    *,
    n_darks: int = 20,
    n_bias: int = 50,
) -> Table:
    """Bias / dark exposures to take so that the plan becomes complete.

    One row per electronic setup and exposure time: darks for blocked light
    exposures and for flat masters without matching darks, and a bias
    series for setups without bias (it allows dark scaling). The camera
    settings are copied from the headers (readout mode as the driver names
    it), so the list can be turned into an acquisition plan (e.g. NINA).
    """
    setups: dict[str, dict] = {}
    for row in plan.frames:
        eid = str(row["electronic_id"]) if "electronic_id" in plan.frames.colnames else ""
        if eid and eid not in setups and str(row["frame_type"]) in (LIGHT, FLAT):
            setups[eid] = dict(zip(plan.frames.colnames, row, strict=True))
    needs: dict[tuple[str, str, float], dict] = {}

    def add(kind: str, eid: str, exptime: float, user: str, reason: str, required: bool) -> None:
        key = (kind, eid, round(float(exptime), 3))
        entry = needs.setdefault(key, {"users": [], "reason": reason, "required": required})
        if user not in entry["users"]:
            entry["users"].append(user)
        entry["required"] = entry["required"] or required

    for unit in plan.units:
        dark_missing = False
        for key, reason in unit.blocked_exptimes.items():
            add("dark", unit.electronic_id, float(key), unit.unit_id, reason, True)
            dark_missing = True
        if dark_missing and not unit.bias_id:
            add("bias", unit.electronic_id, 0.0, unit.unit_id,
                "no bias in the window (allows scaling darks of other exposure times)", False)
    for mid, spec in plan.masters.items():
        if spec.kind != FLAT or not spec.missing_exptimes:
            continue
        for exptime in spec.missing_exptimes:
            add("dark", spec.electronic_id, exptime, mid, "darks for the flat exposures", True)
        if not spec.bias_id:
            add("bias", spec.electronic_id, 0.0, mid,
                "no bias in the window (allows scaling darks of other exposure times)", False)

    table = Table(names=MISSING_COLUMNS,
                  dtype=(str, str, str, str, str, float, float, str, float, float, int, bool,
                         str, str, str))
    for (kind, eid, exptime), entry in sorted(needs.items(), key=lambda kv: (kv[0][1], kv[0][0],
                                                                             kv[0][2])):
        setup = setups.get(eid, {})
        table.add_row((
            kind, camera_id(setup) if setup else eid.split("|")[0],
            str(setup.get("instrume", "")), str(setup.get("telescop", "")),
            binning(setup) if setup else "", _float(setup.get("gain")),
            _float(setup.get("offset")), str(setup.get("readoutm", "")),
            _float(setup.get("set_temp")) if math.isfinite(_float(setup.get("set_temp")))
            else _float(setup.get("ccd_temp")),
            exptime, n_bias if kind == "bias" else n_darks, entry["required"],
            ";".join(entry["users"]), entry["reason"], eid,
        ))
    return table


def write_missing_calibrations(plan: CalibrationPlan, path: str | Path, **kwargs) -> Table:
    """Write :func:`missing_calibrations` as ECSV (also when empty)."""
    table = missing_calibrations(plan, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="ascii.ecsv", overwrite=True)
    return table


def write_plan(plan: CalibrationPlan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(plan_to_dict(plan), sort_keys=False, allow_unicode=True, width=100)
    path.write_text(_HEADER + text)
    return path


def write_frames(plan: CalibrationPlan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.frames.write(path, format="ascii.ecsv", overwrite=True)
    return path


def load_plan(path: str | Path) -> dict:
    """Read and validate a plan YAML; returns the plan dictionary."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if data.get("version") != PLAN_VERSION:
        raise ValueError(f"Unsupported calibration plan version {data.get('version')!r}")
    for key in ("masters", "units", "targets", "overrides"):
        data.setdefault(key, {})
    Overrides.from_mapping(data["overrides"])
    masters = data["masters"]
    for unit_id, unit in data["units"].items():
        for ref in [unit.get("bias"), unit.get("darks"), *dict(unit.get("flats") or {}).values()]:
            if ref and ref not in masters:
                raise ValueError(f"Unit {unit_id} refers to unknown master {ref!r}")
    for mid, master in masters.items():
        for ref in (master.get("bias_id"), master.get("dark_id")):
            if ref and ref not in masters:
                raise ValueError(f"Master {mid} refers to unknown master {ref!r}")
    return data


def read_overrides(path: str | Path) -> Overrides:
    """Overrides of an existing plan (empty if the file does not exist)."""
    path = Path(path)
    if not path.is_file():
        return Overrides()
    data = yaml.safe_load(path.read_text()) or {}
    return Overrides.from_mapping(data.get("overrides"))


__all__ = [
    "CalibrationPlan",
    "MasterSpec",
    "Overrides",
    "PLAN_VERSION",
    "PlanSettings",
    "ReductionUnit",
    "build_calibration_plan",
    "exptime_key",
    "light_block_reason",
    "load_plan",
    "missing_calibrations",
    "orientation_table",
    "plan_to_dict",
    "read_overrides",
    "sessions_table",
    "write_frames",
    "write_missing_calibrations",
    "write_plan",
]
