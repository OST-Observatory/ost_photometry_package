"""Mount sessions: time blocks in which a camera stayed mounted unchanged.

A session ends when the camera or pixel scale changes, when the camera
orientation (modulo 180 degrees) changes by more than ``pa_tolerance``, or
when another camera was used at the same telescope in between (archive
context frames). Unsolved frames inherit the session of their solved
neighbours when both neighbours agree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table

from ...wcs import orientation_difference_mod180
from .orientation import OrientationResult
from .setup_keys import camera_id, telescope_id

UNKNOWN_SESSION = "unknown"


@dataclass
class Session:
    session_id: str
    camera: str
    telescope: str
    start_jd: float
    end_jd: float
    pa_mod180: float = float("nan")
    pa_spread: float = float("nan")
    parity: str = ""
    scale_arcsec: float = float("nan")
    frame_ids: list[str] = field(default_factory=list)
    n_solved: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.n_solved > 0


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _night(jd: float) -> str:
    """Calendar date of the evening the night started (local noon split)."""
    if not math.isfinite(jd):
        return "nodate"
    from astropy.time import Time

    return Time(jd - 0.5, format="jd").datetime.strftime("%Y%m%d")


def _circular_spread_mod180(values: Sequence[float]) -> tuple[float, float]:
    """Mean and max deviation of angles modulo 180 degrees."""
    angles = np.radians(np.asarray(values, dtype=float) * 2.0)
    mean = math.degrees(math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) / 2.0
    mean %= 180.0
    spread = max(orientation_difference_mod180(v, mean) for v in values)
    return mean, spread


def hard_breaks(
    context: Table | None, telescope: str, camera: str
) -> np.ndarray:
    """Times (JD) at which another camera was used at ``telescope``."""
    if context is None or len(context) == 0:
        return np.zeros(0)
    times = []
    for row in context:
        row = dict(zip(context.colnames, row, strict=True))
        if telescope_id(row) != telescope:
            continue
        other = camera_id(row)
        if other in ("unknown", camera):
            continue
        t = _float(row.get("jd"))
        if math.isfinite(t):
            times.append(t)
    return np.sort(np.asarray(times, dtype=float))


def _hard_break_between(
    a: int,
    b: int,
    rows: Sequence[Mapping[str, object]],
    jd: np.ndarray,
    breaks: np.ndarray,
) -> str:
    """Reason for a hard session break between consecutive lights ``a`` and ``b``."""
    cam_a, cam_b = camera_id(rows[a]), camera_id(rows[b])
    if cam_a != cam_b:
        return f"camera change {cam_a} -> {cam_b}"
    if breaks.size and np.any((breaks > jd[a]) & (breaks < jd[b])):
        return "another camera was used at the telescope in between"
    return ""


def _split_segment(
    segment: Sequence[int],
    frame_ids: Sequence[str],
    orientations: Mapping[str, OrientationResult],
    pa_tolerance: float,
    scale_tolerance: float,
) -> list[tuple[list[int], list[str]]]:
    """Split one hard-break-free segment at orientation changes.

    Returns ``[(frame indices, evidence), ...]``. Unsolved frames before the
    first / after the last solution join the adjacent session; unsolved
    frames between two solutions of different sessions are left out
    (ambiguous: the camera was remounted somewhere in that interval).
    """
    solved = [i for i in segment if frame_ids[i] in orientations
              and orientations[frame_ids[i]].solved]
    if not solved:
        return [(list(segment), ["no plate solution in this session (orientation unverified)"])]

    # Session index per solved frame.
    session_of: dict[int, int] = {}
    evidence: list[list[str]] = [[]]
    reference: list[float] = []
    ref_parity = ""
    ref_scale = float("nan")
    for i in solved:
        result = orientations[frame_ids[i]]
        reasons = []
        if reference:
            mean, _ = _circular_spread_mod180(reference)
            diff = orientation_difference_mod180(result.pa_deg, mean)
            if diff > pa_tolerance:
                reasons.append(f"orientation changed by {diff:.2f} deg")
            if ref_parity and result.parity and result.parity != ref_parity:
                reasons.append("image parity changed")
            if math.isfinite(ref_scale) and math.isfinite(result.scale_arcsec) and abs(
                result.scale_arcsec / ref_scale - 1.0
            ) > scale_tolerance:
                reasons.append("pixel scale changed")
        if reasons:
            evidence.append(reasons)
            reference, ref_parity, ref_scale = [], "", float("nan")
        session_of[i] = len(evidence) - 1
        reference.append(result.pa_deg % 180.0)
        ref_parity = ref_parity or result.parity
        if not math.isfinite(ref_scale):
            ref_scale = result.scale_arcsec

    members: list[list[int]] = [[] for _ in evidence]
    position = {i: k for k, i in enumerate(segment)}
    solved_positions = [position[i] for i in solved]
    for k, i in enumerate(segment):
        if i in session_of:
            members[session_of[i]].append(i)
            continue
        before = [p for p in solved_positions if p < k]
        after = [p for p in solved_positions if p > k]
        s_before = session_of[segment[before[-1]]] if before else None
        s_after = session_of[segment[after[0]]] if after else None
        if s_before is None:
            members[s_after].append(i)
        elif s_after is None or s_before == s_after:
            members[s_before].append(i)
        # else: between two different sessions -> ambiguous, left out
    return [(m, e) for m, e in zip(members, evidence, strict=True) if m]


def segment_sessions(
    lights: Table,
    orientations: Mapping[str, OrientationResult],
    *,
    context: Table | None = None,
    pa_tolerance: float = 0.5,
    scale_tolerance: float = 0.02,
    forced_breaks: set[str] | frozenset[str] = frozenset(),
) -> tuple[Table, list[Session]]:
    """Assign ``session_id`` to every light; returns the table copy and sessions.

    ``forced_breaks`` are frame ids that start a new session (manual
    override, e.g. a remount that the orientation did not reveal).

    ``context`` are frames of other instruments (any type, metadata only)
    used for hard breaks between sessions. Lights that cannot be placed get
    ``session_id = "unknown"``.
    """
    table = lights.copy()
    n = len(table)
    rows = [dict(zip(table.colnames, r, strict=True)) for r in table]
    frame_ids = [str(r["frame_id"]) for r in rows]
    jd = np.asarray(table["jd"], dtype=float) if n else np.zeros(0)
    sessions: list[Session] = []

    timelines: dict[str, list[int]] = {}
    for i in np.argsort(np.nan_to_num(jd, nan=np.inf), kind="stable"):
        timelines.setdefault(telescope_id(rows[i]), []).append(int(i))

    for telescope, order in timelines.items():
        segments: list[tuple[list[int], str]] = []
        for k, i in enumerate(order):
            if k == 0:
                segments.append(([i], "first frame"))
                continue
            breaks = hard_breaks(context, telescope, camera_id(rows[i]))
            reason = _hard_break_between(order[k - 1], i, rows, jd, breaks)
            if not reason and frame_ids[i] in forced_breaks:
                reason = "manual session break"
            if reason:
                segments.append(([i], reason))
            else:
                segments[-1][0].append(i)
        for segment, segment_reason in segments:
            parts = _split_segment(segment, frame_ids, orientations, pa_tolerance,
                                   scale_tolerance)
            for part_index, (members, evidence) in enumerate(parts):
                reasons = ([segment_reason] if part_index == 0 else []) + evidence
                sessions.append(Session(
                    session_id="",
                    camera=camera_id(rows[members[0]]),
                    telescope=telescope,
                    start_jd=float(np.nanmin(jd[members])),
                    end_jd=float(np.nanmax(jd[members])),
                    frame_ids=[frame_ids[i] for i in members],
                    evidence=reasons,
                ))

    counters: dict[str, int] = {}
    for s in sorted(sessions, key=lambda s: (np.nan_to_num(s.start_jd, nan=np.inf), s.camera)):
        night = _night(s.start_jd)
        counters[night] = counters.get(night, 0) + 1
        s.session_id = f"S{night}_{counters[night]:02d}_{s.camera}"
        results = [orientations[f] for f in s.frame_ids if f in orientations
                   and orientations[f].solved]
        s.n_solved = len(results)
        if results:
            s.pa_mod180, s.pa_spread = _circular_spread_mod180([r.pa_deg % 180 for r in results])
            s.parity = results[0].parity
            s.scale_arcsec = float(np.median([r.scale_arcsec for r in results]))
    session_of = {f: s.session_id for s in sessions for f in s.frame_ids}
    table["session_id"] = np.array([session_of.get(f, UNKNOWN_SESSION) for f in frame_ids],
                                   dtype=str)
    table["pa_mod180"] = np.array(
        [orientations[f].pa_mod180 if f in orientations and orientations[f].solved else np.nan
         for f in frame_ids],
        dtype=float,
    )
    return table, sessions


def sessions_table(sessions: Sequence[Session]) -> Table:
    if not sessions:
        return Table()
    rows = []
    for s in sessions:
        rows.append({
            "session_id": s.session_id,
            "camera": s.camera,
            "telescope": s.telescope,
            "start_jd": s.start_jd,
            "end_jd": s.end_jd,
            "pa_mod180": s.pa_mod180,
            "pa_spread": s.pa_spread,
            "parity": s.parity,
            "scale_arcsec": s.scale_arcsec,
            "n_frames": len(s.frame_ids),
            "n_solved": s.n_solved,
            "evidence": "; ".join(s.evidence),
        })
    table = Table(rows=rows)
    for name in ("session_id", "camera", "telescope", "parity", "evidence"):
        table[name] = np.array([str(v) for v in table[name]], dtype=str)
    return table


__all__ = ["Session", "UNKNOWN_SESSION", "hard_breaks", "segment_sessions", "sessions_table"]
