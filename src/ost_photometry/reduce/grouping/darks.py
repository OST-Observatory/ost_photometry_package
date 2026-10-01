"""Electronic setups and the choice of bias / dark frames.

Bias and dark frames depend on the camera electronics (camera, binning,
readout mode, gain, offset, sensor temperature) but not on how the camera
is mounted, so they can be reused across nights. Consumers (light or flat
frames of one electronic setup and night) get the bias / dark frames of
the nearest night that covers their exposure times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table

from .classify import BIAS, DARK
from .setup_keys import electronic_key, sensor_temperature, temperature_clusters


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def night_of(jd: float) -> str:
    """Night label ``YYYYMMDD`` (date of the evening; split at local noon UT)."""
    if not math.isfinite(jd):
        return "nodate"
    from astropy.time import Time

    return Time(jd - 0.5, format="jd").datetime.strftime("%Y%m%d")


def add_electronic_ids(table: Table, *, temp_tolerance: float = 2.0) -> Table:
    """Add ``electronic_id`` (setup key plus temperature cluster) and ``night``."""
    out = table.copy()
    rows = [dict(zip(out.colnames, r, strict=True)) for r in out]
    keys = [electronic_key(r) for r in rows]
    temps = [sensor_temperature(r) for r in rows]
    ids = [""] * len(rows)
    for key in dict.fromkeys(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        labels = temperature_clusters([temps[i] for i in idx], temp_tolerance)
        for cluster in set(labels):
            members = [idx[j] for j, lab in enumerate(labels) if lab == cluster]
            values = [temps[i] for i in members if math.isfinite(temps[i])]
            suffix = f"t{np.median(values):+.0f}" if values else "t?"
            for i in members:
                ids[i] = f"{key}|{suffix}"
    out["electronic_id"] = np.array(ids, dtype=str)
    out["night"] = np.array([night_of(_float(r.get("jd"))) for r in rows], dtype=str)
    return out


@dataclass
class CalibrationSet:
    """Bias or dark frames of one electronic setup and night."""

    set_id: str
    kind: str
    electronic_id: str
    night: str
    frame_ids: list[str] = field(default_factory=list)
    exptimes: list[float] = field(default_factory=list)
    jd: float = float("nan")


@dataclass
class DarkAssignment:
    """Bias / dark sets chosen for the consumers of one setup and night."""

    electronic_id: str
    night: str
    exptimes: list[float]
    bias: CalibrationSet | None = None
    darks: CalibrationSet | None = None
    missing_exptimes: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def calibration_sets(frames: Table, kind: str) -> list[CalibrationSet]:
    """Bias or dark sets (one per electronic id and night)."""
    sets: dict[tuple[str, str], CalibrationSet] = {}
    for row in frames:
        if str(row["frame_type"]) != kind:
            continue
        key = (str(row["electronic_id"]), str(row["night"]))
        entry = sets.get(key)
        if entry is None:
            prefix = "B" if kind == BIAS else "D"
            entry = CalibrationSet("", kind, key[0], key[1])
            entry.set_id = f"{prefix}_{key[1]}"
            sets[key] = entry
        entry.frame_ids.append(str(row["frame_id"]))
        exptime = _float(row["exptime"])
        if math.isfinite(exptime):
            entry.exptimes.append(exptime)
    for entry in sets.values():
        jds = [_float(r["jd"]) for r in frames if str(r["frame_id"]) in set(entry.frame_ids)]
        jds = [j for j in jds if math.isfinite(j)]
        entry.jd = float(np.median(jds)) if jds else float("nan")
        entry.exptimes = sorted(set(round(e, 3) for e in entry.exptimes))
    # Unique, readable ids per kind.
    counter: dict[str, int] = {}
    for entry in sorted(sets.values(), key=lambda e: (e.night, e.electronic_id)):
        counter[entry.night] = counter.get(entry.night, 0) + 1
        entry.set_id = f"{entry.set_id}_{counter[entry.night]:02d}"
    return list(sets.values())


def _covers(darks: CalibrationSet, exptime: float, tolerance: float, bias_ok: bool) -> bool:
    if any(abs(d - exptime) <= tolerance for d in darks.exptimes):
        return True
    # Dark scaling needs a bias and a longer dark (never scale up).
    return bias_ok and any(d >= exptime for d in darks.exptimes)


def assign_bias_dark(
    consumers: Table,
    frames: Table,
    *,
    exptime_tolerance: float = 0.5,
    window_days: float = 30.0,
) -> list[DarkAssignment]:
    """Choose bias / dark sets for every (electronic id, night) of ``consumers``.

    ``consumers`` are light or flat frames with ``electronic_id`` / ``night``;
    ``frames`` hold the classified bias and dark frames. The nearest night
    (within ``window_days``) whose darks cover all consumer exposure times
    wins; otherwise the nearest dark night is used and the uncovered
    exposure times are reported.
    """
    bias_sets = calibration_sets(frames, BIAS)
    dark_sets = calibration_sets(frames, DARK)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(consumers):
        groups.setdefault((str(row["electronic_id"]), str(row["night"])), []).append(i)
    assignments: list[DarkAssignment] = []
    jd_all = np.asarray(consumers["jd"], dtype=float)
    exp_all = np.asarray(consumers["exptime"], dtype=float)
    for (eid, night), idx in groups.items():
        times = jd_all[idx]
        centre = float(np.nanmedian(times)) if np.isfinite(times).any() else float("nan")
        exptimes = sorted(set(round(float(e), 3) for e in exp_all[idx] if math.isfinite(e)))
        result = DarkAssignment(eid, night, exptimes)

        def distance(entry: CalibrationSet, centre: float = centre) -> float:
            if not (math.isfinite(entry.jd) and math.isfinite(centre)):
                return float("inf")
            return abs(entry.jd - centre)

        biases = sorted((b for b in bias_sets if b.electronic_id == eid), key=distance)
        biases = [b for b in biases if distance(b) <= window_days]
        result.bias = biases[0] if biases else None
        darks = sorted((d for d in dark_sets if d.electronic_id == eid), key=distance)
        darks = [d for d in darks if distance(d) <= window_days]
        bias_ok = result.bias is not None
        full = [d for d in darks if all(_covers(d, e, exptime_tolerance, bias_ok)
                                        for e in exptimes)]
        if full:
            result.darks = full[0]
        elif darks:
            result.darks = darks[0]
        if result.darks is None:
            result.missing_exptimes = list(exptimes)
            result.notes.append("no dark frames for this electronic setup in the window")
        else:
            result.missing_exptimes = [
                e for e in exptimes
                if not _covers(result.darks, e, exptime_tolerance, bias_ok)
            ]
            if result.darks.night != night:
                result.notes.append(f"darks from night {result.darks.night}")
        if result.bias is None:
            result.notes.append("no bias frames (dark scaling impossible)")
        elif result.bias.night != night:
            result.notes.append(f"bias from night {result.bias.night}")
        if result.missing_exptimes:
            result.notes.append(
                "no usable dark for exposure(s) "
                + ", ".join(f"{e:g} s" for e in result.missing_exptimes)
            )
        assignments.append(result)
    return assignments


def bias_levels(frames: Table) -> dict[str, float]:
    """Median raw level of the bias frames per electronic id (from ``stat_median``)."""
    levels: dict[str, list[float]] = {}
    if "stat_median" not in frames.colnames:
        return {}
    for row in frames:
        if str(row["frame_type"]) == BIAS and math.isfinite(_float(row["stat_median"])):
            levels.setdefault(str(row["electronic_id"]), []).append(_float(row["stat_median"]))
    return {k: float(np.median(v)) for k, v in levels.items()}


__all__ = [
    "CalibrationSet",
    "DarkAssignment",
    "add_electronic_ids",
    "assign_bias_dark",
    "bias_levels",
    "calibration_sets",
    "night_of",
]
