"""Group light frames into targets by sky position.

``OBJECT`` names in the archive are unreliable (case, spelling, missing),
so targets are clusters of field centres: two frames belong to the same
target when their centres are closer than ``overlap_fraction`` times the
smaller field height. Names come from the archive object link, else from
the most common ``OBJECT`` of the cluster, else from the coordinates.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from astropy.table import Table

from .setup_keys import camera_id

GROUP_BY_POSITION = "position"
GROUP_BY_ARCHIVE_OBJECT = "archive_object"
GROUP_BY_NAME = "name"
TARGET_GROUPINGS = (GROUP_BY_POSITION, GROUP_BY_ARCHIVE_OBJECT, GROUP_BY_NAME)
UNKNOWN_TARGET = "unknown"

#: Field height assumed when neither a plate solution nor the header geometry
#: (FOCALLEN, XPIXSZ) is available (degrees; OST CDK20 + QHY600 is ~0.6).
DEFAULT_FIELD_DEG = 0.5

#: Largest mount pointing error (deg) for which a frame with only a header
#: position is matched to a plate-solved field of the same object name.
MAX_POINTING_ERROR_DEG = 10.0


def normalize_name(name: object) -> str:
    """Comparison key for object names (``"M 57"`` == ``"m57"`` == ``"M_57"``)."""
    text = str(name or "").strip().lower()
    if text in {"", "-", "none", "unknown", "object"}:
        return ""
    return "".join(ch for ch in text if ch.isalnum())


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def pixel_scale_arcsec(row: Mapping[str, object]) -> float:
    """Pixel scale from ``XPIXSZ`` (binned, µm) and ``FOCALLEN`` (mm)."""
    pixel = _float(row.get("xpixsz"))
    focal = _float(row.get("focallen"))
    if math.isfinite(pixel) and math.isfinite(focal) and pixel > 0 and focal > 0:
        return 206.265 * pixel / focal
    return float("nan")


def field_height_deg(row: Mapping[str, object], scale_arcsec: float | None = None) -> float:
    scale = _float(scale_arcsec) if scale_arcsec is not None else float("nan")
    if not math.isfinite(scale):
        scale = pixel_scale_arcsec(row)
    sizes = [v for v in (_float(row.get("naxis1")), _float(row.get("naxis2")))
             if math.isfinite(v) and v > 0]
    if math.isfinite(scale) and sizes:
        return scale * min(sizes) / 3600.0
    return DEFAULT_FIELD_DEG


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_positions(
    ra_deg: Sequence[float],
    dec_deg: Sequence[float],
    radius_deg: Sequence[float],
) -> list[int]:
    """Single-linkage clusters: ``i`` and ``j`` join if their separation is
    below ``min(radius_i, radius_j)``. Returns a cluster index per point."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    radius = np.asarray(radius_deg, dtype=float)
    n = ra.size
    uf = _UnionFind(n)
    if n == 0:
        return []
    vectors = _unit_vectors(ra, dec)
    chunk = 512
    for start in range(0, n, chunk):
        block = vectors[start : start + chunk]
        cos_sep = np.clip(block @ vectors.T, -1.0, 1.0)
        sep = np.degrees(np.arccos(cos_sep))
        limit = np.minimum(radius[start : start + chunk, None], radius[None, :])
        rows, cols = np.nonzero(sep < limit)
        for r, c in zip(rows + start, cols, strict=True):
            if c > r:
                uf.union(int(r), int(c))
    roots = [uf.find(i) for i in range(n)]
    order = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return [order[r] for r in roots]


def _cluster_solved_first(labels, notes, ra, dec, radius, solved, names) -> None:
    """Cluster plate-solved centres first; header pointings only after that.

    Header pointings (mount coordinates) can be degrees off. A frame with
    only a header position joins the solved cluster whose object name it
    shares (if exactly one); the remaining frames are clustered by their
    header positions among themselves.
    """
    solved_idx = np.flatnonzero(solved)
    if solved_idx.size:
        labels[solved_idx] = cluster_positions(ra[solved_idx], dec[solved_idx],
                                               radius[solved_idx])
    next_label = int(labels.max()) + 1 if solved_idx.size else 0
    solved_names: dict[str, set[int]] = {}
    for i in solved_idx:
        if names[i]:
            solved_names.setdefault(names[i], set()).add(int(labels[i]))
    header_idx = np.flatnonzero(~solved & np.isfinite(ra) & np.isfinite(dec))
    rest = []
    for i in header_idx:
        clusters = solved_names.get(names[i], set()) if names[i] else set()
        if len(clusters) == 1:
            members = solved_idx[labels[solved_idx] == next(iter(clusters))]
            sep = np.degrees(np.arccos(np.clip(
                _unit_vectors(ra[members], dec[members]) @ _unit_vectors(ra[[i]], dec[[i]])[0],
                -1, 1)))
            if float(np.min(sep)) > MAX_POINTING_ERROR_DEG:
                clusters = set()
        if len(clusters) == 1:
            labels[i] = next(iter(clusters))
            notes[i] = "header pointing; matched to the solved field by object name"
        else:
            rest.append(i)
    if rest:
        # Header-only frames: join a solved cluster when close enough, else
        # form their own clusters.
        rest = np.asarray(rest)
        own = cluster_positions(ra[rest], dec[rest], radius[rest])
        for i, label in zip(rest, own, strict=True):
            if solved_idx.size:
                vec = _unit_vectors(ra[[i]], dec[[i]])
                sep = np.degrees(np.arccos(np.clip(
                    _unit_vectors(ra[solved_idx], dec[solved_idx]) @ vec[0], -1, 1)))
                close = sep < np.minimum(radius[i], radius[solved_idx])
                if np.any(close):
                    labels[i] = labels[solved_idx[np.argmin(np.where(close, sep, np.inf))]]
                    continue
            labels[i] = next_label + label


def _mean_position(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float]:
    vec = _unit_vectors(ra, dec).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        return float("nan"), float("nan")
    x, y, z = vec / norm
    return float(np.degrees(np.arctan2(y, x)) % 360.0), float(np.degrees(np.arcsin(z)))


def _format_field_name(ra: float, dec: float) -> str:
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return UNKNOWN_TARGET
    return f"field_{ra:07.3f}{'+' if dec >= 0 else '-'}{abs(dec):06.3f}"


def assign_targets(
    lights: Table,
    *,
    centers: Mapping[str, tuple[float, float]] | None = None,
    scales: Mapping[str, float] | None = None,
    overlap_fraction: float = 0.5,
    group_by: str = GROUP_BY_POSITION,
    merge_targets: Sequence[Sequence[str]] = (),
    rename: Mapping[str, str] | None = None,
    neighbour_minutes: float = 30.0,
) -> Table:
    """Add ``target_id``, ``target_name``, ``target_ra``/``dec``, ``target_note``.

    ``lights`` are light frames (manifest rows). ``centers`` maps
    ``frame_id`` to a solved field centre (deg) and ``scales`` to a solved
    pixel scale (arcsec); without them the header / archive pointing and
    header geometry are used. ``merge_targets`` lists groups of target names
    or ids to combine (mosaics); ``rename`` maps id or name to a new name.
    Returns a copy of ``lights``.
    """
    if group_by not in TARGET_GROUPINGS:
        raise ValueError(f"group_by must be one of {TARGET_GROUPINGS}, got {group_by!r}")
    table = lights.copy()
    n = len(table)
    centers = dict(centers or {})
    scales = dict(scales or {})
    frame_ids = [str(f) for f in table["frame_id"]]
    rows = [dict(zip(table.colnames, r, strict=True)) for r in table]
    names = [normalize_name(r.get("main_object_name")) or normalize_name(r.get("object"))
             or normalize_name(r.get("main_target")) for r in rows]
    jd = np.asarray(table["jd"], dtype=float) if n else np.zeros(0)

    ra = np.full(n, np.nan)
    dec = np.full(n, np.nan)
    radius = np.full(n, np.nan)
    solved = np.zeros(n, dtype=bool)
    for i, (fid, row) in enumerate(zip(frame_ids, rows, strict=True)):
        if fid in centers and all(math.isfinite(v) for v in centers[fid]):
            ra[i], dec[i] = centers[fid]
            solved[i] = True
        else:
            ra[i], dec[i] = _float(row.get("ra")), _float(row.get("dec"))
        radius[i] = overlap_fraction * field_height_deg(row, scales.get(fid))

    labels = np.full(n, -1, dtype=int)
    notes = [""] * n
    if group_by == GROUP_BY_POSITION:
        _cluster_solved_first(labels, notes, ra, dec, radius, solved, names)
    else:
        key_values = []
        for row, name in zip(rows, names, strict=True):
            object_id = _float(row.get("main_object_id"))
            if group_by == GROUP_BY_ARCHIVE_OBJECT and math.isfinite(object_id) and object_id > 0:
                key_values.append(f"id{int(object_id)}")
            else:
                key_values.append(name)
        mapping: dict[str, int] = {}
        for i, key in enumerate(key_values):
            if key:
                labels[i] = mapping.setdefault(key, len(mapping))

    # Frames without a position: by name, then by time neighbours.
    next_label = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    name_only: dict[str, int] = {}
    cluster_names: dict[int, Counter] = {}
    for i in np.flatnonzero(labels >= 0):
        if names[i]:
            cluster_names.setdefault(int(labels[i]), Counter())[names[i]] += 1
    for i in np.flatnonzero(labels < 0):
        matches = [c for c, counter in cluster_names.items() if names[i] and names[i] in counter]
        if len(matches) == 1:
            labels[i] = matches[0]
            notes[i] = "no position; matched by object name"
            continue
        assigned = labels >= 0
        before = np.flatnonzero(assigned & (jd < jd[i]) & (jd >= jd[i] - neighbour_minutes / 1440))
        after = np.flatnonzero(assigned & (jd > jd[i]) & (jd <= jd[i] + neighbour_minutes / 1440))
        cams = camera_id(rows[i])
        before = [j for j in before if camera_id(rows[j]) == cams]
        after = [j for j in after if camera_id(rows[j]) == cams]
        neighbour_labels = {int(labels[j]) for j in (before[-1:] + after[:1])}
        if len(neighbour_labels) == 1 and (before or after):
            labels[i] = neighbour_labels.pop()
            notes[i] = "no position; matched by time neighbours"
        elif names[i]:
            # Name-only target: no frame with this name has a position.
            if names[i] not in name_only:
                name_only[names[i]] = next_label
                next_label += 1
            labels[i] = name_only[names[i]]
            notes[i] = "no position; grouped by object name"
        else:
            notes[i] = "no position, no name, no neighbours"

    # Merge requests are applied after naming (they may refer to names).
    target_ids, target_names, target_ra, target_dec = _name_clusters(
        labels, rows, names, ra, dec, jd
    )
    target_ids, target_names = _apply_merge_and_rename(
        target_ids, target_names, merge_targets, rename or {}
    )
    # Recompute centres after merging.
    final_ra = np.full(n, np.nan)
    final_dec = np.full(n, np.nan)
    for tid in set(target_ids):
        idx = np.array([i for i, t in enumerate(target_ids) if t == tid])
        good = idx[np.isfinite(ra[idx]) & np.isfinite(dec[idx])]
        if good.size:
            final_ra[idx], final_dec[idx] = _mean_position(ra[good], dec[good])
        else:
            final_ra[idx], final_dec[idx] = target_ra[idx[0]], target_dec[idx[0]]
    table["target_id"] = np.array(target_ids, dtype=str)
    table["target_name"] = np.array(target_names, dtype=str)
    table["target_ra"] = final_ra
    table["target_dec"] = final_dec
    table["target_note"] = np.array(notes, dtype=str)
    return table


def _name_clusters(labels, rows, names, ra, dec, jd):
    n = len(labels)
    target_ids = [UNKNOWN_TARGET] * n
    target_names = [UNKNOWN_TARGET] * n
    t_ra = np.full(n, np.nan)
    t_dec = np.full(n, np.nan)
    clusters = sorted(
        {int(c) for c in labels if c >= 0},
        key=lambda c: float(np.nanmin(np.where(labels == c, jd, np.inf))),
    )
    used_names: Counter = Counter()
    for k, cluster in enumerate(clusters, start=1):
        idx = np.flatnonzero(labels == cluster)
        good = idx[np.isfinite(ra[idx]) & np.isfinite(dec[idx])]
        c_ra, c_dec = _mean_position(ra[good], dec[good]) if good.size else (np.nan, np.nan)
        archive = Counter(str(rows[i].get("main_object_name") or "").strip() for i in idx)
        archive.pop("", None)
        header = Counter(str(rows[i].get("object") or "").strip() for i in idx if names[i])
        if archive:
            name = archive.most_common(1)[0][0]
        elif header:
            name = header.most_common(1)[0][0]
        else:
            name = _format_field_name(c_ra, c_dec)
        used_names[name] += 1
        if used_names[name] > 1:
            name = f"{name}_{used_names[name]}"
        tid = f"T{k:02d}"
        for i in idx:
            target_ids[i] = tid
            target_names[i] = name
            t_ra[i], t_dec[i] = c_ra, c_dec
    return target_ids, target_names, t_ra, t_dec


def _apply_merge_and_rename(target_ids, target_names, merge_targets, rename):
    ids = list(target_ids)
    names = list(target_names)
    by_key: dict[str, str] = {}
    for tid, name in zip(ids, names, strict=True):
        by_key.setdefault(tid, tid)
        by_key.setdefault(name, tid)
        by_key.setdefault(normalize_name(name), tid)
    for group in merge_targets:
        members = [by_key.get(str(m)) or by_key.get(normalize_name(m)) for m in group]
        members = [m for m in members if m]
        if len(members) < 2:
            continue
        keep = sorted(members)[0]
        keep_name = names[ids.index(keep)]
        for i, tid in enumerate(ids):
            if tid in members:
                ids[i] = keep
                names[i] = keep_name
    for key, new_name in rename.items():
        tid = by_key.get(str(key)) or by_key.get(normalize_name(key))
        if not tid:
            continue
        for i, current in enumerate(ids):
            if current == tid:
                names[i] = str(new_name)
    return ids, names


def target_summary(table: Table) -> Table:
    """One row per target: id, name, centre, frames per camera and filter, exposure."""
    from .setup_keys import filter_name

    rows = []
    for tid in dict.fromkeys(str(t) for t in table["target_id"]):
        sel = [r for r in table if str(r["target_id"]) == tid]
        counts = Counter(f"{camera_id(r)}:{filter_name(r)}" for r in sel)
        exposure = float(np.nansum([_float(r["exptime"]) for r in sel]))
        rows.append({
            "target_id": tid,
            "target_name": str(sel[0]["target_name"]),
            "target_ra": float(sel[0]["target_ra"]),
            "target_dec": float(sel[0]["target_dec"]),
            "n_frames": len(sel),
            "frames": ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            "exposure_s": exposure,
        })
    return Table(rows=rows, names=("target_id", "target_name", "target_ra", "target_dec",
                                   "n_frames", "frames", "exposure_s")) if rows else Table()


__all__ = [
    "DEFAULT_FIELD_DEG",
    "GROUP_BY_ARCHIVE_OBJECT",
    "GROUP_BY_NAME",
    "GROUP_BY_POSITION",
    "TARGET_GROUPINGS",
    "UNKNOWN_TARGET",
    "assign_targets",
    "cluster_positions",
    "field_height_deg",
    "normalize_name",
    "pixel_scale_arcsec",
    "target_summary",
]
