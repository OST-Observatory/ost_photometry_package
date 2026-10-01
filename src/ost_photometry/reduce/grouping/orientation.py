"""Camera orientation of light frames (plate solving) for mount sessions.

The orientation of the camera on the sky (position angle of the image +y
axis, modulo 180 degrees to ignore meridian flips) is stable to about
0.2 degrees while a camera stays mounted and changes by many degrees when
it is remounted. Only a sample of frames is solved: block ends, one frame
per interval, frames around time gaps and target changes; disagreeing
neighbours are bisected to locate the change.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from ...wcs import (
    orientation_difference_mod180,
    position_angle_from_cd,
    position_angle_from_wcs,
    solve_astap_copy,
)
from .setup_keys import camera_id, telescope_id
from .targets import field_height_deg, pixel_scale_arcsec

SOURCE_SERVER = "server"
SOURCE_ASTAP = "astap"
SOURCE_FAILED = "failed"


@dataclass
class OrientationResult:
    frame_id: str
    pa_deg: float = float("nan")
    parity: str = ""
    scale_arcsec: float = float("nan")
    center_ra: float = float("nan")
    center_dec: float = float("nan")
    source: str = SOURCE_FAILED
    note: str = ""

    @property
    def solved(self) -> bool:
        return self.source != SOURCE_FAILED and math.isfinite(self.pa_deg)

    @property
    def pa_mod180(self) -> float:
        return self.pa_deg % 180.0 if math.isfinite(self.pa_deg) else float("nan")


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def orientation_from_server(row: Mapping[str, object]) -> OrientationResult | None:
    """Orientation from the archive's plate solution (``wcs_cd*``), if present."""
    if not bool(row.get("plate_solved")):
        return None
    cd = [_float(row.get(k)) for k in ("wcs_cd1_1", "wcs_cd1_2", "wcs_cd2_1", "wcs_cd2_2")]
    if not all(math.isfinite(v) for v in cd) or (cd[0] * cd[3] - cd[1] * cd[2]) == 0:
        return None
    pa, parity, scale = position_angle_from_cd(np.array(cd).reshape(2, 2))
    return OrientationResult(
        frame_id=str(row.get("frame_id")),
        pa_deg=pa,
        parity=parity,
        scale_arcsec=scale,
        center_ra=_float(row.get("wcs_ra")),
        center_dec=_float(row.get("wcs_dec")),
        source=SOURCE_SERVER,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _blocks(rows: Sequence[Mapping[str, object]], jd: np.ndarray, gap_days: float) -> list[list[int]]:
    """Time-ordered runs of frames with the same camera and telescope, split at gaps."""
    order = np.argsort(np.nan_to_num(jd, nan=np.inf), kind="stable")
    blocks: list[list[int]] = []
    last_key = None
    last_jd = None
    for i in order:
        key = (camera_id(rows[i]), telescope_id(rows[i]))
        t = jd[i]
        new = (
            not blocks
            or key != last_key
            or (math.isfinite(t) and last_jd is not None and math.isfinite(last_jd)
                and t - last_jd > gap_days)
        )
        if new:
            blocks.append([int(i)])
        else:
            blocks[-1].append(int(i))
        last_key, last_jd = key, t
    return blocks


def sample_frames_for_solving(
    lights: Table,
    *,
    interval_minutes: float = 30.0,
    gap_hours: float = 1.0,
    target_labels: Sequence[str] | None = None,
) -> list[int]:
    """Row indices to plate-solve.

    Per block (same camera and telescope, no gap above ``gap_hours``): the
    first and last frame, one frame per ``interval_minutes``, and the
    frames on both sides of every change in ``target_labels`` (a target
    change often coincides with an instrument change).
    """
    rows = [dict(zip(lights.colnames, r, strict=True)) for r in lights]
    jd = np.asarray(lights["jd"], dtype=float) if len(lights) else np.zeros(0)
    chosen: set[int] = set()
    for block in _blocks(rows, jd, gap_hours / 24.0):
        chosen.add(block[0])
        chosen.add(block[-1])
        next_time = jd[block[0]] + interval_minutes / 1440.0
        for i in block[1:]:
            if math.isfinite(jd[i]) and jd[i] >= next_time:
                chosen.add(i)
                next_time = jd[i] + interval_minutes / 1440.0
        if target_labels is not None:
            for a, b in zip(block[:-1], block[1:], strict=True):
                if str(target_labels[a]) != str(target_labels[b]):
                    chosen.update((a, b))
    return sorted(chosen, key=lambda i: (np.nan_to_num(jd[i], nan=np.inf), i))


# ---------------------------------------------------------------------------
# Solving with cache
# ---------------------------------------------------------------------------


def _cache_key(row: Mapping[str, object]) -> str:
    sha = str(row.get("sha256") or "")
    if sha:
        return sha
    path = Path(str(row.get("local_path") or ""))
    try:
        stat = path.stat()
    except OSError:
        return str(path)
    return f"{path}|{stat.st_size}|{int(stat.st_mtime)}"


def read_orientation_cache(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        return {}
    table = Table.read(path, format="ascii.ecsv")
    return {str(r["cache_key"]): {k: r[k] for k in table.colnames} for r in table}


def write_orientation_cache(path: str | Path, cache: Mapping[str, Mapping[str, object]]) -> None:
    if not cache:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("cache_key", "pa_deg", "parity", "scale_arcsec", "center_ra", "center_dec",
            "source", "note", "radius_deg")
    rows = [[entry.get(k, "") if k in ("parity", "source", "note", "cache_key")
             else _float(entry.get(k)) for k in keys] for entry in cache.values()]
    table = Table(rows=rows, names=keys)
    for name in ("cache_key", "parity", "source", "note"):
        table[name] = np.array([str(v) for v in table[name]], dtype=str)
    table.write(path, format="ascii.ecsv", overwrite=True)


Solver = Callable[..., object]

#: Default ASTAP search radius (deg). Mount pointings in the archive can be
#: several degrees off; 3 deg missed real frames in the test data.
DEFAULT_SEARCH_RADIUS = 15.0

#: Allowed relative deviation of a solved pixel scale from the header
#: prediction (FOCALLEN, XPIXSZ). Larger deviations are false solutions.
SCALE_TOLERANCE = 0.10


def check_plausible(result: OrientationResult, row: Mapping[str, object]) -> OrientationResult:
    """Reject solutions whose pixel scale contradicts the header geometry."""
    if not result.solved:
        return result
    predicted = pixel_scale_arcsec(row)
    if math.isfinite(predicted) and predicted > 0:
        ratio = result.scale_arcsec / predicted if result.scale_arcsec > 0 else 0.0
        if abs(ratio - 1.0) > SCALE_TOLERANCE:
            return OrientationResult(
                result.frame_id, source=SOURCE_FAILED,
                note=f"implausible solution: scale {result.scale_arcsec:.3f}\"/px, "
                     f"header predicts {predicted:.3f}\"/px",
            )
    return result


def solve_frame(
    row: Mapping[str, object],
    *,
    work_dir: str | Path,
    timeout: float = 120.0,
    solver: Solver = solve_astap_copy,
    radius_deg: float = DEFAULT_SEARCH_RADIUS,
) -> OrientationResult:
    """Plate-solve one frame (server solution first, then the local solver)."""
    server = orientation_from_server(row)
    if server is not None:
        return check_plausible(server, row)
    frame_id = str(row.get("frame_id"))
    path = str(row.get("local_path") or "")
    if not path or not Path(path).is_file():
        return OrientationResult(frame_id, note="no local file")
    scale = pixel_scale_arcsec(row)
    fov = field_height_deg(row) if math.isfinite(scale) else None
    ra, dec = _float(row.get("ra")), _float(row.get("dec"))
    solved = solver(
        path,
        work_dir=work_dir,
        fov_deg=fov,
        ra_deg=ra if math.isfinite(ra) else None,
        dec_deg=dec if math.isfinite(dec) else None,
        radius_deg=radius_deg,
        timeout=timeout,
    )
    if solved is None:
        return OrientationResult(frame_id, note=f"no plate solution (radius {radius_deg:g} deg)")
    pa, parity, scale = position_angle_from_wcs(solved)
    try:
        ny, nx = int(_float(row.get("naxis2"))), int(_float(row.get("naxis1")))
        center = solved.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        c_ra, c_dec = float(center.ra.deg), float(center.dec.deg)
    except (ValueError, TypeError, AttributeError):
        c_ra, c_dec = float(solved.wcs.crval[0]), float(solved.wcs.crval[1])
    return check_plausible(
        OrientationResult(frame_id, pa, parity, scale, c_ra, c_dec, SOURCE_ASTAP), row
    )


def solve_orientations(
    lights: Table,
    indices: Sequence[int],
    *,
    work_dir: str | Path,
    cache_path: str | Path | None = None,
    timeout: float = 120.0,
    solver: Solver = solve_astap_copy,
    progress: Callable[[int, int, str], None] | None = None,
    results: dict[str, OrientationResult] | None = None,
    radius_deg: float = DEFAULT_SEARCH_RADIUS,
) -> dict[str, OrientationResult]:
    """Solve ``indices`` of ``lights``; results keyed by ``frame_id``.

    Solutions (and failures) are cached by file checksum in ``cache_path``.
    """
    rows = [dict(zip(lights.colnames, r, strict=True)) for r in lights]
    results = dict(results or {})
    cache = read_orientation_cache(cache_path) if cache_path else {}
    todo = [i for i in indices if str(rows[i]["frame_id"]) not in results]
    for count, i in enumerate(todo, start=1):
        row = rows[i]
        frame_id = str(row["frame_id"])
        if progress is not None:
            progress(count, len(todo), str(row.get("file_name")))
        key = _cache_key(row)
        entry = cache.get(key)
        retry = False
        if entry is not None and str(entry.get("source")) == SOURCE_FAILED:
            # Retry failures that were searched with a smaller radius (or an
            # unknown one); implausible solutions would only come back again.
            cached_radius = _float(entry.get("radius_deg"))
            smaller = not math.isfinite(cached_radius) or cached_radius < radius_deg
            retry = smaller and "implausible" not in str(entry.get("note", ""))
        if entry is not None and not retry:
            results[frame_id] = OrientationResult(
                frame_id,
                _float(entry.get("pa_deg")),
                str(entry.get("parity", "")),
                _float(entry.get("scale_arcsec")),
                _float(entry.get("center_ra")),
                _float(entry.get("center_dec")),
                str(entry.get("source", SOURCE_FAILED)),
                str(entry.get("note", "")),
            )
            results[frame_id] = check_plausible(results[frame_id], row)
            continue
        result = solve_frame(row, work_dir=work_dir, timeout=timeout, solver=solver,
                             radius_deg=radius_deg)
        results[frame_id] = result
        if result.source != SOURCE_SERVER:
            entry = asdict(result)
            entry.pop("frame_id")
            entry["cache_key"] = key
            entry["radius_deg"] = radius_deg
            cache[key] = entry
    if cache_path:
        write_orientation_cache(cache_path, cache)
    return results


def refine_orientation_changes(
    lights: Table,
    results: dict[str, OrientationResult],
    *,
    work_dir: str | Path,
    cache_path: str | Path | None = None,
    pa_tolerance: float = 0.5,
    timeout: float = 120.0,
    solver: Solver = solve_astap_copy,
    max_extra: int = 200,
    radius_deg: float = DEFAULT_SEARCH_RADIUS,
) -> dict[str, OrientationResult]:
    """Bisect between solved neighbours of the same camera that disagree.

    Narrows each orientation change down to two adjacent frames (or stops
    when frames between cannot be solved, or after ``max_extra`` solves).
    """
    rows = [dict(zip(lights.colnames, r, strict=True)) for r in lights]
    jd = np.asarray(lights["jd"], dtype=float)
    frame_ids = [str(r["frame_id"]) for r in rows]
    extra = 0
    failed: set[int] = set()
    while extra < max_extra:
        target = None
        for block in _blocks(rows, jd, gap_days=1e9):
            solved = [i for i in block if frame_ids[i] in results and results[frame_ids[i]].solved]
            for a, b in zip(solved[:-1], solved[1:], strict=True):
                ra_, rb_ = results[frame_ids[a]], results[frame_ids[b]]
                if orientation_difference_mod180(ra_.pa_deg, rb_.pa_deg) <= pa_tolerance:
                    continue
                pos_a, pos_b = block.index(a), block.index(b)
                between = [i for i in block[pos_a + 1 : pos_b]
                           if frame_ids[i] not in results and i not in failed]
                if between:
                    target = between[len(between) // 2]
                    break
            if target is not None:
                break
        if target is None:
            break
        results = solve_orientations(
            lights, [target], work_dir=work_dir, cache_path=cache_path, timeout=timeout,
            solver=solver, results=results, radius_deg=radius_deg,
        )
        if not results[frame_ids[target]].solved:
            failed.add(target)
        extra += 1
    return results


def orientation_table(results: Mapping[str, OrientationResult]) -> Table:
    rows = [asdict(r) | {"pa_mod180": r.pa_mod180} for r in results.values()]
    if not rows:
        return Table()
    table = Table(rows=rows)
    for name in ("frame_id", "parity", "source", "note"):
        table[name] = np.array([str(v) for v in table[name]], dtype=str)
    return table


__all__ = [
    "DEFAULT_SEARCH_RADIUS",
    "OrientationResult",
    "SCALE_TOLERANCE",
    "check_plausible",
    "SOURCE_ASTAP",
    "SOURCE_FAILED",
    "SOURCE_SERVER",
    "orientation_from_server",
    "orientation_table",
    "read_orientation_cache",
    "refine_orientation_changes",
    "sample_frames_for_solving",
    "solve_frame",
    "solve_orientations",
    "write_orientation_cache",
]
