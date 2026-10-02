"""Register and stack the reduced lights of a calibration plan per target.

Calibration happened per reduction unit (mount session x electronic
setup); stacking happens per target, camera and filter over all sessions
and nights in which the target was observed. All frames of a target -
every filter and camera - are registered onto one reference grid (the
sharpest frame of the finest pixel scale), so the stacks of one target
line up pixel by pixel.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table

from ...archive.cache import safe_name
from ...fits_headers import clear_cosmics_identified, cosmics_identified
from .. import registration
from ..detector_noise import (
    READ_NOISE_KEY,
    SATURATION_KEY,
    stack_noise_values,
    write_noise_header,
)
from ..frame_selection import (
    GLOBAL_REFERENCE_KEY,
    FrameSelection,
    mark_reference_frames,
    mark_selection,
    merge_alignment_result,
    rank_frames,
    stack_weights,
    write_quality_table,
)
from ..grouping.setup_keys import camera_id, filter_name
from ..grouping.targets import normalize_name
from ..quality import (
    measure_directory_quality,
    move_rejected_frames,
    plot_frame_quality,
    write_quality_to_headers,
    write_weights_to_headers,
)
from ..storage import cast_like
from ..weighted_combine import weighted_average_combine
from .stack import stack_filter_images, stack_meta_for_filter

CAMERA_COMBINATIONS = ("separate", "combine")
LIGHT_TYPES = ["Light Frame", "Light", "LIGHT"]


@dataclass
class StackSettings:
    frame_selection: Mapping[str, object] | FrameSelection | None = None
    stack_weighting: str = "fwhm"
    stack_method: str = "average"
    shift_method: str = "wcs"
    wcs_method: str = "astap"
    camera_combination: str = "separate"
    keep_aligned_lights: bool = True
    #: With ``keep_aligned_lights``: False deletes the reduced frame (in
    #: ``reduced/<unit>/``) of every aligned frame, so each is stored once.
    keep_reduced_lights: bool = False
    min_frames: int = 1
    stack_below_min_frames: bool = False
    n_cores_multiprocessing: int | None = None
    dtype: str | None = None


def _selected_targets(plan: Mapping, wanted: Sequence[str] | None) -> list[tuple[str, dict]]:
    targets = []
    keys = {normalize_name(w) for w in wanted} if wanted else None
    for tid, info in (plan.get("targets") or {}).items():
        if not info.get("stack", True):
            continue
        if keys is not None and normalize_name(tid) not in keys and normalize_name(
            info.get("name")
        ) not in keys:
            continue
        targets.append((tid, dict(info)))
    return targets


def remove_aligned_originals(originals: Mapping[str, str | Path], aligned_dir: str | Path) -> int:
    """Delete the reduced frames whose aligned copy is kept.

    ``originals`` maps the frame names in ``aligned_dir`` to the reduced
    files they were made from. Returns the number removed; frames without
    an aligned copy (alignment failed, rejected) are kept.
    """
    aligned_dir = Path(aligned_dir)
    removed = 0
    for name, original in originals.items():
        original = Path(original)
        if (aligned_dir / name).is_file() and original.is_file():
            original.unlink()
            removed += 1
    return removed


def _reference_file(table: Table) -> str | None:
    """Sharpest kept frame of the camera with the finest pixel scale."""
    if len(table) == 0:
        return None
    kept = ~np.asarray(table["rejected"], dtype=bool)
    fwhm = np.asarray(table["fwhm_px"], dtype=float)
    scale = np.asarray(table["pixel_scale"], dtype=float)
    cameras = np.asarray(table["camera"]).astype(str)
    candidates = np.flatnonzero(kept & np.isfinite(fwhm))
    if candidates.size == 0:
        candidates = np.flatnonzero(kept) if np.any(kept) else np.arange(len(table))
        return str(table["file"][candidates[0]])
    per_camera = {}
    for cam in set(cameras[candidates]):
        values = scale[candidates][cameras[candidates] == cam]
        values = values[np.isfinite(values)]
        per_camera[cam] = float(np.median(values)) if values.size else math.inf
    finest = min(per_camera, key=per_camera.get)
    pool = candidates[cameras[candidates] == finest]
    return str(table["file"][rank_frames(table, pool, key="fwhm_px")[0]])


def combine_camera_stacks(
    paths: Sequence[str | Path], out_path: str | Path, *, cameras: Sequence[str]
) -> Path:
    """Noise-weighted average of per-camera stacks that share one pixel grid.

    Weights are the inverse median variance of each stack (from its
    uncertainty extension, else a robust pixel scatter).
    """
    ccds = [CCDData.read(p) for p in paths]
    weights = []
    for ccd in ccds:
        if ccd.uncertainty is not None:
            var = float(np.nanmedian(np.asarray(ccd.uncertainty.array, dtype=float) ** 2))
        else:
            data = np.asarray(ccd.data, dtype=float)
            var = (1.4826 * float(np.nanmedian(np.abs(data - np.nanmedian(data))))) ** 2
        weights.append(1.0 / var if var > 0 and math.isfinite(var) else 1.0)
    weights = np.asarray(weights) / np.mean(weights)
    combined = weighted_average_combine(ccds, weights, sigma_clip=False)
    combined.meta = ccds[0].meta.copy()
    combined.meta["EXPTIME"] = float(sum(float(c.meta.get("EXPTIME", 0.0)) for c in ccds))
    combined.meta["N-IMAGES"] = int(sum(int(c.meta.get("N-IMAGES", 1)) for c in ccds))
    read_noise, saturation = stack_noise_values(
        [c.meta for c in ccds], weights, rate_images=combined.unit == u.electron / u.s,
        total_exptime=combined.meta["EXPTIME"],
    )
    for key in (READ_NOISE_KEY, SATURATION_KEY):
        if key in combined.meta:
            del combined.meta[key]
    write_noise_header(combined.meta, read_noise, saturation)
    if not all(cosmics_identified(c.meta) for c in ccds):
        clear_cosmics_identified(combined.meta)
    combined.meta["NCAMERAS"] = (len(ccds), "Cameras combined")
    combined.meta["CAMERAS"] = (",".join(cameras)[:68], "Cameras combined")
    combined.meta.add_history(
        "Noise-weighted camera combination: "
        + ", ".join(f"{c}={w:.2f}" for c, w in zip(cameras, weights, strict=True))
    )
    cast_like(combined, ccds[0])
    out_path = Path(out_path)
    combined.write(out_path, overwrite=True)
    return out_path


def stack_target(
    target_id: str,
    info: Mapping[str, object],
    frames: Table,
    reduced_paths: Mapping[str, str],
    out_dir: str | Path,
    settings: StackSettings,
    *,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Select, register and stack all reduced lights of one target."""
    name = str(info.get("name") or target_id)
    target_dir = Path(out_dir) / "stacks" / safe_name(name)
    light_dir = target_dir / "light"
    if light_dir.exists():
        shutil.rmtree(light_dir)
    light_dir.mkdir(parents=True)

    file_rows: dict[str, dict] = {}
    originals: dict[str, Path] = {}
    for row in frames:
        row = dict(zip(frames.colnames, row, strict=True))
        if str(row.get("target_id")) != target_id:
            continue
        path = reduced_paths.get(str(row["frame_id"]))
        if not path or not Path(path).is_file():
            continue
        link = light_dir / Path(path).name
        link.symlink_to(Path(path).resolve())
        file_rows[link.name] = row
        originals[link.name] = Path(path).resolve()
    if not file_rows:
        log(f"Target {name}: no reduced frames.")
        return []

    quality_table = measure_directory_quality(
        light_dir, image_type_list=LIGHT_TYPES,
        n_cores_multiprocessing=settings.n_cores_multiprocessing,
    )
    quality_table["camera"] = np.array(
        [camera_id(file_rows.get(str(f), {})) for f in quality_table["file"]], dtype=str
    )
    group_columns = ("camera", "filter")
    write_quality_to_headers(quality_table, light_dir)
    selection = FrameSelection.from_mapping(settings.frame_selection)
    mark_selection(quality_table, selection, group_columns=group_columns)
    reference = _reference_file(quality_table)
    mark_reference_frames(quality_table, {GLOBAL_REFERENCE_KEY: reference} if reference else {})
    move_rejected_frames(quality_table, light_dir, target_dir / "rejected_lights")
    quality_table["stack_weight"] = stack_weights(
        quality_table, settings.stack_weighting, group_columns=group_columns
    )
    write_weights_to_headers(quality_table, light_dir)

    result = registration.align_images(
        light_dir, target_dir, LIGHT_TYPES, shift_method=settings.shift_method,
        n_cores_multiprocessing=settings.n_cores_multiprocessing,
        image_output_directory="aligned_lights", align_filter_wise=False,
        reference_file_names={GLOBAL_REFERENCE_KEY: reference} if reference else None,
        wcs_method=settings.wcs_method, instrument=None,
    )
    merge_alignment_result(quality_table, result)
    write_quality_table(quality_table, target_dir / "frame_quality.ecsv")
    plot_frame_quality(quality_table, target_dir, selection=selection, blocking=True)

    aligned_dir = target_dir / "aligned_lights"
    summary: list[dict[str, object]] = []
    camera_stacks: dict[str, list[tuple[str, Path]]] = {}
    kept = ~np.asarray(quality_table["rejected"], dtype=bool)
    aligned = np.asarray(quality_table["aligned"], dtype=bool)
    files = [str(f) for f in quality_table["file"]]
    cams = np.asarray(quality_table["camera"]).astype(str)
    filters = [filter_name({"filter": f}) for f in quality_table["filter"]]
    for cam in dict.fromkeys(cams):
        for filt in dict.fromkeys(filters):
            idx = [i for i in range(len(files))
                   if cams[i] == cam and filters[i] == filt and kept[i] and aligned[i]
                   and (aligned_dir / files[i]).is_file()]
            if not idx:
                continue
            if len(idx) < settings.min_frames and not settings.stack_below_min_frames:
                log(f"Target {name}, {cam} {filt}: only {len(idx)} frame(s) "
                    f"(min_frames={settings.min_frames}), not stacked.")
                continue
            paths = [str(aligned_dir / files[i]) for i in idx]
            weights = None
            if settings.stack_weighting != "none":
                weights = np.asarray(quality_table["stack_weight"], dtype=float)[idx]
                weights = np.where(np.isfinite(weights), weights, 1.0)
            subset = quality_table[cams == cam]
            meta = stack_meta_for_filter(subset, str(quality_table["filter"][idx[0]]),
                                         weighting=settings.stack_weighting, stacked_files=paths)
            cam_dir = target_dir / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            file_name = stack_filter_images(
                paths, settings.stack_method, settings.dtype,
                str(quality_table["filter"][idx[0]]), cam_dir, name,
                weights=weights, stack_meta=meta,
            )
            stack_path = cam_dir / file_name
            with fits.open(stack_path, mode="update") as hdul:
                hdul[0].header["TARGETID"] = (target_id[:68], "Target id (pipeline)")
                hdul[0].header["CAMERA"] = (cam[:68], "Camera id (pipeline)")
                hdul.flush()
            header = fits.getheader(stack_path)
            camera_stacks.setdefault(filt, []).append((cam, stack_path))
            summary.append({
                "target_id": target_id, "target_name": name, "camera": cam, "filter": filt,
                "n_images": int(header.get("N-IMAGES", len(paths))),
                "exposure_s": float(header.get("EXPTIME", float("nan"))),
                "fwhm_median_px": float(header.get("FWHMMED", float("nan"))),
                "path": str(stack_path),
            })

    if settings.camera_combination == "combine":
        for filt, stacks in camera_stacks.items():
            out_path = target_dir / f"combined_filter_{safe_name(filt)}.fit"
            if len(stacks) == 1:
                shutil.copyfile(stacks[0][1], out_path)
            else:
                combine_camera_stacks([p for _, p in stacks], out_path,
                                      cameras=[c for c, _ in stacks])
            header = fits.getheader(out_path)
            summary.append({
                "target_id": target_id, "target_name": name, "camera": "combined",
                "filter": filt, "n_images": int(header.get("N-IMAGES", 0)),
                "exposure_s": float(header.get("EXPTIME", float("nan"))),
                "fwhm_median_px": float("nan"), "path": str(out_path),
            })
    if not settings.keep_aligned_lights:
        shutil.rmtree(aligned_dir, ignore_errors=True)
    elif not settings.keep_reduced_lights:
        removed = remove_aligned_originals(originals, aligned_dir)
        if removed:
            log(f"Target {name}: {removed} reduced frame(s) removed, aligned copies kept "
                f"in {aligned_dir}")
    log(f"Target {name}: {len(summary)} stack(s) written to {target_dir}")
    return summary


def stack_planned(
    plan: Mapping,
    frames: Table,
    reduction_report: Table,
    out_dir: str | Path,
    settings: StackSettings | None = None,
    *,
    targets: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> Table:
    """Stack every target of the plan (``stack: true``), optionally only ``targets``.

    Writes ``<out>/stacks/<target>/<camera>/combined_filter_<F>.fit`` and the
    overview ``<out>/stacks/summary.ecsv``, which is also returned.
    """
    settings = settings or StackSettings()
    if settings.camera_combination not in CAMERA_COMBINATIONS:
        raise ValueError(f"camera_combination must be one of {CAMERA_COMBINATIONS}")
    reduced = {str(fid): str(path) for fid, status, path in zip(
        reduction_report["frame_id"], reduction_report["status"],
        reduction_report["reduced_path"], strict=True) if str(status) == "reduced"}
    rows: list[dict[str, object]] = []
    for target_id, info in _selected_targets(plan, targets):
        rows.extend(stack_target(target_id, info, frames, reduced, out_dir, settings, log=log))
    names = ("target_id", "target_name", "camera", "filter", "n_images", "exposure_s",
             "fwhm_median_px", "path")
    summary = Table(rows=[[r[n] for n in names] for r in rows], names=names) if rows else Table(
        names=names, dtype=(str, str, str, str, int, float, float, str))
    path = Path(out_dir) / "stacks" / "summary.ecsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.write(path, format="ascii.ecsv", overwrite=True)
    return summary


__all__ = [
    "CAMERA_COMBINATIONS",
    "StackSettings",
    "combine_camera_stacks",
    "stack_planned",
    "stack_target",
]
