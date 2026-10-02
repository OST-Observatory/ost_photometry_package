"""Reduce a calibration plan: masters per calibration group, lights per unit.

The plan (see :mod:`ost_photometry.reduce.grouping.plan`) lists bias,
dark and flat masters and reduction units (mount session x electronic
setup). Every master is built once in ``<out>/masters/<master_id>/`` with
the existing directory-based master functions; the lights of a unit are
reduced with explicit masters and pixel masks through
:func:`~ost_photometry.reduce.workflow.science.reduce_light_image`, so
masters of different cameras or sessions can never be mixed up.
"""

from __future__ import annotations

import glob
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table

from ... import calibration_parameters, style, terminal_output
from ...archive.cache import frame_link_name
from ...core.parallel import Executor
from ..detector_noise import (
    NoiseMeasurement,
    check_binning_mode,
    check_noise_source,
    measure_detector_noise,
)
from ..grouping.classify import BIAS, DARK, FLAT, LIGHT, header_frame_type
from ..image_collection import image_file_collection
from ..masks import load_pixel_mask_files
from .bias import master_bias
from .config import ReduceConfig
from .constants import REDUCE_STATUS_REDUCED
from .dark import master_dark, reduce_dark
from .flat import master_flat, reduce_flat
from .main import resolve_camera_parameters
from .science import check_cosmic_ray_mode, reduce_light_image, resolve_cosmic_ray_removal

#: IMAGETYP written into staged copies of frames whose header type is wrong.
CANONICAL_IMAGETYP = {BIAS: "Bias Frame", DARK: "Dark Frame", FLAT: "Flat Field",
                      LIGHT: "Light Frame"}


@dataclass
class ReductionSettings:
    #: True, False or "auto" (only for frames not stacked from at least
    #: ``cosmic_ray_auto_min_frames`` frames per target, camera and filter)
    rm_cosmic_rays: bool | str = "auto"
    cosmic_ray_auto_min_frames: int = 7
    mask_cosmic_rays: bool = False
    limiting_contrast_rm_cosmic_rays: float = 5.0
    sigma_clipping_value_rm_cosmic_rays: float = 4.0
    exposure_time_tolerance: float = 0.5
    temperature_tolerance: float = 5.0
    gain: float | None = None
    read_noise: float | None = None
    dark_rate: float | None = None
    saturation_level: float | None = None
    add_hot_bad_pixel_mask: bool = True
    n_cores_multiprocessing: int | None = None
    dtype: str | None = None
    reuse_masters: bool = True
    #: "catalog" or "measured" (per electronic setup, see measure_plan_noise)
    camera_noise_source: str = "catalog"
    #: "auto", "digital" or "charge" (read noise per binned pixel)
    binning_mode: str = "auto"


@dataclass
class UnitReport:
    unit_id: str
    reduced: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def _frame_lookup(frames: Table) -> dict[str, dict]:
    return {str(r["frame_id"]): dict(zip(frames.colnames, r, strict=True)) for r in frames}


def stage_frames(
    rows: Sequence[Mapping[str, object]],
    dest: str | Path,
    frame_type: str,
) -> list[Path]:
    """Put frames of one type into ``dest`` for the directory-based workflow.

    Frames whose header ``IMAGETYP`` already names ``frame_type`` with the
    spelling most frames use are symlinked; all others are copied with the
    corrected ``IMAGETYP`` (raw cache files are never modified).
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    spellings = [str(r.get("imagetyp") or "").strip() for r in rows
                 if header_frame_type(r.get("imagetyp")) == frame_type]
    target_spelling = (max(set(spellings), key=spellings.count) if spellings
                       else CANONICAL_IMAGETYP[frame_type])
    staged: list[Path] = []
    for row in rows:
        source = Path(str(row.get("local_path") or ""))
        if not source.is_file():
            continue
        link = dest / frame_link_name(str(row["frame_id"]), str(row.get("file_name") or source.name))
        if str(row.get("imagetyp") or "").strip() == target_spelling:
            link.symlink_to(source.resolve())
        else:
            with fits.open(source) as hdul:
                hdul[0].header["IMAGETYP"] = target_spelling
                hdul[0].header.add_history(
                    f"IMAGETYP corrected to {target_spelling!r} by the archive pipeline"
                )
                hdul.writeto(link, overwrite=True)
        staged.append(link)
    return staged


def _image_type_dir(spelling_dir: Path, frame_type: str) -> dict[str, list[str]]:
    """Default image types plus the spelling found in a staged directory."""
    types = calibration_parameters.get_image_types()
    for path in sorted(spelling_dir.glob("*"))[:1]:
        value = str(fits.getheader(path).get("IMAGETYP", "")).strip()
        key = {BIAS: "bias", DARK: "dark", FLAT: "flat", LIGHT: "light"}[frame_type]
        if value and value not in types[key]:
            types[key] = [value, *types[key]]
    return types


def _camera_parameters(directory: Path, settings: ReductionSettings,
                       image_type_dir: dict[str, list[str]],
                       measurement: NoiseMeasurement | None = None):
    cfg = ReduceConfig(
        image_path=directory, output_dir=directory, image_type_dir=image_type_dir,
        gain=settings.gain, read_noise=settings.read_noise, dark_rate=settings.dark_rate,
        saturation_level=settings.saturation_level,
        temperature_tolerance=settings.temperature_tolerance,
        ignore_readout_mode_mismatch=True, ignore_instrument_mismatch=True,
        camera_noise_source=settings.camera_noise_source, binning_mode=settings.binning_mode,
    )
    return resolve_camera_parameters(image_file_collection(directory), cfg,
                                     measurement=measurement)


# ---------------------------------------------------------------------------
# Detector noise and cosmic rays
# ---------------------------------------------------------------------------


def _gain_key(electronic_id: str) -> str:
    """Camera, binning, readout mode and gain setting (offset and temperature
    do not change the gain)."""
    return "|".join(str(electronic_id).split("|")[:4])


def measure_plan_noise(
    frames: Table,
    *,
    saturation_level: float | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, NoiseMeasurement]:
    """Read noise per electronic setup and gain per gain setting.

    Read noise comes from the bias (else dark) pairs of each
    ``electronic_id``; the gain from flat pairs of all setups that share
    camera, binning, readout mode and gain setting, each flat measured
    against the zero level of its own setup.
    """
    if "electronic_id" not in frames.colnames:
        return {}
    kinds = np.asarray(frames["frame_type"]).astype(str)
    eids = np.asarray(frames["electronic_id"]).astype(str)
    paths = np.asarray(frames["local_path"]).astype(str)
    per_setup: dict[str, NoiseMeasurement] = {}
    for eid in dict.fromkeys(eids[np.isin(kinds, [BIAS, DARK, FLAT])]):
        sel = eids == eid
        files = {kind: [p for p in paths[sel & (kinds == kind)] if Path(p).is_file()]
                 for kind in (BIAS, DARK, FLAT)}
        measurement = measure_detector_noise(files[BIAS], files[FLAT], files[DARK],
                                             saturation_level=saturation_level)
        if measurement is not None:
            per_setup[eid] = measurement
    gains: dict[str, list[tuple[float, int]]] = {}
    for eid, m in per_setup.items():
        if m.gain is not None:
            gains.setdefault(_gain_key(eid), []).append((m.gain, m.n_flat_pairs))
    result: dict[str, NoiseMeasurement] = {}
    for eid, m in per_setup.items():
        shared = gains.get(_gain_key(eid), [])
        gain = float(np.median([g for g, _n in shared])) if shared else None
        result[eid] = NoiseMeasurement(m.read_noise_adu, gain, m.n_zero_pairs,
                                       sum(n for _g, n in shared), m.zero_source)
        log(f"Detector noise {eid}: {result[eid].describe()}")
    return result


def stacked_frame_counts(plan: Mapping, frames: Table) -> dict[str, int]:
    """Frames per stack (target x camera x filter) for every light frame id.

    Lights of targets with ``stack: false`` count 0 (they are not stacked).
    """
    lookup = _frame_lookup(frames)
    targets = plan.get("targets") or {}
    keys: dict[str, tuple[str, str, str] | None] = {}
    for unit in (plan.get("units") or {}).values():
        for fid in unit.get("lights", []):
            row = lookup.get(str(fid))
            if row is None:
                continue
            tid = str(row.get("target_id", ""))
            stacked = bool(dict(targets.get(tid) or {}).get("stack", tid in targets))
            keys[str(fid)] = (tid, str(row.get("camera", "")),
                              str(row.get("filter") or "").strip()) if stacked else None
    sizes: dict[tuple[str, str, str], int] = {}
    for key in keys.values():
        if key is not None:
            sizes[key] = sizes.get(key, 0) + 1
    return {fid: (sizes[key] if key is not None else 0) for fid, key in keys.items()}


# ---------------------------------------------------------------------------
# Masters
# ---------------------------------------------------------------------------


def _master_files(master_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(master_dir / pattern)))


def _link_into(paths: Sequence[Path], dest: Path) -> None:
    for path in paths:
        link = dest / path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(path.resolve())


def build_masters(
    plan: Mapping,
    frames: Table,
    out_dir: str | Path,
    settings: ReductionSettings,
    *,
    log: Callable[[str], None] = print,
    noise: Mapping[str, NoiseMeasurement] | None = None,
) -> dict[str, Path]:
    """Build every master referenced by the plan; returns ``{master_id: dir}``.

    ``noise`` maps electronic ids to measured detector noise (used with
    ``settings.camera_noise_source == "measured"``).
    """
    noise = noise or {}
    out_dir = Path(out_dir)
    lookup = _frame_lookup(frames)
    masters = dict(plan.get("masters") or {})
    needed: set[str] = set()
    for unit in (plan.get("units") or {}).values():
        needed.update(x for x in (unit.get("bias"), unit.get("darks")) if x)
        needed.update(x for x in dict(unit.get("flats") or {}).values() if x)
    for mid in list(needed):
        spec = masters.get(mid, {})
        needed.update(x for x in (spec.get("bias_id"), spec.get("dark_id")) if x)

    built: dict[str, Path] = {}
    order = sorted(needed, key=lambda m: {BIAS: 0, DARK: 1, FLAT: 2}.get(masters[m]["kind"], 3))
    for mid in order:
        spec = masters[mid]
        kind = spec["kind"]
        mdir = out_dir / "masters" / mid
        done_pattern = {BIAS: "combined_bias.fit", DARK: "combined_dark_*.fit",
                        FLAT: "combined_flat_filter_*.fit"}[kind]
        if settings.reuse_masters and _master_files(mdir, done_pattern):
            built[mid] = mdir
            continue
        mdir.mkdir(parents=True, exist_ok=True)
        rows = [lookup[f] for f in spec.get("frame_ids", []) if f in lookup]
        if not rows:
            log(f"Master {mid}: no frames available, skipped.")
            continue
        log(f"Building master {mid} ({kind}, {len(rows)} frames)...")
        input_dir = mdir / "input"
        stage_frames(rows, input_dir, kind)
        types = _image_type_dir(input_dir, kind)
        camera = _camera_parameters(input_dir, settings, types,
                                    noise.get(str(spec.get("electronic_id", ""))))
        if kind == BIAS:
            master_bias(input_dir, mdir, types, dtype=settings.dtype)
        elif kind == DARK:
            bias_dir = built.get(spec.get("bias_id", ""))
            rm_bias = bias_dir is not None
            if rm_bias:
                _link_into(_master_files(bias_dir, "combined_bias.fit"), mdir)
                reduce_dark(input_dir, mdir, types, gain=camera.gain,
                            read_noise=camera.read_noise,
                            n_cores_multiprocessing=settings.n_cores_multiprocessing)
            master_dark(mdir / "dark" if rm_bias else input_dir, mdir, types, gain=camera.gain,
                        read_noise=camera.read_noise, dark_rate=camera.dark_rate,
                        n_cores_multiprocessing=settings.n_cores_multiprocessing,
                        rm_bias=rm_bias, dtype=settings.dtype)
        else:
            bias_dir = built.get(spec.get("bias_id", ""))
            dark_dir = built.get(spec.get("dark_id", ""))
            if dark_dir is None:
                log(f"Master {mid}: no dark master for the flats, skipped.")
                continue
            rm_bias = bias_dir is not None
            if rm_bias:
                _link_into(_master_files(bias_dir, "combined_bias.fit"), mdir)
            _link_into(_master_files(dark_dir, "combined_dark_*.fit"), mdir)
            reduce_flat(input_dir, mdir, types, gain=camera.gain, read_noise=camera.read_noise,
                        rm_bias=rm_bias,
                        exposure_time_tolerance=settings.exposure_time_tolerance,
                        n_cores_multiprocessing=settings.n_cores_multiprocessing)
            master_flat(mdir / "flat", mdir, types,
                        n_cores_multiprocessing=settings.n_cores_multiprocessing,
                        dtype=settings.dtype)
        shutil.rmtree(input_dir, ignore_errors=True)
        if _master_files(mdir, done_pattern):
            built[mid] = mdir
        else:
            log(f"{style.Bcolors.WARNING}Master {mid} could not be built.{style.Bcolors.ENDC}")
    return built


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------


def _unity_flat(shape: tuple[int, ...], filter_: str) -> CCDData:
    ccd = CCDData(np.ones(shape, dtype=np.float32), unit="electron")
    ccd.meta["FILTER"] = filter_
    return ccd


def reduce_unit(
    unit_id: str,
    unit: Mapping,
    plan: Mapping,
    frames: Table,
    masters: Mapping[str, Path],
    out_dir: str | Path,
    settings: ReductionSettings,
    *,
    log: Callable[[str], None] = print,
    noise: Mapping[str, NoiseMeasurement] | None = None,
    stack_counts: Mapping[str, int] | None = None,
) -> UnitReport:
    """Reduce the lights of one unit into ``<out>/reduced/<unit_id>/``.

    ``stack_counts`` (frame id -> frames in its stack, see
    :func:`stacked_frame_counts`) decides ``rm_cosmic_rays="auto"``.
    """
    report = UnitReport(unit_id)
    noise = noise or {}
    if stack_counts is None:
        stack_counts = stacked_frame_counts(plan, frames)
    out_dir = Path(out_dir)
    lookup = _frame_lookup(frames)
    rows = [lookup[f] for f in unit.get("lights", []) if f in lookup]
    if not rows:
        report.notes.append("no light frames")
        return report
    unit_dir = out_dir / "units" / unit_id
    stage_dir = unit_dir / "input"
    reduced_dir = out_dir / "reduced" / unit_id
    stage_frames(rows, stage_dir, LIGHT)
    if reduced_dir.exists():
        shutil.rmtree(reduced_dir)
    reduced_dir.mkdir(parents=True)

    types = _image_type_dir(stage_dir, LIGHT)
    camera = _camera_parameters(stage_dir, settings, types,
                                noise.get(str(unit.get("electronic_id", ""))))
    report.notes.append(
        f"gain {camera.gain} e-/ADU, read noise {camera.read_noise} e- ({camera.noise_source})"
    )

    bias = None
    bias_dir = masters.get(unit.get("bias") or "")
    dark_spec = dict((plan.get("masters") or {}).get(unit.get("darks") or "", {}))
    if bias_dir is not None and dark_spec.get("bias_id") == unit.get("bias"):
        bias = CCDData.read(_master_files(bias_dir, "combined_bias.fit")[0])
    elif bias_dir is not None:
        report.notes.append("bias not used: the dark master was built without it")
    darks: dict[float, CCDData] = {}
    mask_files: list[Path] = []
    dark_dir = masters.get(unit.get("darks") or "")
    if dark_dir is not None:
        for path in _master_files(dark_dir, "combined_dark_*.fit"):
            ccd = CCDData.read(path)
            darks[float(ccd.header["exptime"])] = ccd
        mask_files += _master_files(dark_dir, "mask_from_dark_*.fit")
    if not darks:
        report.notes.append("no dark master: unit skipped")
        report.skipped = [(str(r["frame_id"]), "no dark master") for r in rows]
        return report

    flats: dict[str, CCDData] = {}
    flat_ids: dict[str, str] = {}
    for filt, mid in dict(unit.get("flats") or {}).items():
        flat_dir = masters.get(mid or "")
        if flat_dir is None:
            continue
        for path in _master_files(flat_dir, "combined_flat_filter_*.fit"):
            ccd = CCDData.read(path)
            flats[str(ccd.header.get("filter", filt))] = ccd
            flat_ids[str(ccd.header.get("filter", filt))] = mid
        mask_files += _master_files(flat_dir, "mask_from_ccdmask_*.fit")

    executor = Executor(settings.n_cores_multiprocessing, n_tasks=len(rows),
                        add_progress_bar=True)
    scheduled: list[tuple[dict, Path]] = []
    staged = {p.name: p for p in stage_dir.iterdir()}
    for row in rows:
        name = frame_link_name(str(row["frame_id"]), str(row.get("file_name")))
        path = staged.get(name)
        if path is None:
            report.skipped.append((str(row["frame_id"]), "file missing"))
            continue
        header = fits.getheader(path)
        filt = str(header.get("FILTER", "")).strip()
        shape = (int(header.get("NAXIS2", 0)), int(header.get("NAXIS1", 0)))
        flat_map = dict(flats)
        if filt not in flat_map:
            if filt in dict(unit.get("excluded_filters") or {}):
                report.skipped.append((str(row["frame_id"]), f"filter {filt} excluded"))
                continue
            flat_map[filt] = _unity_flat(shape, filt)
            report.notes.append(f"filter {filt}: reduced without flat field")
        mask = load_pixel_mask_files(mask_files, shape) if settings.add_hot_bad_pixel_mask else None
        executor.schedule(
            reduce_light_image,
            args=(str(path), bias, darks, flat_map, unit_dir, reduced_dir),
            kwargs={
                "gain": camera.gain,
                "read_noise": camera.read_noise,
                "rm_bias": bias is not None,
                "exposure_time_tolerance": settings.exposure_time_tolerance,
                "add_hot_bad_pixel_mask": settings.add_hot_bad_pixel_mask,
                "rm_cosmic_rays": resolve_cosmic_ray_removal(
                    settings.rm_cosmic_rays, int(stack_counts.get(str(row["frame_id"]), 0)),
                    min_frames=settings.cosmic_ray_auto_min_frames),
                "limiting_contrast_rm_cosmic_rays": settings.limiting_contrast_rm_cosmic_rays,
                "sigma_clipping_value_rm_cosmic_rays": settings.sigma_clipping_value_rm_cosmic_rays,
                "saturation_level": camera.saturation_level,
                "mask_cosmics": settings.mask_cosmic_rays,
                "pixel_mask": mask if mask is not None else np.zeros(shape, dtype=bool),
            },
        )
        scheduled.append((row, reduced_dir / path.name))
    if executor.err is not None:
        raise RuntimeError(f"Reduction of unit {unit_id} failed: {executor.err}")
    executor.wait()
    statuses = list(executor.res)
    if statuses.count(REDUCE_STATUS_REDUCED) < len(scheduled):
        report.notes.append(
            f"{len(scheduled) - statuses.count(REDUCE_STATUS_REDUCED)} frame(s) not reduced"
        )

    session_id = str(unit.get("session_id", ""))
    for row, reduced_path in scheduled:
        if not reduced_path.is_file():
            report.skipped.append((str(row["frame_id"]), "reduction failed"))
            continue
        filt = str(row.get("filter") or "").strip()
        flat_id = flat_ids.get(filt, "")
        flat_spec = dict(plan.get("masters", {}).get(flat_id, {}))
        with fits.open(reduced_path, mode="update") as hdul:
            header = hdul[0].header
            header["IMAGETYP"] = ("Light Frame", "normalised by the archive pipeline")
            header["FRAMEID"] = (str(row["frame_id"]), "Archive / manifest frame id")
            header["CALUNIT"] = (unit_id[:68], "Reduction unit")
            header["SESSID"] = (session_id[:68], "Mount session")
            header["FLATGRP"] = (flat_id[:68] or "none", "Flat master")
            p_flat = flat_spec.get("probability")
            if p_flat is not None and math.isfinite(float(p_flat)):
                header["PFLAT"] = (round(float(p_flat), 4), "Flat assignment probability")
            header["TARGETID"] = (str(row.get("target_id", ""))[:68], "Target id (pipeline)")
            header["TARGET"] = (str(row.get("target_name", ""))[:68], "Target name (pipeline)")
            hdul.flush()
        report.reduced.append(str(row["frame_id"]))
    shutil.rmtree(stage_dir, ignore_errors=True)
    log(f"{unit_id}: {len(report.reduced)} reduced, {len(report.skipped)} skipped")
    return report


def reduce_planned(
    plan: Mapping,
    frames: Table,
    out_dir: str | Path,
    settings: ReductionSettings | None = None,
    *,
    units: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, UnitReport], Table]:
    """Build masters and reduce every unit of the plan (non-interactive).

    Returns the unit reports and a table ``frame_id, unit_id, status,
    reduced_path`` (also written to ``<out>/reduction_report.ecsv``).
    """
    settings = settings or ReductionSettings()
    check_noise_source(settings.camera_noise_source)
    check_binning_mode(settings.binning_mode)
    check_cosmic_ray_mode(settings.rm_cosmic_rays)
    out_dir = Path(out_dir)
    noise = (measure_plan_noise(frames, saturation_level=settings.saturation_level, log=log)
             if settings.camera_noise_source == "measured" else {})
    masters = build_masters(plan, frames, out_dir, settings, log=log, noise=noise)
    stack_counts = stacked_frame_counts(plan, frames)
    reports: dict[str, UnitReport] = {}
    rows = []
    for unit_id, unit in (plan.get("units") or {}).items():
        if units is not None and unit_id not in units:
            continue
        terminal_output.print_to_terminal(f"Reduce unit {unit_id}...", indent=1)
        report = reduce_unit(unit_id, unit, plan, frames, masters, out_dir, settings, log=log,
                             noise=noise, stack_counts=stack_counts)
        reports[unit_id] = report
        lookup = _frame_lookup(frames)
        for fid in report.reduced:
            name = frame_link_name(fid, str(lookup[fid].get("file_name")))
            rows.append((fid, unit_id, "reduced", str(out_dir / "reduced" / unit_id / name)))
        for fid, reason in report.skipped:
            rows.append((fid, unit_id, f"skipped: {reason}", ""))
    table = Table(rows=rows, names=("frame_id", "unit_id", "status", "reduced_path"),
                  dtype=(str, str, str, str)) if rows else Table(
        names=("frame_id", "unit_id", "status", "reduced_path"), dtype=(str, str, str, str))
    table.write(out_dir / "reduction_report.ecsv", format="ascii.ecsv", overwrite=True)
    return reports, table


__all__ = [
    "CANONICAL_IMAGETYP",
    "ReductionSettings",
    "UnitReport",
    "build_masters",
    "measure_plan_noise",
    "reduce_planned",
    "reduce_unit",
    "stage_frames",
    "stacked_frame_counts",
]
