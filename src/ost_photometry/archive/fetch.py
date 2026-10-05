"""Fetch the frames of an object or an observation run from the archive.

Science frames of the request are downloaded (``role=target``) together
with every bias / dark / flat candidate of the involved runs and of the
neighbouring runs within ``calib_window_days`` (``role=calibration``).
Other light frames in the window are recorded as metadata only
(``role=context``); they show when a different camera was used at the
telescope, which separates mount sessions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.table import Table

from ..camera_specs import normalize_readout_mode
from .cache import FileCache
from .client import ArchiveClient, ArchiveError
from .manifest import (
    ROLE_CALIBRATION,
    ROLE_CONTEXT,
    ROLE_TARGET,
    apply_header,
    manifest_from_rows,
)

#: Archive exposure-type codes.
BIAS, DARK, FLAT, LIGHT, WAVE, UNKNOWN = "BI", "DA", "FL", "LI", "WA", "UK"
_CALIBRATION_CODES = {BIAS, DARK, FLAT}
_ROLE_PRIORITY = {ROLE_TARGET: 0, ROLE_CALIBRATION: 1, ROLE_CONTEXT: 2}


@dataclass
class FetchReport:
    """What :func:`fetch_dataset` did."""

    mode: str = ""
    request: str = ""
    runs: list[str] = field(default_factory=list)
    neighbour_runs: list[str] = field(default_factory=list)
    targets: dict[str, int] = field(default_factory=dict)
    n_target: int = 0
    n_calibration: int = 0
    n_context: int = 0
    n_downloaded: int = 0
    n_cached: int = 0
    skipped_non_fits: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    #: One line per light setup without matching darks (archive dark finder)
    dark_finder: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"Request: {self.mode} {self.request!r}",
            f"Runs: {', '.join(self.runs) or '-'}",
            f"Neighbour runs (calibration window): {', '.join(self.neighbour_runs) or '-'}",
            f"Frames: {self.n_target} science, {self.n_calibration} calibration candidates, "
            f"{self.n_context} context (metadata only)",
            f"Files: {self.n_downloaded} downloaded, {self.n_cached} from cache, "
            f"{self.skipped_non_fits} non-FITS skipped, {len(self.failures)} failed",
        ]
        if self.targets:
            out.append("Targets in the request (archive names):")
            for name, count in sorted(self.targets.items(), key=lambda kv: (-kv[1], kv[0])):
                out.append(f"  {name or '(no name)'}: {count} frames")
        if self.dark_finder:
            out.append("Dark / bias finder (science frames without matching calibration):")
            out.extend(f"  {line}" for line in self.dark_finder)
        for name, reason in self.failures[:20]:
            out.append(f"  failed: {name}: {reason}")
        return out


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def _code(value: object) -> str:
    return str(value or "").strip().upper()


def is_fits_record(record: dict) -> bool:
    return _code(record.get("file_type")) in {"FITS", "FIT", ""}


def is_spectroscopy_record(record: dict) -> bool:
    return bool(record.get("spectroscopy")) or _code(record.get("spectrograph")) not in {"", "N"}


def record_types(record: dict) -> set[str]:
    """All exposure types the archive assigns (header, ML, user, effective)."""
    types = {_code(record.get("exposure_type")), _code(record.get("effective_exposure_type"))}
    if record.get("exposure_type_user"):
        types.add(_code(record.get("exposure_type_user")))
    if record.get("exposure_type_ml") and not record.get("exposure_type_ml_abstained"):
        types.add(_code(record.get("exposure_type_ml")))
    types.discard("")
    return types


def is_light_record(record: dict) -> bool:
    user = _code(record.get("exposure_type_user"))
    if user:
        return user == LIGHT
    types = record_types(record)
    return LIGHT in types and not (types & _CALIBRATION_CODES)


def is_calibration_candidate(record: dict) -> bool:
    """Bias / dark / flat by any archive classification, or unknown type."""
    user = _code(record.get("exposure_type_user"))
    if user:
        return user in _CALIBRATION_CODES
    types = record_types(record)
    if types & _CALIBRATION_CODES:
        return True
    return types <= {UNKNOWN}


def _finite(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def record_to_row(record: dict, *, role: str) -> dict[str, object]:
    """Manifest row from an archive data-file record."""
    ra, dec = _finite(record.get("ra")), _finite(record.get("dec"))
    if not math.isfinite(ra) or ra < 0 or (ra == 0 and dec == 0):
        ra, dec = float("nan"), float("nan")
    jd = _finite(record.get("hjd"))
    if math.isfinite(jd) and jd <= 0:
        jd = float("nan")
    row: dict[str, object] = {
        "frame_id": str(record.get("pk")),
        "pk": record.get("pk"),
        "run": record.get("observation_run_name", ""),
        "run_pk": record.get("observation_run"),
        "file_name": record.get("file_name", ""),
        "sha256": record.get("content_hash", ""),
        "size": record.get("file_size", 0),
        "role": role,
        "obs_date": record.get("obs_date", ""),
        "jd": jd,
        "exptime": record.get("exptime"),
        "instrument_archive": record.get("instrument", ""),
        "telescope": record.get("telescope", ""),
        "exposure_type": _code(record.get("exposure_type")),
        "exposure_type_ml": _code(record.get("exposure_type_ml")),
        "ml_confidence": record.get("exposure_type_ml_confidence"),
        "exposure_type_user": _code(record.get("exposure_type_user")),
        "spectrograph": _code(record.get("spectrograph")),
        "main_target": record.get("main_target", "") or record.get("header_target_name", ""),
        "main_object_id": record.get("main_object_id") if record.get("main_object_id") else -1,
        "main_object_name": record.get("main_object_name", "") or "",
        "ra": ra,
        "dec": dec,
        "plate_solved": bool(record.get("plate_solved")),
        "naxis1": record.get("naxis1"),
        "naxis2": record.get("naxis2"),
    }
    for key in ("wcs_ra", "wcs_dec", "wcs_cd1_1", "wcs_cd1_2", "wcs_cd2_1", "wcs_cd2_2"):
        row[key] = record.get(key)
    return row


def normalize_target_name(name: object) -> str:
    """Case- and space-insensitive comparison key (``"M 57"`` == ``"m57"``)."""
    return "".join(str(name or "").lower().split()).replace("_", "").replace("-", "")


def record_matches_targets(record: dict, targets: Sequence[str]) -> bool:
    wanted = {normalize_target_name(t) for t in targets}
    names = (
        record.get("main_object_name"),
        record.get("main_target"),
        record.get("header_target_name"),
    )
    return any(normalize_target_name(n) in wanted for n in names if n)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _neighbour_runs(
    all_runs: Iterable[dict], centre_runs: Iterable[dict], window_days: float
) -> list[dict]:
    centres = [
        _finite(r.get("mid_observation_jd")) for r in centre_runs
    ]
    centres = [c for c in centres if math.isfinite(c) and c > 0]
    centre_pks = {r.get("pk") for r in centre_runs}
    out: list[dict] = []
    for run in all_runs:
        if run.get("pk") in centre_pks:
            continue
        jd = _finite(run.get("mid_observation_jd"))
        if not (math.isfinite(jd) and jd > 0):
            continue
        if any(abs(jd - c) <= window_days for c in centres):
            out.append(run)
    return out


def _choose_object(client: ArchiveClient, name: str) -> dict:
    candidates = client.search_objects(name)
    if not candidates:
        raise ArchiveError(f"No archive object matches {name!r}.")
    key = normalize_target_name(name)
    for obj in candidates:
        names = [obj.get("name")] + [i.get("name") for i in obj.get("identifiers", []) or []]
        if any(normalize_target_name(n) == key for n in names if n):
            return obj
    return candidates[0]


def collect_records(
    client: ArchiveClient,
    *,
    object_name: str | None = None,
    run_name: str | None = None,
    targets: Sequence[str] | None = None,
    calib_window_days: float = 7.0,
    report: FetchReport | None = None,
) -> list[dict[str, object]]:
    """Manifest rows (no downloads yet) for an object or a run request."""
    if bool(object_name) == bool(run_name):
        raise ValueError("Give exactly one of object_name or run_name.")
    report = report if report is not None else FetchReport()
    rows: dict[int, dict[str, object]] = {}

    def add(record: dict, role: str) -> None:
        if not is_fits_record(record):
            report.skipped_non_fits += 1
            return
        pk = int(record.get("pk"))
        current = rows.get(pk)
        if current is None or _ROLE_PRIORITY[role] < _ROLE_PRIORITY[str(current["role"])]:
            rows[pk] = record_to_row(record, role=role)

    def add_run_files(run: dict, *, science: bool) -> None:
        for record in client.datafiles(run_pk=int(run["pk"])):
            if is_light_record(record):
                if science and not is_spectroscopy_record(record) and (
                    not targets or record_matches_targets(record, targets)
                ):
                    add(record, ROLE_TARGET)
                else:
                    add(record, ROLE_CONTEXT)
            elif is_calibration_candidate(record):
                add(record, ROLE_CALIBRATION)
            else:
                add(record, ROLE_CONTEXT)

    if object_name:
        report.mode, report.request = "object", object_name
        obj = _choose_object(client, object_name)
        science_pks = set()
        for record in client.object_datafiles(int(obj["pk"])):
            if is_light_record(record) and not is_spectroscopy_record(record):
                add(record, ROLE_TARGET)
                science_pks.add(int(record["pk"]))
        centre_runs = client.object_runs(int(obj["pk"]))
        for run in centre_runs:
            add_run_files(run, science=False)
    else:
        report.mode, report.request = "run", str(run_name)
        centre_runs = [client.find_run(str(run_name))]
        add_run_files(centre_runs[0], science=True)

    report.runs = [str(r.get("name")) for r in centre_runs]
    if calib_window_days > 0:
        all_runs = client.runs(ordering="mid_observation_jd")
        neighbours = _neighbour_runs(all_runs, centre_runs, calib_window_days)
        report.neighbour_runs = [str(r.get("name")) for r in neighbours]
        for run in neighbours:
            add_run_files(run, science=False)

    result = list(rows.values())
    report.n_target = sum(1 for r in result if r["role"] == ROLE_TARGET)
    report.n_calibration = sum(1 for r in result if r["role"] == ROLE_CALIBRATION)
    report.n_context = sum(1 for r in result if r["role"] == ROLE_CONTEXT)
    names: dict[str, int] = {}
    for r in result:
        if r["role"] == ROLE_TARGET:
            key = str(r.get("main_object_name") or r.get("main_target") or "")
            names[key] = names.get(key, 0) + 1
    report.targets = names
    return result


def download_rows(
    client: ArchiveClient,
    rows: list[dict[str, object]],
    cache: FileCache,
    *,
    roles: Sequence[str] = (ROLE_TARGET, ROLE_CALIBRATION),
    report: FetchReport | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Download (or reuse cached) files of ``rows`` and merge their headers."""
    report = report if report is not None else FetchReport()
    todo = [r for r in rows if r["role"] in roles]
    for i, row in enumerate(todo, start=1):
        name = str(row.get("file_name"))
        sha = str(row.get("sha256") or "")
        suffix = Path(name).suffix.lower() or ".fit"
        if progress is not None:
            progress(i, len(todo), name)
        try:
            cached = bool(sha) and cache.has(sha, suffix)
            if not sha:
                raise ArchiveError("archive record has no checksum")
            path = cache.path_for(sha, suffix)
            if not cached:
                client.download(int(row["pk"]), path, expected_sha256=sha)
                cache.protect(path)
            header = fits.getheader(path)
        except (ArchiveError, OSError, ValueError) as exc:
            report.failures.append((name, str(exc)[:160]))
            continue
        if cached:
            report.n_cached += 1
        else:
            report.n_downloaded += 1
        row["local_path"] = str(path)
        row["downloaded"] = True
        apply_header(row, header)


# ---------------------------------------------------------------------------
# Missing darks: archive dark finder
# ---------------------------------------------------------------------------


@dataclass
class DarkNeed:
    """Light frames of one electronic setup (and exposure time) without darks / bias."""

    setup: tuple
    exptime: float
    temperature: float
    jd: float
    example: dict[str, object]
    n_frames: int = 0
    kind: str = "dark"

    def describe(self) -> str:
        camera, nx, ny, xbin, ybin, gain, offset, mode = self.setup
        what = f"{self.exptime:g} s darks" if self.kind == "dark" else "bias"
        return (f"{what}: {camera} {nx}x{ny} bin {xbin}x{ybin} gain {gain:g} "
                f"offset {offset:g} {mode or '?'} at {self.temperature:.0f} C "
                f"({self.n_frames} lights)")


def _number(row: dict, key: str) -> float:
    return _finite(row.get(key))


def _temperature(row: dict) -> float:
    temp = _number(row, "set_temp")
    return temp if math.isfinite(temp) else _number(row, "ccd_temp")


def _integer(row: dict, key: str, default: int) -> int:
    value = _number(row, key)
    return int(value) if math.isfinite(value) else default


def _setup_key(row: dict) -> tuple:
    """Camera, image size, binning, gain, offset and readout mode of a frame."""
    camera = str(row.get("instrument_archive") or row.get("instrume") or "").strip()
    gain, offset = _number(row, "gain"), _number(row, "offset")
    return (
        camera,
        _integer(row, "naxis1", 0),
        _integer(row, "naxis2", 0),
        _integer(row, "xbinning", 1),
        _integer(row, "ybinning", _integer(row, "xbinning", 1)),
        gain if math.isfinite(gain) else -1.0,
        offset if math.isfinite(offset) else -1.0,
        normalize_readout_mode(str(row.get("readoutm") or "")) or "",
    )


def _row_is(row: dict, code: str, word: str) -> bool:
    user = _code(row.get("exposure_type_user"))
    if user:
        return user == code
    return (word in str(row.get("imagetyp") or "").lower()
            or code in {_code(row.get("exposure_type")), _code(row.get("exposure_type_ml"))})


def _row_is_dark(row: dict) -> bool:
    return _row_is(row, DARK, "dark")


def _row_is_bias(row: dict) -> bool:
    return _row_is(row, BIAS, "bias")


def exposure_tolerance(exptime: float, absolute: float = 0.5, fraction: float = 0.05) -> float:
    """Dark / light exposure tolerance ``max(absolute, fraction * exptime)``
    (the same rule as the grouping)."""
    return max(float(absolute), float(fraction) * abs(float(exptime)))


def _same_temperature(a: float, b: float, tolerance: float) -> bool:
    return not math.isfinite(a) or not math.isfinite(b) or abs(a - b) <= tolerance


def dark_needs(
    rows: Sequence[dict[str, object]],
    *,
    exptime_tolerance: float = 0.5,
    exptime_tolerance_fraction: float = 0.05,
    temp_tolerance: float = 2.0,
) -> list[DarkNeed]:
    """Exposure times of the downloaded science frames without a matching dark."""
    darks = [r for r in rows if r.get("local_path") and _row_is_dark(r)]
    needs: dict[tuple, DarkNeed] = {}
    for row in rows:
        if row.get("role") != ROLE_TARGET or not row.get("local_path"):
            continue
        exptime, temp = _number(row, "exptime"), _temperature(row)
        if not math.isfinite(exptime):
            continue
        setup = _setup_key(row)
        tolerance = exposure_tolerance(exptime, exptime_tolerance, exptime_tolerance_fraction)
        covered = any(
            _setup_key(d) == setup
            and abs(_number(d, "exptime") - exptime) <= tolerance
            and _same_temperature(temp, _temperature(d), temp_tolerance)
            for d in darks
        )
        if covered:
            continue
        key = (setup, round(exptime, 1), round(temp) if math.isfinite(temp) else None)
        need = needs.get(key)
        if need is None:
            need = needs[key] = DarkNeed(setup, exptime, temp, _number(row, "jd"), dict(row))
        need.n_frames += 1
    return list(needs.values())


def bias_needs(
    rows: Sequence[dict[str, object]],
    *,
    temp_tolerance: float = 2.0,
) -> list[DarkNeed]:
    """Setups of the downloaded science frames without any bias frame."""
    biases = [r for r in rows if r.get("local_path") and _row_is_bias(r)]
    needs: dict[tuple, DarkNeed] = {}
    for row in rows:
        if row.get("role") != ROLE_TARGET or not row.get("local_path"):
            continue
        setup, temp = _setup_key(row), _temperature(row)
        if any(_setup_key(b) == setup and _same_temperature(temp, _temperature(b), temp_tolerance)
               for b in biases):
            continue
        key = (setup, round(temp) if math.isfinite(temp) else None)
        need = needs.get(key)
        if need is None:
            need = needs[key] = DarkNeed(setup, 0.0, temp, _number(row, "jd"), dict(row),
                                         kind="bias")
        need.n_frames += 1
    return list(needs.values())


def fetch_missing_calibrations(
    client: ArchiveClient,
    rows: list[dict[str, object]],
    cache: FileCache,
    *,
    report: FetchReport,
    kinds: Sequence[str] = ("dark", "bias"),
    exptime_tolerance: float = 0.5,
    exptime_tolerance_fraction: float = 0.05,
    temp_tolerance: float = 2.0,
    max_runs: int = 3,
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Add darks / bias for science frames that have none, via the dark finder.

    For every uncovered setup (darks: and exposure time) the matching
    frames of the run closest in time are downloaded (``role=calibration``);
    a run whose frames were taken in another readout mode (not known to the
    finder) is skipped and the next one tried, up to ``max_runs``.
    """
    needs: list[DarkNeed] = []
    if "dark" in kinds:
        needs += dark_needs(rows, exptime_tolerance=exptime_tolerance,
                            exptime_tolerance_fraction=exptime_tolerance_fraction,
                            temp_tolerance=temp_tolerance)
    if "bias" in kinds:
        needs += bias_needs(rows, temp_tolerance=temp_tolerance)
    if not needs:
        return
    if not client.authenticated:
        for need in needs:
            report.dark_finder.append(f"{need.describe()}: not searched (the dark finder "
                                      "needs an archive login)")
        return
    known = {int(r["pk"]) for r in rows if r.get("pk") not in (None, "", -1)}
    for need in needs:
        camera, nx, ny, xbin, ybin, gain, offset, mode = need.setup
        temperature = need.temperature if math.isfinite(need.temperature) else _number(
            need.example, "ccd_temp")
        try:
            found = client.find_calibration_frames(
                need.kind, exptime=need.exptime if need.kind == "dark" else None,
                ccd_temp=temperature, instrument=camera, naxis1=nx, naxis2=ny,
                binning_x=xbin, binning_y=ybin, gain=gain, offset=offset,
                exptime_tolerance=exposure_tolerance(need.exptime, exptime_tolerance,
                                                     exptime_tolerance_fraction),
                temp_tolerance=temp_tolerance,
            )
        except ArchiveError as exc:
            report.dark_finder.append(f"{need.describe()}: dark finder failed: {exc}")
            continue
        found = [
            f for f in found
            if int(f.get("id", -1)) not in known
            and (gain < 0 or _finite(f.get("gain")) == gain)
            and (offset < 0 or _finite(f.get("offset")) == offset)
        ]
        by_run: dict[int, list[dict]] = {}
        for item in found:
            if item.get("observation_run_id") is not None:
                by_run.setdefault(int(item["observation_run_id"]), []).append(item)

        def distance(run_id: int, by_run: dict = by_run, jd: float = need.jd) -> float:
            jds = [_finite(i.get("hjd")) for i in by_run[run_id]]
            jds = [j for j in jds if math.isfinite(j)]
            if not jds or not math.isfinite(jd):
                return math.inf
            return min(abs(j - jd) for j in jds)

        for run_id in sorted(by_run, key=distance)[:max_runs]:
            ids = {int(i["id"]) for i in by_run[run_id]}
            records = [r for r in client.datafiles(run_pk=run_id) if int(r.get("pk", -1)) in ids]
            new_rows = [record_to_row(r, role=ROLE_CALIBRATION) for r in records]
            download_rows(client, new_rows, cache, report=report, progress=progress)
            usable = [r for r in new_rows if r.get("local_path")
                      and normalize_readout_mode(str(r.get("readoutm") or "")) == (mode or None)]
            run_name = by_run[run_id][0].get("observation_run") or run_id
            if usable:
                rows.extend(usable)
                known.update(int(r["pk"]) for r in usable)
                report.n_calibration += len(usable)
                report.dark_finder.append(
                    f"{need.describe()}: {len(usable)} frames from run {run_name} "
                    f"({distance(run_id):.0f} days away)"
                )
                break
            report.dark_finder.append(
                f"{need.describe()}: frames of run {run_name} skipped (other readout mode)"
            )
        if not by_run:
            report.dark_finder.append(f"{need.describe()}: no matching frames in the archive")


def fetch_missing_darks(client: ArchiveClient, rows: list[dict[str, object]], cache: FileCache,
                        **kwargs) -> None:
    """Darks only, see :func:`fetch_missing_calibrations`."""
    fetch_missing_calibrations(client, rows, cache, kinds=("dark",), **kwargs)


def fetch_dataset(
    client: ArchiveClient,
    *,
    object_name: str | None = None,
    run_name: str | None = None,
    targets: Sequence[str] | None = None,
    calib_window_days: float = 7.0,
    cache_dir: str | Path,
    download_context: bool = False,
    use_dark_finder: bool = True,
    finder_kinds: Sequence[str] = ("dark", "bias"),
    exptime_tolerance: float = 0.5,
    exptime_tolerance_fraction: float = 0.05,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[Table, FetchReport]:
    """Collect, download and describe the frames of an object or a run.

    Returns the manifest (see :mod:`ost_photometry.archive.manifest`) and a
    :class:`FetchReport`. Context frames are downloaded only with
    ``download_context=True``. With ``use_dark_finder`` (needs a login),
    science exposures without matching darks (or setups without bias) in the
    calibration window get them from the archive's dark finder
    (:func:`fetch_missing_calibrations`; ``finder_kinds``). Darks match
    within ``max(exptime_tolerance, exptime_tolerance_fraction * t)``.
    """
    report = FetchReport()
    rows = collect_records(
        client,
        object_name=object_name,
        run_name=run_name,
        targets=targets,
        calib_window_days=calib_window_days,
        report=report,
    )
    roles = (ROLE_TARGET, ROLE_CALIBRATION, ROLE_CONTEXT) if download_context else (
        ROLE_TARGET,
        ROLE_CALIBRATION,
    )
    cache = FileCache(cache_dir)
    download_rows(client, rows, cache, roles=roles, report=report, progress=progress)
    if use_dark_finder:
        fetch_missing_calibrations(
            client, rows, cache, report=report, kinds=finder_kinds,
            exptime_tolerance=exptime_tolerance,
            exptime_tolerance_fraction=exptime_tolerance_fraction, progress=progress,
        )
    return manifest_from_rows(rows), report


__all__ = [
    "DarkNeed",
    "FetchReport",
    "collect_records",
    "bias_needs",
    "dark_needs",
    "exposure_tolerance",
    "fetch_missing_calibrations",
    "download_rows",
    "fetch_dataset",
    "fetch_missing_darks",
    "is_calibration_candidate",
    "is_light_record",
    "normalize_target_name",
    "record_matches_targets",
    "record_to_row",
]
