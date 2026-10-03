"""Assign flat sets to mount sessions with a probability.

A flat set is a time block of flats of one camera, binning, filter and
telescope. Its probability of belonging to a mount session combines:

* **certain**: the flats were taken inside the session (between lights of
  the session);
* **incompatible**: different camera or binning, or another session of
  the same telescope lies between the flats and the session (the camera was
  remounted or replaced in between);
* otherwise a prior that decays with the time gap, ``exp(-dt / tau)``,
  multiplied by likelihood factors from the **dust fingerprint** (sensor
  window dust is filter independent, so flats of any filter that certainly
  belong to the session serve as reference) and the **vignetting** (only
  coarse rotations or a different adapter are visible).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import gaussian_filter, zoom

from .classify import FLAT
from .sessions import Session
from .setup_keys import binning, camera_id, filter_name, telescope_id

CERTAIN, LIKELY, UNCERTAIN, REJECTED = "certain", "likely", "uncertain", "rejected"
NO_FLAT_POLICIES = ("best_available", "skip_flat", "exclude_lights")

DEFAULT_FLAT_SETTINGS: dict[str, float] = {
    "tau_days": 2.0,
    "block_gap_hours": 2.0,
    "p_certain": 0.95,
    "p_likely": 0.7,
    "p_uncertain": 0.4,
    "dust_same": 0.6,  # normalised correlation: same dust state
    "dust_changed": 0.2,  # normalised correlation: dust changed
    "lr_dust_same": 3.0,
    "lr_dust_changed": 0.2,
    "vignetting_floor": 0.004,  # log-ratio rms between filters of one night
    "lr_vignetting": 0.1,
    "max_frames": 6,
}


def _float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


@dataclass
class FlatSet:
    set_id: str
    camera: str
    binning: str
    filter: str
    telescope: str
    electronic_id: str
    start_jd: float
    end_jd: float
    frame_ids: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    #: Observation runs (archive names or directory names) of the frames
    runs: list[str] = field(default_factory=list)

    @property
    def mid_jd(self) -> float:
        return 0.5 * (self.start_jd + self.end_jd)

    def describe(self) -> str:
        """``set_id`` with frame count and source run(s), for reports."""
        runs = f", run {', '.join(self.runs)}" if self.runs else ""
        return f"{self.set_id} ({len(self.frame_ids)} frames{runs})"


@dataclass
class FlatCandidate:
    set_id: str
    probability: float
    category: str
    reason: str


@dataclass
class FlatAssignment:
    """Flat choice for the lights of one session, filter and binning."""

    session_id: str
    filter: str
    binning: str
    flat_set_ids: list[str] = field(default_factory=list)
    probability: float = 0.0
    category: str = REJECTED
    note: str = ""
    candidates: list[FlatCandidate] = field(default_factory=list)


def flat_sets(frames: Table, *, block_gap_hours: float = 2.0) -> list[FlatSet]:
    """Flat sets: flats of one camera / binning / filter / telescope in a time block."""
    rows = [dict(zip(frames.colnames, r, strict=True)) for r in frames
            if str(r["frame_type"]) == FLAT]
    rows.sort(key=lambda r: np.nan_to_num(_float(r.get("jd")), nan=np.inf))
    sets: list[FlatSet] = []
    open_sets: dict[tuple, FlatSet] = {}
    for row in rows:
        key = (camera_id(row), binning(row), filter_name(row), telescope_id(row),
               str(row.get("electronic_id", "")))
        jd = _float(row.get("jd"))
        current = open_sets.get(key)
        if current is None or not math.isfinite(jd) or not math.isfinite(current.end_jd) or (
            jd - current.end_jd > block_gap_hours / 24.0
        ):
            current = FlatSet("", key[0], key[1], key[2], key[3], key[4], jd, jd)
            sets.append(current)
            open_sets[key] = current
        current.frame_ids.append(str(row["frame_id"]))
        current.paths.append(str(row.get("local_path") or ""))
        run = str(row.get("run") or "").strip()
        if run and run not in current.runs:
            current.runs.append(run)
        if math.isfinite(jd):
            current.end_jd = jd if not math.isfinite(current.end_jd) else max(current.end_jd, jd)
    counters: dict[str, int] = {}
    from .darks import night_of

    for s in sets:
        night = night_of(s.start_jd)
        base = f"FS_{night}_{s.camera}_{s.binning}_{s.filter}"
        counters[base] = counters.get(base, 0) + 1
        s.set_id = base if counters[base] == 1 else f"{base}_{counters[base]}"
    return sets


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------


def _load_normalised(path: str, bias_level: float) -> np.ndarray | None:
    try:
        data = fits.getdata(path).astype(np.float32)
    except (OSError, ValueError, TypeError):
        return None
    if data.ndim != 2:
        return None
    ny, nx = data.shape
    data = data[: ny // 2 * 2, : nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    data = data - bias_level
    level = float(np.median(data))
    if not math.isfinite(level) or level <= 0:
        return None
    return data / level


@dataclass
class FlatImages:
    master: np.ndarray | None
    half_a: np.ndarray | None
    half_b: np.ndarray | None


def load_flat_images(flat_set: FlatSet, bias_level: float, max_frames: int = 6) -> FlatImages:
    """Median master (2x2 binned, normalised) and two half masters."""
    frames = [f for f in (_load_normalised(p, bias_level)
                          for p in flat_set.paths[:max_frames] if p) if f is not None]
    if not frames:
        return FlatImages(None, None, None)
    master = np.median(frames, axis=0)
    if len(frames) >= 4:
        half = len(frames) // 2
        return FlatImages(master, np.median(frames[:half], axis=0), np.median(frames[half:], axis=0))
    return FlatImages(master, None, None)


def dust_map(master: np.ndarray, sigma: float = 30.0, border_fraction: float = 0.08) -> np.ndarray:
    """Small-scale structure (dust donuts) of a normalised flat; border removed."""
    hp = master / gaussian_filter(master, sigma)
    ny, nx = hp.shape
    # Cut at least 1.5 sigma: the smoothing creates identical edge artefacts
    # in every map, which would inflate the correlation.
    b = max(int(border_fraction * min(ny, nx)), int(1.5 * sigma))
    return hp[b : ny - b, b : nx - b]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def self_correlation(images: FlatImages) -> float:
    if images.half_a is None or images.half_b is None:
        return float("nan")
    return correlation(dust_map(images.half_a), dust_map(images.half_b))


def dust_similarity(a: FlatImages, b: FlatImages) -> tuple[float, float]:
    """``(r, r_normalised)``; ``r_normalised`` divides by the half-set
    self-correlations, so noisy (low-signal) sets are not mistaken for a
    changed dust state."""
    if a.master is None or b.master is None:
        return float("nan"), float("nan")
    r = correlation(dust_map(a.master), dust_map(b.master))
    sa, sb = self_correlation(a), self_correlation(b)
    if math.isfinite(sa) and math.isfinite(sb) and sa > 0.05 and sb > 0.05:
        return r, min(1.5, r / math.sqrt(sa * sb))
    return r, float("nan")


def vignetting_difference(a: np.ndarray, b: np.ndarray) -> float:
    """rms of the large-scale log ratio (plane removed, central circle)."""
    if a.shape != b.shape:
        return float("nan")
    la = gaussian_filter(zoom(a, 0.25, order=1), 4)
    lb = gaussian_filter(zoom(b, 0.25, order=1), 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(la / lb)
    ny, nx = ratio.shape
    yy, xx = np.indices(ratio.shape)
    mask = ((xx - nx / 2) ** 2 + (yy - ny / 2) ** 2 < (0.46 * min(ny, nx)) ** 2) & np.isfinite(ratio)
    if mask.sum() < 50:
        return float("nan")
    design = np.c_[np.ones(mask.sum()), xx[mask], yy[mask]]
    coeff, *_ = np.linalg.lstsq(design, ratio[mask], rcond=None)
    residual = ratio[mask] - design @ coeff
    return float(np.std(residual))


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def _category(probability: float, settings: Mapping[str, float]) -> str:
    if probability >= settings["p_certain"]:
        return CERTAIN
    if probability >= settings["p_likely"]:
        return LIKELY
    if probability >= settings["p_uncertain"]:
        return UNCERTAIN
    return REJECTED


def _session_lookup(sessions: Sequence[Session]) -> dict[str, Session]:
    return {s.session_id: s for s in sessions}


def _between(sessions: Sequence[Session], telescope: str, t0: float, t1: float,
             exclude: str) -> list[Session]:
    lo, hi = min(t0, t1), max(t0, t1)
    return [s for s in sessions if s.telescope == telescope and s.session_id != exclude
            and s.start_jd < hi and s.end_jd > lo]


def _context_breaks(context: Table | None, telescope: str, camera: str,
                    t0: float, t1: float) -> bool:
    if context is None or len(context) == 0:
        return False
    lo, hi = min(t0, t1), max(t0, t1)
    for row in context:
        row = dict(zip(context.colnames, row, strict=True))
        if telescope_id(row) != telescope:
            continue
        cam = camera_id(row)
        if cam in ("unknown", camera):
            continue
        t = _float(row.get("jd"))
        if math.isfinite(t) and lo < t < hi:
            return True
    return False


def score_flat_set(
    flat_set: FlatSet,
    session: Session,
    sessions: Sequence[Session],
    *,
    session_binnings: set[str],
    context: Table | None = None,
    dust_evidence: tuple[float, float] | None = None,
    vignetting_rms: float | None = None,
    settings: Mapping[str, float] = DEFAULT_FLAT_SETTINGS,
) -> FlatCandidate:
    """Probability that ``flat_set`` belongs to ``session``."""
    if flat_set.camera != session.camera or flat_set.telescope != session.telescope:
        return FlatCandidate(flat_set.set_id, 0.0, REJECTED, "different camera or telescope")
    if flat_set.binning not in session_binnings:
        return FlatCandidate(flat_set.set_id, 0.0, REJECTED, "different binning")
    t_flat = flat_set.mid_jd
    if session.start_jd <= t_flat <= session.end_jd:
        return FlatCandidate(flat_set.set_id, 1.0, CERTAIN, "taken during the session")
    edge = session.end_jd if t_flat > session.end_jd else session.start_jd
    others = _between(sessions, session.telescope, t_flat, edge, session.session_id)
    if others:
        names = ", ".join(sorted(s.session_id for s in others))
        return FlatCandidate(flat_set.set_id, 0.0, REJECTED,
                             f"other session(s) in between: {names}")
    if _context_breaks(context, session.telescope, session.camera, t_flat, edge):
        return FlatCandidate(flat_set.set_id, 0.0, REJECTED,
                             "another camera was used at the telescope in between")

    gap = abs(t_flat - edge)
    prior = min(0.99, math.exp(-gap / settings["tau_days"]))
    odds = prior / (1.0 - prior)
    reasons = [f"{gap * 24:.1f} h from the session"]
    if dust_evidence is not None:
        r, r_norm = dust_evidence
        value = r_norm if math.isfinite(r_norm) else r
        if math.isfinite(value):
            if value >= settings["dust_same"]:
                odds *= settings["lr_dust_same"]
                reasons.append(f"dust pattern matches (r={value:.2f})")
            elif value <= settings["dust_changed"]:
                odds *= settings["lr_dust_changed"]
                reasons.append(f"dust pattern differs (r={value:.2f})")
            else:
                reasons.append(f"dust pattern inconclusive (r={value:.2f})")
    if vignetting_rms is not None and math.isfinite(vignetting_rms):
        if vignetting_rms > 2.0 * settings["vignetting_floor"]:
            odds *= settings["lr_vignetting"]
            reasons.append(f"vignetting differs ({vignetting_rms:.4f})")
    probability = odds / (1.0 + odds)
    return FlatCandidate(flat_set.set_id, probability, _category(probability, settings),
                         "; ".join(reasons))


def _other_candidates(candidates: Sequence[FlatCandidate], by_id: Mapping[str, FlatSet],
                      limit: int = 3) -> str:
    """Report text for flat sets that were not used, with probability and reason."""
    if not candidates:
        return ""
    parts = [f"{by_id[c.set_id].describe()} p={c.probability:.2f} ({c.reason})"
             for c in candidates[:limit]]
    more = f"; {len(candidates) - limit} more" if len(candidates) > limit else ""
    return "; not used: " + "; ".join(parts) + more


def assign_flats(
    lights: Table,
    sessions: Sequence[Session],
    sets: Sequence[FlatSet],
    *,
    context: Table | None = None,
    bias_levels: Mapping[str, float] | None = None,
    settings: Mapping[str, float] | None = None,
    no_flat_policy: str = "best_available",
    use_images: bool = True,
) -> list[FlatAssignment]:
    """Flat choice for every (session, filter, binning) of ``lights``."""
    if no_flat_policy not in NO_FLAT_POLICIES:
        raise ValueError(f"no_flat_policy must be one of {NO_FLAT_POLICIES}")
    limits = dict(DEFAULT_FLAT_SETTINGS)
    limits.update(settings or {})
    bias_levels = dict(bias_levels or {})
    lookup = _session_lookup(sessions)
    images: dict[str, FlatImages] = {}

    def flat_images(fs: FlatSet) -> FlatImages:
        if fs.set_id not in images:
            images[fs.set_id] = load_flat_images(
                fs, bias_levels.get(fs.electronic_id, 0.0), int(limits["max_frames"])
            ) if use_images else FlatImages(None, None, None)
        return images[fs.set_id]

    consumers: dict[tuple[str, str, str], None] = {}
    session_binnings: dict[str, set[str]] = {}
    for row in lights:
        row = dict(zip(lights.colnames, row, strict=True))
        sid = str(row.get("session_id"))
        if sid not in lookup:
            continue
        consumers[(sid, filter_name(row), binning(row))] = None
        session_binnings.setdefault(sid, set()).add(binning(row))

    # Certain sets per session (any filter) are the dust references.
    certain_refs: dict[str, list[FlatSet]] = {}
    for sid, session in lookup.items():
        for fs in sets:
            if fs.camera == session.camera and fs.telescope == session.telescope and (
                session.start_jd <= fs.mid_jd <= session.end_jd
            ) and fs.binning in session_binnings.get(sid, set()):
                certain_refs.setdefault(sid, []).append(fs)

    assignments: list[FlatAssignment] = []
    for sid, filt, binn in consumers:
        session = lookup[sid]
        result = FlatAssignment(sid, filt, binn)
        candidates = [fs for fs in sets if fs.filter == filt and fs.binning == binn
                      and fs.camera == session.camera]
        scored: list[FlatCandidate] = []
        for fs in candidates:
            dust = None
            vignetting = None
            refs = [r for r in certain_refs.get(sid, []) if r.set_id != fs.set_id]
            outside = not (session.start_jd <= fs.mid_jd <= session.end_jd)
            if outside and refs and use_images:
                imgs = flat_images(fs)
                values = [dust_similarity(imgs, flat_images(ref)) for ref in refs]
                values = [v for v in values if math.isfinite(v[0])]
                if values:
                    dust = max(values, key=lambda v: v[1] if math.isfinite(v[1]) else v[0])
                same_filter = [ref for ref in refs if ref.filter == fs.filter]
                if same_filter and imgs.master is not None:
                    ref_master = flat_images(same_filter[0]).master
                    if ref_master is not None:
                        vignetting = vignetting_difference(imgs.master, ref_master)
            scored.append(score_flat_set(
                fs, session, sessions, session_binnings=session_binnings.get(sid, {binn}),
                context=context, dust_evidence=dust, vignetting_rms=vignetting, settings=limits,
            ))
        scored.sort(key=lambda c: -c.probability)
        result.candidates = scored
        usable = [c for c in scored if c.category in (CERTAIN, LIKELY)]
        by_id = {fs.set_id: fs for fs in candidates}
        if usable:
            best = usable[0]
            chosen = [best]
            if best.category != CERTAIN:
                # Flats before and after the session: combine both sides.
                side = by_id[best.set_id].mid_jd > session.end_jd
                other = [c for c in usable[1:]
                         if (by_id[c.set_id].mid_jd > session.end_jd) != side]
                if other:
                    chosen.append(other[0])
            result.flat_set_ids = [c.set_id for c in chosen]
            result.probability = min(c.probability for c in chosen)
            result.category = min((c.category for c in chosen),
                                  key=[CERTAIN, LIKELY, UNCERTAIN, REJECTED].index)
            result.note = "; ".join(f"{by_id[c.set_id].describe()}: {c.reason}" for c in chosen)
            if len(chosen) == 2:
                result.note = "flats before and after the session combined; " + result.note
        elif scored and scored[0].category == UNCERTAIN:
            best = scored[0]
            result.flat_set_ids = [best.set_id]
            result.probability = best.probability
            result.category = UNCERTAIN
            result.note = (f"{by_id[best.set_id].describe()}: {best.reason} (uncertain, check)"
                           + _other_candidates(scored[1:], by_id))
        elif no_flat_policy == "best_available" and scored and scored[0].probability > 0:
            best = scored[0]
            result.flat_set_ids = [best.set_id]
            result.probability = best.probability
            result.category = REJECTED
            result.note = (f"no reliable flat; best available {by_id[best.set_id].describe()} "
                           f"used anyway (p={best.probability:.2f}: {best.reason})"
                           + _other_candidates(scored[1:], by_id))
        else:
            result.note = (f"no applicable flat ({no_flat_policy})"
                           + _other_candidates(scored, by_id) if scored
                           else "no flat frames for this camera / binning / filter")
        assignments.append(result)
    return assignments


__all__ = [
    "CERTAIN",
    "DEFAULT_FLAT_SETTINGS",
    "FlatAssignment",
    "FlatCandidate",
    "FlatImages",
    "FlatSet",
    "LIKELY",
    "NO_FLAT_POLICIES",
    "REJECTED",
    "UNCERTAIN",
    "assign_flats",
    "correlation",
    "dust_map",
    "dust_similarity",
    "flat_sets",
    "load_flat_images",
    "score_flat_set",
    "vignetting_difference",
]
