"""Track / correlation quality-control statistics for one image series.

Everything here is plain numpy over the per-image photometry tables after
:func:`~ost_photometry.analyze.correlate.correlate_preserve_objects`, so it
can be tested without FITS files and plotted in a background process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from ..models import ImageSeries


def _float_col(table: Table, name: str) -> np.ndarray | None:
    if name not in table.colnames:
        return None
    col = table[name]
    return np.asarray(getattr(col, "value", col), dtype=float).ravel()


def _sky_deg(image, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = getattr(image, "wcs", None)
    nan = np.full(x.size, np.nan)
    if w is None or x.size == 0:
        return nan, nan
    try:
        sky = w.pixel_to_world(x, y)
        return np.asarray(sky.ra.deg, dtype=float), np.asarray(sky.dec.deg, dtype=float)
    except Exception:  # noqa: BLE001 - any WCS failure just disables sky stats
        return nan, nan


def collect_track_qc(
    image_series: ImageSeries,
    *,
    filter_: str,
    coordinate_frame: str,
    pixel_radius: float,
    separation_limit_arcsec: float,
    min_detection_fraction: float | None = None,
    ooi_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Per-frame and per-track statistics of a correlated series.

    Per frame: number of tracks matched, median pixel shift of the matched
    tracks relative to the reference frame (registration / drift check).

    Per track: number of detections, median pixel position, pixel and sky
    scatter about the track median (identity check — one star sits at one
    place), instrumental magnitude scatter after removing the per-frame
    median offset (photometric consistency independent of clouds).

    ``coordinate_frame`` (``"pixel"`` / ``"sky"``) records how the tracks were
    matched so the plot draws the matching radius in the right unit.
    """
    images = list(getattr(image_series, "image_list", []) or [])
    n_frames = len(images)
    ref = int(getattr(image_series, "reference_image_index", 0) or 0)
    if ref < 0 or ref >= n_frames:
        ref = 0
    ooi = sorted({int(i) for i in (ooi_ids or []) if i is not None})

    # --- gather per frame -------------------------------------------------
    frame_ids: list[np.ndarray] = []
    frame_x: list[np.ndarray] = []
    frame_y: list[np.ndarray] = []
    frame_mag: list[np.ndarray] = []
    frame_ra: list[np.ndarray] = []
    frame_dec: list[np.ndarray] = []
    frame_jd = np.full(n_frames, np.nan)
    for i, image in enumerate(images):
        phot = getattr(image, "photometry", None)
        jd = getattr(image, "jd", None)
        if jd is not None:
            frame_jd[i] = float(jd)
        if phot is None or len(phot) == 0 or "id" not in phot.colnames:
            empty = np.array([], dtype=float)
            frame_ids.append(np.array([], dtype=np.int64))
            frame_x.append(empty)
            frame_y.append(empty)
            frame_mag.append(empty)
            frame_ra.append(empty)
            frame_dec.append(empty)
            continue
        ids = np.asarray(phot["id"], dtype=np.int64)
        x = _float_col(phot, "x_fit")
        y = _float_col(phot, "y_fit")
        mag = _float_col(phot, "mags_fit")
        if mag is None:
            flux = _float_col(phot, "flux_fit")
            mag = (
                -2.5 * np.log10(np.where(flux > 0, flux, np.nan))
                if flux is not None
                else np.full(ids.size, np.nan)
            )
        ra, dec = _sky_deg(image, x, y)
        keep = ids >= 0
        frame_ids.append(ids[keep])
        frame_x.append(x[keep])
        frame_y.append(y[keep])
        frame_mag.append(mag[keep])
        frame_ra.append(ra[keep])
        frame_dec.append(dec[keep])

    all_ids = (
        np.unique(np.concatenate([f for f in frame_ids if f.size]))
        if any(f.size for f in frame_ids)
        else np.array([], dtype=np.int64)
    )
    n_tracks = int(all_ids.size)
    id_to_col = {int(sid): k for k, sid in enumerate(all_ids)}

    # --- (n_frames, n_tracks) matrices -------------------------------------
    shape = (n_frames, n_tracks)
    X = np.full(shape, np.nan)
    Y = np.full(shape, np.nan)
    M = np.full(shape, np.nan)
    RA = np.full(shape, np.nan)
    DEC = np.full(shape, np.nan)
    for i in range(n_frames):
        if frame_ids[i].size == 0:
            continue
        cols = np.fromiter((id_to_col[int(s)] for s in frame_ids[i]), dtype=int)
        X[i, cols] = frame_x[i]
        Y[i, cols] = frame_y[i]
        M[i, cols] = frame_mag[i]
        RA[i, cols] = frame_ra[i]
        DEC[i, cols] = frame_dec[i]

    detected = np.isfinite(X)
    track_n_det = detected.sum(axis=0)
    frame_n_matched = detected.sum(axis=1)

    # --- per frame shift vs reference --------------------------------------
    frame_dx = np.full(n_frames, np.nan)
    frame_dy = np.full(n_frames, np.nan)
    if n_tracks:
        both = detected & detected[ref][None, :]
        for i in range(n_frames):
            m = both[i]
            if np.count_nonzero(m) >= 3:
                frame_dx[i] = float(np.median(X[i, m] - X[ref, m]))
                frame_dy[i] = float(np.median(Y[i, m] - Y[ref, m]))

    # --- per track scatter ---------------------------------------------------
    with np.errstate(all="ignore"):
        x_med = np.nanmedian(X, axis=0) if n_tracks else np.array([])
        y_med = np.nanmedian(Y, axis=0) if n_tracks else np.array([])
        off_px = np.hypot(X - x_med[None, :], Y - y_med[None, :]) if n_tracks else X
        track_rms_px = np.sqrt(np.nanmean(off_px**2, axis=0)) if n_tracks else np.array([])
        track_max_px = np.nanmax(off_px, axis=0) if n_tracks else np.array([])

        has_sky = n_tracks > 0 and np.any(np.isfinite(RA))
        if has_sky:
            dec_med = np.nanmedian(DEC, axis=0)
            ra_med = np.nanmedian(RA, axis=0)
            dra = (RA - ra_med[None, :]) * np.cos(np.deg2rad(dec_med))[None, :] * 3600.0
            ddec = (DEC - dec_med[None, :]) * 3600.0
            off_sky = np.hypot(dra, ddec)
            track_rms_sky = np.sqrt(np.nanmean(off_sky**2, axis=0))
            track_max_sky = np.nanmax(off_sky, axis=0)
        else:
            track_rms_sky = np.full(n_tracks, np.nan)
            track_max_sky = np.full(n_tracks, np.nan)

        track_mag_med = np.nanmedian(M, axis=0) if n_tracks else np.array([])
        # Remove the per-frame median offset (clouds, airmass) before the RMS so
        # a high value points at the track itself (identity error / variable).
        if n_tracks:
            frame_zp = np.nanmedian(M - track_mag_med[None, :], axis=1)
            frame_zp = np.where(np.isfinite(frame_zp), frame_zp, 0.0)
            track_mag_rms = np.nanstd(M - frame_zp[:, None], axis=0)
        else:
            track_mag_rms = np.array([])

    # --- worst frame (largest median shift) for the residual map -----------
    shift = np.hypot(frame_dx, frame_dy)
    if np.any(np.isfinite(shift)):
        worst = int(np.nanargmax(shift))
    else:
        worst = ref
    worst_dx = X[worst] - X[ref] if n_tracks else np.array([])
    worst_dy = Y[worst] - Y[ref] if n_tracks else np.array([])

    frame_limit = float(pixel_radius)
    if coordinate_frame == "pixel":
        suspect = np.isfinite(track_max_px) & (track_max_px > frame_limit)
    else:
        lim = float(separation_limit_arcsec)
        suspect = np.isfinite(track_max_sky) & (track_max_sky > lim)

    return {
        "filter": str(filter_),
        "coordinate_frame": str(coordinate_frame),
        "reference_index": ref,
        "n_frames": n_frames,
        "n_tracks": n_tracks,
        "pixel_radius": float(pixel_radius),
        "separation_limit_arcsec": float(separation_limit_arcsec),
        "min_detection_fraction": (
            None if min_detection_fraction is None else float(min_detection_fraction)
        ),
        "ooi_ids": ooi,
        "frame_index": np.arange(n_frames),
        "frame_jd": frame_jd,
        "frame_n_matched": frame_n_matched.astype(int),
        "frame_dx": frame_dx,
        "frame_dy": frame_dy,
        "worst_frame_index": worst,
        "track_id": all_ids.astype(np.int64),
        "track_n_det": track_n_det.astype(int),
        "track_x_med": x_med,
        "track_y_med": y_med,
        "track_rms_px": track_rms_px,
        "track_max_px": track_max_px,
        "track_rms_arcsec": track_rms_sky,
        "track_max_arcsec": track_max_sky,
        "track_mag_med": track_mag_med,
        "track_mag_rms": track_mag_rms,
        "track_suspect": suspect,
        "worst_dx": worst_dx,
        "worst_dy": worst_dy,
    }


def track_qc_table(qc: dict[str, Any]) -> Table:
    """Per-track QC rows (``track_qc_<filter>.ecsv``)."""
    n = int(qc["n_tracks"])
    if n == 0:
        return Table()
    return Table(
        {
            "id": qc["track_id"],
            "n_detections": qc["track_n_det"],
            "detection_fraction": qc["track_n_det"] / max(int(qc["n_frames"]), 1),
            "x_median": qc["track_x_med"],
            "y_median": qc["track_y_med"],
            "pos_rms_px": qc["track_rms_px"],
            "pos_max_px": qc["track_max_px"],
            "pos_rms_arcsec": qc["track_rms_arcsec"],
            "pos_max_arcsec": qc["track_max_arcsec"],
            "mag_inst_median": qc["track_mag_med"],
            "mag_inst_rms": qc["track_mag_rms"],
            "suspect": qc["track_suspect"],
            "is_ooi": np.isin(qc["track_id"], np.asarray(qc["ooi_ids"], dtype=np.int64)),
        }
    )


def track_qc_summary(qc: dict[str, Any]) -> str:
    """One-line terminal summary."""
    n_tracks = int(qc["n_tracks"])
    if n_tracks == 0:
        return f"Track QC ({qc['filter']}): no tracks"
    frac = qc["track_n_det"] / max(int(qc["n_frames"]), 1)
    n_bad = int(np.count_nonzero(qc["track_suspect"]))
    shift = np.hypot(qc["frame_dx"], qc["frame_dy"])
    n_shift = int(np.count_nonzero(np.isfinite(shift) & (shift > qc["pixel_radius"])))
    unit = "px" if qc["coordinate_frame"] == "pixel" else '"'
    scatter = (
        qc["track_rms_px"] if qc["coordinate_frame"] == "pixel" else qc["track_rms_arcsec"]
    )
    med_scatter = float(np.nanmedian(scatter)) if np.any(np.isfinite(scatter)) else np.nan
    return (
        f"Track QC ({qc['filter']}): {n_tracks} tracks, median completeness "
        f"{np.median(frac):.0%}, median position scatter {med_scatter:.2f} {unit}, "
        f"{n_bad} suspect track(s); {n_shift} of {int(qc['n_frames'])} frames shifted "
        f"> {qc['pixel_radius']:.0f} px vs reference"
    )


__all__ = ["collect_track_qc", "track_qc_summary", "track_qc_table"]
