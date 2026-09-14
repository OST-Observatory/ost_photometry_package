"""Track / correlation QC figures (``diagnostics/correlation/track_qc_<filter>``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ...output_layout import diagnostics_dir


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def plot_track_qc(
    qc: dict[str, Any],
    output_dir: str | Path,
    file_type: str = "pdf",
) -> Path | None:
    """Six-panel overview of one filter's tracks after intra-filter correlation.

    Panels: (a) tracks matched per frame, (b) median pixel shift per frame vs
    the reference (registration / drift), (c) track completeness histogram
    with the ``min_detection_fraction`` cut, (d) per-track position scatter vs
    instrumental magnitude with the matching radius, (e) instrumental
    magnitude RMS vs magnitude, (f) field map coloured by position scatter
    with the residual vectors of the most-shifted frame.
    """
    n_tracks = int(qc.get("n_tracks", 0))
    n_frames = int(qc.get("n_frames", 0))
    if n_tracks == 0 or n_frames == 0:
        return None

    filt = str(qc["filter"])
    pixel_mode = str(qc["coordinate_frame"]) == "pixel"
    radius_px = float(qc["pixel_radius"])
    sep_arcsec = float(qc["separation_limit_arcsec"])
    ref = int(qc["reference_index"])
    frame_index = np.asarray(qc["frame_index"])
    n_matched = np.asarray(qc["frame_n_matched"], dtype=float)
    dx = np.asarray(qc["frame_dx"], dtype=float)
    dy = np.asarray(qc["frame_dy"], dtype=float)
    ids = np.asarray(qc["track_id"])
    n_det = np.asarray(qc["track_n_det"], dtype=float)
    mag = np.asarray(qc["track_mag_med"], dtype=float)
    mag_rms = np.asarray(qc["track_mag_rms"], dtype=float)
    suspect = np.asarray(qc["track_suspect"], dtype=bool)
    ooi_mask = np.isin(ids, np.asarray(qc.get("ooi_ids", []), dtype=np.int64))
    if pixel_mode:
        scatter_max = np.asarray(qc["track_max_px"], dtype=float)
        scatter_rms = np.asarray(qc["track_rms_px"], dtype=float)
        unit, limit = "px", radius_px
    else:
        scatter_max = np.asarray(qc["track_max_arcsec"], dtype=float)
        scatter_rms = np.asarray(qc["track_rms_arcsec"], dtype=float)
        unit, limit = '"', sep_arcsec
        if not np.any(np.isfinite(scatter_max)):
            scatter_max = np.asarray(qc["track_max_px"], dtype=float)
            scatter_rms = np.asarray(qc["track_rms_px"], dtype=float)
            unit, limit = "px", radius_px

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.5))
    (ax_n, ax_shift), (ax_hist, ax_pos), (ax_mag, ax_map) = axes

    # (a) tracks matched per frame
    ax_n.bar(frame_index, n_matched, color="0.6", width=0.9)
    ax_n.axvline(ref, color="C3", ls="--", lw=1.0, label=f"reference frame {ref}")
    ax_n.set_xlabel("frame index")
    ax_n.set_ylabel("tracks matched")
    ax_n.set_title(
        f"(a) matched tracks per frame  ({n_tracks} tracks, {n_frames} frames)",
        loc="left",
        fontsize=10,
    )
    ax_n.legend(fontsize=8, loc="lower left")
    ax_n.grid(True, color="lightgray", ls="--")

    # (b) per-frame median shift vs reference
    ax_shift.plot(frame_index, dx, ".-", ms=4, lw=0.8, label="median Δx")
    ax_shift.plot(frame_index, dy, ".-", ms=4, lw=0.8, label="median Δy")
    for s in (-radius_px, radius_px):
        ax_shift.axhline(s, color="C3", ls=":", lw=0.9)
    ax_shift.axhline(0.0, color="0.3", lw=0.6)
    ax_shift.set_xlabel("frame index")
    ax_shift.set_ylabel("pixel shift vs reference [px]")
    mode_note = (
        "matched in pixels" if pixel_mode else "matched on the sky (per-frame WCS)"
    )
    ax_shift.set_title(
        f"(b) frame registration — {mode_note}; dotted: ±{radius_px:.0f} px match radius",
        loc="left",
        fontsize=10,
    )
    ax_shift.legend(fontsize=8)
    ax_shift.grid(True, color="lightgray", ls="--")

    # (c) completeness histogram
    frac = n_det / max(n_frames, 1)
    ax_hist.hist(frac, bins=np.linspace(0.0, 1.0, 26), color="0.6")
    mdf = qc.get("min_detection_fraction")
    if mdf is not None:
        ax_hist.axvline(float(mdf), color="C3", ls="--", lw=1.0, label=f"min_detection_fraction={float(mdf):.2f}")
        ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("fraction of frames on which the track is detected")
    ax_hist.set_ylabel("tracks")
    ax_hist.set_title("(c) track completeness", loc="left", fontsize=10)
    ax_hist.grid(True, color="lightgray", ls="--")

    # (d) position scatter vs magnitude
    good = ~suspect
    ax_pos.scatter(mag[good], scatter_max[good], s=10, c="0.5", label="tracks")
    if np.any(suspect):
        ax_pos.scatter(
            mag[suspect], scatter_max[suspect], s=16, c="C3",
            label=f"{int(np.count_nonzero(suspect))} suspect (> match radius)",
        )
    if np.any(ooi_mask):
        ax_pos.scatter(
            mag[ooi_mask], scatter_max[ooi_mask], s=90, marker="*", c="C1",
            edgecolor="k", zorder=5, label="object of interest",
        )
    ax_pos.axhline(limit, color="C3", ls="--", lw=1.0)
    ax_pos.set_yscale("log")
    ax_pos.set_xlabel("median instrumental magnitude")
    ax_pos.set_ylabel(f"max position offset from track median [{unit}]")
    ax_pos.set_title(
        f"(d) identity check — one star = one place (dashed: {limit:g} {unit} match radius)",
        loc="left",
        fontsize=10,
    )
    ax_pos.legend(fontsize=8, loc="upper left")
    ax_pos.grid(True, color="lightgray", ls="--", which="both")

    # (e) instrumental magnitude RMS vs magnitude
    ax_mag.scatter(mag[good], mag_rms[good], s=10, c="0.5")
    if np.any(suspect):
        ax_mag.scatter(mag[suspect], mag_rms[suspect], s=16, c="C3")
    if np.any(ooi_mask):
        ax_mag.scatter(
            mag[ooi_mask], mag_rms[ooi_mask], s=90, marker="*", c="C1",
            edgecolor="k", zorder=5,
        )
    ax_mag.set_yscale("log")
    ax_mag.set_xlabel("median instrumental magnitude")
    ax_mag.set_ylabel("RMS of instrumental magnitude [mag]\n(per-frame offset removed)")
    ax_mag.set_title(
        "(e) photometric consistency per track (identity errors / variables = high RMS)",
        loc="left",
        fontsize=10,
    )
    ax_mag.grid(True, color="lightgray", ls="--", which="both")

    # (f) field map + residual vectors of the most shifted frame
    xm = np.asarray(qc["track_x_med"], dtype=float)
    ym = np.asarray(qc["track_y_med"], dtype=float)
    col = np.where(np.isfinite(scatter_rms), scatter_rms, np.nan)
    finite = np.isfinite(col)
    if np.any(finite):
        # full range so a single mixed track stands out, but do not let one
        # extreme outlier flatten the colour scale for everything else
        vmax = max(min(float(np.nanmax(col)), 5.0 * float(np.nanmedian(col))), 1e-3)
        sc = ax_map.scatter(
            xm[finite], ym[finite], c=col[finite], s=14, vmin=0.0, vmax=vmax,
            cmap="viridis",
        )
        cb = fig.colorbar(sc, ax=ax_map, pad=0.01)
        cb.set_label(f"position RMS [{unit}]", fontsize=8)
    worst = int(qc.get("worst_frame_index", ref))
    wdx = np.asarray(qc["worst_dx"], dtype=float)
    wdy = np.asarray(qc["worst_dy"], dtype=float)
    ok = np.isfinite(wdx) & np.isfinite(wdy)
    scale_note = ""
    if worst != ref and np.count_nonzero(ok) >= 3:
        med_shift = float(np.nanmedian(np.hypot(wdx[ok], wdy[ok])))
        if med_shift > 1.0:
            # arrows show the residual pattern (pure shift vs rotation/scale)
            ax_map.quiver(
                xm[ok], ym[ok], wdx[ok], wdy[ok], color="C3", alpha=0.6,
                angles="xy", scale_units="xy", scale=None, width=0.003,
            )
            scale_note = (
                f"; arrows: frame {worst} − reference "
                f"(median {med_shift:.1f} px, auto-scaled)"
            )
        else:
            scale_note = "; all frames within 1 px of the reference"
    if np.any(ooi_mask):
        ax_map.scatter(
            xm[ooi_mask], ym[ooi_mask], s=120, marker="*", facecolor="none",
            edgecolor="C1", linewidth=1.5, zorder=6,
        )
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("x [px]")
    ax_map.set_ylabel("y [px]")
    ax_map.set_title(f"(f) field map{scale_note}", loc="left", fontsize=10)
    ax_map.grid(True, color="lightgray", ls="--")

    fig.suptitle(
        f"Track QC — filter {filt}: {n_tracks} tracks over {n_frames} frames, "
        f"{int(np.count_nonzero(suspect))} suspect",
        fontsize=12,
    )
    fig.tight_layout()
    out = diagnostics_dir(output_dir, "correlation")
    path = out / f"track_qc_{_sanitize(filt)}.{file_type}"
    fig.savefig(path, bbox_inches="tight", format=file_type)
    plt.close(fig)
    return path


__all__ = ["plot_track_qc"]
