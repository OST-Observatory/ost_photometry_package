"""Timeline plots of the calibration grouping (one PDF per telescope and night)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...output_layout import diagnostics_dir
from .classify import BIAS, DARK, FLAT, LIGHT, SPECTROSCOPY

#: Categorical slots in fixed order (identity follows the entity, never rank).
_CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#4a3aa7", "#e34948")
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e1"
_MUTED = "#a3a29d"
_ROWS = (LIGHT, FLAT, DARK, BIAS, SPECTROSCOPY, "other camera")


def _hours(jd: np.ndarray, night_start: float) -> np.ndarray:
    return (jd - night_start) * 24.0


def plot_night_timelines(plan, output_dir: str | Path) -> list[Path]:
    """One timeline per telescope and night with lights or flats.

    Top: camera orientation (mod 180) of solved frames, sessions shaded.
    Bottom: frame types over time; lights coloured by target, flats
    labelled with filter and the probability of their best session.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frames = plan.frames
    if len(frames) == 0:
        return []
    out_dir = diagnostics_dir(output_dir, "calibration_groups")
    jd = np.asarray(frames["jd"], dtype=float)
    nights = np.asarray(frames["night"]).astype(str)
    kinds = np.asarray(frames["frame_type"]).astype(str)
    telescopes = np.array([str(c) for c in frames["telescop"]]) if "telescop" in frames.colnames \
        else np.array([""] * len(frames))
    cameras = np.asarray(frames["camera"]).astype(str)
    targets = np.asarray(frames["target_name"]).astype(str)
    session_ids = np.asarray(frames["session_id"]).astype(str)
    flat_sets = np.asarray(frames["flat_set_id"]).astype(str)
    pa = np.asarray(frames["pa_mod180"], dtype=float)

    target_order = list(dict.fromkeys(t for t in targets if t and t != "unknown"))
    target_color = {t: _CATEGORICAL[i % len(_CATEGORICAL)] for i, t in enumerate(target_order)}
    session_order = [s.session_id for s in plan.sessions]
    session_color = {s: _CATEGORICAL[i % len(_CATEGORICAL)] for i, s in enumerate(session_order)}
    best_p: dict[str, float] = {}
    for assignment in plan.flat_assignments:
        for candidate in assignment.candidates:
            best_p[candidate.set_id] = max(best_p.get(candidate.set_id, 0.0),
                                           candidate.probability)

    written: list[Path] = []
    keys = sorted({(n, t) for n, t, k in zip(nights, telescopes, kinds, strict=True)
                   if k in (LIGHT, FLAT) and n != "nodate"})
    for night, telescope in keys:
        sel = (nights == night) & ((telescopes == telescope) | np.isin(kinds, [BIAS, DARK]))
        if not np.any(sel & np.isin(kinds, [LIGHT, FLAT])):
            continue
        start = math.floor(np.nanmin(jd[sel]) - 0.5) + 0.5  # noon UT before the night
        fig, (ax_pa, ax_t) = plt.subplots(
            2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]},
            constrained_layout=True,
        )
        for session in plan.sessions:
            if night not in session.session_id:
                continue
            x0, x1 = _hours(np.array([session.start_jd, session.end_jd]), start)
            color = session_color.get(session.session_id, _MUTED)
            ax_pa.axvspan(x0, max(x1, x0 + 0.05), color=color, alpha=0.12, lw=0)
            label = session.session_id.split("_", 2)[-1]
            ax_pa.text(x0, 1.02, label, transform=ax_pa.get_xaxis_transform(), fontsize=8,
                       color=_INK_SECONDARY, va="bottom")
        solved = sel & np.isfinite(pa)
        for sid in dict.fromkeys(session_ids[solved]):
            m = solved & (session_ids == sid)
            ax_pa.plot(_hours(jd[m], start), pa[m], "o", ms=6,
                       color=session_color.get(sid, _MUTED), label=sid)
        ax_pa.set_ylabel("orientation mod 180 [deg]", color=_INK)
        if np.any(solved):
            lo, hi = np.nanmin(pa[solved]), np.nanmax(pa[solved])
            pad = max(1.0, 0.1 * (hi - lo))
            ax_pa.set_ylim(lo - pad, hi + pad)
        else:
            ax_pa.text(0.5, 0.5, "no plate solution this night", transform=ax_pa.transAxes,
                       ha="center", color=_INK_SECONDARY)

        rows = list(_ROWS)
        for k, row_name in enumerate(rows):
            if row_name == "other camera":
                main = cameras[sel & (kinds == LIGHT)]
                main_cam = main[0] if main.size else ""
                m = sel & (cameras != main_cam) & np.isin(kinds, [LIGHT, FLAT])
            else:
                m = sel & (kinds == row_name)
            if not np.any(m):
                continue
            x = _hours(jd[m], start)
            if row_name == LIGHT:
                colors = [target_color.get(t, _MUTED) for t in targets[m]]
                ax_t.scatter(x, np.full(x.size, k), c=colors, s=24, marker="|", linewidths=2)
            else:
                ax_t.scatter(x, np.full(x.size, k), color=_INK_SECONDARY, s=24, marker="|",
                             linewidths=2)
            if row_name == FLAT:
                for j, fs in enumerate(f for f in dict.fromkeys(flat_sets[m]) if f):
                    mm = m & (flat_sets == fs)
                    label = fs.split("_")[-1]
                    p = best_p.get(fs)
                    text = f"{label} p={p:.2f}" if p is not None else label
                    # Stagger labels: flat sets of several filters are taken
                    # back to back and would overlap.
                    ax_t.text(float(np.nanmean(_hours(jd[mm], start))), k - 0.3 - 0.22 * (j % 3),
                              text, fontsize=6.5, ha="center", color=_INK)
        ax_t.set_yticks(range(len(rows)))
        ax_t.set_yticklabels(rows)
        ax_t.set_ylim(-0.6, len(rows) - 0.4)
        ax_t.invert_yaxis()
        ax_t.set_xlabel("hours after noon UT", color=_INK)
        handles = [plt.Line2D([], [], marker="|", ls="", ms=10, mew=2, color=target_color[t],
                              label=t) for t in target_order
                   if np.any(sel & (targets == t))]
        if handles:
            ax_t.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                        frameon=False, fontsize=8, title="targets")
        for ax in (ax_pa, ax_t):
            ax.grid(True, color=_GRID, lw=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(colors=_INK_SECONDARY)
        fig.suptitle(f"Calibration grouping: night {night}, telescope {telescope or '?'}",
                     color=_INK)
        safe_tel = "".join(c if c.isalnum() else "_" for c in (telescope or "unknown"))
        path = out_dir / f"timeline_{night}_{safe_tel}.pdf"
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


__all__ = ["plot_night_timelines"]
