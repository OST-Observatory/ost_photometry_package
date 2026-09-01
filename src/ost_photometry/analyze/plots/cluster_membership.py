"""QC figures for Gaia (μ, π) cluster membership."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ...output_layout import diagnostics_dir

plt.switch_backend("Agg")

_FIELD_COLOR = "0.65"
_MEMBER_CMAP = "cividis"
_N_PLOT_MAX = 12_000
# Half-widths so a tight clump is still readable, without stretching to Gaia tails.
_PM_MIN_HALF = 2.5  # mas/yr
_PLX_MIN_HALF = 0.35  # mas
_FOCUS_N_SIGMA = 5.5


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    keep = np.ones(arrays[0].size, dtype=bool)
    for arr in arrays:
        keep &= np.isfinite(np.asarray(arr, dtype=float))
    return keep


def _focus_limits(
    values: np.ndarray,
    member: np.ndarray | None = None,
    *,
    extras: tuple[float | None, ...] = (),
    min_half: float,
    n_sigma: float = _FOCUS_N_SIGMA,
    pad: float = 0.2,
) -> tuple[float, float]:
    """Axis range around the member clump (median ± N σ_MAD), not min/max.

    High-PM stars or nearby dwarfs (ϖ ≳ 10 mas) stay outside the view.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    core = values[finite]
    if member is not None:
        mem = finite & np.asarray(member, dtype=bool)
        if int(np.count_nonzero(mem)) >= 8:
            core = values[mem]
    if core.size == 0:
        return -float(min_half), float(min_half)
    center = float(np.median(core))
    mad = float(np.median(np.abs(core - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        lo_c, hi_c = np.percentile(core, [10.0, 90.0])
        scale = max(0.5 * (float(hi_c) - float(lo_c)), 0.0)
    half = max(float(min_half), float(n_sigma) * scale)
    for extra in extras:
        if extra is None:
            continue
        try:
            point = float(extra)
        except (TypeError, ValueError):
            continue
        if np.isfinite(point):
            half = max(half, abs(point - center) + 0.25 * float(min_half))
    half *= 1.0 + float(pad)
    lo, hi = center - half, center + half
    return _shrink_empty_edges(
        lo,
        hi,
        values[finite],
        extras=extras,
        pad_abs=0.12 * float(min_half),
    )


def _shrink_empty_edges(
    lo: float,
    hi: float,
    values: np.ndarray,
    *,
    extras: tuple[float | None, ...] = (),
    pad_frac: float = 0.04,
    pad_abs: float = 0.04,
) -> tuple[float, float]:
    """Pull limits in where the window is empty (e.g. ϖ cut at ~0.25 mas)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return lo, hi
    span = hi - lo
    pad = max(pad_frac * span, float(pad_abs))
    inside = finite[(finite >= lo) & (finite <= hi)]
    if inside.size < 8:
        return lo, hi
    vlo = float(np.min(inside))
    vhi = float(np.max(inside))
    for extra in extras:
        if extra is None:
            continue
        try:
            point = float(extra)
        except (TypeError, ValueError):
            continue
        if np.isfinite(point) and lo <= point <= hi:
            vlo = min(vlo, point)
            vhi = max(vhi, point)
    if vlo - lo > pad:
        lo = vlo - pad
    if hi - vhi > pad:
        hi = vhi + pad
    if hi <= lo:
        mid = 0.5 * (vlo + vhi)
        return mid - pad, mid + pad
    return lo, hi


def _square_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    return (cx - half, cx + half), (cy - half, cy + half)


def _subsample_indices(n: int, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if n <= n_max:
        return np.arange(n)
    return np.sort(rng.choice(n, size=n_max, replace=False))


def _subtitle(
    *,
    method: str,
    cluster_component: int,
    reason: str,
    pmem_min: float,
    n_member: int,
    n_total: int,
) -> str:
    why = f" — {reason}" if reason else ""
    return (
        f"{method.upper()} in (μ_α*, μ_δ, ϖ): cluster = component "
        f"{cluster_component}{why}. "
        f"Members: P_mem ≥ {pmem_min:.2f} ({n_member}/{n_total})."
    )


def _save(fig: plt.Figure, out, stem: str, file_type: str) -> None:
    fig.savefig(
        out / f"{stem}.{file_type}",
        bbox_inches="tight",
        format=file_type,
    )
    plt.close(fig)


def plot_cluster_membership_diagnostics(
    *,
    output_dir: str,
    file_type: str = "pdf",
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    p_mem: np.ndarray,
    gmag: np.ndarray | None = None,
    pmem_min: float = 0.5,
    method: str = "gmm",
    cluster_component: int = 0,
    reason: str = "",
    simbad_pm_ra: float | None = None,
    simbad_pm_de: float | None = None,
    simbad_plx: float | None = None,
) -> None:
    """Write μ–μ, parallax, P_mem, and 3-D QC under ``diagnostics/cluster/``."""
    pm_ra = np.asarray(pm_ra, dtype=float)
    pm_de = np.asarray(pm_de, dtype=float)
    plx = np.asarray(plx, dtype=float)
    p_mem = np.asarray(p_mem, dtype=float)
    keep = _finite_mask(pm_ra, pm_de, plx, p_mem)
    pm_ra, pm_de, plx, p_mem = pm_ra[keep], pm_de[keep], plx[keep], p_mem[keep]
    if gmag is not None:
        gmag = np.asarray(gmag, dtype=float)[keep]
    n = p_mem.size
    if n == 0:
        return

    member = p_mem >= float(pmem_min)
    n_mem = int(np.count_nonzero(member))
    note = _subtitle(
        method=method,
        cluster_component=cluster_component,
        reason=reason,
        pmem_min=pmem_min,
        n_member=n_mem,
        n_total=n,
    )
    rng = np.random.default_rng(0)
    show = _subsample_indices(n, _N_PLOT_MAX, rng)
    out = diagnostics_dir(output_dir, "cluster")
    pm_xlim, pm_ylim = _square_limits(
        _focus_limits(
            pm_ra,
            member,
            extras=(simbad_pm_ra,),
            min_half=_PM_MIN_HALF,
        ),
        _focus_limits(
            pm_de,
            member,
            extras=(simbad_pm_de,),
            min_half=_PM_MIN_HALF,
        ),
    )
    plx_lim = _focus_limits(
        plx,
        member,
        extras=(simbad_plx,),
        min_half=_PLX_MIN_HALF,
    )

    _plot_proper_motion(
        out,
        file_type,
        pm_ra[show],
        pm_de[show],
        p_mem[show],
        member[show],
        pmem_min=pmem_min,
        note=note,
        n_shown=show.size,
        n_total=n,
        simbad_pm_ra=simbad_pm_ra,
        simbad_pm_de=simbad_pm_de,
        xlim=pm_xlim,
        ylim=pm_ylim,
    )
    _plot_parallax(
        out,
        file_type,
        plx,
        p_mem,
        member,
        gmag,
        pmem_min=pmem_min,
        note=note,
        simbad_plx=simbad_plx,
        show=show,
        plx_lim=plx_lim,
    )
    _plot_pmem_histogram(
        out,
        file_type,
        p_mem,
        pmem_min=pmem_min,
        note=note,
    )
    _plot_mu_plx_3d(
        out,
        file_type,
        pm_ra[show],
        pm_de[show],
        plx[show],
        member[show],
        note=note,
        simbad_pm_ra=simbad_pm_ra,
        simbad_pm_de=simbad_pm_de,
        simbad_plx=simbad_plx,
        xlim=pm_xlim,
        ylim=pm_ylim,
        zlim=plx_lim,
    )


def _plot_proper_motion(
    out,
    file_type: str,
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    p_mem: np.ndarray,
    member: np.ndarray,
    *,
    pmem_min: float,
    note: str,
    n_shown: int,
    n_total: int,
    simbad_pm_ra: float | None,
    simbad_pm_de: float | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    field = ~member
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    if np.any(field):
        ax.scatter(
            pm_ra[field],
            pm_de[field],
            s=6,
            c=_FIELD_COLOR,
            alpha=0.35,
            linewidths=0,
            label=f"Field (P_mem < {pmem_min:.2f})",
            zorder=1,
        )
    points = None
    if np.any(member):
        points = ax.scatter(
            pm_ra[member],
            pm_de[member],
            s=10,
            c=p_mem[member],
            cmap=_MEMBER_CMAP,
            vmin=float(pmem_min),
            vmax=1.0,
            alpha=0.85,
            linewidths=0,
            label=f"Members (P_mem ≥ {pmem_min:.2f})",
            zorder=2,
        )
    if (
        simbad_pm_ra is not None
        and simbad_pm_de is not None
        and np.isfinite(simbad_pm_ra)
        and np.isfinite(simbad_pm_de)
    ):
        ax.scatter(
            [float(simbad_pm_ra)],
            [float(simbad_pm_de)],
            marker="*",
            s=180,
            c="crimson",
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
            label="Simbad (μ)",
        )
        ax.axvline(float(simbad_pm_ra), color="crimson", ls=":", lw=0.8, alpha=0.6)
        ax.axhline(float(simbad_pm_de), color="crimson", ls=":", lw=0.8, alpha=0.6)
    if points is not None:
        cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$P_{\mathrm{mem}}$")
    ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
    ax.set_ylabel(r"$\mu_{\delta}$ [mas/yr]")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cluster selection in proper motion")
    if n_shown < n_total:
        sample = (
            f"{n_shown} of {n_total} quality Gaia stars shown; "
            "view clipped to the member clump."
        )
    else:
        sample = f"{n_total} quality Gaia stars; view clipped to the member clump."
    ax.text(
        0.02,
        0.98,
        sample,
        transform=ax.transAxes,
        fontsize=8,
        color="0.25",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.8,
            "lw": 0.3,
        },
        zorder=4,
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, wrap=True)
    ax.legend(loc="best", fontsize=8, markerscale=1.4)
    ax.grid(True, color="lightgray", linestyle="--", alpha=0.35)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "cluster_pm_members", file_type)


def _plot_parallax(
    out,
    file_type: str,
    plx: np.ndarray,
    p_mem: np.ndarray,
    member: np.ndarray,
    gmag: np.ndarray | None,
    *,
    pmem_min: float,
    note: str,
    simbad_plx: float | None,
    show: np.ndarray,
    plx_lim: tuple[float, float],
) -> None:
    field = ~member
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    plo, phi = plx_lim
    if not np.isfinite(plo) or not np.isfinite(phi) or phi <= plo:
        plo, phi = -0.5, 2.0
    bins = np.linspace(plo, phi, 50)

    ax_h = axes[0]
    if np.any(field):
        ax_h.hist(
            plx[field],
            bins=bins,
            color=_FIELD_COLOR,
            alpha=0.7,
            label=f"Field (P_mem < {pmem_min:.2f})",
        )
    if np.any(member):
        ax_h.hist(
            plx[member],
            bins=bins,
            color="C0",
            alpha=0.75,
            label=f"Members (P_mem ≥ {pmem_min:.2f})",
        )
    if simbad_plx is not None and np.isfinite(simbad_plx):
        ax_h.axvline(
            float(simbad_plx),
            color="crimson",
            ls="--",
            lw=1.2,
            label=f"Simbad ϖ = {float(simbad_plx):.2f} mas",
        )
    ax_h.set_xlabel(r"$\varpi$ [mas]")
    ax_h.set_ylabel("Number of Gaia stars")
    ax_h.set_xlim(plo, phi)
    ax_h.set_title("Parallax: members pile at one ϖ")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, color="lightgray", linestyle="--", alpha=0.35)

    ax_s = axes[1]
    if gmag is not None and np.any(np.isfinite(gmag)):
        g_show = gmag[show]
        plx_show = plx[show]
        p_show = p_mem[show]
        mem_show = member[show]
        field_show = ~mem_show
        if np.any(field_show):
            ax_s.scatter(
                plx_show[field_show],
                g_show[field_show],
                s=6,
                c=_FIELD_COLOR,
                alpha=0.3,
                linewidths=0,
                label=f"Field (P_mem < {pmem_min:.2f})",
            )
        if np.any(mem_show):
            points = ax_s.scatter(
                plx_show[mem_show],
                g_show[mem_show],
                s=10,
                c=p_show[mem_show],
                cmap=_MEMBER_CMAP,
                vmin=float(pmem_min),
                vmax=1.0,
                alpha=0.85,
                linewidths=0,
                label=f"Members (P_mem ≥ {pmem_min:.2f})",
            )
            cbar = fig.colorbar(points, ax=ax_s, fraction=0.046, pad=0.04)
            cbar.set_label(r"$P_{\mathrm{mem}}$")
        if simbad_plx is not None and np.isfinite(simbad_plx):
            ax_s.axvline(float(simbad_plx), color="crimson", ls="--", lw=1.2)
        ax_s.set_xlabel(r"$\varpi$ [mas]")
        ax_s.set_ylabel(r"$G$ [mag]")
        ax_s.set_xlim(plo, phi)
        ax_s.invert_yaxis()
        ax_s.set_title(r"$G$ vs ϖ (not $d=1/\varpi$)")
        ax_s.legend(fontsize=8)
        ax_s.grid(True, color="lightgray", linestyle="--", alpha=0.35)
    else:
        ax_s.scatter(plx[show], p_mem[show], s=8, c="C0", alpha=0.4, linewidths=0)
        ax_s.axhline(float(pmem_min), color="k", ls="--", lw=1, label="threshold")
        ax_s.set_xlabel(r"$\varpi$ [mas]")
        ax_s.set_ylabel(r"$P_{\mathrm{mem}}$")
        ax_s.set_xlim(plo, phi)
        ax_s.set_title(r"$P_{\mathrm{mem}}$ vs ϖ")
        ax_s.legend(fontsize=8)
        ax_s.grid(True, color="lightgray", linestyle="--", alpha=0.35)

    fig.suptitle("Cluster selection in parallax", fontsize=12)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _save(fig, out, "cluster_parallax", file_type)


def _plot_pmem_histogram(
    out,
    file_type: str,
    p_mem: np.ndarray,
    *,
    pmem_min: float,
    note: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.hist(p_mem, bins=np.linspace(0.0, 1.0, 41), color="C0", alpha=0.85)
    ax.axvline(
        float(pmem_min),
        color="k",
        ls="--",
        lw=1.4,
        label=rf"Threshold $P_{{\mathrm{{mem}}}} \geq {pmem_min:.2f}$",
    )
    n_hi = int(np.count_nonzero(p_mem >= float(pmem_min)))
    n_lo = int(p_mem.size - n_hi)
    ax.set_xlabel(r"$P_{\mathrm{mem}}$ (probability of the cluster Gaussian)")
    ax.set_ylabel("Number of quality Gaia stars")
    ax.set_title(
        r"Why these stars: $P_{\mathrm{mem}}$ is bimodal if cluster and field separate"
    )
    ax.legend(fontsize=8)
    ax.text(
        0.98,
        0.95,
        f"below threshold: {n_lo}\nabove threshold: {n_hi}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "lw": 0.4},
    )
    ax.grid(True, color="lightgray", linestyle="--", alpha=0.35)
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out, "cluster_pmem", file_type)


def _plot_mu_plx_3d(
    out,
    file_type: str,
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    member: np.ndarray,
    *,
    note: str,
    simbad_pm_ra: float | None,
    simbad_pm_de: float | None,
    simbad_plx: float | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    field = ~member
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle(
        r"Cluster membership in $(\mu_{\alpha*},\,\mu_{\delta},\,\varpi)$ "
        r"(QC only — not a distance axis)",
        fontsize=13,
    )
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.view_init(22, 40 + i * 90)
        if np.any(field):
            ax.scatter(
                pm_ra[field],
                pm_de[field],
                plx[field],
                s=4,
                c=_FIELD_COLOR,
                alpha=0.25,
                linewidths=0,
                label="Field",
            )
        if np.any(member):
            ax.scatter(
                pm_ra[member],
                pm_de[member],
                plx[member],
                s=8,
                c="C0",
                alpha=0.7,
                linewidths=0,
                label="Members",
            )
        if (
            simbad_pm_ra is not None
            and simbad_pm_de is not None
            and simbad_plx is not None
            and np.isfinite(simbad_pm_ra)
            and np.isfinite(simbad_pm_de)
            and np.isfinite(simbad_plx)
        ):
            ax.scatter(
                [float(simbad_pm_ra)],
                [float(simbad_pm_de)],
                [float(simbad_plx)],
                marker="*",
                s=120,
                c="crimson",
                edgecolors="k",
                linewidths=0.4,
                label="Simbad",
            )
        ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
        ax.set_ylabel(r"$\mu_{\delta}$ [mas/yr]")
        ax.set_zlabel(r"$\varpi$ [mas]")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8, wrap=True)
    _save(fig, out, "cluster_mu_plx_3d", file_type)


__all__ = ["plot_cluster_membership_diagnostics"]
