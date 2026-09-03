"""Cluster / region selection using Gaia DR3 astrometry."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from ... import terminal_output
from ...core.parallel import start_plot_process
from ..plots.cluster_membership import plot_cluster_membership_diagnostics
from ..post_processing.coords import (
    plot_starmap_from_imaging_context,
    table_object_sky_coords,
)
from ..post_processing.imaging import ImagingPlotContext, imaging_context_from_image_series
from .cluster_membership import (
    as_float_column,
    gaia_quality_mask,
    match_photometry_to_gaia,
    membership_from_astrometry,
    plx_min_mas_from_distance_kpc,
    propagate_gaia_positions,
    years_since_gaia_dr3,
)

if TYPE_CHECKING:
    from .. import analyze

GAIA_DR3_CATALOG = "I/355/gaiadr3"
GAIA_DR3_COLUMNS = (
    "RA_ICRS",
    "DE_ICRS",
    "Gmag",
    "Plx",
    "e_Plx",
    "pmRA",
    "e_pmRA",
    "pmDE",
    "e_pmDE",
    "RUWE",
)


def _resolve_imaging_plot_context(
    *,
    image_series: analyze.ImageSeries | None = None,
    plot_context: ImagingPlotContext | None = None,
) -> ImagingPlotContext:
    if plot_context is not None:
        return plot_context
    if image_series is not None:
        return imaging_context_from_image_series(image_series)
    raise TypeError(
        "Provide plot_context=... or image_series=... (for example "
        "plot_context=imaging_context_from_image_series(series))."
    )


def _vizier_field_cone(
    ctx: ImagingPlotContext,
    image_series: analyze.ImageSeries | None,
) -> tuple[SkyCoord, u.Quantity]:
    """Center and radius for ``Vizier.query_region`` (Gaia cone)."""
    if ctx.field_center_icrs is not None and ctx.field_radius_arcmin is not None:
        return ctx.field_center_icrs, ctx.field_radius_arcmin * u.arcmin
    if image_series is not None:
        return (
            image_series.coordinates_image_center,
            image_series.field_of_view_x * u.arcmin,
        )
    raise TypeError(
        "ImagingPlotContext.field_center_icrs and field_radius_arcmin must be set, "
        "or pass image_series=..., for Gaia / Vizier cone queries."
    )


def query_gaia_dr3_cone(
    center: SkyCoord,
    radius: u.Quantity,
    *,
    catalog: str = GAIA_DR3_CATALOG,
    g_mag_limit: float = 20.0,
) -> Table:
    """Download Gaia DR3 rows in a cone (Vizier)."""
    vizier = Vizier(
        columns=list(GAIA_DR3_COLUMNS),
        row_limit=1_000_000,
        catalog=catalog,
        column_filters={"Gmag": f"<{float(g_mag_limit)}"},
    )
    result = vizier.query_region(center, radius=radius)
    if result is None or len(result) == 0:
        raise RuntimeError("Gaia Vizier cone query returned no tables.")
    tbl = result[0]
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("Gaia Vizier cone query returned an empty table.")
    return tbl


def _simbad_field(table: Table, *names: str):
    cols = {str(c).lower(): c for c in table.colnames}
    for name in names:
        key = cols.get(name.lower())
        if key is not None:
            return table[key][0]
    return None


def _simbad_scalar(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "--", "masked", "None"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def query_simbad_astrometry(name: str) -> tuple[float | None, float | None, float | None]:
    """Simbad μ_α*, μ_δ (mas/yr) and π (mas). Missing fields are ``None``."""
    if not str(name).strip() or str(name).strip() in {"?", "--"}:
        return None, None, None
    custom = Simbad()
    try:
        custom.add_votable_fields("pmra", "pmdec", "plx")
    except Exception:
        pass
    try:
        result = custom.query_object(str(name).strip())
    except Exception:
        return None, None, None
    if result is None or len(result) == 0:
        return None, None, None
    pm_ra = _simbad_scalar(_simbad_field(result, "pmra", "PMRA"))
    pm_de = _simbad_scalar(_simbad_field(result, "pmdec", "pmde", "PMDEC"))
    plx = _simbad_scalar(_simbad_field(result, "plx", "PLX_VALUE", "plx_value"))
    return pm_ra, pm_de, plx


def _observation_jd(
    image_series: analyze.ImageSeries | None,
    observation_jd: float | None,
) -> float | None:
    if observation_jd is not None:
        return float(observation_jd)
    if image_series is None:
        return None
    try:
        jd = float(image_series.median_observation_time())
    except Exception:
        return None
    return jd


def _gaia_skycoord(gaia: Table, *, years: float) -> SkyCoord:
    ra = as_float_column(gaia["RA_ICRS"])
    dec = as_float_column(gaia["DE_ICRS"])
    pm_ra = as_float_column(gaia["pmRA"])
    pm_de = as_float_column(gaia["pmDE"])
    ra_obs, dec_obs = propagate_gaia_positions(ra, dec, pm_ra, pm_de, years)
    return SkyCoord(ra_obs, dec_obs, unit=(u.degree, u.degree), frame="icrs")


def _gaia_astrometry_arrays(gaia: Table) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    pm_ra = as_float_column(gaia["pmRA"])
    pm_de = as_float_column(gaia["pmDE"])
    plx = as_float_column(gaia["Plx"])
    ruwe = as_float_column(gaia["RUWE"]) if "RUWE" in gaia.colnames else None
    plx_err = as_float_column(gaia["e_Plx"]) if "e_Plx" in gaia.colnames else None
    gmag = as_float_column(gaia["Gmag"]) if "Gmag" in gaia.colnames else None
    return pm_ra, pm_de, plx, ruwe, plx_err, gmag


def _empty_cluster_table(tbl: Table) -> Table:
    empty = tbl[0:0]
    empty.meta["cluster_membership"] = {
        "ids": [],
        "p_mem": [],
        "p_mem_by_id": {},
    }
    return empty


def _plot_membership_diagnostics(
    *,
    plot_stub: str,
    file_type: str,
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    p_mem: np.ndarray,
    gmag: np.ndarray | None,
    pmem_min: float,
    method: str,
    cluster_component: int,
    reason: str,
    simbad_pm_ra: float | None,
    simbad_pm_de: float | None,
    simbad_plx: float | None,
) -> None:
    start_plot_process(
        plot_cluster_membership_diagnostics,
        (),
        {
            "output_dir": plot_stub,
            "file_type": file_type,
            "pm_ra": pm_ra,
            "pm_de": pm_de,
            "plx": plx,
            "p_mem": p_mem,
            "gmag": gmag,
            "pmem_min": pmem_min,
            "method": method,
            "cluster_component": cluster_component,
            "reason": reason,
            "simbad_pm_ra": simbad_pm_ra,
            "simbad_pm_de": simbad_pm_de,
            "simbad_plx": simbad_plx,
        },
    )


def find_cluster(
    tbl: Table,
    object_names: list[str],
    *,
    image_series: analyze.ImageSeries | None = None,
    plot_context: ImagingPlotContext | None = None,
    catalog: str = GAIA_DR3_CATALOG,
    g_mag_limit: float = 20.0,
    separation_limit: float = 1.0,
    max_distance: float = 6.0,
    parameter_set: int | None = None,
    file_type_plots: str = "pdf",
    use_wcs_projection_for_star_maps: bool = True,
    cluster_selection_id: int | None = None,
    ruwe_max: float = 1.4,
    plx_snr_min: float | None = None,
    pmem_min: float = 0.5,
    membership_method: str = "gmm",
    cluster_component_id: int | None = None,
    observation_jd: float | None = None,
    indent: int = 1,
) -> tuple[Table, np.ndarray, np.ndarray, np.ndarray]:
    """Identify cluster members in scaled Gaia (μ_α*, μ_δ, ϖ).

    Quality-filter the Gaia cone, fit membership, then match photometry
    (~1 arcsec). Returns the **member subset** of ``tbl`` (P_mem ≥ ``pmem_min``).
    All Gaia-matched stars and their ``P_mem`` are stored on
    ``returned.meta['cluster_membership']`` for the full-table ECSV write.

    ``parameter_set`` is ignored (deprecated SpectralClustering presets).
    ``cluster_selection_id`` is an alias for ``cluster_component_id``.
    """
    if parameter_set is not None:
        warnings.warn(
            "find_cluster(parameter_set=...) is deprecated and ignored; "
            "membership uses a GMM/HDBSCAN in (μ_α*, μ_δ, ϖ).",
            DeprecationWarning,
            stacklevel=2,
        )
    if cluster_component_id is None and cluster_selection_id is not None:
        cluster_component_id = int(cluster_selection_id)

    detail = indent + 1
    terminal_output.print_to_terminal(
        "Cluster selection (Gaia astrometry)",
        indent=indent,
        style_name="HEADER",
    )
    ctx = _resolve_imaging_plot_context(
        image_series=image_series, plot_context=plot_context
    )
    obj_coordinates = table_object_sky_coords(tbl, ctx.wcs)
    plot_stub = str(ctx.out_path_stub)
    v_center, v_radius = _vizier_field_cone(ctx, image_series)

    gaia = query_gaia_dr3_cone(
        v_center,
        v_radius,
        catalog=catalog,
        g_mag_limit=g_mag_limit,
    )

    object_name = ""
    if object_names:
        object_name = str(object_names[0]).strip()
    simbad_pm_ra, simbad_pm_de, simbad_plx = query_simbad_astrometry(object_name)

    years = years_since_gaia_dr3(_observation_jd(image_series, observation_jd))
    pm_ra, pm_de, plx, ruwe, plx_err, gmag = _gaia_astrometry_arrays(gaia)
    quality = gaia_quality_mask(
        pm_ra=pm_ra,
        pm_de=pm_de,
        plx=plx,
        ruwe=ruwe,
        plx_err=plx_err,
        ruwe_max=ruwe_max,
        plx_snr_min=plx_snr_min,
        plx_min_mas=plx_min_mas_from_distance_kpc(max_distance),
    )
    if not np.any(quality):
        terminal_output.print_to_terminal(
            "No Gaia sources survived the quality cuts (RUWE / π).",
            indent=detail,
            style_name="WARNING",
        )
        empty = _empty_cluster_table(tbl)
        return empty, np.zeros(0, dtype=int), quality, np.zeros(0, dtype=bool)

    gaia_q = gaia[quality]
    pm_ra_q = pm_ra[quality]
    pm_de_q = pm_de[quality]
    plx_q = plx[quality]
    gmag_q = None if gmag is None else gmag[quality]

    result = membership_from_astrometry(
        pm_ra_q,
        pm_de_q,
        plx_q,
        method=membership_method,
        simbad_pm_ra=simbad_pm_ra,
        simbad_pm_de=simbad_pm_de,
        simbad_plx=simbad_plx,
        component_id=cluster_component_id,
    )
    gaia_member = result.p_mem >= float(pmem_min)
    n_gaia_mem = int(np.count_nonzero(gaia_member))

    _plot_membership_diagnostics(
        plot_stub=plot_stub,
        file_type=file_type_plots,
        pm_ra=pm_ra_q,
        pm_de=pm_de_q,
        plx=plx_q,
        p_mem=result.p_mem,
        gmag=gmag_q,
        pmem_min=float(pmem_min),
        method=result.method,
        cluster_component=int(result.cluster_component),
        reason=result.reason,
        simbad_pm_ra=simbad_pm_ra,
        simbad_pm_de=simbad_pm_de,
        simbad_plx=simbad_plx,
    )

    gaia_coordinates = _gaia_skycoord(gaia_q, years=years)
    id_img, id_gaia, _sep = match_photometry_to_gaia(
        obj_coordinates,
        gaia_coordinates,
        separation_arcsec=separation_limit,
    )
    if id_img.size == 0:
        terminal_output.print_to_terminal(
            "No photometry–Gaia matches within the separation limit.",
            indent=detail,
            style_name="WARNING",
        )
        empty = _empty_cluster_table(tbl)
        return empty, id_img, quality, gaia_member

    p_mem_matched = np.asarray(result.p_mem[id_gaia], dtype=float)
    member_mask = p_mem_matched >= float(pmem_min)
    n_phot_mem = int(np.count_nonzero(member_mask))
    why = f", {result.reason}" if result.reason else ""
    terminal_output.print_to_terminal(
        f"Gaia membership ({result.method}): {n_gaia_mem}/{gaia_member.size} "
        f"quality stars with P_mem ≥ {float(pmem_min):.2f} "
        f"(component {result.cluster_component}{why}); "
        f"{n_phot_mem} photometry matches kept in memory.",
        indent=detail,
        style_name="GOOD",
    )

    phot_ids = np.asarray(tbl["id"][id_img]) if "id" in tbl.colnames else id_img
    p_mem_by_id = {
        int(sid): float(p)
        for sid, p in zip(phot_ids, p_mem_matched, strict=True)
    }

    members = tbl[id_img][member_mask]
    members["cluster_p_mem"] = p_mem_matched[member_mask]
    members.meta["cluster_membership"] = {
        "ids": [int(i) for i in phot_ids],
        "p_mem": [float(p) for p in p_mem_matched],
        "method": result.method,
        "cluster_component": int(result.cluster_component),
        "reason": result.reason,
        "p_mem_by_id": p_mem_by_id,
    }

    if len(tbl) > 0:
        plot_starmap_from_imaging_context(
            ctx,
            tbl,
            tbl_2=members if len(members) > 0 else None,
            filter_=ctx.filter_name,
            x_name="x",
            y_name="y",
            rts_pre="Gaia membership",
            label="Stars with photometric extractions",
            label_2="Cluster members (Gaia μ, π)",
            covariance_on_tbl_2=True,
            add_image_id=False,
            use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
            file_type_plots=file_type_plots,
        )
    return members, id_img, quality, member_mask


def proper_motion_selection(
    tbl: Table,
    *,
    image_series: analyze.ImageSeries | None = None,
    plot_context: ImagingPlotContext | None = None,
    catalog: str = GAIA_DR3_CATALOG,
    g_mag_limit: int | float = 20,
    separation_limit: float = 1.0,
    sigma: float = 3.0,
    max_n_iterations_sigma_clipping: int = 3,
    use_wcs_projection_for_star_maps: bool = True,
    file_type_plots: str = "pdf",
    object_names: list[str] | None = None,
    max_distance: float = 6.0,
    ruwe_max: float = 1.4,
    plx_snr_min: float | None = None,
    pmem_min: float = 0.5,
    membership_method: str = "gmm",
    cluster_component_id: int | None = None,
    observation_jd: float | None = None,
) -> Table:
    """Deprecated alias: same Gaia (μ, π) membership as :func:`find_cluster`.

    The old 1-D σ-clip (which kept PM *outliers*) is gone. ``sigma`` and
    ``max_n_iterations_sigma_clipping`` are ignored.
    """
    warnings.warn(
        "proper_motion_selection is deprecated; it now calls find_cluster "
        "(GMM/HDBSCAN in μ, π). Prefer identify_cluster_gaia_data and leave "
        "clean_objects_using_proper_motion=False.",
        DeprecationWarning,
        stacklevel=2,
    )
    if sigma != 3.0 or max_n_iterations_sigma_clipping != 3:
        warnings.warn(
            "proper_motion_selection(sigma=..., max_n_iterations_sigma_clipping=...) "
            "is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    names = list(object_names) if object_names else [""]
    members, _, _, _ = find_cluster(
        tbl,
        names,
        image_series=image_series,
        plot_context=plot_context,
        catalog=catalog,
        g_mag_limit=float(g_mag_limit),
        separation_limit=separation_limit,
        max_distance=max_distance,
        file_type_plots=file_type_plots,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        ruwe_max=ruwe_max,
        plx_snr_min=plx_snr_min,
        pmem_min=pmem_min,
        membership_method=membership_method,
        cluster_component_id=cluster_component_id,
        observation_jd=observation_jd,
    )
    return members


def region_selection(
    coordinates_target: SkyCoord | list[SkyCoord],
    tbl: Table,
    *,
    image_series: analyze.ImageSeries | None = None,
    plot_context: ImagingPlotContext | None = None,
    radius: float = 600.0,
    file_type_plots: str = "pdf",
    use_wcs_projection_for_star_maps: bool = True,
) -> tuple[Table, np.ndarray]:
    """Keep sources within ``radius`` arcsec of the cluster (Simbad / OOI) position."""
    ctx = _resolve_imaging_plot_context(
        image_series=image_series, plot_context=plot_context
    )
    obj_coordinates = table_object_sky_coords(tbl, ctx.wcs)

    if isinstance(coordinates_target, list):
        mask = np.zeros(len(obj_coordinates), dtype=bool)
        for target_coordinates in coordinates_target:
            sep = obj_coordinates.separation(target_coordinates)
            mask = mask | (sep.arcsec <= radius)
    else:
        sep = obj_coordinates.separation(coordinates_target)
        mask = sep.arcsec <= radius

    tbl = tbl[mask]
    plot_starmap_from_imaging_context(
        ctx,
        tbl,
        filter_=ctx.filter_name,
        x_name="x",
        y_name="y",
        rts_pre="radius selection, image",
        label=f"Objects selected within {radius}'' of the target",
        add_image_id=True,
        use_wcs_projection_for_star_maps=use_wcs_projection_for_star_maps,
        file_type_plots=file_type_plots,
    )
    return tbl, mask


__all__ = [
    "GAIA_DR3_CATALOG",
    "GAIA_DR3_COLUMNS",
    "find_cluster",
    "proper_motion_selection",
    "query_gaia_dr3_cone",
    "query_simbad_astrometry",
    "region_selection",
]
