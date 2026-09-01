"""Astrometric open-cluster membership in Gaia (μ_α*, μ_δ, ϖ).

No catalog I/O: callers pass arrays or an already-downloaded Gaia table.
Distances are not used; ``d = 1/ϖ`` is biased and mishandles π ≤ 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, matching
from astropy.time import Time

from .duplicates import clear_duplicates

GAIA_DR3_JYEAR = 2016.0
_MAS_PER_DEGREE = 3.6e6


def as_float_column(column) -> np.ndarray:
    """Numeric column as float, with masked values filled by NaN."""
    arr = np.asanyarray(column)
    if np.ma.isMaskedArray(arr):
        return np.ma.filled(arr.astype(float), np.nan)
    return np.asarray(arr, dtype=float)


def gaia_quality_mask(
    *,
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    ruwe: np.ndarray | None = None,
    plx_err: np.ndarray | None = None,
    ruwe_max: float = 1.4,
    plx_snr_min: float | None = None,
    plx_min_mas: float | None = None,
) -> np.ndarray:
    """Finite μ/π, optional RUWE, π/σ_π, and π-min (from a max-distance pre-cut)."""
    pm_ra = as_float_column(pm_ra)
    pm_de = as_float_column(pm_de)
    plx = as_float_column(plx)
    keep = np.isfinite(pm_ra) & np.isfinite(pm_de) & np.isfinite(plx)
    if ruwe is not None and ruwe_max is not None and np.isfinite(ruwe_max):
        ruwe_arr = as_float_column(ruwe)
        ruwe_ok = np.isfinite(ruwe_arr) & (ruwe_arr <= float(ruwe_max))
        keep &= ruwe_ok
    if (
        plx_err is not None
        and plx_snr_min is not None
        and np.isfinite(plx_snr_min)
        and plx_snr_min > 0
    ):
        err = as_float_column(plx_err)
        snr_ok = np.isfinite(err) & (err > 0) & ((plx / err) >= float(plx_snr_min))
        keep &= snr_ok
    if plx_min_mas is not None and np.isfinite(plx_min_mas):
        keep &= plx >= float(plx_min_mas)
    return keep


def plx_min_mas_from_distance_kpc(max_distance_kpc: float | None) -> float | None:
    """``ϖ [mas] ≥ 1 / d [kpc]`` (since d[kpc] = 1/ϖ[mas]). None if unset."""
    if max_distance_kpc is None:
        return None
    try:
        dist = float(max_distance_kpc)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(dist) or dist <= 0:
        return None
    return 1.0 / dist


def years_since_gaia_dr3(jd: float | None) -> float:
    """Julian years from Gaia DR3 epoch J2016.0 to ``jd``. ``0`` if ``jd`` is unset."""
    if jd is None:
        return 0.0
    try:
        value = float(jd)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value <= 0:
        return 0.0
    obs = Time(value, format="jd")
    gaia = Time(GAIA_DR3_JYEAR, format="jyear")
    return float((obs - gaia).to_value(u.yr))


def propagate_gaia_positions(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order sky shift: Gaia ``pmRA`` is μ_α* = μα cos δ (mas/yr)."""
    ra = as_float_column(ra_deg)
    dec = as_float_column(dec_deg)
    mu_a = as_float_column(pm_ra)
    mu_d = as_float_column(pm_de)
    dt = float(years)
    if not np.isfinite(dt) or dt == 0.0:
        return ra.copy(), dec.copy()
    cos_dec = np.cos(np.deg2rad(dec))
    cos_dec = np.where(np.abs(cos_dec) < 1e-6, np.nan, cos_dec)
    ra_new = ra + (mu_a * dt) / (_MAS_PER_DEGREE * cos_dec)
    dec_new = dec + (mu_d * dt) / _MAS_PER_DEGREE
    return ra_new, dec_new


def match_photometry_to_gaia(
    obj_coordinates: SkyCoord,
    gaia_coordinates: SkyCoord,
    *,
    separation_arcsec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique nearest pairs within ``separation_arcsec`` (keep smallest separation)."""
    id_img, id_gaia, d2d, _ = matching.search_around_sky(
        obj_coordinates,
        gaia_coordinates,
        float(separation_arcsec) * u.arcsec,
    )
    sep = np.asarray(d2d.arcsec, dtype=float)
    id_img, sep, id_gaia = clear_duplicates(id_img, sep, id_gaia)
    id_gaia, sep, id_img = clear_duplicates(id_gaia, sep, id_img)
    return (
        np.asarray(id_img, dtype=int),
        np.asarray(id_gaia, dtype=int),
        np.asarray(sep, dtype=float),
    )


@dataclass(frozen=True)
class MembershipResult:
    """Soft membership for one sample in (μ_α*, μ_δ, ϖ)."""

    p_mem: np.ndarray
    labels: np.ndarray
    cluster_component: int
    method: str


def _scaled_features(
    pm_ra: np.ndarray, pm_de: np.ndarray, plx: np.ndarray
):
    from sklearn.preprocessing import StandardScaler

    features = np.column_stack(
        (
            as_float_column(pm_ra),
            as_float_column(pm_de),
            as_float_column(plx),
        )
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def _simbad_scaled_point(
    scaler,
    simbad_pm_ra: float | None,
    simbad_pm_de: float | None,
    simbad_plx: float | None,
) -> np.ndarray | None:
    if simbad_pm_ra is None or simbad_pm_de is None:
        return None
    plx = 0.0 if simbad_plx is None else float(simbad_plx)
    try:
        point = np.array(
            [[float(simbad_pm_ra), float(simbad_pm_de), plx]],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(point)):
        return None
    if simbad_plx is None:
        # Match in the μ plane only: use the sample-mean π (scaled ≈ 0).
        point[0, 2] = scaler.mean_[2]
    return scaler.transform(point)[0]


def _gmm_cluster_component(
    gmm,
    *,
    simbad_scaled: np.ndarray | None,
    component_id: int | None,
) -> int:
    n_comp = int(gmm.n_components)
    if component_id is not None:
        cid = int(component_id)
        if 0 <= cid < n_comp:
            return cid
    if simbad_scaled is not None:
        means = np.asarray(gmm.means_, dtype=float)
        dist = np.linalg.norm(means - simbad_scaled[None, :], axis=1)
        return int(np.argmin(dist))
    spreads = []
    for k in range(n_comp):
        cov = np.atleast_2d(gmm.covariances_[k])
        spreads.append(float(np.trace(cov)))
    return int(np.argmin(spreads))


def _gmm_components_look_alike(gmm, *, ratio_min: float = 1.5) -> bool:
    if gmm.n_components < 2:
        return False
    traces = []
    for k in range(gmm.n_components):
        cov = np.atleast_2d(gmm.covariances_[k])
        traces.append(float(np.trace(cov)))
    lo = min(traces)
    hi = max(traces)
    if lo <= 0:
        return True
    return (hi / lo) < float(ratio_min)


def _hdbscan_membership(
    scaled: np.ndarray,
    *,
    simbad_scaled: np.ndarray | None,
    component_id: int | None,
    random_state: int,
) -> MembershipResult:
    n = scaled.shape[0]
    _ = random_state
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        HDBSCAN = None
    if HDBSCAN is None or n < 8:
        return MembershipResult(
            p_mem=np.zeros(n, dtype=float),
            labels=np.zeros(n, dtype=int),
            cluster_component=-1,
            method="none",
        )
    min_size = max(8, n // 40)
    clusterer = HDBSCAN(
        min_cluster_size=min_size,
        min_samples=max(3, min_size // 3),
    )
    labels = np.asarray(clusterer.fit_predict(scaled), dtype=int)
    unique = [int(c) for c in np.unique(labels) if c >= 0]
    if not unique:
        return MembershipResult(
            p_mem=np.zeros(n, dtype=float),
            labels=labels,
            cluster_component=-1,
            method="hdbscan",
        )
    if component_id is not None and int(component_id) in unique:
        chosen = int(component_id)
    elif simbad_scaled is not None:
        centroids = []
        for cid in unique:
            centroids.append(scaled[labels == cid].mean(axis=0))
        dist = np.linalg.norm(
            np.asarray(centroids, dtype=float) - simbad_scaled[None, :],
            axis=1,
        )
        chosen = unique[int(np.argmin(dist))]
    else:
        counts = [int(np.count_nonzero(labels == cid)) for cid in unique]
        chosen = unique[int(np.argmax(counts))]
    strength = np.asarray(getattr(clusterer, "probabilities_", np.ones(n)), dtype=float)
    p_mem = np.zeros(n, dtype=float)
    in_cluster = labels == chosen
    p_mem[in_cluster] = np.clip(strength[in_cluster], 0.0, 1.0)
    return MembershipResult(
        p_mem=p_mem,
        labels=labels,
        cluster_component=chosen,
        method="hdbscan",
    )


def membership_from_astrometry(
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    *,
    method: str = "gmm",
    simbad_pm_ra: float | None = None,
    simbad_pm_de: float | None = None,
    simbad_plx: float | None = None,
    component_id: int | None = None,
    random_state: int = 0,
) -> MembershipResult:
    """P(member) from a 2-component GMM in z-scaled (μ_α*, μ_δ, ϖ).

    The cluster component is the one nearest Simbad (μ, π) when given, otherwise
    the tighter Gaussian (smaller covariance trace). ``component_id`` overrides.
    If both Gaussians look like field (similar spread) and no Simbad prior,
    fall back to HDBSCAN when ``method='gmm'``.
    """
    pm_ra = as_float_column(pm_ra)
    pm_de = as_float_column(pm_de)
    plx = as_float_column(plx)
    n = pm_ra.size
    if n == 0:
        return MembershipResult(
            p_mem=np.zeros(0, dtype=float),
            labels=np.zeros(0, dtype=int),
            cluster_component=-1,
            method="none",
        )
    scaled, scaler = _scaled_features(pm_ra, pm_de, plx)
    simbad_scaled = _simbad_scaled_point(
        scaler, simbad_pm_ra, simbad_pm_de, simbad_plx
    )
    method_key = str(method).strip().lower()
    if method_key == "hdbscan":
        return _hdbscan_membership(
            scaled,
            simbad_scaled=simbad_scaled,
            component_id=component_id,
            random_state=random_state,
        )
    if n < 4:
        return MembershipResult(
            p_mem=np.ones(n, dtype=float),
            labels=np.zeros(n, dtype=int),
            cluster_component=0,
            method="gmm",
        )
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(random_state),
        n_init=5,
    )
    gmm.fit(scaled)
    if (
        method_key == "gmm"
        and simbad_scaled is None
        and component_id is None
        and _gmm_components_look_alike(gmm)
    ):
        fallback = _hdbscan_membership(
            scaled,
            simbad_scaled=None,
            component_id=None,
            random_state=random_state,
        )
        if fallback.cluster_component >= 0:
            return fallback
    cluster_k = _gmm_cluster_component(
        gmm, simbad_scaled=simbad_scaled, component_id=component_id
    )
    proba = gmm.predict_proba(scaled)
    labels = gmm.predict(scaled)
    p_mem = np.asarray(proba[:, cluster_k], dtype=float)
    return MembershipResult(
        p_mem=np.clip(p_mem, 0.0, 1.0),
        labels=np.asarray(labels, dtype=int),
        cluster_component=int(cluster_k),
        method="gmm",
    )


__all__ = [
    "GAIA_DR3_JYEAR",
    "MembershipResult",
    "as_float_column",
    "gaia_quality_mask",
    "match_photometry_to_gaia",
    "membership_from_astrometry",
    "plx_min_mas_from_distance_kpc",
    "propagate_gaia_positions",
    "years_since_gaia_dr3",
]
