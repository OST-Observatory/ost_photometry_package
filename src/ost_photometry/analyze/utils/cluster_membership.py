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
    reason: str = ""


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
) -> tuple[int, str]:
    n_comp = int(gmm.n_components)
    if component_id is not None:
        cid = int(component_id)
        if 0 <= cid < n_comp:
            return cid, f"manual override (component {cid})"
    if simbad_scaled is not None:
        if hasattr(gmm, "component_log_density"):
            # a compact cluster component is far denser at the cluster's
            # (μ, π) than a broad field component whose mean happens to lie closer
            return int(np.argmax(gmm.component_log_density(simbad_scaled))), \
                "highest density at Simbad (μ, π)"
        means = np.asarray(gmm.means_, dtype=float)
        dist = np.linalg.norm(means - simbad_scaled[None, :], axis=1)
        return int(np.argmin(dist)), "nearest Simbad (μ, π)"
    spreads = []
    for k in range(n_comp):
        cov = np.atleast_2d(gmm.covariances_[k])
        spreads.append(float(np.trace(cov)))
    return int(np.argmin(spreads)), "tighter Gaussian"


class _XDModel:
    """Gaussian mixture fitted by extreme deconvolution (Bovy et al. 2011).

    Each star ``i`` is drawn from component ``k`` with the covariance
    ``V_k + S_i`` (intrinsic width plus the star's own errors); EM in log
    space, so stars far from every component (large proper motions in a
    wide field) cannot underflow all responsibilities to 0/0 (astroML's
    ``XDGMM`` fails there with NaN). ``means_`` / ``covariances_`` are the
    deconvolved component parameters.
    """

    #: Added to every intrinsic covariance (scaled units) to keep it positive definite.
    regularization = 1e-6

    def __init__(self, n_components: int, random_state: int, *, max_iter: int = 300, tol: float = 1e-6):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _log_resp(self, x: np.ndarray, x_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from scipy.special import logsumexp

        t = x_cov[:, None, :, :] + self.covariances_[None, :, :, :]
        t_inv = np.linalg.inv(t)
        _sign, logdet = np.linalg.slogdet(t)
        diff = x[:, None, :] - self.means_[None, :, :]
        maha = np.einsum("nki,nkij,nkj->nk", diff, t_inv, diff)
        d = x.shape[1]
        log_n = -0.5 * (maha + logdet + d * np.log(2.0 * np.pi)) + np.log(self.weights_)[None, :]
        log_norm = logsumexp(log_n, axis=1)
        return log_n - log_norm[:, None], log_norm, t_inv, diff

    def fit(self, x: np.ndarray, x_cov: np.ndarray) -> _XDModel:
        from sklearn.mixture import GaussianMixture

        n, d = x.shape
        init = GaussianMixture(self.n_components, covariance_type="full",
                               random_state=self.random_state, n_init=3).fit(x)
        self.means_ = np.asarray(init.means_, dtype=float)
        self.covariances_ = np.asarray(init.covariances_, dtype=float)
        self.weights_ = np.asarray(init.weights_, dtype=float)
        eye = np.eye(d) * self.regularization
        log_l = -np.inf
        for _ in range(self.max_iter):
            log_q, log_norm, t_inv, diff = self._log_resp(x, x_cov)
            new_log_l = float(np.sum(log_norm))
            q = np.exp(log_q)
            # E-step: expected true values b and their covariance B (eqs. 16-17)
            v = self.covariances_
            b = self.means_[None, :, :] + np.einsum("kij,nkjl,nkl->nki", v, t_inv, diff)
            big_b = v[None, :, :, :] - np.einsum("kij,nkjl,klm->nkim", v, t_inv, v)
            # M-step
            qj = np.maximum(q.sum(axis=0), 1e-300)
            self.weights_ = np.maximum(qj / n, 1e-12)
            self.means_ = np.einsum("nk,nki->ki", q, b) / qj[:, None]
            dev = b - self.means_[None, :, :]
            self.covariances_ = (
                np.einsum("nk,nki,nkj->kij", q, dev, dev) + np.einsum("nk,nkij->kij", q, big_b)
            ) / qj[:, None, None] + eye
            if new_log_l - log_l < self.tol * n:
                break
            log_l = new_log_l
        self.log_likelihood_ = float(np.sum(self._log_resp(x, x_cov)[1]))
        return self

    def bic(self, n: int, d: int) -> float:
        params = self.n_components * (d + d * (d + 1) / 2) + self.n_components - 1
        return -2.0 * self.log_likelihood_ + params * np.log(n)

    def predict_proba(self, x: np.ndarray, x_cov: np.ndarray) -> np.ndarray:
        return np.exp(self._log_resp(x, x_cov)[0])

    def component_log_density(self, point: np.ndarray) -> np.ndarray:
        """Log density of each component at ``point`` (intrinsic covariance, no weight)."""
        diff = point[None, :] - self.means_
        inv = np.linalg.inv(self.covariances_)
        _sign, logdet = np.linalg.slogdet(self.covariances_)
        return -0.5 * (np.einsum("ki,kij,kj->k", diff, inv, diff) + logdet)


def _fit_xd(x: np.ndarray, x_cov: np.ndarray, random_state: int,
            components: tuple[int, ...] = (2, 3, 4, 5)) -> _XDModel:
    """XD mixture with the number of components chosen by BIC.

    Two components are not enough for a wide field: the field itself is
    not one Gaussian, takes both components, and the cluster disappears.
    """
    n, d = x.shape
    best = None
    for k in components:
        if n < 10 * k:
            break
        model = _XDModel(k, random_state).fit(x, x_cov)
        if best is None or model.bic(n, d) < best.bic(n, d):
            best = model
    return best if best is not None else _XDModel(2, random_state).fit(x, x_cov)


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
            reason="HDBSCAN unavailable or too few stars",
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
            reason="HDBSCAN found no core",
        )
    if component_id is not None and int(component_id) in unique:
        chosen = int(component_id)
        reason = f"manual override (component {chosen})"
    elif simbad_scaled is not None:
        centroids = []
        for cid in unique:
            centroids.append(scaled[labels == cid].mean(axis=0))
        dist = np.linalg.norm(
            np.asarray(centroids, dtype=float) - simbad_scaled[None, :],
            axis=1,
        )
        chosen = unique[int(np.argmin(dist))]
        reason = "nearest Simbad (μ, π)"
    else:
        counts = [int(np.count_nonzero(labels == cid)) for cid in unique]
        chosen = unique[int(np.argmax(counts))]
        reason = "largest HDBSCAN core"
    strength = np.asarray(getattr(clusterer, "probabilities_", np.ones(n)), dtype=float)
    p_mem = np.zeros(n, dtype=float)
    in_cluster = labels == chosen
    p_mem[in_cluster] = np.clip(strength[in_cluster], 0.0, 1.0)
    return MembershipResult(
        p_mem=p_mem,
        labels=labels,
        cluster_component=chosen,
        method="hdbscan",
        reason=reason,
    )


def membership_from_astrometry(
    pm_ra: np.ndarray,
    pm_de: np.ndarray,
    plx: np.ndarray,
    *,
    cov: np.ndarray | None = None,
    method: str = "gmm",
    simbad_pm_ra: float | None = None,
    simbad_pm_de: float | None = None,
    simbad_plx: float | None = None,
    component_id: int | None = None,
    random_state: int = 0,
) -> MembershipResult:
    """P(member) from a Gaussian mixture in z-scaled (μ_α*, μ_δ, ϖ).

    With ``cov`` (per-star 3×3 covariance of (μ_α*, μ_δ, ϖ), e.g. from
    :func:`~ost_photometry.analyze.utils.cluster_selection.gaia_astrometry_covariance`)
    the mixture is fitted by extreme deconvolution (:class:`_XDModel`, EM in
    log space): the components describe the intrinsic distributions, and
    each star's responsibility uses its own errors, so faint stars with
    large errors are not pushed into the broad field component. The number
    of components (2-5) is chosen by BIC, so a wide field can take several.
    Without ``cov`` a plain 2-component Gaussian mixture is used.
    ``method='xd'`` requires ``cov``.

    The cluster component is the one with the highest density at the
    Simbad (μ, π) for XD (nearest mean for the plain mixture) when given,
    otherwise the tighter Gaussian (smaller covariance trace).
    ``component_id`` overrides.
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
            reason="no stars",
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
            reason="too few stars; all kept",
        )
    scaled_cov = None
    if cov is not None:
        c = np.asarray(cov, dtype=float)
        s = np.asarray(scaler.scale_, dtype=float)
        scaled_cov = c / (s[:, None] * s[None, :])[None, :, :]
        bad = ~np.all(np.isfinite(scaled_cov), axis=(1, 2))
        # a star without errors: tiny isotropic error (treated as exact)
        scaled_cov[bad] = np.eye(3) * 1e-6
    elif method_key == "xd":
        raise ValueError("method='xd' needs the per-star covariance (cov=...)")
    if scaled_cov is not None:
        gmm = _fit_xd(scaled, scaled_cov, int(random_state))
    else:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=int(random_state),
            n_init=5,
        )
        gmm.fit(scaled)
    if (
        method_key in ("gmm", "xd")
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
            extra = "GMM components had similar spread"
            reason = f"{fallback.reason}; {extra}" if fallback.reason else extra
            return MembershipResult(
                p_mem=fallback.p_mem,
                labels=fallback.labels,
                cluster_component=fallback.cluster_component,
                method=fallback.method,
                reason=reason,
            )
    cluster_k, reason = _gmm_cluster_component(
        gmm, simbad_scaled=simbad_scaled, component_id=component_id
    )
    if scaled_cov is not None:
        proba = gmm.predict_proba(scaled, scaled_cov)
    else:
        proba = gmm.predict_proba(scaled)
    labels = np.argmax(proba, axis=1)
    p_mem = np.asarray(proba[:, cluster_k], dtype=float)
    return MembershipResult(
        p_mem=np.clip(p_mem, 0.0, 1.0),
        labels=np.asarray(labels, dtype=int),
        cluster_component=int(cluster_k),
        method="xd" if scaled_cov is not None else "gmm",
        reason=reason,
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
