"""Isochrone fit statistics: fiducials, χ² with nuisance offsets, age interval.

The CMD is binned in magnitude; each bin gives a fiducial point (colour,
magnitude) with a 2×2 covariance. An isochrone is scored by the Mahalanobis
distance of every fiducial to its nearest (in that metric) isochrone point:

``χ² = Σ_i min_k r_ik^T C_i^{-1} r_ik``.

Distance and reddening shift the isochrone by ``(δE·c, δM + δE·a)`` (colour,
magnitude) with Gaussian priors ``δM ~ N(0, σ_M)`` (distance modulus and the
R_V part of the extinction), ``δE ~ N(0, σ_E)`` (colour excess); they are
profiled (minimised) per isochrone instead of being added to every star's
error. The age interval follows from the profile ``χ²(age)`` with
``Δχ² = s`` (``s = max(1, χ²_min/ν)``, Birge scaling when the model does not
describe the fiducials to their errors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_MAD_TO_SIGMA = 1.4826
_MEDIAN_FACTOR = 1.2533  # σ of the median / σ of the mean for a normal sample


@dataclass
class Fiducials:
    """Binned CMD points with their 1σ errors and covariance."""

    mag: np.ndarray
    color: np.ndarray
    mag_err: np.ndarray
    color_err: np.ndarray
    cov: np.ndarray  # cov(colour, mag) per fiducial
    n: np.ndarray

    def __len__(self) -> int:
        return int(self.mag.size)


def _robust_sigma(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size < 3:
        return float("nan")
    return float(_MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def _detrended(mag: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Colour minus a straight line in magnitude (the sequence within a bin)."""
    if mag.size < 5 or np.ptp(mag) <= 0:
        return color - np.median(color)
    slope, intercept = np.polyfit(mag - np.mean(mag), color, 1)
    return color - (intercept + slope * (mag - np.mean(mag)))


def _weighted_mean(values, err):
    """IVW mean and its error; plain mean / NaN error without usable errors."""
    if err is not None:
        e = np.asarray(err, dtype=float)
        ok = np.isfinite(values) & np.isfinite(e) & (e > 0)
        if np.any(ok):
            w = 1.0 / e[ok] ** 2
            return float(np.sum(w * values[ok]) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w))), w, ok
    ok = np.isfinite(values)
    return float(np.mean(values[ok])), float("nan"), None, ok


def fiducial_sigma(ivw_err: float, scatter: float, n: int) -> float:
    """``max(IVW error, 1.2533 · σ_robust / √N)``; NaN when neither exists.

    The scatter term covers the intrinsic width of the sequence (binaries,
    differential reddening, field contamination) that the photometric
    errors do not describe.
    """
    terms = []
    if np.isfinite(ivw_err) and ivw_err > 0:
        terms.append(float(ivw_err))
    if np.isfinite(scatter) and scatter > 0 and n > 0:
        terms.append(_MEDIAN_FACTOR * float(scatter) / np.sqrt(n))
    return max(terms) if terms else float("nan")


def bin_fiducials(
    mag: np.ndarray,
    color: np.ndarray,
    *,
    mag_err: np.ndarray | None = None,
    color_err: np.ndarray | None = None,
    color_mag_cov: np.ndarray | None = None,
    bins: np.ndarray,
    clip: float = 3.0,
    min_stars: int = 3,
) -> Fiducials:
    """Fiducial point per magnitude bin with σ and covariance.

    Within a bin the colour follows the sequence (slope × magnitude spread);
    that is removed with a linear fit of colour vs magnitude before the
    scatter is measured, and stars more than ``clip`` robust σ off that line
    are not used. Centre: IVW mean (plain mean without errors). σ of the
    colour: :func:`fiducial_sigma` of the IVW error and the detrended
    scatter. σ of the magnitude: the IVW error (the spread of magnitudes in a
    bin is the bin width, not an error); without errors the scatter term.
    Covariance: IVW combination of the per-star
    ``cov(colour, mag)`` (``−σ₂²`` for colour ``m₁ − m₂`` and magnitude
    ``m₂`` with independent filters), scaled with the σ inflation so the
    correlation coefficient is kept.
    """
    mag = np.asarray(mag, dtype=float)
    color = np.asarray(color, dtype=float)
    me = None if mag_err is None else np.broadcast_to(np.asarray(mag_err, dtype=float), mag.shape)
    ce = None if color_err is None else np.broadcast_to(np.asarray(color_err, dtype=float), mag.shape)
    cv = None if color_mag_cov is None else np.broadcast_to(np.asarray(color_mag_cov, dtype=float), mag.shape)
    digit = np.digitize(mag, bins)
    out = {k: [] for k in ("mag", "color", "mag_err", "color_err", "cov", "n")}
    for i in range(1, len(bins)):
        sel = np.flatnonzero((digit == i) & np.isfinite(mag) & np.isfinite(color))
        if sel.size < min_stars:
            continue
        resid = _detrended(mag[sel], color[sel])
        c_sig = _robust_sigma(resid)
        if np.isfinite(c_sig) and c_sig > 0:
            keep = np.abs(resid - np.median(resid)) <= clip * c_sig
            sel, resid = sel[keep], resid[keep]
        if sel.size < min_stars:
            continue
        n = int(sel.size)
        m_mean, m_ivw, w_m, ok_m = _weighted_mean(mag[sel], None if me is None else me[sel])
        c_mean, c_ivw, w_c, ok_c = _weighted_mean(color[sel], None if ce is None else ce[sel])
        m_err = m_ivw if np.isfinite(m_ivw) else fiducial_sigma(np.nan, _robust_sigma(mag[sel]), n)
        c_err = fiducial_sigma(c_ivw, _robust_sigma(_detrended(mag[sel], color[sel])), n)
        cov = 0.0
        if cv is not None and w_m is not None and w_c is not None:
            both = ok_m & ok_c
            wm = np.zeros(n)
            wc = np.zeros(n)
            wm[ok_m] = w_m
            wc[ok_c] = w_c
            cvi = np.nan_to_num(cv[sel])
            cov_ivw = float(np.sum((wm * wc * cvi)[both]) / (np.sum(wm) * np.sum(wc)))
            if np.isfinite(m_ivw) and np.isfinite(c_ivw) and m_ivw > 0 and c_ivw > 0:
                cov = cov_ivw * (m_err * c_err) / (m_ivw * c_ivw)
        out["mag"].append(m_mean)
        out["color"].append(c_mean)
        out["mag_err"].append(m_err)
        out["color_err"].append(c_err)
        out["cov"].append(cov)
        out["n"].append(n)
    return Fiducials(**{k: np.asarray(v, dtype=float if k != "n" else int) for k, v in out.items()})


def densify_isochrone(
    mag: np.ndarray,
    color: np.ndarray,
    *,
    step: float = 0.005,
    max_gap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation between consecutive isochrone points.

    Segments longer than ``max_gap`` mag (jumps between evolutionary
    branches) are not filled.
    """
    mag = np.asarray(mag, dtype=float)
    color = np.asarray(color, dtype=float)
    ok = np.isfinite(mag) & np.isfinite(color)
    mag, color = mag[ok], color[ok]
    if mag.size < 2:
        return mag, color
    out_m = [mag[:1]]
    out_c = [color[:1]]
    for k in range(1, mag.size):
        d = float(np.hypot(mag[k] - mag[k - 1], color[k] - color[k - 1]))
        n = int(np.ceil(d / step)) if d <= max_gap else 1
        t = np.arange(1, n + 1) / n
        out_m.append(mag[k - 1] + t * (mag[k] - mag[k - 1]))
        out_c.append(color[k - 1] + t * (color[k] - color[k - 1]))
    return np.concatenate(out_m), np.concatenate(out_c)


def _inverse_covariances(fid: Fiducials) -> np.ndarray:
    """``(n, 3)`` entries (a, b, d) of ``C⁻¹ = [[a, b], [b, d]]`` in (colour, mag)."""
    vc = fid.color_err**2
    vm = fid.mag_err**2
    cov = np.clip(fid.cov, -0.99 * np.sqrt(vc * vm), 0.99 * np.sqrt(vc * vm))
    det = vc * vm - cov**2
    return np.column_stack([vm / det, -cov / det, vc / det])


def isochrone_chi_square(
    fid: Fiducials,
    iso_mag: np.ndarray,
    iso_color: np.ndarray,
    *,
    inverse: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """``χ² = Σ min_k r^T C⁻¹ r`` and the per-fiducial residuals (colour, mag)."""
    inv = _inverse_covariances(fid) if inverse is None else inverse
    dc = fid.color[:, None] - np.asarray(iso_color, dtype=float)[None, :]
    dm = fid.mag[:, None] - np.asarray(iso_mag, dtype=float)[None, :]
    d2 = inv[:, 0:1] * dc**2 + 2.0 * inv[:, 1:2] * dc * dm + inv[:, 2:3] * dm**2
    k = np.argmin(d2, axis=1)
    rows = np.arange(len(fid))
    return float(np.sum(d2[rows, k])), dc[rows, k], dm[rows, k]


@dataclass
class ProfileResult:
    chi_square: float
    chi_square_data: float
    chi_square_mag: float
    chi_square_color: float
    delta_m: float
    delta_e: float


def profile_chi_square(
    fid: Fiducials,
    iso_mag: np.ndarray,
    iso_color: np.ndarray,
    *,
    a_per_e: float,
    c_per_e: float,
    sigma_m: float | None,
    sigma_e: float | None,
    data_scale: float = 1.0,
) -> ProfileResult:
    """χ² of an isochrone minimised over the nuisance offsets δM and δE.

    The isochrone is shifted by ``δE·c_per_e`` in colour and
    ``δM + δE·a_per_e`` in magnitude; the Gaussian priors add
    ``(δM/σ_M)² + (δE/σ_E)²``. A missing or zero σ fixes that offset at 0.
    ``data_scale`` divides the data χ² before the priors are added: when
    the fiducial errors are too small by √s (χ²_ν = s), the unscaled data
    would outweigh the priors by that factor and pull the offsets away.
    ``chi_square`` is the minimised objective, ``chi_square_data`` the
    unscaled data χ² at its minimum.
    """
    iso_m, iso_c = densify_isochrone(iso_mag, iso_color)
    if iso_m.size == 0 or len(fid) == 0:
        return ProfileResult(np.nan, np.nan, np.nan, np.nan, 0.0, 0.0)
    inv = _inverse_covariances(fid)
    free_m = sigma_m is not None and np.isfinite(sigma_m) and sigma_m > 0
    free_e = sigma_e is not None and np.isfinite(sigma_e) and sigma_e > 0

    def unpack(p):
        dm = p[0] if free_m else 0.0
        de = (p[1] if free_m else p[0]) if free_e else 0.0
        return dm, de

    def objective(p):
        dm, de = unpack(p)
        chi2, _, _ = isochrone_chi_square(fid, iso_m + dm + de * a_per_e, iso_c + de * c_per_e, inverse=inv)
        chi2 /= data_scale
        prior = (dm / sigma_m) ** 2 if free_m else 0.0
        prior += (de / sigma_e) ** 2 if free_e else 0.0
        return chi2 + prior

    n_free = int(free_m) + int(free_e)
    if n_free:
        start = np.zeros(n_free)
        steps = [s for s, f in ((sigma_m, free_m), (sigma_e, free_e)) if f]
        simplex = np.vstack([start] + [np.eye(n_free)[j] * steps[j] for j in range(n_free)])
        res = minimize(objective, start, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-4})
        dm, de = unpack(res.x)
        total = float(res.fun)
    else:
        dm, de = 0.0, 0.0
        total = objective(np.zeros(0))
    data, rc, rm = isochrone_chi_square(fid, iso_m + dm + de * a_per_e, iso_c + de * c_per_e, inverse=inv)
    return ProfileResult(
        chi_square=total,
        chi_square_data=float(data),
        chi_square_mag=float(np.sum((rm / fid.mag_err) ** 2)),
        chi_square_color=float(np.sum((rc / fid.color_err) ** 2)),
        delta_m=float(dm),
        delta_e=float(de),
    )


@dataclass
class AgeInterval:
    best_age: float
    age_err_lo: float
    age_err_hi: float
    chi_square: float
    dof: int
    chi_square_nu: float
    scale: float
    index: int


def age_interval(
    ages: np.ndarray,
    chi_square: np.ndarray,
    *,
    n_fiducials: int,
    n_free: int,
    scale: float | None = None,
) -> AgeInterval:
    """Best age and its 1σ interval from the profile ``χ²(age)``.

    Each fiducial slides along the isochrone to its nearest point, so only
    its offset across the sequence counts: ``ν = n_fiducials − 1 − n_free``.
    With ``χ²_min/ν > 1`` the profile is scaled by ``s = χ²_min/ν``; a given
    ``scale`` is used instead (for a profile already divided by it, pass 1).

    Near the grid minimum a parabola through it and its neighbours gives the
    best age (its vertex, so the result is not tied to the grid) and, when
    ``Δχ²/s = 1`` is reached within one grid step, the symmetric σ. Wider
    minima use the crossings of ``Δχ²/s = 1`` (linear interpolation). NaN
    where the crossing lies outside the grid (minimum at an edge).
    """
    ages = np.asarray(ages, dtype=float)
    chi2 = np.asarray(chi_square, dtype=float)
    ok = np.isfinite(ages) & np.isfinite(chi2)
    order = np.argsort(ages[ok])
    idx_map = np.flatnonzero(ok)[order]
    a, c = ages[ok][order], chi2[ok][order]
    if a.size == 0:
        return AgeInterval(np.nan, np.nan, np.nan, np.nan, 0, np.nan, 1.0, -1)
    k = int(np.argmin(c))
    dof = max(int(n_fiducials) - 1 - int(n_free), 1)
    chi_nu = float(c[k] / dof)
    s = max(1.0, chi_nu) if scale is None else float(scale)
    delta = (c - c[k]) / s
    best = float(a[k])

    def crossing(side: int) -> float:
        j = k
        while 0 <= j + side < a.size:
            j += side
            if delta[j] >= 1.0:
                t = (1.0 - delta[j - side]) / (delta[j] - delta[j - side])
                return float(a[j - side] + t * (a[j] - a[j - side]))
        return float("nan")

    lo_edge, hi_edge = crossing(-1), crossing(+1)
    if 0 < k < a.size - 1:
        curv, lin, _const = np.polyfit(a[k - 1:k + 2], delta[k - 1:k + 2], 2)
        if curv > 0:
            best = float(np.clip(-lin / (2.0 * curv), a[k - 1], a[k + 1]))
            sigma = float(np.sqrt(1.0 / curv))
            if np.isfinite(lo_edge) and lo_edge >= a[k - 1]:
                lo_edge = best - sigma
            if np.isfinite(hi_edge) and hi_edge <= a[k + 1]:
                hi_edge = best + sigma
    lo = best - lo_edge if np.isfinite(lo_edge) else float("nan")
    hi = hi_edge - best if np.isfinite(hi_edge) else float("nan")
    return AgeInterval(best, max(lo, 0.0) if np.isfinite(lo) else lo,
                       max(hi, 0.0) if np.isfinite(hi) else hi,
                       float(c[k]), dof, chi_nu, s, int(idx_map[k]))


__all__ = [
    "AgeInterval",
    "Fiducials",
    "ProfileResult",
    "age_interval",
    "bin_fiducials",
    "densify_isochrone",
    "fiducial_sigma",
    "isochrone_chi_square",
    "profile_chi_square",
]
