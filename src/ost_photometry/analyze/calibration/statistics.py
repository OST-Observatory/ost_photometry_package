"""Zero-point / colour-term fits with honest errors.

Model for a calibrator ``i`` of one filter::

    Δ_i = m_std,i - m_inst,i = ZP + T · C_i + ε_i,
    Var(ε_i) = σ_inst,i² + σ_cat,i² + T²·σ_C,i² + σ_int²

* ``σ_inst`` includes photon, sky and scintillation noise;
* ``σ_cat`` is the catalog error (``err_std_<f>``; 0.02 mag where the
  catalog gives none or zero, the APASS convention);
* ``σ_C`` is the catalog colour error (it scatters the points along x);
* ``σ_int`` is the intrinsic scatter (catalog systematics, colour-term
  mismatch, crowding), solved so that χ²_ν = 1. With it the weights are
  absolute: the parameter errors come from ``(AᵀWA)⁻¹`` without rescaling.
* When the errors over-explain the scatter (χ²_ν < 1 with σ_int = 0) and at
  least :data:`MIN_STARS_CATALOG_SCALE` calibrators are used, the catalog
  errors are scaled by ``s < 1`` until χ²_ν = 1 (``catalog_error_scale``).
  The instrumental errors are validated on their own; catalog errors are
  not (APASS B errors are about twice the scatter on the OST test fields).
* With σ-clipping the target is not χ²_ν = 1 but the variance of a normal
  truncated at the clip level (:func:`clipped_variance`, 0.91 at 2.5σ):
  the kept residuals scatter less than their errors.

The colour term is fitted around the weighted mean colour ``C_ref`` (ZP and T
are then nearly uncorrelated) and reported at colour 0 with its full
covariance. When ``σ_int`` had to be estimated from the scatter, the errors
are multiplied by the Student-t quantile of 1σ for ``n - p`` degrees of
freedom (matters for few calibrators). Outliers are clipped on
residuals normalised by their own error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

#: Catalog error per band where the catalog gives none (APASS convention).
CATALOG_ERR_DEFAULT = 0.02
#: Below this many calibrators the result is flagged ``few_calibrators``.
FEW_CALIBRATORS = 5
#: Catalog errors are only scaled down (χ²_ν < 1) with at least this many
#: calibrators; with fewer a low χ² is too likely by chance.
MIN_STARS_CATALOG_SCALE = 10
_ONE_SIGMA = stats.norm.cdf(1.0)


def clipped_variance(k: float | None) -> float:
    """Variance of a unit normal truncated at ±k (1 without clipping).

    After clipping at kσ the kept residuals scatter less than σ; σ_int and
    the catalog-error scale aim at this χ²_ν instead of 1 (k = 2.5: 0.91).
    """
    if k is None or not np.isfinite(k) or k <= 0:
        return 1.0
    return float(1.0 - 2.0 * k * stats.norm.pdf(k) / (2.0 * stats.norm.cdf(k) - 1.0))


@dataclass
class ZeroPointFit:
    zero_point: float = float("nan")
    zero_point_err: float = float("nan")
    color_term: float = 0.0
    color_term_err: float = 0.0
    cov_tz: float = 0.0
    color_ref: float = 0.0
    sigma_int: float = 0.0
    chi2_nu: float = float("nan")
    rms: float = float("nan")
    n_used: int = 0
    used: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    quality_flag: str = "OK"
    #: Covariance of (ZP at colour 0, T) split into the photometric part
    #: (photon, sky, scintillation: independent from epoch to epoch) and the
    #: catalog part (catalog errors and σ_int of the *same* calibrators:
    #: common to every epoch that uses them). Their sum is the full one.
    cov_phot: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan))
    cov_cat: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan))
    #: Factor applied to the catalog errors (1: as given; < 1: they
    #: over-explained the scatter).
    catalog_error_scale: float = 1.0


def catalog_errors(errors: np.ndarray | None, n: int) -> np.ndarray:
    """Catalog σ per star; 0.02 mag where missing, non-finite or zero."""
    if errors is None:
        return np.full(n, CATALOG_ERR_DEFAULT)
    e = np.abs(np.asarray(errors, dtype=float))
    return np.where(np.isfinite(e) & (e > 0), e, CATALOG_ERR_DEFAULT)


def small_sample_factor(dof: int) -> float:
    """Student-t 1σ quantile / normal 1σ quantile for ``dof`` degrees of freedom."""
    if dof <= 0:
        return float("nan")
    return float(stats.t.ppf(_ONE_SIGMA, dof))


def _sandwich(color, var, var_part, c_ref, *, fit_color):
    """Covariance of (zp_ref, T) caused by the variance part ``var_part``."""
    w = 1.0 / var
    if fit_color:
        a = np.column_stack([np.ones_like(color), color - c_ref])
        bread = np.linalg.inv(a.T @ (a * w[:, None]))
        meat = (a * (w**2 * var_part)[:, None]).T @ a
        return bread @ meat @ bread
    sw = float(np.sum(w))
    return np.array([[float(np.sum(w**2 * var_part)) / sw**2, 0.0], [0.0, 0.0]])


def _to_color_zero(cov: np.ndarray, c_ref: float) -> np.ndarray:
    """(zp_ref, T) covariance -> (ZP at colour 0, T), ZP0 = zp_ref - T·c_ref."""
    jac = np.array([[1.0, -c_ref], [0.0, 1.0]])
    return jac @ cov @ jac.T


def _weighted_fit(delta, color, var, *, fit_color, fixed_t):
    """One weighted solve; returns (zp_ref, t, cov(zp_ref, t) 2x2, c_ref)."""
    w = 1.0 / var
    c_ref = float(np.sum(w * color) / np.sum(w)) if fit_color else 0.0
    if fit_color:
        x = color - c_ref
        a = np.column_stack([np.ones_like(x), x])
        aw = a * w[:, None]
        cov = np.linalg.inv(a.T @ aw)
        zp_ref, t = cov @ (aw.T @ delta)
        return float(zp_ref), float(t), cov, c_ref
    resid = delta - fixed_t * color
    zp = float(np.sum(w * resid) / np.sum(w))
    cov = np.array([[1.0 / np.sum(w), 0.0], [0.0, 0.0]])
    return zp, float(fixed_t), cov, 0.0


def _chi2(delta, color, var, zp_ref, t, c_ref):
    model = zp_ref + t * (color - c_ref)
    return float(np.sum((delta - model) ** 2 / var))


def fit_zero_point(
    delta: np.ndarray,
    color: np.ndarray,
    sigma_inst: np.ndarray,
    sigma_cat: np.ndarray,
    *,
    fit_color: bool,
    sigma_color: np.ndarray | None = None,
    fixed_color_term: float = 0.0,
    fixed_color_term_err: float = 0.0,
    sigma_clip: float | None = 2.5,
    max_iterations: int = 5,
    min_stars: int = 3,
) -> ZeroPointFit:
    """Fit ``Δ = ZP + T·C`` (or ZP with a fixed T) with intrinsic scatter.

    Returns NaN parameters with fewer than ``min_stars`` usable stars.
    ``fixed_color_term_err`` enters ``color_term_err`` (the colour term is
    not fitted, its error stays a systematic of every star's magnitude).
    """
    delta = np.asarray(delta, dtype=float)
    color = np.asarray(color, dtype=float)
    inst2 = np.asarray(sigma_inst, dtype=float) ** 2
    cat2_raw = np.asarray(sigma_cat, dtype=float) ** 2
    sig_c2_raw = (
        np.zeros_like(delta) if sigma_color is None else np.asarray(sigma_color, dtype=float) ** 2
    )
    base = inst2 + cat2_raw
    valid = np.isfinite(delta) & np.isfinite(base) & (base > 0)
    if fit_color or fixed_color_term != 0.0:
        valid &= np.isfinite(color)
    color = np.where(np.isfinite(color), color, 0.0)
    p = 2 if fit_color else 1
    out = ZeroPointFit(used=np.zeros(delta.size, dtype=bool))
    use = valid.copy()
    t_now = fixed_color_term
    sigma_int = 0.0
    target = clipped_variance(sigma_clip)
    for _ in range(max(1, max_iterations)):
        n = int(use.sum())
        if n < max(min_stars, p + 1):
            return out
        d, c = delta[use], color[use]
        base, sig_c2, cat_scale = inst2 + cat2_raw, sig_c2_raw, 1.0
        var0 = base[use] + t_now**2 * sig_c2[use]
        zp_ref, t_now, cov, c_ref = _weighted_fit(
            d, c, var0, fit_color=fit_color, fixed_t=fixed_color_term
        )
        chi2_0 = _chi2(d, c, var0, zp_ref, t_now, c_ref) / (n - p)
        # intrinsic scatter: σ_int with χ²_ν = 1 (bisection on the variance)
        sigma_int = 0.0
        if chi2_0 < target and n >= MIN_STARS_CATALOG_SCALE:
            # catalog errors over-explain the scatter: scale them by s < 1
            lo, hi = 0.0, 1.0
            for _bisect in range(50):
                mid = 0.5 * (lo + hi)
                var = inst2[use] + mid**2 * (cat2_raw[use] + t_now**2 * sig_c2_raw[use])
                zp_ref, t_now, cov, c_ref = _weighted_fit(
                    d, c, var, fit_color=fit_color, fixed_t=fixed_color_term
                )
                if _chi2(d, c, var, zp_ref, t_now, c_ref) / (n - p) > target:
                    lo = mid
                else:
                    hi = mid
            cat_scale = hi
            base = inst2 + cat_scale**2 * cat2_raw
            sig_c2 = cat_scale**2 * sig_c2_raw
            var0 = base[use] + t_now**2 * sig_c2[use]
        elif chi2_0 > target:
            lo, hi = 0.0, max(1.0, float(np.var(d)))
            for _bisect in range(60):
                mid = 0.5 * (lo + hi)
                var = var0 + mid
                zp_ref, t_now, cov, c_ref = _weighted_fit(
                    d, c, var, fit_color=fit_color, fixed_t=fixed_color_term
                )
                if _chi2(d, c, var, zp_ref, t_now, c_ref) / (n - p) > target:
                    lo = mid
                else:
                    hi = mid
            sigma_int = math.sqrt(hi)
        var_all = base + t_now**2 * sig_c2 + sigma_int**2
        resid = delta - (zp_ref + t_now * (color - c_ref))
        if sigma_clip is None:
            break
        new_use = valid & (np.abs(resid) <= float(sigma_clip) * np.sqrt(var_all))
        if np.array_equal(new_use, use) or int(new_use.sum()) < max(min_stars, p + 1):
            break
        use = new_use

    n = int(use.sum())
    var = (base + t_now**2 * sig_c2 + sigma_int**2)[use]
    zp_ref, t_now, cov, c_ref = _weighted_fit(
        delta[use], color[use], var, fit_color=fit_color, fixed_t=fixed_color_term
    )
    var_phot = inst2[use]
    cov_phot = _sandwich(color[use], var, var_phot, c_ref, fit_color=fit_color)
    cov_cat = _sandwich(color[use], var, var - var_phot, c_ref, fit_color=fit_color)
    # Student-t only where the error scale was estimated from the scatter
    # (σ_int > 0); with σ_int = 0 the variances are known and Gaussian.
    if sigma_int > 0:
        factor2 = small_sample_factor(n - p) ** 2
        cov, cov_phot, cov_cat = cov * factor2, cov_phot * factor2, cov_cat * factor2
    if fit_color:
        # back to colour 0: ZP0 = zp_ref - T·C_ref
        zp0 = zp_ref - t_now * c_ref
        var_zp0 = cov[0, 0] + c_ref**2 * cov[1, 1] - 2.0 * c_ref * cov[0, 1]
        cov_tz = cov[0, 1] - c_ref * cov[1, 1]
        t_err = math.sqrt(max(cov[1, 1], 0.0))
    else:
        zp0 = zp_ref
        var_zp0 = cov[0, 0]
        cov_tz = 0.0
        t_err = float(fixed_color_term_err)
    resid = (delta - (zp0 + t_now * color))[use]
    out.zero_point = float(zp0)
    out.zero_point_err = float(math.sqrt(max(var_zp0, 0.0)))
    out.color_term = float(t_now)
    out.color_term_err = float(t_err)
    out.cov_tz = float(cov_tz)
    out.color_ref = float(c_ref)
    out.sigma_int = float(sigma_int)
    out.chi2_nu = float(chi2_0)
    out.rms = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")
    out.n_used = n
    out.used = use
    out.catalog_error_scale = float(cat_scale)
    cov_phot0, cov_cat0 = _to_color_zero(cov_phot, c_ref), _to_color_zero(cov_cat, c_ref)
    if not fit_color:
        cov_phot0[1, 1] = 0.0
        cov_cat0[1, 1] = float(fixed_color_term_err) ** 2
    out.cov_phot = cov_phot0
    out.cov_cat = cov_cat0
    flags = []
    if n < FEW_CALIBRATORS:
        flags.append("few_calibrators")
    if not fit_color and fixed_color_term != 0.0:
        flags.append("fixed_color_term")
    out.quality_flag = ",".join(flags) if flags else "OK"
    return out


def combine_epoch_fits(
    zero_points: np.ndarray,
    color_terms: np.ndarray,
    cov_phot: np.ndarray,
    cov_cat: np.ndarray,
    color_ref: float = 0.0,
) -> dict[str, float]:
    """Combine per-epoch (ZP, T) of the same calibrators into one night value.

    ``cov_phot`` / ``cov_cat`` have shape ``(n, 2, 2)`` for ``(ZP at colour
    0, T)``. Weights come from the full covariances. The photometric parts
    are independent between epochs and average down; the catalog parts
    belong to the same stars in every epoch and do not (their mean is added
    unchanged). The Birge ratio ``max(1, χ²/ν)`` of the epochs about the
    combined value, measured against the photometric covariances, scales the
    photometric part. ``zero_point_scatter`` is the robust scatter of the
    epoch zero points about the combined one at ``color_ref`` beyond what
    their photometric errors explain: what a single epoch's true zero point
    differs from the night value (transparency).
    """
    zp = np.asarray(zero_points, dtype=float)
    t = np.asarray(color_terms, dtype=float)
    cp = np.asarray(cov_phot, dtype=float)
    cc = np.asarray(cov_cat, dtype=float)
    ct = cp + cc
    ok = (
        np.isfinite(zp) & np.isfinite(t)
        & np.all(np.isfinite(ct), axis=(1, 2)) & (ct[:, 0, 0] > 0)
    )
    zp, t, cp, cc, ct = zp[ok], t[ok], cp[ok], cc[ok], ct[ok]
    n = zp.size
    nan = float("nan")
    if n == 0:
        return {"zero_point": nan, "zero_point_err": nan, "color_term": nan,
                "color_term_err": nan, "cov_tz": nan, "birge": nan,
                "zero_point_scatter": nan, "n_epochs": 0,
                "cov_phot": np.full((2, 2), nan), "cov_cat": np.full((2, 2), nan)}
    jac = np.array([[1.0, color_ref], [0.0, 1.0]])
    x = np.column_stack([zp + t * color_ref, t])

    def to_ref(c):
        return np.einsum("ij,njk,lk->nil", jac, c, jac)

    cp_r, cc_r, ct_r = to_ref(cp), to_ref(cc), to_ref(ct)
    fixed_t = bool(np.all(ct_r[:, 1, 1] <= 0) or (np.all(cp_r[:, 1, 1] <= 0) and np.all(t == t[0])))
    k = 1 if fixed_t else 2
    sub = (slice(0, k), slice(0, k))
    weights = np.linalg.inv(ct_r[:, :k, :k])
    info_inv = np.linalg.inv(weights.sum(axis=0))
    mean = info_inv @ np.einsum("nij,nj->i", weights, x[:, :k])
    phot = info_inv @ np.einsum("nij,njk,nkl->il", weights, cp_r[:, :k, :k], weights) @ info_inv
    resid = x[:, :k] - mean
    phot_inv = np.linalg.pinv(cp_r[:, :k, :k])
    chi2 = float(np.einsum("ni,nij,nj->", resid, phot_inv, resid))
    dof = k * (n - 1)
    birge = max(1.0, chi2 / dof) if dof > 0 else 1.0
    cat = cc_r[:, :k, :k].mean(axis=0)
    out_phot = np.zeros((2, 2))
    out_cat = np.zeros((2, 2))
    out_phot[sub] = phot * birge
    out_cat[sub] = cat
    if fixed_t:
        out_mean = np.array([mean[0], t[0]])
        out_cat[1, 1] = float(cc_r[:, 1, 1].mean())
    else:
        out_mean = mean
    total = out_phot + out_cat
    back = np.array([[1.0, -color_ref], [0.0, 1.0]])
    total0 = back @ total @ back.T
    zp0 = float(out_mean[0] - out_mean[1] * color_ref)
    dev = x[:, 0] - out_mean[0]
    if n >= 3:
        spread = float(1.4826 * np.median(np.abs(dev - np.median(dev))))
        phot_typ = float(np.sqrt(np.median(cp_r[:, 0, 0])))
        scatter = float(np.sqrt(max(spread**2 - phot_typ**2, 0.0)))
    else:
        scatter = nan
    return {
        "zero_point": zp0,
        "zero_point_err": float(np.sqrt(max(total0[0, 0], 0.0))),
        "color_term": float(out_mean[1]),
        "color_term_err": float(np.sqrt(max(total0[1, 1], 0.0))),
        "cov_tz": float(total0[0, 1]),
        "birge": float(birge),
        "zero_point_scatter": scatter,
        "n_epochs": int(n),
        "cov_phot": back @ out_phot @ back.T,
        "cov_cat": back @ out_cat @ back.T,
    }


def average_epochs_per_star(
    epochs: list,
    filters: list[str],
    *,
    mag_prefix: str = "mag_",
    err_prefix: str = "err_",
):
    """One row per star: inverse-variance mean of its magnitudes over epochs.

    For ensemble calibrations: every star enters the fit once, so the
    catalog error of a star (the same in every epoch) is counted once, not
    once per epoch. The mean's error is ``1/√Σw``; other columns are taken
    from the star's first row. The epochs must already be extinction
    corrected; airmass columns are dropped.
    """
    from astropy.table import Table, vstack

    stacked = vstack(list(epochs), metadata_conflicts="silent")
    ids = np.asarray(stacked["id"]).astype(np.int64)
    uniq, first = np.unique(ids, return_index=True)
    out = stacked[np.sort(first)]
    order_ids = np.asarray(out["id"]).astype(np.int64)
    for f in filters:
        mc, ec = f"{mag_prefix}{f}", f"{err_prefix}{f}"
        if mc not in stacked.colnames or ec not in stacked.colnames:
            continue
        m = np.asarray(stacked[mc], dtype=float)
        e = np.abs(np.asarray(stacked[ec], dtype=float))
        ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
        w = np.where(ok, 1.0 / np.where(ok, e, 1.0) ** 2, 0.0)
        idx = np.searchsorted(uniq, ids)
        sw = np.bincount(idx, weights=w, minlength=uniq.size)
        swm = np.bincount(idx, weights=w * np.where(ok, m, 0.0), minlength=uniq.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(sw > 0, swm / sw, np.nan)
            err = np.where(sw > 0, 1.0 / np.sqrt(sw), np.nan)
        pos = np.searchsorted(uniq, order_ids)
        out[mc] = mean[pos]
        out[ec] = err[pos]
    drop = [c for c in out.colnames if c == "airmass" or c.startswith("airmass_")]
    if drop:
        out.remove_columns(drop)
    return Table(out)


def parse_fixed_color_terms(spec: dict | None) -> dict[str, tuple[float, float]]:
    """``{filter: (T, σ_T)}`` from ``transformation_coefficients_dict``.

    Accepts ``{"V": {"color_term": T, "color_term_err": σ}}`` and the legacy
    OST campaign format of ``calibration_parameters.Tcs``
    (``"Filter 1"``/``"T_1"``/``"T_1_err"``, ``"Filter 2"``/``"T_2"``/...):
    the colour term of filter f is ``T_1`` if f is filter 1, else ``T_2``.
    """
    out: dict[str, tuple[float, float]] = {}
    for filter_, entry in (spec or {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"transformation_coefficients_dict[{filter_!r}] must be a dict")
        if "color_term" in entry:
            out[str(filter_)] = (float(entry["color_term"]), float(entry.get("color_term_err", 0.0)))
            continue
        if "T_1" in entry and "T_2" in entry:
            idx = "1" if str(entry.get("Filter 1")) == str(filter_) else "2"
            out[str(filter_)] = (float(entry[f"T_{idx}"]), float(entry.get(f"T_{idx}_err", 0.0)))
            continue
        raise ValueError(
            f"transformation_coefficients_dict[{filter_!r}]: need 'color_term' "
            "(and 'color_term_err') or the Tcs keys 'T_1'/'T_2'"
        )
    return out


def mean_airmass(epochs, filter_: str) -> float | None:
    """Mean airmass of ``filter_`` over epoch tables (``airmass_<f>`` or ``airmass``)."""
    values = []
    for tbl in epochs:
        for col in (f"airmass_{filter_}", "airmass"):
            if col in tbl.colnames:
                x = np.asarray(tbl[col], dtype=float)
                x = x[np.isfinite(x)]
                if x.size:
                    values.append(float(np.median(x)))
                break
    return float(np.mean(values)) if values else None


def set_airmass_reference(result, epochs, filters: list[str]) -> None:
    """Store the epochs' mean airmass on a zero point they share."""
    for f in filters:
        tc = result.transformation.get(f)
        if tc is not None:
            tc.airmass_ref = mean_airmass(epochs, f)


__all__ = [
    "mean_airmass",
    "set_airmass_reference",
    "average_epochs_per_star",
    "parse_fixed_color_terms",
    "CATALOG_ERR_DEFAULT",
    "FEW_CALIBRATORS",
    "ZeroPointFit",
    "catalog_errors",
    "combine_epoch_fits",
    "fit_zero_point",
    "small_sample_factor",
]
