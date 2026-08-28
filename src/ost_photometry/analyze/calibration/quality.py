"""Pre-fit quality cuts for calibration-star candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table

from .zp import comparison_mask_from_std_columns


def _as_float(table: Table, name: str) -> np.ndarray | None:
    if name not in table.colnames:
        return None
    col = table[name]
    return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)


def _binned_p84(mag: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    mag, err = mag[ok], err[ok]
    if mag.size < 16:
        return np.array([]), np.array([])
    order = np.argsort(mag)
    mag_s, err_s = mag[order], err[order]
    n_bins = int(min(12, mag.size // 8))
    n_bins = max(n_bins, 2)
    edges = np.linspace(0, mag.size, n_bins + 1, dtype=int)
    centers: list[float] = []
    p84: list[float] = []
    for i0, i1 in zip(edges[:-1], edges[1:], strict=True):
        if i1 - i0 < 8:
            continue
        centers.append(float(np.median(mag_s[i0:i1])))
        p84.append(float(np.percentile(err_s[i0:i1], 84.0)))
    return np.asarray(centers), np.asarray(p84)


def _interp_ridge(mag: np.ndarray, centers: np.ndarray, p84: np.ndarray) -> np.ndarray:
    """Piecewise-linear p84(mag), with slope continuation past the end bins."""
    mag = np.asarray(mag, dtype=float)
    ridge = np.interp(mag, centers, p84)
    if centers.size < 2:
        return ridge
    ridge = np.array(ridge, dtype=float, copy=True)
    d_lo = centers[1] - centers[0]
    d_hi = centers[-1] - centers[-2]
    if d_lo != 0.0:
        slope_lo = (p84[1] - p84[0]) / d_lo
        lo = mag < centers[0]
        ridge[lo] = p84[0] + slope_lo * (mag[lo] - centers[0])
    if d_hi != 0.0:
        slope_hi = (p84[-1] - p84[-2]) / d_hi
        hi = mag > centers[-1]
        ridge[hi] = p84[-1] + slope_hi * (mag[hi] - centers[-1])
    return np.maximum(ridge, 0.0)


def _photon_envelope(mag: np.ndarray, err: np.ndarray) -> tuple[float, float] | None:
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if np.count_nonzero(ok) < 12:
        return None
    centers, _p84 = _binned_p84(mag[ok], err[ok])
    if centers.size < 3:
        return None
    # Re-bin median σ for the envelope fit.
    mag_ok, err_ok = mag[ok], err[ok]
    order = np.argsort(mag_ok)
    mag_s, err_s = mag_ok[order], err_ok[order]
    n_bins = min(12, mag_s.size // 8)
    n_bins = max(n_bins, 3)
    edges = np.linspace(0, mag_s.size, n_bins + 1, dtype=int)
    m_fit: list[float] = []
    s_fit: list[float] = []
    for i0, i1 in zip(edges[:-1], edges[1:], strict=True):
        if i1 - i0 < 8:
            continue
        m_fit.append(float(np.median(mag_s[i0:i1])))
        s_fit.append(float(np.median(err_s[i0:i1])))
    m_fit_a = np.asarray(m_fit)
    s_fit_a = np.asarray(s_fit)
    if m_fit_a.size < 3:
        return None

    def _model(m, floor, faint_scale):
        return np.sqrt(floor**2 + (faint_scale * 10.0 ** (0.4 * m)) ** 2)

    p0 = (
        float(np.nanmin(s_fit_a)),
        float(np.median(s_fit_a) / max(np.median(10.0 ** (0.4 * m_fit_a)), 1e-12)),
    )
    try:
        from scipy.optimize import curve_fit

        popt, _cov = curve_fit(_model, m_fit_a, s_fit_a, p0=p0, maxfev=2000)
    except (ImportError, RuntimeError, ValueError, TypeError):
        return None
    floor, faint_scale = float(popt[0]), float(popt[1])
    if not np.isfinite(floor) or not np.isfinite(faint_scale) or floor <= 0:
        return None
    return floor, max(faint_scale, 0.0)


def _in_range(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    finite = np.isfinite(values)
    return (~finite) | ((values >= lo) & (values <= hi))


def _bright_percentile_cap(
    mag: np.ndarray,
    quality: np.ndarray,
    percentile: float,
    bright_frac: float,
) -> float | None:
    ok = np.isfinite(mag) & np.isfinite(quality)
    if np.count_nonzero(ok) < 8:
        return None
    mag_ok, q_ok = mag[ok], quality[ok]
    n_bright = max(5, int(np.ceil(bright_frac * mag_ok.size)))
    bright = np.argsort(mag_ok)[:n_bright]
    return float(np.percentile(q_ok[bright], percentile))


def _shape_ok(
    table: Table,
    *,
    sharpness_range: tuple[float, float],
    roundness_range: tuple[float, float],
) -> np.ndarray:
    n = len(table)
    ok = np.ones(n, dtype=bool)
    sharp = _as_float(table, "sharpness")
    if sharp is not None:
        lo, hi = sharpness_range
        ok &= _in_range(sharp, float(lo), float(hi))
    lo_r, hi_r = roundness_range
    for name in ("roundness", "roundness1", "roundness2"):
        rnd = _as_float(table, name)
        if rnd is not None:
            ok &= _in_range(rnd, float(lo_r), float(hi_r))
    return ok


def _psf_quality_ok(
    table: Table,
    filters: list[str],
    *,
    qfit_max: float | None,
    cfit_max: float | None,
    qfit_bright_percentile: float | None,
    cfit_bright_percentile: float | None,
    bright_frac: float,
) -> np.ndarray:
    n = len(table)
    ok = np.ones(n, dtype=bool)
    mag = None
    for filt in filters:
        mag = _as_float(table, f"mag_{filt}")
        if mag is not None:
            break
    for col, cap, pct in (
        ("qfit", qfit_max, qfit_bright_percentile),
        ("cfit", cfit_max, cfit_bright_percentile),
    ):
        values = _as_float(table, col)
        if values is None:
            continue
        limit = cap
        if mag is not None and pct is not None:
            bright_cap = _bright_percentile_cap(mag, values, float(pct), bright_frac)
            if bright_cap is not None:
                limit = bright_cap if limit is None else min(float(limit), bright_cap)
        if limit is None:
            continue
        finite = np.isfinite(values)
        ok &= (~finite) | (values <= float(limit))
    return ok


def _error_ridge_ok(
    table: Table,
    filters: list[str],
    *,
    p84_clip: bool,
    photon_factor: float | None,
) -> np.ndarray:
    n = len(table)
    ok = np.ones(n, dtype=bool)
    for filt in filters:
        mag = _as_float(table, f"mag_{filt}")
        err = _as_float(table, f"err_{filt}")
        if mag is None or err is None:
            continue
        err_abs = np.abs(err)
        if p84_clip:
            centers, p84 = _binned_p84(mag, err_abs)
            if centers.size >= 2:
                ridge = _interp_ridge(mag, centers, p84)
                finite = np.isfinite(mag) & np.isfinite(err_abs)
                ok &= (~finite) | (err_abs <= ridge)
        if photon_factor is not None and photon_factor > 0:
            params = _photon_envelope(mag, err_abs)
            if params is not None:
                floor, faint_scale = params
                model = np.sqrt(floor**2 + (faint_scale * 10.0 ** (0.4 * mag)) ** 2)
                finite = np.isfinite(mag) & np.isfinite(err_abs) & np.isfinite(model)
                ok &= (~finite) | (err_abs <= float(photon_factor) * model)
    return ok


def calibrator_quality_kwargs(config: Any) -> dict[str, Any]:
    """Map ``PipelineConfig`` fields onto :func:`calibrator_candidate_mask` kwargs."""
    sharpness = getattr(config, "finder_sharpness_range", (0.2, 1.0))
    roundness = getattr(config, "finder_roundness_range", (-1.0, 1.0))
    return {
        "error_p84_clip": bool(getattr(config, "calibrator_error_p84_clip", True)),
        "photon_factor": getattr(config, "calibrator_photon_factor", 2.0),
        "qfit_max": getattr(config, "calibrator_qfit_max", 0.2),
        "cfit_max": getattr(config, "calibrator_cfit_max", 0.2),
        "qfit_bright_percentile": getattr(config, "calibrator_qfit_bright_percentile", 90.0),
        "cfit_bright_percentile": getattr(config, "calibrator_cfit_bright_percentile", 90.0),
        "bright_frac": float(getattr(config, "calibrator_bright_frac", 0.4)),
        "sharpness_range": tuple(sharpness),
        "roundness_range": tuple(roundness),
        "apply_finder_shape_cuts": bool(
            getattr(config, "calibrator_apply_finder_shape_cuts", True)
        ),
        "min_keep": int(getattr(config, "calibrator_min_keep", 3)),
    }


def calibrator_candidate_mask(
    table: Table,
    filters: list[str],
    config: Any | None = None,
    *,
    min_keep: int | None = None,
    error_p84_clip: bool = True,
    photon_factor: float | None = 2.0,
    qfit_max: float | None = 0.2,
    cfit_max: float | None = 0.2,
    qfit_bright_percentile: float | None = 90.0,
    cfit_bright_percentile: float | None = 90.0,
    bright_frac: float = 0.4,
    sharpness_range: tuple[float, float] = (0.2, 1.0),
    roundness_range: tuple[float, float] = (-1.0, 1.0),
    apply_finder_shape_cuts: bool = True,
) -> np.ndarray:
    """Catalog matches that also pass photometric-error and PSF/finder quality cuts.

    Cuts are applied in order (shape, qfit/cfit, error ridge). A cut is skipped
    if it would leave fewer than ``min_keep`` candidates.
    """
    if config is not None:
        kw = calibrator_quality_kwargs(config)
        if min_keep is not None:
            kw["min_keep"] = min_keep
        return calibrator_candidate_mask(table, filters, None, **kw)

    keep = 3 if min_keep is None else int(min_keep)
    mask = comparison_mask_from_std_columns(table, filters)
    if np.count_nonzero(mask) < keep:
        return mask

    def _try(next_mask: np.ndarray) -> None:
        nonlocal mask
        if np.count_nonzero(next_mask) >= keep:
            mask = next_mask

    if apply_finder_shape_cuts:
        _try(
            mask
            & _shape_ok(
                table,
                sharpness_range=sharpness_range,
                roundness_range=roundness_range,
            )
        )
    _try(
        mask
        & _psf_quality_ok(
            table,
            filters,
            qfit_max=qfit_max,
            cfit_max=cfit_max,
            qfit_bright_percentile=qfit_bright_percentile,
            cfit_bright_percentile=cfit_bright_percentile,
            bright_frac=bright_frac,
        )
    )
    _try(
        mask
        & _error_ridge_ok(
            table,
            filters,
            p84_clip=error_p84_clip,
            photon_factor=photon_factor,
        )
    )
    return mask


__all__ = [
    "calibrator_candidate_mask",
    "calibrator_quality_kwargs",
]
