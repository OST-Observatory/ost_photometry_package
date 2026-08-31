"""Reddening and optional R_V / E(B-V) error terms for absolute CMDs."""

from __future__ import annotations

import numpy as np

_RV_FINITE_DIFFERENCE = 1e-4


def _optional_sigma(value: float | None) -> float:
    """Return a non-negative finite sigma, or 0 if the value is unused."""
    if value is None:
        return 0.0
    try:
        sigma = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    return sigma


def _rss(*terms: float | np.ndarray | None) -> float | np.ndarray | None:
    """Combine non-zero independent terms in quadrature; ``None`` if empty."""
    accumulated: float | np.ndarray | None = None
    for term in terms:
        if term is None:
            continue
        values = np.asarray(term, dtype=float)
        if np.all(values == 0.0):
            continue
        if accumulated is None:
            accumulated = values
        else:
            accumulated = np.sqrt(np.square(accumulated) + np.square(values))
    return accumulated


def combine_cmd_error_bars(
        photometric_err: np.ndarray | float | None,
        reddening_err: float | None,
    ) -> np.ndarray | float | None:
    """Quadrature of per-star photometry and a common reddening sigma."""
    return _rss(photometric_err, _optional_sigma(reddening_err) or None)


def _fitzpatrick_k(rv: float, inverse_um: float) -> float:
    from ost_photometry.calibration_parameters import fitzpatrick_extinction_curve

    return float(fitzpatrick_extinction_curve(rv)(inverse_um))


def _fitzpatrick_dk_drv(rv: float, inverse_um: float) -> float:
    delta = _RV_FINITE_DIFFERENCE
    rv_lo = max(rv - delta, delta)
    rv_hi = rv + delta
    span = rv_hi - rv_lo
    if span <= 0.0:
        return 0.0
    return (_fitzpatrick_k(rv_hi, inverse_um) - _fitzpatrick_k(rv_lo, inverse_um)) / span


def reddening_for_absolute_cmd(
        filter_1: str, filter_2: str, rv: float, e_b_v: float,
        *, e_b_v_err: float | None = None, rv_err: float | None = None,
    ) -> tuple[float, float, float, float]:
    """
    Absolute extinction in ``filter_2`` and colour excess, with optional
    independent 1-sigma uncertainties on E(B-V) and R_V.

    Returns
    -------
    a_filter_2, relative_extinction, a_filter_2_err, relative_extinction_err
    """
    sigma_ebv = _optional_sigma(e_b_v_err)
    sigma_rv = _optional_sigma(rv_err)

    if filter_1 == "B" and filter_2 == "V":
        a_filter_2 = rv * e_b_v
        relative_extinction = e_b_v
        d_a_de = rv
        d_a_drv = e_b_v
        d_rel_de = 1.0
        d_rel_drv = 0.0
    else:
        from ost_photometry.calibration_parameters import (
            filter_effective_wavelength,
            fitzpatrick_extinction_curve,
        )

        inverse_um_1 = 10000.0 / filter_effective_wavelength[filter_1]
        inverse_um_2 = 10000.0 / filter_effective_wavelength[filter_2]
        extinction_curve = fitzpatrick_extinction_curve(rv)
        k1 = float(extinction_curve(inverse_um_1))
        k2 = float(extinction_curve(inverse_um_2))
        a_filter_2 = k2 * e_b_v
        relative_extinction = (k1 - k2) * e_b_v
        d_a_de = k2
        d_rel_de = k1 - k2
        if sigma_rv:
            dk1 = _fitzpatrick_dk_drv(rv, inverse_um_1)
            dk2 = _fitzpatrick_dk_drv(rv, inverse_um_2)
            d_a_drv = e_b_v * dk2
            d_rel_drv = e_b_v * (dk1 - dk2)
        else:
            d_a_drv = 0.0
            d_rel_drv = 0.0

    a_filter_2_err = _rss(d_a_de * sigma_ebv, d_a_drv * sigma_rv)
    relative_extinction_err = _rss(d_rel_de * sigma_ebv, d_rel_drv * sigma_rv)
    return (
        a_filter_2,
        relative_extinction,
        0.0 if a_filter_2_err is None else float(a_filter_2_err),
        0.0 if relative_extinction_err is None else float(relative_extinction_err),
    )


def cmd_correction_offsets(
        a_filter_2: float, relative_extinction: float, m_m: float,
        *, apply_to: str = "observation",
    ) -> tuple[float, float, float, float]:
    """
    Signed offsets ``(dmag_obs, dcolor_obs, dmag_iso, dcolor_iso)``.

    ``observation`` (default): subtract extinction and distance from the stars.
    ``isochrone``: add the same terms to theoretical isochrones so they sit on
    the apparent CMD.
    """
    target = str(apply_to).strip().lower()
    if target in ("observation", "data", "stars"):
        return (-(a_filter_2 + m_m), -relative_extinction, 0.0, 0.0)
    if target in ("isochrone", "isochrones"):
        return (0.0, 0.0, a_filter_2 + m_m, relative_extinction)
    raise ValueError(
        f"apply_to must be 'observation' or 'isochrone', got {apply_to!r}"
    )
