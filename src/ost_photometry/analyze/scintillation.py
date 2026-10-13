"""Atmospheric scintillation noise for photometric uncertainties.

Scintillation is a fractional flux error that does not depend on the
brightness of the star and is uncorrelated between stars more than a few
arcsec apart, so it adds in quadrature to the photon / sky error of every
source and does not cancel in differential photometry. For bright stars in
short exposures it dominates (HAT-P-20 test night, CDK20, 20 s: 0.002 mag
against 0.0023 mag photon noise).

Estimate after Young (1967) in the form of Osborn et al. (2015, MNRAS 452,
1707, eq. 7)::

    sigma_S**2 = 1e-5 * C_Y**2 * D**(-4/3) * t**(-1) * X**3 * exp(-2 h / H)

with the aperture diameter ``D`` in m, exposure time ``t`` in s, airmass
``X``, site altitude ``h`` and scale height ``H = 8000 m``. Young's original
coefficient underestimates the measured scintillation; Osborn et al. find a
median correction ``C_Y ≈ 1.5`` over several sites (the default). Central
obstruction is ignored.
"""

from __future__ import annotations

import math

import numpy as np
from astropy.table import Table

from .. import terminal_output
from ..telescope_specs import aperture_diameter_m, resolve_telescope

SCALE_HEIGHT_M = 8000.0
DEFAULT_COEFFICIENT = 1.5
_MAG_PER_FRACTION = 2.5 / math.log(10.0)


def scintillation_fraction(
    aperture_m: float,
    exposure_s: float,
    airmass: float,
    altitude_m: float = 0.0,
    coefficient: float = DEFAULT_COEFFICIENT,
) -> float:
    """1σ relative flux error from scintillation (Osborn et al. 2015, eq. 7).

    Returns NaN for a non-positive aperture or exposure time; an airmass
    below 1 (missing header value) is treated as 1.
    """
    try:
        d = float(aperture_m)
        t = float(exposure_s)
        x = float(airmass)
        h = float(altitude_m)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(d) and d > 0 and math.isfinite(t) and t > 0):
        return float("nan")
    if not math.isfinite(x) or x < 1.0:
        x = 1.0
    if not math.isfinite(h):
        h = 0.0
    variance = (
        1e-5
        * float(coefficient) ** 2
        * d ** (-4.0 / 3.0)
        / t
        * x**3
        * math.exp(-2.0 * h / SCALE_HEIGHT_M)
    )
    return math.sqrt(variance)


def add_scintillation_to_photometry(tbl: Table, fraction: float) -> Table:
    """Add ``fraction · flux`` in quadrature to ``flux_err`` and ``mags_unc``.

    The photometric error is kept as ``flux_err_phot``; calling this again
    starts from that column, so the term is never added twice.
    """
    if tbl is None or len(tbl) == 0 or "flux_err" not in tbl.colnames:
        return tbl
    if not math.isfinite(fraction) or fraction <= 0:
        return tbl
    if "flux_err_phot" not in tbl.colnames:
        tbl["flux_err_phot"] = tbl["flux_err"]
    flux = np.abs(_values(tbl["flux_fit"]))
    phot = _values(tbl["flux_err_phot"])
    err = np.hypot(phot, fraction * flux)
    unit = getattr(tbl["flux_err"], "unit", None)
    tbl["flux_err"] = err * unit if unit is not None else err
    if "mags_unc" in tbl.colnames:
        with np.errstate(divide="ignore", invalid="ignore"):
            mag_err = _MAG_PER_FRACTION * err / flux
        mag_unit = getattr(tbl["mags_unc"], "unit", None)
        tbl["mags_unc"] = mag_err * mag_unit if mag_unit is not None else mag_err
    tbl.meta["scintillation_fraction"] = float(fraction)
    return tbl


def _values(column) -> np.ndarray:
    """Plain float values of a ``Column`` or ``Quantity`` column."""
    return np.asarray(getattr(column, "value", column), dtype=float)


def _header_float(header, key: str) -> float:
    try:
        return float(header.get(key))
    except (TypeError, ValueError):
        return float("nan")


def apply_scintillation_to_image_series(
    image_series,
    *,
    altitude_m: float = 0.0,
    coefficient: float = DEFAULT_COEFFICIENT,
    aperture_m: float | None = None,
) -> dict | None:
    """Add the scintillation term to every extracted image of one filter.

    The aperture comes from ``aperture_m`` or the telescope catalog
    (``TELESCOP``, else ``FOCALLEN`` of the first image). Without a known
    aperture nothing changes and a warning names the telescope. Returns a
    summary (telescope, aperture, median / range in mag) or ``None``.
    """
    images = [
        im
        for im in getattr(image_series, "image_list", [])
        if getattr(im, "photometry", None) is not None and len(im.photometry) > 0
    ]
    if not images:
        return None
    header = images[0].get_header()
    telescope = resolve_telescope(header.get("TELESCOP"), header.get("FOCALLEN"))
    diameter = aperture_m if aperture_m is not None else aperture_diameter_m(telescope)
    filter_ = getattr(image_series, "filter_", "")
    if diameter is None:
        terminal_output.print_to_terminal(
            f"Scintillation ({filter_}): telescope "
            f"{header.get('TELESCOP')!r} (FOCALLEN {header.get('FOCALLEN')}) "
            "is not in the telescope catalog; no scintillation term. Set "
            "telescope_aperture_m to add it.",
            indent=1,
            style_name="WARNING",
        )
        return None

    fractions: list[float] = []
    n_default_airmass = 0
    for im in images:
        hdr = im.get_header()
        exposure = _header_float(hdr, "EXPTIME")
        airmass = getattr(im, "air_mass", None)
        try:
            airmass = float(airmass)
        except (TypeError, ValueError):
            airmass = float("nan")
        if not math.isfinite(airmass) or airmass < 1.0:
            n_default_airmass += 1
        frac = scintillation_fraction(diameter, exposure, airmass, altitude_m, coefficient)
        if math.isfinite(frac):
            im.photometry = add_scintillation_to_photometry(im.photometry, frac)
            fractions.append(frac)

    if not fractions:
        terminal_output.print_to_terminal(
            f"Scintillation ({filter_}): no usable EXPTIME; no scintillation term.",
            indent=1,
            style_name="WARNING",
        )
        return None
    mags = _MAG_PER_FRACTION * np.asarray(fractions)
    name = telescope or "telescope_aperture_m"
    msg = (
        f"Scintillation ({filter_}): D = {diameter:.3f} m ({name}), "
        f"{float(np.median(mags)):.4f} mag median "
        f"({float(np.min(mags)):.4f}–{float(np.max(mags)):.4f}) added to "
        f"{len(fractions)} image(s)"
    )
    if n_default_airmass:
        msg += f"; {n_default_airmass} without AIRMASS used X = 1"
    terminal_output.print_to_terminal(msg, indent=1, style_name="INFO")
    return {
        "telescope": telescope,
        "aperture_m": diameter,
        "median_mag": float(np.median(mags)),
        "n_images": len(fractions),
    }


__all__ = [
    "DEFAULT_COEFFICIENT",
    "add_scintillation_to_photometry",
    "apply_scintillation_to_image_series",
    "scintillation_fraction",
]
