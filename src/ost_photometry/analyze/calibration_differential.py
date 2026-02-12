"""
Differential photometry calibration with extinction correction.

Integrated from example_code/photometry_calibration.py.
Provides flexible extinction correction and airmass calculation
as an alternative to the legacy calibration pipeline.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time


class CoefficientMode(Enum):
    """Mode for coefficient determination."""

    PER_IMAGE = auto()  # Each image individually
    PER_NIGHT = auto()  # Averaged over a night
    FIXED = auto()  # Fixed, preset values
    ENSEMBLE = auto()  # Ensemble photometry (all images together)


class ExtinctionOrder(Enum):
    """Order of extinction correction."""

    NONE = 0
    FIRST = 1  # m_0 = m - k' * X
    SECOND = 2  # m_0 = m - k' * X - k'' * X * (color)


@dataclass
class ExtinctionCoefficients:
    """Extinction coefficients for a filter."""

    filter_name: str
    k_prime: float  # First order [mag/airmass]
    k_prime_err: float = 0.0
    k_second: float = 0.0  # Second order [mag/airmass/mag_color]
    k_second_err: float = 0.0
    color_filter_1: str = ""  # For k'' required color index
    color_filter_2: str = ""
    valid: bool = True

    def __repr__(self) -> str:
        s = f"k'_{self.filter_name} = {self.k_prime:.4f}±{self.k_prime_err:.4f}"
        if self.k_second != 0:
            ci = f"({self.color_filter_1}-{self.color_filter_2})"
            s += f", k''_{self.filter_name} = {self.k_second:.4f}±{self.k_second_err:.4f} × {ci}"
        return s


# Default extinction coefficients for typical good observing sites
DEFAULT_EXTINCTION = {
    "U": ExtinctionCoefficients(
        "U", k_prime=0.55, k_prime_err=0.05, k_second=0.03,
        color_filter_1="U", color_filter_2="B",
    ),
    "B": ExtinctionCoefficients(
        "B", k_prime=0.30, k_prime_err=0.03, k_second=0.02,
        color_filter_1="B", color_filter_2="V",
    ),
    "V": ExtinctionCoefficients(
        "V", k_prime=0.18, k_prime_err=0.02, k_second=0.01,
        color_filter_1="B", color_filter_2="V",
    ),
    "R": ExtinctionCoefficients(
        "R", k_prime=0.12, k_prime_err=0.02, k_second=0.01,
        color_filter_1="V", color_filter_2="R",
    ),
    "I": ExtinctionCoefficients(
        "I", k_prime=0.07, k_prime_err=0.02, k_second=0.005,
        color_filter_1="R", color_filter_2="I",
    ),
}


def calculate_airmass(
    coords: SkyCoord,
    obstime: Time,
    location: EarthLocation,
    method: str = "secz",
) -> np.ndarray:
    """
    Calculate airmass for given coordinates and time.

    Use when FITS header does not provide AIRMASS.

    Parameters
    ----------
    coords : SkyCoord
        Celestial coordinates
    obstime : Time
        Observation time(s)
    location : EarthLocation
        Observatory location
    method : str
        'secz' (simple) or 'pickering' (better at large zenith angle)

    Returns
    -------
    np.ndarray
        Airmass values
    """
    altaz = coords.transform_to(AltAz(obstime=obstime, location=location))
    alt = altaz.alt.deg

    if method == "secz":
        zenith_angle = 90 - alt
        airmass = 1 / np.cos(np.radians(zenith_angle))
    elif method == "pickering":
        airmass = 1 / np.sin(np.radians(alt + 244 / (165 + 47 * alt**1.1)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(airmass, 1.0, 10.0)


class ExtinctionCorrector:
    """
    Correct atmospheric extinction.

    Atmosphere absorbs light wavelength-dependently.
    Correction: m_0 = m_obs - k' × X - k'' × X × (color index)
    where X = Airmass
    """

    def __init__(
        self,
        coefficients: Optional[dict[str, ExtinctionCoefficients]] = None,
        order: ExtinctionOrder = ExtinctionOrder.FIRST,
    ):
        """
        Parameters
        ----------
        coefficients : dict, optional
            Extinction coefficients per filter. If None, defaults are used.
        order : ExtinctionOrder
            FIRST (k' only) or SECOND (k' and k'')
        """
        self.coefficients = coefficients or {k: ExtinctionCoefficients(
            v.filter_name, v.k_prime, v.k_prime_err, v.k_second,
            v.k_second_err, v.color_filter_1, v.color_filter_2, v.valid,
        ) for k, v in DEFAULT_EXTINCTION.items()}
        self.order = order

    def correct(
        self,
        data: Table,
        airmass_col: str = "airmass",
        mag_col_prefix: str = "mag_",
        output_prefix: str = "mag_ext_",
        filters: Optional[list[str]] = None,
        inplace: bool = False,
    ) -> Table:
        """
        Apply extinction correction to magnitudes.

        Parameters
        ----------
        data : Table
            Input table with magnitudes and airmass
        airmass_col : str
            Name of airmass column
        mag_col_prefix : str
            Prefix of magnitude columns
        output_prefix : str
            Prefix for corrected magnitudes
        filters : list, optional
            Filters to correct. Default: all available
        inplace : bool
            If True, modify original table

        Returns
        -------
        Table
            Table with extinction-corrected magnitudes
        """
        import warnings

        if not inplace:
            data = data.copy()

        if airmass_col not in data.colnames:
            raise ValueError(f"Airmass column '{airmass_col}' not found!")

        X = np.array(data[airmass_col])

        if filters is None:
            filters = [
                col.replace(mag_col_prefix, "")
                for col in data.colnames
                if col.startswith(mag_col_prefix)
            ]

        for filt in filters:
            mag_col = f"{mag_col_prefix}{filt}"
            if mag_col not in data.colnames:
                continue

            coeff = self.coefficients.get(filt)
            if coeff is None or not coeff.valid:
                warnings.warn(
                    f"No extinction coefficients for {filt}. Skipping.",
                    stacklevel=2,
                )
                continue

            m_obs = np.array(data[mag_col], dtype=float)
            correction = coeff.k_prime * X

            if self.order == ExtinctionOrder.SECOND and coeff.k_second != 0:
                ci_col1 = f"{mag_col_prefix}{coeff.color_filter_1}"
                ci_col2 = f"{mag_col_prefix}{coeff.color_filter_2}"
                if ci_col1 in data.colnames and ci_col2 in data.colnames:
                    color = np.array(data[ci_col1]) - np.array(data[ci_col2])
                    correction += coeff.k_second * X * color

            m_corrected = m_obs - correction
            data[f"{output_prefix}{filt}"] = m_corrected

        return data


__all__ = [
    "CoefficientMode",
    "ExtinctionCoefficients",
    "ExtinctionCorrector",
    "ExtinctionOrder",
    "calculate_airmass",
    "DEFAULT_EXTINCTION",
]
