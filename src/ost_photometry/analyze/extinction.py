"""
Atmospheric extinction for differential (and general) photometry.

Extinction coefficients, first/second-order correction (:class:`ExtinctionCorrector`),
airmass calculation, and fits from flux vs airmass or from comparison stars.

Also defines :class:`CoefficientMode`, which controls how *transformation*
coefficients are combined in differential calibration (per image, per night, etc.);
it lives here because the differential pipeline imports extinction and catalog code
together.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import warnings

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

from .warnings_types import OstPhotometryAnalyzeWarning


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
    """Extinction coefficients for a filter.

    Coefficients depend strongly on observing conditions: humidity, aerosols,
    altitude, and light pollution (Bortle) all affect k. They can vary
    night-to-night and season-to-season. For precision, determine from your
    own data via fit_extinction_from_value_airmass or fit_extinction_from_comparison_stars.
    """

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


# Default extinction coefficients for Bortle 5–6 (light-polluted) sites.
# Higher values than pristine sites due to increased aerosols/humidity.
# Typical ranges: U 0.46–0.61, B 0.19–0.31, V 0.12–0.21, R 0.07–0.12 mag/airmass.
# Sources: Berry & Burnell, Handbook of Astronomical Image Processing (Willmann-Bell);
# AAVSO https://www.aavso.org/content/typical-values-2nd-order-extinction-coefficients;
# cat-star.org http://cat-star.org/SOCO/PROCESSING/extinction.html
# Note: Coefficients vary night-to-night and season-to-season (humidity, aerosols).
# For precision use fit_extinction_from_value_airmass or fit_extinction_from_comparison_stars.
DEFAULT_EXTINCTION = {
    "U": ExtinctionCoefficients(
        "U", k_prime=0.60, k_prime_err=0.06, k_second=0.03,
        color_filter_1="U", color_filter_2="B",
    ),
    "B": ExtinctionCoefficients(
        "B", k_prime=0.33, k_prime_err=0.04, k_second=0.02,
        color_filter_1="B", color_filter_2="V",
    ),
    "V": ExtinctionCoefficients(
        "V", k_prime=0.20, k_prime_err=0.03, k_second=0.01,
        color_filter_1="B", color_filter_2="V",
    ),
    "R": ExtinctionCoefficients(
        "R", k_prime=0.13, k_prime_err=0.02, k_second=0.01,
        color_filter_1="V", color_filter_2="R",
    ),
    "I": ExtinctionCoefficients(
        "I", k_prime=0.08, k_prime_err=0.02, k_second=0.005,
        color_filter_1="R", color_filter_2="I",
    ),
}


def fit_extinction_from_value_airmass(
    data: Table,
    value_col: str | None = None,
    value_cols: dict[str, str] | None = None,
    fallback_airmass_col: str = "airmass",
    id_col: str | None = "id",
    use_magnitude: bool = False,
    output_dir: str | None = None,
    file_type: str = "pdf",
) -> dict[str, ExtinctionCoefficients]:
    """
    Determine extinction coefficients from flux or magnitude vs airmass (cat-star.org method).

    Observe stars (e.g. G2V) over several hours as they rise/set.
    - With flux: ln(flux) vs X has slope -k (extinction dims the star).
    - With magnitude: m vs X has slope +k (m = m_0 + k*X).
    Best done on clear moonlit nights when other imaging is not possible.

    Parameters
    ----------
    data : Table
        Rows = individual measurements. Must have airmass and flux or magnitude.
    value_col : str, optional
        Single column (flux or magnitude). Mutually exclusive with value_cols.
    value_cols : dict, optional
        {filter_name: column_name}. Use for multi-filter.
    fallback_airmass_col : str
        Column with airmass X.
    id_col : str, optional
        Column identifying the star/source. If given, fit per-source then average k.
    use_magnitude : bool
        If True, column contains magnitude (slope of m vs X = k). If False, flux.
    output_dir : str, optional
        If provided, save diagnostic plots to output_dir/extinction_fit/.
    file_type : str
        Plot file format when output_dir is set. Default is ``pdf``.

    Returns
    -------
    dict[str, ExtinctionCoefficients]
        Extinction coefficients per filter.

    References
    ----------
    cat-star.org: http://cat-star.org/SOCO/PROCESSING/extinction.html
    Berry & Burnell: Handbook of Astronomical Image Processing (Willmann-Bell)
    """
    from scipy import stats

    airmass = np.asarray(data[fallback_airmass_col], dtype=float)
    valid = np.isfinite(airmass) & (airmass >= 1.0)
    if not np.any(valid):
        return {}

    if value_col is not None and value_cols is None:
        value_cols = {"default": value_col}
    elif value_cols is None:
        raise ValueError("Provide value_col or value_cols")

    result = {}
    for filter_, col in value_cols.items():
        if col not in data.colnames:
            continue
        vals = np.asarray(data[col], dtype=float)
        if use_magnitude:
            ok = valid & np.isfinite(vals)
            y = vals[ok]
            sign = 1.0  # slope of m vs X = k
        else:
            ok = valid & np.isfinite(vals) & (vals > 0)
            y = np.log(vals[ok])
            sign = -1.0  # slope of ln(flux) vs X = -k
        if np.sum(ok) < 3:
            continue
        airmass_ok = airmass[ok]

        if id_col is not None and id_col in data.colnames:
            ids = np.asarray(data[id_col])[ok]
            slopes = []
            for uid in np.unique(ids):
                mask = ids == uid
                if np.sum(mask) < 2:
                    continue
                slope, intercept, r, p, se = stats.linregress(airmass_ok[mask], y[mask])
                slopes.append((slope, se))
            if not slopes:
                continue
            k_vals = np.array([sign * s[0] for s in slopes])
            k_errs = np.array([s[1] for s in slopes])
            k_prime = float(np.mean(k_vals))
            k_prime_err = float(np.sqrt(np.mean(k_errs**2) + np.var(k_vals)))
        else:
            slope, intercept, r, p, se = stats.linregress(airmass_ok, y)
            k_prime = float(sign * slope)
            k_prime_err = float(se)

        result[filter_] = ExtinctionCoefficients(
            filter_, k_prime=k_prime, k_prime_err=k_prime_err, k_second=0.0
        )

    if output_dir and result:
        from . import plots
        data_by_filter = {}
        for filter_, col in value_cols.items():
            if col not in data.colnames or filter_ not in result:
                continue
            vals = np.asarray(data[col], dtype=float)
            if use_magnitude:
                ok = valid & np.isfinite(vals)
                y = vals[ok]
            else:
                ok = valid & np.isfinite(vals) & (vals > 0)
                y = np.log(vals[ok])
            if np.sum(ok) < 3:
                continue
            airmass_ok = airmass[ok]
            data_by_filter[filter_] = (airmass_ok, y)
        if data_by_filter:
            plots.plot_extinction_fit_value_airmass(
                output_dir,
                data_by_filter,
                result,
                use_magnitude=use_magnitude,
                file_type=file_type,
            )

    return result


def observation_to_extinction_fit_table(
    observation: "object",
    filter_list: list[str],
    mag_col: str = "mags_fit",
    use_flux: bool = False,
) -> Table:
    """
    Build table for fit_extinction_from_value_airmass from reduced observation.

    Collects (id, airmass, mag/flux) for each star in each image. Requires
    WCS, extraction, and correlation to be done. Stars must be observed at
    different airmasses (e.g. over several hours as they rise/set).

    Parameters
    ----------
    observation : object
        Observation with image_series_dict, each image with photometry.
    filter_list : list[str]
        Filters to include.
    mag_col : str
        Photometry column for magnitude (default mags_fit).
    use_flux : bool
        If True, expect flux column (e.g. flux_fit). If False, use magnitude.

    Returns
    -------
    Table
        Columns: id, airmass, and mag_<filter> or flux_<filter> per filter.
    """
    from astropy.table import vstack

    rows = []
    for filter_ in filter_list:
        if filter_ not in getattr(observation, "image_series_dict", {}):
            continue
        image_series = observation.image_series_dict[filter_]
        for image in image_series.image_list:
            if image.photometry is None:
                continue
            phot = image.photometry
            if mag_col not in phot.colnames and not use_flux:
                continue
            col = "flux_fit" if use_flux else mag_col
            if col not in phot.colnames:
                continue
            vals = np.asarray(phot[col], dtype=float)
            if hasattr(vals, "value"):
                vals = vals.value
            ids = np.arange(len(phot)) if "id" not in phot.colnames else phot["id"]
            airmass = getattr(image, "air_mass", None)
            if airmass is None:
                airmass = 1.0
            airmass = float(airmass)
            mag_col_name = f"mag_{filter_}" if not use_flux else f"flux_{filter_}"
            tbl = Table()
            tbl["id"] = np.asarray(ids, dtype=int)
            tbl["airmass"] = np.full(len(phot), airmass)
            tbl[mag_col_name] = np.asarray(vals, dtype=float)
            rows.append(tbl)

    if not rows:
        return Table()
    combined = vstack(rows)
    return combined


def extinction_airmass_ready(
    data: Table,
    filters: list[str],
    fallback_airmass_col: str = "airmass",
) -> bool:
    """
    True if extinction correction can obtain X for every ``filters`` entry:
    either a global ``fallback_airmass_col`` or per-filter ``airmass_<f>`` for each f.
    """
    if fallback_airmass_col in data.colnames:
        return True
    return all(f"airmass_{f}" in data.colnames for f in filters)


def fit_extinction_from_comparison_stars(
    epochs: dict[str, Table],
    mag_col_prefix: str = "mag_",
    std_col_prefix: str = "mag_std_",
    fallback_airmass_col: str = "airmass",
    output_dir: str | None = None,
    file_type: str = "pdf",
) -> dict[str, ExtinctionCoefficients]:
    """
    Determine extinction coefficients from APASS-matched comparison stars.

    For each filter: m_obs = m_std + k*X + ZP. Fit k from mean(m_obs - m_std) vs X
    across calibration epochs. One point per epoch (mean residual over comparison stars).
    ZP can vary per epoch; the slope k is shared.

    Parameters
    ----------
    epochs : dict[str, Table]
        {epoch_id: Table} with columns: airmass, mag_<filter>, mag_std_<filter>.
        Tables must be from APASS crossmatch or similar.
    mag_col_prefix : str
        Prefix for instrumental magnitude columns (e.g. ``mag_B``). Columns that
        also start with ``std_col_prefix`` (e.g. ``mag_std_B`` when it is
        ``mag_std_``) are skipped so catalog mags are not mistaken for
        instrument mags.
    std_col_prefix : str
        Prefix for standard magnitude columns.
    fallback_airmass_col : str
        Column with airmass.
    output_dir : str, optional
        If provided, save diagnostic plots to output_dir/extinction_fit/.
    file_type : str
        Plot file format when output_dir is set. Default is ``pdf``.

    Returns
    -------
    dict[str, ExtinctionCoefficients]
        Extinction coefficients per filter.
    """
    from scipy import stats

    # Model (per filter): m_obs = m_std + k' * X + ZP_epoch.
    # ZP_epoch absorbs transparency + zero point per exposure; k' is the first-order
    # extinction (mag/airmass). Averaging residuals over comparison stars in one epoch
    # removes star-to-star offsets; regressing mean(m_obs - m_std) vs X isolates k'.

    if not epochs:
        warnings.warn(
            "fit_extinction_from_comparison_stars: no epochs provided.",
            category=OstPhotometryAnalyzeWarning,
            stacklevel=2,
        )
        return {}

    # One (X, delta) per epoch per filter: X = airmass, delta = mean residual vs catalog.
    by_filter: dict[str, list[tuple[float, float]]] = {}

    for epoch_id, tbl in epochs.items():
        for col in tbl.colnames:
            if not col.startswith(mag_col_prefix):
                continue
            # e.g. mag_std_B also starts with mag_; treat as std column, not mag_*.
            if col.startswith(std_col_prefix):
                continue
            filter_ = col[len(mag_col_prefix) :]
            band_airmass_col = f"airmass_{filter_}"
            if band_airmass_col in tbl.colnames:
                airmass_median = float(
                    np.median(np.asarray(tbl[band_airmass_col], dtype=float))
                )
            elif fallback_airmass_col in tbl.colnames:
                airmass_median = float(
                    np.median(np.asarray(tbl[fallback_airmass_col], dtype=float))
                )
            else:
                warnings.warn(
                    f"fit_extinction_from_comparison_stars: epoch {epoch_id!r} filter "
                    f"{filter_!r}: no {band_airmass_col!r} or {fallback_airmass_col!r}; skipping this band.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
                continue
            std_col = f"{std_col_prefix}{filter_}"
            if std_col not in tbl.colnames:
                warnings.warn(
                    f"fit_extinction_from_comparison_stars: epoch {epoch_id!r} filter "
                    f"{filter_!r}: has {col!r} but missing {std_col!r}.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
                continue
            m_obs = np.asarray(tbl[col], dtype=float)
            m_std = np.asarray(tbl[std_col], dtype=float)
            valid = np.isfinite(m_obs) & np.isfinite(m_std)
            if not np.any(valid):
                warnings.warn(
                    f"fit_extinction_from_comparison_stars: epoch {epoch_id!r} filter "
                    f"{filter_!r}: no finite pairs for instrumental vs standard magnitude.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
                continue

            # Mean over stars: epoch-level offset cancels in the slope; slope is k'.
            delta = np.nanmean((m_obs - m_std)[valid])
            if by_filter.get(filter_) is None:
                by_filter[filter_] = []
            by_filter[filter_].append((airmass_median, delta))

    # Per filter: linear regression delta ≈ k' * X + const (intercept absorbs mean ZP).
    result = {}
    data_by_filter = {}
    for filter_, pairs in by_filter.items():
        if len(pairs) < 3:
            warnings.warn(
                f"fit_extinction_from_comparison_stars: filter {filter_!r} has only "
                f"{len(pairs)} epoch point(s), need >= 3 for regression; skipping.",
                stacklevel=2,
            )
            continue
        airmass_arr = np.array([p[0] for p in pairs])
        d_arr = np.array([p[1] for p in pairs])
        valid = np.isfinite(airmass_arr) & np.isfinite(d_arr)
        if np.sum(valid) < 3:
            warnings.warn(
                f"fit_extinction_from_comparison_stars: filter {filter_!r} has only "
                f"{int(np.sum(valid))} finite (airmass, delta) point(s) after NaN filter; skipping.",
                category=OstPhotometryAnalyzeWarning,
                stacklevel=2,
            )
            continue
        slope, intercept, r, p, se = stats.linregress(
            airmass_arr[valid], d_arr[valid]
        )
        result[filter_] = ExtinctionCoefficients(
            filter_, k_prime=float(slope), k_prime_err=float(se), k_second=0.0
        )
        if output_dir:
            data_by_filter[filter_] = (airmass_arr[valid], d_arr[valid])

    # Plots only for filters that got a successful fit (same arrays as used in linregress).
    if output_dir and data_by_filter:
        from . import plots
        plots.plot_extinction_fit_comparison_stars(
            output_dir,
            data_by_filter,
            result,
            file_type=file_type,
        )

    if not result and epochs:
        warnings.warn(
            "fit_extinction_from_comparison_stars: no coefficients returned "
            "(no valid mag/std pairs across epochs, or insufficient airmass points per filter).",
            category=OstPhotometryAnalyzeWarning,
            stacklevel=2,
        )

    return result


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
    Correction: m_0 = m_obs - k' * X - k'' * X * (color index)
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
        fallback_airmass_col: str = "airmass",
        mag_col_prefix: str = "mag_",
        output_prefix: str = "mag_ext_",
        filters: Optional[list[str]] = None,
        inplace: bool = False,
        std_col_prefix: Optional[str] = "mag_std_",
        catalog_color_prefix: Optional[str] = "mag_std_",
    ) -> Table:
        """
        Apply extinction correction to magnitudes.

        Parameters
        ----------
        data : Table
            Input table with magnitudes and airmass
        fallback_airmass_col : str
            Global airmass column, used when per-filter ``airmass_<f>`` is absent.
            At least one of ``fallback_airmass_col`` or all ``airmass_<f>`` for active filters
            must be present (see :func:`extinction_airmass_ready`).
        mag_col_prefix : str
            Prefix of magnitude columns
        output_prefix : str
            Prefix for corrected magnitudes
        filters : list, optional
            Filters to correct. Default: all available
        inplace : bool
            If True, modify original table
        std_col_prefix : str, optional
            When ``filters`` is None, column names starting with this prefix are
            excluded from the auto-detected filter list (same convention as
            :func:`fit_extinction_from_comparison_stars`). Pass ``None`` to
            disable.

        Returns
        -------
        Table
            Table with extinction-corrected magnitudes
        """
        if not inplace:
            data = data.copy()

        if filters is None:
            filters = [
                col[len(mag_col_prefix) :]
                for col in data.colnames
                if col.startswith(mag_col_prefix)
                and (
                    std_col_prefix is None
                    or not col.startswith(std_col_prefix)
                )
            ]

        if not extinction_airmass_ready(data, filters, fallback_airmass_col):
            raise ValueError(
                f"Need '{fallback_airmass_col}' or all of "
                f"{[f'airmass_{f}' for f in filters]} for extinction correction."
            )

        for filter_ in filters:
            mag_col = f"{mag_col_prefix}{filter_}"
            if mag_col not in data.colnames:
                continue

            coeff = self.coefficients.get(filter_)
            if coeff is None or not coeff.valid:
                warnings.warn(
                    f"No extinction coefficients for {filter_}. Skipping.",
                    category=OstPhotometryAnalyzeWarning,
                    stacklevel=2,
                )
                continue

            band_airmass_col = f"airmass_{filter_}"
            if band_airmass_col in data.colnames:
                airmass = np.asarray(data[band_airmass_col], dtype=float)
            else:
                airmass = np.asarray(data[fallback_airmass_col], dtype=float)

            m_obs = np.array(data[mag_col].ravel(), dtype=float)
            if self.order == ExtinctionOrder.NONE:
                data[f"{output_prefix}{filter_}"] = m_obs
                continue

            correction = coeff.k_prime * airmass

            if self.order == ExtinctionOrder.SECOND and coeff.k_second != 0:
                ci_col1 = f"{mag_col_prefix}{coeff.color_filter_1}"
                ci_col2 = f"{mag_col_prefix}{coeff.color_filter_2}"
                if ci_col1 in data.colnames and ci_col2 in data.colnames:
                    color = np.array(data[ci_col1].ravel()) - np.array(data[ci_col2].ravel())
                    correction += coeff.k_second * airmass * color
                elif catalog_color_prefix:
                    s1 = f"{catalog_color_prefix}{coeff.color_filter_1}"
                    s2 = f"{catalog_color_prefix}{coeff.color_filter_2}"
                    if s1 in data.colnames and s2 in data.colnames:
                        color = np.asarray(data[s1].ravel(), dtype=float) - np.asarray(
                            data[s2].ravel(), dtype=float
                        )
                        color = np.where(np.isfinite(color), color, 0.0)
                        correction += coeff.k_second * airmass * color
                    else:
                        missing = [c for c in (ci_col1, ci_col2) if c not in data.colnames]
                        warnings.warn(
                            f"Second-order extinction for filter {filter_} skipped: missing "
                            f"instrumental color {missing} and catalog columns "
                            f"{[c for c in (s1, s2) if c not in data.colnames]}.",
                            category=OstPhotometryAnalyzeWarning,
                            stacklevel=2,
                        )
                else:
                    missing = [c for c in (ci_col1, ci_col2) if c not in data.colnames]
                    warnings.warn(
                        f"Second-order extinction for filter {filter_} skipped: missing "
                        f"columns {missing} (need {coeff.color_filter_1}-{coeff.color_filter_2} "
                        f"under prefix '{mag_col_prefix}').",
                        category=OstPhotometryAnalyzeWarning,
                        stacklevel=2,
                    )

            m_corrected = m_obs - correction
            data[f"{output_prefix}{filter_}"] = m_corrected

        return data


__all__ = [
    "CoefficientMode",
    "ExtinctionCoefficients",
    "ExtinctionCorrector",
    "ExtinctionOrder",
    "calculate_airmass",
    "DEFAULT_EXTINCTION",
    "extinction_airmass_ready",
    "fit_extinction_from_value_airmass",
    "fit_extinction_from_comparison_stars",
    "observation_to_extinction_fit_table",
]
