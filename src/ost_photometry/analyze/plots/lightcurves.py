"""Light-curve plotting helpers."""
from __future__ import annotations

import os

import numpy as np
from astropy.stats import sigma_clip as sigma_clipping
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u
import matplotlib.pyplot as plt

from ... import checks

plt.switch_backend("Agg")

def _nonnegative_errorbar_yerr(values) -> np.ndarray:
    """``matplotlib.errorbar`` rejects negative ``yerr``; use absolute uncertainty."""
    return np.abs(np.asarray(values, dtype=float))


def _safe_ylim_from_series(
    y_values,
    err_values,
    *,
    own_scaling: bool,
    invert_axis: bool,
    mag_like_threshold: tuple[float, float] = (0.9, 1.1),
    magnitude_system: str = "vega",
) -> str:
    """
    Choose y-axis label suffix; set ``plt.ylim`` only when limits are finite.

    Skips ``plt.ylim`` when stats are non-finite (e.g. NaN magnitudes left in series).
    """
    from ..post_processing.magnitude_systems import magnitude_system_axis_suffix

    mag_suffix = magnitude_system_axis_suffix(magnitude_system)
    y = np.asarray(y_values, dtype=float)
    fin = np.isfinite(y)
    if not np.any(fin):
        return mag_suffix

    median_data = float(np.nanmedian(y))
    min_data = float(np.nanmin(y))
    max_data = float(np.nanmax(y))

    if invert_axis and (median_data > 1.5 or median_data < 0.5):
        plt.gca().invert_yaxis()

    y_err = _nonnegative_errorbar_yerr(err_values)
    y_err_sigma = sigma_clipping(y_err, sigma=1.5)
    max_err = float(np.nanmax(y_err_sigma))
    if not np.isfinite(max_err) or max_err <= 0:
        max_err = 0.1

    lo_mag, hi_mag = mag_like_threshold
    if median_data > hi_mag or median_data < lo_mag:
        y_lim = max(float(np.nanmax(np.array([max_err * 1.5, 0.1]))), 0.05)
        if own_scaling and np.isfinite(median_data):
            plt.ylim([median_data + y_lim, median_data - y_lim])
        return mag_suffix

    y_lim = max(max_err * 1.2, 0.05)
    if own_scaling and np.isfinite(min_data) and np.isfinite(max_data):
        plt.ylim([min_data - y_lim, max_data + y_lim])
    return " [flux] (normalized)"


def _light_curve_set_ylabel_and_ylim(
    data_column: str,
    y_values,
    err_values,
    *,
    y_axis_style: str = "magnitude",
    own_scaling: bool = True,
    invert_axis: bool = True,
    fontsize: int = 15,
    magnitude_system: str = "vega",
) -> None:
    """
    Y-axis label and limits for :func:`light_curve_jd` / :func:`light_curve_fold`.

    ``y_axis_style`` is ``"magnitude"`` (default mag scaling) or ``"flux"``
    (linear flux / ADU-style, no inverted axis).
    """
    if y_axis_style == "flux":
        y = np.asarray(y_values, dtype=float)
        ev = _nonnegative_errorbar_yerr(err_values)
        fin = np.isfinite(y)
        if np.any(fin) and own_scaling:
            lo, hi = float(np.nanmin(y[fin])), float(np.nanmax(y[fin]))
            margin = float(np.nanmedian(ev[fin])) * 2.0 if np.any(np.isfinite(ev[fin])) else 0.0
            span = hi - lo
            margin = max(margin, span * 0.1 if np.isfinite(span) else 0.0, 1e-30)
            plt.ylim(lo - margin, hi + margin)
        plt.ylabel(f"{data_column} [flux]", fontsize=fontsize)
        return
    y_label_text = _safe_ylim_from_series(
        y_values,
        err_values,
        own_scaling=own_scaling,
        invert_axis=invert_axis,
        magnitude_system=magnitude_system,
    )
    plt.ylabel(data_column + y_label_text, fontsize=fontsize)



def light_curve_jd(
        ts: TimeSeries, data_column: str, err_column: str, output_dir: str,
        error_bars: bool = True, name_object: str | None = None,
        file_name_suffix: str = '', subdirectory: str = '',
        file_type: str = 'pdf', own_scaling: bool = True,
        invert_axis: bool = True,
        y_axis_style: str = "magnitude",
        magnitude_system: str = "vega") -> None:
    """
    Plot the light curve over Julian Date

    Parameters
    ----------
    ts
        Time series

    data_column
        Filter

    err_column
        Name of the error column

    output_dir
        Output directory

    error_bars
        If True error bars will be plotted.
        Default is ``False``.

    name_object
        Name of the object
        Default is ``None``.

    file_name_suffix
        Suffix to add to the file name
        Default is ``''``

    subdirectory
        Name of the subdirectory in which to save the plots

    file_type
        Type of plot file to be created
        Default is ``pdf``.

    own_scaling
        If ``True``, the Y-axis is subject to the normal mathplotlib
        autoscaling.
        Default is ``True``.

    invert_axis
        If ``True``, the Y-axis will be inverted.
        Default is ``True``.

    y_axis_style
        ``"magnitude"`` (default) or ``"flux"`` for extracted flux light curves
        (linear scale, ``[flux]`` label).

    magnitude_system
        ``vega`` / ``ab`` / … for the magnitude y-axis suffix (ignored for flux).
    """
    #   Check output directories
    if subdirectory != '':
        checks.check_output_directories(
            output_dir,
            f'{output_dir}/lightcurve{subdirectory}',
        )
    else:
        checks.check_output_directories(
            output_dir,
            os.path.join(output_dir, 'lightcurve'),
        )

    #   Make plot
    fig = plt.figure(figsize=(20, 9))

    #   Plot grid
    plt.grid(True, color='lightgray', linestyle='--')

    #   Set tick size
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    #   Set title
    if name_object is None:
        fig.suptitle(f'Light curve', fontsize=30)
    else:
        fig.suptitle(f'Light curve - {name_object}', fontsize=30)

    #   Plot data with or without error bars
    if not error_bars:
        plt.plot(ts.time.jd, ts[data_column], 'k.', markersize=3)
    else:
        plt.errorbar(
            ts.time.jd,
            np.array(ts[data_column]),
            yerr=_nonnegative_errorbar_yerr(ts[err_column]),
            marker='.',
            markersize=4,
            linestyle='none',
            capsize=2,
            ecolor='dodgerblue',
            color='darkred',
        )

    #   Set x and y axis label
    plt.xlabel('Julian Date', fontsize=15)
    _light_curve_set_ylabel_and_ylim(
        data_column,
        ts[data_column].value,
        ts[err_column].value,
        y_axis_style=y_axis_style,
        own_scaling=own_scaling,
        invert_axis=invert_axis,
        fontsize=15,
        magnitude_system=magnitude_system,
    )

    #   Save plot
    plt.savefig(
        f'{output_dir}/lightcurve{subdirectory}/lightcurve_jd_{name_object}'
        f'_{data_column}{file_name_suffix}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()


def light_curve_fold(
        time_series: TimeSeries, data_column: str, err_column: str,
        output_dir: str, transit_time: str, period: float,
        binning_factor: float | None = None, error_bars: bool = True,
        name_object: str | None = None, file_name_suffix: str = '',
        subdirectory: str = '', file_type: str = 'pdf',
        y_axis_style: str = "magnitude",
        magnitude_system: str = "vega") -> None:
    """
    Plot a folded light curve

    Parameters
    ----------
    time_series
        Time series

    data_column
        Filter

    err_column
        Name of the error column

    output_dir
        Output directory

    transit_time
        Time of the transit - Format example: "2020-09-18T01:00:00"

    period
        The period in days

    binning_factor
        Light-curve binning-factor in days
        Default is ``None``.

    error_bars
        If True error bars will be plotted.
        Default is ``False``.

    name_object
        Name of the object
        Default is ``None``.

    file_name_suffix
        Suffix to add to the file name
        Default is ``''``

    subdirectory
        Name of the subdirectory in which to save the plots

    file_type
        Type of plot file to be created
        Default is ``pdf``.
    """
    #   Check output directories
    if subdirectory != '':
        checks.check_output_directories(
            output_dir,
            f'{output_dir}/lightcurve{subdirectory}',
        )
    else:
        checks.check_output_directories(
            output_dir,
            os.path.join(output_dir, 'lightcurve'),
        )

    #   Make a time object for the  transit times
    transit_time = Time(transit_time, format='isot', scale='utc')

    #   Fold lightcurve
    ts_folded = time_series.fold(
        period=float(period) * u.day,
        epoch_time=transit_time,
    )

    #   Make plot
    fig = plt.figure(figsize=(20, 9))

    #   Plot grid
    plt.grid(True, color='lightgray', linestyle='--')

    #   Set tick size
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    #   Set title
    if name_object is None:
        fig.suptitle('Folded light curve', fontsize=30)
    else:
        fig.suptitle(f'Folded light curve - {name_object}', fontsize=30)

    #   Calculate binned lightcurve => plot
    if binning_factor is not None:
        ts_binned = aggregate_downsample(
            ts_folded,
            time_bin_size=binning_factor * u.day,
        )

        #   Remove zero entries in case the binning time is smaller than the
        #   time between the data points
        mask = np.array(ts_binned[data_column]) == 0.
        mask = np.invert(mask)

        if error_bars:
            plt.errorbar(
                ts_binned.time_bin_start.jd[mask],
                np.array(ts_binned[data_column][mask]),
                yerr=_nonnegative_errorbar_yerr(ts_binned[err_column][mask]),
                # fmt='k.',
                marker='o',
                ls='none',
                elinewidth=1,
                markersize=3,
                capsize=2,
                ecolor='dodgerblue',
                color='darkred',
            )
        else:
            plt.plot(
                ts_binned.time_bin_start.jd[mask],
                ts_binned[data_column][mask],
                'k.',
                markersize=3,
            )
    else:
        if error_bars:
            plt.errorbar(
                ts_folded.time.jd,
                np.array(ts_folded[data_column]),
                yerr=_nonnegative_errorbar_yerr(ts_folded[err_column]),
                # fmt='k.',
                marker='o',
                ls='none',
                elinewidth=1,
                markersize=3,
                capsize=2,
                ecolor='dodgerblue',
                color='darkred',
            )
        else:
            plt.plot(
                ts_folded.time.jd,
                ts_folded[data_column],
                'k.',
                markersize=3,
            )

    #   Set x and y axis label
    plt.xlabel('Time (days)', fontsize=16)
    _light_curve_set_ylabel_and_ylim(
        data_column,
        ts_folded[data_column].value,
        time_series[err_column].value,
        y_axis_style=y_axis_style,
        own_scaling=True,
        invert_axis=True,
        fontsize=16,
        magnitude_system=magnitude_system,
    )

    #   Save plot
    plt.savefig(
        f'{output_dir}/lightcurve{subdirectory}/lightcurve_folded_{name_object}'
        f'_{data_column}{file_name_suffix}.{file_type}',
        bbox_inches='tight',
        format=file_type,
    )
    plt.close()
