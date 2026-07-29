"""Plotting helpers for photometry analysis (package facade)."""

from .calibration_qc import (
    plot_aperture_growth_curve,
    plot_calibration_color_color_cal_stars,
    plot_calibration_crossmatch_separations,
    plot_calibration_night_summary,
    plot_calibration_transformation,
    plot_combined_separation_histograms,
    plot_extinction_fit_comparison_stars,
    plot_extinction_fit_value_airmass,
    plot_instrumental_vs_catalog_magnitudes,
    plot_inter_filter_correlation_separations,
    plot_inter_filter_correlation_separations_overview,
    plot_photometry_mag_vs_error,
    plot_zeropoint_residual_distribution,
    plot_zeropoint_residual_vs_color,
)
from .cmds import MakeCMDs
from .extraction import plot_apertures, plot_cutouts, plot_epsf, plot_residual
from .lightcurves import light_curve_fold, light_curve_jd
from .scatter import d3_scatter, scatter
from .starmaps import (
    compare_images,
    plot_annotated_image,
    plot_limiting_mag_sky_apertures,
    starmap,
)
from .style import (
    MaxRecursionError,
    initialize_plot,
    mk_color_cycler_error_bars,
    mk_color_cycler_symbols,
    mk_colormap,
    mk_line_cycler,
    mk_ticks_labels,
)

__all__ = [
    "MakeCMDs",
    "MaxRecursionError",
    "compare_images",
    "d3_scatter",
    "initialize_plot",
    "light_curve_fold",
    "light_curve_jd",
    "mk_color_cycler_error_bars",
    "mk_color_cycler_symbols",
    "mk_colormap",
    "mk_line_cycler",
    "mk_ticks_labels",
    "plot_annotated_image",
    "plot_aperture_growth_curve",
    "plot_apertures",
    "plot_calibration_color_color_cal_stars",
    "plot_calibration_crossmatch_separations",
    "plot_calibration_night_summary",
    "plot_calibration_transformation",
    "plot_combined_separation_histograms",
    "plot_cutouts",
    "plot_epsf",
    "plot_extinction_fit_comparison_stars",
    "plot_extinction_fit_value_airmass",
    "plot_instrumental_vs_catalog_magnitudes",
    "plot_inter_filter_correlation_separations",
    "plot_inter_filter_correlation_separations_overview",
    "plot_limiting_mag_sky_apertures",
    "plot_photometry_mag_vs_error",
    "plot_residual",
    "plot_zeropoint_residual_distribution",
    "plot_zeropoint_residual_vs_color",
    "scatter",
    "starmap",
]
