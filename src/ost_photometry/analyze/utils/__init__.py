"""Analysis utility implementations (named modules under this package)."""

from .cluster_selection import find_cluster, proper_motion_selection, region_selection
from .cmd_defaults import check_variable_absolute_cmd, check_variable_apparent_cmd
from .duplicates import clear_duplicates
from .errors import err_prop
from .legacy_magnitudes import (
    calibrated_epochs_to_legacy_wide_table,
    find_filter_for_magnitude_transformation,
    mk_magnitudes_table,
    save_calibration,
    save_magnitudes_ascii,
    transformation_keys_for_table_magnitudes,
)
from .limiting_magnitude import derive_limiting_magnitude
from .photometry import flux_to_magnitudes, rm_edge_objects
from .series_wcs import find_wcs
from .simbad_annotate import mark_simbad_objects_on_image, query_simbad_objects
from .starmaps import (
    prepare_and_plot_starmap,
    prepare_and_plot_starmap_from_image_series,
    prepare_and_plot_starmap_from_observation,
)

__all__ = [
    "calibrated_epochs_to_legacy_wide_table",
    "check_variable_absolute_cmd",
    "check_variable_apparent_cmd",
    "clear_duplicates",
    "derive_limiting_magnitude",
    "err_prop",
    "find_cluster",
    "find_filter_for_magnitude_transformation",
    "find_wcs",
    "flux_to_magnitudes",
    "mark_simbad_objects_on_image",
    "mk_magnitudes_table",
    "prepare_and_plot_starmap",
    "prepare_and_plot_starmap_from_image_series",
    "prepare_and_plot_starmap_from_observation",
    "proper_motion_selection",
    "query_simbad_objects",
    "region_selection",
    "rm_edge_objects",
    "save_calibration",
    "save_magnitudes_ascii",
    "transformation_keys_for_table_magnitudes",
]
