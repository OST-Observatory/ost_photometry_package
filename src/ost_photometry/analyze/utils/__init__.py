"""Analysis utility implementations (named modules under this package)."""

from .cluster_selection import find_cluster, proper_motion_selection, region_selection
from .cmd_defaults import check_variable_absolute_cmd, check_variable_apparent_cmd
from .duplicates import clear_duplicates
from .epsf_selection import n_epsf_stars_to_select
from .errors import err_prop
from .legacy_magnitudes import (
    find_filter_for_magnitude_transformation,
    transformation_keys_for_table_magnitudes,
)
from .limiting_magnitude import derive_limiting_magnitude
from .photometry import flux_to_magnitudes, rm_edge_objects
from .series_wcs import find_wcs
from .starmaps import (
    prepare_and_plot_starmap,
    prepare_and_plot_starmap_from_image_series,
    prepare_and_plot_starmap_from_observation,
)

# Not in ``__all__``: star-import from ``utilities`` must not load post_processing.
_SIMBAD_EXPORTS = (
    "annotate_reference_image_with_simbad",
    "mark_simbad_objects_on_image",
    "query_simbad_objects",
)

__all__ = [
    "check_variable_absolute_cmd",
    "check_variable_apparent_cmd",
    "clear_duplicates",
    "derive_limiting_magnitude",
    "err_prop",
    "find_cluster",
    "find_filter_for_magnitude_transformation",
    "find_wcs",
    "flux_to_magnitudes",
    "n_epsf_stars_to_select",
    "prepare_and_plot_starmap",
    "prepare_and_plot_starmap_from_image_series",
    "prepare_and_plot_starmap_from_observation",
    "proper_motion_selection",
    "region_selection",
    "rm_edge_objects",
    "transformation_keys_for_table_magnitudes",
]


def __getattr__(name: str):
    # Lazy: post_processing.simbad_annotate must not load during utilities import
    # (post_processing.__init__ imports cluster_field → utilities).
    if name in _SIMBAD_EXPORTS:
        from . import simbad_annotate as _simbad

        return getattr(_simbad, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
