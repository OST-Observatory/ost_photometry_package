"""Cross-image and cross-filter source correlation."""

from .core import correlate_datasets, correlation_astropy, correlation_own
from .intra import (
    assign_correlated_object_ids_single_series,
    correlate_image_series_images,
    correlate_preserve_objects,
    correlate_preserve_variable,
)
from .inter import (
    assign_global_correlated_object_ids,
    correlate_image_series,
    correlate_preserve_calibration_objects,
    determine_object_position,
    inter_filter_correlation_separations_arcsec,
    inter_filter_correlation_separations_for_images,
    inter_filter_pair_image_label,
    inter_filter_pair_title_suffix,
)
from .protection import (
    merge_protected_object_ids,
    resolve_calibration_object_ids,
    resolve_protected_object_ids_for_intra,
    resolve_protected_object_ids_for_inter,
)
from .ooi import (
    find_objects_of_interest_astropy,
    find_objects_of_interest_srcor,
    identify_object_of_interest_in_dataset,
    verify_objects_of_interest_global_correlated_ids,
)

__all__ = [
    "assign_correlated_object_ids_single_series",
    "assign_global_correlated_object_ids",
    "correlate_datasets",
    "correlate_image_series",
    "correlate_image_series_images",
    "correlate_preserve_calibration_objects",
    "correlate_preserve_objects",
    "correlate_preserve_variable",
    "correlation_astropy",
    "correlation_own",
    "determine_object_position",
    "find_objects_of_interest_astropy",
    "find_objects_of_interest_srcor",
    "identify_object_of_interest_in_dataset",
    "inter_filter_correlation_separations_arcsec",
    "inter_filter_correlation_separations_for_images",
    "inter_filter_pair_image_label",
    "inter_filter_pair_title_suffix",
    "merge_protected_object_ids",
    "resolve_calibration_object_ids",
    "resolve_protected_object_ids_for_intra",
    "resolve_protected_object_ids_for_inter",
    "verify_objects_of_interest_global_correlated_ids",
]
