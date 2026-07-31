"""Post-processing: cluster-field filters, magnitude conversion, I/O, light curves.

HiPS/HOTPANTS subtraction lives in :mod:`ost_photometry.analyze.post_processing.hips_reference_subtract`
(not re-exported here) to avoid import cycles with :mod:`ost_photometry.analyze.utilities`.
"""

from .adapters import (
    ensure_epoch_native_photometry_table,
    legacy_wide_table_to_epoch_native,
)
from .cluster_field import (
    apply_cluster_field_phase,
    post_process_cluster_field,
    write_post_processed_cluster_field_table,
)
from .coords import plot_starmap_from_imaging_context, table_object_sky_coords
from .convert import (
    convert_magnitudes_to_other_system,
)
from .imaging import ImagingPlotContext, imaging_context_from_image_series
from .io import (
    read_epoch_native_magnitudes,
    write_epoch_native_magnitudes,
)
from .magnitude_convert import apply_magnitude_system_convert_on_observation
from .magnitude_systems import (
    infer_filter_set,
    magnitude_system_axis_suffix,
    resolve_catalog_magnitude_system,
    validate_magnitude_output_request,
)
from .light_curve import (
    attach_observation_jd_column,
    epoch_native_mag_err_columns,
    is_epoch_native_photometry_table,
    load_calibration_epoch_meta_json,
    mk_time_series,
    mk_time_series_flux,
    object_id_from_epoch_native_sky,
    plot_light_curve_from_epoch_native_ecsv,
    prepare_plot_time_series,
    prepare_time_series_data,
    prepare_time_series_epoch_native,
    save_calibration_epoch_meta_json,
)
from .schema import (
    PHOTOMETRY_TABLE_SCHEMA_ID,
    REQUIRED_EPOCH_NATIVE_COLUMNS,
    validate_epoch_native_table,
)

__all__ = [
    "PHOTOMETRY_TABLE_SCHEMA_ID",
    "REQUIRED_EPOCH_NATIVE_COLUMNS",
    "convert_magnitudes_to_other_system",
    "ImagingPlotContext",
    "imaging_context_from_image_series",
    "plot_starmap_from_imaging_context",
    "table_object_sky_coords",
    "apply_cluster_field_phase",
    "epoch_native_mag_err_columns",
    "ensure_epoch_native_photometry_table",
    "legacy_wide_table_to_epoch_native",
    "apply_magnitude_system_convert_on_observation",
    "attach_observation_jd_column",
    "infer_filter_set",
    "is_epoch_native_photometry_table",
    "load_calibration_epoch_meta_json",
    "magnitude_system_axis_suffix",
    "mk_time_series",
    "mk_time_series_flux",
    "object_id_from_epoch_native_sky",
    "plot_light_curve_from_epoch_native_ecsv",
    "post_process_cluster_field",
    "prepare_plot_time_series",
    "prepare_time_series_data",
    "prepare_time_series_epoch_native",
    "resolve_catalog_magnitude_system",
    "save_calibration_epoch_meta_json",
    "read_epoch_native_magnitudes",
    "validate_epoch_native_table",
    "validate_magnitude_output_request",
    "write_epoch_native_magnitudes",
    "write_post_processed_cluster_field_table",
]
