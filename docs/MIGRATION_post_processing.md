# Post-processing migration (epoch-native layout)

## API renames

- `utilities.post_process_results` was removed. Use `post_process_cluster_field` from `ost_photometry.analyze.post_processing` (or `post_processing.cluster_field`).
- `convert_magnitudes_to_other_system` expects **epoch-native** columns `mag_cal_<filter>` / `err_cal_<filter>` and adds SDSS bands as `mag_sdss_<band>` / `err_sdss_<band>` (Jordi et al. 2005). Legacy wide column names are no longer supported.; `utilities` still re-exports these for compatibility.
- Light-curve helpers `prepare_time_series_data`, `mk_time_series`, `prepare_time_series_epoch_native`, and `prepare_plot_time_series` live in `ost_photometry.analyze.post_processing.light_curve`. The pipeline `LightCurveStep` uses `prepare_plot_time_series` from there.
- For **epoch-native** tables, pass `epoch_meta=context.calibration_epoch_meta` into `prepare_plot_time_series` (and `observation_times=None`); JDs come from each epoch’s `jd_by_filter` (legacy key `filter_jds` still read as fallback). `AnalysisPipeline` includes `LightCurveStep` last; it is off until `PipelineConfig.skip_light_curve=False` (plus `plot_light_curve_*` toggles).

## Output format

- Epoch-native magnitude tables are written as **ECSV** (`*.ecsv`) via `write_epoch_native_magnitudes` (calibration and post-processing), with table `meta["photometry_schema"] = "ost_photometry.epoch_native.v1"`. Use `read_epoch_native_magnitudes` to load them.
- Downstream scripts should use `Table.read(..., format="ascii.ecsv")` or `post_processing.io.read_epoch_native_magnitudes`.

## Pipeline configuration

- `Observation.run_pipeline(..., skip_post_process=True)` is ignored (unknown attribute on `PipelineConfig`). Use the new skip flags or `skip_calibration`.

- `PipelineConfig.skip_post_process` was **removed**. When `skip_calibration` is `True`, cluster post-process and limiting-magnitude steps still skip (unchanged for workflows such as extinction-only runs).
- New optional skips (all default `False`):
  - `skip_cluster_region_step`
  - `skip_cluster_gaia_step`
  - `skip_cluster_pm_step`
  - `skip_magnitude_convert_step`
  - `skip_save_post_processed_magnitudes` (skips `PostProcessSaveMagnitudesStep`; filter steps still run)
  - `skip_derive_limiting_magnitude` (skips `DeriveLimitingMagnitudeStep`; differential calibration already skips this step)

## Pipeline steps

- `PostProcessStep` was replaced by split cluster-field steps, `DeriveLimitingMagnitudeStep`, and `LightCurveStep` (see `pipeline/orchestrator.py`).
- Cluster / field post-processing: `PostProcessRegionStep`, `PostProcessClusterGaiaStep`, `PostProcessProperMotionStep`, then **`PostProcessMagnitudeConvertStep`** (module `pipeline/steps/magnitude_convert.py`, step name `magnitude_system_convert` — not cluster-specific), then `PostProcessSaveMagnitudesStep`. The convert step uses `apply_magnitude_system_convert_on_observation` in `post_processing/magnitude_convert.py`. Flags: `skip_cluster_*`, `skip_magnitude_convert_step`, `skip_save_post_processed_magnitudes`, plus `extract_only_circular_region`, `identify_cluster_gaia_data`, `clean_objs_using_pm`, `convert_magnitudes`.
- Low-level helpers: `apply_cluster_field_phase` (phases `region` | `gaia` | `pm` only) requires keyword `plot_context` (`ImagingPlotContext`, e.g. `imaging_context_from_image_series(reference_series)`). `write_post_processed_cluster_field_table` lives in the same module. `post_process_cluster_field` also requires `plot_context=`; it chains region→Gaia→PM, then optional magnitude conversion, then optional save. `filter_list` remains used for writing outputs.
- If more than one usable filter combination is returned by `find_filter_for_magnitude_transformation`, the split pipeline emits a `UserWarning`: phase order across combinations differs from calling `post_process_cluster_field` once per combination. Typical two-filter runs have a single combination and are unchanged.

## Schema

- See `schema.py`: `REQUIRED_EPOCH_NATIVE_COLUMNS`, `validate_epoch_native_table`, `PHOTOMETRY_TABLE_SCHEMA_ID`.

## Cluster field (region / Gaia)

- `utilities.region_selection`, `utilities.find_cluster`, and `utilities.proper_motion_selection` take **`plot_context=`** and/or **`image_series=`** (keyword-only); at least one is required. `ImagingPlotContext` holds WCS, reference image array, `filter_name`, output stub, and for Gaia steps `field_center_icrs` / `field_radius_arcmin` (filled by `imaging_context_from_image_series`). They use sky positions from the table when columns `ra`/`dec` or `ra (deg)`/`dec (deg)` exist; otherwise WCS + `x`/`y`.
- Starmaps go through `post_processing.coords.plot_starmap_from_imaging_context`.
- Multi-epoch differential tables (`epoch_id` with more than one value): cluster phases run region/Gaia/PM on the **first** `epoch_id`, then expand surviving `id` values to **all epochs**. `apply_magnitude_system_convert_on_observation` runs on the **full** table (all epochs).

## Differential calibration (pipeline)

- After `DifferentialCalibrationStep`, `observation.table_magnitudes` is the **epoch-native** vstack from `PhotometryCalibrator.get_calibrated_photometry`, normalized with `ensure_epoch_native_photometry_table` (schema metadata).
- Primary table file: `calibrated_magnitudes_<method>_<filters>.ecsv` under `tables/` (same writer as post-process ECSV). Legacy wide `.dat` is **not** written unless `PipelineConfig.write_differential_legacy_magnitudes_dat` is `True` (then `differential_calibrated_to_legacy_table` + `save_magnitudes_ascii`).

## Adapters

- `legacy_wide_table_to_epoch_native` / `ensure_epoch_native_photometry_table`: attach `photometry_schema` to differential vstack tables, or expand legacy **wide** rows (`{filter} (transformed|simple, image=...)`) into long form with `mag_cal_*` / `err_cal_*` and `epoch_id`.
