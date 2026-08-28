# Architecture and migration archive

Consolidated reference for the epoch-native pipeline, breaking API changes, and
upgrade paths. **Current options and decision tables** live in
[PIPELINE_CONFIG.md](PIPELINE_CONFIG.md) and [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md).

---

## Table of contents

1. [Calibration convergence](#1-calibration-convergence)
2. [Calibration epochs](#2-calibration-epochs)
3. [Calibration catalog sources](#3-calibration-catalog-sources)
4. [Post-processing](#4-post-processing)

---

## 1. Calibration convergence

Legacy (`calibration_module`, `CalibrationDataStep`, …) and differential-specific
config prefixes are removed. Use **one epoch-native calibration path**:

- **Step:** `CalibrationStep` + `CalibrationEngine`
- **Config:** `calibration_strategy`, `calibration_grouping`, `extinction_mode`, …
- **Presets:** `PipelineConfig.from_preset("median_zp_per_image" | "linear_fit_per_night" | …)`
- **Results:** `context.calibration_results`

### Presets

| Preset | strategy | grouping | extinction_mode | Use case |
|--------|----------|----------|-----------------|----------|
| `median_zp_per_image` | `median_zp` | `per_image` | `none` | Stacked multi-filter fields, cluster photometry |
| `linear_fit_per_night` | `linear_fit` | `per_night` | `none` | Multi-epoch light curves |
| `linear_fit_per_night_extinction` | `linear_fit` | `per_night` | `from_comparison_stars` | Multi-epoch light curves with significant airmass range |

Deprecated aliases (still accepted): `n2_stack` → `median_zp_per_image`, `c7_variable` → `linear_fit_per_night`, `c7_variable_extinction` → `linear_fit_per_night_extinction`, `mk_calib_trans` → `extract_protect_calibrators`, `mk_calib_calibrate` → `linear_fit_ensemble`, `ost_site` → `tabulated_extinction`.

Additional presets: `extract_protect_calibrators` (extract/intra-correlate, protect calibrators, skip apply), `linear_fit_ensemble` (ensemble derive-transform), `tabulated_extinction` (site extinction table).

### Removed (breaking)

| Removed | Replacement |
|---------|-------------|
| `calibration_module` | `calibration_strategy` + `calibration_grouping` + `extinction_mode` |
| `differential_*` config fields | Neutral names (`fit_sigma_clip`, `exposure_pairing`, …) |
| `CalibrationDataStep`, `CalibrationApplyStep`, `DifferentialCalibrationStep` | `CalibrationStep` |
| `context.differential_calib_parameters` | `context.calibration_results` |
| `derive_transformation_coefficients` | `derive_transform_from_data` (within `linear_fit`, 2 filters) |
| `calculate_zero_point_statistic` | `zp_subsample_statistic` |
| `write_differential_legacy_magnitudes_dat` | `write_legacy_wide_magnitudes_dat` (later removed; ECSV only) |
| `differential_calibrated_to_legacy_table()` | `calibrated_epochs_to_legacy_wide_table()` (later removed) |
| `mk_magnitudes_table` / `save_calibration` / `save_magnitudes_ascii` | `write_epoch_native_magnitudes` |
| `zp_method` (`linear` / `median` / `auto`) | `color_term_fit` (`always` / `never` / `auto`) |
| `extinction_mode="fitted"` | `extinction_mode="from_comparison_stars"` |
| `fit_extinction_from_data` | `extinction_mode` (`from_comparison_stars` or `from_value_airmass`) |
| `skip_extinction_fit` | `extinction_mode="from_value_airmass"` runs `ExtinctionFitStep` |
| (implicit first-order only) | `extinction_order` (`first` / `second`) + optional `k_second` overrides |
| `tabulated` (builtin only) | `tabulated` + site JSON; see [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) |
| `Observation.extract_flux` / `extract_flux_multi` | `Observation.run_pipeline` |
| `analyze.calibration_data` / `derive_calibration` / `CalibParameters` | `CalibrationStep` + `calibration_sources`; protect via `correlate.protection` |
| `standard_catalog_to_legacy` / `legacy_adapter` | Standard schema only (`ra`/`dec`/`mag_std_*`) |
| `select_calibration_objects` / `correlate_with_calibration_objects` | Catalog match in `CalibrationStep` / `crossmatch_standard_catalog` |

### Script pattern (N2 / C7)

Both supervisor (N2) and student (C7) flux scripts support `calibration_config_mode`:

```python
calibration_config_mode = "preset"   # or "custom"
calibration_preset = "median_zp_per_image"      # or "linear_fit_per_night", "linear_fit_per_night_extinction"

# custom mode:
calibration_strategy = "median_zp"   # or "linear_fit"
calibration_grouping = "per_image"   # per_image | per_night | ensemble | fixed
extinction_mode = "none"             # none | tabulated | from_comparison_stars | from_value_airmass
color_term_fit = "never"             # always | auto | never  (linear_fit only)
derive_transform_from_data = False   # catalog-color derive-transform (linear_fit, 2 filters)
```

| `extinction_mode` | ExtinctionFitStep | Calibration coefficients |
|-------------------|-------------------|--------------------------|
| `none` | skipped | no extinction correction |
| `tabulated` | skipped | bundled/custom site JSON (`path_extinction_coefficients`) |
| `from_comparison_stars` | skipped | fit from catalog comparison stars in epochs |
| `from_value_airmass` | runs | fit from flux/magnitude vs airmass |

### Derive-transform (`derive_transform_from_data`)

Within `linear_fit`, set `derive_transform_from_data=True` for the catalog-color
derive-transform (two filters only, e.g. B and V). Preset `linear_fit_per_night` enables
this. Otherwise `color_term_fit` controls the standard `PhotometryCalibrator` fit.

QC plots (when `output_dir` is set) go under `<output>/diagnostics/calibration/`:

- `derive_transform_<epoch>_<filter>.pdf` — fit of `m_std−m_inst` vs catalog color
  (slopes used to build applied `c` factors), with residuals; gray = outliers rejected
  via `fit_sigma_clip`
- `derive_transform_fit_overview_<filters>.pdf` — catalog-color T/ZP, fit RMS, and
  n_stars vs epoch
- `derive_transform_summary_<filters>.pdf` — applied `c` and median ZP across epochs

### Tests

```bash
pytest tests/test_calibration_comparison.py tests/test_pipeline_config.py tests/test_extinction_io.py -m comparison
```

---

## 2. Calibration epochs

The calibration bridge builds **calibration epochs** — one multi-band table per
matched set of exposures — with keys `epoch_000`, `epoch_001`, … (not `B_0`, `V_0`).

### API changes

| Before | After |
|--------|--------|
| `context.frames` | `context.calibration_epochs` |
| `observation_to_frame_tables(context)` | `observation_to_calibration_epochs(context, config)` |
| Table keys `"{filter}_{pd}"` | `epoch_{nnn}` |
| Single `airmass` column (typical) | `airmass_<filter>` per band + mean `airmass` |

`PipelineConfig` fields for exposure pairing:

- `exposure_pairing`: `"jd_nearest"` (default) or `"index"`
- `exposure_jd_tolerance`: JD match tolerance in days
- `reference_filter`: reference band for pairing (`None` → first filter)

Skipped pairings are listed in `context.calibration_epochs_skipped` and logged
from `CalibrationStep`.

### `PhotometryCalibrator.add_epoch`

- Tables contain `mag_<f>` / `err_<f>` for all filters (rows aligned by `id`).
- `airmass_<f>` may be pre-filled by the bridge; otherwise use `filter_obstimes`.

Epoch-native APIs: `add_epoch`, `epochs`, `epoch_metadata`, `fit_transformation_epoch`,
`fit_extinction_from_epochs`, column `epoch_id`. The alias `observation_to_frame_tables`
was removed.

### Scripts outside the pipeline

```python
from ost_photometry.analyze.pipeline import observation_to_calibration_epochs, PipelineConfig

observation_to_calibration_epochs(context, PipelineConfig(...))
epochs = context.calibration_epochs
meta = context.calibration_epoch_meta
```

---

## 3. Calibration catalog sources

Calibration catalog download and normalization live in
`ost_photometry.analyze.calibration_sources`. Standard schema:

- `ra` / `dec` (ICRS, degrees)
- `mag_std_{filter}` / `err_std_{filter}`
- Optional Sloan `mag_std_g/r/i` and Lupton Johnson `R`/`I`

### Breaking renames

| Removed / old | New |
|---------------|-----|
| `analyze.calibration_differential_catalog` | `analyze.differential_photometry` |
| `PipelineConfig.magnitude_range` | `calibration_catalog_mag_range` |
| `PhotometryCalibrator.setup_apass(...)` | `setup_calibration_source(...)` |
| `APASSCatalog` (public) | `fetch_standard_calibration_catalog` + `crossmatch_standard_catalog` |

`get_vizier_catalog` takes `center: SkyCoord` and `field_of_view_arcmin` (no
`image_like_object`). Lupton Sloan → Johnson R/I is in `calibration_sources.transforms`.

Consumers of the standard schema: `CalibrationStep` (cross-match),
`correlate.protection.resolve_calibration_object_ids` (protect calibrators),
and `PhotometryCalibrator.setup_calibration_source`.

---

## 4. Post-processing

### Magnitude systems (filter set + Vega/AB)

- Module: `post_processing.magnitude_systems` — catalog→system map, validation, meta keys
  (`ost_photometry.magnitude_system`, `ost_photometry.filter_set`, …).
- Config: `output_filter_set` (`auto`|`bessell`|`sdss`), `output_magnitude_system` (`auto`|`vega`|`ab`),
  `convert_magnitudes`. Deprecated: `target_filter_system` (`SDSS`|`AB`|`BESSELL`).
- Early validation in `AnalysisPipeline.run`; SDSS+Vega aborts. Calibration requires matching
  `mag_std_*` bands. Conversion: Vega↔AB offsets; Bessell→SDSS (Jordi); SDSS→Bessell (Lupton).
- Light-curve axis labels use table meta (no hardcoded “(Vega)”).

### API renames

- `utilities.post_process_results` → `post_process_cluster_field` (`post_processing`)
- `convert_magnitudes_to_other_system` expects epoch-native `mag_cal_<filter>` columns
- Light-curve helpers in `post_processing.light_curve`; `LightCurveStep` uses `prepare_plot_time_series`
- Pass `epoch_meta=context.calibration_epoch_meta` for epoch-native light curves

### Output format

- Primary output: **ECSV** (`*.ecsv`), schema `ost_photometry.epoch_native.v1`
- Old legacy wide `.dat` files can still be **read** via `legacy_wide_table_to_epoch_native`

### Pipeline configuration

- `skip_post_process` removed; use granular `skip_cluster_*`, `skip_magnitude_convert_step`, etc.
- When `skip_calibration=True`, cluster post-process steps skip (extinction-only runs)

### Pipeline steps

`PostProcessStep` was split into: `PostProcessRegionStep`, `PostProcessClusterGaiaStep`,
`PostProcessProperMotionStep`, `PostProcessMagnitudeConvertStep`, `PostProcessSaveMagnitudesStep`,
`DeriveLimitingMagnitudeStep`, `LightCurveStep`. Optional Simbad overlay on reference
images is `SimbadAnnotateStep` (after extraction; `annotate_image` /
`annotate_reference_image`).

Cluster helpers require `plot_context=` (`ImagingPlotContext`). Multi-epoch tables:
region/Gaia/PM on first `epoch_id`, then expand surviving `id` to all epochs.

### Adapters

`legacy_wide_table_to_epoch_native` / `ensure_epoch_native_photometry_table` convert
between legacy wide rows and epoch-native long form.
