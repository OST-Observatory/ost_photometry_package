# Migration: calibration convergence (breaking release)

## Summary

Legacy (`calibration_module`, `CalibrationDataStep`, …) and differential-specific
config prefixes are removed. Use **one epoch-native calibration path**:

- **Step:** `CalibrationStep` + `CalibrationEngine`
- **Config:** `calibration_strategy`, `calibration_grouping`, `extinction_mode`, …
- **Presets:** `PipelineConfig.from_preset("n2_stack" | "c7_variable" | …)`
- **Results:** `context.calibration_results`

## Presets

| Preset | strategy | grouping | extinction_mode | Use case |
|--------|----------|----------|-----------------|----------|
| `n2_stack` | `median_zp` | `per_image` | `none` | Stacked B/V, clusters (N2) |
| `c7_variable` | `linear_fit` | `per_night` | `none` | Multi-epoch light curves (C7) |
| `c7_variable_extinction` | `linear_fit` | `per_night` | `fitted` | Variables with airmass spread |

## Removed (breaking)

| Removed | Replacement |
|---------|-------------|
| `calibration_module` | `calibration_strategy` + `calibration_grouping` + `extinction_mode` |
| `differential_*` config fields | Neutral names (`fit_sigma_clip`, `exposure_pairing`, …) |
| `CalibrationDataStep`, `CalibrationApplyStep`, `DifferentialCalibrationStep` | `CalibrationStep` |
| `context.differential_calib_parameters` | `context.calibration_results` |
| `derive_transformation_coefficients` | `derive_transform_from_data` |
| `calculate_zero_point_statistic` | `zp_subsample_statistic` |
| `write_differential_legacy_magnitudes_dat` | `write_legacy_wide_magnitudes_dat` |
| `differential_calibrated_to_legacy_table()` | `calibrated_epochs_to_legacy_wide_table()` |

## Script pattern (N2 / C7)

Both supervisor (N2) and student (C7) scripts support `calibration_config_mode`:

```python
calibration_config_mode = "preset"   # or "custom"
calibration_preset = "n2_stack"      # or "c7_variable", "c7_variable_extinction"

# custom mode:
calibration_strategy = "median_zp"   # or "linear_fit"
calibration_grouping = "per_image"   # per_image | per_night | ensemble | fixed
extinction_mode = "none"             # none | tabulated | fitted
zp_method = "median"                 # median | linear | auto
```

## C7 / N2 recommendations

**N2:** `from_preset("n2_stack")` or custom `median_zp` / `per_image` / `none`.

**C7:** `from_preset("c7_variable")`; try `zp_method="median"` if comparing to old legacy runs.

## Tests

```bash
pytest tests/test_calibration_comparison.py tests/test_pipeline_config.py -m comparison
```
