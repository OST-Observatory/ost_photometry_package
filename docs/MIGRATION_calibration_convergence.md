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
| `c7_variable_extinction` | `linear_fit` | `per_night` | `from_comparison_stars` | Variables with airmass spread |

## Removed (breaking)

| Removed | Replacement |
|---------|-------------|
| `calibration_module` | `calibration_strategy` + `calibration_grouping` + `extinction_mode` |
| `differential_*` config fields | Neutral names (`fit_sigma_clip`, `exposure_pairing`, …) |
| `CalibrationDataStep`, `CalibrationApplyStep`, `DifferentialCalibrationStep` | `CalibrationStep` |
| `context.differential_calib_parameters` | `context.calibration_results` |
| `derive_transformation_coefficients` | `derive_transform_from_data` (within `linear_fit`, 2 filters) |
| `calculate_zero_point_statistic` | `zp_subsample_statistic` |
| `write_differential_legacy_magnitudes_dat` | `write_legacy_wide_magnitudes_dat` |
| `differential_calibrated_to_legacy_table()` | `calibrated_epochs_to_legacy_wide_table()` |
| `zp_method` (`linear` / `median` / `auto`) | `color_term_fit` (`always` / `never` / `auto`) |
| `extinction_mode="fitted"` | `extinction_mode="from_comparison_stars"` |
| `fit_extinction_from_data` | Use `extinction_mode` (`from_comparison_stars` or `from_value_airmass`) |
| `skip_extinction_fit` | `extinction_mode="from_value_airmass"` runs `ExtinctionFitStep`; otherwise skipped |
| `tabulated` (builtin only) | `tabulated` reads bundled/custom site JSON; see [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) |

## Script pattern (N2 / C7)

Both supervisor (N2) and student (C7) scripts support `calibration_config_mode`:

```python
calibration_config_mode = "preset"   # or "custom"
calibration_preset = "n2_stack"      # or "c7_variable", "c7_variable_extinction"

# custom mode:
calibration_strategy = "median_zp"   # or "linear_fit"
calibration_grouping = "per_image"   # per_image | per_night | ensemble | fixed
extinction_mode = "none"             # none | tabulated | from_comparison_stars | from_value_airmass
color_term_fit = "never"             # always | auto | never  (linear_fit only)
derive_transform_from_data = False   # catalog-color derive-transform (linear_fit, 2 filters)
```

`extinction_mode` controls both whether extinction is applied in calibration and how coefficients are obtained:

| Mode | ExtinctionFitStep | Calibration coefficients |
|------|-------------------|--------------------------|
| `none` | skipped | no extinction correction |
| `tabulated` | skipped | bundled/custom site JSON (`path_extinction_coefficients`) |
| `from_comparison_stars` | skipped | fit from catalog comparison stars in epochs |
| `from_value_airmass` | runs | fit from flux/magnitude vs airmass (`context.extinction_coefficients`) |

Preset `c7_variable_extinction` sets `from_comparison_stars` for multi-epoch variables with airmass spread.

## Derive-transform (`derive_transform_from_data`)

Within ``linear_fit``, set ``derive_transform_from_data=True`` to use the
catalog-color derive-transform (ported ``derive_transformation_onthefly``):
median ZP per filter plus differential color-term slopes fitted from catalog
colors. Requires exactly **two** filters (e.g. B and V). Preset ``c7_variable``
sets this flag to ``True``. Otherwise ``color_term_fit`` controls the standard
PhotometryCalibrator linear fit.

## C7 / N2 recommendations

**N2:** `from_preset("n2_stack")` or custom `median_zp` / `per_image` / `none`.

**C7:** `from_preset("c7_variable")`; try `color_term_fit="never"` if comparing to old legacy runs.

## Tests

```bash
pytest tests/test_calibration_comparison.py tests/test_pipeline_config.py -m comparison
```
