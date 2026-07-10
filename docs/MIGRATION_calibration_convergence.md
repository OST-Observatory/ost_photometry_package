# Migration: calibration convergence (legacy + differential → CalibrationEngine)

## Summary

Legacy (`calibration.py` + `CalibrationDataStep` / `CalibrationApplyStep`) and differential
(`PhotometryCalibrator` + `DifferentialCalibrationStep`) paths are converging on a single
**epoch-native** workflow:

- **Data:** `context.calibration_epochs` via `observation_to_calibration_epochs`
- **Fit / apply:** `CalibrationEngine.fit()` / `.apply()` with strategy backends
- **Pipeline step:** `CalibrationStep` (replaces the three legacy calibration steps)
- **Configuration:** `calibration_strategy`, `calibration_grouping`, `extinction_mode` + presets

## Presets

| Preset | strategy | grouping | extinction_mode | Typical use |
|--------|----------|----------|-----------------|-------------|
| `n2_stack` | `median_zp` | `per_image` | `none` | Stacked B/V, clusters (N2) |
| `c7_variable` | `linear_fit` | `per_night` | `none` | Multi-epoch light curves (C7) |
| `c7_variable_extinction` | `linear_fit` | `per_night` | `fitted` | Variables with airmass spread |

```python
from ost_photometry.analyze.pipeline import PipelineConfig

config = PipelineConfig.from_preset("c7_variable", overrides={"fit_sigma_clip": 3.0})
```

## Field mapping

| Old (`PipelineConfig`) | New | Notes |
|------------------------|-----|-------|
| `calibration_module="legacy"` | `strategy=median_zp`, `grouping=per_image`, `extinction_mode=none` | Deprecated alias |
| `calibration_module="differential"` | `strategy=linear_fit`, `grouping=per_night`, `extinction_mode=tabulated` | Until presets override |
| `differential_coefficient_mode` | `calibration_grouping` | Same values |
| `differential_extinction_order` | `extinction_mode` | `none`/`first`/`second` → `none`/`tabulated`/`tabulated` |
| `differential_fit_sigma_clip` | `fit_sigma_clip` | |
| `differential_exposure_pairing` | `exposure_pairing` | |
| `differential_exposure_jd_tolerance` | `exposure_jd_tolerance` | |
| `differential_reference_filter` | `reference_filter` | |
| `differential_color_indices` | `color_indices` | |
| `calculate_zero_point_statistic` | `zp_subsample_statistic` | MC subsample QC |
| `derive_transformation_coefficients` | `derive_transform_from_data` | |
| `write_differential_legacy_magnitudes_dat` | `write_legacy_wide_magnitudes_dat` | |

## Context

| Before | After |
|--------|-------|
| `context.differential_calib_parameters` | `context.calibration_results` (preferred) |
| `obs.calib_parameters` (legacy `CalibParameters`) | Read-only alias; deprecated |

## C7 / N2 recommendations

**C7 (variable stars):** start with `from_preset("c7_variable")` — linear T/ZP per night,
no extinction correction by default. Set `zp_method="median"` if legacy median ZP is closer to
your reference runs.

**N2 (stacked cluster):** use `from_preset("n2_stack")` — median zero point per image,
no extinction, matches classic legacy behaviour on stacked frames.

## Comparison tests

Run comparison harness:

```bash
pytest tests/test_calibration_comparison.py -m comparison
```

Tests cover synthetic T/ZP agreement, median vs linear ZP, and extinction `none` vs constant
airmass (should be identical when coefficients are unused).
