# Migration: `calibration_epochs` (multi-band differential pipeline)

## Summary

The differential calibration bridge no longer builds one table per filter per image with keys like `B_0`, `V_0`. It builds **calibration epochs** — one multi-band table per matched set of exposures — with keys `epoch_000`, `epoch_001`, …

## API changes

| Before | After |
|--------|--------|
| `context.frames` | `context.calibration_epochs` |
| `observation_to_frame_tables(context)` | `observation_to_calibration_epochs(context, config)` or `observation_to_epoch_tables(context)` |
| Table keys `"{filter}_{pd}"` | `epoch_{nnn}` |
| Single `airmass` column (typical) | `airmass_<filter>` per band + mean `airmass` |

New `PipelineConfig` fields:

- `differential_exposure_pairing`: `"jd_nearest"` (default) or `"index"`
- `differential_exposure_jd_tolerance`: JD match tolerance in days
- `differential_reference_filter`: reference band for pairing / row order (`None` → first filter in `filter_list`)

Skipped pairings (no epoch created) are listed in `context.calibration_epochs_skipped` and logged from `DifferentialCalibrationStep`.

## `PhotometryCalibrator.add_epoch`

- Tables should contain `mag_<f>` / `err_<f>` for all filters in the epoch (rows aligned by correlated `id`).
- `airmass_<f>` may be pre-filled by the bridge; otherwise use `filter_obstimes={"B": Time(...), ...}` or legacy `obstime` / `airmass`.

Differential-calibration APIs use **epoch** naming (`add_epoch`, `epochs`, `epoch_metadata`, `fit_transformation_epoch`, `fit_extinction_from_epochs`, table column `epoch_id`). `PhotometryCalibrator.fit_transformation_parameters` fits T/ZP; `get_calibrated_photometry` applies them. Use `observation_to_epoch_tables` (the old `observation_to_frame_tables` alias was removed).

## Scripts outside the pipeline

Call `observation_to_calibration_epochs` with a `PipelineConfig` instance and read `context.calibration_epochs` / `calibration_epoch_meta`.

`observation_to_epoch_tables` returns `dict(context.calibration_epochs)` (empty if epochs were never built).
