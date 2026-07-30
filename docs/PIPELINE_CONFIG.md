# Pipeline configuration

Analysis scripts pass a `[PipelineConfig](../src/ost_photometry/analyze/pipeline/config.py)` to `Observation.run_pipeline()`. The options below are the main **typed choices** that control WCS, extraction, correlation, and calibration. They are independent of the N2/C7 course scripts; use this guide to pick settings for any field, survey, or lab run.

For breaking renames and named presets (`median_zp_per_image`, `linear_fit_per_night`, …) see [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md).

## Quick start

```python
from ost_photometry.analyze.pipeline import PipelineConfig

# Named recipe (calibration fields only; other steps keep defaults)
config = PipelineConfig.from_preset("linear_fit_per_night")

# Or build explicitly
config = PipelineConfig(
    wcs_method="astrometry",
    photometry_extraction_method="PSF",
    correlation_method="astropy",
    calibration_strategy="linear_fit",
    calibration_grouping="per_night",
    extinction_mode="none",
    color_term_fit="auto",
)
```



## Option reference


| Config field                   | Values                                                             | Pipeline step                   | Role                                                            |
| ------------------------------ | ------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------- |
| `wcs_method`                   | `astrometry`, `astap`, `twirl`                                     | WCS                             | Astrometric solution per image                                  |
| `photometry_extraction_method` | `PSF`, `APER`                                                      | Extraction                      | PSF fitting vs aperture photometry                              |
| `correlation_method`           | `astropy`, `own`                                                   | Correlation (intra/inter)       | Match detections across exposures                               |
| `protected_object_ids`         | list of row indices                                                | Correlation (intra/inter)       | Explicit reference-image rows to keep (any object type)         |
| `protect_ooi`                  | `True`, `False`                                                    | Correlation (intra/inter)       | Auto-add identified objects of interest to the protected set    |
| `protect_calibration_objects`  | `True`, `False`                                                    | Correlation (intra/inter)       | Auto-add catalog-matched calibration stars to the protected set |
| `calibration_strategy`         | `median_zp`, `linear_fit`                                          | Calibration                     | ZP-only vs linear T/ZP fit                                      |
| `calibration_grouping`         | `per_image`, `per_night`, `ensemble`, `fixed`                      | Calibration                     | How coefficients are shared across epochs                       |
| `extinction_mode`              | `none`, `tabulated`, `from_comparison_stars`, `from_value_airmass` | Extinction fit + calibration    | Whether/how to correct for airmass                              |
| `extinction_order`             | `first`, `second`                                                  | Calibration                     | Apply k′ only, or also k″ (tabulated / `k_second`)              |
| `k_second`                     | `{filter: float}` or `None`                                        | Calibration                     | Optional per-filter k″ overrides                                |
| `color_term_fit`               | `always`, `auto`, `never`                                          | Calibration (`linear_fit` only) | Color-term handling in linear fit                               |
| `uncertainty_mode`             | `fit_errors`, `flux_monte_carlo`, `both`                           | Calibration                     | Propagated `err_cal_*` after calibration apply                  |




### `wcs_method`


| Value        | When to use                                    | Requirements / notes                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| `astrometry` | Default; best accuracy when it converges       | Local [astrometry.net](https://nova.astrometry.net/) install |
| `astap`      | Fast blind solve; few stars; no astrometry.net | ASTAP binary on `PATH`                                       |
| `twirl`      | Small fields with Gaia-visible stars           | Internet for catalog query                                   |


WCS must succeed (or be copied from another filter) before extraction and correlation. `hips_reference_subtraction_wcs_method` can override `wcs_method` for the optional HiPS subtraction step only.

### `photometry_extraction_method`


| Value  | When to use                                        | Notes                                                                          |
| ------ | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `PSF`  | Crowded fields, variable seeing, course default    | Needs enough PSF stars (`minimum_n_eps_stars`–`maximum_n_eps_stars`); ePSF path in single-image mode |
| `APER` | Very sparse fields, quick checks, extended sources | Set `radius_aperture`, annulus radii; no ePSF build                            |


Both methods feed the same downstream tables (`mag_<filter>`, `err_<filter>`). Choice does not change calibration strategy, but PSF is usually preferred when stars overlap.

### `correlation_method`


| Value     | When to use                                 | Notes                                       |
| --------- | ------------------------------------------- | ------------------------------------------- |
| `astropy` | Default; standard OST pipelines             | Duplicate handling: smallest sky separation |
| `own`     | Legacy / reproducibility with older scripts | Duplicate handling: first match in list     |


Requires a valid WCS. Affects intra-filter tracking (same object across exposures) and inter-filter matching (B with V on the same night). Tune `separation_limit`, `max_pixel_between_objects`, and `exposure_pairing` if matches fail.

**Protected objects during correlation** — sources are combined (deduplicated) into one set of reference-image row indices:

1. `protected_object_ids` — explicit list (any science case)
2. `protect_ooi=True` — objects of interest after identification (C7 variables)
3. `protect_calibration_objects=True` — catalog-matched comparison stars (mk_calib, extinction fields)

All three can be active at once, e.g. a variable star in a field that also needs calibration stars preserved:

```python
PipelineConfig(
    protect_ooi=True,
    protect_calibration_objects=True,
    protected_object_ids=[42],  # optional extra row, e.g. a known comparison star
)
```

Intra correlation uses `correlate_preserve_objects`; inter correlation resolves IDs on the reference filter before matching across bands.

**Rename (July 2026):** `protect_reference_obj` was renamed to `protect_ooi`. The old name still works as a `PipelineConfig` alias.

### `calibration_strategy` and `calibration_grouping`

**Strategy** — what is fitted:


| `calibration_strategy` | Fits                           | Typical science case                                              |
| ---------------------- | ------------------------------ | ----------------------------------------------------------------- |
| `median_zp`            | Zero point per epoch (T = 0)   | Single epoch or stacked images; cluster colour–magnitude diagrams |
| `linear_fit`           | Color term T and zero point ZP | Multi-epoch light curves; passband mismatch matters               |


**Grouping** — how epochs share coefficients (applies to both strategies):


| `calibration_grouping` | Behaviour                                      | When to use                                                             |
| ---------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| `per_image`            | One coefficient set per exposure epoch         | Each image calibrated independently (e.g. one stacked frame per filter) |
| `per_night`            | One set per night (epochs combined before fit) | Multi-epoch variable stars on the same night                            |
| `ensemble`             | Single set from all epochs stacked             | Stable transform from many nights; assumes constant instrument response |
| `fixed`                | No fit; use `transformation_coefficients_dict` | Known T/ZP from external calibration                                    |


`median_zp` ignores `color_term_fit`. `linear_fit` with `per_image` can smooth night-to-night drift via `per_image_rolling_*` options.

**Related (not in the Literal list but coupled):**

- `derive_transform_from_data` — only with `linear_fit`, exactly **two** filters; alternative to `PhotometryCalibrator` linear fit (catalog-color slopes + median ZP). Preset `linear_fit_per_night` enables this. Incompatible with using `color_term_fit` on the standard calibrator path (derive path bypasses it). Writes QC under `<output>/calibration/`: `derive_transform_<epoch>_<filter>.*` (catalog-color slope fits), `derive_transform_fit_overview_*.*` (T/ZP/RMS/n vs epoch), and `derive_transform_summary_*.*` (applied `c`/ZP vs epoch). Outlier rejection uses `fit_sigma_clip` (default `2.5`; lower = stricter) on both `zp_sum` and fit residuals — gray points on the QC plots are excluded stars.
- `exposure_pairing` (`jd_nearest` / `index`) and `reference_filter` — build multi-band epochs before calibration; use `jd_nearest` when B and V exposures are not strictly paired by index.
- `zp_subsample_statistic` — extra ZP stability reporting for `median_zp` only.



### `diagnostic_plots`


Most QC figures under `<output>/diagnostics/` are **on by default** (growth curves stay off). Toggle via nested overrides, e.g. `diagnostic_plots__photometry_mag_vs_error_scatter=False`.

| Flag | What it checks |
|------|----------------|
| `calibration_crossmatch_separation_histogram` | Catalog match separations |
| `combined_separation_histograms` | Combined separation panels |
| `photometry_mag_vs_error_scatter` | Mag vs photometric error (reference image) |
| `photometry_radial_growth_curve` | Aperture growth for brightest star (off by default) |
| `correlation_inter_filter_separation_plot` | Inter-filter match separations: reference-image pair, up to 25 exposure pairs (same `exposure_pairing` as calibration), plus overview (all pairs) |
| `calibration_instrumental_vs_catalog` | Instrumental vs catalog magnitudes |
| `calibration_zeropoint_residual_histogram` | ZP residual distribution |
| `calibration_zeropoint_residual_vs_color` | ZP residuals vs color |
| `calibration_color_check_cal_stars` | Color–color check of comparison stars |

Standard `linear_fit` (non-derive) also writes transformation panels under `<output>/calibration/` (`calibration_<epoch>_<filter>.*`, night/per-image summaries). Those are separate from `diagnostic_plots`.


### `extinction_mode`


| Value                   | `ExtinctionFitStep` | Coefficients in calibration                                                                                  | When to use                                                                              |
| ----------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `none`                  | skipped             | no correction                                                                                                | Short runs, low airmass range, or extinction absorbed in ZP drift                        |
| `tabulated`             | skipped             | bundled site JSON (`path_extinction_coefficients` or package default) + builtin fallback for missing filters | Routine science with maintained OST k′ table                                             |
| `from_comparison_stars` | skipped             | fit from catalog stars across epochs                                                                         | Multi-epoch data with airmass span and enough comparison stars (≥ 3 epochs, spread in X) |
| `from_value_airmass`    | **runs**            | fit from mag/flux vs airmass, then passed to calibration                                                     | Dedicated extinction fields (same stars over several hours)                              |


Requires `observatory_location` (or per-epoch airmass columns) for any mode except `none`. `from_comparison_stars` needs `linear_fit` and catalog cross-match (`mag_std_*`). `from_value_airmass` runs before `CalibrationStep` and writes `extinction_coefficients.json`.

**Related config:**

| Field | Role |
|-------|------|
| `path_extinction_coefficients` | Custom site JSON for `tabulated` (default: bundled `ost_potsdam_extinction.json`) |
| `extinction_order` | `first` (default) or `second` — whether to apply k″ as well as k′ when extinction is enabled |
| `k_second` | Optional `{filter: k''}` overrides (mag/airmass/mag_color); applied on top of tabulated / fitted coefficients |

With `extinction_order="second"`, k″ comes from the site table (or builtin defaults) when the night fit only determined k′; user `k_second` always wins. Color indices for the k″ term use `color_indices` when set, otherwise the tabulated / default color pair. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md).

### `extinction_order`


| Value | Behaviour |
|-------|-----------|
| `first` | Correct `m_0 = m − k'·X` only (default) |
| `second` | Also apply `k''·X·(color)` using tabulated and/or `k_second` overrides |

Ignored when `extinction_mode="none"`. Example:

```python
PipelineConfig(
    extinction_mode="tabulated",
    extinction_order="second",
    k_second={"B": 0.03, "V": 0.01},  # optional override
)
```

### `color_term_fit` (`linear_fit` only)


| Value    | Behaviour                                                  |
| -------- | ---------------------------------------------------------- |
| `never`  | Median ZP, T = 0; no extinction correction before ZP       |
| `always` | Always fit T and ZP when color columns exist               |
| `auto`   | Linear fit if catalog color spread > 0.1 mag, else ZP-only |


Use `never` for single-filter data or when colors are unreliable. Use `auto` as the general default for multi-filter work. Use `always` when you need a color term even with narrow color range (may be poorly constrained).

### `uncertainty_mode`

Applied by :class:`~ost_photometry.analyze.pipeline.steps.calibration.CalibrationStep` after calibrated magnitudes are written.


| Value              | Behaviour                                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit_errors`       | Default. `err_cal_*` from the linear fit / derive-transform apply path (instrumental + T/ZP propagation).                                            |
| `flux_monte_carlo` | Replace `err_cal_*` with Monte Carlo spread from `flux_<filter>` / `flux_err_<filter>` (+ ZP). Falls back to fit errors if flux columns are missing. |
| `both`             | Combine fit and MC errors in quadrature: `sqrt(err_fit² + err_mc²)`.                                                                                 |


Uses `distribution_samples` (default 1000) for the MC draw count. Requires flux columns in epoch tables (populated by the extraction bridge when PSF/aperture flux is available).

## Decision tables



### 1. What kind of observation do you have?


| Your situation                                        | Suggested starting point                                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| One or few images per filter, stacked or single epoch | `median_zp` + `per_image` + `extinction_mode="none"`                                              |
| Multi-epoch light curve, same filters each night      | `linear_fit` + `per_night` + `color_term_fit="auto"`                                              |
| Multi-epoch, large airmass range, many cal stars      | add `extinction_mode="from_comparison_stars"`                                                     |
| Dedicated extinction field (hours-long star trail)    | `extinction_mode="from_value_airmass"`                                                            |
| Two filters, legacy C7-style derive transform         | `linear_fit` + `derive_transform_from_data=True` (2 filters only)                                 |
| Known T/ZP from elsewhere                             | `linear_fit` or `median_zp` + `calibration_grouping="fixed"` + `transformation_coefficients_dict` |




### 2. Calibration strategy × grouping


|              | `per_image`                                | `per_night`                     | `ensemble`              | `fixed`           |
| ------------ | ------------------------------------------ | ------------------------------- | ----------------------- | ----------------- |
| `median_zp`  | One ZP per exposure (e.g. stacked fields)  | One ZP per night                | One ZP for all data     | Apply preset ZP   |
| `linear_fit` | T/ZP per exposure; rolling smooth optional | One T/ZP per night (default for time series) | One T/ZP for all nights | Apply preset T/ZP |




### 3. Extinction choice


| Airmass span | Comparison stars in field? | Dedicated extinction observations? | Choose                    |
| ------------ | -------------------------- | ---------------------------------- | ------------------------- |
| small        | —                          | —                                  | `none`                    |
| any          | no                         | no                                 | `tabulated` (approximate) |
| large        | yes, multi-epoch           | no                                 | `from_comparison_stars`   |
| large        | —                          | yes (same stars, many X)           | `from_value_airmass`      |




### 4. Extraction and correlation (upstream of calibration)


| Condition                           | `photometry_extraction_method` | `correlation_method` | Other                                                   |
| ----------------------------------- | ------------------------------ | -------------------- | ------------------------------------------------------- |
| Stellar PSF, course / OST default   | `PSF`                          | `astropy`            | Ensure `minimum_n_eps_stars` met; dense fields capped by `maximum_n_eps_stars` (default 50; `None` = no cap) |
| Quick test, very sparse field       | `APER`                         | `astropy`            | Widen aperture if SNR low                               |
| Reproduce pre-2024 script behaviour | `PSF`                          | `own`                | Check `duplicate_handling_object_identification`        |
| B/V not aligned by index            | either                         | `astropy`            | `exposure_pairing="jd_nearest"`, set `reference_filter` |




## Valid combinations and constraints


| Constraint                                | Details                                                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `derive_transform_from_data=True`         | Only with `calibration_strategy="linear_fit"` and exactly 2 filters; bypasses standard `color_term_fit` path |
| `color_term_fit`                          | Only affects `linear_fit`; ignored by `median_zp`                                                            |
| `extinction_mode="from_comparison_stars"` | Meaningful with `linear_fit`, multi-epoch data, and catalog matches                                          |
| `extinction_mode="from_value_airmass"`    | Runs `ExtinctionFitStep`; calibration should use `linear_fit` to consume coefficients                        |
| `calibration_grouping="fixed"`            | Requires `transformation_coefficients_dict` (set on calibrator via advanced API; pipeline field exists)      |
| `calibration_grouping="ensemble"`         | All epochs must be combinable; poor if instrument response drifts between nights                             |
| WCS failure                               | Extraction and correlation cannot proceed; try `astap` or `twirl`, or `force_wcs_determination`              |




## Named presets (shortcuts)


| Preset                   | Key settings                                                                                                 | Typical use                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `median_zp_per_image`               | `median_zp`, `per_image`, `extinction_mode="none"`                                                           | Stacked multi-filter fields / cluster photometry                                                     |
| `linear_fit_per_night`            | `linear_fit`, `per_night`, `derive_transform_from_data=True`                                                 | Multi-epoch light curves                                                            |
| `linear_fit_per_night_extinction` | `linear_fit`, `per_night`, `from_comparison_stars`                                                           | Light curves with significant airmass range                                                      |
| `extract_protect_calibrators` | `protect_calibration_objects=True`, `skip_calibration`, `skip_correlation_inter` | Extract + intra-correlate while protecting catalog calibrators (no apply) |
| `linear_fit_ensemble` | `linear_fit`, `derive_transform_from_data=True`, `calibration_grouping="ensemble"` | Single ensemble transform / derive-transform over all epochs |
| `tabulated_extinction` | `extinction_mode="tabulated"`, bundled site JSON when path is `None` | Apply a site extinction table |


Field transformation output: legacy ASCII `trans_para_<field>.dat` (unchanged column names) plus JSON sidecar `trans_para_<field>.json` with structured coefficients. Second-order scripts accept either format.

```python
config = PipelineConfig.from_preset("linear_fit_per_night", overrides={"fit_sigma_clip": 3.0})
config_mk = PipelineConfig.from_preset("extract_protect_calibrators", overrides={"calibration_source": "APASS"})
```



## Further reading

- [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) — site extinction table and dedicated-night best practices
- [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md) — breaking changes and epoch-native architecture
- [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) — site extinction table
- Source of truth for defaults: `[pipeline/config.py](../src/ost_photometry/analyze/pipeline/config.py)`

