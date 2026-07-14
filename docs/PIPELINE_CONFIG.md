# Pipeline configuration

Analysis scripts pass a [`PipelineConfig`](../src/ost_photometry/analyze/pipeline/config.py) to `Observation.run_pipeline()`. The options below are the main **typed choices** that control WCS, extraction, correlation, and calibration. They are independent of the N2/C7 course scripts; use this guide to pick settings for any field, survey, or lab run.

For breaking renames and course presets (`n2_stack`, `c7_variable`, …) see [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md).

## Quick start

```python
from ost_photometry.analyze.pipeline import PipelineConfig

# Named recipe (calibration fields only; other steps keep defaults)
config = PipelineConfig.from_preset("c7_variable")

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

| Config field | Values | Pipeline step | Role |
|--------------|--------|---------------|------|
| `wcs_method` | `astrometry`, `astap`, `twirl` | WCS | Astrometric solution per image |
| `photometry_extraction_method` | `PSF`, `APER` | Extraction | PSF fitting vs aperture photometry |
| `correlation_method` | `astropy`, `own` | Correlation (intra/inter) | Match detections across exposures |
| `calibration_strategy` | `median_zp`, `linear_fit` | Calibration | ZP-only vs linear T/ZP fit |
| `calibration_grouping` | `per_image`, `per_night`, `ensemble`, `fixed` | Calibration | How coefficients are shared across epochs |
| `extinction_mode` | `none`, `tabulated`, `from_comparison_stars`, `from_value_airmass` | Extinction fit + calibration | Whether/how to correct for airmass |
| `color_term_fit` | `always`, `auto`, `never` | Calibration (`linear_fit` only) | Color-term handling in linear fit |
| `uncertainty_mode` | `fit_errors`, `flux_monte_carlo`, `both` | *(reserved)* | Defined in config; not yet applied by pipeline steps |

### `wcs_method`

| Value | When to use | Requirements / notes |
|-------|-------------|----------------------|
| `astrometry` | Default; best accuracy when it converges | Local [astrometry.net](https://nova.astrometry.net/) install |
| `astap` | Fast blind solve; few stars; no astrometry.net | ASTAP binary on `PATH` |
| `twirl` | Small fields with Gaia-visible stars | Internet for catalog query |

WCS must succeed (or be copied from another filter) before extraction and correlation. `hips_reference_subtraction_wcs_method` can override `wcs_method` for the optional HiPS subtraction step only.

### `photometry_extraction_method`

| Value | When to use | Notes |
|-------|-------------|-------|
| `PSF` | Crowded fields, variable seeing, course default | Needs enough PSF stars (`minimum_n_eps_stars`); ePSF path in single-image mode |
| `APER` | Very sparse fields, quick checks, extended sources | Set `radius_aperture`, annulus radii; no ePSF build |

Both methods feed the same downstream tables (`mag_<filter>`, `err_<filter>`). Choice does not change calibration strategy, but PSF is usually preferred when stars overlap.

### `correlation_method`

| Value | When to use | Notes |
|-------|-------------|-------|
| `astropy` | Default; standard OST pipelines | Duplicate handling: smallest sky separation |
| `own` | Legacy / reproducibility with older scripts | Duplicate handling: first match in list |

Requires a valid WCS. Affects intra-filter tracking (same object across exposures) and inter-filter matching (B with V on the same night). Tune `separation_limit`, `max_pixel_between_objects`, and `exposure_pairing` if matches fail.

### `calibration_strategy` and `calibration_grouping`

**Strategy** — what is fitted:

| `calibration_strategy` | Fits | Typical science case |
|------------------------|------|----------------------|
| `median_zp` | Zero point per epoch (T = 0) | Single epoch or stacked images; cluster colour–magnitude diagrams |
| `linear_fit` | Color term T and zero point ZP | Multi-epoch light curves; passband mismatch matters |

**Grouping** — how epochs share coefficients (applies to both strategies):

| `calibration_grouping` | Behaviour | When to use |
|------------------------|-----------|-------------|
| `per_image` | One coefficient set per exposure epoch | Each image calibrated independently (e.g. one stacked frame per filter) |
| `per_night` | One set per night (epochs combined before fit) | Multi-epoch variable stars on the same night |
| `ensemble` | Single set from all epochs stacked | Stable transform from many nights; assumes constant instrument response |
| `fixed` | No fit; use `transformation_coefficients_dict` | Known T/ZP from external calibration |

`median_zp` ignores `color_term_fit`. `linear_fit` with `per_image` can smooth night-to-night drift via `per_image_rolling_*` options.

**Related (not in the Literal list but coupled):**

- `derive_transform_from_data` — only with `linear_fit`, exactly **two** filters; alternative to `PhotometryCalibrator` linear fit (catalog-color slopes + median ZP). Preset `c7_variable` enables this. Incompatible with using `color_term_fit` on the standard calibrator path (derive path bypasses it).
- `exposure_pairing` (`jd_nearest` / `index`) and `reference_filter` — build multi-band epochs before calibration; use `jd_nearest` when B and V exposures are not strictly paired by index.
- `zp_subsample_statistic` — extra ZP stability reporting for `median_zp` only.

### `extinction_mode`

| Value | `ExtinctionFitStep` | Coefficients in calibration | When to use |
|-------|-------------------|----------------------------|-------------|
| `none` | skipped | no correction | Short runs, low airmass range, or extinction absorbed in ZP drift |
| `tabulated` | skipped | bundled site JSON (`path_extinction_coefficients` or package default) + builtin fallback for missing filters | Routine science with maintained OST k′ table |
| `from_comparison_stars` | skipped | fit from catalog stars across epochs | Multi-epoch data with airmass span and enough comparison stars (≥ 3 epochs, spread in X) |
| `from_value_airmass` | **runs** | fit from mag/flux vs airmass, then passed to calibration | Dedicated extinction fields (same stars over several hours) |

Requires `observatory_location` (or per-epoch airmass columns) for any mode except `none`. `from_comparison_stars` needs `linear_fit` and catalog cross-match (`mag_std_*`). `from_value_airmass` runs before `CalibrationStep` and writes `extinction_coefficients.json`.

**Related config:** `path_extinction_coefficients` — custom site JSON for `tabulated` (default: bundled `ost_potsdam_extinction.json`). See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) for maintaining the site table.

### `color_term_fit` (`linear_fit` only)

| Value | Behaviour |
|-------|-----------|
| `never` | Median ZP, T = 0; no extinction correction before ZP |
| `always` | Always fit T and ZP when color columns exist |
| `auto` | Linear fit if catalog color spread > 0.1 mag, else ZP-only |

Use `never` for single-filter data or when colors are unreliable. Use `auto` as the general default for multi-filter work. Use `always` when you need a color term even with narrow color range (may be poorly constrained).

### `uncertainty_mode`

Values `fit_errors`, `flux_monte_carlo`, and `both` are reserved for propagated calibration uncertainties (`calibration/uncertainty.py`). The default pipeline steps do not branch on this field yet; leave at `fit_errors` unless you call the uncertainty helpers directly.

## Decision tables

### 1. What kind of observation do you have?

| Your situation | Suggested starting point |
|----------------|-------------------------|
| One or few images per filter, stacked or single epoch | `median_zp` + `per_image` + `extinction_mode="none"` |
| Multi-epoch light curve, same filters each night | `linear_fit` + `per_night` + `color_term_fit="auto"` |
| Multi-epoch, large airmass range, many cal stars | add `extinction_mode="from_comparison_stars"` |
| Dedicated extinction field (hours-long star trail) | `extinction_mode="from_value_airmass"` |
| Two filters, legacy C7-style derive transform | `linear_fit` + `derive_transform_from_data=True` (2 filters only) |
| Known T/ZP from elsewhere | `linear_fit` or `median_zp` + `calibration_grouping="fixed"` + `transformation_coefficients_dict` |

### 2. Calibration strategy × grouping

| | `per_image` | `per_night` | `ensemble` | `fixed` |
|---|-------------|-------------|------------|---------|
| **`median_zp`** | One ZP per exposure (N2 stacks) | One ZP per night | One ZP for all data | Apply preset ZP |
| **`linear_fit`** | T/ZP per exposure; rolling smooth optional | One T/ZP per night (C7 default) | One T/ZP for all nights | Apply preset T/ZP |

### 3. Extinction choice

| Airmass span | Comparison stars in field? | Dedicated extinction observations? | Choose |
|--------------|---------------------------|-----------------------------------|--------|
| small | — | — | `none` |
| any | no | no | `tabulated` (approximate) |
| large | yes, multi-epoch | no | `from_comparison_stars` |
| large | — | yes (same stars, many X) | `from_value_airmass` |

### 4. Extraction and correlation (upstream of calibration)

| Condition | `photometry_extraction_method` | `correlation_method` | Other |
|-----------|----------------------------------|--------------------|-------|
| Stellar PSF, course / OST default | `PSF` | `astropy` | Ensure `minimum_n_eps_stars` met |
| Quick test, very sparse field | `APER` | `astropy` | Widen aperture if SNR low |
| Reproduce pre-2024 script behaviour | `PSF` | `own` | Check `duplicate_handling_object_identification` |
| B/V not aligned by index | either | `astropy` | `exposure_pairing="jd_nearest"`, set `reference_filter` |

## Valid combinations and constraints

| Constraint | Details |
|------------|---------|
| `derive_transform_from_data=True` | Only with `calibration_strategy="linear_fit"` and exactly 2 filters; bypasses standard `color_term_fit` path |
| `color_term_fit` | Only affects `linear_fit`; ignored by `median_zp` |
| `extinction_mode="from_comparison_stars"` | Meaningful with `linear_fit`, multi-epoch data, and catalog matches |
| `extinction_mode="from_value_airmass"` | Runs `ExtinctionFitStep`; calibration should use `linear_fit` to consume coefficients |
| `calibration_grouping="fixed"` | Requires `transformation_coefficients_dict` (set on calibrator via advanced API; pipeline field exists) |
| `calibration_grouping="ensemble"` | All epochs must be combinable; poor if instrument response drifts between nights |
| WCS failure | Extraction and correlation cannot proceed; try `astap` or `twirl`, or `force_wcs_determination` |

## Course presets (shortcuts)

| Preset | Key settings | Typical use |
|--------|--------------|-------------|
| `n2_stack` | `median_zp`, `per_image`, `extinction_mode="none"` | Supervisor cluster exercise (stacked B/V) |
| `c7_variable` | `linear_fit`, `per_night`, `derive_transform_from_data=True` | Student variable-star light curves |
| `c7_variable_extinction` | `linear_fit`, `per_night`, `from_comparison_stars` | Variables with significant airmass range |

```python
config = PipelineConfig.from_preset("c7_variable", overrides={"fit_sigma_clip": 3.0})
```

## Further reading

- [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) — site extinction table and dedicated-night best practices
- [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md) — breaking changes and epoch-native architecture
- [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) — site extinction table
- Source of truth for defaults: [`pipeline/config.py`](../src/ost_photometry/analyze/pipeline/config.py)
