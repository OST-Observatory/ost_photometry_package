# Technical debt

Re-evaluated after the calibration-convergence and site-extinction work (July 2026).
Living docs: [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md).

**Forward-looking backlog:** [TODO.md](TODO.md) — intended to replace this file once the items below are largely closed.

**Priority legend:** P1 = worthwhile next sprint · P2 = track, no urgency · P3 = optional / long-term · — = operational note, not code debt

---

## Recommended next (P1)

| Item | Status | Recommendation |
|------|--------|----------------|
| ~~**Protect calibration stars in pipeline correlation**~~ | **Done** | `protect_calibration_objects` in `CorrelationConfig`; wired in `CorrelationIntraStep` / `CorrelationInterStep`; preset `extract_protect_calibrators`. |
| ~~**Migrate `mk_calib_photometry/2_mk_trans*.py` off legacy calibration**~~ | **Done** | `2_mk_trans*.py` use `CalibrationEngine` via `calibrate_mk_calib_filter_pair` / `write_field_transformation_table`; legacy `trans_para_*.dat` plus JSON sidecar. `3_second_order_extinction*.py` use `run_second_order_campaign`. |

---

## Pipeline / config (P2–P3)

| Item | Priority | Status | Notes |
|------|----------|--------|-------|
| `uncertainty_mode` not wired | P2 | **Done** | `CalibrationStep` calls `apply_uncertainty_mode_to_calibrated_table` (`calibration/uncertainty.py`). |
| ~~`tabulated_extinction` preset~~ | P3 | **Done** | `PipelineConfig.from_preset("tabulated_extinction")` → `extinction_mode="tabulated"`. |
| Aggregator `--plot` QC PDFs | P3 | **Done** | `scripts/aggregate_site_extinction.py --plot` writes per-filter night scatter + site summary PDFs. |

---

## Legacy modules — reclassified

These are **not** short-term removal candidates; the table describes real dependencies.

| Module | Priority | Status | Notes |
|--------|----------|--------|-------|
| `analyze/calibration/_legacy.py` | P3 | **Active** | `calculate_trans` retained for backward compatibility; mk_calib now uses `CalibrationEngine` / `mk_calib.py`. Variable-star pipeline uses `CalibrationEngine` only. |
| `analyze/plots/_legacy.py` | P3 | **Active** | `plots/__init__.py` is `from ._legacy import *`. All course plotting scripts depend on it. Cleanup is a large refactor, not post-migration housekeeping. |
| `calibration_parameters.py` | — | **Active infrastructure** | Used for VizieR catalog props, filter systems, chip dimensions, extinction curves, image types — **not** legacy dead code. Do not remove; optional rename/docs only. |
| `analyze/calibration_data.py` | P3 | **Legacy bridge** | Still documents old `derive_calibration` path; pipeline prefers `CalibrationStep`. mk_calib field tables use `calibrate_mk_calib_filter_pair`. |
| `analyze/utils/_legacy.py` | P3 | **Active** | Wide helper surface; many utilities still routed here. |

---

## In-code TODOs — re-evaluated

| Location | Priority | Still valid? | Notes |
|----------|----------|--------------|-------|
| ~~`analyze/analyze.py` OOI IDs~~ | — | **Stale** | Logic lives in `observation.py` and `pipeline/helpers.py`; no open TODO there. |
| `utilities.py` — split `Image` | P3 | Yes | Architectural; affects reduce + analyze. |
| `calibration/_legacy.py` — filter metadata | P3 | Yes | Only relevant while legacy transformation export remains. |
| `extraction.py` — CCDData units, FWHM helper | P3 | **Done** | Shared helpers in `ost_photometry/fwhm.py`; used by extraction and `reduce/utilities.estimate_fwhm`. |
| `reduce/redu.py` | P2 | **Done** | `reduce/validation.py`: flat coverage, sanity sample, graceful science skips; `check_master_files_on_disk` fixed; MP stacking. Facade → modular `reduce/workflow/` package. |
| `reduce/registration.py` | P3 | **Partial** | Trim paths unified via `trim_ccd` / `ccd_trim_slices`; `trim_image` + `trim_image_simple` share the helper. Package split still optional. |
| ~~`correlate/inter.py` — cal-star protection / ID determination~~ | — | **Addressed** | `resolve_calibration_object_ids`; pipeline config flag. |
| `plots/_legacy.py` | P3 | Yes | Many TODOs; module is intentionally the plot backend for now. |
| `correlate/core.py` | P3 | **Done** | `argwhere` cleanup replaced with `_drop_protected_from_rejected_object_ids`; identical-position fallback via try/except; tests in `test_correlation_core.py`. |
| ~~`reduce/workflow/_legacy.py`~~ | — | **Done** | Split into `workflow/{config,constants,main,bias,dark,flat,science,stack,dispatch}.py`; `_legacy.py` removed. |
| `post_processing/hips_reference_subtract.py` | P3 | **Done** | Uses `find_wcs_for_image` on single `Image`; reuse accepts `Image` or `ImageSeries`. |

Approximate open marker count in `src/`: **~43 `TODO`** across the files above (`rg 'TODO' src/`).

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` still literature seed until a dedicated campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Completed / closed by recent work

| Item | Resolution |
|------|------------|
| Calibration convergence (`CalibrationStep`, presets, `extinction_mode`) | Done — see [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md) |
| Site extinction IO + `tabulated` loading | Done — `extinction_io.py`, bundled JSON, aggregator script |
| `skip_extinction_fit` / `fit_extinction_from_data` | Removed — replaced by `extinction_mode` |
| Migration docs sprawl | Consolidated into `ARCHITECTURE_AND_MIGRATION.md` |
| `migration_reports/` duplicate folder | Removed |
| Pipeline `protect_calibration_objects` | Done — unified `protected_object_ids` + independent auto flags; `correlate_preserve_objects` |
| mk_calib WCS/extraction/correlation | Done — `run_pipeline` via `mk_calib_pipeline.py` |
| mk_calib CalibrationEngine + second-order extinction | Done — `mk_calib.py`, `second_order_extinction.py`, preset `linear_fit_ensemble` |
| `uncertainty_mode` in CalibrationStep | Done — `fit_errors` / `flux_monte_carlo` / `both` |
| Site extinction aggregator QC plots | Done — `--plot` on `aggregate_site_extinction.py` |
| `tabulated_extinction` pipeline preset | Done |
| Reduction edge cases (`reduce/validation.py`, `reduce_main` checks) | Done |
| Reduce workflow modularization (`reduce/workflow/` package) | Done — `_legacy.py` removed; `redu.py` facade unchanged |
| `correlate/core.py` argwhere / protected-object cleanup | Done — `test_correlation_core.py` |
| Shared FWHM helpers (`ost_photometry/fwhm.py`) | Done — `test_fwhm.py`; extraction + reduction |
| `find_wcs_for_image` + HiPS `Image` WCS reuse | Done — `test_find_wcs_for_image.py` |
| Registration trim unification (`trim_ccd`) | Done — `test_registration_trim.py` |

---

## Suggested order of implementation

1. **P3 (optional):** further split `reduce/registration.py` into `align` / `shifts` / `trim` packages after more alignment tests.
2. **Long-term:** Gradual shrink of remaining `_legacy.py` modules in `analyze/` (`calculate_trans` no longer on mk_calib hot path).

GitHub issues are preferred for tracking individual P1/P2 items.
