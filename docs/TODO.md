# TODO

Open, prioritized work on the `ost_photometry` package.

This document is intended to **replace** [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) once the backlog documented there is largely cleared. Completed items and history remain in `TECHNICAL_DEBT.md` for now. For ongoing architecture and API references, see also [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md), [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), and [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** July 2026 (after workflow modularization, reduction edge cases, and review of an older numbered backlog)

**Priorities:** P1 = next sprint · P2 = track, no urgency · P3 = optional / long-term · — = operational, not code debt

Prefer GitHub issues for individual P1/P2 items.

---

## Calibration accuracy & photometry systems

### T/ZP fit covariance in calibrated errors (P2)

`T` and `ZP` come from the same linear fit and are correlated. `weighted_linear_fit` computes the full covariance but only returns diagonal errors; propagation currently assumes uncorrelated terms (explicit note in `differential_photometry.py`: `cov(T,ZP) … is neglected`). `TransformationCoefficients` has no off-diagonal term.

**Direction:** carry `cov_tz` (or full 2×2) on `TransformationCoefficients` and include `2·color·cov(T,ZP)` in error propagation (`uncertainty.py` / apply-transform paths).

### Vega vs AB / magnitude-system transforms (P3)

Plots often hardcode “(Vega)”. `convert_magnitudes_to_other_system` implements SDSS/Jordi paths; `AB` / `BESSELL` still warn “not implemented”. The photometric system is primarily that of the **calibration catalog** (via `filter_systems`), not a free pipeline switch — but conversion and labeling should be consistent and explicit.

**Direction:** document catalog→system mapping; finish or clearly gate AB/Bessell conversion; optionally unify calibration-side and post-processing transforms (`PostProcessMagnitudeConvertStep`).

---

## Pipeline architecture

### Optional hard prerequisites between steps (P3)

Steps already use soft `skip_*` flags and context markers (`extraction_done`, etc.). There is **no** dependency graph: a later step can run (or fail opaquely) if earlier work was skipped.

**Direction:** optional validation (e.g. correlation requires extraction; calibration requires correlation) that fails fast with a clear message — without forcing a full DAG rewrite.

### Checkpoint / resume after extraction + correlation (P3)

Steps are already modular via config. What is missing is **persist and continue later** (disk checkpoint of `AnalysisContext` / intermediate tables).

**Direction:** only if course/supervisor workflows need “extract+correlate → stop → calibrate later”. Otherwise leave as soft `skip_*` usage.

### Unified correlated object index (P3)

After correlation, table `id` is typically the row index (`assign_global_correlated_object_ids`). OOI still uses `id_in_image_series: dict[str, int]`; calibration objects keep a separate ID list. A third parallel index is unnecessary — prefer one correlated `id` everywhere.

**Direction:** audit `ObjectOfInterest.id_in_image_series`, `get_ids_object_of_interest` / `reference_obj_ids`, and `ids_calibration_objects`; migrate callers to the correlated `id` where safe.

---

## Extraction / ePSF

### `fraction_epsf_stars` can select too-faint stars (P2)

With many detections, `int(n_stars * fraction_epsf_stars)` still pulls in faint stars for the ePSF (crash / poor ePSF risk). Only a `minimum_n_stars` floor exists today.

**Direction:** prefer a **max count** and/or an explicit **min–max range** of stars for ePSF construction (config on `PipelineConfig` / `main_extract`).

### Finite-value checks around `extract_stars` (P3)

Cutouts / inputs are not systematically checked for non-finite values before ePSF building; finite checks appear later mainly for `error` arrays.

**Direction:** reject or mask non-finite pixels/stars early with a clear warning.

### Move `mark_simbad_objects_on_image` to post-processing (P3)

Still invoked from `main_extract` when annotating. Belongs with other optional annotation / post-process tasks, not in the extraction hot path.

---

## Warnings & small fixes

| Item | Prio | Notes |
|------|------|-------|
| ASCII `formats` key `{'i'}` vs column `id` | P2 | Epoch-native schema uses `id`; some writers still pass format key `i` → Astropy warning. Align keys in `post_processing/io.py` / legacy `save_magnitudes`. |
| `FITSFixedWarning` (`datfix` / `MJD-OBS` from `DATE-OBS`) | P3 | Harmless but noisy; filter or set MJD-OBS explicitly when building WCS/Time from FITS. |
| NaNs clipped in `ImageDepth` / limiting magnitude | P3 | `_derive_limiting_magnitude_one_epoch` → `ImageDepth`; pre-clean non-finite pixels or acknowledge warning. |

---

## Reduce

### `reduce/registration.py` — modularize (P3, as needed)

**~1720 lines**, 15 functions. Contains orchestration (`align_images`), shift algorithms (phase correlation, optical flow, astroalign), and trim logic in a single file.

**External API (keep stable):**

| Consumer | Symbols |
|----------|---------|
| `reduce/workflow/main.py` | `align_images`, `make_big_images` |
| `n1_baches/1_masterimages.py` | `trim_image_simple` |

**Open in-code TODOs (5):** including merge of `trim_image` / `trim_image_simple`; loop optimizations in `align_image_main`; footprint/mask for background.

**Recommendation:** No separate large refactor. Worth doing **only when actively working** on alignment/trim. Before a larger split, add small regression tests with synthetic `CCDData` arrays (shifts, trim edges) — the workflow split was a pure move; risk is higher here.

**Possible target structure (if undertaken):**

```
reduce/registration/
  align.py    # align_images, align_image_main
  shifts.py   # calculate_xy_image_shifts*, astro_align, optical_flow_*
  trim.py     # trim_image, trim_image_simple
```

**First sensible step:** unify `trim_image` and `trim_image_simple` (existing TODO in code), without splitting everything at once.

---

### `reduce/utilities.py` — optional further split (P3)

**~940 lines**, 14 functions. Already delegates to submodules (`exposure`, `instrument`, `masks`, `wcs_reduce`, …); the file is a facade plus reduction-specific helpers.

**External API (keep stable):**

| Consumer | Symbols |
|----------|---------|
| Course `c7`, `n2` | `prepare_reduction` |
| Course `n1` | `flip_image`, `bin_image`, `inverse_median` |
| Workflow modules | Master checks, exposure times, WCS, FWHM |

**Recommendation:** **Low priority.** Incremental only when touching affected areas — e.g. `preprocessing.py` (flip/bin), `masters.py`, `fwhm.py`. No dedicated cleanup PR needed.

**Note:** The “split `Image`” entry in `TECHNICAL_DEBT.md` primarily concerns `ost_photometry.utilities.Image` (reduce + analyze), not `reduce/utilities.py` as a whole.

---

## Analyze — legacy modules (P3, long-term)

These modules are **actively in use**; not short-term removal candidates. Shrink gradually as respective code paths are migrated.

| Module | Notes |
|--------|-------|
| `analyze/calibration/_legacy.py` | `calculate_trans` for backward compatibility; mk_calib and variable-star pipeline use `CalibrationEngine`. |
| `analyze/plots/_legacy.py` | Plot backend for all course plotting scripts. |
| `analyze/utils/_legacy.py` | Wide helper surface; many utilities still routed here. |
| `analyze/calibration_data.py` | Legacy bridge to `derive_calibration`; pipeline prefers `CalibrationStep`. |

---

## In-code TODOs (still open)

Roughly **~43 `TODO` markers** in `src/` (`rg 'TODO' src/`). Selected items that remain valid:

| Location | Prio | Content |
|----------|------|---------|
| ~~`correlate/core.py`~~ | — | **Done** — see `test_correlation_core.py` |
| ~~`extraction.py`~~ | — | **Done** | FWHM via `ost_photometry/fwhm.py` |
| `ost_photometry.utilities` — `Image` | P3 | Architectural split; affects reduce + analyze |
| `calibration/_legacy.py` | P3 | Filter metadata; relevant only while legacy transformation export remains |
| `plots/_legacy.py` | P3 | Many TODOs; intentionally the plot backend |
| ~~`post_processing/hips_reference_subtract.py`~~ | — | **Done** | `find_wcs_for_image` on `Image` |

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` is a literature seed until a campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Suggested order

1. **P2:** ASCII `formats` key `i` vs `id` (small, visible warning).
2. **P2:** `fraction_epsf_stars` → max count / min–max range.
3. **P2:** T/ZP covariance in calibrated magnitude errors.
4. **P3:** Vega/AB labeling and magnitude-system conversion consistency.
5. **On registration changes:** merge `trim_image` / `trim_image_simple`; optionally extract `trim.py`.
6. **On utilities changes:** extract only the affected area.
7. **Long-term:** gradually shrink remaining `analyze/*/_legacy.py` modules (`calculate_trans` no longer on mk_calib hot path).

---

## Reviewed older backlog — not added

These came from an earlier numbered list; they are **obsolete, vague, or already addressed** and were intentionally not promoted:

| Old # | Verdict |
|-------|---------|
| 7 Warning formatting | Soft polish only; `OstPhotometryAnalyzeWarning` + terminal styles already exist. |
| 8 Everything English? | `src/` mostly English; leftover German is mainly in some docs. |
| 12 Comments / rename for readability | Do when touching code; not a discrete backlog item. |
| 18 Rework LimitingMagnitudeStep | Largely done (`DeriveLimitingMagnitudeStep` / epoch-native path); only small polish remains (see NaN warning above). |
| 25 Remove unused imports | Hygiene via `ruff`, not tracked debt. |
| 28 Improve extinction fit? | Too vague; site seed is operational (above). Re-open only with a concrete defect. |
| 29 Clear filter / legacy vs differential in `2_obtain_flux` | Course path uses `run_pipeline`; Clear uses flux fallback in `LightCurveStep`. Obsolete as worded. |
| 31 EPSFBuilder → `EPSFBuildResult` | Tuple unpack still supported under `photutils>=3`; optional modernization only. |
| 36 `kground_clipping` typo | Fixed as `sigma_value_background_clipping`. |
| 39 Global `np.argwhere` / `.ravel()` audit | `correlate/core.py` done; remaining uses are local — fix when editing. |

---

## Out of scope (intentionally not listed)

- `calibration_parameters.py` — active infrastructure (catalogs, filters, chips, extinction), not dead legacy code.
- `reduce/workflow/` — modularized; `_legacy.py` removed.
- Course scripts — API via `reduce.redu.reduce_main` unchanged; see [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).
