# TODO

Open, prioritized work on the `ost_photometry` package.

This document is the **forward-looking backlog**. Closed migration notes live in
[ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md); a short open-debt
summary remains in [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md). Also see
[PIPELINE_CONFIG.md](PIPELINE_CONFIG.md) and [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** July 2026 (after `_legacy` bag removal, workflow modularization, and backlog review)

**Priorities:** P1 = next sprint · P2 = track, no urgency · P3 = optional / long-term · — = operational, not code debt

Prefer GitHub issues for individual P1/P2 items.

---

## Calibration accuracy & photometry systems

### Star-wise fit for second-order extinction k″ (P3)

Today k″ is **not** fitted in the routine pipeline. Options:

| Path | What it does |
|------|----------------|
| `extinction_mode` + `extinction_order` | Apply k′ (and optionally tabulated / user k″) |
| `run_second_order_campaign` (mk_calib) | Field-level `C = T + k'·X`, then `k" = -k'` |

A **star-wise** alternative would fit per star × epoch:

`m_obs − m_std = ZP + k'·X + k''·X·(color)`

That is a **new** method (not the same as `run_second_order_campaign`, which reduces stars to a field `C` first). Needs wide airmass **and** color span, and careful separation from the linear color term `T`.

**Direction (optional):** optional `extinction_mode` / helper that fits star-level residuals; keep mk_calib campaign and tabulated/user k″ as the supported paths for applying SECOND order. Do **not** bolt this onto `fit_extinction_from_comparison_stars` (epoch-mean vs X destroys color information).

### T/ZP fit covariance in calibrated errors — done

`TransformationCoefficients.cov_tz` stores `Cov(T, ZP)` from `weighted_linear_fit`. Apply-transform / `calibrated_magnitude_variance` include `2·color·cov(T, ZP)`. Rolling/IV-combined T/ZP set `cov_tz=0` (independent mixing).

### Vega vs AB / magnitude-system transforms — done

Two axes: **filter set** (`bessell` / `sdss`) and **magnitude system** (`vega` / `ab`). Config: `output_filter_set`, `output_magnitude_system` (`auto` = catalog/calibrated), `convert_magnitudes`. Early abort on SDSS+Vega. Conversions: Vega↔AB offsets, Bessell→SDSS (Jordi), SDSS→Bessell (Lupton). Table meta + light-curve labels. See [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md).

### OST filter throughput → Vega↔AB offsets via synphot (P3)

Published Bessell/SDSS per-filter constants (`m_AB = m_Vega + Δ`) are correct for same-bandpass ZP conversion. If OST/instrument **throughput curves** are added later, recompute `Δ` with `synphot`/`speclite` instead of literature defaults.

**Advantages:**

- Offsets match **actual OST bandpasses**, so Vega↔AB on `mag_cal_*` matches how the site observes.
- Can cover **filter variants / chip × filter** combinations when curves differ.
- Same stack can check whether literature Jordi/Lupton colour terms fit OST throughputs (synthetic colours).
- Transparent provenance: regenerate from curves + CALSPEC Vega + AB definition.
- Method unchanged (still per-filter constants); only the numerical `Δ` become instrument-specific.

Optional until curves exist in-repo and a lightweight synphot/speclite dependency is accepted.

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

**Done.** Selection uses ``n_epsf_stars_to_select``:
``clamp(int(n_stars * fraction_epsf_stars), minimum_n_eps_stars, maximum_n_eps_stars)``.
Config: ``minimum_n_eps_stars`` (default 15), ``maximum_n_eps_stars`` (default 50; ``None`` = no upper cap), ``fraction_epsf_stars`` (default 0.2).

### Finite-value checks around `extract_stars` (P3)

Cutouts / inputs are not systematically checked for non-finite values before ePSF building; finite checks appear later mainly for `error` arrays.

**Direction:** reject or mask non-finite pixels/stars early with a clear warning.

### Move `mark_simbad_objects_on_image` to post-processing (P3)

Still invoked from `main_extract` when annotating. Belongs with other optional annotation / post-process tasks, not in the extraction hot path.

---

## Warnings & small fixes

| Item | Prio | Notes |
|------|------|-------|
| ASCII `formats` key `{'i'}` vs column `id` | — | **Done.** `schema.ascii_write_formats_for_columns` filters format keys to columns present on the table (`post_processing/io.py`, `utils/legacy_magnitudes.py`). |
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

**Open in-code TODOs (4):** loop optimizations in `align_image_main`; footprint/mask for background. Trim helpers unified via `trim_ccd` / `ccd_trim_slices` (`trim_image` + `trim_image_simple`).

**Recommendation:** No separate large refactor. Worth doing **only when actively working** on alignment/trim. Before a larger split, add small regression tests with synthetic `CCDData` arrays (shifts, trim edges) — the workflow split was a pure move; risk is higher here.

**Possible target structure (if undertaken):**

```
reduce/registration/
  align.py    # align_images, align_image_main
  shifts.py   # calculate_xy_image_shifts*, astro_align, optical_flow_*
  trim.py     # trim_ccd, trim_image, trim_image_simple
```

**Done (first step):** unify trim paths on `trim_ccd` / `ccd_trim_slices` (see `tests/test_registration_trim.py`). Full package split still optional.

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

**Note:** The former “split `Image`” item (`ost_photometry.utilities.Image`) is **done** — see `ost_photometry/image.py` and `analyze/image.py`. This section is only about `reduce/utilities.py`.

---

## Remaining analyze bridge

| Module | Notes |
|--------|-------|
| `analyze/calibration_data.py` | Deprecated `derive_calibration` path; pipeline prefers `CalibrationStep`. Optional cleanup when no external callers remain. |

### Drop legacy wide magnitude tables / column `i` (P3)

When the optional wide `.dat` path (`write_legacy_wide_magnitudes_dat`, `calibrated_epochs_to_legacy_wide_table`, `save_magnitudes_ascii` with column `i`) is retired in favour of epoch-native tables only (`id`):

- Remove column `i` and the `"i"` entry from `_ASCII_COLUMN_FORMATS` / `ascii_write_formats_for_columns`.
- Drop dual-read branches such as `"i" if "i" in colnames else "id"` (e.g. adapters).
- Remove or gate the wide-table builders and the `write_legacy_wide_magnitudes_dat` config flag.

Until then, keeping `"i"` in the filtered formats helper is correct.

---

## In-code TODOs

Roughly **~14 `TODO` markers** in `src/` (`rg 'TODO' src`). Hotspots: `reduce/registration.py`, `analyze/plots/cmds.py`, `analyze/calibration_data.py`.

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` is a literature seed until a campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Suggested order

1. **P3:** Star-wise k″ fit (optional alternative to mk_calib campaign) — see section above.
2. **P3:** OST filter throughput → synphot Vega↔AB offsets — see section above.
3. **P3 (when ready):** Drop legacy wide tables / column `i` — see section above.
4. **On further registration work:** optionally extract `trim.py` / split align+shifts; remaining TODOs in `align_image_main`.
5. **On utilities changes:** extract only the affected area.

**Recently done:** Vega/AB + filter-set conversion model; T/ZP `cov_tz`; ASCII `formats` key `i` vs `id`; `fraction_epsf_stars` + `maximum_n_eps_stars` clamp.

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

- `calibration_parameters.py` — active infrastructure (catalogs, filters, chips, extinction).
- Modular packages already in place: `reduce/workflow/`, `analyze/utils/`, `analyze/plots/`, `analyze/calibration/` (no `_legacy` bags).
- Course scripts — API via `reduce.redu.reduce_main` unchanged; see [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).
- Optional wide `.dat` export (`write_legacy_wide_magnitudes_dat`) — compatibility flag, not tech debt.
