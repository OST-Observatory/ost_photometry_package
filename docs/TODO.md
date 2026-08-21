# TODO

Open, prioritized work on the `ost_photometry` package.

This document is the **forward-looking backlog**. Closed migration notes live in
[ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md). Also see
[PIPELINE_CONFIG.md](PIPELINE_CONFIG.md) and [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** August 2026

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

### Unified correlated object index — done

After correlation, photometry table ``id`` is the aligned row index. Objects of interest store that as ``correlated_id`` (see `ooi_photometry_id`). ``id_in_image_series`` remains the pre-alignment per-filter row map for intra-filter protection. Calibration stars are matched to the same table ``id`` (no separate stored index).

---

## Extraction / ePSF

### Finite-value checks around `extract_stars` (P3)

Cutouts / inputs are not systematically checked for non-finite values before ePSF building; finite checks appear later mainly for `error` arrays.

**Direction:** reject or mask non-finite pixels/stars early with a clear warning.

### Move `mark_simbad_objects_on_image` to post-processing (P3)

Still invoked from `main_extract` when annotating. Belongs with other optional annotation / post-process tasks, not in the extraction hot path.

---

## Warnings & small fixes

| Item | Prio | Notes |
|------|------|-------|
| `FITSFixedWarning` (`datfix` / `MJD-OBS` from `DATE-OBS`) | P3 | Harmless but noisy; filter or set MJD-OBS explicitly when building WCS/Time from FITS. |
| NaNs clipped in `ImageDepth` / limiting magnitude | P3 | `_derive_limiting_magnitude_one_epoch` → `ImageDepth`; pre-clean non-finite pixels or acknowledge warning. |

---

## Reduce

### `reduce/registration.py` — modularize (P3, as needed)

**~1700 lines**, 15 functions. Orchestration (`align_images`), shift algorithms, and trim logic in one file.

**External API (keep stable):**

| Consumer | Symbols |
|----------|---------|
| `reduce/workflow/main.py` | `align_images`, `make_big_images` |
| `n1_baches/1_masterimages.py` | `trim_image_simple` |

Trim helpers are already unified (`trim_ccd` / `ccd_trim_slices` / `aa_common_trim_margins`). AA footprint stays on `CCDData.mask`.

**Recommendation:** no dedicated split. If touching alignment/trim, optionally:

```
reduce/registration/
  align.py    # align_images, align_image_main
  shifts.py   # calculate_xy_image_shifts*, astro_align, optical_flow_*
  trim.py     # trim_ccd, trim_image, trim_image_simple
```

### `reduce/utilities.py` — optional further split (P3)

Facade plus reduction-specific helpers; already delegates to `exposure`, `instrument`, `masks`, `wcs_reduce`, …. Incremental only when touching the affected area.

---

## Remaining analyze bridge

### Drop legacy wide magnitude tables / column `i` (P3)

When the optional wide `.dat` path (`write_legacy_wide_magnitudes_dat`, `calibrated_epochs_to_legacy_wide_table`, `save_magnitudes_ascii` with column `i`) is retired in favour of epoch-native tables only (`id`):

- Remove column `i` and the `"i"` entry from `_ASCII_COLUMN_FORMATS` / `ascii_write_formats_for_columns`.
- Drop dual-read branches such as `"i" if "i" in colnames else "id"` (e.g. adapters).
- Remove or gate the wide-table builders and the `write_legacy_wide_magnitudes_dat` config flag.

Until then, keeping `"i"` in the filtered formats helper is correct.

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` is a literature seed until a campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Suggested order

1. **P3:** Star-wise k″ fit (optional alternative to mk_calib campaign).
2. **P3:** OST filter throughput → synphot Vega↔AB offsets.
3. **P3 (when ready):** Drop legacy wide tables / column `i`.
4. **On further registration work:** optionally extract `trim.py` / split align+shifts.
5. **On utilities changes:** extract only the affected area.

---

## Out of scope (intentionally not listed)

- `calibration_parameters.py` — active infrastructure (catalogs, filters, chips, extinction).
- Modular packages already in place: `reduce/workflow/`, `analyze/utils/`, `analyze/plots/`, `analyze/calibration/` (no `_legacy` bags).
- Course scripts — API via `reduce.redu.reduce_main` unchanged; see [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).
- Optional wide `.dat` export (`write_legacy_wide_magnitudes_dat`) — compatibility flag, not tech debt.
- In-code `TODO` markers under `src/` — none remaining (August 2026).
