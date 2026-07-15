# TODO

Open, prioritized work on the `ost_photometry` package.

This document is intended to **replace** [TECHNICAL_DEBT.md](TECHNICAL_DEBT.md) once the backlog documented there is largely cleared. Completed items and history remain in `TECHNICAL_DEBT.md` for now. For ongoing architecture and API references, see also [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md), [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), and [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** July 2026 (after workflow modularization and reduction edge cases)

**Priorities:** P1 = next sprint · P2 = track, no urgency · P3 = optional / long-term · — = operational, not code debt

Prefer GitHub issues for individual P1/P2 items.

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

1. **On registration changes:** merge `trim_image` / `trim_image_simple`; optionally extract `trim.py`.
2. **On utilities changes:** extract only the affected area.
3. **Long-term:** gradually shrink remaining `analyze/*/_legacy.py` modules (`calculate_trans` no longer on mk_calib hot path).

---

## Out of scope (intentionally not listed)

- `calibration_parameters.py` — active infrastructure (catalogs, filters, chips, extinction), not dead legacy code.
- `reduce/workflow/` — modularized; `_legacy.py` removed.
- Course scripts — API via `reduce.redu.reduce_main` unchanged; see [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).
