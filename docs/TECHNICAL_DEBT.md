# Technical debt

**Primary backlog:** [TODO.md](TODO.md). Living API docs: [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md), [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md).

**As of:** July 2026 — all `src/**/_legacy.py` bags removed (`reduce/workflow`, `analyze/utils`, `analyze/calibration`, `analyze/plots`). Package layout uses named modules + stable facades.

**Priority legend:** P1 = next sprint · P2 = track · P3 = optional · — = operational

---

## Still open (summary)

| Item | Priority | Notes |
|------|----------|-------|
| `fraction_epsf_stars` max / min–max | P2 | See [TODO.md](TODO.md) |
| T/ZP covariance in calibrated errors | P2 | See [TODO.md](TODO.md) |
| Vega/AB labeling & magnitude-system convert | P3 | See [TODO.md](TODO.md) |
| Star-wise k″ fit (optional) | P3 | See [TODO.md](TODO.md) |
| `reduce/registration.py` further package split | P3 | Trim helpers already unified (`trim_ccd`) |
| `analyze/calibration_data.py` | P3 | Deprecated `derive_calibration` bridge; pipeline prefers `CalibrationStep` |
| ~14 in-code `TODO` markers in `src/` | — | Mostly registration, CMD plots, `calibration_data` |

---

## Not debt

| Item | Notes |
|------|-------|
| `calibration_parameters.py` | Active infrastructure (catalogs, filters, chips, extinction curves). |
| Wide `.dat` / `write_legacy_wide_magnitudes_dat` | Optional compatibility export, not a `_legacy` module. |
| `old_legacy_function/` | Archive only; not part of the package API. |
| Site extinction JSON seed | Operational — [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Recently closed (pointer)

ASCII `formats` `i` vs `id` (`ascii_write_formats_for_columns`); calibration convergence, extinction modes/`extinction_order`, mk_calib engine path, reduce workflow modularization, utils/plots/calibration splits, `Image`/`AnalysisImage`, FWHM helpers, registration trim unification, protect-calibrators, diagnostic plot hooks — details in git history and [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md).

GitHub issues are preferred for tracking individual P1/P2 items.
