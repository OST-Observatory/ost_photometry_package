# Technical debt

GitHub issues are preferred for ongoing tracking. This file lists known follow-ups
not covered by the living docs ([PIPELINE_CONFIG.md](PIPELINE_CONFIG.md),
[EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md)).

## Pipeline / config

| Item | Location | Notes |
|------|----------|-------|
| `uncertainty_mode` not wired | `pipeline/config.py`, `calibration/uncertainty.py` | Config field exists; steps do not branch on it yet |
| `ost_site` preset | `pipeline/config.py` | Optional convenience preset for `extinction_mode="tabulated"` not added |
| Aggregator QC plots | `scripts/aggregate_site_extinction.py` | `--plot` for review PDFs not implemented |

## Legacy modules (candidates for removal)

| Module | Notes |
|--------|-------|
| `analyze/calibration/_legacy.py` | Superseded by `CalibrationEngine` + backends; still referenced in places |
| `analyze/plots/_legacy.py` | Large plot module; cleanup and type-hint fixes open |
| `calibration_parameters.py` | Legacy parameter-file format |

## In-code TODOs (representative)

| Location | Summary |
|----------|---------|
| `analyze/analyze.py` | `get_ids_object_of_interest` filter/index selection |
| `utilities.py` | Split `Image` into base + analysis class |
| `analyze/calibration/_legacy.py` | Filter metadata on transformed magnitudes |
| `analyze/extraction.py` | Unit check for CCDData; FWHM helper reuse |
| `reduce/redu.py` | Reduction edge cases |
| `reduce/registration.py` | Registration refinements, merge with `trim_image` |
| `analyze/correlate/inter.py` | Protect calibration objects; rewrite ID determination |
| `analyze/plots/_legacy.py` | Plot generalization, error propagation, cleanup |

Full list: `rg 'TODO|FIXME|HACK' src/`

## Site extinction

Bundled `data/ost_potsdam_extinction.json` is seeded from literature defaults until
the first dedicated extinction campaign updates it via `scripts/aggregate_site_extinction.py`.
