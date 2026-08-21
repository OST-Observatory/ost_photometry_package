# OST Photometry Package

Photometry reduction and analysis package for the
[OST Observatory](https://polaris.astro.physik.uni-potsdam.de/) of
the University of Potsdam.

The package provides data reduction and analysis for photometric observations
from the observatory's telescopes. It is used in astrophysics laboratory courses
and for on-the-fly reduction in the observatory archive, but the APIs are
intended for general differential photometry workflows as well.

## Package structure

The installable package lives under `src/ost_photometry/` and splits into two
main layers:

| Module | Role |
|--------|------|
| **`ost_photometry.reduce`** | CCD preprocessing: bias/dark/flat correction, registration, masking, instrument helpers |
| **`ost_photometry.analyze`** | Science analysis: WCS, extraction, correlation, calibration, post-processing, light curves |

Supporting code includes `terminal_output` and `style` (console output),
`checks`, `calibration_parameters` (catalogs, filters, chips, extinction curves),
and `core` (shared utilities such as parallel execution).

Typical workflow: reduce raw FITS with **`reduce`**, then run **`analyze`** on
the calibrated image series via an `Observation` object and `run_pipeline()`.

## Capabilities

**Reduction (`reduce`)** — standard CCD stack building, image alignment, cosmic-ray
handling hooks, and WCS helpers used by the analysis layer.

**Analysis pipeline (`analyze.pipeline`)** — configurable step sequence driven by
[`PipelineConfig`](src/ost_photometry/analyze/pipeline/config.py):

1. WCS determination (`astrometry`, `astap`, `twirl`)
2. PSF or aperture photometry extraction
3. Intra- and inter-filter source correlation
4. Optional extinction coefficient fit
5. Differential calibration (median ZP or linear T/ZP; epoch-native tables)
6. Post-processing (region selection, cluster/Gaia matching, magnitude conversion)
7. Optional light-curve plots and HiPS reference subtraction

**Calibration** — catalog cross-match (APASS, VizieR, custom tables), extinction
correction, `PhotometryCalibrator` / `CalibrationEngine`, and named presets
(`median_zp_per_image`, `linear_fit_per_night`, …).

**Libraries usable outside the pipeline** — `main_extract`, `PhotometryCalibrator`,
`ExtinctionCorrector`, correlation helpers, plotting, image subtraction, and
calibration-source fetch/crossmatch APIs.

Entry point for scripted runs:

```python
from ost_photometry.analyze import Observation, PipelineConfig

observation = Observation.from_config(...)
config = PipelineConfig.from_preset("linear_fit_per_night")
observation.run_pipeline(filter_list, image_paths=..., output_dir=..., config=config)
```

## Requirements

Core Python dependencies:

* [ccdproc](https://github.com/astropy/ccdproc)
* [photutils](https://github.com/astropy/photutils)
* [astropy](https://github.com/astropy/astropy)
* [astroquery](https://github.com/astropy/astroquery)
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [matplotlib](https://github.com/matplotlib/matplotlib)

For the default WCS method (`astrometry`), a local
[astrometry.net](https://nova.astrometry.net/) installation is required.

## Documentation

| Topic | Document |
|-------|----------|
| Pipeline options and decision tables | [docs/PIPELINE_CONFIG.md](docs/PIPELINE_CONFIG.md) |
| Site extinction table and observation campaigns | [docs/EXTINCTION_COEFFICIENTS.md](docs/EXTINCTION_COEFFICIENTS.md) |
| Breaking changes and architecture (archive) | [docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md) |
| Script compatibility matrix | [docs/COMPATIBILITY_REPORT.md](docs/COMPATIBILITY_REPORT.md) |
| Known follow-ups | [docs/TECHNICAL_DEBT.md](docs/TECHNICAL_DEBT.md) |

Config defaults and all pipeline fields:
[`src/ost_photometry/analyze/pipeline/config.py`](src/ost_photometry/analyze/pipeline/config.py).

## Development

Install the package with lint/test tools, then enable the git hook so
[Ruff](https://docs.astral.sh/ruff/) runs (and auto-fixes) before each commit —
the same check as CI (`ruff check src tests`):

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run the linter on demand with `ruff check src tests` (add `--fix` to apply
safe fixes). Tests: `pytest tests -m "not slow"`.
