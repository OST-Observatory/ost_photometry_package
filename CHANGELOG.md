# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Releases are annotated Git tags `v<version>` on `main`. The process is in
[docs/RELEASING.md](docs/RELEASING.md). GitHub Release notes are the matching
section of this file.

## [Unreleased]

## [0.5.0] - 2026-09-15

First tagged release after `0.4.4`. This is the epoch-native pipeline: one
calibration engine, `PipelineConfig` presets, and the dual legacy / differential
paths removed.

### Added

- Unified **`CalibrationEngine`** / **`CalibrationStep`** with named presets
  (`linear_fit_per_image`, `linear_fit_per_image_extinction`,
  `median_zp_per_image`, `linear_fit_per_night`,
  `linear_fit_per_night_extinction`, `linear_fit_ensemble`,
  `tabulated_extinction`, `extract_protect_calibrators`). Catalog-color
  derive-transform, T/ZP covariance in `err_cal_*`, and
  `calibration_match_radius`.
- Site extinction table and observation campaigns
  (`docs/EXTINCTION_COEFFICIENTS.md`); `extinction_mode` /
  `extinction_order` / optional k″.
- Alard–Lupton image subtraction (spatial kernel, affine template warp,
  PanSTARRS HiPS default; HOTPANTS still available).
- `shift_method="wcs"`: reproject science frames onto the reference celestial
  WCS. Alignment backends share one shift-method dictionary.
- Sparse tracks (`require_complete_intersection=False`,
  `min_detection_fraction`), pixel matching when frames share a grid,
  sequential linking across gaps, auto pixel-vs-sky coordinates, and
  intra-filter **track QC** (`diagnostics/correlation/track_qc_<filter>`).
- Independent OOI match radius (`ooi_separation_limit`); photometry IDs bound
  after correlation so light curves use track ids, not row indices.
- Multi-night light curves; FWHM-scaled apertures
  (`aperture_scale_with_fwhm`); `cosmic_ray_removal` `auto` / `always` /
  `never`; `maximum_n_epsf_stars`; skip unusable frames in multi-image
  extraction instead of aborting the series.
- Camera catalog rebuilt from manufacturer CSVs (`scripts/build_camera_catalog.py`);
  chip sizes for QHY5III462C and QHY5III485C.
- Simbad annotation as a pipeline step (otype / magnitude / common-name cuts).
- Gaia cluster membership (`is_cluster_member`, `cluster_p_mem`), membership
  QC plots, and interactive cluster-id selection.
- CMD post-processing: series tables, isochrone metadata box, correction
  offsets, distance-modulus and reddening uncertainties on absolute CMDs.
- Calibration / correlation diagnostics: inter-filter residual geometry,
  exposure-pairing overview, catalog-color derive-transform across epochs,
  photometry mag-vs-error (and overview), pre-fit calibrator quality cuts,
  known-variable exclusion via CDS xMatch.
- Keep a Changelog, `docs/RELEASING.md`, and a GitHub Release workflow on
  `v*` tags. Python 3.13 in the CI test matrix.

### Changed

- `Observation.run_pipeline` is the analysis entry point (replaces
  `extract_flux` / `extract_flux_multi`). Results are epoch-native ECSV
  tables (`mag_cal_*`, `err_cal_*`). Magnitude-system conversions
  (Vega↔AB, Bessell↔SDSS) live in post-processing.
- Diagnostic plots go to `<output>/diagnostics/<step>/`; many run in a
  subprocess so they do not block the pipeline. Per-epoch calibration QC
  PDFs are capped.
- `Image` split from `AnalysisImage`; trim / registration / endianness
  helpers unified; worker pools default to half the CPUs.
- Bad-pixel masks: only significantly negative pixels after dark
  subtraction; fill defects before resampling; warp masks nearest-neighbour
  so they do not grow. Warn if more than 10 % of the frame interior stays
  masked after align.
- Extinction is applied in the colour-term fit even when
  `color_term_fit="never"`. OOI is kept out of the calibrator pool.
- Finder / FWHM cuts: IRAF roundness, FWHM-scale window, cosmic-ray header
  handshake with reduction.

### Removed

- Dual calibration APIs: `calibration_module`, `differential_*` config,
  `CalibrationDataStep` / `CalibrationApplyStep` /
  `DifferentialCalibrationStep`, `derive_calibration` / `CalibParameters`,
  legacy wide magnitude tables / `.dat` writers.
- Renamed presets (`c7_variable`, `n2_stack`, `mk_calib_*`, `ost_site`);
  use the canonical names above. See
  [docs/ARCHITECTURE_AND_MIGRATION.md](docs/ARCHITECTURE_AND_MIGRATION.md).

### Fixed

- Clear / catalog-free light curves: the epoch quasi-ZP uses relative fluxes
  of stars detected in most frames, so faint-star dropout no longer tilts
  the continuum away from 1. Crash on Clear-only campaigns (no catalog ZP).
- Reduce yes/no prompts (reuse masters / previous science frames) time out
  after 30 s and default to ``no`` (stdlib ``select``; ``pytimedinput`` is
  no longer required).
- APER flux errors combine aperture-sum and sky σ in quadrature (the
  aperture-sum term was previously treated as a variance and the sky term
  scaled twice). PSF photometry sanitizes non-finite pixel errors.
- Light-curve outlier flagging: short faint runs and bright spikes; coherent
  eclipse dips stay unflagged. Y-limits padded so dips remain visible.
- ASTAP WCS and detector-gain handling (merged from `main` / `patch_gain`).
- Background RMS mask; derive-transform sigma-clip; limiting-magnitude
  apertures on sources; starmap ellipses and filenames; ImageFileCollection
  init warning; photutils v2/v3 CI.

## [0.4.4] - 2026-07-07

Last version on `main` before this changelog. It was not tagged; detailed notes
start with later versions. Git history before that date remains the record.

[Unreleased]: https://github.com/OST-Observatory/ost_photometry_package/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/OST-Observatory/ost_photometry_package/compare/2648b88...v0.5.0
[0.4.4]: https://github.com/OST-Observatory/ost_photometry_package/commit/2648b88
