# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Releases are annotated Git tags `v<version>` on `main`. The process is in
[docs/RELEASING.md](docs/RELEASING.md). GitHub Release notes are the matching
section of this file.

## [Unreleased]

### Added

- **Frame quality** in the reduction (`reduce.quality`, `reduce.frame_selection`):
  per-frame FWHM (Gaussian fits of selected stars), roundness, star count, sky
  background / RMS and masked fraction, measured on the reduced lights before
  alignment. Results in `<output>/frame_quality.ecsv` and the frame headers
  (`FWHM`, `FWHMAS`, `PIXSCALE`, `ROUNDNES`, `NSTARS`, `BACKGRND`, `BKGRMS`,
  `MASKFRAC`, `QCSTAT`). `reduce_main(measure_frame_quality=True)` is the default.
- **Frame selection** (`frame_selection=`): `fwhm_max` in px or arcsec,
  `best_fraction` (Siril "best X %"), `fwhm_sigma_clip`, `roundness_max`,
  `n_stars_min`, `background_max`, `masked_fraction_max`, `min_frames` floor,
  `rank_by="fwhm_weighted"` (wFWHM analogue). Rejected frames are moved to
  `<output>/rejected_lights/` with `QCREJ` / `QCREASON`, never deleted.
- **Reference frame = sharpest frame** (`reference_image_selection="best_fwhm"`,
  default; per filter, or one global frame with `shift_all`). An explicit
  `reference_image_index` still wins.
- **Weighted stacking** (`stack_weighting="fwhm" | "n_stars" | "noise"`) for
  `stack_method="average"`; weights live in `FRMWGHT`, the stack records
  `WEIGHTNG`, `NFRAMES0`, `NREJECT`, `NALIGNFL`, `FWHMMED`, `FWHMMAX`.
  `keep_aligned_lights=True` keeps the registered frames for re-stacking.
- **Alignment accounting**: `align_images` returns an `AlignmentResult`
  ("n of N frames aligned; skipped: …"); skipped frames are recorded in the
  quality table (`aligned`, `align_note`).
- Frame-quality QC plots per filter under `diagnostics/frame_quality/`.
- **Archive access** (`ost_photometry.archive`): `ArchiveClient` for the OST
  data archive (session + CSRF login, runs, data files, object search,
  checksum-verified downloads, rate limiting), content-addressed cache,
  manifests from the archive or a local directory (`manifest_from_directory`),
  `fetch_dataset` for an object or an observation run including calibration
  candidates of neighbouring runs.
- **Calibration grouping** (`reduce.grouping`): frame types from image
  statistics (header types are often wrong; spectroscopy is recognised),
  electronic setups for bias / darks, targets by sky position, camera
  orientation from plate solving (archive WCS or local ASTAP, cached, with
  bisection of changes), mount sessions, flat sets with a probability per
  session (time prior, dust fingerprint, vignetting), reduction units and an
  editable `calibration_plan.yaml` with overrides; timeline plots under
  `diagnostics/calibration_groups/`. See `docs/ARCHIVE_PIPELINE.md`.
- **Group-wise reduction and per-target stacking**
  (`reduce.workflow.groups.reduce_planned`, `reduce.workflow.combine.stack_planned`):
  masters per calibration group, lights per unit with explicit masters, one
  grid per target, weighted stacks per camera and filter over all nights,
  optional noise-weighted camera combination.
- `wcs.solve_astap_copy` (plate solving without touching the file),
  `wcs.position_angle_from_wcs`, `camera_specs.normalize_instrument_name`
  (QHY268 6252×4176 and ZWO ASI2600 behind generic driver names),
  `reduce_main(interactive=False)`, `reduce_light_image(pixel_mask=)`,
  `masks.load_pixel_mask_files`, `workflow.main.resolve_camera_parameters`.
- **Measured detector noise** (`reduce.detector_noise`,
  `reduce_main(camera_noise_source="measured")`, also in the archive
  pipeline per electronic setup): read noise from bias (or dark) pairs, gain
  from the photon transfer of flat pairs; falls back to the catalog.
- `binning_mode` (`auto` / `digital` / `charge`) and the catalog field
  `binning_mode` in `data/cameras.json` (`camera_specs.binning_mode`).
- `rm_cosmic_rays="auto"` with `cosmic_ray_auto_min_frames` (default 7).

### Changed

- `reference_image_index` defaults to `None` (was `0`); with
  `measure_frame_quality=True` the sharpest frame is the reference. Pass
  `reference_image_index=0` or `reference_image_selection="first"` for the
  old behaviour.
- `apply_astro_align`, `apply_wcs_align`, `apply_optical_flow` and
  `apply_xy_image_shift` return `(basename, success, note)` instead of `None`.
- `stack_image` prints a per-filter summary and accepts `stack_weighting`,
  `quality_table`, `keep_input_frames`; `stack_filter_images` accepts
  `weights` and `stack_meta`.
- Stacks record the summed exposure time of the combined frames (`EXPTIME`)
  instead of `n x` the first frame's; `INSTRU` follows `INSTRUME`; missing
  `EGAIN` / `OBJECT` / `FILTER` no longer abort the header update.
- Frame selection, weights and reference choice accept `group_columns`
  (e.g. camera and filter); FWHM weighting uses arcsec when pixel scales are
  mixed.
- `find_wcs_astap` solves blind (radius 180°) when the header has no
  pointing instead of searching around RA = Dec = 0, runs with a timeout,
  and uses the correct image shape for the corner check.
- `requests` and `pyyaml` are declared dependencies.
- `reduce_main(rm_cosmic_rays=)` defaults to `"auto"` (was `True`): with
  stacking, filters with at least 7 frames leave cosmic rays to the
  sigma-clipped stack. Pass `True` for the old behaviour.

### Fixed

- `check_exposure_times` checked only the first exposure time and named the
  wrong file in its error message.
- **Read noise of binned CMOS frames** was the catalog value per native
  pixel; a 3×3 binned QHY600 pixel has three times that. L.A.Cosmic took sky
  noise for cosmic rays (13 % of an Hα frame masked, sky 12 % too low). The
  catalog read noise is now scaled with `sqrt(xbin * ybin)` for digitally
  binning cameras.
- **Uncertainty maps** of lights, flats and darks counted the bias pedestal
  as photon noise (`create_deviation` before the bias subtraction), which
  overestimated the pixel noise by 17–66 % on the OST test data. The
  uncertainty is now computed from the bias-free signal (after the dark when
  the darks carry the bias); master uncertainties add in quadrature.

### Deprecated

- `reduce_main(estimate_fwhm=True)` and `reduce.utilities.estimate_fwhm`
  (use `measure_frame_quality` / `reduce.quality`).

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
