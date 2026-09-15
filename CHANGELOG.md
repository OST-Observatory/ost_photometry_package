# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Releases are annotated Git tags `v<version>` on `main`. The process is in
[docs/RELEASING.md](docs/RELEASING.md). GitHub Release notes are the matching
section of this file.

## [Unreleased]

### Added

- Intra-filter **track QC** (`diagnostics/correlation/track_qc_<filter>`) and
  identity checks (pixel / on-sky scatter).
- `shift_method="wcs"`: reproject science frames onto the reference celestial WCS.
- Sparse tracks (`require_complete_intersection=False`, `min_detection_fraction`)
  and pixel-coordinate correlation when frames share a grid.
- Independent OOI match radius (`ooi_separation_limit`).
- Alard–Lupton image subtraction (alternative to HOTPANTS), multi-night light
  curves, FWHM-scaled apertures, camera catalog rebuild helpers.
- Python 3.13 in the CI test matrix.

### Changed

- Bad-pixel masks: only significantly negative pixels after dark subtraction;
  fill defects before resampling; warp masks nearest-neighbour so they do not
  grow. Warn if more than 10 % of the frame interior stays masked after align.
- Sequential correlation bridges single-frame gaps; extinction is applied in
  the colour-term fit even when `color_term_fit="never"`.
- Calibration diagnostics cap per-epoch PDFs; many plots run in a subprocess
  so they do not block the pipeline.

### Fixed

- Reduce yes/no prompts (reuse masters / previous science frames) time out
  again after 30 s and default to ``no``, using stdlib ``select`` instead of
  the optional ``pytimedinput`` package (without it, ``input()`` blocked forever).
- Aperture uncertainty combination (hypotenuse of variances, not of sigmas).
- PSF photometry non-finite pixel errors; extraction skip of unusable frames;
  light-curve outlier flagging around eclipse dips; Clear-filter light curves;
  OOI photometry IDs after sparse correlation.

## [0.4.4] - 2026-07-07

Last version on `main` before this changelog. It was not tagged; detailed notes
start with later versions. Git history before that date remains the record.

[Unreleased]: https://github.com/OST-Observatory/ost_photometry_package/compare/2648b88...HEAD
[0.4.4]: https://github.com/OST-Observatory/ost_photometry_package/commit/2648b88
