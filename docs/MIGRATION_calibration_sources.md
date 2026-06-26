# Migration: shared `calibration_sources` module

## Overview

Calibration catalog download and normalization live in
`ost_photometry.analyze.calibration_sources`. Tables use a **standard schema**:
`ra` / `dec` (ICRS, degrees), `mag_std_{filter}` / `err_std_{filter}`, optional
Sloan columns `mag_std_g`, `mag_std_r`, `mag_std_i`, and Lupton Johnson `R`/`I`
where applicable.

Legacy `derive_calibration` still receives `(Table, column_dict, ra_unit)` via
`standard_catalog_to_legacy`.

## Breaking renames

| Removed / old | New |
|---------------|-----|
| Module `analyze.calibration_differential_catalog` (removed) | `analyze.differential_photometry` |
| `PipelineConfig.magnitude_range` | `calibration_catalog_mag_range` |
| `PipelineConfig.differential_apass_radius` | `calibration_catalog_radius_arcmin` |
| `PipelineConfig.differential_apass_mag_limit` | (use tuple upper bound in `calibration_catalog_mag_range`) |
| `derive_calibration(..., magnitude_range=...)` | `calibration_catalog_mag_range=...` |
| `load_calibration_data_table(..., magnitude_range=...)` | `calibration_catalog_mag_range=...` |
| `PhotometryCalibrator.setup_apass(...)` | `setup_calibration_source(..., calibration_method=..., radius_arcmin=..., calibration_catalog_mag_range=..., vizier_dict=..., path_calibration_file=...)` |
| `APASSCatalog` (public) | Removed; use `calibration_sources.fetch_standard_calibration_catalog` and `crossmatch_standard_catalog` |

Differential calibration previously used a single faint limit (`mag_limit=16`);
that is now the **upper** end of `calibration_catalog_mag_range`, e.g.
`(0.0, 16.0)`, if you need the same cut.

## Vizier API

`get_vizier_catalog` now takes `center: SkyCoord` and
`field_of_view_arcmin: float` (no `image_like_object`).

## Lupton (Sloan → Johnson R/I)

Implemented in `calibration_sources.transforms` and applied for APASS in
`fetch.py`, and optionally for other Vizier catalogs that have Sloan `r`/`i`
but not Johnson `R`/`I` when `apply_sloan_to_johnson_ri=True` (default).
