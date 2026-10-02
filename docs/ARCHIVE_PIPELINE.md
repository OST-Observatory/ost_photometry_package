# Archive pipeline: fetch, group, reduce, stack

Reference for `ost_photometry.archive`, `ost_photometry.reduce.grouping`
and the group-wise reduction (`reduce.workflow.groups`, `reduce.workflow.combine`).
The scripts live in `auxiliary_scripts/archive_pipeline/`
(`1_fetch.py`, `2_classify_and_group.py`, `3_reduce_and_stack.py`).

## Why

At the OST the instruments (cameras, spectrographs, eyepieces) share one
focus and are exchanged often, sometimes several times per night.
Calibration frames from other days apply only if the camera stayed mounted
unchanged. The pipeline therefore decides per frame which bias, dark and
flat frames belong to it, and with which probability.

## Findings behind the design (archive test data, 15 nights)

| Night | QHY600 orientation mod 180° |
|-------|------------------------------|
| 2021-02-24 | 18.2° |
| 2022-03-08 | 28.4° |
| 2022-06-23 | 20.0° |
| 2022-12-14 | 3.4° |
| 2022-12-26 | 69.9° |

Orientation = position angle of the image +y axis, north through east
(`wcs.position_angle_from_wcs`), as reported in `calibration_plan.yaml`.

- Within a night the plate-solved orientation is stable to about 0.2°, also
  across meridian flips (180° apart). Between nights it changes by 8°–66°.
- Flats reveal a rotation only above roughly 90°: the sharp dust donuts sit
  on the sensor window and rotate with the camera; only the large-scale
  vignetting is telescope-fixed, and its differences between filters of
  one night are as large as a 90° rotation. A known 7° rotation was
  recovered as 8.25°.
- The dust donuts are a good fingerprint of "same camera, same dust state":
  the high-pass maps of flats of different filters of one night correlate
  at r ≈ 0.6 (the noise limit of two 5-frame masters), a changed pattern at
  r ≈ 0.
- Header image types are unreliable (night-time lights labelled `Flat
  Field`), and offset, readout mode and temperature change between nights.

## Modules

| Module | Role |
|--------|------|
| `archive.client` | `ArchiveClient`: session + CSRF login (`OST_ARCHIVE_USER` / `OST_ARCHIVE_PASSWORD` or prompt), runs, data files, object search, headers, checksum-verified downloads, rate limiting and `Retry-After` |
| `archive.fetch` | `fetch_dataset(object_name= / run_name=, targets=, calib_window_days=)`: science frames, calibration candidates of the runs and neighbouring runs, context metadata |
| `archive.cache` | content-addressed read-only cache, unique symlink names |
| `archive.manifest`, `archive.local` | manifest schema (archive + header fields), `manifest_from_directory` for local trees |
| `reduce.grouping.classify` | frame type from image statistics (stars with spatial extent, level above bias, saturation, spectral structure) confirmed by header / archive ML / user types |
| `reduce.grouping.setup_keys` | camera / telescope ids, electronic key, temperature clusters |
| `reduce.grouping.targets` | targets by sky position (single-linkage on field centres), naming, merge / rename |
| `reduce.grouping.orientation` | sampling, archive WCS or `wcs.solve_astap_copy`, cache, bisection of changes |
| `reduce.grouping.sessions` | mount sessions (camera change, other camera in between, orientation / parity / scale change, manual breaks) |
| `reduce.grouping.darks` | electronic ids, bias / dark sets, nearest covering night |
| `reduce.grouping.flats` | flat sets, dust / vignetting metrics, probability per session |
| `reduce.grouping.plan` | orchestration, masters, reduction units, YAML plan with overrides |
| `reduce.grouping.plots` | timeline per telescope and night (`diagnostics/calibration_groups/`) |
| `reduce.workflow.groups` | masters per group, lights per unit with explicit masters and pixel masks |
| `reduce.workflow.combine` | per target: quality selection, one grid, weighted stack per camera and filter, optional camera combination |

## Grouping levels

| Level | Key | Used for |
|-------|-----|----------|
| Electronic setup | camera, binning, readout mode, gain, offset, temperature (±2 K) | bias / darks, reusable across nights |
| Mount session | camera, telescope, orientation mod 180°, parity, pixel scale | time block without remounting |
| Flat group | camera, binning, filter, mount session | flats |
| Target | cluster of field centres | reference, selection, stack |

Calibration runs per reduction unit (mount session × electronic setup); all
targets of a night share the masters. Stacking runs per target × camera ×
filter over all sessions and nights.

A session ends at a camera or pixel-scale change, at an orientation change
above `pa_tolerance` (0.5°), or when another camera was used at the same
telescope in between (archive context frames). Unsolved frames between two
different sessions are left out (`session_id = unknown`).

## Flat probability

```
P = odds / (1 + odds),  odds = p0 / (1 - p0) * LR_dust * LR_vignetting
p0 = exp(-dt / tau)  (tau = 2 days, dt = gap to the session)
LR_dust = 3 if r_norm >= 0.6,  0.2 if r_norm <= 0.2,  else 1
LR_vignetting = 0.1 if the large-scale log-ratio rms > 2 x 0.004
```

`r_norm` is the dust-map correlation with flats that certainly belong to
the session (any filter), divided by the half-set self-correlations so
that noisy, low-signal flats are not mistaken for a changed dust state.
Flats taken during the session are certain; flats with another session or
camera in between, or with a different binning, are rejected.
Categories: certain ≥ 0.95, likely ≥ 0.7, uncertain ≥ 0.4.

## Plan file

`calibration_plan.yaml` holds sessions, flat sets with all candidates and
probabilities, masters, units, targets and an `overrides` block
(`exclude_frames`, `frame_types`, `session_breaks`, `merge_sessions`,
`force_flats`, `merge_targets`, `rename_targets`, `no_stack_targets`).
Overrides are read back when the grouping is rebuilt. Plate solutions are
cached in `orientation_cache.ecsv`.

Header stamps of reduced lights: `FRAMEID`, `CALUNIT`, `SESSID`, `FLATGRP`,
`PFLAT`, `TARGETID`, `TARGET`. Stacks: `TARGETID`, `CAMERA`, plus the
frame-quality keywords (`WEIGHTNG`, `NREJECT`, `FWHMMED`, …); camera
combinations add `NCAMERAS` / `CAMERAS`.

## Detector noise and cosmic rays

`ReductionSettings(camera_noise_source="measured")` measures the read noise
per electronic setup (bias pairs, else dark pairs) and the gain per gain
setting (camera, binning, readout mode, gain; flat pairs of all offsets,
each against the zero level of its own setup), see
`reduce.workflow.groups.measure_plan_noise`. The default `catalog` scales the
catalog read noise to the binned pixel (`binning_mode`). With
`rm_cosmic_rays="auto"` (default) L.A.Cosmic runs only on frames whose stack
(target × camera × filter) has fewer than `cosmic_ray_auto_min_frames`
frames, or whose target is not stacked.

## Limits

- Spectroscopy frames are excluded (archive classification, or spectral
  structure along one axis in the image statistics).
- Moving objects: `target_grouping="archive_object"`; tracking the object
  in the stack is not implemented.
- Mount sessions are not written back to the archive (planned extension).
