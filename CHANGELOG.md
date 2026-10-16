# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Releases are annotated Git tags `v<version>` on `main`. The process is in
[docs/RELEASING.md](docs/RELEASING.md). GitHub Release notes are the matching
section of this file.

## [Unreleased]

### Added

- **Differential light curves** (`analyze.post_processing.differential_light_curve`,
  `light_curve_differential=True`): per filter a quiet comparison ensemble
  (the quietest field stars of similar brightness to the OOI, all field
  stars rather than only catalog calibrators; robust preselection, then RMS
  without bad epochs) gives the common mode per epoch. `light_curves.ecsv`
  gains `mag_inst` / `mag_inst_err`, `ens_offset`, `ens_offset_err`, `ens_n`,
  `ens_scatter`, `dmag`, `dmag_err`, `dflux`, `dflux_err` and `is_ensemble`;
  differential errors carry no catalog term, and ensemble stars are measured
  leave-one-out. Candidates and members are listed in
  `tables/light_curve_ensemble.ecsv`. New plots `lightcurve_diff_jd_*` /
  `lightcurve_diff_folded_*` and `diagnostics/lightcurves/ensemble_qc_*`.
  On the HAT-P-20 test night the point-to-point scatter is 0.004 mag
  instead of 0.010 mag for `mag_cal_V`.
- **Global bad-epoch flags** `flag_epoch` / `flag_epoch_reason`
  (`ensemble_incomplete`, `transparency`, `ensemble_scatter`) from the
  ensemble common mode, drawn distinct from per-source outliers in all
  light-curve plots. Options `light_curve_ensemble_*` and
  `light_curve_epoch_*`.

- **Scintillation in the photometric errors** (`analyze.scintillation`,
  `scintillation_noise=True`): after the extraction every source gets the
  scintillation term of Osborn et al. (2015, eq. 7, `C_Y = 1.5`) in
  quadrature on `flux_err` / `mags_unc`; the photon / sky error is kept as
  `flux_err_phot`. Aperture from the new telescope catalog
  (`ost_photometry.telescope_specs`, `data/telescopes.json`: CDK20 and the
  mobile OST telescopes from the OST wiki; matched by `TELESCOP`, else
  `FOCALLEN`) or `telescope_aperture_m`; site altitude from
  `observatory_location`. On the HAT-P-20 test night the differential
  scatter was 1.3–2.5× the error bars without it.

### Changed

- **QHY readout mode not in the header (MaxIm DL):** `READOUTM` = `Fast` /
  `Normal` / `Slow` is MaxIm's exposure-tab setting, not the QHY driver
  mode. These labels (and a missing keyword) now mean an unknown readout
  mode; it was silently assumed to be "Extend Fullwell 2CMS" (reduction)
  or replaced by fixed 7.904 e⁻ / 1.292 e⁻/ADU (`camera_info`). New default
  `camera_noise_source="auto"` measures gain and read noise on bias and
  flat pairs for such data, else uses the OST measurements in the new
  `data/camera_measurements.json` (QHY600M, GAIN 0: 1.31 e⁻/ADU, read noise
  per binning). The saturation level comes from the catalog modes that
  agree with the measured values (2CMS modes: 62 386 ADU per native pixel,
  the ADC limit for 2x2 sums). The archive pipeline no longer splits
  electronic setups by these labels unless their bias levels differ. On
  the OST QHY600M data the catalog gain was 16 % too high (1.53 vs
  1.31 e⁻/ADU, photon noise 7 % too small relative to the signal) and the
  bin-3 read noise 50 % too high. On HAT-P-20 the final errors stay right
  (scatter / error 0.93–1.08); the empty-sky factor now carries the extra
  sky noise of the light frames (1.27–1.30, was 1.06–1.11 because the two
  errors hid it).
- **Isochrone fit with an age error** (`analyze.isochrone_fit`): fiducials
  with σ = max(IVW, 1.2533·σ_robust/√N) of the detrended colour scatter and
  a colour–magnitude covariance (was IVW or scatter, else 1.0); χ² as the
  Mahalanobis distance to the nearest isochrone point (was Euclidean
  nearest point); distance and reddening profiled as nuisance offsets with
  Gaussian priors and the data χ² scaled by χ²_ν (were added to every
  star's error bar); age from the parabola vertex and its 1σ interval from
  Δχ² = 1. `tables/cmd_isochrone_fit_*.ecsv` is always written with
  `age_err_lo` / `age_err_hi`, `chi_square_nu`, `delta_m` / `delta_e`,
  `m_m_fit` / `e_b_v_fit`; the info box shows them. Simulated 1σ coverage
  0.69–0.72.
- **CMD input:** epochs are averaged per star (was the first epoch;
  `load_cmd_table(epoch_id=...)` selects one), colour errors and the
  colour–magnitude covariance come from the joint calibration solution, and
  errors are loaded for the fit whether or not error bars are drawn
  (`do_error_bars` only draws).
- **Gaia cluster membership by extreme deconvolution:** with the Gaia
  errors and correlations (`pmRApmDEcor`, `PlxpmRAcor`, `PlxpmDEcor`, now
  queried) the mixture is fitted by XD (Bovy et al. 2011, own EM in log
  space; astroML's `XDGMM` produced NaN on wide fields when stars far from
  every component underflowed all responsibilities) and each star's P_mem
  uses its own covariance. 2–5 components are chosen by BIC (two were not
  enough for a wide field: the field took both and up to 28 000 of 34 000
  stars came out as members); the cluster is the component with the
  highest density at the Simbad (μ, ϖ). Option value `xd` requires the
  errors. In simulations the plain GMM either absorbed half of the field or
  lost the faint members; XD keeps 91–94 % of them at 3 % contamination.
- **Light-curve errors with a stated meaning:** `mag_err` is what changes
  from point to point (`err_cal_rand` ⊕ the new `err_cal_epoch`), the new
  `mag_err_sys` the offset common to all points sharing the calibrators
  (`err_cal_common`; `err_cal_sys` is split into the two, a zero point shared
  by a night or an ensemble is common, `shared_by_epochs` in
  `calibration_epochs.ecsv`). Colour rows use colour errors of the joint
  solution (`err_cal_*_B-V`) instead of adding correlated magnitude errors
  in quadrature. `apply_nightly_zero_point` skips flagged points and writes
  the error of the night median (`night_norm_err`, replaces `mag_err_sys`).
- **Differential ensemble error:** the common-mode error includes the
  members' excess scatter (`1/√Σw`) and a Birge factor from the members'
  scatter about it (the quietest stars underrate their excess); was the
  photometric error of the mean only. `excess_rms` uses `mean(σ²)` and
  `ddof=1` instead of the median error.
- **Outlier flags** use residuals normalised by `√(σ² + s²)` (a point with
  a large error is no longer clipped for it); a MAD of 0 flags nothing
  (flagged every point off the median).
- **Flux fallback (Clear):** analytic relative fluxes against the weighted
  leave-one-out common mode of the field (`relative_flux_light_curves`)
  replace the Monte Carlo `quasi_flux_calibration_*` /
  `flux_normalization_*` functions (removed; missing errors became
  10⁻⁶·flux). `ImageSeries.get_flux_array(return_ids=True)`.
- **Limiting magnitude** is shifted to the table magnitudes by the measured
  aperture/table flux ratio of bright isolated stars (was the aperture flux
  of a radius-r aperture against aperture-corrected / PSF magnitudes).
- **Difference images:** HOTPANTS `-ig` from the image unit (`EXPTIME` for
  e⁻/s, `EGAIN` for ADU; was 1), science σ as `-ini`; both backends write
  the difference σ as `UNCERT`. The Alard–Lupton kernel fit is weighted by
  1/σ.
- **Calibration fits with honest errors** (`analyze.calibration.statistics`,
  used by `linear_fit`, `median_zp`, derive-transform): calibrators weighted
  by instrumental + catalog (`err_std_*`, 0.02 where missing) + colour +
  intrinsic scatter σ_int solved for χ²_ν = 1 (fits were unweighted with
  `ddof=0`); with χ²_ν < 1 and ≥ 10 calibrators the catalog errors are
  scaled down instead (`catalog_error_scale`, APASS B errors are about
  twice the scatter); with clipping the target χ²_ν is the truncated-normal
  variance (0.91 at 2.5σ); colour term fitted around the mean colour; clipping on
  normalised residuals; Student-t factor when σ_int is estimated; n < 5
  flagged `few_calibrators`. Simulated 1σ coverage 0.68–0.70.
- **Error columns:** `err_cal_rand_*` (independent per star) and
  `err_cal_sys_*` (common to the epoch: ZP, T·C, cov, `zero_point_sys`,
  extinction σ_k·ΔX) besides the total `err_cal_*`; all calibrated filters
  are solved jointly (`m = A⁻¹(i+Z)`), so (1 + T) factors and the colour /
  magnitude correlation are included. New `tables/calibration_epochs.ecsv`
  (ZP, T, errors split into photometric / catalog parts, σ_int, χ²_ν,
  `catalog_error_scale`,
  calibrators, flags per epoch and filter).
- **Groupings:** `per_night` combines per-epoch fits by GLS with full
  covariance (the catalog part of the same calibrators no longer averages
  down; Birge ratio; epoch scatter as `zero_point_sys`; was independent IV
  means with cov = 0); `ensemble` fits per-star means of the
  extinction-corrected epochs (each star once; was all epochs stacked);
  `fixed` is implemented (known colour terms from
  `transformation_coefficients_dict`, also in the OST `Tcs` format; zero
  point per epoch; it always raised before). Rolling smoothing keeps the
  covariance and adds the epochs' scatter about the smoothed zero point.
- **Presets** `linear_fit_per_night`, `linear_fit_per_night_extinction`,
  `linear_fit_ensemble` use the linear path with their grouping
  (`derive_transform_from_data=False`; the derive path ignored grouping and
  extinction). `derive_transform_from_data` requires `per_image`.
- `uncertainty_mode`: `fit_errors` (default) and `monte_carlo` (ZP/T drawn
  per epoch, flux and catalog colour per star, joint solution;
  `cal_low_snr_<f>`). `both` (double-counted the instrumental error) and
  `flux_monte_carlo` (dropped ZP/T errors) are removed.
- Aggregated site extinction errors include the night-to-night scatter of
  k; the comparison-star extinction fit uses a fixed star set and a clipped
  median per epoch.
- **Stacks and master frames propagate the frames' uncertainties**
  (`reduce.weighted_combine.combine_images`, replaces `ccdproc.combine` for
  bias, dark, flat, stacks and the camera combination): mean
  `σ² = Σw²σ²/(Σw)²`, median `k(n)·n/(Σ1/σ)²` (k = 1.35 for n = 3 … π/2),
  sum. Before, σ was the scatter between the frames: 0 where one frame
  covered a pixel, biased low for few frames, inflated on stars by seeing
  changes, and without the median factor. Frames without uncertainties (raw
  bias) still use the scatter, now with `ddof=1`. Integer raw frames are read
  block-wise with their `BZERO`/`BSCALE`.
- **Registration resamples the variance** with a linear kernel (masked σ
  filled first) and scales it by the pixel-area ratio k:
  `σ = √(k·interp(σ²))`. WCS reprojection multiplied σ by k (errors 2× too
  small from bin 2 onto bin 1) and astroalign warped σ bicubically (negative
  σ possible). `SATLEVEL` × k, `RDNOISE` × √k and `PIXAREAR` are written.
- Stack headers: `MJD_MID` / `DATE-MID` and `AIRMASS` (with `AIRMSMIN` /
  `AIRMSMAX`) are weighted means over the frames; `RDNOISE` of median stacks
  includes the median factor.
- Frames reduced with darks that carry the bias get the dark shot noise of
  warm pixels and amplifier glow; cosmic-ray pixels interpolated by
  L.A.Cosmic get the σ of their neighbourhood.
- The photon-transfer gain uses neighbouring-pixel differences of the flat
  pair (a changing twilight gradient lowered the gain); clamped camera
  curves warn.
- **APER sky and errors** (`analyze.photometry_quality`): the sky is the
  3σ-clipped median of the annulus with all detected stars masked (was the
  unclipped annulus mean, about −40σ for an 8× brighter neighbour in the
  annulus in simulations; now < 0.2σ). The flux error replaces the
  background part of the map error by the empty-sky measurement
  `bkg_noise_factor` (about 400 sky apertures per image measured like
  stars) and includes the median's (π/2) variance. `background_estimate_simple`
  and the helpers `compute_aperture_photometry_uncertainties` /
  `sky_subtraction_error` are removed.
- **Saturation:** stars with a pixel within the measurement radius at
  `saturation_fraction` (0.9) × `SATLEVEL` of the frame before the 2D
  background subtraction are `saturated`; they are never calibrators or
  differential-ensemble members and are drawn as flagged points
  (`flag_saturated` in `light_curves.ecsv`).
- **Aperture correction per image** to the annulus inner radius from bright
  isolated unsaturated stars (`aperture_correction`,
  `aperture_correction_err`), for APER and PSF.
- **PSF photometry:** local background (annulus 2.8–4 FWHM) on by default;
  `flux_err` scaled by `√max(1, reduced_chi2)`; `reduced_chi2`, `local_bkg`
  kept. The `strict_cleaning_epsf_results` option is removed (invalid
  errors are always dropped).
- **Faint stars with non-positive net flux are kept** (`nonpositive_flux`,
  magnitude NaN) instead of being dropped (flux light curves of faint
  sources were biased upward).
- Masked aperture pixels get σ from their neighbours plus the neighbours'
  spread; calibrators with more than `calibrator_max_masked_fraction`
  (0.05) of the aperture masked are excluded.
- Per-filter quality columns `<name>_<filter>` (`saturated`,
  `nonpositive_flux`, `aperture_masked_fraction`, `flux_err_phot`,
  `reduced_chi2`, `local_bkg`, `sky_sigma_map`, `bkg_noise_factor`,
  `aperture_correction*`) in the epoch tables.
- New `docs/UNCERTAINTIES.md`: error model and validation.
- **Missing `AIRMASS` is computed, not 1.0:** `Image.air_mass` is `None`
  without a usable header value; the extraction step computes it at
  mid-exposure from the WCS centre (else `OBJCTRA`/`OBJCTDEC` or NINA's
  `RA`/`DEC` in degrees), JD and `observatory_location`
  (`extinction.airmass_for_image`, `fill_missing_airmass`). Epoch tables and
  the calibrator use NaN instead of 1.0 when nothing is known. The image
  centre now also reads NINA's `RA`/`DEC` (it fell back to 0/0).
- **Colour of the colour term:** a filter's colour comes from calibrated
  magnitudes only when both colour filters are calibrated in that epoch,
  else from the catalog colour (with `err_std_*`), else NaN; never from
  instrumental magnitudes, which carry the zero-point difference of the
  two filters. A colour term of 0 needs no colour (stars missing in the
  other filter are no longer NaN). New column `cal_color_source_<f>`.
- **No silent error floors:** `weighted_linear_fit` leaves out points with
  non-finite or non-positive weights and returns NaN errors for
  underdetermined fits (was weight 1 and errors 0); the per-night
  inverse-variance combination drops epochs without a finite σ (the
  `inverse_variance_min_error` floor is removed); magnitude-system
  conversions give NaN errors where an input error is missing (was 0);
  the differential ensemble ignores sources without errors.
- Lupton/Jester SDSS→Johnson errors include the published scatter of each
  relation; U is propagated from u, g, r (`johnson_u_from_sloan_ugr`
  replaces `johnson_u_from_sloan_ug_and_b`).
- Phase bins of folded light curves leave out flagged points and use the
  inverse-variance mean; the bin error is the propagated error or, for
  n ≥ 3, the scatter of the mean if larger (never 0).
- Colour light-curve rows inherit `flag_outlier` and `flag_epoch` of both
  filters; the differential product is now built before the colour rows.
- Stacks and reduced frames write `GAIN` as electrons per data unit
  (EXPTIME for e-/s) instead of the raw `EGAIN`; a camera gain setting in
  `GAIN` is kept as `GAINSET`.
- Images without an uncertainty extension get per-pixel errors from the
  frame's sky scatter and the source photons (APER crashed, PSF raised).
- `PipelineConfig` rejects values outside an option's allowed set and warns
  about unknown option names (both were silently accepted).
- Track flux arrays: an id that occurs twice in one image is NaN for that
  image with a warning (the last row won silently).

### Fixed

- **Stacks of frames from different nights were noisy:** the per-pixel 5σ
  clipping kept the frames with the brighter sky in some pixels and rejected
  them in others, so the sky difference became pixel-to-pixel noise (T CrB,
  Clear, 3 + 10 frames from two nights: 2.9× the propagated σ, sky +12 %).
  Stacks now shift every frame to the common sky level before clipping
  (`combine_images(match_background=True)`); afterwards 0.96–1.11× the
  propagated σ in all five filters.
- **ASTAP found no stars on reduced frames with one extreme pixel:** the
  temporary 16-bit copy was scaled to the maximum, so a single masked dead
  flat pixel (10⁹ e⁻/s in a QHY600 I frame) pushed every star to the lowest
  ADU. Masked pixels are set to the background and the scale is capped at
  the brightest extended source (largest 3×3 median among the brightest
  pixels), so single hot pixels are ignored and bright star cores keep
  their order (a cap at a percentile of the image flattened ~760 star
  pixels and made a NGC 7789 B stack unsolvable).
- Archive pipeline: a local directory whose nights are symlinks was read as
  empty (`os.walk` did not follow links; now followed, each real directory
  once); `reduce_planned` crashed writing `reduction_report.ecsv` when the
  output directory did not exist yet.
- **Flux-fallback light curves were labelled with column numbers** 0…n−1
  instead of the photometry ids (wrong star whenever the ids had gaps).
- **HOTPANTS crashed under numpy 2** (`ndarray.newbyteorder` was removed).
- **Alard–Lupton on noisy images:** the stamp FWHM clipped the noise at 0, so
  the positive noise of a 41×41 stamp gave a ~3× too broad PSF, a far too
  large kernel and a 1.5× flux mismatch; it now uses the half-maximum area
  and moments within 2·FWHM.
- **Derive-transform biased stars without catalog magnitudes** by the colour
  term times the instrumental zero-point difference of the two filters (row-
  wise apply with instrumental colour); its errors had no colour-term part.
  It now fits weighted T/ZP and uses the standard joint apply; its fit sees
  the same extinction correction as the apply.
- **`median_zp` with extinction biased every magnitude by −k·X:** the zero
  point was fitted on raw magnitudes while apply subtracted k·X. The median
  backend now fits on the same extinction-corrected magnitudes.
- **Jordi Bessell→SDSS:** the piecewise branches (`V−R ≤ 0.93`,
  `V−I ≤ 1.8`) were chosen with a Python `if` on whole distribution arrays
  (crash or one branch for all stars); they are now chosen per star. The
  `V−R > 0.93` branch of r was missing `+ R`.
- `determine_background` subtracted in e-/s regardless of the image unit;
  dark-frame plots and the hot-pixel mask truncated `EGAIN` to an integer.
- **Night boundary at midnight UT:** `night_id_from_jd` used
  `floor(JD - 0.5)`, which changes at 0 h UT and split every European night
  in two (night bands, per-night zero point in `4_compare_nights.py`,
  per-night transparency model). It is now `floor(JD)` (noon UT to noon UT).
  Combining and comparing nights recompute `night_id` from `jd`, so tables
  written before the fix are grouped correctly. Night labels give the
  evening date instead of the date of the median time. The calibration
  timelines of the archive pipeline started at midnight UT although the axis
  reads "hours after noon UT"; they now start at noon UT.

## [0.6.0] - 2026-10-12

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
- `storage_dtype` (`reduce_main`, `ReductionSettings`; module `reduce.storage`):
  floating type of written masters, reduced / aligned frames and stacks.
- Archive pipeline: `StackSettings.keep_reduced_lights` (default `False`)
  deletes the reduced frame of every aligned frame that is kept, so each
  frame is stored once.
- Archive pipeline: darks for science exposures without matching darks
  come from the archive's **dark finder** (`ArchiveClient.find_darks`,
  `fetch.fetch_missing_darks`, `use_dark_finder=True`, needs a login).
  The grouping accepts covering darks outside the calibration window.
  Flat and dark notes name the sets and their source runs; low-probability
  flats used with `no_flat_policy="best_available"` say so and list the
  candidates that were not used, with their reasons.
- Archive pipeline: **only complete calibration is reduced**
  (`PlanSettings.require_complete`, `accepted_flat_categories`): lights
  without matching darks or without an accepted flat for their session are
  marked incomplete and skipped (`overrides.force_units`,
  `ReductionSettings.reduce_incomplete`); `missing_calibrations.ecsv` lists
  the dark / bias series to take.
- Archive pipeline: bias frames from the archive finder
  (`ArchiveClient.find_calibration_frames`, `fetch_missing_calibrations`);
  dark / light exposure tolerance `max(0.5 s, 5 %)` in fetch, grouping and
  reduction.
- Archive pipeline: `reduce.workflow.combine.restack_planned` /
  `restack_target` (script `4_restack.py`) re-stack registered frames with
  another selection / weighting; `StackSettings.align_rejected` registers
  rejected frames too so the selection can be loosened later.
  `3_reduce_and_stack.py` has the full frame-quality parameter block of the
  imaging script.
- `ArchiveClient(log=)` reports retries (timeouts, HTTP 429 / 5xx with the
  wait time) and `fetch_dataset(log=)` / `collect_records(log=)` each query
  step; `1_fetch.py` prints both, so slow archive answers no longer look
  like a hang. File listings are paged with 200 entries, a page that times
  out is requested again with half the size, run listings ask for FITS files
  only, and `1_fetch.py` has `archive_timeout` (120 s).
- **Colour images** (`ost_photometry.imaging`, scripts
  `astro_imaging/3_color_composite.py`, `archive_pipeline/5_color_composite.py`):
  palettes RGB, LRGB, SHO, HOO and custom mixes; sky neutralisation, crop to
  the common field, cold-pixel correction, optional polynomial /
  `Background2D` gradient removal and star-based white balance; auto
  (midtone), asinh or linear stretch, linked or per channel; 16-bit TIFF,
  PNG, JPEG. See `docs/COLOR_COMPOSITE.md`. `tifffile` and `pillow` are
  declared dependencies.
- Header keywords `RDNOISE` / `SATLEVEL` (electrons for the image times
  `EXPTIME`) on reduced frames, stacks and camera combinations; sigma-clipped
  stacks of at least three frames are marked `CRIDENT` / `CRCLIP`.

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
- Images are written as **float32** by default (`storage_dtype="float32"`,
  was float64): half the disk space and I/O; the rounding (relative 6e-8)
  is far below the pixel noise. Calculations still run in float64;
  alignment and stacking keep the float type of their input. Pass
  `storage_dtype="float64"` for the old behaviour.
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
- The mask warning after alignment counted the part of the reference field
  a frame does not cover (pointing offset, dither, meridian flip) as masked
  pixels (e.g. "21.5 % of interior pixels are masked"). It now warns only
  about masked pixels inside the frame's own field and reports large
  offsets as a note.
- **Weighted average stacks** (`stack_weighting != "none"` with
  `stack_method="average"`, and the camera combination) were too dark
  wherever some frames were masked or clipped: ccdproc divides by the
  weights of the rejected values too. Holes and dark columns in the
  background, star flux about 1 % low. Now computed by
  `reduce.weighted_combine.weighted_average_combine` (mask-aware, in row
  blocks of at most 2 GB); see `docs/TODO.md` for the ccdproc check.
- Telescope names: "OST CDK20" was not recognised as the CDK20 (only
  "Planewave CDK20"), so the grouping treated it as another telescope
  (separate sessions, flats rejected as "different camera or telescope").
  `setup_keys.telescope_id` also recognises the model inside longer names.
  Timeline plots use the normalised telescope and draw context frames
  (archive metadata without header) inside the timelines instead of extra
  "unknown" ones.
- QHY readout modes with the newer SDK names (`High Gain Mode 16BIT`,
  `Photographic DSO 16BIT`, `Extend Fullwell Mode`) were not found in the
  camera catalog, so gain and read noise fell back to 1.292 e-/ADU and
  7.904 e-. `normalize_readout_mode` now ignores the `16BIT` / `Mode`
  words.
- **L.A.Cosmic saturation in the reduction** was the raw ADU limit applied to
  calibrated electrons; it is now `(saturation - bias) x gain / brightest
  flat`, so saturated star cores are protected.
- **L.A.Cosmic in the analysis** used 8 e- and 65535 for every image,
  including stacks whose read noise and saturation scale with the number of
  frames. `read_noise` / `saturation_level` default to `None` and come from
  `RDNOISE` / `SATLEVEL`.
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

[Unreleased]: https://github.com/OST-Observatory/ost_photometry_package/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/OST-Observatory/ost_photometry_package/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/OST-Observatory/ost_photometry_package/compare/2648b88...v0.5.0
[0.4.4]: https://github.com/OST-Observatory/ost_photometry_package/commit/2648b88
