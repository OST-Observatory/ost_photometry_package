# Uncertainties

This page describes how each uncertainty in `ost_photometry` is computed, from
the pixel to the published magnitude, and what was done to check it on real
data. The aim is that every error column is a 1σ uncertainty with a stated
meaning:

- **random:** independent for every measurement;
- **systematic:** shared by all stars of an epoch or a night.

A missing value is NaN, never a silent 0 or 1.

See also [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md) (options) and
[DIAGNOSTICS.md](DIAGNOSTICS.md) (plots and tables).

## Rules

- **Unknown is NaN.**
  - A missing error stays NaN; it is never replaced by 0 or a tiny floor.
  - A missing `AIRMASS` is computed from pointing, time and site (or is NaN);
    it is never 1.0.
- **Bad measurements are flagged, not used.**
  - Saturated, masked or non-positive measurements carry a flag.
  - They are never calibrators or ensemble members.
- **Random and systematic terms stay apart.**
  - Terms common to an epoch (zero point, colour term, extinction, aperture
    correction) are kept apart from the per-star random error, so that light
    curves and colour–magnitude diagrams can use the right part.

## Pixel level (reduction)

`UNCERT` is the per-pixel σ of every reduced frame:
`σ² = gain·S + RN²` in electrons on the bias-free signal, plus the master
frame errors in quadrature.

The read noise is that of the binned pixel: `√(xbin·ybin)` × the native read
noise for digitally binning CMOS. Frames in e⁻/s carry `RDNOISE` and
`SATLEVEL` in electrons for the image times `EXPTIME`.

### Master frames and stacks

`reduce.weighted_combine.combine_images` combines frames by mean, weighted
mean, median or sum. It propagates the frames' own `UNCERT` over the values
that survive masking and 5σ clipping:

```
mean:   σ² = Σ w_i² σ_i² / (Σ w_i)²
median: σ² = k(n) · n / (Σ 1/σ_i)²     k = Var(median)/Var(mean): 1.35 (n=3) … π/2
sum:    the mean × n
```

- **Raw bias frames:** they carry no `UNCERT`, so their master gets the
  scatter between frames, `std(ddof=1)/√n`.
- **Flats:** each frame is scaled to median 1, data and σ alike.
- **Science stacks:** every frame is first shifted to the common (weighted
  mean) sky level. Otherwise frames from nights with different sky
  brightness are clipped in some pixels and kept in others, and the sky
  difference turns into pixel-to-pixel noise.
- **Why it changed:** the earlier stacks and masters used the scatter
  between frames for every pixel. That gave σ = 0 where one frame covered a
  pixel and was biased low for few frames (0.56 for n = 2). On stars it was
  inflated by seeing changes between frames, and for medians the factor
  π/2 was missing.

Stack headers:
- `MJD_MID` / `DATE-MID` and `AIRMASS` are the weighted means over the
  frames, with `AIRMSMIN` / `AIRMSMAX` as the range.
- `RDNOISE` of a median stack includes `√k(n)`.

### Registration

- **Method:** the *variance* is resampled with a linear kernel, after masked
  pixels got their neighbours' variance, and scaled by the pixel area ratio
  `k` (new/old pixel area): `σ_out = √(k·interp(σ²))`. Aperture sums keep
  their variance, also onto a finer grid.
- **Header keywords:** they follow the same `k`: `SATLEVEL` × k,
  `RDNOISE` × √k, and `PIXAREAR` = k.
- **Why it changed:**
  - WCS reprojection used to scale σ by `k` itself, which made errors too
    small by a factor of 2 when projecting bin 2 onto bin 1.
  - astroalign warped σ bicubically, which can overshoot to negative values
    near hot pixels.

### Other reduction terms

- **Darks without bias frames:** the dark shot noise of the frame's own
  dark current is added, with the signal estimated as the master dark above
  its median. This covers warm pixels and amplifier glow.
- **Cosmic-ray pixels:** pixels interpolated by L.A.Cosmic get the σ of
  their neighbourhood, not the Poisson σ of the cosmic.
- **Photon-transfer gain:** measured from differences of neighbouring
  pixels of a flat pair, so a twilight gradient that changes between the
  two flats does not lower the gain.

## Extraction

### Aperture photometry

- **Sky:** 3σ-clipped median of the annulus. Pixels within 2 FWHM of every
  detected star are masked, but always inside the annulus inner radius, so a
  star never masks its own sky.
  - *Why:* the earlier unclipped annulus mean put about half of a
    neighbour's annulus flux into the target. In a simulation with an 8×
    brighter neighbour at 3 FWHM that was a −40σ bias; it is now below
    0.2σ.
- **Flux error:**
  ```
  σ_F² = σ_source² + f²·(A·σ_sky² + (π/2)·A²·σ_sky²/n_sky)
  ```
  - `σ_source` is the source part of the map error.
  - `σ_sky` is the median map σ in the annulus.
  - `π/2` is the variance of a median relative to a mean.
  - `f` (`bkg_noise_factor`, applied as `max(1, f)`) is measured on every
    image with about 400 apertures on empty sky, measured exactly like
    stars. Faint undetected stars, flat-field structure, scattered light and
    noise correlated by resampling all raise it.
- **Masked pixels:** filled with the median of their neighbours. Their σ is
  the neighbours' σ plus the neighbours' spread (`1.4826·MAD`). The masked
  fraction of each aperture is kept as `aperture_masked_fraction`.
- **Faint stars:** stars with zero or negative net flux are kept
  (`nonpositive_flux`, magnitude NaN). Dropping them biased faint flux
  light curves upward.

### PSF photometry

- **Local background:** fitted in an annulus at 2.8–4 FWHM, on top of the
  80 px `Background2D` map.
- **Flux error:** the formal fit error scaled by `√max(1, reduced_chi2)`.
  Where the ePSF does not describe the star (mismatch, blends), the formal
  error is too small.

### Both methods

- **Saturation:** `saturated` is set when a pixel within the measurement
  radius reaches `saturation_fraction` (0.9) × `SATLEVEL`. The test uses the
  frame *before* the 2D background subtraction, so the sky counts towards
  saturation.
- **Aperture correction per image:** the flux is scaled to the flux inside
  the annulus inner radius. The factor is measured on bright, isolated,
  unsaturated stars (`aperture_correction` ± `aperture_correction_err`).
  - *Why:* the flux outside a fixed aperture (or outside the ePSF
    normalisation radius) changes with seeing.
  - Per-image calibration absorbs this change; per-night and ensemble
    calibrations would see it as variability.
- **Scintillation:** added to every source after extraction (Osborn et al.
  2015, eq. 7, `C_Y = 1.5`); see `scintillation_noise`. The error without it
  is kept as `flux_err_phot`.
- **Missing `UNCERT`:** pixel errors are estimated as sky scatter plus
  source photons.

These quality values reach the epoch tables as `<name>_<filter>` (for
example `saturated_V`) and the light-curve rows as `flag_saturated`.

## Calibration

### Zero point and colour term (`analyze.calibration.statistics`)

For each calibrator, `Δ = m_std − m_inst = ZP + T·C + ε` with

```
Var(ε) = σ_inst² + σ_cat² + T²·σ_C² + σ_int²
```

- `σ_inst` is the instrumental error (photon, sky, scintillation).
- `σ_cat` and `σ_C` are the catalog errors of the magnitude and the colour
  (0.02 mag where the catalog has none).
- `σ_int` is the intrinsic scatter: catalog systematics, colour-term
  mismatch, crowding. It is solved so that χ²_ν = 1, so the weights are
  absolute.
- When the errors over-explain the scatter (χ²_ν < 1 at σ_int = 0) with at
  least 10 calibrators, the catalog errors are scaled by a factor s < 1
  until χ²_ν = 1 (`catalog_error_scale`). The instrumental errors are
  validated separately (see below); the catalog errors are not. On the OST
  test fields APASS overestimates the B errors (s = 0.29–0.87) and in the
  HAT-P-20 field also the V errors (s ≈ 0.54).
- With clipping at kσ the target is the variance of a normal truncated at
  ±k instead of 1 (0.91 at the default 2.5σ), since the kept residuals
  scatter less than their errors.
- `T` is fitted around the mean colour.
- When σ_int had to be estimated, the errors get the Student-t 1σ factor.

Simulated 1σ coverage of ZP and T:

| Case | Coverage |
|---|---|
| 25 or 8 calibrators | 0.68–0.70 |
| 4 calibrators, ZP only | 0.70 |
| 4 calibrators, errors known exactly | 0.76 (conservative) |
| 40 / 15 calibrators, catalog errors twice too large | 0.68 / 0.68 (s ≈ 0.5) |

The covariance of (ZP, T) is split in two parts:
- **photometric:** photon, sky and scintillation of the calibrators;
  independent from epoch to epoch;
- **catalog:** catalog errors and σ_int of the same calibrators; common to
  all epochs that use them.

### Groupings

- **`per_night`:** generalised least squares of the per-epoch fits. The
  photometric parts average down, the catalog part does not. A Birge ratio
  covers epochs that scatter beyond their errors, and that extra scatter is
  `zero_point_sys` of every epoch.
- **`ensemble`:** one fit to the per-star means of the extinction-corrected
  epochs, so each catalog error counts once.
- **`fixed`:** known colour terms (campaign), zero point per epoch, σ_T
  kept.
- **Rolling smoothing:** keeps the covariance and adds the epochs' scatter
  about the smoothed zero point as `zero_point_sys`.
- **Shared zero points:** with extinction applied, σ_k·(X − X_ref) is added
  to the systematic error. The aggregated site k error includes the scatter
  from night to night.

### Error columns

- **`err_cal_rand_<f>`:** independent from star to star. It is the
  instrumental error and, with a catalog colour, T²σ_C².
- **`err_cal_sys_<f>`:** common to the epoch. It is ZP, T·C, cov,
  `zero_point_sys` and the extinction term. It is split further:
  - **`err_cal_epoch_<f>`:** changes from epoch to epoch: the photometric
    part of a zero point fitted to this epoch alone (`per_image`, `fixed`)
    and `zero_point_sys`;
  - **`err_cal_common_<f>`:** one offset for all epochs that share the
    calibrators: the catalog part (catalog errors, σ_int), a zero point
    shared by several epochs (`per_night`, `ensemble`, rolling smoothing;
    `shared_by_epochs` in `calibration_epochs.ecsv`) and the extinction term.
- **`err_cal_<f>`:** all in quadrature.
- **`err_cal_rand_<c1>-<c2>` etc.:** the same four columns for each colour
  index of the joint solution (e.g. `B-V`). The two magnitudes are
  correlated through the colour terms; the colour error uses
  `∂(m₁ − m₂)/∂x = A⁻¹₁ − A⁻¹₂` instead of adding the magnitude errors in
  quadrature.

All filters whose colours are calibrated are solved jointly:
`m = A⁻¹(i + Z)`, `A = I − diag(T)K`. The (1 + T) factors and the
correlation of a colour with its own magnitude are therefore included.
Checked against a Monte Carlo of the instrumental magnitudes and of ZP/T
(agreement within 5–6 %). `uncertainty_mode="monte_carlo"` repeats that
Monte Carlo on real tables as a cross-check (non-Gaussian low-S/N errors,
`cal_low_snr_<f>`).

## Light curves and post-processing

### Transformed product (`mag`)

- **`mag_err`:** what changes from point to point,
  `err_cal_rand ⊕ err_cal_epoch`. The scatter of a constant star about its
  mean should match it (HAT-P-20: 0.98–1.09, see the validation table).
- **`mag_err_sys`:** the offset common to all points that share the
  calibrators (`err_cal_common`). It moves the whole curve and does not
  belong in a fit of the shape.
- **Colour rows (`B-V`):** from the colour columns of the joint solution;
  where they are missing (catalog colour, instrumental tables) the
  components are added in quadrature.
- **Nightly zero point** (`apply_nightly_zero_point`): the night median
  skips flagged points. Its error `1.2533·σ_robust/√n` is common to the
  night and replaces `mag_err_sys` (`night_norm_err`): the common
  calibration offset cancels in the difference to the night median.

### Differential product (`dmag`)

- The ensemble common mode is the weighted mean with weights
  `1/(σ² + excess²)`. Its error is `1/√Σw`, so the members' excess scatter
  counts.
- The members are the quietest stars, so their excess is biased low. A
  Birge factor `√(mean χ²_ν)` of the members about the common mode, trimmed
  over epochs and at least 1, scales the common-mode error.
- `dmag_err = √(σ² + σ_cm²)`; ensemble members use the leave-one-out
  common mode.
- `excess_rms` is `√max(0, var − mean(σ²))` (ddof = 1). The median error
  underrated the expected variance for mixed errors.

### Outlier flags

Residuals are normalised by `√(σ² + s²)`, where `s` is the series' own
scatter beyond the errors (from the MAD). A point with a large error is not
flagged for its error alone. A series with MAD 0 (constant or quantised)
flags nothing.

### Flux fallback (filters without catalog calibration)

`relative_flux_light_curves` uses the same common mode as the differential
product:
- the ensemble is all stars detected in at least `min_detection_fraction`
  of the epochs;
- points beyond 5σ do not enter;
- ensemble stars are measured leave-one-out.

Errors are analytic: the star's own flux error ⊕ the common-mode error. A
missing flux error stays NaN; the former Monte Carlo replaced it by
10⁻⁶·flux. Columns are labelled with the photometry ids; the former
labels 0…n−1 did not match the ids of the tables.

### Limiting magnitude

ImageDepth measures the 5σ flux in an aperture of radius r, but the table
magnitudes are total fluxes (aperture-corrected APER or PSF). The limit is
shifted by `2.5·log₁₀ ρ`, where ρ is the aperture/table flux ratio of
bright, isolated, unsaturated stars (median, error `1.2533·σ/√n`).

### Difference images

- **HOTPANTS:**
  - `-ig` is the number of electrons per image unit (`EXPTIME` for e⁻/s
    images, `EGAIN` for ADU; it was always 1);
  - the science σ is passed as `-ini`, and the difference σ (`-oni`) is
    stored as the `UNCERT` extension of the difference image.
- **Alard–Lupton:**
  - the kernel fit weights every pixel by `1/σ` of the fitted image (the
    `UNCERT` image, or the stamp's border σ);
  - the difference carries `UNCERT`: the science σ when the template is
    convolved, `√(K² ⊗ σ²)` when the science image is.
- The template noise is not included (HiPS cutouts carry no noise model and
  are much deeper).

## Colour–magnitude diagrams

### Input

- Several epochs are averaged per star (`combine_cmd_epochs`, default of
  `load_cmd_table`; was the first epoch). Weights are the point errors;
  the error of the mean is `√(max(1/Σw, var/n) + mean(common²))`, so the
  catalog part does not average down and variable stars keep their scatter.
- The colour error comes from the joint calibration solution
  (`err_cal_<f1>-<f2>`), and `cov(colour, mag)` follows from the three
  variances (`−σ₂²` for independent filters).
- Errors are always loaded; `do_error_bars` only controls drawing. The fit
  used no errors when error bars were off.

### Isochrone fit (`analyze.isochrone_fit`)

- **Fiducials per magnitude bin:**
  - colour: IVW mean with σ = max(IVW error, 1.2533·σ_robust/√N) of the
    colour scatter about a linear trend in the bin (the sequence slope is no
    error);
  - magnitude: the IVW error;
  - their covariance;
  - no fallback σ of 1.
- **χ²:** Mahalanobis distance of each fiducial to its nearest isochrone
  point (isochrone densified, gaps between branches not filled). Each
  fiducial slides along the isochrone, so `ν = n_fid − 1 − n_free`.
- **Distance and reddening** are nuisance offsets (δM: distance modulus and
  the R_V part of the extinction; δE: colour excess), profiled per isochrone
  with Gaussian priors from `m_m_err` / `distance_err`, `e_b_v_err`,
  `rv_err`. They are no longer added to every star's error bar. Two passes:
  1. offsets fixed → `s = max(1, χ²_ν)`;
  2. offsets free, data χ² divided by `s`, so fiducials that are too
     optimistic cannot drag the offsets away from their priors.
- **Age:** vertex of a parabola through the χ² minimum on the age grid and
  its neighbours; the 1σ interval where the profile rises by 1 (after
  scaling with s).
- **Output:** `tables/cmd_isochrone_fit_*.ecsv` is written for every fit and
  contains `best_age`, `age_err_lo` / `age_err_hi`, `chi_square_nu`, `dof`,
  `chi_square_scale`, `delta_m` / `delta_e`, `m_m_fit` / `e_b_v_fit`. The
  info box and the terminal show them too.
- **Fit range:** binning in magnitude assumes one colour per magnitude. Keep
  the giant branch and the red clump out of `magnitude_fit_range`; fit the
  main sequence and the turn-off.

### Cluster membership

Extreme deconvolution in (μ_α*, μ_δ, ϖ) with each star's Gaia covariance
(errors and the three correlations) is described in
[DIAGNOSTICS.md](DIAGNOSTICS.md#cluster-membership-gaia).

## Not included

- **Vega/AB offsets:** taken from Blanton & Roweis (2007) without an
  uncertainty. Their real uncertainty is how the OST passbands differ from
  the ideal ones (TODO: synphot with OST throughput curves).
- **Template noise in difference images** (see above).
- **Jordi et al. (2006) relations:** no intrinsic scatter is given in the
  paper, only coefficient errors; the coefficient errors are propagated.
  Lupton/Jester relations carry their published scatter.

## Validation

| Check | Data | Result |
|---|---|---|
| APER pulls, simulated frames (Poisson + read noise, neighbour in annulus) | `tests/test_aperture_uncertainty.py` | pull σ 1.00 ± 0.1; neighbour bias < 0.2σ |
| Empty-sky factor, white noise / added structure | same | 1.0 ± 0.15 / > 1.3 |
| Computed vs header `AIRMASS` | HAT-P-20, 10 V frames | ≤ 0.0011 |
| Stack pulls, simulated frames (mean, weighted, median, 1 frame, no `UNCERT`) | `tests/test_weighted_combine.py` | σ 1.00 ± 0.06 (median ± 0.1) |
| Aperture variance after reprojection bin 2 → bin 1 | `tests/test_wcs_align.py` | 1.00 ± 0.15 (old scaling: 0.5) |
| Pixel pulls between two independent 10-frame stacks (sky pixels) | NGC 7789 B/V (N2) | 0.87–0.89 (interpolation smooths pixels); 4×4 sums 1.14–1.20 |
| Empty-sky noise factor on the stacks | NGC 7789 | 1.7–2.1: structured background at airmass 2.1, captured by `bkg_noise_factor` |
| Faint-star pulls between the two stacks (APER, with the factor) | NGC 7789 B / V | 0.87–1.17 / 1.09 (bright stars probe PSF differences, not noise) |
| ZP / T 1σ coverage, simulated calibrators (σ_int, catalog errors) | `tests/test_calibration_statistics.py` | 0.68–0.70 |
| `err_cal_rand` / `err_cal_sys` vs Monte Carlo (joint B/V solution) | `tests/test_calibration_color_source.py` | within 5–6 % |
| `monte_carlo` mode vs analytic at S/N 200 | `tests/test_uncertainty.py` | within 5 % |
| Robust scatter / error of all constant stars, X < 1.3, six magnitude bins (package 2: median sky, empty-sky factor, scintillation) | HAT-P-20 C7 night, 355 (V) / 320 (B) stars | 0.97–1.01; brightest V bin (11 stars) 0.91 |
| Calibrator 1σ coverage, random half splits (fit on one half, predict the other; σ_inst, s·σ_cat, σ_int, prediction error) | HAT-P-20 (172 epochs) / NGC 7789 N2, APASS, B and V | \|pull\| < 1: 0.684–0.693 (nominal 0.683); with target χ²_ν = 1 instead of the clipped value: 0.64–0.67 |
| Robust scatter / `dmag_err` and / `mag_err`, all constant stars, X < 1.3, six magnitude bins (package 5) | HAT-P-20 C7 night, B and V | 0.95–1.02 / 0.94–1.08 |
| Robust scatter / error of the flux fallback and `dmag` | ATO (Clear, 202 stars; frames reduced before package 3) | 0.88–0.92 |
| Alard–Lupton difference σ in the sky, noisy science frame | `tests/test_alard_lupton.py` | pull σ 0.99 |
| Aperture scatter / propagated σ on registered frames, bin 2 reprojected onto bin 1 (k = 0.25: `SATLEVEL` 91 000 → 22 760, `RDNOISE` 11.66 → 5.83 e⁻) | T CrB 2024-06-08 (bin 1, 18 frames) + 06-09 (bin 2, 50 frames), QHY600M | bin 1: 1.01–1.19, bin 2: 0.93–1.17 |
| Same for the mixed stacks (local sky) | T CrB, B V R I Clear | 0.96–1.11 after sky matching (Clear 2.86 before) |
| Isochrone age 1σ coverage, simulated clusters (distance / reddening profiled, true age on and between grid ages) | `tests/test_isochrone_fit.py` | 0.69–0.72; χ²_ν 0.94 |
| NGC 7789 isochrone fit (PARSEC Z = 0.0152, priors E(B−V) = 0.22 ± 0.03, d = 1.795 ± 0.1 kpc, no membership) | N2, B/V | 2.08 (+0.18/−0.15) Gyr for V 1.0–4.5, 2.44 ± 0.2 Gyr for V 0.5–4.5; χ²_ν 11–14 (field stars, giants in the bright bins); offsets within ~1σ of the priors |
| Gaia membership, simulated cluster with faint members (errors 12× the intrinsic width) | `tests/test_membership_xd.py` | XD: recall 0.91–0.94, contamination 3 %, Σ P_mem within 2 % of the truth; plain GMM: 53 % contamination or recall 0.01–0.03 |
| Gaia membership NGC 7789 (12′, G < 18) | Gaia DR3 | XD vs GMM: 1890 vs 1875 members, +4 % at G 17–18 |
| Fitted catalog-error scale s | same | B 0.29 / 0.87, V 0.54 / 1.0 (σ_int 0.022 in N2 V) |
| Epoch scatter of `mag_cal` for stars with ≥ 20 epochs / (`err_cal_rand` ⊕ photometric ZP error), four error bins | HAT-P-20, B and V | 0.98–1.09; brightest bin (err ≈ 0.012) 1.18: a ~0.007 mag term beyond the model for bright stars (not in the differential light curves, see above) |

Real-data checks are in `scripts/validation/`.
- **`substack_consistency.py`, `substack_pixel_pulls.py`:** pulls between
  two stacks of disjoint frame sets, per star and per pixel.
- **`calibration_checks.py`:** fit statistics per filter, random
  cross-validation of the calibrators, epoch scatter of constant stars
  against `err_cal_rand` ⊕ the photometric zero-point error. Its
  "hold-out catalog stars" are mostly stars the clipping or the quality cuts
  rejected; their pulls (1.5–3) are biased high and are no test of the
  errors.
- **`scatter_over_error.py`:** robust point-to-point scatter of `dmag`,
  `mag` or the fallback `flux` divided by its error, per magnitude bin. Use all constant stars, not the
  differential ensemble: the ensemble is chosen for low scatter and would
  give a ratio below 1 (about 0.85 on HAT-P-20).
