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
| Robust scatter / error of all constant stars, X < 1.3, six magnitude bins (package 2: median sky, empty-sky factor, scintillation) | HAT-P-20 C7 night, 355 (V) / 320 (B) stars | 0.97–1.01; brightest V bin (11 stars) 0.91 |

Real-data checks are in `scripts/validation/`.
- **`substack_consistency.py`, `substack_pixel_pulls.py`:** pulls between
  two stacks of disjoint frame sets, per star and per pixel.
- **`scatter_over_error.py`:** robust point-to-point scatter of `dmag`
  divided by `dmag_err`, per magnitude bin. Use all constant stars, not the
  differential ensemble: the ensemble is chosen for low scatter and would
  give a ratio below 1 (about 0.85 on HAT-P-20).
