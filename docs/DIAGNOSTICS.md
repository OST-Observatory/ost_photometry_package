# Diagnostic plots

QC figures live under `<output>/diagnostics/<step>/` when the corresponding
`PipelineConfig.diagnostic_plots` flags are on (most are on by default). Toggles
are listed in [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md).

```
<output>/
  diagnostics/
    extraction/      # mag–error, growth, starmaps, aperture, ePSF, residual
    correlation/     # inter-filter separations/geometry, exposure pairing
    calibration/     # catalog match, fit panels, residuals, night summary
    extinction/      # k′ vs airmass
    cluster/         # PM vs distance, membership scatter
  results/
    lightcurves/     # by_id/, calibration/
    cmds/
    starmaps/        # Simbad-annotated maps
  tables/
  work/
    wcs_images/
    extraction/      # all-image galleries when plots_for_all_images
    subtract/
```

Former top-level plot folders (`calibration/`, `lightcurve/`, `extinction_fit/`,
`compare/`, `scatter/`, `wcs_images/`, `subtract/`) are no longer written.

This page explains the **magnitude vs. uncertainty** plots
(`photometry_mag_vs_error_*`), the **inter-filter residual geometry**
figures that sit next to the correlation separation histograms, the
**catalog cross-match** geometry/diagnostics next to the calibration
separation histogram, and the **catalog vs. extraction** checks that use
the same residual as the calibration fit.

## Magnitude vs. uncertainty

Two kinds of figure, on purpose:

| File (stem) | When | Content |
|-------------|------|---------|
| `photometry_mag_vs_error_<filter>` | After **extraction** (reference image) | Density of instrumental mag vs \(\sigma_m\) for **all** detections. One extra panel per quality column (`qfit`, `cfit`, `sharpness`, `roundness*`, finder FWHM) when those columns exist. **No** comparison-star overlay — the later fit still rejects stars. |
| `photometry_mag_vs_error_overview_<filter>` | After extraction, if more than one image | Same density for **all** images, plus median \(\sigma\) vs time and/or airmass |
| `photometry_mag_vs_error_<filter>_<epoch>` | After **calibration** | Same density for that epoch, with stars **used in the fit** (open stars) and catalog matches that were clipped or otherwise unused (grey crosses) |

\(\sigma_m\) is the **1σ magnitude uncertainty** (always plotted positive, log
y-axis). After a new extraction it is
\((2.5 / \ln 10)\,\sigma_F / |F|\). Older tables that still stored the signed
derivative are shown with \(|\sigma|\).

Finder quality (`sharpness`, `roundness*`) and PSF-fit quality (`qfit`, `cfit`,
`flags`) are copied onto the photometry table at extraction so both figures can
use them. Comparison / calibrator flags are written only after the calibration
step (`is_comparison` = catalog match; `is_calibrator_<filter>` = survived the
pre-fit quality cuts and the residual clip for that band). The quality cuts
are: σ above the binned p84 ridge of all detections (and optionally the
photon+sky envelope), `qfit`/`cfit` caps, and the finder sharpness/roundness
windows. See [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md) (`calibrator_*`). APER
tables often have no `qfit`; that cut is then skipped.

### How to read the top panel

- **X:** instrumental magnitude (brighter = smaller / more negative, to the left).
- **Y:** \(\sigma_m\) on a **log** scale, so the useful 0.01-mag locus stays visible
  while faint non-detections at \(\sigma \sim 1\) mag do not squash the plot.
- **Colour:** number of stars per bin (log). The dense ridge is the typical
  photometric error at that magnitude.
- **Cyan line + white band:** binned **median** and 16–84% range (high contrast
  on the viridis density; the legend uses a black-edged patch so the band stays
  visible). Compare nights with this curve, not with individual faint outliers.
- **Dashed orange curve:** two-component noise envelope (see below). The data
  ridge should follow it. A high floor at the bright end often means saturation
  or a poor PSF; a ridge well above the curve at all mags often means extra
  sky/read noise or underestimated flux errors.
- **Navy dash-dot / dark dotted lines:** \(\sigma_m\) for **10σ** (≈ 0.11 mag)
  and **5σ** (≈ 0.22 mag). Stars above the 5σ line are barely detections.

The stats box (N, median \(\sigma\), 90th percentile) summarises the same sample.
After calibration it also counts stars used in the fit and catalog matches that
were not used.

### Extra panels (single-image figure)

One panel per available quality column (`qfit`, `cfit`, `sharpness`,
`roundness` / `roundness1` / `roundness2`, finder `fwhm`). If none of those
exist, the fallback is distance to the image edge or offset from the field
centre. After calibration, stars used in the fit are open stars; catalog
matches rejected by the quality cuts are grey crosses (overplotted on each
quality panel). Used stars should sit on the \(\sigma\) ridge and inside the
qfit/shape windows. Remaining outliers in the used sample mean a cut was
skipped (`calibrator_min_keep`) or the column was absent.

### Overview extra panels

Median \(\sigma_m\) of the **brightest 40%** of stars, vs **JD** and/or
**airmass** (image index if neither is available). A jump at high airmass or
late in the night is clouds, tracking, or extinction, not a single faint star.

## The dashed “photon + sky/read” curve

The overlay is

\[
\sigma(m) = \sqrt{\sigma_0^2 + \bigl(c \cdot 10^{0.4\,m}\bigr)^2}.
\]

\(\sigma_0\) is a **constant floor** (bright-end systematics). The second term
is **additive noise in flux** (sky in the aperture, read noise, dark) converted
to magnitudes: \(\sigma_m \propto \sigma_F / F\) and \(F \propto 10^{-0.4 m}\),
so a flux-independent \(\sigma_F\) becomes \(\propto 10^{0.4 m}\).

**This is not CCD-specific.** Photon arrival is Poisson for any silicon
detector. CMOS has per-pixel amplifiers instead of a CCD serial register, so the 
**read-noise number** can differ and may have a slightly non-Gaussian tail, but 
the same two-component shape is the right first-order QC model. Use the curve 
as a **guide to the ridge**, not as a fitted camera characterisation.

A source-Poisson term \(\propto 10^{0.2 m}\) is **not** fitted separately; it is
only loosely absorbed into \(\sigma_0\) and \(c\).

## Inter-filter residual geometry

When a separation histogram has a **tail to large offsets**, these figures show
*where* on the reference image those offsets sit. They are written together
with the existing inter-filter separation plots
(`correlation_inter_filter_separation_plot`; same
`correlation_inter_filter_max_pair_plots` cap).

| File (stem) | When | Content |
|-------------|------|---------|
| `inter_filter_correlation_geometry_reference_<other>` | After inter-filter correlation | 2×2: quiver on the **reference image**, \|offset\| vs radius, radial vs \(r\), tangential vs \(r\) |
| `inter_filter_correlation_geometry_pair_NNN_*_<other>` | Per paired exposure (capped) | Same 2×2 for that pair |
| `inter_filter_correlation_geometry_overview` | If more than one pair | RMS radial vs RMS tangential (1:1 line) and RMS vs pair |

Other-filter sky positions are reprojected onto the **reference WCS**. Arrows
are \((x_\mathrm{other\ on\ ref} - x_\mathrm{ref},\, y_\mathrm{other\ on\ ref} - y_\mathrm{ref})\)
in reference pixels (magnified so they are visible). Colour is on-sky
separation in arcsec.

A median \((\mathrm{d}x, \mathrm{d}y)\) is subtracted **only** for the radial /
tangential panels, so a bulk pointing/WCS offset does not leak into those
components. The quiver still shows the raw vectors.

### How to tell the patterns apart

- **Bulk shift** (wrong CRVAL / a constant pointing offset): all arrows nearly
  parallel and similar length. Histogram moves as a whole; no radius tail.
  Stats box: large median \((\mathrm{d}x, \mathrm{d}y)\); after removal, both
  RMS values are small.
- **Rotation** (images rotated relative to each other, or a CROTA/CD mismatch):
  arrows swirl around the field centre. \|offset\| and the **tangential**
  residual grow linearly with radius (dashed median \(d/r\) line). Points
  **above** the 1:1 line on the overview. Corner stars make the separation
  **tail**.
- **Plate-scale / magnification**: arrows point radially in or out.
  **Radial** residual grows linearly with radius. Points **below** the 1:1 line.
- **Field distortion** (pincushion/barrel, unmodelled SIP, differential
  refraction): radial (or mixed) residuals that **curve** or fan away from the
  dashed line, usually worse in the corners. Quiver is not a clean swirl or
  a clean radial field.
- **Wrong matches**: large arrows scattered with no spatial pattern; both RMS
  values high and the histogram tail is not concentrated at large \(r\).

The implied rotation (arcmin) and scale (%) in the stats box are the median
\(d/r\) converted to those units — a first-order number, not a fitted WCS.

## Catalog cross-match separations

The histogram (`differential_catalog_crossmatch_separations`) of
`match_sep_arcsec` often shows a **sharp core** (true matches, limited by
astrometry and centroids) plus a **tail** out to the search radius
(`calibration_match_radius`, default 2″). That tail is expected in a crowded
field such as NGC 7789; these extra figures say *which* of the usual causes it
is.

| File (stem) | Content |
|-------------|---------|
| `calibration_crossmatch_diagnostics` | Log-y histogram; \|offset\| vs magnitude; \|offset\| vs radius; nearest vs second-nearest catalog star |
| `calibration_crossmatch_geometry` | Same 2×2 quiver / radial / tangential layout as the inter-filter geometry (catalog projected onto the image WCS) |

Written with the histogram when `calibration_crossmatch_separation_histogram` is on.

### How to read them

- **Chance coincidences / crowding** (typical in an open cluster): tail is
  **not** a clean function of radius; many points sit near the **1:1 line** in
  nearest vs second-nearest (`match_sep2_arcsec` ≈ `match_sep_arcsec`); faint
  stars dominate the tail. The quiver looks **random**. Tighten
  `calibration_match_radius`, drop ambiguous pairs (`sep2` close to `sep`), or
  restrict to brighter catalog stars.
- **WCS rotation / scale / distortion**: \|offset\| **grows with radius**;
  quiver is a swirl (rotation) or radial field (plate scale). Same language as
  the inter-filter geometry section above. Re-solve WCS (SIP) or check the
  plate scale.
- **Bulk CRVAL offset**: histogram peak **not** at zero; arrows nearly parallel
  and similar length. Median \((\mathrm{d}x, \mathrm{d}y)\) in the stats box is
  large; after removing it, RMS radial/tangential are small.
- **Saturated / blended centroids**: tail at the **bright** end of \|offset\| vs
  mag, often in the cluster core. Those stars should not enter the ZP fit
  (they usually fail `is_calibrator` already).
- **Proper motion / epoch mismatch**: modest extra scatter with no spatial
  pattern, more for nearby high-\(\mu\) stars. Gaia with epoch correction is
  the next catalog step if the core itself is wider than the expected
  astrometric floor.

Orange stars on the diagnostic scatter are stars **used in the calibration
fit**; grey points are catalog matches that were rejected by the pre-fit
quality cuts or the residual clip. A tail that is almost only grey is already
handled. A tail that still contains orange stars is contaminating the ZP
(usually because a cut was skipped to keep `calibrator_min_keep` stars).

## Katalog vs. Extraktion

The four catalog-check figures use **one plot API** and the **same residual as
the fit** that produced the calibrated magnitudes. There is no second,
median-only ZP hidden in the diagnostics.

\[
r = m_\mathrm{cat} - m_\mathrm{inst} - T\cdot c - \mathrm{ZP}.
\]

| Fit | \(T\) | \(\mathrm{ZP}\) | What a slope of \(r\) vs color means |
|-----|-------|-----------------|--------------------------------------|
| Median-ZP (teaching script and pipeline `median_zp`) | \(0\) | median of \(m_\mathrm{cat}-m_\mathrm{inst}\) | the **missing color term** |
| Linear fit (pipeline `linear_fit`) | `color_term` from `TransformationCoefficients` | `zero_point` from the same object | a **leftover** trend after \(T\cdot c+\mathrm{ZP}\) |

Color \(c\) is the catalog index from `color_index_filters` (not hardcoded
\(B-V\)). Stars used in the fit (`is_calibrator_<filter>`) are overplotted on
catalog matches that were clipped. The student extract script has no clip, so
it passes no mask.

| File (stem) | Content |
|-------------|---------|
| `instrumental_vs_catalog_<filter>_<epoch>` | Observed vs catalog mag, **always** with \(r\) vs mag underneath. A 1:1 line only when x is already on the catalog scale (`show_one_to_one`, student script). |
| `zeropoint_residual_distribution_<epoch>` | Histogram of **fit** residuals (optional Gaussian with \(\sigma=\mathrm{RMS}\)). Overlay of several filters: the stats box lists **N / median / RMS per filter**. |
| `zeropoint_residual_vs_color_<f1>_<f2>_<epoch>` | Same \(r\) vs catalog color, with per-filter median, RMS, and slope. Title “Residual color term with pure ZP?” vs “Residuals after \(T\cdot c+\mathrm{ZP}\)”. |
| `calibration_color_color_cal_stars_<epoch>` | Catalog color vs **calibrated** observed color (after \(\Delta\mathrm{ZP}\), or after the full transformation). Guide line is slope 1 through the median offset, not a naive 1:1 on instrumental \((B-V)\). |

The transformation panels under `<output>/diagnostics/calibration/` are the fit view
with the line \(T\cdot c+\mathrm{ZP}\). These diagnostic figures must agree
with that residual, not invent a second one.
