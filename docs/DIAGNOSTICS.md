# Diagnostic plots

QC figures are written to `<output>/diagnostics/` when the corresponding
`PipelineConfig.diagnostic_plots` flags are on (most are on by default). Toggles
are listed in [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md).

This page explains the **magnitude vs. uncertainty** plots
(`photometry_mag_vs_error_*`) and the **inter-filter residual geometry**
figures that sit next to the correlation separation histograms.

## Magnitude vs. uncertainty

Two kinds of figure, on purpose:

| File (stem) | When | Content |
|-------------|------|---------|
| `photometry_mag_vs_error_<filter>` | After **extraction** (reference image) | Density of instrumental mag vs \(\sigma_m\) for **all** detections. Quality panel (`qfit` / `sharpness` / …) when those columns exist. **No** comparison-star overlay — the later fit still rejects stars. |
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
fit cuts for that band).

### How to read the top panel

- **X:** instrumental magnitude (brighter = smaller / more negative, to the left).
- **Y:** \(\sigma_m\) on a **log** scale, so the useful 0.01-mag locus stays visible
  while faint non-detections at \(\sigma \sim 1\) mag do not squash the plot.
- **Colour:** number of stars per bin (log). The dense ridge is the typical
  photometric error at that magnitude.
- **Magenta line + white band:** binned **median** and 16–84% range (high contrast
  on the viridis density). Compare nights with this curve, not with individual
  faint outliers.
- **Dashed red curve:** two-component noise envelope (see below). The data ridge
  should follow it. A high floor at the bright end often means saturation or a
  poor PSF; a ridge well above the curve at all mags often means extra sky/read
  noise or underestimated flux errors.
- **Dotted grey lines:** \(\sigma_m\) for **10σ** (≈ 0.11 mag) and **5σ**
  (≈ 0.22 mag). Stars above the 5σ line are barely detections.

The stats box (N, median \(\sigma\), 90th percentile) summarises the same sample.
After calibration it also counts stars used in the fit and catalog matches that
were not used.

### Second panel (single-image figure)

Drawn when comparison/calibrator flags or a quality column exist (`qfit` /
`cfit` / `sharpness` / `roundness*`, otherwise distance to the image edge or
offset from the field centre). After calibration, stars used in the fit are
open stars; catalog matches rejected by the quality cuts are grey crosses. If
the used stars sit in the high-\(\sigma\) tail, they are a poor ensemble.

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
