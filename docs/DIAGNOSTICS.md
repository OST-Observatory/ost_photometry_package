# Diagnostic plots

QC figures are written to `<output>/diagnostics/` when the corresponding
`PipelineConfig.diagnostic_plots` flags are on (most are on by default). Toggles
are listed in [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md).

This page explains the **magnitude vs. uncertainty** plots
(`photometry_mag_vs_error_*`).

## Magnitude vs. uncertainty

Two figures:

| File (stem) | When | Content |
|-------------|------|---------|
| `photometry_mag_vs_error_<filter>` | After extraction (reference image) and again per calibration epoch | Density of instrumental mag vs \(\sigma_m\) |
| `photometry_mag_vs_error_overview_<filter>` | After extraction, if more than one image | Same density for **all** images, plus median \(\sigma\) vs time and/or airmass |

\(\sigma_m\) is the **1σ magnitude uncertainty** (always plotted positive, log
y-axis). After a new extraction it is
\((2.5 / \ln 10)\,\sigma_F / |F|\). Older tables that still stored the signed
derivative are shown with \(|\sigma|\).

### How to read the top panel

- **X:** instrumental magnitude (brighter = smaller / more negative, to the left).
- **Y:** \(\sigma_m\) on a **log** scale, so the useful 0.01-mag locus stays visible
  while faint non-detections at \(\sigma \sim 1\) mag do not squash the plot.
- **Colour:** number of stars per bin (log). The dense ridge is the typical
  photometric error at that magnitude.
- **White/black line + band:** binned **median** and 16–84% range. Compare nights
  with this curve, not with individual faint outliers.
- **Dashed red curve:** two-component noise envelope (see below). The data ridge
  should follow it. A high floor at the bright end often means saturation or a
  poor PSF; a ridge well above the curve at all mags often means extra sky/read
  noise or underestimated flux errors.
- **Dotted grey lines:** \(\sigma_m\) for **10σ** (≈ 0.11 mag) and **5σ**
  (≈ 0.22 mag). Stars above the 5σ line are barely detections.

The stats box (N, median \(\sigma\), 90th percentile) summarises the same sample.
Comparison-star counts appear when `is_comparison` is in the table.

### Second panel (single-image figure)

Drawn when comparison flags or a quality column exist (`qfit` / `cfit` /
`sharpness`, otherwise distance to the image edge or offset from the field
centre). Comparison stars are marked with open stars. If those stars sit in the
high-\(\sigma\) tail, they are a poor ensemble for calibration.

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
detector. CMOS (including the QHY cameras used at OST) has per-pixel amplifiers
instead of a CCD serial register, so the **read-noise number** can differ and
may have a slightly non-Gaussian tail, but the same two-component shape is the
right first-order QC model. Use the curve as a **guide to the ridge**, not as a
fitted camera characterisation.

A source-Poisson term \(\propto 10^{0.2 m}\) is **not** fitted separately; it is
only loosely absorbed into \(\sigma_0\) and \(c\).
