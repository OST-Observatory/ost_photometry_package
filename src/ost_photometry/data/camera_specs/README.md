# Camera specification curves

Digitized manufacturer plots (QHY, ZWO, SBIG). These CSVs and the source
JPG/PNG files are **provenance**. The reduction pipeline loads
``ost_photometry/data/cameras.json``, not these files.

Rebuild the catalog after changing a CSV:

```text
python scripts/build_camera_catalog.py
```

## Why JSON at runtime, CSV on disk

The traces come from plot digitizers: two columns, no header, no units.
Filenames encode camera, quantity, and readout mode, and some files are
unusable (swapped axes, all zeros, scatter). That is a reasonable *ingest*
format. It is a poor *runtime* format:

- No units, source, or mode metadata next to the numbers
- Filename parsing is brittle (`fuelwell` vs `fullwell`, combined modes)
- The wheel should not parse 60 headerless files on every `camera_info` call

A single JSON catalog (same pattern as `ost_potsdam_extinction.json`) stores
aliases, chip size, units, readout-mode names, a `usable` flag, and the
`(x, y)` arrays. Linear interpolation is enough; do not go back to inlined
BSpline knots in Python.

SQLite or FITS tables are unnecessary for a few dozen 1-D curves.

## File format (source CSVs)

Each **CSV** has **no header** and two columns. **PNG/JPG** files are the
plots the CSVs were traced from. All CSVs use the same axis convention:

| Quantity | x | y |
|----------|---|---|
| `system_gain` | camera GAIN setting | system gain (e-/ADU) |
| `readout_noise` | camera GAIN setting | read noise (e-) |
| `dark_current` | CCD temperature (°C) | dark current (e-/s) |
| `fullwell` / `fuelwell` | GAIN setting | full well (e-) |
| `dynamic_range` | GAIN setting | dynamic range (stops or dB) |
| `QE` / `response_curve` | wavelength (nm) | QE or relative response |
| `linearity` | signal | measured counts |

A few digitizer exports originally had columns reversed (485C system gain and
QE; 462 vs-GAIN traces). Those CSVs were rewritten in place so they match
this table; the builder does not swap axes.

Readout-mode tokens map to FITS `READOUTM` names:

- `photography` → PhotoGraphic DSO
- `photography_2cms` → Photography Mode 2CMS
- `high_gain` → High Gain Mode
- `extend_fullwell` / `extend_fullwell_2cms` → Extend Fullwell [2CMS]
- `readout_mode_0` / `_1` → QHY5III485C numeric modes
- QHY5III462 has a **single** readout mode (`readout_mode` is null)

`high_gain_and_high_gain_2cms` is one digitized figure: the two high-gain
modes overlap, so the same curve is stored for both.

Notes on individual files:

- **QHY5III485C QE**: wavelength values are ~400–1000 and stored as **nm**
  (4000–10000 Å).
- **ASI2600 `stats_*`**: `red` = RN vs GAIN, `pink` = e-/ADU vs GAIN,
  `blue` = DR vs GAIN. The empty dark CSV/JPG were removed.
- **QHY268 photography fullwell** CSVs are skipped (scattered digitization).

## What the pipeline uses

`camera_info` interpolates **system gain**, **read noise**, and **dark current**
when those curves exist and are marked usable. Chip size comes from the
catalog. Other quantities (QE, full well, DR, linearity) are stored only.

## Missing parameters

Reduction today needs: system gain (e-/ADU), read noise, dark current, chip size.
Optional later: QE, full well, dynamic range, linearity.

| Camera | system gain | read noise | dark | chip mm | QE | full well | DR | linearity |
|--------|-------------|------------|------|---------|----|-----------|----|-----------|
| QHY600M | all 6 modes | all 6 modes | yes | yes | yes | all 6 modes | all 6 modes | yes |
| QHY268M/C | **missing (all modes)** | all 6 modes | yes | yes | yes | missing photography (+ 2CMS); other modes yes | no | no |
| STF-8300 | n/a (use EGAIN) | scalar 9.3 e- | yes | yes | yes | no | no | yes |
| QHY5III485C | mode 0 and 1 | mode 0 and 1 | **missing** | 11.21×6.32 (3864×2180 @ 2.9 µm) | RGB | **axes unclear** | **y looks like full well** | no |
| QHY5III462C | single mode | single mode | **missing** | 5.57×3.13 (1920×1080 @ 2.9 µm) | **missing** | single mode | single mode | no |
| ASI2600 | yes (`stats_pink`) | yes (`stats_red`) | **missing** (file was empty) | IMX571 = 268 | RGB | **missing** | yes (`stats_blue`) | no |

Highest priority for a new digitization: **QHY268 system gain vs GAIN** (every
readout mode), then dark current for 485C / 462 / ASI2600.
