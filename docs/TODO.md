# TODO

Open, prioritized work on the `ost_photometry` package.

Closed migration notes live in [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md). Also see
[PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), [DIAGNOSTICS.md](DIAGNOSTICS.md), and
[COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** September 2026

**Priorities:** P1 = next sprint · P2 = track, no urgency · P3 = optional / long-term · — = operational, not code debt

---

## Calibration accuracy & photometry systems

### Star-wise fit for second-order extinction k″ (P3)

Today k″ is **not** fitted in the routine pipeline. Options:

| Path | What it does |
|------|----------------|
| `extinction_mode` + `extinction_order` | Apply k′ (and optionally tabulated / user k″) |
| `run_second_order_campaign` (mk_calib) | Field-level `C = T + k'·X`, then `k" = -k'` |

A **star-wise** alternative would fit per star × epoch:

`m_obs − m_std = ZP + k'·X + k''·X·(color)`

That is a **new** method (not the same as `run_second_order_campaign`, which reduces stars to a field `C` first). Needs wide airmass **and** color span, and careful separation from the linear color term `T`.

**Direction (optional):** optional `extinction_mode` / helper that fits star-level residuals; keep mk_calib campaign and tabulated/user k″ as the supported paths for applying SECOND order. Do **not** bolt this onto `fit_extinction_from_comparison_stars` (epoch-mean vs X destroys color information).

### OST filter throughput → Vega↔AB offsets via synphot (P3)

Published Bessell/SDSS per-filter constants (`m_AB = m_Vega + Δ`) are correct for same-bandpass ZP conversion. If OST/instrument **throughput curves** are added later, recompute `Δ` with `synphot`/`speclite` instead of literature defaults.

**Advantages:**

- Offsets match **actual OST bandpasses**, so Vega↔AB on `mag_cal_*` matches how the site observes.
- Can cover **filter variants / chip × filter** combinations when curves differ.
- Same stack can check whether literature Jordi/Lupton colour terms fit OST throughputs (synthetic colours).
- Transparent provenance: regenerate from curves + CALSPEC Vega + AB definition.
- Method unchanged (still per-filter constants); only the numerical `Δ` become instrument-specific.

Optional until curves exist in-repo and a lightweight synphot/speclite dependency is accepted.

---

## Pipeline architecture

### Optional hard prerequisites between steps (P3)

Steps already use soft `skip_*` flags and context markers (`extraction_done`, etc.). There is **no** dependency graph: a later step can run (or fail opaquely) if earlier work was skipped.

**Direction:** optional validation (e.g. correlation requires extraction; calibration requires correlation) that fails fast with a clear message — without forcing a full DAG rewrite.

### Checkpoint / resume after extraction + correlation (P3)

Steps are already modular via config. What is missing is **persist and continue later** (disk checkpoint of `AnalysisContext` / intermediate tables).

**Direction:** only if course/supervisor workflows need “extract+correlate → stop → calibrate later”. Otherwise leave as soft `skip_*` usage.

---

## Reduce

### `reduce/utilities.py` — optional further split (P3)

Facade plus reduction-specific helpers; already delegates to `exposure`, `instrument`, `masks`, `wcs_reduce`, …. Incremental only when touching the affected area.

### WCS-based registration (`shift_method="wcs"`)

Optional alignment backend: solve a celestial WCS per frame if missing
(``wcs_method``, default ASTAP) and **reproject onto the reference image
WCS**. Default remains ``aa_true`` (star-triangle / astroalign) for dense
fields and sub-pixel stacks. Use ``wcs`` for large dithers, sparse fields,
filter-to-filter stacks, or as a fallback when astroalign fails.

Do **not** run ASTAP twice: alignment writes WCS onto the reduced frames;
``find_wcs_of_all_images`` is skipped after ``shift_method="wcs"`` unless
``force_wcs_determination``. ``make_big_images`` is skipped for ``wcs``
(same as ``aa_true``). Keep ``aa_true`` for N2 cluster stacks unless WCS
align is clearly better on that dataset.

---

## Diagnostic plots

### Mag vs. uncertainty — optional polish (P3)

The current figure already uses a log \(\sigma_m\) axis, a binned median,
a photon+sky/read envelope, SNR guides, and JD/airmass panels. See
[DIAGNOSTICS.md](DIAGNOSTICS.md). Left as optional:

| Item | Why it might still help |
|------|-------------------------|
| Hard y-limit | Clip the axis at e.g. the 99.5th percentile or \(\sigma < 1\) mag so a handful of non-detections cannot stretch the log range. |
| Source-Poisson term | Add a \(\propto 10^{0.2 m}\) component to the envelope (currently only floor \(+\) additive \(\propto 10^{0.4 m}\)). |

Do **not** add a third linear y-axis on the same panel or overlay every epoch in one scatter colour (the overview density already pools images).

---

## Light curves

The pipeline already writes one long table `tables/light_curves.ecsv` and
plots **views** of it (JD / folded / check-star QC / calibrator excess-RMS).
See [DIAGNOSTICS.md](DIAGNOSTICS.md#light-curves). Check-star QC ranks the
**most variable** calibrators on purpose (teaching / debug). The items below
are about **determination** (ensemble, errors, epoch quality) and
**evaluation** (period, colour vs phase), not another PDF layout.

Do **not** add more per-star PDFs, interactive HTML/plotly, a prettier
`ImageSeries` flux backend, or a full eclipse MCMC. Those cement the old dual
path or assume the table physics below already exists.

### Two light-curve products, not one API (P1)

Build **two versions**, same photometry table, different physics — not one
function with a `mode=` switch that shares defaults.

| Product | Scale | What it is for |
|---------|--------|----------------|
| **Catalog-transformed** | Calibrated magnitudes (`mag_cal_*`, Vega/AB as labelled) | Absolute level, colour, comparison to catalogs / other nights / other sites. ZP, colour term, and catalog \(\sigma\) stay in the error budget. |
| **Relative / differential** | Flux ratio (OOI vs a comparison; continuum ≈ 1) | Shape and **depth** without a catalog ZP. Store **flux** (or a flux ratio). \(\Delta m = -2.5\log_{10}(F/F_{\mathrm{ref}})\) is only a display transform — do not difference `mag_cal_*`. |

Today the catalog-transformed curve exists (`mag_cal_*` in `light_curves.ecsv`).
Bands with **no catalog standards** (Clear, luminance, white, …) already use
the flux fallback: epoch quasi-ZP (field median) then per-star median so the
continuum sits at 1. That is **not** yet the quiet-ensemble product below.

**Direction:** both live on the long table (e.g. `mag` / `mag_err` for the
transformed curve; `flux` / `flux_err` with `quantity="flux"` or dedicated
`dflux` rows for the relative curve). Plots and C7 scripts say which product
they show. Do not silently subtract the ensemble from `mag_cal_*` and still
call it calibrated.

A magnitude-valued “differential” column is optional later for quoting eclipse
depth in mag; it must be \(-2.5\log_{10}\) of the **same** flux ratio
(instrumental or quasi-ZP), never catalog-calibrated magnitudes.

### Quiet comparison ensemble — differential product (P1)

A C7 light curve whose **relative depth** can be defended against check stars
needs the opposite of today’s QC ranking: the **quietest** \(N\) field stars
(smallest excess RMS, similar brightness to the OOI) as a comparison ensemble.
Their median residual is the common mode (cloud, tracking, extinction).
Divide the OOI by that ensemble in **flux** (or subtract in instrumental /
quasi-ZP magnitudes — the same ratio). Catalog \(\sigma\) must not enter
`flux_err` / `dmag_err`.

Today’s Clear-filter path only removes a **global** epoch median (all stars)
and then sets each star’s own continuum to 1. The quiet-ensemble product
restricts that common mode to the quietest similar-mag stars.

This is not a replacement for the catalog-transformed curve: that one keeps
`mag_cal_*` and catalog systematics on purpose.

**Direction:** select quiet stars from `calibrator_lc_stats.ecsv` (and/or
similar-mag field stars, not only catalog calibrators); write ensemble
residual and OOI/ensemble flux ratio on `light_curves.ecsv` as the
differential product. Keep the “most variable” QC panel as a separate
diagnostic of the transformed set.

### Global bad-epoch flags (P1)

`flag_outlier` is per source. A cloud hits everyone. Use the median residual of
the **quiet** ensemble per epoch; if that common mode is large, the epoch is
bad, not the OOI.

**Direction:** `flag_epoch` (or equivalent) on the long table, plotted
distinctly from per-star outliers. Do not delete rows.

### Inflate photometric uncertainties (P1)

Error bars on the **transformed** curve include photon/sky \(\sigma_m\) and,
honestly, catalog/ZP scatter. On the **differential** curve they must not
include catalog \(\sigma\): only photometry of OOI and ensemble, plus the
ensemble’s excess RMS as systematic floor. If the quiet ensemble has 0.02 mag
excess RMS, that floor applies to `dmag_err`. A 0.03 mag “variation” on
`mag_cal_*` can still be the night **or** the catalog; on `dmag` it should
not be the catalog.

**Direction:** `mag_err_inflated` for the transformed product (optional ZP
floor); `dmag_err = \sqrt{\sigma_{\mathrm{OOI}}^2 + \sigma_{\mathrm{ens}}^2 +
\sigma_{\mathrm{ens,exc}}^2}` for the differential product. Periodograms use
the product they are run on.

### Residuals vs observables (P1)

One diagnostic sheet: OOI residual (after the **differential** ensemble) vs
airmass, FWHM, sky, pixel \((x,y)\). The catalog-transformed curve can share
the same \(x\)-observables; do not mix `mag_cal` residuals into the
differential depth QC. Correlation with airmass → extinction/colour; with
FWHM → seeing/blend; with position → tracking.

**Direction:** add `fwhm`, `sky`, `x`, `y` (and existing `airmass`) to
`light_curves.ecsv` when extraction already has them. Views cannot invent
those columns later.

### Period search from the table (P2)

`period` / \(t_0\) are still typed into the C7 script. Lomb–Scargle (or BLS
for eclipses) on the **differential** curve (smallest extra scatter), with a
false-alarm probability, then fold on the peak. Report the same \(P\) on the
catalog-transformed fold so students see depth on a mag scale vs a cleaner
relative curve. Teaching core: measure \(P\), do not paste it.

**Direction:** optional step/helper reading `light_curves.ecsv`; write
periodogram PDF under `diagnostics/lightcurves/` and annotate the folded
science plot. Default off for non-C7 runs.

### Colour vs phase (P2)

Colour vs JD exists (`light_curve_color`, e.g. \(B-V\)). Colour vs **phase**
distinguishes eclipse (minimum cooler) from pulsation (colour tracks
brightness).

**Direction:** one extra view of the colour rows already in the long table,
folded with the same \(P\) / \(t_0\) as the magnitude curve.

### Simple shape benchmark (P2)

No MCMC. Overlay a constant, optionally a sinusoid or trapezoid minimum, with
\(\chi^2/\nu\) in the annotation. Enough to read “flat vs variable” and
“period fit vs noise”.

**Direction:** optional; science PDF stays the data. Do not couple this to
isochrone MCMC.

### APER vs PSF as two curves (P3)

C7 detection thresholds and blending in tight pairs change amplitude. Same
`id` from APER and PSF tables on one plot: if the amplitude agrees, blend is
unlikely.

**Direction:** only when both extractions exist; do not force dual extraction
in the default C7 script.

### Comparison stars by magnitude and colour (P3)

Calibrators are catalog matches. For the variable, stars of **similar mag and
colour** in the field matter more. Otherwise differential red/blue remains in
the residual.

**Direction:** optional extra cut when building the quiet ensemble (P1), not a
second calibration engine.

### Blend / neighbour in the aperture (P3)

Per OOI epoch: brightest neighbour inside the aperture radius and its flux
fraction. If the neighbour contributes ~20 %, the observed amplitude is a
lower bound.

**Direction:** needs extraction positions + aperture radius already on the
run; store a column on the long table. Skip if crowding products are not
worth the join.

---

## Difference images

HiPS archival subtraction is still **one science image**, optional,
skip-by-default. Fetch hardening is **done**: bandpass-matched survey, cache
under `work/subtract/`, retries with backoff, CDS fallback server, the same
WCS on science CCD and HiPS query, and the step **warns and continues** on
network/subtraction failure (`context.hips_subtract_result`). The subtractor
is **Alard–Lupton in Python** when HOTPANTS is not on `PATH`
(`hips_reference_subtraction_backend="auto"`); HOTPANTS stays optional.

What remains is architecture: HiPS is one **template source**, not the whole
difference-image product. Do **not** start with the legacy trim
`(0, 1599, 0, 2501)`, extra survey strings in config, or HOTPANTS flag
tuning while there is still no night template, no detection, and one
overwritten `diff.fits`.

### Split Template / Subtractor / Detection (P2)

Three layers, like the two light-curve products: do not grow
`HipsReferenceSubtractStep` into a second pipeline.

| Layer | Job |
|-------|-----|
| **Template** | Archive (HiPS / PanSTARRS) **or** a night median / other epoch of the same series |
| **Subtractor** | Alard–Lupton (Python, default) or HOTPANTS; shared WCS, kernel, masks |
| **Detection** | Sources on the difference; match known photometry `id`s vs new vs moving |

Photometry must not block on this. Difference search is its own step (or
night job) after WCS + extraction, with a clear skip when the template is
missing.

### Internal night template (P2)

Asteroids, comets, and transients do **not** need CDS. Template = median of
the other epochs (leave-one-out) or the nearest exposure, regridded to a
common WCS. That scales with RASA: same data, no hips2fits. HiPS stays the
question “was this already on the DSS plate?”.

**Direction:** a template provider next to the HiPS fetch, not a second
copy of subtractor wiring. Reuse `subtract_science_template` on whatever
FITS the provider returns.

### Detection on ±diff and linking (P2)

Starfinder on the positive and negative difference (new/brighter vs
disappeared). Match against photometry `id`s: a large residual on a known
star is a bad kernel, not a transient. Unmatched positive sources linked
across epochs: linear on the sky in minutes–hours → mover; fixed over the
night → transient, variable, cosmic, ghost. Optional SkyBoT/MPC for known
movers.

**Direction:** `tables/diff_candidates.ecsv` (RA/Dec, JD, flux on the diff,
FWHM, dipole flag, `matched_id` or empty, `motion_arcsec_per_h` if linked).
That table is the science product; FITS stay under `work/`. Diagnostics
under `diagnostics/subtract/` with `epoch_id` / `image_id` in the filename,
not one overwritten `diff.fits`. Result list on `context`.

### Subtractor backends (P3)

Alard–Lupton is the in-tree default (`subtract_science_template`,
`backend="auto"`). HOTPANTS remains an optional binary. ZOGY is **not**
next: extra stack, and it duplicates finder/PSF/WCS we already have.
`subtract(science, template) → diff FITS` stays analogous to “one table,
plots as views”; a noise map can wait until detection exists.

### RASA / wide field (P3)

- **Once per pointing, not per frame:** cache one archive template per
  field; all epochs of the night against the **internal** template. HiPS
  only if you care about “not in DSS”.
- **Tiles or downsample:** hips2fits and HOTPANTS on a full RASA frame are
  unreliable. Tile the template, or difference a binned preview plus
  cutouts around candidates.
- **Separate preset from C7 photometry:** RASA monitoring can skip
  calibration (extraction + diff + candidates). C7: HiPS optional on the
  reference image; the time series uses the internal template.

---

## CMD / isochrones

### Overhaul isochrone handling (P2)

The loader is still a 2018-era hand parser: magic column indices, a brittle
``Age``/``age`` scrape, per-library YAML (PARSEC **3.6**, YY V2, BaSTI-IAC),
and files under ``~/isochrone_database/``. Composition metadata is copied into
YAML by hand. That stack is due a **full redesign**, not another YAML tweak.

**Today’s pain**

- Grids are stale (Padova CMD is at **3.9** / PARSEC v2.0 as of late 2025).
- Column maps break when a new download shifts a filter by one index.
- One \(Z\) per YAML; no named columns, no cache, no fetch.
- Student/supervisor scripts only know local paths.

**Direction (do as one project)**

1. **Refresh the on-disk grids** used in the course (PARSEC first; then MIST /
   BaSTI if still wanted). Drop or archive YY if it is no longer maintained.
2. **Fetch + cache** instead of a frozen home-directory dump. PARSEC can be
   queried from the CMD web form ([CMD 3.9](https://stev.oapd.inaf.it/cgi-bin/cmd))
   via [ezpadova](https://github.com/mfouesneau/ezpadova)
   (``get_isochrones``, named columns, age and \([M/H]\) ranges). Evaluate
   whether to vendor a thin wrapper or shell out; cache results under a
   package/data or user cache so the course does **not** hit the network at
   plot time (site downtime, rate limits).
3. **One loader per library** that reads **headers and named columns**, not
   ``isochrone_column: V: 31``. YAML then declares library + photometric system +
   age/\(Z\) grid (or a cache key), not file-offset maps. That subsumes
   “parse isochrone headers” below.
4. Keep a **small cached course set** (e.g. solar ± a metal-poor grid, UBVRI)
   so students plot offline. Supervisors may fetch extra \([M/H]\)/age slices.
5. Do **not** mix this with MCMC or interactive plots; those assume a sane
   table API.

Survey MIST (web/API) and current BaSTI-IAC download paths in the same pass;
only keep libraries we can load without magic indices.

### Colour window for the isochrone fit (P3)

`magnitude_fit_range` already limits fiducials / χ² to a magnitude interval.
There is no matching colour cut, so binaries, red giants, or remaining field
stars redward of the sequence still enter the binned fiducials.

**Direction:** optional `color_fit_range: tuple[float | None, float | None] = (None, None)`,
same shape as `magnitude_fit_range`, applied **together** with the mag window and
only on the fit sample (cluster members when `is_cluster_member` exists). The
CMD still plots stars outside the window. Default off. Student script stays
unset; supervisor script can set a window when a sequence is obvious.

### Hess / density underlay for dense clusters (P3)

Crowded CMDs (rich clusters) overplot: scatter + error bars hide the sequence.

**Direction:** optional 2D histogram / hexbin (Hess diagram) **under** the member
scatter; field stars stay grey points. Default off so student plots stay a
simple CMD. Do **not** replace the scatter — individual stars and σ-colouring
should remain on top.

### MCMC in age–\(Z\) (P3)

The fit is 1-D: best age at the **fixed** \(Z\) of the YAML grid. Age,
metallicity, and reddening shift the sequence in similar ways, so a single
\(\chi^2\) minimum looks more precise than it is.

**What it would add:** a posterior (or even a discrete age\(\times Z\)
\(\chi^2\) map) so the age–\(Z\) degeneracy and uncertainties are visible,
instead of “best isochrone: 1.2 Gyr”. Needs a 2-D grid, interpolation, and
priors — i.e. the overhauled loader above, not the current one-file YAML.

**Not next:** residuals and membership already say whether \(\chi^2\) hits the
main sequence. A cheap precursor is a discrete age\(\times Z\) map before a
full MCMC (emcee/dynesty).

### Interactive CMD plots (P3)

Plots are Agg → PDF/PNG. Supervisors cannot pan/zoom, hover ``id``, click
outliers out of the fit, or slide \(E(B-V)\).

**What it would add:** a faster iterate-on-reddening / fit-window loop. It does
not change the published figure or the age number. Keep student and pollux
batch output static; any GUI (matplotlib widget / plotly) stays supervisor-only
and optional.

### Parse isochrone file headers (P3)

PARSEC/YY/BaSTI headers already carry \(Z\), \(Y\), \([\mathrm{Fe}/\mathrm{H}]\),
photometric system, and column names. We scrape age from comment lines and copy
the rest into YAML.

**What it would add:** info-box metadata from the **file**, and a mismatch
warning (YAML “solar” vs file \([\mathrm{Fe}/\mathrm{H}]=-1.5\)). Prefer to
do this **inside the loader overhaul** (named columns), not as a one-off regex
per library on the current parser. Does not change the fit.

---

## Remaining analyze bridge

### Drop legacy wide magnitude tables / column `i` — write path done

Pipeline and post-process **write** only epoch-native ECSV (``id``).
``write_legacy_wide_magnitudes_dat``, ``calibrated_epochs_to_legacy_wide_table``,
``mk_magnitudes_table``, ``save_calibration``, and ``save_magnitudes_ascii``
are removed.

**Still kept (read old files):** ``legacy_wide_table_to_epoch_native`` /
``ensure_epoch_native_photometry_table`` accept wide tables with column ``i``.
N2 ``2b_post_process.py`` prefers ECSV and converts a leftover ``.dat`` on
input; it does not write wide tables. Keep this dual-read until old course
``.dat`` files no longer matter, then drop it (P3 below).

---

## Code health / structure

Structural revision (not new science features). Prefer touching these when
already working in the area; do not block P1 light curves on a pure cleanup
sprint.

### `plots/cmds.py` monolith (P2)

``MakeCMDs`` / ``plot_absolute_cmd`` (~1700-line module) still mixes plotting,
isochrone I/O, χ² scoring, and diagnostics. Couples tightly to the brittle
isochrone YAML loader.

**Direction:** split plot helpers, isochrone scoring, and diagnostics writers
alongside the **isochrone handling overhaul** above — not a drive-by rename.
Keep the public ``plot_cmds_from_table`` / ``MakeCMDs`` entry points stable for
course scripts.

### `extraction.py` size (P2)

``analyze/extraction.py`` (~1700+ lines) is the photometry kernel (APER/PSF,
masks, ePSF, growth). Hard to test and change in isolation.

**Direction:** extract cohesive pieces (e.g. aperture vs PSF path, ePSF
selection, diagnostic hooks) when changing that area. Do not rewrite the
detection API for N2/C7 scripts.

### `subtraction_alard_lupton.py` vs difference-image layers (P2)

The Python Alard–Lupton subtractor is large and will grow with night templates
and detection. Keep subtractor code separate from HiPS fetch and from
``diff_candidates`` linking — see **Difference images** above.

### Drop ``differential_photometry`` shim (P3)

Live code is ``calibration/photometer.py`` (``DifferentialPhotometer``) and
``calibration/calibrator.py`` (``PhotometryCalibrator``).
``analyze/differential_photometry.py`` is only a ``DeprecationWarning`` re-export.

**Direction:** remove the shim once external scripts import from
``ost_photometry.analyze.calibration``.

### Small compatibility leftovers (P3)

- Re-export stubs such as ``analyze/utils/simbad_annotate.py`` → real module in
  ``post_processing``.
- ``populate_legacy_calibration_epoch_meta`` / legacy column helpers used only
  for old wide tables — remove with the legacy-wide **read** drop.
- Deprecated preset aliases (``n2_stack``, ``c7_variable``, …) are **already
  removed**; use canonical preset names only.

### `old_legacy_function/` — removed

Archived Image-based ZP/plot helpers outside ``src/`` are gone. Do not restore.

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` is a literature seed until a campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Suggested order

1. **P1:** Light curves — two products (catalog-transformed mag scale vs differential depth without catalog \(\sigma\)); quiet ensemble + `flag_epoch`; then inflate \(\sigma\) and residuals vs airmass/FWHM/sky/\(x,y\).
2. **P2:** Difference images — internal night template + detection/linking + `diff_candidates.ecsv` (HiPS fetch hardening is done; do not start with legacy trim or extra survey strings).
3. **P2:** Light curves — period search (Lomb–Scargle / BLS) from `light_curves.ecsv`; colour vs phase; simple \(\chi^2\) shape overlay.
4. **P2:** Overhaul isochrone handling (refresh grids, fetch+cache, named-column loaders) **and** split `plots/cmds.py` in the same pass.
5. **P2:** Split `extraction.py` only when touching that area; keep script-facing APIs stable.
6. **P3:** Light curves — APER vs PSF amplitude, mag/colour-matched ensemble, aperture blend fraction.
7. **P3:** Difference images — ZOGY backend; RASA field-wise HiPS cache / tiles; separate monitoring preset.
8. **P3:** Star-wise k″ fit (optional alternative to mk_calib campaign).
9. **P3:** OST filter throughput → synphot Vega↔AB offsets.
10. **P3:** Drop remaining **read** support for legacy wide tables / column `i` (adapter dual-read), when old `.dat` files no longer matter; drop related legacy meta helpers.
11. **P3:** Remove `differential_photometry` deprecation shim once imports use `analyze.calibration`.
12. **P3:** Mag vs. uncertainty optional ylim / source-Poisson term, only if the log-scale QC is still hard to read.
13. **P3:** CMD colour window for the isochrone fit (`color_fit_range`), when a mag-only cut is not enough.
14. **P3:** Hess / density underlay on crowded CMDs (default off).
15. **P3:** Parse isochrone headers — only if not already done by the loader overhaul.
16. **P3:** Discrete age×\(Z\) map / MCMC, after the new loader exists.
17. **P3:** Interactive supervisor CMD (optional GUI; batch/PDF stay static).
18. **On utilities changes:** extract only the affected area of `reduce/utilities.py`.
