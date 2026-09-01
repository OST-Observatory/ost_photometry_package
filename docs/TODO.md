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
input; it does not write wide tables.

---

## Operational (not code debt)

| Item | Notes |
|------|-------|
| **Site extinction seed** | `data/ost_potsdam_extinction.json` is a literature seed until a campaign updates it via `scripts/aggregate_site_extinction.py`. See [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md). |

---

## Suggested order

1. **P3:** Star-wise k″ fit (optional alternative to mk_calib campaign).
2. **P3:** OST filter throughput → synphot Vega↔AB offsets.
3. **P3:** Drop remaining **read** support for legacy wide tables / column `i` (adapter dual-read), when old `.dat` files no longer matter.
4. **P3:** Mag vs. uncertainty optional ylim / source-Poisson term, only if the log-scale QC is still hard to read.
5. **P2:** Overhaul isochrone handling (refresh grids, fetch+cache, named-column loaders).
6. **P3:** CMD colour window for the isochrone fit (`color_fit_range`), when a mag-only cut is not enough.
7. **P3:** Hess / density underlay on crowded CMDs (default off).
8. **P3:** Parse isochrone headers — only if not already done by the loader overhaul.
9. **P3:** Discrete age×\(Z\) map / MCMC, after the new loader exists.
10. **P3:** Interactive supervisor CMD (optional GUI; batch/PDF stay static).
11. **On utilities changes:** extract only the affected area.
