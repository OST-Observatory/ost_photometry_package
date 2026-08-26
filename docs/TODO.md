# TODO

Open, prioritized work on the `ost_photometry` package.

Closed migration notes live in [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md). Also see
[PIPELINE_CONFIG.md](PIPELINE_CONFIG.md), [DIAGNOSTICS.md](DIAGNOSTICS.md), and
[COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md).

**As of:** August 2026

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
5. **On utilities changes:** extract only the affected area.
