# Script compatibility matrix

**Stand:** Juli 2026 — epoch-native pipeline, calibration convergence release, site extinction table.

For pipeline options see [PIPELINE_CONFIG.md](PIPELINE_CONFIG.md). For breaking API
changes see [ARCHITECTURE_AND_MIGRATION.md](ARCHITECTURE_AND_MIGRATION.md).

---

## 1. reduction_scripts_students

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `c7/1_reduce_images.py` | ✅ | `reduce.redu.reduce_main`; optional `validate_inputs`, `fail_on_missing_flat` (defaults on) |
| `c7/2_obtain_flux.py` | ✅ | `run_pipeline`, `PipelineConfig` (`preset`/`custom`), `extraction_mode="multi"` |
| `c7/3_plot_lightcurve.py` | ✅ | `analyze.plots`, epoch-native ECSV input |
| `n1_baches/1_masterimages.py` | ✅ | `reduce.utilities`, `reduce.registration`, `checks` |
| `n2/1_add_images.py` | ✅ | `reduce.redu.reduce_main` (stacking via MP per filter) |
| `n2/3_plot_cmd.py` | ✅ | `load_cmd_table`, `plot_cmds_from_table`, `style.Bcolors` |

**Hinweis N2:** In `reduction_scripts_students/n2/` gibt es **kein** `2_obtain_flux.py`.
Die Photometrie-Extraktion für N2 läuft über die Supervisor-Skripte (Abschnitt 2).

Legacy `Observation.extract_flux` / `extract_flux_multi` entfallen zugunsten von `run_pipeline`.

---

## 2. reduction_scripts_supervisors

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `n2/2_extract_data_supervisors.py` | ✅ | `run_pipeline`, `PipelineConfig`, Preset `median_zp_per_image` / custom |
| `n2/2_extract_data_students.py` | ✅ | Wie supervisors; student-facing variant |
| `n2/3_plot_cmd_supervisors.py` | ✅ | wie `n2/3_plot_cmd` (plus Fit/Cali/E(B-V)-Fehler) |
| `n2/2b_post_process.py` | ✅ | `post_processing`, `analyze.utilities`, `checks` |

---

## 3. auxiliary_scripts

### 3.1 `img_shift_cut_etc/`

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `remove_overscan.py`, `cut_imgs.py`, `bin_imgs.py` | ✅ | `ost_photometry.checks` |
| `flip_imgs_fits.py` | ✅ | `reduce.utilities.flip_image` |
| `flip_imgs.py` | ✅ | `mk_file_list(..., add_path_to_file_names=True)` |
| `imgs_shifts.py` | ✅ | `mk_file_list(..., add_path_to_file_names=True)` |

### 3.2 `mk_calib_photometry/`

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `1_reduce_images.py` | ✅ | `reduce.redu`, `reduce.utilities`, `reduce_main(...)` keywords |
| `2_mk_trans.py`, `2_mk_trans_add.py` | ✅ | `run_pipeline` (`extract_protect_calibrators`), `CalibrationEngine` via `mk_calib_pipeline.write_field_transformation_table` → `trans_para_*.dat` + `.json` |
| `3_second_order_extinction.py`, `3_second_order_extinction_add.py` | ✅ | `run_second_order_campaign` (reads `.dat` or `.json` field tables) |
| `new_pipeline/determine_extinction_coefficients.py` | ✅ | `extinction_mode="from_value_airmass"`, `protect_calibration_objects=True`, `skip_calibration=True`; siehe [EXTINCTION_COEFFICIENTS.md](EXTINCTION_COEFFICIENTS.md) |

### 3.3 Sonstiges

Skripte unter `align_sun_imgs_mk_video/` nutzen ein lokales `aux`-Modul und sind
nicht Teil dieser Matrix.

---

## 4. Style

Es gilt **`ost_photometry.style.Bcolors`** (großes „B“). Kein `style.bcolors` verwenden.

---

## 5. Checklist bei API-Änderungen

- `Observation.run_pipeline` + `PipelineConfig` (inkl. Presets)
- `calibration_strategy`, `calibration_grouping`, `extinction_mode`, `color_term_fit`
- `path_extinction_coefficients` für `extinction_mode="tabulated"`
- `mk_file_list(..., add_path_to_file_names=...)`
- `reduce_main`-Keyword-Argumente (`validate_inputs`, `fail_on_missing_flat`, `sanity_check_sample_size`)
- Öffentliche Reduktions-API: `reduce.redu.reduce_main` (Implementierung in `ost_photometry.reduce.workflow`, intern modularisiert)
- Epoch-native ECSV (legacy wide `.dat` is read-only via `legacy_wide_table_to_epoch_native`)
