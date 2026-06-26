# Kompatibilitätsprüfung: Skripte vs. aktuelle ost_photometry-Pipeline

**Stand:** März 2026 (nach API-Bereinigung: `run_pipeline`, `mk_file_list(..., add_path_to_file_names=...)`, `reduce_main`-Keyword-Argumente, `analyze.utilities` für Hilfsfunktionen, `Bcolors` im Style-Modul).

---

## 1. reduction_scripts_students

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `c7/1_reduce_images.py` | ✅ | `reduce.redu`, `reduce.utilities`, `style` |
| `c7/2_obtain_flux.py` | ✅ | `Observation.run_pipeline`, `PipelineConfig`, `extraction_mode="multi"` |
| `c7/3_plot_lightcurve.py` | ✅ | `analyze.plots` |
| `n1_baches/1_masterimages.py` | ✅ | `reduce.utilities`, `reduce.registration`, `checks`, `terminal_output` |
| `n2/1_add_images.py` | ✅ | `reduce.redu`, `reduce.utilities` |
| `n2/2_obtain_flux.py` | ✅ | Wie c7/2, N2-Docstring; `run_pipeline` + Multi-Filter-Pfade |
| `n2/3_plot_cmd.py` | ✅ | `analyze.plots`, `analyze.utilities`, `style.Bcolors`, `checks`, `ost_photometry.utilities` |

**Hinweis:** Frühere `Observation.extract_flux` / `extract_flux_multi` entfallen zugunsten von `run_pipeline`.

---

## 2. reduction_scripts_supervisors

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `n2/2_extract_data_supervisors.py` | ✅ | `analyze.analyze`, `style` |
| `n2/2_extract_data_students.py` | ✅ | u. a. `main_extract`, `find_wcs_astrometry` über `ost_photometry.utilities` |
| `n2/3_plot_cmd_supervisors.py` | ✅ | wie `n2/3_plot_cmd` (students) |
| `n2/2b_post_process.py` | ✅ | `analyze`, `analyze.utilities`, `checks`, `style` |

---

## 3. auxiliary_scripts

### 3.1 `img_shift_cut_etc/`

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `remove_overscan.py`, `cut_imgs.py`, `bin_imgs.py` | ✅ | `ost_photometry.checks` |
| `flip_imgs_fits.py` | ✅ | `reduce.utilities.flip_image` |
| `flip_imgs.py` | ✅ | `utilities.mk_file_list(..., add_path_to_file_names=True)` |
| `imgs_shifts.py` | ✅ | `mk_file_list(..., add_path_to_file_names=True)` |

### 3.2 `mk_calib_photometry/`

| Skript | Status | Kurznotiz |
|--------|--------|-----------|
| `1_reduce_images.py` | ✅ | `from ost_photometry.reduce import redu, utilities`; `reduce_main(..., read_noise=..., dark_rate=..., rm_cosmic_rays=..., saturation_level=..., limiting_contrast_rm_cosmic_rays=..., sigma_clipping_value_rm_cosmic_rays=..., debug=...)` |
| `2_mk_trans.py`, `2_mk_trans_add.py` | ✅ | `Observation`, `ImageSeries`, `utilities.find_wcs`, `correlate`, `calibration` |
| `3_second_order_extinction.py`, `3_second_order_extinction_add.py` | ✅ | `from ost_photometry.analyze.utilities import lin_func, fit_curve`; `from ost_photometry.style import Bcolors as bcolors` |
| `new_pipeline/determine_extinction_coefficients.py` | ✅ | `analyze`, `PipelineConfig`, `style` |

### 3.3 Sonstiges

Skripte unter `align_sun_imgs_mk_video/` nutzen ein **lokales** `aux`-Modul (`mkfilelist`, `addpath`) und sind **nicht** Teil dieser ost_photometry-Kompatibilitätsmatrix.

---

## 4. Style: `Bcolors`

Es gilt **`ost_photometry.style.Bcolors`** (großes „B“). Kein `style.bcolors` / `import bcolors` verwenden.

---

## 5. Kurzfassung

Alle in den Abschnitten 1–3 genannten Skripte sind mit der beschriebenen Pipeline-API abgeglichen (**✅**). Bei künftigen Umstellungen vor allem prüfen: `run_pipeline`, `PipelineConfig`, `mk_file_list`-Parameter und `reduce_main`-Keywords.
