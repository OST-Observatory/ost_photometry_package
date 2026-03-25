# Kompatibilitätsprüfung: Skripte vs. aktuelle ost_photometry-Pipeline

Stand der Prüfung: nach Restrukturierung (Phase 1–3)

---

## 1. reduction_scripts_students

### c7/1_reduce_images.py ✅
- **Imports:** `ost_photometry.reduce.redu`, `ost_photometry.reduce.utilities`, `ost_photometry.style`
- **Status:** Kompatibel – alle Imports und Aufrufe existieren weiterhin.

### c7/2_obtain_flux.py ✅
- **Imports:** `ost_photometry.analyze.analyze`, `ost_photometry.style`
- **Verwendung:** `analyze.Observation`, `observation.extract_flux_multi`
- **Status:** Kompatibel.

### c7/3_plot_lightcurve.py ✅
- **Imports:** `ost_photometry.analyze.plots`
- **Status:** Kompatibel – `plots` ist ein Submodul von `analyze`.

### n1_baches/1_masterimages.py ⚠️
- **Imports:** `ost_photometry.reduce.utilities`, `ost_photometry.reduce.registration`
- **Verwendung:** `import ost_photometry.reduce.utilities as utilities`
- **Status:** Kompatibel – utilities re-exportiert die aufgeteilten Funktionen.

### n2/1_add_images.py ✅
- **Imports:** `ost_photometry.reduce.redu`, `ost_photometry.reduce.utilities`
- **Status:** Kompatibel.

### n2/2_obtain_flux.py
- (Nicht in den Suchergebnissen – vermutlich wie c7/2.)

### n2/3_plot_cmd.py ✅
- **Imports:** `ost_photometry.analyze.plots`, `ost_photometry.analyze.utilities`, `ost_photometry.style.Bcolors`, `ost_photometry.checks`, `ost_photometry.utilities as base_utilities`
- **Hinweis:** `ost_photometry.style` enthält `Bcolors` (groß geschrieben), nicht `bcolors`.
- **Status:** Kompatibel, wenn `Bcolors` verwendet wird.

---

## 2. reduction_scripts_supervisors

### n2/2_extract_data_supervisors.py ✅
- **Imports:** `ost_photometry.analyze.analyze`, `ost_photometry.style`
- **Status:** Kompatibel.

### n2/2_extract_data_students.py ✅
- **Imports:** `ost_photometry.analyze.analyze.main_extract`, `ost_photometry.analyze.plots`, `ost_photometry.utilities` (find_wcs_astrometry, Image), `ost_photometry.analyze.utilities`
- **Status:** Kompatibel – `find_wcs_astrometry` wird weiter über `utilities` re-exportiert.

### n2/3_plot_cmd_supervisors.py ✅
- **Imports:** `ost_photometry.analyze.plots`, `ost_photometry.analyze.utilities`, `ost_photometry.checks`, `ost_photometry.utilities`, `ost_photometry.style.Bcolors`
- **Status:** Kompatibel.

### n2/2b_post_process.py ✅
- **Imports:** `ost_photometry.analyze.analyze`, `ost_photometry.analyze.utilities`, `ost_photometry.style`, `ost_photometry.checks`
- **Status:** Kompatibel.

---

## 3. auxiliary_scripts

### 3.1 img_shift_cut_etc/

#### remove_overscan.py ✅
- **Imports:** `ost_photometry.checks`
- **Status:** Kompatibel.

#### cut_imgs.py ✅
- **Imports:** `ost_photometry.checks`
- **Status:** Kompatibel.

#### bin_imgs.py ✅
- **Imports:** `ost_photometry.checks`
- **Status:** Kompatibel.

#### flip_imgs_fits.py ✅
- **Imports:** `ost_photometry.reduce.utilities`
- **Verwendung:** `utilities.flip_image`
- **Status:** Kompatibel – `flip_image` bleibt in `reduce.utilities`.

#### flip_imgs.py ⚠️
- **Imports:** `ost_photometry.checks`, `ost_photometry.utilities`
- **Verwendung:** `utilities.mk_file_list(..., add_path_to_file_names=True)`
- **Hinweis:** API verwendet `add_path_to_file_names`, nicht `addpath`.
- **Status:** Kompatibel, falls `add_path_to_file_names=True` verwendet wird.

#### imgs_shifts.py ❌ **Anpassung nötig**
- **Imports:** `ost_photometry.checks`, `ost_photometry.utilities`, `ost_photometry.reduce.utilities`
- **Problem 1:** `utilities_base.mk_file_list(..., addpath=True)` – Parameter heißt heute `add_path_to_file_names=True`.
- **Anpassung:** `addpath=True` ersetzen durch `add_path_to_file_names=True`.

### 3.2 mk_calib_photometry/

#### 1_reduce_images.py ❌ **Anpassung nötig**
- **Imports:** `ost_photometry.reduce.redu`, `ost_photometry.reduce.aux`
- **Problem 1:** Modul `ost_photometry.reduce.aux` existiert nicht. `prepare_reduction` liegt in `ost_photometry.reduce.utilities`.
- **Problem 2:** `redu.reduce_main` wird mit veralteten Parametern aufgerufen: `cosmics`, `dr`, `readnoise`, `satlevel`, `objlim`, `sigclip`, `verbose` statt `rm_cosmic_rays`, `dark_rate`, `read_noise`, `saturation_level`, `limiting_contrast_rm_cosmic_rays`, `sigma_clipping_value_rm_cosmic_rays`, `debug`.
- **Vorschlag:**
  ```python
  from ost_photometry.reduce import redu, utilities as aux
  # ...
  rawfiles = aux.prepare_reduction(...)
  redu.reduce_main(
      rawfiles, outdir, img_type,
      gain=gain,
      read_noise=readnoise,          # war: readnoise
      dark_rate=dark_rate,            # war: dr
      rm_cosmic_rays=rmcos,          # war: cosmics
      saturation_level=satlevel,     # war: satlevel
      limiting_contrast_rm_cosmic_rays=objlim,  # war: objlim
      sigma_clipping_value_rm_cosmic_rays=sigclip,  # war: sigclip
      debug=verbose,                  # war: verbose
  )
  ```

#### 2_mk_trans.py ✅ **Migriert**
- **Neue Imports:** `ost_photometry.analyze` (Observation, ImageSeries, analyze, calibration, correlate, utilities)
- **Umstellung:** `image_container` → `Observation`, `image_ensemble` → `ImageSeries`, `aux.find_wcs` → `utilities.find_wcs`, `correlate_preserve_calibs` → `correlate.correlate_preserve_calibration_objects`, `trans.calculate_trans` → `calibration.calculate_trans`
- **Hinweis:** `aux.add_median_table` hat in der aktuellen API keine Entsprechung und wurde entfernt.

#### 2_mk_trans_add.py ✅ **Migriert**
- **Wie 2_mk_trans.py** – gleiche Umstellung auf die neue API.

#### 3_second_order_extinction.py ❌ **Anpassung nötig**
- **Imports:** `ost_photometry.analyze.aux` (lin_func, fit_curve)
- **Problem:** `ost_photometry.analyze.aux` existiert nicht. `lin_func` und `fit_curve` liegen in `ost_photometry.analyze.utilities`.
- **Anpassung:**
  ```python
  from ost_photometry.analyze.utilities import lin_func, fit_curve
  ```

#### 3_second_order_extinction_add.py ❌ **Anpassung nötig**
- **Wie 3_second_order_extinction.py** – gleiche Anpassung.

#### style-Referenz (3_second_order_extinction*.py)
- **Import:** `ost_photometry.style.bcolors`
- **Hinweis:** Korrekt ist `Bcolors` (Großbuchstabe): `from ost_photometry.style import Bcolors` oder `ost_photometry.style.Bcolors`.

---

## 4. Übersicht der notwendigen Änderungen

| Skript | Änderung |
|--------|----------|
| **auxiliary_scripts/img_shift_cut_etc/imgs_shifts.py** | `addpath=True` → `add_path_to_file_names=True` |
| **auxiliary_scripts/mk_calib_photometry/1_reduce_images.py** | `aux` → `utilities as aux`, Parameter von `reduce_main` anpassen |
| **auxiliary_scripts/mk_calib_photometry/3_second_order_extinction.py** | Import von `lin_func`, `fit_curve` aus `analyze.utilities` statt `analyze.aux` |
| **auxiliary_scripts/mk_calib_photometry/3_second_order_extinction_add.py** | Wie oben |
| **auxiliary_scripts/mk_calib_photometry/2_mk_trans.py** | ✅ Migriert |
| **auxiliary_scripts/mk_calib_photometry/2_mk_trans_add.py** | ✅ Migriert |

---

## 5. Style-Modul: Bcolors vs. bcolors

Das Modul `ost_photometry.style` definiert die Klasse `Bcolors` (mit großem „B“). Skripte, die `style.bcolors` oder `from ost_photometry.style import bcolors` verwenden, müssen auf `Bcolors` umgestellt werden.

---

## 6. Zusammenfassung

- **reduction_scripts_students:** Alle geprüften Skripte sind kompatibel oder benötigen nur kleine Anpassungen (z. B. Bcolors).
- **reduction_scripts_supervisors:** Geprüfte Skripte sind kompatibel.
- **auxiliary_scripts:**
  - Einfache Anpassungen: `imgs_shifts.py`, `1_reduce_images.py`, `3_second_order_extinction*.py` (erledigt)
  - Migration abgeschlossen: `2_mk_trans.py` und `2_mk_trans_add.py` auf neue API umgestellt
