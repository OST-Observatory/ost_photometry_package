# OST Photometry Package
Photometry reduction and analysis package for the 
[OST Observatory](https://polaris.astro.physik.uni-potsdam.de/) of 
the University of Potsdam.

The package aims to provide easy-to-use data reduction and analysis 
functionality for photometric observations made with the observatory's 
telescopes and instruments. It is mainly used in the astrophysics laboratory 
courses of the university and provides on-the-fly data reduction for the 
data archive of the observatory. It is designed with these applications in 
mind. 

## Requirements

The following Python packages provide the core functionality of this 
package:
   
   * [ccdproc](https://github.com/astropy/ccdproc)
   * [photutils](https://github.com/astropy/photutils)
   * [astropy](https://github.com/astropy/astropy)
   * [astroquery](https://github.com/astropy/astroquery)
   * [numpy](https://github.com/numpy/numpy)
   * [scipy](https://github.com/scipy/scipy)
   * [mathplotlib](https://github.com/matplotlib/matplotlib)

For astrometric solutions a local installation of the 
[astronomy.net](https://nova.astrometry.net/) software is required.

## Differential pipeline: calibration epochs

The analysis pipeline’s differential calibration step builds **multi-band epoch tables** (`epoch_000`, …) instead of per-image keys such as `B_0`. See [MIGRATION_calibration_epochs.md](src/ost_photometry/analyze/pipeline/MIGRATION_calibration_epochs.md) if you have custom code using older `PhotometryCalibrator` method names (`add_frame`, …) or obsolete bridge helpers.