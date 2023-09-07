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
   * [uncertainties](https://github.com/lebigot/uncertainties)
   * [mathplotlib](https://github.com/matplotlib/matplotlib)

For astrometric solutions a local installation of the 
[astronomy.net](https://nova.astrometry.net/) software is required.