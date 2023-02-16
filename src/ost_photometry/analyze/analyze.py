
############################################################################
####                            Libraries                               ####
############################################################################

import os

import numpy as np

from uncertainties import unumpy

from collections import Counter

from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

from photutils import (
    DAOStarFinder,
    EPSFBuilder,
    )
from photutils.psf import (
    extract_stars,
    DAOGroup,
    IterativelySubtractedPSFPhotometry,
    )
from photutils.detection import IRAFStarFinder
from photutils.background import (
    MMMBackground,
    MADStdBackgroundRMS,
    )

import ccdproc as ccdp

from astropy.stats import SigmaClip

from astropy.table import Table
from astropy.time import Time
from astropy.nddata import NDData
from astropy.stats import (gaussian_sigma_to_fwhm, sigma_clipped_stats)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.coordinates import SkyCoord, Angle, matching
import astropy.units as u

#   hips2fits module is not in the Ubuntu 22.04 package version
#   of astroquery (0.4.1)
#from astroquery.hips2fits import hips2fits
from astroquery.hips2fits import hips2fitsClass

from astropy.nddata import CCDData

from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    CircularAnnulus,
    )
from photutils.background import Background2D, MedianBackground
#from photutils.utils import calc_total_error

import multiprocessing as mp

from . import aux, calib, trans, plot, correlate, subtraction

from .. import style, checks, terminal_output, calibration_data

from .. import aux as base_aux

############################################################################
####                        Routines & definitions                      ####
############################################################################

class image_container:
    '''
        Container class for image class objects
    '''
    def __init__(self, **kwargs):
        #   Prepare dictionary
        self.ensembles = {}

        #   Add additional key words
        self.__dict__.update(kwargs)

        #   Check for right ascension and declination
        ra  =  kwargs.get('ra', None)
        dec = kwargs.get('dec', None)
        if ra is not None:
            self.ra  = Angle(ra, unit='hour').degree
        else:
            self.ra = None
        if dec is not None:
            self.dec = Angle(dec, unit='degree').degree
        else:
            self.dec = None

        #   Check for a object name
        self.name  = kwargs.get('name', None)

        #   Create SkyCoord object
        if self.name is not None and self.ra is None and self.dec is None:
            self.coord = SkyCoord.from_name(self.name)
        elif self.ra is not None and self.dec is not None:
            self.coord = SkyCoord(
                ra=self.ra,
                dec=self.dec,
                unit=(u.degree, u.degree),
                frame="icrs"
                )
        else:
            self.coord = None

        #   Check if uncertainty should be calculated by means of the
        #   "uncertainties" package. Default is ``True``.
        self.unc = kwargs.get('unc', True)


    #   Get ePSF objects of all images
    def get_epsf(self):
        epsf_dict = {}
        for key, ensemble in self.ensembles.items():
            epsf_list = []
            for img in ensemble.image_list:
                epsf_list.append(img.epsf)
            epsf_dict[key] = epsf_list

        return epsf_dict

    #   Get ePSF object of the reference image
    def get_ref_epsf(self):
        epsf_dict = {}
        for key, ensemble in self.ensembles.items():
            ref_id = ensemble.ref_id

            img = ensemble.image_list[ref_id]

            epsf_dict[key] = img.epsf

        return epsf_dict

    #   Get reference image
    def get_ref_img(self):
        img_dict = {}
        for key, ensemble in self.ensembles.items():
            ref_id = ensemble.ref_id

            img = ensemble.image_list[ref_id]

            img_dict[key] = img.get_data()

        return img_dict

    #   Get residual image belonging to the reference image
    def get_ref_residual_img(self):
        img_dict = {}
        for key, ensemble in self.ensembles.items():
            ref_id = ensemble.ref_id

            img = ensemble.image_list[ref_id]

            img_dict[key] = img.residual_image

        return img_dict

    #   Get image ensembles for a specific set of filter
    def get_ensembles(self, filter_list):
        ensembles = {}
        for filt in filter_list:
            ensembles[filt] = self.ensembles[filt]

        return ensembles


class image_ensemble:
    '''
        Image ensemble class: Used to handle multiple images, e.g.,
        an image series taken in a specific filter
    '''
    def __init__(self, filt, obj_name, path, outdir, ref_ID):
        ###
        #   Get file list, if path is a directory, if path is a file put
        #   base name of this file in a list
        #
        if os.path.isdir(path):
            formats=[".FIT",".fit",".FITS",".fits"]
            fileList = os.listdir(path)

            #   Remove not FITS entries
            tempList = []
            for file_i in fileList:
                for j, form in enumerate(formats):
                    if file_i.find(form)!=-1:
                        tempList.append(file_i)
            fileList=tempList
        elif os.path.isfile(path):
            fileList=[str(path).split('/')[-1]]
            path = os.path.dirname(path)
        else:
            raise RuntimeError(
                f'{style.bcolors.FAIL}ERROR: Provided path is neither a file'
                f' nor a directory -> EXIT {style.bcolors.ENDC}'
                )

        ###
        #   Check if the id of the reference image is valid
        #
        if ref_ID > len(fileList):
            raise ValueError(
                f'{style.bcolors.FAIL} ERROR: Reference image ID [ref_ID] '
                'is larger than the total number of images!'
                f' -> EXIT {style.bcolors.ENDC}'
                )

        #   Set filter
        self.filt = filt

        #   Set number of images
        self.nfiles = len(fileList)

        #   Set ID of the reference image
        self.ref_id = ref_ID

        #   Prepare image list
        self.image_list = []

        #   Set path to output directory
        self.outpath = Path(outdir)

        #   Set object name
        self.objname = obj_name

        #   Fill image list
        terminal_output.print_terminal(
            string="Read images and calculate FOV, PIXEL scale, etc. ... ",
            indent=2,
            )
        for pd, file_name in enumerate(fileList):
            self.image_list.append(
                #   Prepare image class instance
                self.image(pd, filt, obj_name, path, file_name, outdir)
                )

            #   Calculate field of view and additional quantities and add
            #   them to the image class instance
            base_aux.cal_fov(self.image_list[pd], verbose=False)

        #   Set reference image
        self.ref_img = self.image_list[ref_ID]

        #   Set field of view
        self.fov = self.ref_img.fov

        #   Set PixelRegion for the field of view
        self.region_pix = self.ref_img.region_pix

        #   Set pixel scale
        self.pixscale = self.ref_img.pixscale

        #   Set coordinates of image center
        self.coord = self.ref_img.coord

        #   Set instrument
        self.instrument = self.ref_img.instrument

        #   Get image shape
        self.img_shape = self.ref_img.get_data().shape

    #   Image class
    class image:
        def __init__(self, pd, filt, obj_name, path, file_name, outdir):
            #   Set image ID
            self.pd       = pd
            #   Set filter
            self.filt     = filt
            #   Set object name
            self.objname  = obj_name
            #   Set file name
            self.filename = file_name
            #   Set complete path
            self.path     = Path(Path(path) / file_name)
            #   Set path to output directory
            self.outpath  = Path(outdir)

        #   Read image
        def read_image(self):
            return CCDData.read(self.path)

        #   Get header
        def get_header(self):
            return CCDData.read(self.path).meta

        #   Get data
        def get_data(self):
            return CCDData.read(self.path).data

        #   Get shape
        def get_shape(self):
            return CCDData.read(self.path).data.shape

    #   Set wcs
    def set_wcs(self, w):
        self.wcs = w
        for img in self.image_list:
            img.wcs = w

    #   Get extracted photometry of all images
    def get_photometry(self):
        photo_dict = {}
        for img in self.image_list:
            photo_dict[str(img.pd)] = img.photometry

        return photo_dict

    #   Get image IDs of all images
    def get_image_ids(self):
        img_ids = []
        for img in self.image_list:
            img_ids.append(img.pd)

        return img_ids

    #   Get sigma clipped mean of the air mass
    def mean_sigma_clip_air_mass(self):
        am_list = []
        for img in self.image_list:
            am_list.append(img.air_mass)

        return sigma_clipped_stats(am_list, sigma=1.5)[0]

    #   Get median of the air mass
    def median_air_mass(self):
        am_list = []
        for img in self.image_list:
            am_list.append(img.air_mass)

        return np.median(am_list)

    #   Get air mass
    def get_air_mass(self):
        am_list = []
        for img in self.image_list:
            am_list.append(img.air_mass)

        return am_list

    #   Get observation times
    def get_obs_time(self):
        obs_time_list = []
        for img in self.image_list:
            obs_time_list.append(img.jd)

        return np.array(obs_time_list)

    #   Get median of the observation time
    def median_obs_time(self):
        obs_time_list = []
        for img in self.image_list:
            obs_time_list.append(img.jd)

        return np.median(obs_time_list)

    #   Get list with dictionary and image class objects
    def get_list_dict(self):
        dict_list = []
        for img in self.image_list:
            dict_list.append({img.filt:img})

        return dict_list


def rm_cosmic(image, objlim=5., readnoise=8., sigclip=4.5, satlevel=65535.,
              verbose=False, addmask=True):
    '''
        Remove cosmic rays

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        objlim          : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip         : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel        : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        addmask         : `boolean`, optional
            If True add hot and bad pixel mask to the reduced science images.
            Default is ``True``.
    '''
    terminal_output.print_terminal(
        #indent=1,
        string="Remove cosmic rays ...",
        )

    #   Get image
    ccd = image.read_image()

    #   Get status cosmic ray removal status
    status = ccd.meta.get('cosmics_rm', False)

    #   Get unit of the image to check if the image was scaled with the
    #   exposure time
    if ccd.unit == u.electron / u.s:
        scaled   = True
        exposure = ccd.meta.get('exptime', 1.)
        reduced  = ccd.multiply(exposure * u.second)
    else:
        scaled  = False
        reduced = ccd

    if not status:
        #   Remove cosmic rays
        reduced = ccdp.cosmicray_lacosmic(
            reduced,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
            verbose=verbose,
            )
        if not addmask:
            reduced.mask = np.zeros(reduced.shape, dtype=bool)
        if verbose:
            terminal_output.print_terminal()

        #   Add Header keyword to mark the file as combined
        reduced.meta['cosmics_rm'] = True

        #   Reapply scaling if image was scaled with the exposure time
        if scaled:
            reduced  = reduced.divide(exposure * u.second)

        #   Set file name
        basename  = base_aux.get_basename(image.filename)
        file_name = basename+'_cosmic-rm.fit'

        #   Set new file name and path
        image.filename = file_name
        image.path     = os.path.join(
            str(image.outpath),
            'cosmics_rm',
            file_name,
            )

        #   Check if the 'cosmics_rm' directory already exits.
        #   If not, create it.
        checks.check_out(os.path.join(str(image.outpath), 'cosmics_rm'))

        #   Save image
        reduced.write(image.path, overwrite=True)


def mk_bg(image, sigma_bkg=5., D2=True, apply_background=True,
          verbose=False):
    '''
        Determine background, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        sigma_bkg           : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        D2                  : `boolean`, optional
            If True a 2D background will be estimated and subtracted.
            Default is ``True``.

        apply_background    : `boolean`, optional
            If True path and file name will be set to the background
            subtracted images, so that those will automatically be used in
            further processing steps.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.
    '''
    if verbose:
        terminal_output.print_terminal(
            image.filt,
            string="Determine background: {:s} band",
            indent=2,
            )

    #   Load image data
    img    = image.read_image()

    #   Set up sigma clipping
    sigma_clip = SigmaClip(sigma=sigma_bkg)

    #   Calculate background RMS
    bkgrms        = MADStdBackgroundRMS(sigma_clip=sigma_clip)
    image.std_rms = bkgrms(img.data)

    #   2D background?
    if D2:
        #   Estimate 2D background
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            img.data,
            (50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            )

        #   Remove background
        img_no_bg = img.subtract(bkg.background * u.electron / u.s)

        #   Put meta data back on the image, because it is lost while
        #   subtracting the background
        img_no_bg.meta = img.meta
        img_no_bg.meta['HIERARCH'] = '2D background removed'

        #   Add Header keyword to mark the file as background subtracted
        img_no_bg.meta['NO_BG'] = True

        #   Get median of the background
        bkg_value = bkg.background_median
    else:
        #   Estimate 1D background
        mmm_bkg    = MMMBackground(sigma_clip=sigma_clip)
        bkg_value  = mmm_bkg.calc_background(img.data)

        #   Remove background
        img_no_bg = img.subtract(bkg_value)

        #   Put meta data back on the image, because it is lost while
        #   subtracting the background
        img_no_bg.meta = img_meta
        img_no_bg.meta['HIERARCH'] = '1D background removed'

        #   Add Header keyword to mark the file as background subtracted
        img_no_bg.meta['NO_BG'] = True


    #   Define name and save image
    file_name = base_aux.get_basename(image.filename)+'_nobg.fit'
    outpath = image.outpath / 'nobg'
    checks.check_out(outpath)
    img_no_bg.write(outpath / file_name, overwrite=True)


    #   Set new path and file
    #   -> Background subtracted image will be used in further processing steps
    if apply_background:
        image.path     = outpath / file_name
        image.filename = file_name

    #   Add background value to image class
    image.bkg_value  = bkg_value


def find_stars(image, sigma_psf, multi_start=5., method='IRAF',
               verbose=False, condense=False, indent=2):
    '''
        Find the stars on the images, using photutils and search and select
        stars for the ePSF stars

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        verbose         : `boolean`, optional
            If True addtional information will be printed to the terminal.
            Default is ``False``.

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal
    '''
    outstring = terminal_output.print_terminal(
        indent=indent,
        string="Identify stars",
        condense=condense,
        )

    #   Load image data
    img = image.read_image()

    #   Get background RMS
    sigma = image.std_rms

    #   Distinguish between different finder options
    if method == 'DAO':
        #   Set up DAO finder
        daofind = DAOStarFinder(
            fwhm=sigma_psf*gaussian_sigma_to_fwhm,
            threshold=multi_start*sigma
            )

        #   Find stars - make table
        tbl_posi_all = daofind(img.data)
    elif method == 'IRAF':
        #   Set up IRAF finder
        iraffind = IRAFStarFinder(
                threshold=multi_start*sigma,
                fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                minsep_fwhm=0.01,
                roundhi=5.0,
                roundlo=-5.0,
                sharplo=0.0,
                sharphi=2.0,
                )

        #   Find stars - make table
        tbl_posi_all = iraffind(img.data)
    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL}\nExtraction method ({method}) not valid: "
            f"use either IRAF or DAO {style.bcolors.ENDC}"
            )

    #   Add positions to image class
    image.positions = tbl_posi_all

    if condense:
        return outstring


def check_epsf_stars(image, size=25, min_stars=25, frac_epsf=0.2,
                     condense=False, strict=True, indent=2):
    '''
        Select ePSF stars and check if there are enough

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        size            : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        min_stars       : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        frac_epsf       : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        strict          : `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal
    '''
    #   Get object positions
    tbl = image.positions

    #   Number of objects
    num_stars = len(tbl)

    #   Get image data
    data = image.get_data()

    #   Combine identification string
    istring = str(image.pd)+'. '+image.filt

    #   Useful information
    outstring = terminal_output.print_terminal(
        num_stars,
        istring,
        indent=indent+1,
        string="{:d} sources identified in the {:s} band image",
        style_name='OKBLUE',
        condense=condense,
        )

    ##  Determine sample of stars used for estimating the ePSF
    #   (rm the brightest 1% of all stars because those are often saturated)
    #   Sort list with star positions according to flux
    tbl_sort = tbl.group_by('flux')
    # Determine the 99 percentile
    p99=np.percentile(tbl_sort['flux'], 99)
    #   Determine the position of the 99 percentile in the position list
    id_p99=np.argmin(np.absolute(tbl_sort['flux']-p99))

    #   Based on the input list, set the minimal number of stars
    frac=int(num_stars*frac_epsf)
    #   If the minimal number of stars ('frac') is lower than 'min_stars'
    #   set it to 'min_stars' (the default is 25 as required by the cutout
    #   plots, 25 also appears to be reasonable for a good ePSF)
    if frac < min_stars:
        frac = min_stars

    #   Check if enough stars have been identified
    if (id_p99-frac < min_stars and strict) or (id_p99-frac<1 and not strict):
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNot enough stars ({id_p99-frac}) found "
            f"to determine the ePSF in the {istring} band{style.bcolors.ENDC}"
            )

    #   Resize table -> limit it to the suitable stars
    tbl_posi = tbl_sort[:][id_p99-frac:id_p99]

    ##  Exclude stars that are too close to the image boarder
    #   Size of the extraction box around each star
    hsize = (size - 1) / 2

    #   New lists with x and y positions
    x = tbl_posi['xcentroid']
    y = tbl_posi['ycentroid']

    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] -1 - hsize)))

    #   Updated positions table
    tbl_posi  = tbl_posi[:][mask]
    num_clean = len(tbl_posi)

    #   Check if there are still enough stars
    if (num_clean < min_stars and strict) or (num_clean<1 and not strict):
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNot enough stars ({num_clean}) for the "
            f"ePSF determination in the {istring} band image. Too many "
            "potential ePSF stars have been removed, because they are too "
            "close to the image border. Check first that enough stars have "
            "been identified, using the starmap_?.pdf files.\n If that is "
            "the case, shrink extraction region or allow for higher fraction "
            "of ePSF stars (size_epsf) from all identified stars "
            f"(frac_epsf_stars). {style.bcolors.ENDC}"
            )


    ##  Find all potential ePSF stars with close neighbors
    dist_min = size

    #   Define and fill new arrays
    x1 = tbl_sort['xcentroid']
    y1 = tbl_sort['ycentroid']
    x2 = tbl_posi['xcentroid']
    y2 = tbl_posi['ycentroid']
    nmax = np.max((len(x1),len(x2)))
    xall = np.zeros((nmax,2))
    yall = np.zeros((nmax,2))
    xall[0:len(x1),0] = x1
    xall[0:len(x2),1] = x2
    yall[0:len(y1),0] = y1
    yall[0:len(y2),1] = y2

    id_p99 = correlate.newsrcor(
        xall,
        yall,
        dist_min,
        option=3,
        silent=True,
        )[1]

    #   Determine multiple entries -> stars that are contaminated
    id_p99_mult = [ite for ite,count in Counter(id_p99).items() if count>1]
    num_spoiled = len(id_p99_mult)

    #   Determine unique entries -> stars that are not contaminated
    id_p99_uniq = [ite for ite,count in Counter(id_p99).items() if count==1]
    num_clean   = len(id_p99_uniq)

    #   Remove ePSF stars with close neighbors from the corresponding table
    tbl_posi.remove_rows(id_p99_mult)

    #   Check if there are still enough stars
    if (num_clean < min_stars and strict) or (num_clean<1 and not strict):
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNot enough stars ({num_clean}) for the "
            f"ePSF determination in the {istring} band image. Too many "
            "potential ePSF stars have been removed, because other "
            "stars are in the extraction region. Check first that enough "
            "stars have been identified, using the starmap_?.pdf files.\n"
            "If that is the case, shrink extraction region or allow for "
            "higher fraction of ePSF stars (size_epsf) from all identified "
            f"stars (frac_epsf_stars). {style.bcolors.ENDC}"
            )

    #   Add ePSF stars to image class
    image.positions_epsf = tbl_posi

    if condense:
        return outstring


def mk_epsf(image, size=25, oversampling=2, maxiters=7,
                min_stars=25, multi=True, condense=False, indent=2):
    '''
        Main function to determine the ePSF, using photutils

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        size            : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Dewfault is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        min_stars       : `float`, optional
            Minimal number of stars required for the ePSF calculations
            Default is ``25``.

        multi           : `boolean`, optional
            If True multi processing is used for plotting.
            Default is ``True``.

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal
    '''
    #   Get image data
    data = image.get_data()

    #   Get ePSF star positions
    tbl_posi = image.positions_epsf

    #   Number of ePSF stars
    num_fit =  len(tbl_posi)

    #   Get object name
    nameobj = image.objname

    outstring = terminal_output.print_terminal(
            indent=indent,
            string="Determine the point spread function",
            condense=condense,
            )
    outstring = terminal_output.print_terminal(
            num_fit,
            indent=indent+1,
            string="{:d} bright stars used",
            style_name='OKBLUE',
            condense=condense,
            )


    #   Create new table with the names required by "extract_stars"
    stars_tbl = Table()
    stars_tbl['x'] = tbl_posi['xcentroid']
    stars_tbl['y'] = tbl_posi['ycentroid']

    #   Put image into NDData container (required by "extract_stars")
    nddata = NDData(data=data)

    #   Extract cutouts of the selected stars
    stars = extract_stars(nddata, stars_tbl, size=size)

    #   Combine plot identification string
    string = 'img-'+str(image.pd)+'-'+image.filt

    #   Get output directory
    outdir = image.outpath.name

    #   Plot the brightest ePSF stars
    if multi:
        p = mp.Process(
            target=plot.plot_cutouts,
            args=(outdir, stars, string),
            kwargs={'nameobj':nameobj,}
            )
        p.start()
    else:
        if condense:
            tmpstr = plot.plot_cutouts(
                outdir,
                stars,
                string,
                nameobj=nameobj,
                condense=True,
                )
            outstring += tmpstr
        else:
            plot.plot_cutouts(
                outdir,
                stars,
                string,
                max_plot_stars=min_stars,
                nameobj=nameobj,
                )

    #   Build the ePSF (set oversampling and max. number of iterations)
    epsf_builder = EPSFBuilder(
        oversampling=oversampling,
        maxiters=maxiters,
        progress_bar=False,
        )
    epsf, fitted_stars = epsf_builder(stars)

    #   Add ePSF and fitted stars to image class
    image.epsf         = epsf
    image.fitted_stars = fitted_stars

    if condense:
    	return outstring


def epsf_extract(image, sigma_psf, sigma_bkg=5., use_init_guesses=True,
                 method_finder='IRAF', size_epsf=25.,  multi=5.0,
                 multi_grouper=2.0, strict_cleaning=True, condense=False,
                 rmbackground=False, indent=2):
    '''
        Main function to perform the eEPSF photometry, using photutils

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        sigma_psf           : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg           : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        use_init_guesses    : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method_finder       : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        size_epsf           : `integer`, optional
            Size of the extraction region in pixel
            Default is ``25``.

        multi               : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper       : `float`, optional
            Multiplier for the DAO grouper
            Default is ``2.0``.

        strict_cleaning     : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        condense            : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        rmbackground        : `boolean`, optional
            If True the background will be estimated and considered.
            Default is ``False``. -> It is expected that the background
            was removed before.

        indent              : `integer`, optional
            Indentation for the console output lines
            Default is ``2`.

    '''
    #   Get output path
    out_path = image.outpath

    #   Check output directories
    checks.check_out(
        out_path,
        out_path / 'tables',
        )

    #   Get image data
    data = image.get_data()

    #   Get filter
    filt = image.filt

    #   Get already identified objects (position and flux)
    if use_init_guesses:
        try:
            #   Get position information
            positions_flux = image.positions
            init_guesses = Table(
                names=['x_0','y_0', 'flux_0'],
                data=[
                    positions_flux['xcentroid'],
                    positions_flux['ycentroid'],
                    positions_flux['flux'],
                    ]
                )
        except:
            #   Switch to backup in case positions and fluxes are not
            #   available
            use_init_guesses=False

    #   Set output and plot identification string
    istring = str(image.pd)+'-'+filt

    #   Get background RMS
    sigma = image.std_rms

    #   Get ePSF
    epsf = image.epsf

    outstr = terminal_output.print_terminal(
            istring,
            indent=indent,
            string="Performing the actual PSF photometry ({:s} image)",
            condense=condense,
            )

    ##  Set up all necessary classes
    if method_finder == 'IRAF':
        #   IRAF finder
        finder = IRAFStarFinder(
            threshold=multi*sigma,
            fwhm=sigma_psf*gaussian_sigma_to_fwhm,
            minsep_fwhm=0.01,
            roundhi=5.0,
            roundlo=-5.0,
            sharplo=0.0,
            sharphi=2.0,
            )
    elif method_finder == 'DAO':
        #   DAO finder
        finder = DAOStarFinder(
            fwhm=sigma_psf*gaussian_sigma_to_fwhm,
            threshold=multi*sigma,
            exclude_border=True,
            )
    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nExtraction method ({method_finder}) "
            f"not valid: use either IRAF or DAO {style.bcolors.ENDC}"
            )
    #   Fitter used
    fitter = LevMarLSQFitter()

    #   Size of the extraction region
    if size_epsf % 2 == 0:
        sizepho = size_epsf + 1
    else:
        sizepho = size_epsf

    #   Number of iterations
    niter=1

    #   Set up sigma clipping
    if rmbackground:
        sigma_clip = SigmaClip(sigma=sigma_bkg)
        mmm_bkg    = MMMBackground(sigma_clip=sigma_clip)
    else:
        mmm_bkg = None

    try:
        #   DAO grouper
        daogroup = DAOGroup(multi_grouper*sigma_psf*gaussian_sigma_to_fwhm)

        #  Set up the overall class to extract the data
        photometry = IterativelySubtractedPSFPhotometry(
            finder=finder,
            group_maker=daogroup,
            bkg_estimator=mmm_bkg,
            psf_model=epsf,
            fitter=fitter,
            niters=niter,
            fitshape=(sizepho, sizepho),
            aperture_radius=(sizepho-1)/2
            )

        #   Extract the photometry and make a table
        if use_init_guesses:
            result_tbl = photometry(image=data, init_guesses=init_guesses)
        else:
            result_tbl = photometry(image=data)
    except RuntimeError as e:
        if multi_grouper != 1.:
            terminal_output.print_terminal(
                indent=indent,
                string="IterativelySubtractedPSFPhotometry failed. " \
                       "Will try again with 'multi_grouper' set to 1...",
                style_name='WARNING',
                )
            multi_grouper = 1.
            #   DAO grouper
            daogroup = DAOGroup(
                multi_grouper*sigma_psf*gaussian_sigma_to_fwhm
                )

            #  Set up the overall class to extract the data
            photometry = IterativelySubtractedPSFPhotometry(
                finder=finder,
                group_maker=daogroup,
                bkg_estimator=mmm_bkg,
                psf_model=epsf,
                fitter=fitter,
                niters=niter,
                fitshape=(sizepho, sizepho),
                aperture_radius=(sizepho-1)/2
                )

            #   Extract the photometry and make a table
            if use_init_guesses:
                result_tbl = photometry(image=data, init_guesses=init_guesses)
            else:
                result_tbl = photometry(image=data)
        else:
            terminal_output.print_terminal(
                indent=0,
                string=e,
                )
            #print(e)

    #   Check if result table contains a 'flux_unc' column
    #   For some reason, it's missing for some extractions....
    if 'flux_unc' not in result_tbl.colnames:
        #   Calculate a very very rough approximation of the uncertainty
        #   by means of the actual extraction result 'flux_fit' and the
        #   early estimate 'flux_0'
        est_unc = np.absolute(
            result_tbl['flux_fit'] - result_tbl['flux_0']
            )
        result_tbl.add_column(est_unc, name='flux_unc')

    #   Clean output for objects with negative uncertainties
    try:
        num_spoiled  = 0
        spoiled_fits = np.where(result_tbl['flux_fit'].data < 0.)
        result_tbl.remove_rows(spoiled_fits)
        num_spoiled  = np.size(spoiled_fits)
        if strict_cleaning:
            spoiled_fits = np.where(result_tbl['flux_unc'].data < 0.)
            num_spoiled  += len(spoiled_fits)
            result_tbl.remove_rows(spoiled_fits)
    except:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nProblem with cleanup of negative "
            f"uncertainties... {style.bcolors.ENDC}"
            )

    #   Clean output for objects with negative pixel coordinates
    try:
        spoiled_fits  = np.where(result_tbl['x_fit'].data < 0.)
        num_spoiled  += np.size(spoiled_fits)
        result_tbl.remove_rows(spoiled_fits)
        spoiled_fits  = np.where(result_tbl['y_fit'].data < 0.)
        num_spoiled  += np.size(spoiled_fits)
        result_tbl.remove_rows(spoiled_fits)
    except:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nProblem with cleanup of negative pixel "
            f"coordinates... {style.bcolors.ENDC}"
            )

    if num_spoiled != 0:
        strout = terminal_output.print_terminal(
                num_spoiled,
                indent=indent+1,
                string="{:d} objects removed because of poor fit quality",
                condense=condense,
                )
        if condense: outstr += strout

    try:
        nstars = len(result_tbl['flux_fit'].data)
    except:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nTable produced by "
            "IterativelySubtractedPSFPhotometry is empty after cleaning up "
            "of objects with negative pixel coordinates and negative "
            f"uncertainties {style.bcolors.ENDC}"
            )

    strout = terminal_output.print_terminal(
            nstars,
            indent=indent+1,
            string="{:d} good stars extracted from the image",
            style_name='OKBLUE',
            condense=condense,
            )
    if condense: outstr += strout


    #   Remove objects that are too close to the image edges
    result_tbl, strout = aux.rm_edge_objects(
        result_tbl,
        data,
        (sizepho-1)/2,
        condense=condense,
        )
    if condense: outstr += strout

    #   Write table
    filename = 'table_photometry_{}_PSF.dat'.format(istring)
    result_tbl.write(
        out_path / 'tables' / filename,
        format='ascii',
        overwrite=True,
        )

    #  Make residual image
    residual_image = photometry.get_residual_image()

    #   Add photometry and residual image to image class
    image.photometry     = result_tbl
    image.residual_image = residual_image

    if condense:
        return outstr


def compute_phot_error(flux_variance, ap_area, nsky, stdev, gain=1.0):
    '''
        This function is largely borrowed from the Space Telescope Science
        Institute's wfc3_photometry package:

        https://github.com/spacetelescope/wfc3_photometry

        It computes the flux errors using the DAOPHOT style computation:

        err = sqrt (Poisson_noise / gain
            + ap_area * stdev**2
            + ap_area**2 * stdev**2 / nsky)

        Parameters
        ----------
        flux_variance       : `numpy.ndarray`
            Extracted aperture flux data or the error^2 of the extraction
            if available -> proxy for the Poisson noise

        ap_area             : `float`
            Photometric aperture area

        nsky                : `fLoat`
            Sky annulus area

        stdev               : `numpy.ndarray`
            Uncertainty in the sky measurement

        gain               : `float`, optional
            Electrons per ADU
            Default is ``1.0``. Usually we already work with gain corrected
            data.
    '''

    #   Calculate flux error as above
    bg_variance_terms = (ap_area * stdev ** 2. ) * (1. + ap_area/nsky)
    variance = flux_variance / gain + bg_variance_terms
    flux_error = variance ** .5

    return flux_error


def define_apertures(image, r, r_in, r_out, r_unit):
    '''
        Define stellar and background apertures

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        r                   : `float`
            Radius of the stellar aperture

        r_in                : `float`
            Inner radius of the background annulus

        r_out               : `float`
            Outer radius of the background annulus

        r_unit              : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        Returns
        -------
        aperture            : `photutils.aperture.CircularAperture`
            Stellar aperture

        annulus_aperture    : `photutils.aperture.CircularAnnulus`
            Background annulus
    '''
    #   Get position information
    tbl = image.positions

    #   Extract positions and prepare a position list
    try:
        lst1 = tbl['x_fit']
        lst2 = tbl['y_fit']
    except:
        lst1 = tbl['xcentroid']
        lst2 = tbl['ycentroid']
    positions = list(zip(lst1,lst2))

    #   Check unit of radii
    if r_unit not in ['pixel', 'arcsec']:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nUnit of the aperture radii not valid: "
            f"set it either to pixel or arcsec {style.bcolors.ENDC}"
            )

    #   Convert radii in arcsec to pixel
    #   (this part is prone to errors and needs to be rewritten)
    pixscale = image.pixscale
    if pixscale != None and r_unit == 'arcsec':
        r     = r/pixscale
        r_in  = r_in/pixscale
        r_out = r_out/pixscale

    #   Make stellar aperture
    aperture = CircularAperture(positions, r=r)

    #   Make background annulus
    annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    return aperture, annulus_aperture


def background_simple(image, annulus_aperture):
    '''
        Calculate background from annulus

        Parameters
        ----------
        image               : `image.class`
            Image class with all image specific properties

        annulus_aperture    : `photutils.aperture.CircularAnnulus`
            Background annulus

        Returns
        -------
        bkg_median          : `float`
            Median of the background

        bkg_stdev           : `float`
            Standard deviation of the background
    '''
    bkg_median = []
    bkg_stdev  = []

    #   Calculate mask from background annulus
    annulus_masks = annulus_aperture.to_mask(method='center')

    #   Loop over all masks
    for mask in annulus_masks:
        #   Extract annulus data
        annulus_data = mask.multiply(image.get_data())

        #   Convert annulus data to 1D
        annulus_data_1d = annulus_data[mask.data > 0]

        #   Sigma clipping
        _, median_sigclip, median_stdev = sigma_clipped_stats(annulus_data_1d)

        #   Add to list
        bkg_median.append(median_sigclip)
        bkg_stdev.append(median_stdev)

    #   Convert to numpy array
    bkg_median = np.array(bkg_median)
    bkg_stdev  = np.array(bkg_stdev)

    return bkg_median, bkg_stdev


def aperture_extract(image, r, r_in, r_out, r_unit='pixel', bg_simple=False,
                     plotaper=False, condense=False, indent=2):
    '''
        Perform aperture photometry using the photutils.aperture package

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        r               : `float`
            Radius of the stellar aperture

        r_in            : `float`
            Inner radius of the background annulus

        r_out           : `float`
            Outer radius of the background annulus

        r_unit          : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        bg_simple       : `boolean`, optional
            If True the background will be extract by a simple algorithm that
            calculates the median within the background annulus. If False the
            background will be extracted using
            photutils.aperture.aperture_photometry.
            Default is ``False``.

        plotaper        : `boolean`, optional
            IF true a plot showing the apertures in relation to image is
            created.
            Default is ``False``.

        condense        : `boolean`, optional
            If True the terminal output will be returned to the calling
            function.
            Default is ``False``.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        Returns
        -------
        outstring       : `string`, optional
            Information to be printed to the terminal

            img_err     - numpy.ndarray or None
                          Error array for 'image'
    '''
    #   Load image data and uncertainty
    img  = image.read_image()
    data = img.data
    err  = img.uncertainty.array

    #   Get filter
    filt = image.filt

    outstr = terminal_output.print_terminal(
        filt,
        indent=indent,
        string="Performing aperture photometry ({:s} image)",
        condense=condense,
        )


    ###
    #   Define apertures
    #
    aperture, annulus_aperture = define_apertures(
        image,
        r,
        r_in,
        r_out,
        r_unit,
        )


    ###
    #   Extract background and calculate median
    #
    if bg_simple:
        bkg_median, bkg_stdev = background_simple(image, annulus_aperture)


    ###
    #   Extract photometry
    #
    #   Extract aperture
    phot = aperture_photometry(data, aperture, mask=img.mask, error=err)

    #   Extract background aperture
    if not bg_simple:
        bkg_phot = aperture_photometry(
            data,
            annulus_aperture,
            mask=img.mask,
            error=err,
            )

        #   Calculate aperture background and the corresponding error
        phot['aper_bkg'] = bkg_phot['aperture_sum'] * aperture.area \
                        / annulus_aperture.area

        phot['aper_bkg_err'] = bkg_phot['aperture_sum_err'] * aperture.area \
                        / annulus_aperture.area
    else:
        #   Add median background to the output table
        phot['annulus_median'] = bkg_median

        #   Calculate background for the apertures add to the output table
        phot['aper_bkg'] = bkg_median * aperture.area

    #   Subtract background from aperture flux and add it to the
    #   output table
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']

    #   Define flux column
    #   (necessary to have the same column names for aperture and PSF
    #   photometry)
    phot['flux_fit'] = phot['aper_sum_bkgsub']

    # Error estimate
    if err is not None:
        err_column = phot['aperture_sum_err']
    else:
        err_column = phot['flux_fit']**(0.5)

    if bg_simple:
        bg_err = bkg_stdev
    else:
        bg_err = phot['aper_bkg_err']

    phot['flux_unc'] = compute_phot_error(
        err_column,
        aperture.area,
        annulus_aperture.area,
        bg_err,
        )

    #   Rename position columns
    phot.rename_column('xcenter', 'x_fit')
    phot.rename_column('ycenter', 'y_fit')

    #   Remove objects that are too close to the image edges
    phot, strout = aux.rm_edge_objects(phot, data, r_out, condense=condense)
    if condense:
        outstr += strout

    #   Replace negative flux values with 10^-10
    #   -> arbitrary, but very small
    flux = np.array(phot['flux_fit'])
    mask = np.where(flux <= 0.)
    phot['flux_fit'][mask] = 1E-10

    #   Add photometry to image class
    image.photometry = phot


    ###
    #   Plot star map with aperture overlay
    #
    if plotaper:
        plot.plot_apertures(
            image.outpath.name,
            data,
            aperture,
            annulus_aperture,
            filt,
            )

    #   Number of stars
    nstars = len(flux)

    #   Useful info
    strout = terminal_output.print_terminal(
        nstars,
        indent=indent+1,
        string="{:d} good stars extracted from the image",
        style_name='OKBLUE',
        condense=condense,
        )

    if condense: return outstr+strout


def correlate_images(*args, **kwargs):
    '''
        Wrapper function: distinguish between astropy table
                          and pandas data frame
    '''
    if base_aux.dict_vs_df(args[1]):
        return correlate_pd(*args, **kwargs)
    else:
        return correlate_tbl(*args, **kwargs)


#@timeis
def correlate_tbl(outdir, result_tbl, arr_img_IDs, dcr=3., option=1,
                  maxid=1, refORI=0, refOBJ=[], nmissed=1, bfrac=1.0,
                  s_refOBJ=True):
    '''
        WARNING: This function is not up to date

        Correlate star lists from the stacked images of all filters to find
        those stars that are visible on all images -> write calibrated CMD

        Parameters
        ----------
        outdir              : `string`
            Output directory

        result_tbl          : `dictionary` - `astropy.table.Table`
            Dictionary of astropy tables with the position and flux data

        arr_img_IDs         : `numpy.ndarray` or `list`
            Image IDs

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        refORI              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        refOBJ              : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        Returns
        -------
        ind_sort            : `numpy.ndarray`
            IDs of the images

        x_sort              : `numpy.ndarray`
            Position of the objects on the image in pixel in X direction

        y_sort              : `numpy.ndarray`
            Position of the objects on the image in pixel in X direction

        flux_arr            : `numpy.ndarray`
            Numpy array with the data of all stars such as magnitudes

        reject              : `numpy.ndarray`
            IDs of the rejected images

        rej_obj             : `numpy.ndarray`
            IDs of the rejected objects

        count               : `integer
            Number of matches found
    '''
    terminal_output.print_terminal(
        arr_img_IDs,
        indent=1,
        string="Correlate results from the images ({:s})",
        )

    #   Define variables
    nimg      = len(arr_img_IDs)
    nmax_list = []
    x = []
    y = []

    #   Number of objects in each table/image
    for i, img_ID in enumerate(arr_img_IDs):
        x.append(result_tbl[str(img_ID)]['x_fit'])
        y.append(result_tbl[str(img_ID)]['y_fit'])
        nmax_list.append(len(x[i]))

    #   Max. number of objects
    nmax = np.max(nmax_list)

    #   Define and fill new arrays
    xall = np.zeros((nmax,nimg))
    yall = np.zeros((nmax,nimg))
    for i in range(0,nimg):
        xall[0:len(x[i]),i] = x[i]
        yall[0:len(y[i]),i] = y[i]

    #   Correlate the results from the two images
    indSR, reject, count, rej_obj = correlate.newsrcor(
        xall,
        yall,
        dcr,
        bfrac=bfrac,
        option=option,
        maxid=maxid,
        refORI=refORI,
        refOBJ=refOBJ,
        nmissed=nmissed,
        s_refOBJ=s_refOBJ,
        )

    if count == 1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nOnly one common object "
            f"found!{style.bcolors.ENDC}"
            )
    elif count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo common objects "
            f"found!{style.bcolors.ENDC}"
            )

    nbad = len(reject)
    if nbad > 0:
        terminal_output.print_terminal(
            nbad,
            indent=2,
            string="{:d} images do not meet the criteria -> removed",
            )

    #   Remove bad origins listed in 'reject'
    indSR       = np.delete(indSR, reject, 0)
    arr_img_IDs = np.delete(arr_img_IDs, reject, 0)

    xall        = np.delete(xall, reject, 1)
    yall        = np.delete(yall, reject, 1)

    #   Calculate shift for the reference origin
    shiftID = np.argwhere(reject < refORI)
    Nshift  = len(shiftID)
    refORI -= Nshift

    # Number of clean images
    nclean = len(indSR[:,0])

    #   Rearrange arrays based on the newsrcor results
    x_sort   = np.zeros((count))
    y_sort   = np.zeros((count))
    ind_sort = np.arange(count)

    for i in range(0,count):
        x_sort[i]   = xall[indSR[refORI][i]][refORI]
        y_sort[i]   = yall[indSR[refORI][i]][refORI]

    #   Array for the flux and uncertainty
    flux_arr = np.zeros(nclean, dtype=[('flux_fit', 'f8', (count)),
                                       ('flux_unc', 'f8', (count)),
                                      ]
                    )

    #   Fill arrays
    for j, img_ID in enumerate(arr_img_IDs):
        img_ID = str(img_ID)

        for i in range(0, count):
            flux     = result_tbl[img_ID]['flux_fit'][indSR[j,i]]
            flux_err = result_tbl[img_ID]['flux_unc'][indSR[j,i]]
            flux_arr['flux_fit'][j][i] = flux
            flux_arr['flux_unc'][j][i] = flux_err

    #   Remove nans
    flux_arr['flux_unc'] = np.nan_to_num(
        flux_arr['flux_unc'],
        nan=9999.,
        posinf=9999.,
        neginf=9999.,
        )

    return ind_sort, x_sort, y_sort, flux_arr, reject, rej_obj, count


def correlate_ensemble_img(img_ensemble, dcr=3., option=1, maxid=1,
                           refORI=0, refOBJ=[], nmissed=1, bfrac=1.0,
                           s_refOBJ=True, correl_method='astropy',
                           seplimit=2.*u.arcsec):
    '''
        Correlate star lists from the stacked images of all filters to find
        those stars that are visible on all images -> write calibrated CMD

        Parameters
        ----------
        img_ensemble        : `image ensemble`
            Ensemble of images, e.g., taken in one filter

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        refORI              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        refOBJ              : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    '''
    #   Get image IDs
    arr_img_IDs = img_ensemble.get_image_ids()

    terminal_output.print_terminal(
        arr_img_IDs,
        indent=1,
        string="Correlate results from the images ({})",
        )

    #   Get WCS
    w = img_ensemble.wcs

    #   Get dictionary with astropy tables with the position and flux data
    result_tbl = img_ensemble.get_photometry()

    #   Number of images
    nimg      = len(arr_img_IDs)

    #   Extract pixel positions of the objects -> needs to be improved!
    nmax_list = []
    x = []
    y = []
    for i, img_ID in enumerate(arr_img_IDs):
        x.append(result_tbl[str(img_ID)]['x_fit'])
        y.append(result_tbl[str(img_ID)]['y_fit'])
        nmax_list.append(len(x[i]))

    #   Max. number of objects
    nmax = np.max(nmax_list)

    #   Define and fill new arrays
    xall = np.zeros((nmax,nimg))
    yall = np.zeros((nmax,nimg))

    for i in range(0,nimg):
        xall[0:len(x[i]),i] = x[i]
        yall[0:len(y[i]),i] = y[i]

    #   Correlate the object positions from the images
    #   -> find common objects
    if correl_method == 'astropy':
        #   Astropy version: 2x faster than own
        indSR, reject = correlate.astropycor(
            x,
            y,
            w,
            refORI=refORI,
            refOBJ=refOBJ,
            nmissed=nmissed,
            s_refOBJ=s_refOBJ,
            seplimit=seplimit,
            )
        count = len(indSR[0])

    elif correl_method == 'own':
        #   Own version based on srcor from the IDL Astro Library
        indSR, reject, count, rej_obj = correlate.newsrcor(
            xall,
            yall,
            dcr,
            bfrac=bfrac,
            option=option,
            maxid=maxid,
            refORI=refORI,
            refOBJ=refOBJ,
            nmissed=nmissed,
            s_refOBJ=s_refOBJ,
            )
    else:
        raise ValueError(
            f'{style.bcolors.FAIL}Correlation method not known. Expected: '
            f'"own" or astropy, but got "{correl_method}"{style.bcolors.ENDC}'
            )

    ###
    #   Print correlation infos or raise error if not enough common
    #   objects were detected
    #
    if count == 1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nOnly one common object "
            f"found! {style.bcolors.ENDC}"
            )
    elif count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo common objects "
            f"found!{style.bcolors.ENDC}"
            )
    else:
        terminal_output.print_terminal(
            count,
            indent=2,
            string="{} objects identified on all images",
            )

    nbad = len(reject)
    if nbad > 0:
        terminal_output.print_terminal(
            nbad,
            indent=2,
            string="{:d} images do not meet the criteria -> removed",
            )
    if nbad > 1:
        terminal_output.print_terminal(
            reject,
            indent=2,
            string="Rejected image IDs: {}",
            )
    elif nbad == 1:
        terminal_output.print_terminal(
            reject,
            indent=2,
            string="ID of the rejected image: {}",
            )
    terminal_output.print_terminal()


    ###
    #   Post process corelation results
    #

    #   Remove "bad" datasets from index array
    #   (only necessary for 'own' method)
    if correl_method == 'own':
        indSR       = np.delete(indSR, reject, 0)

    # Number of "clean" datasets
    nclean = len(indSR[:,0])

    #   Remove "bad" datasets from image IDs -> used in a later step
    arr_img_IDs = np.delete(arr_img_IDs, reject, 0)

    #   Calculate new index of the reference origin
    shiftID = np.argwhere(reject < refORI)
    Nshift  = len(shiftID)
    refORI_new = refORI-Nshift


    ###
    #   Rearrange arrays based on the correlation results
    #
    #   Prepare new arrays for positions and indexes
    x_sort   = np.zeros((count))
    y_sort   = np.zeros((count))
    ind_sort = np.arange(count)

    #   Fill position arrays -> distinguish between input sources
    #                           depending on correlation method
    if correl_method == 'astropy':
        x_sort = x[refORI][indSR[refORI_new]].value
        y_sort = y[refORI][indSR[refORI_new]].value

    elif correl_method == 'own':
        #   Remove "bad" datasets first
        xall        = np.delete(xall, reject, 1)
        yall        = np.delete(yall, reject, 1)

        x_sort = xall[indSR[refORI_new]][:,refORI_new]
        y_sort = yall[indSR[refORI_new]][:,refORI_new]

    #   Prepare array for the flux and uncertainty (all datasets)
    flux_arr = np.zeros(nclean, dtype=[('flux_fit', 'f8', (count)),
                                       ('flux_unc', 'f8', (count)),
                                      ]
                    )

    #   Fill flux arrays
    for j, img_ID in enumerate(arr_img_IDs):
        img_ID_str = str(img_ID)

        #   Flux and uncertainty array for individual images
        flux_img = np.zeros(
            count,
            dtype=[('flux_fit', 'f8'), ('flux_unc', 'f8')],
            )

        #   Rearrange flux and error
        flux_img['flux_fit'] = result_tbl[img_ID_str]['flux_fit'][indSR[j,:]]
        flux_img['flux_unc'] = result_tbl[img_ID_str]['flux_unc'][indSR[j,:]]

        #   Remove nans etc. in error
        flux_img['flux_unc'] = np.nan_to_num(
            flux_img['flux_unc'],
            nan=9999.,
            posinf=9999.,
            neginf=9999.,
            )

        #   Remove '--' entries in error
        flux_err_dash = np.argwhere(flux_img['flux_unc'] == '--')
        flux_img['flux_unc'][flux_err_dash] = 9999.

        uflux_img = unumpy.uarray(
            flux_img['flux_fit'],
            flux_img['flux_unc']
            )

        #   Add sorted flux data and positions back to the image
        img_ensemble.image_list[img_ID].flux    = flux_img
        img_ensemble.image_list[img_ID].uflux   = uflux_img
        img_ensemble.image_list[img_ID].x_sort  = x_sort
        img_ensemble.image_list[img_ID].y_sort  = y_sort
        img_ensemble.image_list[img_ID].id_sort = ind_sort


        #   Add to overall array
        flux_arr['flux_fit'][j] = flux_img['flux_fit']
        flux_arr['flux_unc'][j] = flux_img['flux_unc']

    uflux_arr = unumpy.uarray(
            flux_arr['flux_fit'],
            flux_arr['flux_unc']
            )

    #   Update image ensemble object and add IDs, pixel coordinates, and
    #   flux of the correlated objects
    img_list = img_ensemble.image_list
    img_list = np.delete(img_list, reject)
    img_ensemble.image_list = img_list
    img_ensemble.nfiles     = len(img_list)
    img_ensemble.ref_id     = refORI_new

    img_ensemble.id_s = ind_sort
    img_ensemble.x_s  = x_sort
    img_ensemble.y_s  = y_sort
    img_ensemble.flux = flux_arr
    img_ensemble.uflux = uflux_arr


def correlate_ensemble(img_container, filt_list, dcr=3., option=1, maxid=1,
                       refORI=0, refOBJ=[], nmissed=1, bfrac=1.0,
                       s_refOBJ=True, correl_method='astropy',
                       seplimit=2.*u.arcsec):
    '''
        Correlate star lists from the stacked images of all filters to find
        those stars that are visible on all images -> write calibrated CMD

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filt_list           : `list` of `string`
            List with filter identifiers.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        refORI              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        refOBJ              : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    '''
    terminal_output.print_terminal(
        indent=1,
        string="Correlate results from image ensembles",
        )

    #   Get image ensembles
    ensemble_dict = img_container.get_ensembles(filt_list)

    #   Define variables
    nobj_list = []
    keys = []
    x = []
    y = []
    w = []

    #   Number of objects in each table/image
    for key, ensemble in ensemble_dict.items():
        keys.append(key)
        x.append(ensemble.x_s)
        y.append(ensemble.y_s)
        nobj_list.append(len(ensemble.x_s))
        w.append(ensemble.wcs)

    #   Max. number of objects
    nobj_max = np.max(nobj_list)

    #   Number of ''images''/image ensembles
    nimg = len(x)

    #   Define and fill new arrays
    xall = np.zeros((nobj_max, nimg))
    yall = np.zeros((nobj_max, nimg))

    for i in range(0,nimg):
        xall[0:len(x[i]),i] = x[i]
        yall[0:len(y[i]),i] = y[i]

    #   Correlate the object positions from the images
    #   -> find common objects
    if correl_method == 'astropy':
        #   Astropy version: ~2x faster than own
        indSR, reject = correlate.astropycor(
            x,
            y,
            w[refORI],
            refORI,
            refOBJ,
            nmissed,
            s_refOBJ,
            cleanup_advanced=False,
            seplimit=seplimit,
            )
        count = len(indSR[0])

    elif correl_method == 'own':
        #   Own version based on srcor from the IDL Astro Library
        indSR, reject, count, rej_obj = correlate.newsrcor(
            xall,
            yall,
            dcr,
            bfrac=bfrac,
            option=option,
            maxid=maxid,
            refORI=refORI,
            refOBJ=refOBJ,
            nmissed=nmissed,
            s_refOBJ=s_refOBJ,
            )
    else:
        raise ValueError(
            f'{style.bcolors.FAIL}Correlation method not known. Expected: '
            f'"own" or astropy, but got "{correl_method}"{style.bcolors.ENDC}'
            )

    ###
    #   Print correlation infos or raise error if not enough common
    #   objects were detected
    #
    if count == 1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nOnly one common object "
            f"found! {style.bcolors.ENDC}"
            )
    elif count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo common objects "
            f"found!{style.bcolors.ENDC}"
            )
    else:
        terminal_output.print_terminal(
            count,
            indent=2,
            string="{} objects identified on all ensemble",
            )

    nbad = len(reject)
    if nbad > 0:
        terminal_output.print_terminal(
            nbad,
            indent=2,
            string="{:d} images do not meet the criteria -> removed",
            )
    if nbad > 1:
        terminal_output.print_terminal(
            reject,
            indent=2,
            string="Rejected ensemble IDs: {}",
            )
    elif nbad == 1:
        terminal_output.print_terminal(
            reject,
            indent=2,
            string="ID of the rejected ensembles: {}",
            )
    terminal_output.print_terminal()


    ###
    #   Post process corelation results
    #

    #   Remove "bad" datasets from index array
    #   (only necessary for 'own' method)
    if correl_method == 'own':
        indSR = np.delete(indSR, reject, 0)

    #   Remove "bad"/rejected ensembles
    for ject in reject:
        ensemble_dict.pop(keys[ject])

    #   Calculate shift for the reference origin
    shiftID = np.argwhere(reject < refORI)
    Nshift  = len(shiftID)
    refORI_new = refORI-Nshift


    ###
    #   Rearrange arrays based on the correlation results
    #
    #   Prepare new arrays for positions and indexes
    x_sort   = np.zeros((count))
    y_sort   = np.zeros((count))
    ind_sort = np.arange(count)

    #   Fill position arrays -> distinguish between input sources
    #                           depending on correlation method
    if correl_method == 'astropy':
        if (isinstance(x[refORI], u.quantity.Quantity) or
            isinstance(x[refORI], Table)):
                x_sort = x[refORI][indSR[refORI_new]].value
                y_sort = y[refORI][indSR[refORI_new]].value
        elif isinstance(x[refORI], np.ndarray):
            x_sort = x[refORI][indSR[refORI_new]]
            y_sort = y[refORI][indSR[refORI_new]]
        else:
            raise TypeError(
                f"{style.bcolors.FAIL} \nType of the position arrays not "
                "known. Expect numpy.float or astropy.units.quantity.Quantity "
                f"but got {type(x[refORI])} {style.bcolors.ENDC}"
                )

    elif correl_method == 'own':
        #   Remove "bad" datasets first
        xall = np.delete(xall, reject, 1)
        yall = np.delete(yall, reject, 1)

        x_sort = xall[indSR[refORI_new]][:,refORI_new]
        y_sort = yall[indSR[refORI_new]][:,refORI_new]


    #   Loop over image ensembles to rearrange flux arrays
    for j, ensemble in enumerate(ensemble_dict.values()):

        #   Get image list
        img_list = ensemble.image_list

        #   Get number of images
        nimg = len(img_list)

        #   Overall array for the flux and uncertainty
        flux_arr = np.zeros(nimg, dtype=[('flux_fit', 'f8', (count)),
                                          ('flux_unc', 'f8', (count)),
                                          ]
                        )

        #   Loop over images -> assumes that the images/results within each
        #   ensemble are already correlated such that the objects have the
        #   same indexes
        for z, img in enumerate(img_list):
            #   Get flux
            flux = img.flux
            uflux = img.uflux

            #   Define new flux array
            flux_sort = np.zeros(
                count,
                dtype=[('flux_fit', 'f8'), ('flux_unc', 'f8')]
                )

            #   Rearrange flux
            flux_sort['flux_fit'] = flux['flux_fit'][indSR[j,:]]
            flux_sort['flux_unc'] = flux['flux_unc'][indSR[j,:]]

            uflux_sort = uflux[indSR[j,:]]

            #   Add sorted flux data and positions back to the image
            img.flux_es = flux_sort
            img.uflux_es = uflux_sort
            img.x_es = x_sort
            img.y_es = y_sort
            img.id_es = ind_sort

            #   Add to overall array
            flux_arr['flux_fit'][z] = flux_sort['flux_fit']
            flux_arr['flux_unc'][z] = flux_sort['flux_unc']


        uflux_arr = getattr(ensemble, 'uflux', None)
        if uflux_arr is None:
            uflux_arr_sort = unumpy.uarray(
                flux_arr['flux_fit'],
                flux_arr['flux_unc']
                )
        else:
            uflux_arr_sort = uflux_arr[:,indSR[j,:]]

        #   Update image ensemble object and add IDs, pixel coordinates, and
        #   flux of the correlated objects
        ensemble.id_es = ind_sort
        ensemble.x_es  = x_sort
        ensemble.y_es  = y_sort
        ensemble.flux_es = flux_arr
        ensemble.uflux_es = uflux_arr_sort


#@timeis
def correlate_pd(outdir, input_df, img_IDs, dcr, option, maxid=1,
                 refORI=0, refOBJ=[], nmissed=1, bfrac=1.0):
    '''
        WARNING: This function is not up to date

        Correlate star lists from the stacked images of all filters to find
        those stars that are visible on all images -> write calibrated CMD

        Parameters
        ----------
        outdir              : `string`
            Output directory

        input_df            : `pandas.DataFrame`
            Position and flux data

        img_IDs             : `numpy.ndarray` or `list`
            Image IDs (can be multidimensional, e.g., several images for
            different filter)

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        refORI              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        refOBJ              : `list` of `integer`, optional
            IDs of the reference objects. The reference objects will not be
            removed from the list of objects.
            Default is ``[]``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.


        Returns
        -------
        ind_sort            : `numpy.ndarray`
            IDs of the images

        x_sort              : `numpy.ndarray`
            Position of the objects on the image in pixel in X direction

        y_sort              : `numpy.ndarray`
            Position of the objects on the image in pixel in X direction

        flux_arr            : `numpy.ndarray`
            Numpy array with the data of all stars such as magnitudes

        reject              : `numpy.ndarray`
            IDs of the rejected images

        rej_obj             : `numpy.ndarray`
            IDs of the rejected objects

        count               : `integer
            Number of matches found
    '''
    terminal_output.print_terminal(
        img_IDs,
        indent=indent,
        string="Correlate results from the images ({:d})",
        )

    #   Number of images
    nimg = len(img_IDs)

    #   Define data frames for X and Y coordinates
    df_x = pd.DataFrame()
    df_y = pd.DataFrame()

    #   Loop over all images
    for ID in img_IDs:
        #   Restrict input data frame to current image
        mask    = input_df['type'] == ID
        ID_data = input_df[mask]

        #   Compile data frames with all X and Y positions
        df_x = pd.concat([df_x, ID_data['x_fit']], axis=1)
        df_y = pd.concat([df_y, ID_data['y_fit']], axis=1)

    #   Replace potential Nans with 0
    xall = df_x.fillna(0).to_numpy()
    yall = df_y.fillna(0).to_numpy()

    #   Correlate the results from the two images
    indSR, reject, count, rej_obj = correlate.newsrcor(
        xall,
        yall,
        dcr,
        option=option,
        maxid=maxid,
        refORI=refORI,
        refOBJ=refOBJ,
        nmissed=nmissed,
        bfrac=bfrac
        )

    #   Checks
    if count == 1:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nOnly one common object "
            f"found! {style.bcolors.ENDC}"
            )
    elif count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo common objects "
            f"found! {style.bcolors.ENDC}"
            )

    nbad = len(reject)
    if nbad > 0:
        terminal_output.print_terminal(
            nbad,
            indent=2,
            string="{:d} images do not meet the criteria -> removed",
            )

    #   Remove bad origins listed in 'reject'
    indSR   = np.delete(indSR, reject, 0)
    xall    = np.delete(xall,  reject, 1)
    yall    = np.delete(yall,  reject, 1)
    img_IDs = np.delete(img_IDs, reject, 0)

    #   Calculate shift for the reference origin
    shiftID = np.argwhere(reject < refORI)
    Nshift  = len(shiftID)
    refORI -= Nshift

    # Number of clean images
    nclean = len(indSR[:,0])

    #   Initialize new arrays for X, Y positions, and index
    x_sort   = np.zeros((count))
    y_sort   = np.zeros((count))
    ind_sort = np.arange(count)

    #   Sort X and Y positions according to correlation results
    for i in range(0,count):
        x_sort[i]   = xall[indSR[refORI][i]][refORI]
        y_sort[i]   = yall[indSR[refORI][i]][refORI]

    #   New data frame to return
    result = pd.DataFrame()

    #   Loop over all images --> Check if this loop can be removed by
    #                            dropping the data of removed images from
    #                            'input_df'
    for j, ID in enumerate(img_IDs):
        #   Restrict input data frame to current image
        mask = input_df['type'] == ID
        data = input_df[mask]

        #   Sort according to newsrcor results
        pre_result = data.copy().iloc[indSR[j]]

        #   Add new index
        pre_result['corr_index'] = ind_sort

        #    Rearrange index
        pre_result = pre_result.reset_index(level=0).set_index(['corr_index'])

        #   Add to data frame
        result = pd.concat([result, pre_result], axis=0)

    return ind_sort, x_sort, y_sort, result, reject, rej_obj, count


def correlate_preserve_calibs(img_ensemble, filter_list,
                              calib_method='APASS', mag_range=(0.,18.5),
                              vizier_dict={}, calib_file=None, dcr=3,
                              option=1, verbose=False, maxid=1, ref_ID=0,
                              nmissed=1, bfrac=1.0, s_refOBJ=True,
                              plot_test=True, correl_method='astropy',
                              seplimit=2.*u.arcsec):
    '''
        Correlate results from all images, while preserving the calibration
        stars

        Parameters
        ----------
        img_ensemble        : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        filter_list         : 'list` with `strings`
            Filter list

        calib_method       : `string`, optional
            Calibration method
            Default is ``APASS``.

        mag_range           : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        vizier_dict         : `dictionary`, optional
            Identifiers of catalogs, containing calibration data
            Derfault is ``{}``.

        calib_file          : `string`, optional
            Path to the calibration file
            Default is ``None``.

        dcr             : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option          : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        ref_ID              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        plot_test       : `boolean`, optional
            If True only the masterplot for the reference image will
            be created.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    '''
    ###
    #   Load calibration data
    #
    calib_tbl, col_names, ra_unit = calib.load_calib(
        img_ensemble.image_list[ref_ID],
        filter_list,
        calib_method=calib_method,
        mag_range=mag_range,
        vizier_dict=vizier_dict,
        calib_file=calib_file,
        )

    #   Number of calibration stars
    n_calib = len(calib_tbl)

    if n_calib == 0:
        raise Exception(
            f"{style.bcolors.FAIL} \nNo match between calibrations stars and "
            f"the\n extracted stars detected. -> EXIT {style.bcolors.ENDC}"
            )


    ###
    #   Find IDs of calibration stars to ensure they are not deleted in
    #   the correlation process
    #
    #   Lists for IDs, and xy coordinates
    calib_IDs = []
    calib_xs  = []
    calib_ys  = []

    #   Loop over all calibration stars
    for k in range(0,n_calib):
        #   Find the calibration star
        inds_obj, ref_count, x_obj, y_obj  = correlate.posi_obj_srcor_img(
            img_ensemble.image_list[ref_ID],
            calib_tbl[col_names['ra']].data[k],
            calib_tbl[col_names['dec']].data[k],
            img_ensemble.wcs,
            dcr=dcr,
            option=option,
            ra_unit=ra_unit,
            verbose=verbose,
            )
        if verbose:
            terminal_output.print_terminal()

        #   Add ID and coordinates of the calibration star to the lists
        if ref_count != 0:
            calib_IDs.append(inds_obj[1][0])
            calib_xs.append(x_obj)
            calib_ys.append(y_obj)
    terminal_output.print_terminal(
        len(calib_IDs),
        indent=3,
        string="{:d} matches",
        style_name='OKBLUE',
        )
    terminal_output.print_terminal()


    ###
    #   Correlate the results from all images
    #
    correlate_ensemble_img(
        img_ensemble,
        dcr=dcr,
        option=option,
        maxid=maxid,
        refORI=ref_ID,
        refOBJ=calib_IDs,
        nmissed=nmissed,
        bfrac=bfrac,
        s_refOBJ=s_refOBJ,
        correl_method=correl_method,
        seplimit=seplimit,
        )


    ###
    #   Plot image with the final positions overlaid (final version)
    #
    aux.prepare_and_plot_starmap_final_3(
        img_ensemble,
        calib_xs,
        calib_ys,
        plot_test=plot_test,
        )


def correlate_preserve_variable(img_ensemble, ra_obj, dec_obj, dcr=3.,
                                option=1, maxid=1, ref_ID=0, nmissed=1,
                                bfrac=1.0, s_refOBJ=True,
                                correl_method='astropy',
                                seplimit=2.*u.arcsec, verbose=False,
                                plot_test=True):
    '''
        Correlate results from all images, while preserving the variable
        star

        Parameters
        ----------
        img_ensemble    : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        ra_obj          : `float`
            Right ascension of the object

        dec_obj         : `float`
            Declination of the object

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        ref_ID              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True only the masterplot for the reference image will
            be created.
            Default is ``True``.
    '''
    ###
    #   Find position of the variable star I
    #
    terminal_output.print_terminal(
        indent=1,
        string="Identify the variable star",
        )

    if correl_method == 'astropy':
        variable_ID, count, x_obj, y_obj = correlate.posi_obj_astropy_img(
            img_ensemble.image_list[ref_ID],
            ra_obj,
            dec_obj,
            img_ensemble.wcs,
            seplimit=seplimit,
            )

    elif correl_method == 'own':
        inds_obj, count, x_obj, y_obj  = correlate.posi_obj_srcor_img(
            img_ensemble.image_list[ref_ID],
            ra_obj,
            dec_obj,
            img_ensemble.wcs,
            dcr=dcr,
            option=option,
            verbose=verbose,
            )

        #   Current object ID
        variable_ID  = inds_obj[1]

        if verbose:
            terminal_output.print_terminal()


    ###
    #   Check if variable star was detected I
    #
    if count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \tERROR: The variable object was not "
            f"detected in the reference image.\n\t-> EXIT{style.bcolors.ENDC}"
            )


    ###
    #   Correlate the stellar positions from the different filter
    #
    correlate_ensemble_img(
        img_ensemble,
        dcr=dcr,
        option=option,
        maxid=maxid,
        refORI=ref_ID,
        refOBJ=int(variable_ID),
        nmissed=nmissed,
        bfrac=bfrac,
        s_refOBJ=s_refOBJ,
        correl_method=correl_method,
        seplimit=seplimit,
        )


    ###
    #   Find position of the variable star II
    #
    terminal_output.print_terminal(
        indent=1,
        string="Reidentify the variable star",
        )

    if correl_method == 'astropy':
        variable_ID, count, x_obj, y_obj = correlate.posi_obj_astropy(
            img_ensemble.x_s,
            img_ensemble.y_s,
            ra_obj,
            dec_obj,
            img_ensemble.wcs,
            seplimit=seplimit,
            )

    elif correl_method == 'own':
        inds_obj, count, x_obj, y_obj  = correlate.posi_obj_srcor(
            img_ensemble.x_s,
            img_ensemble.y_s,
            ra_obj,
            dec_obj,
            img_ensemble.wcs,
            dcr=dcr,
            option=option,
            verbose=verbose,
            )
        if verbose:
            terminal_output.print_terminal()

        #   Current object ID
        variable_ID  = inds_obj[1]


    ###
    #   Check if variable star was detected II
    #
    if count == 0:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \tERROR: The variable was not detected "
            f"in the reference image.\n\t-> EXIT{style.bcolors.ENDC}"
            )


    ###
    #   Plot image with the final positions overlaid (final version)
    #
    aux.prepare_and_plot_starmap_final_3(
        img_ensemble,
        [x_obj],
        [y_obj],
        plot_test=plot_test,
        )

    #   Add ID of the variable star to the image ensemble
    img_ensemble.variable_ID = variable_ID


def extract_multiprocessing(img_ensemble, ncores, sigma_psf, sigma_bkg=5.,
                            multi_start=5., size_epsf=25,
                            frac_epsf_stars=0.2,
                            oversampling=2, maxiters=7,
                            epsf_use_init_guesses=True, method='IRAF',
                            multi=5.0, multi_grouper=2.0,
                            strict_cleaning=True, min_eps_stars=25,
                            photometry='PSF', rstars=5., rbg_in=7.,
                            rbg_out=10., r_unit='arcsec', strict_eps=True,
                            search_image=True, plot_ifi=False, plot_test=True):
    '''
        Extract flux and object positions using multiprocessing

        Parameters
        ----------
        img_ensemble    : `image.ensemble` object
            Ensemble class object with all image data taken in a specific
            filter

        ncores          : `integer`
            Number of cores to use during multiprocessing.

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg       : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf       : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi           : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper   : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars   : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry      : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars          : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in          : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit          : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps      ; `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        search_image    : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plot_ifi        : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    '''
    #   Get filter
    filt = img_ensemble.filt


    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    if not search_image:
        mk_bg(img_ensemble.ref_img, sigma_bkg=sigma_bkg)

        find_stars(
            img_ensemble.ref_img,
            sigma_psf[filt],
            multi_start=multi_start,
            method=method,
            )


    ###
    #   Main loop: Extract stars and info from all images, using
    #              multiprocessing
    #
    #   Initialize multiprocessing object
    executor = aux.Executor(ncores)

    #   Main loop
    for image in img_ensemble.image_list:
        #   Set positions of the reference image if required
        if not search_image:
            image.positions = img_ensemble.ref_img.positions

        #   Extract photometry
        executor.schedule(
            main_extract_condense,
            args=(
                image,
                sigma_psf[filt],
                ),
            kwargs={
                'sigma_bkg':sigma_bkg,
                'multi_start':multi_start,
                'size_epsf':size_epsf,
                'frac_epsf_stars':frac_epsf_stars,
                'oversampling':oversampling,
                'maxiters':maxiters,
                'epsf_use_init_guesses':epsf_use_init_guesses,
                'method':method,
                'multi':multi,
                'multi_grouper':multi_grouper,
                'strict_cleaning':strict_cleaning,
                'min_eps_stars':min_eps_stars,
                'strict_eps':strict_eps,
                'refid':img_ensemble.ref_id,
                'photometry':photometry,
                'rstars':rstars,
                'rbg_in':rbg_in,
                'rbg_out':rbg_out,
                'r_unit':r_unit,
                'search_image':search_image,
                'plot_ifi':plot_ifi,
                'plot_test':plot_test,
                }
            )
    #   Close multiprocessing pool and wait until it finishes
    executor.wait()

    #   Exit if exceptions occurred
    if executor.err is not None:
        raise RuntimeError(
            f'\n{style.bcolors.FAIL}Extraction using multiprocessing failed '
            f'for {filt} :({style.bcolors.ENDC}'
            )


    ###
    #   Sort multiprocessing results
    #
    #   Extract results
    res = executor.res

    #   Sort observation times and images & build dictionary for the
    #   tables with the extraction results
    tmp_list = []
    for j in range(0, img_ensemble.nfiles):
        for img in res:
            pd = img.pd
            if pd == j:
                tmp_list.append(img)

    img_ensemble.image_list = tmp_list


def main_extract_condense(image, sigma_psf, sigma_bkg=5., multi_start=5.,
                          size_epsf=25, frac_epsf_stars=0.2, oversampling=2,
                          maxiters=7, epsf_use_init_guesses=True,
                          method='IRAF', multi=5.0, multi_grouper=2.0,
                          strict_cleaning=True, min_eps_stars=25, refid=0,
                          photometry='PSF', rstars=5., rbg_in=7.,
                          rbg_out=10., r_unit='arcsec', strict_eps=True,
                          search_image=True, plot_ifi=False, plot_test=True):
    '''
        Main function to extract the information from the individual images

        Parameters
        ----------
        image           : `image.class`
            Image class with all image specific properties

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg       : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5``.

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf       : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi           : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper   : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars   : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        refid           : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry      : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars          : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in          : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit          : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps      ; `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        search_image    : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plot_ifi        : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    '''
    ###
    #   Initialize output string
    #
    outstring  = "      "
    outstring += style.bcolors.UNDERLINE
    outstring += "Image: "+str(image.pd)
    outstring += style.bcolors.ENDC+"\n"


    ###
    #   Estimate and remove background
    #
    mk_bg(image, sigma_bkg=sigma_bkg)


    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    if search_image:
        out_str = find_stars(
            image,
            sigma_psf,
            multi_start=multi_start,
            method=method,
            condense=True,
            )
        outstring += out_str

    if photometry == 'PSF':
        ###
        #   Check if enough stars have been detected to allow ePSF
        #   calculations
        #
        out_str = check_epsf_stars(
            image,
            size=size_epsf,
            min_stars=min_eps_stars,
            frac_epsf=frac_epsf_stars,
            condense=True,
            strict=strict_eps,
            )
        outstring += out_str


        ###
        #   Plot images with the identified stars overlaid
        #
        if plot_ifi or (plot_test and image.pd == refid):
            out_str = plot.starmap(
                image.outpath.name,
                image.get_data(),
                image.filt,
                image.positions,
                tbl_2=image.positions_epsf,
                label='identified stars',
                label_2='stars used to determine the ePSF',
                rts='initial-img-'+str(image.pd),
                nameobj=image.objname,
                condense=True,
                )
        else:
            out_str = ''
        outstring += out_str


        ###
        #   Calculate the ePSF
        #
        out_str = mk_epsf(
            image,
            size=size_epsf,
            oversampling=oversampling,
            maxiters=maxiters,
            min_stars=min_eps_stars,
            multi=False,
            condense=True,
            )
        outstring += out_str


        ###
        #   Plot the ePSFs
        #
        out_str = plot.plot_epsf(
            image.outpath.name,
            {'img-'+str(image.pd)+'-'+image.filt:image.epsf},
            condense=True,
            nameobj=image.objname,
            indent=2,
            )
        outstring += out_str


        ###
        #   Performing the PSF photometry
        #
        out_str = epsf_extract(
            image,
            sigma_psf,
            sigma_bkg=sigma_bkg,
            use_init_guesses=epsf_use_init_guesses,
            method_finder=method,
            size_epsf=size_epsf,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            condense=True,
            )
        outstring += out_str


        ###
        #   Plot original and residual image
        #
        out_str = plot.plot_residual(
            image.objname,
            {f'{image.pd}-{image.filt}':image.get_data()},
            {f'{image.pd}-{image.filt}':image.residual_image},
            #{str(image.pd)+'-'+image.filt:image.get_data()},
            #{str(image.pd)+'-'+image.filt:image.residual_image},
            image.outpath.name,
            condense=True,
            nameobj=image.objname,
            indent=2,
            )
        outstring += out_str

    elif photometry == 'APER':
        ###
        #   Perform aperture photometry
        #
        if image.pd == refid:
            plotaper=True
        else:
            plotaper=False

        out_str = aperture_extract(
            image,
            rstars,
            rbg_in,
            rbg_out,
            r_unit=r_unit,
            plotaper=plotaper,
            condense=True,
            indent=3,
            )
        outstring += out_str

    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nExtraction method ({photometry}) not "
            f"valid: use either aper or PSF {style.bcolors.ENDC}"
            )


    ###
    #   Plot images with extracted stars overlaid
    #
    if plot_ifi or (plot_test and image.pd == refid):
        out_str = aux.prepare_and_plot_starmap(image, condense=True)
    else:
        out_str = ''
    outstring += out_str
    outstring += '\n'
    terminal_output.print_terminal(
        indent=0,
        string=outstring,
        )

    return image



def main_extract(image, sigma_psf, sigma_bkg=5., multi_start=5.,
                 size_epsf=25, frac_epsf_stars=0.2, oversampling=2,
                 maxiters=7, epsf_use_init_guesses=True,
                 method='IRAF', multi=5.0, multi_grouper=2.0,
                 strict_cleaning=True, min_eps_stars=25, refid=0,
                 photometry='PSF', rstars=5., rbg_in=7., rbg_out=10.,
                 r_unit='arcsec', strict_eps=True, rmcos=False, objlim=5.,
                 readnoise=8., sigclip=4.5, satlevel=65535., plot_ifi=False,
                 plot_test=True):
    '''
        Main function to extract the information from the individual images

        Parameters
        ----------
        image                   : `image.class`
            Image class with all image specific properties

        sigma_psf               : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        sigma_bkg               : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start             : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        size_epsf               : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars         : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling            : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters                : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method                 : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi                   : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper           : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning         : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars           : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        refid                   : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry              : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars                  : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in                  : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out                 : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit                  : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps              ; `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        rmcos                   : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        objlim                  : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise               : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip                 : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel                : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plot_ifi                : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test               : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    '''
    ###
    #   Initialize output string
    #
    terminal_output.print_terminal(
        image.pd,
        indent=2,
        string="Image: {:d}",
        style_name='UNDERLINE',
        )


    ###
    #   Remove cosmics (optional)
    #
    if rmcos:
        rm_cosmic(
            image,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
            )


    ###
    #   Estimate and remove background
    #
    mk_bg(image, sigma_bkg=sigma_bkg)


    ###
    #   Find the stars (via DAO or IRAF StarFinder)
    #
    find_stars(
        image,
        sigma_psf,
        multi_start=multi_start,
        method=method,
        )


    if photometry == 'PSF':
        ###
        #   Check if enough stars have been detected to allow ePSF
        #   calculations
        #
        check_epsf_stars(
            image,
            size=size_epsf,
            min_stars=min_eps_stars,
            frac_epsf=frac_epsf_stars,
            strict=strict_eps,
            )


        ###
        #   Plot images with the identified stars overlaid
        #
        if plot_ifi or (plot_test and image.pd == refid):
            plot.starmap(
                image.outpath.name,
                image.get_data(),
                image.filt,
                image.positions,
                tbl_2=image.positions_epsf,
                label='identified stars',
                label_2='stars used to determine the ePSF',
                rts='initial-img-'+str(image.pd),
                nameobj=image.objname,
                )


        ###
        #   Calculate the ePSF
        #
        mk_epsf(
            image,
            size=size_epsf,
            oversampling=oversampling,
            maxiters=maxiters,
            min_stars=min_eps_stars,
            multi=False,
            )


        ###
        #   Plot the ePSFs
        #
        plot.plot_epsf(
            image.outpath.name,
            {'img-'+str(image.pd)+'-'+image.filt:image.epsf},
            nameobj=image.objname,
            indent=2,
            )


        ###
        #   Performing the PSF photometry
        #
        epsf_extract(
            image,
            sigma_psf,
            sigma_bkg=sigma_bkg,
            use_init_guesses=epsf_use_init_guesses,
            method_finder=method,
            size_epsf=size_epsf,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            )


        ###
        #   Plot original and residual image
        #
        plot.plot_residual(
            image.objname,
            {str(image.pd)+'-'+image.filt:image.get_data()},
            {str(image.pd)+'-'+image.filt:image.residual_image},
            image.outpath.name,
            nameobj=image.objname,
            indent='      ',
            )

    elif photometry == 'APER':
        ###
        #   Perform aperture photometry
        #
        if image.pd == refid:
            plotaper=True
        else:
            plotaper=False

        aperture_extract(
            image,
            rstars,
            rbg_in,
            rbg_out,
            r_unit=r_unit,
            plotaper=plotaper,
            indent=3,
            )
    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nExtraction method ({photometry}) not "
            f"valid: use either APER or PSF {style.bcolors.ENDC}"
            )

    #   Add flux array to image (is this really necessary?)
    flux_img = np.zeros(
        len(image.photometry['x_fit']),
        dtype=[('flux_fit', 'f8'), ('flux_unc', 'f8')],
        )
    flux_img['flux_fit'] = image.photometry['flux_fit']
    flux_img['flux_unc'] = image.photometry['flux_unc']

    uflux_img = unumpy.uarray(
            flux_img['flux_fit'],
            flux_img['flux_unc']
            )
    image.flux = flux_img
    image.uflux = uflux_img


    ###
    #   Plot images with extracted stars overlaid
    #
    if plot_ifi or (plot_test and image.pd == refid):
        aux.prepare_and_plot_starmap(image)

    terminal_output.print_terminal()


def extract_flux(img_container, filter_list, name, img_paths, outdir,
                 sigma_psf, wcs_method='astrometry', force_wcs_determ=False,
                 sigma_bkg=5., multi_start=5., size_epsf=25,
                 frac_epsf_stars=0.2, oversampling=2, maxiters=7,
                 epsf_use_init_guesses=True, method='IRAF', multi=5.0,
                 multi_grouper=2.0, strict_cleaning=True, min_eps_stars=25,
                 refid=0, strict_eps=True, photometry='PSF', rstars=5.,
                 rbg_in=7., rbg_out=10., r_unit='arcsec', rmcos=False,
                 objlim=5., readnoise=8., sigclip=4.5, satlevel=65535.,
                 plot_ifi=False, plot_test=True):
    '''
        Extract flux and fill the image container

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        filter_list     : `list` of `string`
            Filter list

        name            : `string`
            Name of the object

        img_paths       : `dictionary`
            Paths to images: key - filter name; value - path

        outdir          : `string`
            Path, where the output should be stored.

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        wcs_method      : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.


        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_bkg       : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        size_epsf       : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        epsf_use_init_guesses   : `boolean`, optional
            If True the initial positions from a previous object
            identification procedure will be used. If False the objects
            will be identified by means of the ``method_finder`` method.
            Default is ``True``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi           : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper   : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars   : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        refid           : `integer`, optional
            ID of the reference image
            Default is ``0``.

        photometry      : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars          : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in          : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit          : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps      ; `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        rmcos           : `bool`
            If True cosmic rays will be removed from the image.
            Default is ``False``.

        objlim          : `float`, optional
            Parameter for the cosmic ray removal: Minimum contrast between
            Laplacian image and the fine structure image.
            Default is ``5``.

        readnoise       : `float`, optional
            The read noise (e-) of the camera chip.
            Default is ``8`` e-.

        sigclip         : `float`, optional
            Parameter for the cosmic ray removal: Fractional detection limit
            for neighboring pixels.
            Default is ``4.5``.

        satlevel        : `float`, optional
            Saturation limit of the camera chip.
            Default is ``65535``.

        plot_ifi        : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    '''
    ###
    #   Check output directories
    #
    checks.check_out(
        outdir,
        os.path.join(outdir,'tables'),
        )

    ###
    #   Loop over all filter
    #
    for filt in filter_list:
        terminal_output.print_terminal(
            filt,
            string="Analyzing {:s} images",
            style_name='HEADER',
            )

        ###
        #   Check paths
        #
        checks.check_file(img_paths[filt])


        #   Initialize image ensemble object
        img_container.ensembles[filt] = image_ensemble(
            filt,
            name,
            img_paths[filt],
            outdir,
            0,
            )


        ###
        #   Find the WCS solution for the image
        #
        aux.find_wcs(
            img_container.ensembles[filt],
            ref_id=0,
            method=wcs_method,
            force_wcs_determ=force_wcs_determ,
            indent=3,
            )


        ###
        #   Main extraction
        #
        main_extract(
            img_container.ensembles[filt].image_list[0],
            sigma_psf[filt],
            sigma_bkg=sigma_bkg,
            multi_start=multi_start,
            size_epsf=size_epsf,
            frac_epsf_stars=frac_epsf_stars,
            oversampling=oversampling,
            maxiters=maxiters,
            epsf_use_init_guesses=epsf_use_init_guesses,
            method=method,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            min_eps_stars=min_eps_stars,
            strict_eps=strict_eps,
            photometry=photometry,
            rstars=rstars,
            rbg_in=rbg_in,
            rbg_out=rbg_out,
            r_unit=r_unit,
            rmcos=rmcos,
            objlim=objlim,
            readnoise=readnoise,
            sigclip=sigclip,
            satlevel=satlevel,
            plot_ifi=plot_ifi,
            plot_test=plot_test,
            )

        #   Add stellar positions to ensemble class
        #   TODO: Shift this into main extract
        photo = img_container.ensembles[filt].image_list[0].photometry
        img_container.ensembles[filt].x_s = photo['x_fit']
        img_container.ensembles[filt].y_s = photo['y_fit']


    if photometry == 'PSF':
        ###
        #   Plot the ePSFs
        #
        p = mp.Process(
            target=plot.plot_epsf,
            args=(outdir, img_container.get_ref_epsf(),),
            )
        p.start()


        ###
        #   Plot original and residual image
        #
        p = mp.Process(
            target=plot.plot_residual,
            args=(
                name,
                img_container.get_ref_img(),
                img_container.get_ref_residual_img(),
                outdir,
                ),
            kwargs={
                'nameobj':'reference image'
                }
            )
        p.start()


def extract_flux_multi(img_container, filter_list, name, img_paths, outdir,
                       sigma_psf, ra_obj, dec_obj, ncores=6,
                       wcs_method='astrometry', force_wcs_determ=False,
                       sigma_bkg=5., multi_start=5., size_epsf=25,
                       frac_epsf_stars=0.2, oversampling=2, maxiters=7,
                       method='IRAF', multi=5.0, multi_grouper=2.0,
                       strict_cleaning=True, min_eps_stars=25, strict_eps=True,
                       photometry='PSF', rstars=5., rbg_in=7., rbg_out=10.,
                       r_unit='arcsec', dcr=3., option=1, maxid=1, ref_ID=0,
                       nmissed=1, bfrac=1.0, s_refOBJ=True,
                       correl_method='astropy', seplimit=2.*u.arcsec,
                       verbose=False, search_image=True, plot_ifi=False,
                       plot_test=True):
    '''
        Extract flux from multiple images per filter and add results to
        the image container

        Parameters
        ----------
        img_container   : `image.container`
            Container object with image ensemble objects for each filter

        filter_list     : `list` of `string`
            Filter list

        name            : `string`
            Name of the object

        img_paths       : `dictionary`
            Paths to images: key - filter name; value - path

        outdir          : `string`
            Path, where the output should be stored.

        sigma_psf       : `float`
            Sigma of the objects PSF, assuming it is a Gaussian

        ra_obj          : `float`
            Right ascension of the object

        dec_obj         : `float`
            Declination of the object

        ncores          : `integer`, optional
            Number of cores to use for multi core processing
            Default is ``6``.

        wcs_method      : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        force_wcs_determ    : `boolean`, optional
            If ``True`` a new WCS determination will be calculated even if
            a WCS is already present in the FITS Header.
            Default is ``False``.

        sigma_bkg       : `float`, optional
            Sigma used for the sigma clipping of the background
            Default is ``5.``.

        multi_start     : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``7``.

        size_epsf       : `integer`, optional
            Size of the extraction region in pixel
            Default is `25``.

        frac_epsf_stars : `float`, optional
            Fraction of all stars that should be used to calculate the ePSF
            Default is ``0.2``.

        oversampling    : `integer`, optional
            ePSF oversampling factor
            Default is ``2``.

        maxiters        : `integer`, optional
            Number of ePSF iterations
            Default is ``7``.

        method         : `string`, optional
            Finder method DAO or IRAF
            Default is ``IRAF``.

        multi           : `float`, optional
            Multiplier for the background RMS, used to calculate the
            threshold to identify stars
            Default is ``5.0``.

        multi_grouper   : `float`, optional
            Multiplier for the DAO grouper
            Default is ``5.0``.

        strict_cleaning : `boolean`, optional
            If True objects with negative flux uncertainties will be removed
            Default is ``True``.

        min_eps_stars   : `integer`, optional
            Minimal number of required ePSF stars
            Default is ``25``.

        photometry      : `string`, optional
            Switch between aperture and ePSF photometry.
            Possibilities: 'PSF' & 'APER'
            Default is ``PSF``.

        rstars          : `float`, optional
            Radius of the stellar aperture
            Default is ``5``.

        rbg_in          : `float`, optional
            Inner radius of the background annulus
            Default is ``7``.

        rbg_out         : `float`, optional
            Outer radius of the background annulus
            Default is ``10``.

        r_unit          : `string`, optional
            Unit of the radii above. Allowed are ``pixel`` and ``arcsec``.
            Default is ``pixel``.

        strict_eps      ; `boolean`, optional
            If True a stringent test of the ePSF conditions is applied.
            Default is ``True``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        ref_ID              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose         : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        search_image    : `boolean`, optional
            If `True` the objects on the image will be identified. If `False`
            it is assumed that object identification was performed in advance.
            Default is ``True``.

        plot_ifi        : `boolean`, optional
            If True star map plots for all stars are created
            Default is ``False``.

        plot_test       : `boolean`, optional
            If True a star map plots only for the reference image [refid] is
            created
            Default is ``True``.
    '''
    ###
    #   Check output directories
    #
    checks.check_out(outdir, os.path.join(outdir,'tables'))


    ###
    #   Check image directories
    #
    checks.check_dir(img_paths)


    #   Outer loop over all filter
    for filt in filter_list:
        terminal_output.print_terminal(
            filt,
            string="Analyzing {:s} images",
            style_name='HEADER',
            )


        #   Initialize image ensemble object
        img_container.ensembles[filt] = image_ensemble(
            filt,
            name,
            img_paths[filt],
            outdir,
            ref_ID,
            )


        ###
        #   Find the WCS solution for the image
        #
        aux.find_wcs(
            img_container.ensembles[filt],
            ref_id=ref_ID,
            method=wcs_method,
            force_wcs_determ=force_wcs_determ,
            indent=3,
            )


        ###
        #   Main extraction of object positions and object fluxes
        #   using multiprocessing
        #
        extract_multiprocessing(
            img_container.ensembles[filt],
            ncores,
            sigma_psf,
            sigma_bkg=sigma_bkg,
            multi_start=multi_start,
            size_epsf=size_epsf,
            frac_epsf_stars=frac_epsf_stars,
            oversampling=oversampling,
            maxiters=maxiters,
            method=method,
            multi=multi,
            multi_grouper=multi_grouper,
            strict_cleaning=strict_cleaning,
            min_eps_stars=min_eps_stars,
            strict_eps=strict_eps,
            photometry=photometry,
            rstars=rstars,
            rbg_in=rbg_in,
            rbg_out=rbg_out,
            r_unit=r_unit,
            search_image=search_image,
            plot_ifi=plot_ifi,
            plot_test=plot_test,
            )


        if search_image:
            ###
            #   Correlate results from all images, while preserving
            #   the variable star
            #
            correlate_preserve_variable(
                img_container.ensembles[filt],
                ra_obj,
                dec_obj,
                dcr=dcr,
                option=option,
                maxid=maxid,
                ref_ID=ref_ID,
                nmissed=nmissed,
                bfrac=bfrac,
                s_refOBJ=s_refOBJ,
                verbose=verbose,
                plot_test=plot_test,
                correl_method=correl_method,
                seplimit=seplimit,
                )

        else:
            img_ensemble = img_container.ensembles[filt]

            ###
            #   Find position of the variable star
            #
            terminal_output.print_terminal(
                indent=1,
                string="Identify the variable star",
                )

            if correl_method == 'astropy':
                variable_ID, count, x_obj, y_obj = correlate.posi_obj_astropy_img(
                    img_ensemble.image_list[ref_ID],
                    ra_obj,
                    dec_obj,
                    img_ensemble.wcs,
                    )

            elif correl_method == 'own':
                inds_obj, count, x_obj, y_obj  = correlate.posi_obj_srcor_img(
                    img_ensemble.image_list[ref_ID],
                    ra_obj,
                    dec_obj,
                    img_ensemble.wcs,
                    dcr=dcr,
                    option=option,
                    verbose=verbose,
                    )

                #   Current object ID
                variable_ID  = inds_obj[1]

                if verbose:
                    terminal_output.print_terminal()


            ###
            #   Check if variable star was detected
            #
            if count == 0:
                raise RuntimeError(
                    f"{style.bcolors.FAIL} \tERROR: The variable object was "
                    f"not detected in the reference image.\n"
                    f"\t-> EXIT {style.bcolors.ENDC}"
                    )

            #   Add ID of the variable star to the image ensemble
            img_ensemble.variable_ID = variable_ID


            ###
            #
            #

            #   Get dictionary with astropy tables with the position and flux data
            result_tbl = img_ensemble.get_photometry()

            x_s = result_tbl[str(ref_ID)]['x_fit']
            y_s = result_tbl[str(ref_ID)]['y_fit']

            count = len(x_s)
            id_s = np.arange(count)

            #   Get image IDs
            arr_img_IDs = img_ensemble.get_image_ids()

            #   Number of images
            nimg = len(arr_img_IDs)

            #   Prepare array for the flux and uncertainty (all datasets)
            flux_arr = np.zeros(nimg, dtype=[('flux_fit', 'f8', (count)),
                                             ('flux_unc', 'f8', (count)),
                                            ]
                            )

            #   Fill flux arrays
            for j, img_ID in enumerate(arr_img_IDs):
                img_ID_str = str(img_ID)

                #   Flux and uncertainty array for individual images
                flux_img = np.zeros(
                    count,
                    dtype=[('flux_fit', 'f8'), ('flux_unc', 'f8')],
                    )

                #   Rearrange flux and error
                flux_img['flux_fit'] = result_tbl[img_ID_str]['flux_fit']
                flux_img['flux_unc'] = result_tbl[img_ID_str]['flux_unc']

                #   Remove nans etc. in error
                flux_img['flux_unc'] = np.nan_to_num(
                    flux_img['flux_unc'],
                    nan=9999.,
                    posinf=9999.,
                    neginf=9999.,
                    )

                #   Remove '--' entries in error
                flux_err_dash = np.argwhere(flux_img['flux_unc'] == '--')
                flux_img['flux_unc'][flux_err_dash] = 9999.

                uflux_img = unumpy.uarray(
                    flux_img['flux_fit'],
                    flux_img['flux_unc']
                    )

                #   Add sorted flux data and positions back to the image
                img_ensemble.image_list[img_ID].flux    = flux_img
                img_ensemble.image_list[img_ID].uflux   = uflux_img
                img_ensemble.image_list[img_ID].x_sort  = x_s
                img_ensemble.image_list[img_ID].y_sort  = y_s
                img_ensemble.image_list[img_ID].id_sort = id_s


                #   Add to overall array
                flux_arr['flux_fit'][j] = flux_img['flux_fit']
                flux_arr['flux_unc'][j] = flux_img['flux_unc']

            uflux_arr = unumpy.uarray(
                    flux_arr['flux_fit'],
                    flux_arr['flux_unc']
                    )

            #   Update image ensemble object and add IDs, pixel coordinates, and
            #   flux of the correlated objects
            img_ensemble.id_s = id_s
            img_ensemble.x_s  = x_s
            img_ensemble.y_s  = y_s
            img_ensemble.flux = flux_arr
            img_ensemble.uflux = uflux_arr



            ###
            #   Plot image with the final positions overlaid (final version)
            #
            aux.prepare_and_plot_starmap_final_3(
                img_ensemble,
                [x_obj],
                [y_obj],
                plot_test=plot_test,
                )


def correlate_calibrate(img_container, filter_list, dcr=3, option=1,
                        ref_img=0, calib_method='APASS', vizier_dict={},
                        calib_file=None, ID=None, ra_unit=u.deg,
                        dec_unit=u.deg, mag_range=(0.,18.5), Tcs=None,
                        derive_Tcs=False, plot_sigma=False, photo_type='',
                        region=False, radius=600, data_cluster=False,
                        pm_median=False, max_distance_cluster=6.,
                        find_cluster_para_set=1, correl_method='astropy',
                        seplimit=2.*u.arcsec):
    '''
        Correlate photometric extraction results from 2 images and calibrate
        the magnitudes.

        Parameters
        ----------
        img_container           : `image.container`
            Container object with image ensemble objects for each filter

        filter_list             : `list` of `strings`
            List with filter names

        dcr                     : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option                  : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        ref_img                 : `integer`, optional
            Reference image ID
            Default is ``0``.

        calib_method           : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict             : `dictionary`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``{}``.

        calib_file              : `string`, optional
            Path to the calibration file
            Default is ``None``.

        ID                      : `integer`, optional
            ID of the object
            Default is ``None``.

        ra_unit                 : `astropy.unit`, optional
            Right ascension unit
            Default is ``u.deg``.

        dec_unit                : `astropy.unit`, optional
            Declination unit
            Default is ``u.deg``.

        mag_range               : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        Tcs             : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_Tcs      : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the data base.
            Default is ``False``

        plot_sigma      : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

        photo_type      : `string`, optional
            Applied extraction method. Posibilities: ePSF or APER`
            Default is ``''``.

        region          : `boolean`, optional
            If True the extracted objects will be filtered such that only
            objects with ``radius`` will be returned.
            Default is ``False``.

        radius          : `float`, optional
            Radius around the object in arcsec.
            Default is ``600``.

        data_cluster    : `boolean`, optional
            If True cluster in the Gaia distance and proper motion data
            will be identified.
            Default is ``False``.

        pm_median       : `boolean`, optional
            If True only the objects that are close to the median
            proper motion will be returned.
            Default is ``False``.

        max_distance_cluster    : `float`, optional
            Expected maximal distance of the cluster in kpc. Used to
            restrict the parameter space to facilitate an easy
            identification of the star cluster.
            Default is ``6``.

        find_cluster_para_set   : `integer`, optional
            Parameter set used to identify the star cluster in proper
            motion and distance data.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.
    '''
    ###
    #   Correlate the stellar positions from the different filter
    #
    correlate_ensemble(
        img_container,
        filter_list,
        dcr=dcr,
        option=option,
        correl_method=correl_method,
        seplimit=seplimit,
        )


    ###
    #   Plot image with the final positions overlaid (final version)
    #
    if len(filter_list) > 1:
        aux.prepare_and_plot_starmap_final(
            img_container,
            filter_list,
            )


    ###
    #   Calibrate the magnitudes
    #
    #   Load calibration information
    calib.deter_calib(
        img_container,
        filter_list,
        calib_method=calib_method,
        dcr=dcr,
        option=option,
        vizier_dict=vizier_dict,
        calib_file=calib_file,
        mag_range=mag_range,
        )

    #   Apply calibration and perform magnitude transformation
    trans.apply_calib(
        img_container,
        filter_list,
        Tcs=Tcs,
        derive_Tcs=derive_Tcs,
        plot_sigma=plot_sigma,
        photo_type=photo_type,
        refid=ref_img,
        )


    ###
    #   Restrict results to specific areas of the image and filter by means
    #   of proper motion and distance using Gaia
    aux.postprocess_results(
        img_container,
        filter_list,
        photo_type=photo_type,
        region=region,
        radius=radius,
        data_cluster=data_cluster,
        pm_median=pm_median,
        max_distance_cluster=max_distance_cluster,
        find_cluster_para_set=find_cluster_para_set,
        )


    ###
    #   Determine limiting magnitudes
    #
    aux.derive_limiting_mag(img_container, filter_list, ref_img)


def calibrate_data_mk_lc(img_container, filter_list, ra_obj, dec_obj, nameobj,
                         outdir, transit_time, period, valid_calibs=None,
                         binn=None, Tcs=None, derive_Tcs=False, ref_ID=0,
                         calib_method='APASS', vizier_dict={}, calib_file=None,
                         mag_range=(0.,18.5), dcr=3., option=1, maxid=1,
                         nmissed=1, bfrac=1.0, s_refOBJ=True, photo_type='',
                         correl_method='astropy', seplimit=2.*u.arcsec,
                         verbose=False, plot_test=True, plot_ifi=False,
                         plot_sigma=False):
    '''
        Calculate magnitudes, calibrate, and plot light curves

        Parameters
        ----------
        img_container       : `image.container`
            Container object with image ensemble objects for each filter

        filt_list           : `list` of `strings`
            List with filter names

        ra_obj              : `float`
            Right ascension of the object

        dec_obj             : `float`
            Declination of the object

        nameobj             : `string`
            Name of the object

        outdir              : `string`
            Path, where the output should be stored.

        transit_time        : `float`
            Date and time of the transit.
            Format: "yyyy:mm:ddThh:mm:ss" e.g., "2020-09-18T01:00:00"

        period              : `float`
            Period in [d]

        valid_calibs        : `list` of 'list` of `string` or None, optional
            Valid filter combinations to calculate magnitude transformation
            Default is ``None``.

        binn                : `float`, optional
            Binning factor for the light curve.
            Default is ``None```.

        Tcs                 : `dictionary`, optional
            Calibration coefficients for the magnitude transformation
            Default is ``None``.

        derive_Tcs      : `boolean`, optional
            If True the magnitude transformation coefficients will be
            calculated from the current data even if calibration coefficients
            are available in the data base.
            Default is ``False``

        ref_ID              : `integer`, optional
            ID of the reference origin
            Default is ``0``.

        calib_method       : `string`, optional
            Calibration method
            Default is ``APASS``.

        vizier_dict         : `dictionary`, optional
            Dictionary with identifiers of the Vizier catalogs with valid
            calibration data
            Default is ``{}``.

        calib_file          : `string`, optional
            Path to the calibration file
            Default is ``None``.

        mag_range           : `tupel` or `float`, optional
            Magnitude range
            Default is ``(0.,18.5)``.

        dcr                 : `float`, optional
            Maximal distance between two objects in Pixel
            Default is ``3``.

        option              : `integer`, optional
            Option for the srcor correlation function
            Default is ``1``.

        maxid               : `integer`, optional
            Max. number of allowed identical cross identifications between
            objects from a specific origin
            Default is ``1``.

        nmissed             : `integer`, optional
            Maximum number an object is allowed to be not detected in an
            origin. If this limit is reached the object will be removed.
            Default is ``i`.

        bfrac               : `float`, optional
            Fraction of low quality source position origins, i.e., those
            origins, for which it is expected to find a reduced number of
            objects with valid source positions.
            Default is ``1.0``.

        s_refOBJ            : `boolean`, optional
            If ``False`` also reference objects will be rejected, if they do
            not fulfill all criteria.
            Default is ``True``.

        photo_type          : `string`, optional
            Applied extraction method. Posibilities: ePSF or APER`
            Default is ``''``.

        correl_method       : `string`, optional
            Correlation method to be used to find the common objects on
            the images.
            Possibilities: ``astropy``, ``own``
            Default is ``astropy``.

        seplimit            : `astropy.units`, optional
            Allowed separation between objects.
            Default is ``2.*u.arcsec``.

        verbose             : `boolean`, optional
            If True additional output will be printed to the command line.
            Default is ``False``.

        plot_test           : `boolean`, optional
            If True only the star map for the reference image will be
            plotted.
            Default is ``True``.

        plot_ifi            : `boolean`, optional
            If True star map plots for all stars will be created.
            Default is ``False``.

        plot_sigma          : `boolean', optional
            If True sigma clipped magnitudes will be plotted.
            Default is ``False``.

    '''
    if valid_calibs is None:
        valid_calibs = calibration_data.valid_calibs

    for filt in filter_list:
        terminal_output.print_terminal(
            filt,
            string="Working on filter: {:s}",
            style_name='OKBLUE',
            )

        ###
        #   Try magnitude transformation
        #
        success = False
        #   Loop over valid filter combination for the transformation
        for calib_fil in valid_calibs:
            if filt in calib_fil:
                #   Check if filter combination is valid
                if calib_fil[0] in filter_list and calib_fil[1] in filter_list:

                    if filt == calib_fil[0]:
                        i = 0
                    else:
                        i = 1

                    #   Get object ID
                    objID = img_container.ensembles[filt].variable_ID

                    ###
                    #   Correlate star positions from the different filter
                    #
                    correlate_ensemble(
                        img_container,
                        calib_fil,
                        dcr=dcr,
                        option=option,
                        maxid=maxid,
                        refOBJ=[objID],
                        nmissed=nmissed,
                        bfrac=bfrac,
                        s_refOBJ=s_refOBJ,
                        correl_method=correl_method,
                        seplimit=seplimit,
                        )


                    ###
                    #   Reidentify position of the variable star
                    #
                    terminal_output.print_terminal(
                        string="Identify the variable star",
                        )

                    if correl_method == 'astropy':
                        objID, count, _, _ = correlate.posi_obj_astropy(
                            img_container.ensembles[filt].x_es,
                            img_container.ensembles[filt].y_es,
                            ra_obj,
                            dec_obj,
                            img_container.ensembles[filt].wcs,
                            seplimit=seplimit,
                            )

                    elif correl_method == 'own':
                        inds_obj, count, _, _  = correlate.posi_obj_srcor(
                            img_container.ensembles[filt].x_es,
                            img_container.ensembles[filt].y_es,
                            ra_obj,
                            dec_obj,
                            img_container.ensembles[filt].wcs,
                            dcr=dcr,
                            option=option,
                            verbose=verbose,
                            )
                        if verbose:
                            terminal_output.print_terminal()

                        #   Current object ID
                        objID  = inds_obj[1]

                    #   Set new object ID
                    img_container.ensembles[filt].variable_ID = objID


                    #   Check if variable star was detected
                    if count == 0:
                        raise RuntimeError(
                            f"{style.bcolors.FAIL} \tERROR: The variable "
                            "star was not detected in the reference image.\n"
                            f"\t-> EXIT {style.bcolors.ENDC}"
                            )


                    ###
                    #   Load calibration information
                    #
                    calib.deter_calib(
                        img_container,
                        calib_fil,
                        calib_method=calib_method,
                        dcr=dcr,
                        option=option,
                        vizier_dict=vizier_dict,
                        calib_file=calib_file,
                        mag_range=mag_range,
                        correl_method=correl_method,
                        seplimit=seplimit,
                        )
                    terminal_output.print_terminal()

                    #   Stop here if calibration data is not available
                    filter_calib = img_container.calib_parameters.column_names
                    if ('mag'+calib_fil[0] not in filter_calib or
                        'mag'+calib_fil[1] not in filter_calib):
                            if 'mag'+calib_fil[0] not in filter_calib:
                                err_filter = calib_fil[0]
                            if 'mag'+calib_fil[1] not in filter_calib:
                                err_filter = calib_fil[1]
                            terminal_output.print_terminal(
                                err_filter,
                                indent=2,
                                string="Magnitude transformation not "\
                                "possible because no calibration data "\
                                "available for filter {}",
                                style_name='WARNING',
                                )
                            break


                    ###
                    #   Calibrate magnitudes
                    #

                    #   Set boolean regarding magnitude plot
                    plot_mags = True if plot_test or plot_ifi else False

                    #   Apply calibration and perform magnitude
                    #   transformation
                    trans.apply_calib(
                        img_container,
                        calib_fil,
                        Tcs=Tcs,
                        derive_Tcs=derive_Tcs,
                        plot_mags=plot_mags,
                        ID=filt,
                        photo_type=photo_type,
                        refid=ref_ID,
                        plot_sigma=plot_sigma,
                        )
                    cali_mags = getattr(img_container, 'cali', None)
                    if not checks.check_unumpy_array(cali_mags):
                        cali = cali_mags['mag']
                    else:
                        cali = cali_mags
                    if np.all(cali == 0):
                        break


                    ###
                    #   Plot light curve
                    #
                    #   Create a Time object for the observation times
                    otime = Time(
                        img_container.ensembles[filt].get_obs_time(),
                        format='jd',
                        )

                    #   Create mask for time series to remove images
                    #   without entries
                    #mask_ts = np.isin(
                        ##cali_mags['med'][i][:,objID],
                        #cali_mags[i][:,objID],
                        #[0.],
                        #invert=True
                        #)

                    #   Create a time series object
                    ts = aux.mk_ts(
                        otime,
                        cali_mags[i],
                        filt,
                        objID,
                        )

                    #   Write time series
                    ts.write(
                        outdir+'/tables/light_curce_'+filt+'.dat',
                        format='ascii',
                        overwrite=True,
                        )
                    ts.write(
                        outdir+'/tables/light_curce_'+filt+'.csv',
                        format='ascii.csv',
                        overwrite=True,
                        )

                    #   Plot light curve over JD
                    plot.light_curve_jd(
                        ts,
                        filt,
                        filt+'_err',
                        outdir,
                        nameobj=nameobj
                        )

                    #   Plot the light curve folded on the period
                    plot.light_curve_fold(
                        ts,
                        filt,
                        filt+'_err',
                        outdir,
                        transit_time,
                        period,
                        binn=binn,
                        nameobj=nameobj,
                        )

                    success = True
                    break


        if not success:
            #   Load calibration information
            calib.deter_calib(
                img_container,
                [filt],
                calib_method=calib_method,
                dcr=dcr,
                option=option,
                vizier_dict=vizier_dict,
                calib_file=calib_file,
                mag_range=mag_range,
                )

            #   Check if calibration data is available
            filter_calib = img_container.calib_parameters.column_names
            if 'mag'+filt not in filter_calib:
                terminal_output.print_terminal(
                    filt,
                    indent=2,
                    string="Magnitude calibration not "\
                    "possible because no calibration data is "\
                    "available for filter {}. Use normalized flux for light "\
                    "curve.",
                    style_name='WARNING',
                    )

                #   Get ensemble
                ensemble = img_container.ensembles[filt]

                #   Quasi calibration of the flux data
                trans.flux_calibrate_ensemble(ensemble)

                #   Normalize data if no calibration magnitudes are available
                trans.flux_normalize_ensemble(ensemble)

                plot_quantity = ensemble.uflux_norm
            else:
                #   Set boolean regarding magnitude plot
                plot_mags = True if plot_test or plot_ifi else False

                #   Apply calibration
                trans.apply_calib(
                    img_container,
                    [filt],
                    plot_mags=plot_mags,
                    photo_type=photo_type,
                    )
                plot_quantity = getattr(img_container, 'noT', None)[0]


            ###
            #   Plot light curve
            #
            #   Create a Time object for the observation times
            otime = Time(
                img_container.ensembles[filt].get_obs_time(),
                format='jd',
                )

            #   Create a time series object
            ts = aux.mk_ts(
                otime,
                plot_quantity,
                filt,
                img_container.ensembles[filt].variable_ID,
                )

            #   Write time series
            ts.write(
                outdir+'/tables/light_curce_'+filt+'.dat',
                format='ascii',
                overwrite=True,
                )
            ts.write(
                outdir+'/tables/light_curce_'+filt+'.csv',
                format='ascii.csv',
                overwrite=True,
                )

            #   Plot light curve over JD
            plot.light_curve_jd(
                ts,
                filt,
                filt+'_err',
                outdir,
                nameobj=nameobj)


            #   Plot the light curve folded on the period
            plot.light_curve_fold(
                ts,
                filt,
                filt+'_err',
                outdir,
                transit_time,
                period,
                binn=binn,
                nameobj=nameobj,
                )


def subtract_archive_img_from_img(filt, name, path, outdir,
                                  wcs_method='astrometry', plot_comp=True,
                                  hips_source='CDS/P/DSS2/blue'):
    '''
        Subtraction of a reference/archival image from the input image.
        The installation of Hotpants is required.

        Parameters
        ----------
        filt            : `string`
            Filter identifier

        name            : `string`
            Name of the object

        path            : `dictionary`
            Paths to images: key - filter name; value - path

        outdir          : `string`
            Path, where the output should be stored.

        wcs_method      : `string`, optional
            Method that should be used to determine the WCS.
            Default is ``'astrometry'``.

        hips_source     : `string`
            ID string of the image catalog that will be queried using the
            hips service.
            Default is ``CDS/P/DSS2/blue``.
    '''
    ###
    #   Check output directories
    #
    checks.check_out(
        outdir,
        os.path.join(outdir,'subtract'),
        )
    outdir = os.path.join(outdir,'subtract')


    ###
    #   Check input path
    #
    checks.check_file(path)


    ###
    #   Trim image as needed (currently images with < 4*10^6 are required)
    #
    #   Load image
    img = CCDData.read(path)

    #   Trim
    xtrim = 2501
    #xtrim = 2502
    ytrim = 1599
    img = ccdp.trim_image(img[0:ytrim, 0:xtrim])
    img.meta['NAXIS1'] = xtrim
    img.meta['NAXIS2'] = ytrim

    #   Save trimmed file
    basename  = base_aux.get_basename(path)
    file_name = basename+'_trimmed.fit'
    file_path = os.path.join(outdir, file_name)
    img.write(file_path, overwrite=True)


    ###
    #   Initialize image ensemble object
    #
    img_ensemble = image_ensemble(
        filt,
        name,
        #file_path,
        path,
        outdir,
        0,
        )


    ###
    #   Find the WCS solution for the image
    #
    aux.find_wcs(
        img_ensemble,
        ref_id=0,
        method=wcs_method,
        indent=3,
        )


    ###
    #   Get image via hips2fits
    #
    #from astropy.utils import data
    #data.Conf.remote_timeout=600
    hipsInstance = hips2fitsClass()
    hipsInstance.timeout = 120000
    #hipsInstance.timeout = 1200000000
    #hipsInstance.timeout = (200000000, 200000000)
    hipsInstance.server = "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits"
    print(hipsInstance.timeout)
    print(hipsInstance.server)
    #hips_hdus = hips2fits.query_with_wcs(
    hips_hdus = hipsInstance.query_with_wcs(
        hips=hips_source,
        wcs=img_ensemble.wcs,
        get_query_payload=False,
        format='fits',
        verbose=True,
        )
    #   Save downloaded file
    hips_hdus.writeto(os.path.join(outdir, 'hips.fits'), overwrite=True)


    ###
    #   Plot original and reference image
    #
    if plot_comp:
        plot.comp_img(
            outdir,
            img_ensemble.image_list[0].get_data(),
            hips_hdus[0].data,
            )


    ###
    #   Perform image subtraction
    #
    #   Get image and image data
    img = img_ensemble.image_list[0].read_image()
    hips_data = hips_hdus[0].data.astype('float64').byteswap().newbyteorder()

    #   Run hotpants
    subtraction.run_hotpants(
        img.data,
        hips_data,
        img.mask,
        np.zeros(hips_data.shape, dtype=bool),
        image_gain=1.,
        #template_gain=1,
        template_gain=None,
        err=img.uncertainty.array,
        #err=True,
        template_err=True,
        #verbose=True,
        _workdir=outdir,
        #_exe=exe_path,
        )
