############################################################################
####                            Libraries                               ####
############################################################################

import numpy as np

from scipy import stats

from matplotlib import pyplot as plt

from astropy.visualization import hist, simple_norm

from .. import checks

############################################################################
####                        Routines & definitions                      ####
############################################################################

def debug_plot_cc_matrix(img, cc):
    '''
        Debug plot showing the cc matrix, created during image correlation

        Parameters
        ----------
        img         : `numpy.ndarray`
            Image data array

        cc          : `numpy.ndarray`
            Array with the data of the cc matrix
    '''
    #   Norm of image
    norm = simple_norm(img, 'log', percent=99.)

    #   Initialize sub plots
    plt.subplot(121)

    #   Plot image
    plt.imshow(img, norm=norm, cmap = 'gray')

    #   Set title & ticks
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    #   Norm of cc matrix
    norm = simple_norm(np.absolute(cc), 'log', percent=99.)

    #   Plot cc matrix
    plt.subplot(122),plt.imshow(np.absolute(cc), norm=norm, cmap = 'gray')

    #   Set title & ticks
    plt.title('cc'), plt.xticks([]), plt.yticks([])
    plt.show()


def plot_dark_with_distributions(img, rn, dark_rate, outdir, n_images=1,
                                 exposure=1, gain=1, show_poisson=True,
                                 show_gaussian=True):
    """
        Plot the distribution of dark pixel values, optionally overplotting the
        expected Poisson and normal distributions corresponding to dark current
        only or read noise only.

        Parameters
        ----------
        img             : numpy array
            Dark frame to histogram

        rn              : float
            The read noise, in electrons

        dark_rate       : float
            The dark current in electrons/sec/pixel

        outdir          : pathlib.Path
            Path pointing to the main storage location

        n_images        : float, optional
            If the image is formed from the average of some number of dark
            frames then the resulting Poisson distribution depends on the
            number of images, as does the expected standard deviation of the
            Gaussian.

        exposure        : float, optional
            Exposure time, in seconds

        gain            : float, optional
            Gain of the camera, in electron/ADU

        show_poisson    : bool, optional
            If ``True``, overplot a Poisson distribution with mean equal to the
            expected dark counts for the number of images

        show_gaussian   : bool, optional
            If ``True``, overplot a normal distribution with mean equal to the
            expected dark counts and standard deviation equal to the read
            noise, scaled as appropriate for the number of images
    """
    #   Check output directories
    checks.check_out(
        outdir,
        outdir / 'reduce_plots',
        )

    #   Scale image
    img =  img * gain / exposure

    #   Use bmh style
    #plt.style.use('bmh')

    #   Set layout of image
    fig = plt.figure(figsize=(20,9))

    #   Get
    h = plt.hist(
        img.flatten(),
        bins=20,
        align='mid',
        density=True,
        label="Dark frame",
        )

    bins = h[1]

    #   Dark rate is dictionary of different values for different
    #   temperatures
    #   Get value for the entry closes to the exposure time
    dr_temp = list(dark_rate.keys())
    delta_temp = np.array(dr_temp) - exposure
    id_min = np.argmin(delta_temp)

    #   Expected mean of the dark
    expected_mean_dark = list(dark_rate.values())[id_min] * exposure / gain

    #   Plot Poisson
    if show_poisson:
        #   Account for number of exposures
        pois = stats.poisson(expected_mean_dark * n_images)

        #   X range
        pois_x = np.arange(0, 300, 1)

        #   Prepare normalization
        new_area = np.sum(1/n_images * pois.pmf(pois_x))

        plt.plot(
            pois_x / n_images, pois.pmf(pois_x) / new_area,
            label="Poisson dsitribution, mean of {:5.2f} counts"\
                .format(expected_mean_dark),
            )

    #   Plot Gaussian
    if show_gaussian:
        #   The expected width of the Gaussian depends on the number of images
        expected_scale = rn / gain * np.sqrt(n_images)

        #   Mean value is same as for the Poisson distribution (account for
        #   number of images)
        expected_mean = expected_mean_dark * n_images

        #
        gauss = stats.norm(loc=expected_mean, scale=expected_scale)

        #   X range
        gauss_x = np.linspace(
            expected_mean - 5 * expected_scale,
            expected_mean + 5 * expected_scale,
            num=100,
            )

        plt.plot(
            gauss_x / n_images,
            gauss.pdf(gauss_x) * n_images,
            label='Gaussian, standard dev is read noise in counts',
            )

    #   Labels
    plt.xlabel("Dark counts in {} sec exposure".format(exposure))
    plt.ylabel("Fraction of pixels (area normalized to 1)")
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_with_distributions_{}.pdf'.format(
        str(exposure).replace("''", "p")
        )
    plt.savefig(
        outdir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def plot_hist(img, outdir, gain, exposure):
    '''
        Plot image histogram for dark images

        Parameters
        ----------
        img         : numpy array
            Dark frame to histogram

        outdir          : pathlib.Path
            Path pointing to the main storage location

        gain        : float
            Gain of the camera, in electron/ADU

        exposure    : float
            Exposure time, in seconds
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        outdir / 'reduce_plots',
        )

    #   Scale image
    img =  img * gain / exposure

    #   Use bmh style
    #plt.style.use('bmh')

    #   Set layout of image
    fig = plt.figure(figsize=(20,9))

    #   Create histogram
    hist(
        img.flatten(),
        bins=5000,
        density=False,
        label=str(exposure)+' sec dark',
        alpha=0.4,
        )

    #   Labels
    plt.xlabel('Dark current, $e^-$/sec')
    plt.ylabel('Number of pixels')
    plt.loglog()
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_hist_{}.pdf'.format(str(exposure).replace("''", "p"))
    plt.savefig(
        outdir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def plot_flat_median(ifc, image_type, outdir, filt):
    '''
        Plot median and mean of each flat field in a file collection

        Parameters
        ----------
        ifc             : ccdproc.ImageFileCollection
            File collection with the flat fields to analyze

        image_type      : string
            Header keyword characterizing the flats

        outdir          : pathlib.Path
            Path pointing to the main storage location

        filt            : string
            Filter

        Idea/Reference
        --------------
            https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        outdir / 'reduce_plots',
        )

    #   Calculate median and mean for each image
    median_count = []
    mean_count   = []
    for data in ifc.data(imagetyp=image_type, filter=filt):
        median_count.append(np.median(data))
        mean_count.append(np.mean(data))

    #   Use bmh style
    #plt.style.use('bmh')

    #   Set layout of image
    fig = plt.figure(figsize=(20,9))

    #   Plot mean & median
    plt.plot(median_count, label='median')
    plt.plot(mean_count, label='mean')

    #   Plot labels
    plt.xlabel('Image number')
    plt.ylabel('Count (ADU)')
    plt.title('Pixel value in calibrated flat frames')
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'flat_median_{}.pdf'.format(filt.replace("''", "p"))
    plt.savefig(
        outdir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def subplots_stars_fwhm_estimate(outdir, nstars, stars, filt, basename):
    '''
        Plots cutouts around the stars used to estimate the FWHM

        Parameters
        ----------
        outdir          : `string`
            Path to the directory where the master files should be saved to

        nstars          : `integer`
            Number of stars

        stars           : `photutils.psf.EPSFStars object
            Sub images (squares) extracted around the FWHM stars

        filt            : `string`
            Filter name

        basename        : `string`
            Name of the image file
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        outdir / 'cutouts',
        )

    #   Set number of rows and columns for the plot
    nrows = 5
    ncols = 5

    #   Prepare plot
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20, 20),
        squeeze=True,
    )
    ax = ax.ravel()

    #   Set title of the complete plot
    fig.suptitle(
        'Cutouts of the FWHM stars ('+str(filt)+', '+str(basename)+')',
        fontsize=20,
        )

    #   Loop over the cutouts (default: 25)
    for i in range(nstars):
        # Set up normalization for the image
        norm = simple_norm(stars[i], 'log', percent=99.)

        # Plot individual cutouts
        ax[i].set_xlabel("Pixel")
        ax[i].set_ylabel("Pixel")
        ax[i].imshow(
            stars[i],
            norm=norm,
            origin='lower',
            cmap='viridis',
        )

    #   Write the plot to disk
    plt.savefig(
        str(outdir)+'/cutouts/cutouts_FWHM-stars_'+str(filt)+'_'+
        str(basename)+'.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()
