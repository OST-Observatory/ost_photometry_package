############################################################################
#                               Libraries                                  #
############################################################################

import numpy as np

from scipy import stats

from matplotlib import pyplot as plt

from astropy.visualization import hist, simple_norm

from .. import checks


############################################################################
#                           Routines & definitions                         #
############################################################################


def cross_correlation_matrix(image_data, cross_correlation_data):
    """
        Debug plot showing the cc matrix, created during image correlation

        Parameters
        ----------
        image_data              : `numpy.ndarray`
            Image data array

        cross_correlation_data  : `numpy.ndarray`
            Array with the data of the cc matrix
    """
    #   Norm of image
    norm = simple_norm(image_data, 'log', percent=99.)

    #   Initialize sub plots
    plt.subplot(121)

    #   Plot image
    plt.imshow(image_data, norm=norm, cmap='gray')

    #   Set title & ticks
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    #   Norm of cc matrix
    norm = simple_norm(
        np.absolute(cross_correlation_data),
        'log',
        percent=99.,
    )

    #   Plot cc matrix
    plt.subplot(122), plt.imshow(
        np.absolute(cross_correlation_data),
        norm=norm,
        cmap='gray',
    )

    #   Set title & ticks
    plt.title('cc'), plt.xticks([]), plt.yticks([])
    plt.show()


def plot_dark_with_distributions(image_data, read_noise, dark_current,
                                 output_dir, n_images=1, exposure_time=1,
                                 gain=1, show_poisson_distribution=True,
                                 show_gaussian_distribution=True):
    """
        Plot the distribution of dark pixel values, optionally over-plotting
        the expected Poisson and normal distributions corresponding to dark
        current only or read noise only.

        Parameters
        ----------
        image_data             : `numpy.ndarray`
            Image data

        read_noise              : `float`
            The read noise, in electrons

        dark_current       : `float`
            The dark current in electrons/sec/pixel

        output_dir          : `pathlib.Path`
            Path pointing to the main storage location

        n_images        : `float`, optional
            If the image is formed from the average of some number of dark
            frames then the resulting Poisson distribution depends on the
            number of images, as does the expected standard deviation of the
            Gaussian.

        exposure_time        : `float`, optional
            Exposure time, in seconds

        gain            : `float`, optional
            Gain of the camera, in electron/ADU

        show_poisson_distribution    : `bool`, optional
            If ``True``, over plot a Poisson distribution with mean equal to
            the expected dark counts for the number of images

        show_gaussian_distribution   : `bool`, optional
            If ``True``, over plot a normal distribution with mean equal to the
            expected dark counts and standard deviation equal to the read
            noise, scaled as appropriate for the number of images
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Scale image
    image_data = image_data * gain / exposure_time

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

    #   Get
    h = plt.hist(
        image_data.flatten(),
        bins=20,
        align='mid',
        density=True,
        label="Dark frame",
    )

    #   TODO: Check plot - bins and histogram not used
    bins = h[1]

    #   Expected mean of the dark
    expected_mean_dark = dark_current * exposure_time / gain

    #   Plot Poisson
    if show_poisson_distribution:
        #   Account for number of exposures
        poisson_distribution = stats.poisson(expected_mean_dark * n_images)

        #   X range
        x_axis_poisson = np.arange(0, 300, 1)

        #   Prepare normalization
        #   TODO: Check if this is correct
        new_area = np.sum(
            1 / n_images * poisson_distribution.pmf(x_axis_poisson)
        )

        plt.plot(
            x_axis_poisson / n_images,
            poisson_distribution.pmf(x_axis_poisson) / new_area,
            label=f"Poisson distribution, mean of {expected_mean_dark:5.2f} "
                  f"counts",
        )

    #   Plot Gaussian
    if show_gaussian_distribution:
        #   The expected width of the Gaussian depends on the number of images
        #   TODO: Check if this is correct
        expected_scale = read_noise / gain * np.sqrt(n_images)

        #   Mean value is same as for the Poisson distribution (account for
        #   number of images)
        expected_mean = expected_mean_dark * n_images

        #
        gauss = stats.norm(loc=expected_mean, scale=expected_scale)

        #   X range
        x_axis_gauss = np.linspace(
            expected_mean - 5 * expected_scale,
            expected_mean + 5 * expected_scale,
            num=100,
        )

        plt.plot(
            x_axis_gauss / n_images,
            gauss.pdf(x_axis_gauss) * n_images,
            label='Gaussian, standard dev is read noise in counts',
        )

    #   Labels
    plt.xlabel(f"Dark counts in {exposure_time} sec exposure")
    plt.ylabel("Fraction of pixels (area normalized to 1)")
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_with_distributions_{}.pdf'.format(
        str(exposure_time).replace("''", "p")
    )
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_histogram(image_data, output_dir, gain, exposure_time):
    """
        Plot image histogram for dark images

        Parameters
        ----------
        image_data      : `numpy.ndarray`
            Dark frame to histogram

        output_dir      : `pathlib.Path`
            Path pointing to the main storage location

        gain            : `float`
            Gain of the camera, in electron/ADU

        exposure_time   : `float`
            Exposure time, in seconds
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Scale image
    image_data = image_data * gain / exposure_time

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

    #   Create histogram
    hist(
        image_data.flatten(),
        bins=5000,
        density=False,
        label=f'{exposure_time} sec dark',
        alpha=0.4,
    )

    #   Labels
    plt.xlabel('Dark current, $e^-$/sec')
    plt.ylabel('Number of pixels')
    plt.loglog()
    plt.grid()
    plt.legend()

    #   Write the plot to disk
    file_name = 'dark_hist_{}.pdf'.format(
        str(exposure_time).replace("''", "p")
    )
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def plot_median_of_flat_fields(image_file_collection, image_type, output_dir,
                               filter_):
    """
        Plot median and mean of each flat field in a file collection

        Parameters
        ----------
        image_file_collection   : `ccdproc.ImageFileCollection`
            File collection with the flat fields to analyze

        image_type              : `string`
            Header keyword characterizing the flats

        output_dir              : `pathlib.Path`
            Path pointing to the main storage location

        filter_                 : `string`
            Filter

        Idea/Reference
        --------------
            # https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'reduce_plots',
    )

    #   Calculate median and mean for each image
    median_count = []
    mean_count = []
    for data in image_file_collection.data(imagetyp=image_type, filter=filter_):
        median_count.append(np.median(data))
        mean_count.append(np.mean(data))

    #   Use bmh style
    # plt.style.use('bmh')

    #   Set layout of image
    plt.figure(figsize=(20, 9))

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
    file_name = 'flat_median_{}.pdf'.format(filter_.replace("''", "p"))
    plt.savefig(
        output_dir / 'reduce_plots' / file_name,
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()


def cutouts_fwhm_stars(output_dir, n_stars, sub_images_fwhm_stars, filter_,
                       basename):
    """
        Plots cutouts around the stars used to estimate the FWHM

        Parameters
        ----------
        output_dir              : `pathlib.Path`
            Path to the directory where the master files should be saved to

        n_stars                 : `integer`
            Number of stars

        sub_images_fwhm_stars   : `photutils.psf.EPSFStars object
            Sub images (squares) extracted around the FWHM stars

        filter_                 : `string`
            Filter name

        basename                : `string`
            Name of the image file
    """
    #   Check output directories
    checks.check_output_directories(
        output_dir,
        output_dir / 'cutouts',
    )

    #   Set number of rows and columns for the plot
    n_rows = 5
    n_columns = 5

    #   Prepare plot
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(20, 20),
        squeeze=True,
    )
    ax = ax.ravel()

    #   Set title of the complete plot
    fig.suptitle(
        f'Cutouts of the FWHM stars ({filter_}), {basename})',
        fontsize=20,
    )

    #   Loop over the cutouts (default: 25)
    for i in range(n_stars):
        # Set up normalization for the image
        norm = simple_norm(sub_images_fwhm_stars[i], 'log', percent=99.)

        # Plot individual cutouts
        ax[i].set_xlabel("[pixel]")
        ax[i].set_ylabel("[pixel]")
        ax[i].imshow(
            sub_images_fwhm_stars[i],
            norm=norm,
            origin='lower',
            cmap='viridis',
        )

    #   Write the plot to disk
    plt.savefig(
        f'{output_dir}/cutouts/cutouts_FWHM-stars_{filter_}_{basename}.pdf',
        bbox_inches='tight',
        format='pdf',
    )
    plt.close()
