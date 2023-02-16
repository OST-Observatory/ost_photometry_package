############################################################################
####                            Libraries                               ####
############################################################################

import numpy as np

import os

from pathlib import Path

import time

import itertools

from astropy.visualization import (
    ImageNormalize,
    ZScaleInterval,
    simple_norm,
    )

from astropy.stats import sigma_clip as sigma_clipping
from astropy.time import Time
from astropy.timeseries import aggregate_downsample
import astropy.units as u

from itertools import cycle

import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
#plt.switch_backend('TkAgg')

from .. import checks, style, terminal_output


############################################################################
####                        Routines & definitions                      ####
############################################################################

def comp_img(outdir, o_image, c_image):
    '''
        Plot two images for comparison

        Parameters
        ----------
        outdir          : `string`
            Output directory

        o_image         : `numpy.ndarray`
            Original image data

        c_image         : `numpy.ndarray`
            Comparison image data
    '''
    #   Prepare plot
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    #   Original image: normalize and plot
    norm = simple_norm(o_image.data, 'log', percent=99.)
    ax1.imshow(o_image.data, norm=norm, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Original image')

    #   Comparison image: normalize and plot
    norm = simple_norm(c_image, 'log', percent=99.)
    ax2.imshow(c_image, norm=norm, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Downloaded image')

    #   Save the plot
    plt.savefig(outdir+'/img_comparison.pdf', bbox_inches='tight',format='pdf')
    plt.close()


def starmap(outdir, image, band, tbl, indent=2, tbl_2=None,
            label='Identified stars', label_2='Identified stars (set 2)',
            rts=None, mode=None, nameobj='', condense=False):
    '''
        Plot star maps  -> overlays of the determined star positions on FITS
                        -> supports different versions

        Parameters
        ----------
        outdir          : `string`
            Output directory

        image           : `numpy.ndarray`
            Image data

        band            : `string`
            Filter identifier

        tbl             : `astropy.table.Table`
            Astropy table with data of the objects

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``2``.

        tbl_2           : `astropy.table.Table`, optional
            Second astropy table with data of special objects
            Default is ``None``

        label           : `string`, optional
            Identifier for the objects in `tbl`
            Default is ``Identified stars``

        label_2         : `string`, optional
            Identifier for the objects in `tbl_2`
            Default is ``Identified stars (set 2)``

        rts             : `string`, optional
            Expression characterizing the plot
            Default is ``None``

        mode            : `string`, optional
            String used to switch between different plot modes
            Default is ``None``

        nameobj         : `string`, optional
            Name of the object
            Default is ````

        condense        : `boolean`, optional
            If True pass the console output to the calling function.
            Default is ``False``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'starmaps'),
        )

    if rts is not None:
        outstring = terminal_output.print_terminal(
            band,
            rts,
            indent=indent,
            string="Plot {} band image with stars overlaid ({})",
            condense=condense,
            )

    #   Check if column with X and Y coordinates are available for table 1
    if 'x' in tbl.colnames:
        x_column = 'x'
        y_column = 'y'
    elif 'xcentroid' in tbl.colnames:
        x_column = 'xcentroid'
        y_column = 'ycentroid'
    elif 'xfit' in tbl.colnames:
        x_column = 'xfit'
        y_column = 'yfit'
    else:
        raise RuntimeError(
            f"{style.bcolors.FAIL} \nNo valid X and Y column found for "
            f"table 1. {style.bcolors.ENDC}"
            )
    #   Check if column with X and Y coordinates are available for table 2
    if tbl_2 is not None:
        if 'x' in tbl_2.colnames:
            x_column_2 = 'x'
            y_column_2 = 'y'
        elif 'xcentroid' in tbl_2.colnames:
            x_column_2 = 'xcentroid'
            y_column_2 = 'ycentroid'
        elif 'xfit' in tbl_2.colnames:
            x_column_2 = 'xfit'
            y_column_2 = 'yfit'
        else:
            raise RuntimeError(
                f"{style.bcolors.FAIL} \nNo valid X and Y column found for "
                f"table 2. {style.bcolors.ENDC}"
                )

    #   Set layout of image
    fig = plt.figure(figsize=(20,9))

    #   Set title of the complete plot
    if rts is not None:
        fig.suptitle(
            'Star map ('+str(band)+' band, '+rts+') - '+nameobj,
            fontsize=20,
            )
    else:
        fig.suptitle(
            'Star map ('+str(band)+' band) - '+nameobj,
            fontsize=20,
            )

    #   Set up normalization for the image
    norm = ImageNormalize(image, interval=ZScaleInterval())

    #   Display the actual image
    plt.imshow(image, cmap='Greys', origin='lower', norm=norm,
               interpolation='nearest')

    #   Plot apertures
    plt.scatter(
        tbl[x_column],
        tbl[y_column],
        s=40,
        facecolors='none',
        edgecolors='#0547f9',
        lw=0.9,
        label=label,
        )
    if tbl_2 is not None:
        plt.scatter(
            tbl_2[x_column_2],
            tbl_2[y_column_2],
            s=40,
            facecolors='none',
            edgecolors='red',
            lw=0.9,
            label=label_2,
            )

    #   Set plot limits
    plt.xlim(0, image.shape[1]-1)
    plt.ylim(0, image.shape[0]-1)

    # Plot labels next to the apertures
    if mode == 'mags':
        for i in range(0,len(tbl[x_column])):
            plt.text(
                tbl[x_column][i],
                tbl[y_column][i],
                " "+str(f"{tbl['mags'][i]:.1f}"),
                fontdict=style.font,
                color='blue',
                )
    elif mode == 'list':
        for i in range(0,len(tbl[x_column])):
            plt.text(
                tbl[x_column][i],
                tbl[y_column][i],
                " "+str(i),
                fontdict=style.font,
                color='blue',
                )
    else:
        for i in range(0,len(tbl[x_column])):
            plt.text(
                tbl[x_column][i]+6,
                tbl[y_column][i]+6,
                " "+str(tbl['id'][i]),
                fontdict=style.font,
                color='blue',
                )
            #plt.text(
                #tbl['xcentroid'][i],
                #tbl['ycentroid'][i],
                #" "+str(tbl['id'][i]),
                #fontdict=font,
                #color='blue',
                #)
            #plt.annotate(
                #str(tbl['id'][i]),
                #(tbl['xcentroid'][i], tbl['ycentroid'][i]),
                #xytext=(tbl['xcentroid'][i], tbl['ycentroid'][i]),
                ##textcoords='offset points',
                #textcoords='offset pixels',
                ##fontdict=font,
                #color='blue',
                #)

    #   Define the ticks
    plt.tick_params(axis='both',which='both',top=True,right=True,
                    direction='in')
    plt.minorticks_on()

    #   Set labels
    plt.xlabel("Pixel", fontsize=16)
    plt.ylabel("Pixel", fontsize=16)

    #   Plot legend
    plt.legend(bbox_to_anchor=(0.,1.02,1.0,0.102),loc=3,ncol=2,
               mode='expand',borderaxespad=0.)

    #   Write the plot to disk
    if rts is None:
        plt.savefig(
            outdir+'/starmaps/starmap_'+band+'.pdf',
            bbox_inches='tight',
            format='pdf',
            )
    else:
        plt.savefig(
            outdir+'/starmaps/starmap_'+band+'_'+rts+'.pdf',
            bbox_inches='tight',
            format='pdf',
            )
    # plt.show()
    plt.close()

    if condense:
    	return outstring


def plot_apertures(outdir, image, aperture, annulus_aperture, string):
    '''
        Plot the apertures used for extracting the stellar fluxes
               (star map plot for aperture photometry)

        Parameters
        ----------
        outdir      : `string`
            Output directory

        image       : `numpy.ndarray`
            Image data (2D)

        aperture    : `photutils.aperture.CircularAperture`
            Apertures used to extract the stellar flux

        annulus     : `photutils.aperture.CircularAnnulus`
            Apertures used to extract the background flux

        string      : `string`
            String characterizing the output file
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'aperture'),
        )

    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20,9))

    #   Normalize the image
    norm = ImageNormalize(image, interval=ZScaleInterval())

    #   Plot the image
    plt.imshow(
        image,
        cmap='Greys',
        origin='lower',
        norm=norm,
        interpolation='nearest',
        )

    #   Plot stellar apertures
    ap_patches = aperture.plot(
        color='red',
        lw=0.2,
        label='Photometry aperture',
        )

    #   Plot background apertures
    ann_patches = annulus_aperture.plot(
        color='blue',
        lw=0.2,
        label='Background annulus',
        )

    #
    handles = (ap_patches[0], ann_patches[0])

    #   Plot legend
    plt.legend(
        loc=(0.17, 0.05),
        facecolor='#458989',
        labelcolor='white',
        handles=handles,
        prop={'weight': 'bold', 'size': 9},
        )

    #   Save figure
    plt.savefig(
        outdir+'/aperture/aperture_'+str(string)+'.pdf',
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def plot_cutouts(outdir, stars, string, condense=False, max_plot_stars=25,
                 nameobj='', indent=2):
    '''
        Plot the cutouts of the stars used to estimated the ePSF

        Parameters
        ----------
        outdir          : `string`
            Output directory

        stars           : `nump.ndarray`
            Numpy array with cutouts of the ePSF stars

        string          : `string`
            String characterizing the plot

        condense        : `boolean`, optional
            If True pass the console output to the calling function
            Default is ``False``.

        max_plot_stars  : `integer`, optional
            Maximum number of cutouts to plot
            Default is ``25``.

        nameobj         : `string`, optional
            Name of the object
            Default is ````.

        indent          : `integer`, optional
            Indentation for the console output lines.
            Default is ``2``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'cutouts'),
        )

    #   Set number of cutouts
    if len(stars) > max_plot_stars:
        n_cutouts = max_plot_stars
    else:
        n_cutouts = len(stars)

    outstr = terminal_output.print_terminal(
        string,
        indent=indent,
        string="Plot ePSF cutouts ({})",
        condense=condense,
        )

    ##  Plot the first cutouts (default: 25)
    #   Set number of rows and columns
    nrows = 5
    ncols = 5

    #   Prepare plot
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15),
                           squeeze=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.25)

    #   Set title of the complete plot
    fig.suptitle(
        'Cutouts of the '+str(n_cutouts)+' faintest stars ('
        +str(string)+') - '+nameobj,
        fontsize=20,
        )

    ax = ax.ravel()                             # flatten the image?

    #   Loop over the cutouts (default: 25)
    for i in range(n_cutouts):
        # Remove bad pixels that would spoil the image normalization
        data_image = np.where(stars[i].data<=0, 1E-7, stars[i].data)
        # Set up normalization for the image
        norm = simple_norm(data_image, 'log', percent=99.)
        # Plot individual cutouts
        ax[i].set_xlabel("Pixel")
        ax[i].set_ylabel("Pixel")
        ax[i].imshow(data_image, norm=norm, origin='lower', cmap='viridis')
    plt.savefig(outdir+'/cutouts/cutouts_'+str(string)
                +'.pdf',bbox_inches='tight',format='pdf')
    # plt.show()
    plt.close()

    if condense:
        return outstr


def plot_epsf(outdir, epsf, condense=False, nameobj='', indent=1):
    '''
        Plot the ePSF image of all filters

        Parameters
        ----------
        outdir          : `string`
            Output directory

        epsf            : `epsf.object` ???
            ePSF object, usually constructed by epsf_builder

        condense        : `boolean`, optional
            If True pass the console output to the calling function
            Default is ``False``.

        nameobj         : `string`, optional
            Name of the object
            Default is ````.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'epsfs'),
        )

    outstr = terminal_output.print_terminal(
        indent=indent,
        string="Plot ePSF image",
        condense=condense,
        )

    #   Set font size
    rcParams['font.size'] = 13

    #   Set up plot
    nplots = len(epsf)
    if nplots == 1:
        fig = plt.figure(figsize=(6,5))
    elif nplots == 2:
        fig = plt.figure(figsize=(13,5))
    else:
        fig = plt.figure(figsize=(20,15))


    #   Set title of the complete plot
    fig.suptitle('ePSF ('+nameobj+')', fontsize=20)

    #   Plot individual subplots
    for i, (band, eps) in enumerate(epsf.items()):
        #   Remove bad pixels that would spoil the image normalization
        epsf_clean = np.where(eps.data<=0, 1E-7, eps.data)
        #   Set up normalization for the image
        norm = simple_norm(epsf_clean, 'log', percent=99.)

        #   Make the sub plot
        if nplots == 1:
            ax = fig.add_subplot(1,1,i+1)
        elif nplots == 2:
            ax = fig.add_subplot(1,2,i+1)
        else:
            ax = fig.add_subplot(nplots,nplots,i+1)

        #   Plot the image
        im1 = ax.imshow(epsf_clean, norm=norm, origin='lower',
                        cmap='viridis')

        #   Set title of subplot
        ax.set_title(band)

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im1, ax=ax)

    if i == 2:
        plt.savefig(
            outdir+'/epsfs/epsf_'+band+'.pdf',
            bbox_inches='tight',
            format='pdf',
            )
    else:
        plt.savefig(
            outdir+'/epsfs/epsf.pdf',
            bbox_inches='tight',
            format='pdf',
            )
    # plt.show()
    plt.close()

    if condense:
        return outstr


def plot_residual(name, image_orig, residual_image, outdir,
                  condense=False, nameobj='', indent=1):
    '''
        Plot the original and the residual image

        Parameters
        ----------
        name            : `string`
            Name of the plot, can be name of the object

        image_orig      : `numpy.ndarray`
            Original image data

        residual_image  : `numpy.ndarray`
            Residual image data

        outdir          : `string`
            Output directory

        condense        : `boolean`, optional
            If True pass the console output to the calling function
            Default is ``False``.

        nameobj         : `string`, optional
            Name of the object
            Default is ````.

        indent          : `integer`, optional
            Indentation for the console output lines
            Default is ``1``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'residual'),
        )

    outstr = terminal_output.print_terminal(
        indent=indent,
        string="Plot original and the residual image",
        condense=condense,
        )

    #   Set font size
    rcParams['font.size'] = 13

    #   Set up plot
    nplots = len(image_orig)
    if nplots == 1:
        fig = plt.figure(figsize=(20,5))
    elif nplots == 2:
        fig = plt.figure(figsize=(20,10))
    else:
        fig = plt.figure(figsize=(20,20))

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=None,
        hspace=0.25,
        )

    #   Set title of the complete plot
    fig.suptitle(name+' ('+nameobj+')', fontsize=20)

    i = 1
    for band, image in image_orig.items():
        ##  Plot original image
        #   Set up normalization for the image
        norm = ImageNormalize(image, interval=ZScaleInterval())

        if nplots == 1:
            ax = fig.add_subplot(1,2,i)
        elif nplots == 2:
            ax = fig.add_subplot(2,2,i)
        else:
            ax = fig.add_subplot(nplots,2,i)

        #   Plot image
        im1 = ax.imshow(
            image,
            norm=norm,
            cmap='viridis',
            aspect=1,
            interpolation='nearest',
            origin='lower',
            )

        #   Set title of subplot
        ax.set_title('Original Image ('+band+')')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im1, ax=ax)

        i += 1

        ##  Plot residual image
        #   Set up normalization for the image
        norm = ImageNormalize(residual_image[band],
                              interval=ZScaleInterval())

        if nplots == 1:
            ax = fig.add_subplot(1,2,i)
        elif nplots == 2:
            ax = fig.add_subplot(2,2,i)
        else:
            ax = fig.add_subplot(nplots,2,i)

        #   Plot image
        im2 = ax.imshow(
            residual_image[band],
            norm=norm,
            cmap='viridis',
            aspect=1,
            interpolation='nearest',
            origin='lower',
            )

        #   Set titel of subplot
        ax.set_title('Residual Image ('+band+')')

        #   Set labels
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        #   Set color bar
        fig.colorbar(im2, ax=ax)

        i += 1

    #   Write the plot to disk
    if nplots == 1:
        plt.savefig(outdir+'/residual/residual_images_'+band+'.pdf',
                    bbox_inches='tight',format='pdf')
    else:
        plt.savefig(outdir+'/residual/residual_images.pdf',
                    bbox_inches='tight',format='pdf')
    # plt.show()
    plt.close()

    if condense:
        return outstr


def plot_mags(mag1, name1, mag2, name2, rts, outdir, err1=None, err2=None,
              nameobj='', fit=None):
    '''
        Plot magnitudes

        Parameters
        ----------
        mag1        : `numpy.ndarray`
            Magnitudes of filter 1

        name1       : `string`
            Filter 1

        mag2        : `numpy.ndarray`
            Magnitudes of filter 2

        name2       : `string`
            Filter 2

        rts         : `string`
            Expression characterizing the plot

        outdir      : `string`
            Output directory

        err1        : `numpy.ndarray' or ``None``, optional
            Error of the filter 1 magnitudes
            Default is ``None``.

        err2        : `numpy.ndarray' or ``None``, optional
            Error of the filter 1 magnitudes
            Default is ``None``.

        nameobj     : `string`, optional
            Name of the object
            Default is ````

        fit             : ` astropy.modeling.fitting` instance, optional
            Fit to plot
            Default is ``None``.
    '''
    scatter(mag1, name1+" [mag]", mag2, name2+" [mag]", rts, outdir, err1=err1,
              err2=err2, nameobj=nameobj, fit=fit)


def sigma_plot(bv, mags, bands, band, nr, outdir, nameobj='', fit=None):
    '''
        Illustrate sigma clipping of magnitudes

        Parameters
        ----------
        bv          : `numpy.ndarray`
            Delta color - (mag_2-mag_1)_observed - (mag_2-mag_1)_literature

        mags        : `numpy.ndarray`
            Magnitudes

        bands       : `list` of `string`
            Filter list

        band        : `list` of `string`
            Filter name

        nr          : `integer`
            Number of the star to plot

        outdir      : `string`
            Output directory

        nameobj     : `string`, optional
            Name of the object
            Default is ````.

        fit             : ` astropy.modeling.fitting` instance, optional
            Fit to plot
            Default is ``None``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'sigmag'),
        )

    #   Sigma clip magnitudes
    clip = sigma_clipping(mags, sigma=1.5)
    mask     = np.invert(clip.recordmask)
    clip_bv  = bv[mask]
    mag_clip = mags[mask]


    #   Plot sigma clipped magnitudes
    fig = plt.figure(figsize=(8,8))

    #   Set title
    fig.suptitle(
        'Sigma clipped magnitudes -- star: '+str(nr)+' ('+nameobj+')',
        fontsize=20,
        )

    #   Plot data
    plt.plot(mags,bv,color='blue',marker='.',linestyle='none')
    plt.plot(mag_clip,clip_bv,color='red',marker='.',linestyle='none')


    #   Plot fit
    if fit is not None:
        mags_sort = np.sort(mags)
        plt.plot(
            mags_sort,
            fit(mags_sort),
            color='r',
            linewidth=3,
            label='Polynomial fit',
            )

    #   Set x and y axis label
    plt.xlabel(str(band)+" [mag]")
    plt.ylabel("Delta "+bands[0]+"-"+bands[1])

    #   Save plot
    plt.savefig(outdir+'/sigmag/'+str(nr)+'_'+str(band)
                +'.png',bbox_inches='tight',format='png')
    plt.close()
    #plt.show()


def light_curve_jd(ts, data_column, err_column, outdir, error_bars=True,
                   nameobj=''):
    '''
        Plot the light curve over Julian Date

        Parameters
        ----------
        ts          : `astropy.timeseries.TimeSeries`
            Time series

        data_column : `string`
            Filter

        err_column  : `string`
            Name of the error column

        outdir      : `string`
            Output directory

        error_bars  : `boolean`, optional
            If True error bars will be plotted.
            Default is ``False``.

        nameobj     : `string`, optional
            Name of the object
            Default is ````.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'lightcurve'),
        )


    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20,9))

    #   Plot grid
    plt.grid(True)

    #   Set title
    fig.suptitle(f'Light curve - {nameobj}', fontsize=30)

    #   Plot data with or without error bars
    if not error_bars:
        plt.plot(ts.time.jd, ts[data_column], 'k.', markersize=3)
    else:
        plt.errorbar(
            ts.time.jd,
            np.array(ts[data_column]),
            yerr=np.array(ts[err_column]),
            fmt='k.',
            markersize=3,
            capsize=2,
            ecolor='lightgray',
            )

    #   Get median of the data
    median_data = np.median(ts[data_column].value)
    min_data = np.min(ts[data_column].value)
    max_data = np.max(ts[data_column].value)

    #   Invert y-axis
    if median_data > 1.5 or median_data < 0.5:
        plt.gca().invert_yaxis()

    #   Set plot limits
    yerr = ts[err_column].value
    yerr_sigma = sigma_clipping(yerr, sigma=1.5)
    max_err = np.max(yerr_sigma)

    if median_data > 1.1 or median_data < 0.9:
        ylim = np.max([max_err*1.5, 0.1])
        #ylim = np.max([max_err*2.0, 0.1])
        plt.ylim([median_data+ylim,median_data-ylim])
        #plt.ylim([max_data+ylim, min_data-ylim])
        ylabel_text = ' [mag] (Vega)'
    else:
        ylim = max_err*1.2
        #plt.ylim([median_data+ylim,median_data-ylim])
        plt.ylim([min_data-ylim,max_data+ylim])
        ylabel_text = ' [flux] (normalized)'

    #   Set x and y axis label
    plt.xlabel('Julian Date')
    plt.ylabel(data_column+ylabel_text)

    #   Save plot
    plt.savefig(
        f'{outdir}/lightcurve/lightcurve_jd_{data_column}.pdf',
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def light_curve_fold(ts, data_column, err_column, outdir, transit_time,
                     period, binn=None, error_bars=True, nameobj=''):
    '''
        Plot a folded light curve

        Parameters
        ----------
        ts              : `astropy.timeseries.TimeSeries`
            Time series

        data_column     : `string`
            Filter

        err_column      : `string`
            Name of the error column

        outdir          : `string`
            Output directory

        transit_time    : `string`
            Time of the transit - Format example: "2020-09-18T01:00:00"

        period          : `float`
            Period in days

        binn            : `float`, optional
            Light-curve binning-factor in days
            Default is ``None``.

        error_bars  : `boolean`, optional
            If True error bars will be plotted.
            Default is ``False``.

        nameobj     : `string`, optional
            Name of the object
            Default is ````.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'lightcurve'),
        )

    #   Make a time object for the  transit times
    transit_time=Time(transit_time, format='isot', scale='utc')

    #   Fold lightcurve
    ts_folded = ts.fold(period=period*u.day, epoch_time=transit_time)


    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20,9))

    #   Set title
    fig.suptitle('Folded light curve - '+nameobj, fontsize=30)

    #   Calculate binned lightcurve
    if binn is not None:
        ts_binned = aggregate_downsample(
            ts_folded,
            time_bin_size=binn * u.day,
            )

        #   Remove zero entries in case the binning time is smaller than the
        #   time between the data points
        mask = np.array(ts_binned[data_column]) == 0.
        mask = np.invert(mask)

    #   Plot data
    if error_bars:
        if binn is None:
            plt.errorbar(
                ts_folded.time.jd,
                np.array(ts_folded[data_column]),
                yerr=np.array(ts_folded[err_column]),
                fmt='k.',
                markersize=3,
                capsize=2,
                ecolor='lightgray',
                )
        else:
            plt.errorbar(
                ts_binned.time_bin_start.jd[mask],
                np.array(ts_binned[data_column][mask]),
                yerr=np.array(ts_binned[err_column][mask]),
                fmt='k.',
                markersize=3,
                capsize=2,
                ecolor='lightgray',
                )
    else:
        if binn is None:
            plt.plot(
                ts_folded.time.jd,
                ts_folded[data_column],
                'k.',
                markersize=3,
                )
        else:
            plt.plot(
                ts_binned.time_bin_start.jd[mask],
                ts_binned[data_column][mask],
                'k.',
                markersize=3,
                )


    #   Get median of the data
    median_data = np.median(ts_folded[data_column].value)

    #   Invert y-axis
    if median_data > 1.5 or median_data < 0.5:
        plt.gca().invert_yaxis()

    #plt.ylim([0.97,1.03])

    #   Set plot limits
    yerr = ts[err_column].value
    yerr_sigma = sigma_clipping(yerr, sigma=1.5)
    max_err = np.max(yerr_sigma)

    if median_data > 1.1 or median_data < 0.9:
        ylim = np.max([max_err*1.5, 0.1])
        plt.ylim([median_data+ylim,median_data-ylim])
        ylabel_text = ' [mag] (Vega)'
    else:
        ylim = max_err*1.3
        plt.ylim([median_data-ylim,median_data+ylim])
        ylabel_text = ' [flux] (normalized)'


    #   Set x and y axis label
    plt.xlabel('Time (days)')
    plt.ylabel(data_column+ylabel_text)

    #   Save plot
    plt.savefig(outdir+'/lightcurve/lightcurve_folded_'+str(data_column)
                +'.pdf', bbox_inches='tight', format='pdf')
    plt.close()


def plot_transform(outdir, color1, color2, color_lit, fit_var, a, b, b_err,
                   fit_func, airmass, filt=None, color_lit_err=None,
                   fit_var_err=None, nameobj=''):
    '''
        Make the plots to determine the calibration factors for the
        magnitude transformation

        Parameters
        ----------
        outdir          : `string`
            Output directory

        color1          : `string`
            Filter 1

        color2          : `string`
            Filter 2

        color_lit       : `numpy.ndarray`
            Colors of the calibration stars

        fit_var         : `numpy.ndarray`
            Fit variable

        a               : `float`
            First parameter of the fit

        b               : `float`
            Second parameter of the fit
            Currently only two fit parameters are supported
            -> Needs to generalized

        b_err           : `float`
            Error of `b`

        fit_func        : `fit.function`
            Fit function, used for determining the fit

        air_mass        : `float`
            Air mass

        filt            : `string`, optional
            Filter, used to distinguish between the different plot options
            Default is ``None``

        color_lit_err   : `numpy.ndarray`, optional
            Color errors of the calibration stars
            Default is ``None``.

        fit_var_err     : `numpy.ndarray`, optional
            Fit varaiable errors
            Default is ``None``.

        nameobj         : `string`
            Name of the object
            Default is ````.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'trans_plots'),
        )

    #   Fit data
    x_lin = np.sort(color_lit)
    y_lin = fit_func(x_lin,a,b)

    #   Set labels etc.
    airmass = round(airmass,2)
    if filt == None:
        #   coeff  = 1./b
        titel  = 'Color transform ('+color1.lower()+'-'+color2.lower()\
                 +' vs. '+color1+'-'+color2+') - '+nameobj\
                 +' (X = '+str(airmass)+')'
        ylabel = color1.lower()+'-'+color2.lower()+' [mag]'
        path   = outdir+'/trans_plots/'+color1.lower()+color2.lower()\
                 +'_'+color1+color2+'.pdf'
        plabel = 'slope = '+str(b)+', T'+color1.lower()+color2.lower()\
                 +' = '+str(1./b)+' +/- '+str(b_err)
    else:
        #   coeff  = b
        titel  = filt+color1.lower()+color2.lower()\
                 +'-mag transform ('+filt+'-'+filt.lower()\
                 +' vs. '+color1+'-'+color2+') - '+nameobj\
                 +' (X = '+str(airmass)+')'
        ylabel = filt+'-'+filt.lower()+' [mag]'
        path   = outdir+'/trans_plots/'+filt+filt.lower()\
                 +'_'+color1+color2+'.pdf'
        plabel = 'slope = '+str(b)+', C'+filt.lower()+'_'\
                 +color1.lower()+color2.lower()+' = '+str(b)\
                 +' +/- '+str(b_err)
    xlabel = color1+'-'+color2+' [mag]'

    #   Make plot
    fig = plt.figure(figsize=(20,9))

    #   Set title
    fig.suptitle(titel, fontsize=30)

    #   Plot data
    plt.errorbar(
        color_lit,
        fit_var,
        xerr=color_lit_err,
        yerr=fit_var_err,
        color='blue',
        marker='.',
        mew=0.0,
        linestyle='none',
        )

    #   Plot fit
    plt.plot(
        x_lin,
        y_lin,
        linestyle='-',
        color='red',
        linewidth=0.8,
        label=plabel,
        )

    #   Set legend
    plt.legend(
        bbox_to_anchor=(0.,1.02,1.0,0.102),
        loc=3,
        ncol=4,
        mode='expand',
        borderaxespad=0.,
        )

    #   Add grid
    plt.grid(color='0.95')

    #   Set x and y axis label
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    #   Add grid
    plt.grid(color='0.95')

    #   Get median of the data
    y_min = np.min(fit_var)
    y_max = np.max(fit_var)

    #   Set plot limits
    if fit_var_err is not None:
        yerr = fit_var_err
        yerr_sigma = sigma_clipping(yerr, sigma=1.5)
        max_err = np.max(yerr_sigma)

        ylim = np.max([max_err*1.5, 0.1])
        plt.ylim([y_max+ylim, y_min-ylim])

    #   Save plot
    plt.savefig(path,bbox_inches='tight',format='pdf')
    plt.close()



def check_plot(size_x, size_y, mag_filt, mag_color, yRangeMax, yRangeMin,
               xRangeMax, xRangeMin):
    '''
        Check the plot dimensions and set defaults

        Parameters
        ----------
        size_x      : `float`
            Figure size in cm (x direction)

        size_y      : `float`
            Figure size in cm (y direction)

        mag_filt    : `numpy.ndarray`
            Filter magnitude - 1D

        mag_color   : `numpy.ndarray`
            Color - 1D

        yRangeMax   : `float`
            The maximum of the plot range in Y direction

        yRangeMin   : `float`
            The minimum of the plot range in Y direction

        xRangeMax   : `float`
            The maximum of the plot range in X direction

        xRangeMin   : `float`
            The minimum of the plot range in X direction
    '''
    #   Set figure size
    if size_x == "" or size_x == "?" or size_y == "" or size_y == "?":
        terminal_output.print_terminal(
            string="[Info] No Plot figure size given, use default: 8cm x 8cm",
            style_name='WARNING',
            )
        plt.figure(figsize=(8,8))
    else:
        plt.figure(figsize=(int(size_x),int(size_y)))

    #   Set plot range -> automatic adjustment
    #   Y range
    try:
        float(yRangeMax)
    except ValueError:
        plt.ylim([float(np.max(mag_filt))+0.5, float(np.min(mag_filt))-0.5])
        terminal_output.print_terminal(
            string="[Info] Use automatic plot range for Y",
            style_name='WARNING',
            )
    else:
        try:
            float(yRangeMin)
        except ValueError:
            plt.ylim([float(np.max(mag_filt))+0.5,float(np.min(mag_filt))-0.5])
            terminal_output.print_terminal(
                string="[Info] Use automatic plot range for Y",
                style_name='WARNING',
                )
        else:
            plt.ylim([float(yRangeMin),float(yRangeMax)])

    #   X range
    try:
        float(xRangeMax)
    except ValueError:
        plt.xlim([float(np.min(mag_color))-0.5, float(np.max(mag_color))+0.5])
        terminal_output.print_terminal(
            string="[Info] Use automatic plot range for Y",
            style_name='WARNING',
            )
    else:
        try:
            float(xRangeMin)
        except ValueError:
            plt.xlim(
                [float(np.min(mag_color))-0.5, float(np.max(mag_color))+0.5]
                )
            terminal_output.print_terminal(
                string="[Info] Use automatic plot range for Y",
                style_name='WARNING',
                )
        else:
            plt.xlim([float(xRangeMin),float(xRangeMax)])


def mk_ticks_labels(filt, color):
    '''
        Set default ticks and labels

        Parameters
        ----------
        filt    : `string`
            Filter

        color   : `string`
            Color
    '''
    #   Set ticks
    plt.tick_params(
        axis='both',
        which='both',
        top=True,
        right=True,
        direction='in',
        )
    plt.minorticks_on()

    #   Set labels
    plt.xlabel(r'${}$'.format(color))
    plt.ylabel(r'${}$'.format(filt))


def fill_lists(liste, ISOcolumn, ISOcolumntype, filt_1, filt_2, iso_mag1,
               iso_mag2, iso_color):
    '''
        Sort magnitudes into lists and calculate the color if necessary

        Parameters
        ----------
        liste           : `list` of `string`
            List of strings

        ISOcolumn       : `dictionary`
            Columns to use from the ISO file.
            Keys = filter           : `string`
            Values = column numbers : `integer`

        ISOcolumntype   : `dictionary`
            Type of the columns from the ISO file
            Keys = filter : `string`
            Values = type : `string`

        filt_1          : `string`
            First filter

        filt_2          : `string`
            Second filter

        iso_mag1        : `list` of `float`
            Magnitude list (first filter)

        iso_mag2        : `list` of `float`
            Magnitude list (second filter)

        iso_color       : `list` of `float`
            Color list
    '''
    mag1 = float(liste[ISOcolumn[filt_1]-1])
    iso_mag1.append(mag1)
    if ISOcolumntype[filt_2] == 'color':
        color = float(liste[ISOcolumn[filt_2]-1])
        iso_color.append(color)
    elif ISOcolumntype[filt_2] == 'single':
        mag2 = float(liste[ISOcolumn[filt_2]-1])
        iso_mag2.append(mag2)
        iso_color.append(mag2-mag1)

    return iso_mag1, iso_mag2, iso_color


def mk_colormap(Niso):
    '''
        Make a color map e.g. for isochrones

        Parameters
        ----------
        Niso    : `integer`
            Number of isochrone files
    '''
    #   Prepare colors for the isochrones
    #   Self defined colormap
    cm1 = mcol.LinearSegmentedColormap.from_list(
        "MyCmapName",
        ['orchid',
         'blue',
         'cyan',
         'forestgreen',
         'limegreen',
         'gold',
         'orange',
         "red",
         'saddlebrown',
         ]
        )
    cnorm = mcol.Normalize(vmin=0,vmax=Niso)
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    return cpick


def mk_cycler():
    '''
        Make a line cycler
    '''
    lines = ["-","--","-.",":"]
    return cycle(lines)


def write_cmd(nameOfStarcluster, filename, filt_1, color, filetype, ptype,
              outdir='output'):
    '''
        Write plot to disk

        Parameters
        ----------
        nameOfStarcluster   : `string`
            Name of cluster

        filename            : `string`
            Base name of the file to write

        filt_1              : `string`
            Filter

        color               : `string`
            Color

        filetype            : `string`
            File type

        ptype               : `string`
            Plot type

        outdir          : `string`, optional
            Output directory
            Default is ``output``.
    '''
    if nameOfStarcluster == "" or nameOfStarcluster == "?":
        terminal_output.print_terminal(
            outdir,
            filename,
            ptype,
            filt_1,
            color,
            filetype,
            string="Write plotfile: ./{}/{}_{}_{}_{}.{}",
            )
        plt.savefig(
            f'./{outdir}/{filename}_{ptype}_{filt_1}_{color}.{filetype}',
            format=filetype,
            bbox_inches="tight",
            )
    else:
        nameOfStarcluster = nameOfStarcluster.replace(' ','_')
        terminal_output.print_terminal(
            outdir,
            filename,
            nameOfStarcluster,
            ptype,
            filt_1,
            color,
            filetype,
            string="Write plotfile: ./{}/{}_{}_{}_{}_{}.{}",
            )
        plt.savefig(
            f'./{outdir}/{filename}_{nameOfStarcluster}_{ptype}_{filt_1}'
            f'_{color}.{filetype}',
            format=filetype,
            bbox_inches="tight",
            )


def plot_apparent_cmd(mag_color, mag_filt, nameOfStarcluster, filename,
                      filetype, filt_1, filt_2, size_x='', size_y='',
                      yRangeMax='', yRangeMin='', xRangeMax='',
                      xRangeMin='', outdir='output', mag_filt_err=None,
                      color_err=None):
    '''
        Plot calibrated cmd with apparent magnitudes

        Parameters
        ----------
        mag_color           : `numpy.ndarray`
            Color - 1D

        mag_filt            : `numpy.ndarray`
            Filter magnitude - 1D

        nameOfStarcluster   : `string`
            Name of cluster

        filename            : `string`
            Base name of the file to write

        filetype            : `string`
            File type

        filt_1              : `string`
            First filter

        filt_2              : `string`
            Second filter

        size_x              : `float`
            Figure size in cm (x direction)

        size_y              : `float`
            Figure size in cm (y direction)

        yRangeMax           : `float`
            The maximum of the plot range in Y direction

        yRangeMin           : `float`
            The minimum of the plot range in Y direction

        xRangeMax           : `float`
            The maximum of the plot range in X direction

        xRangeMin           : `float`
            The minimum of the plot range in X direction

        outdir              : `string`, optional
            Output directory
            Default is ``output``.

        mag_filt_err        : `numpy.ndarray' or ``None``, optional
            Error for ``mag_filt``
            Default is ``None``.

        color_err           : `numpy.ndarray' or ``None``, optional
            Error for ``mag_color``
            Default is ``None``.

    '''
    #   Check plot dimensions and set defaults
    check_plot(
        size_x,
        size_y,
        mag_filt,
        mag_color,
        yRangeMax,
        yRangeMin,
        xRangeMax,
        xRangeMin,
        )

    #   Plot the stars
    terminal_output.print_terminal(indent=1, string="Add stars")
    plt.errorbar(
        mag_color,
        mag_filt,
        xerr=mag_filt_err,
        yerr=color_err,
        mew=0.0,
        fmt='b.',
        markersize=3,
        capsize=2,
        ecolor='lightgray',
        #ecolor='gainsboro',
        #ecolor='silver',
        #ecolor='linen',
        #ecolor='beige',
        #ecolor='lavender',
        #ecolor='honeydew',
        )

    #   Set ticks and labels
    color = filt_2+'-'+filt_1
    mk_ticks_labels(filt_1, color)

    #   Write plot to disk
    write_cmd(nameOfStarcluster, filename, filt_1, color, filetype,
              'apparent', outdir=outdir)
    plt.close()


def plot_absolute_cmd(mag_color, mag_filt, nameOfStarcluster, filename,
                      filetype, filt_1, filt_2, isos, isotype,
                      ISOcolumntype, ISOcolumn, logAGE, keyword, IsoLabels,
                      size_x='', size_y='', yRangeMax='', yRangeMin='',
                      xRangeMax='', xRangeMin='', outdir='output',
                      mag_filt_err=None, color_err=None):
    '''
        Plot calibrated CMD with
            * magnitudes corrected for reddening and distance
            * isochrones

        Parameters
        ----------
        mag_color           : `numpy.ndarray`
            Color - 1D

        mag_filt            : `numpy.ndarray`
            Filter magnitude - 1D

        nameOfStarcluster   : `string`
            Name of cluster

        filename            : `string`
            Base name of the file to write

        filetype            : `string`
            File type

        filt_1              : `string`
            First filter

        filt_2              : `string`
            Second filter

        isos                : `string`
            Path to the isochrone directory or the isochrone file

        isotype             : `string`
            Type of 'isos'
            Possibilities: 'directory' or 'file'

        ISOcolumntype       : `dictionary`
            Keys = filter : `string`
            Values = type : `string`

        ISOcolumn           : `dictionary`
            Keys = filter           : `string`
            Values = column numbers : `integer`

        logAGE              : `boolean`
            Logarithmic age

        keyword             : `string`
            Keyword to identify a new isochrone

        IsoLabels           : `boolean`
            If True plot legend for isochrones.

        size_x              : `float`, optional
            Figure size in cm (x direction)
            Default is ````.

        size_y              : `float`, optional
            Figure size in cm (y direction)
            Default is ````.

        yRangeMax           : `float`, optional
            The maximum of the plot range in Y
                                direction
            Default is ````.

        yRangeMin           : `float`, optional
            The minimum of the plot range in Y
                                direction
            Default is ````.

        xRangeMax           : `float`, optional
            The maximum of the plot range in X
                                direction
            Default is ````.

        xRangeMin           : `float`, optional
            The minimum of the plot range in X direction

        outdir          : `string`, optional
            Output directory
            Default is ``output``.

        mag_filt_err        : `numpy.ndarray' or ``None``, optional
            Error for ``mag_filt``
            Default is ``None``.

        color_err           : `numpy.ndarray' or ``None``, optional
            Error for ``mag_color``
            Default is ``None``.
    '''
    #   Check plot dimensions and set defaults
    check_plot(size_x, size_y, mag_filt, mag_color, yRangeMax, yRangeMin,
               xRangeMax, xRangeMin)

    #   Plot the stars
    terminal_output.print_terminal(string="Add stars")
    plt.errorbar(
        mag_color,
        mag_filt,
        xerr=mag_filt_err,
        yerr=color_err,
        mew=0.0,
        fmt='b.',
        markersize=3,
        capsize=2,
        ecolor='lightgray',
        )



    ###
    #   Plot isochrones
    #

    #   Check if isochrones are specified
    if isos != '' and isos != '?':
        #   OPTION I: Individual isochrone files in a specific directory
        if isotype == 'directory':
            #   Resolve iso path
            isos = Path(isos).expanduser()

            #   Make list of isochrone files
            fileList = os.listdir(isos)

            #   Number of isochrones
            Niso = len(fileList)
            terminal_output.print_terminal(
                Niso,
                string="Plot {} isochrone(s)",
                style_name='OKGREEN',
                )

            #   Make color map
            cpick = mk_colormap(Niso)

            #   Prepare cycler for the line styles
            linecycler = mk_cycler()

            #   Cycle through iso files
            for i in range(0,Niso):
                #   Load file
                isodata = open(isos / fileList[i])

                #   Prepare variables for the isochrone data
                iso_mag1  = []
                iso_mag2  = []
                iso_color = []
                age_num   = ''
                age_unit  = ''

                #   Extract B and V values & make lists
                #   Loop over all lines in the file
                for line in isodata:
                    liste=line.split()

                    #   Check that the entries are not HEADER key words
                    try:
                        float(liste[0])
                    except:
                        #   Try to find and extract age information
                        if 'Age' in liste or 'age' in liste:
                            try:
                                age_index = liste.index('age')
                            except:
                                age_index = liste.index('Age')

                            for string in liste[age_index+1:]:
                                #   Find age unit
                                if string.rfind("yr") != -1:
                                    age_unit = string
                                #   Find age value
                                try:
                                    if isinstance(age_num, str):
                                        age_num = int(float(string))
                                except:
                                    pass
                        continue

                    #   Fill lists
                    iso_mag1, iso_mag2, iso_color = fill_lists(
                        liste,
                        ISOcolumn,
                        ISOcolumntype,
                        filt_1,
                        filt_2,
                        iso_mag1,
                        iso_mag2,
                        iso_color,
                        )

                #   Construct label
                if not isinstance(age_num, str):
                    lable = str(age_num)
                    if age_unit != '':
                        lable += ' '+age_unit
                else:
                    lable=os.path.splitext(fileList[i])[0]

                #   Plot iso lines
                plt.plot(
                    iso_color,
                    iso_mag1,
                    linestyle=next(linecycler),
                    color=cpick.to_rgba(i),
                    linewidth=0.8,
                    label=lable,
                    )

                #   Close file with the iso data
                isodata.close()


        #   OPTION II: Isochrone file containing many individual isochrones
        if isotype == 'file':
            #   Resolve iso path
            isos = Path(isos).expanduser()

            #   Load file
            isodata = open(isos)

            #   Overall list for the isochrones
            age            = []
            age_list       = []
            iso_mag1_list  = []
            iso_mag2_list  = []
            iso_color_list = []

            #   Initialize bool to signalize a new isochrone
            newiso = False

            #   Loop over all lines in the file
            for line in isodata:
                liste  = line.split()

                #   Check for a key word to distinguish the isochrones
                try:
                    if line[0:len(keyword)] == keyword:
                        #   Save age
                        if ISOcolumn['AGE'] == 0:
                            age_key = line.split('=')[1].split()[0]

                        #   Add iso data to the overall lists
                        #   for the isochrones
                        if newiso:
                            age_list.append(age)
                            iso_mag1_list.append(iso_mag1)
                            iso_mag2_list.append(iso_mag2)
                            iso_color_list.append(iso_color)

                        #   Prepare/reset lists for the single isochrones
                        iso_mag1  = []
                        iso_mag2  = []
                        iso_color = []
                        age       = []

                        #   Set bool to signalize a new isochrone
                        newiso = True
                        continue
                except:
                    continue

                #   Check that the entries are not HEADER key words
                try:
                    float(liste[0])
                except:
                    continue

                #   Fill lists
                if ISOcolumn['AGE'] != 0:
                    a = float(liste[ISOcolumn['AGE']-1])
                    age.append(a)
                else:
                    age.append(age_key)
                iso_mag1, iso_mag2, iso_color = fill_lists(
                    liste,
                    ISOcolumn,
                    ISOcolumntype,
                    filt_1,
                    filt_2,
                    iso_mag1,
                    iso_mag2,
                    iso_color,
                    )

            #   Close isochrone file
            isodata.close()

            #   Number of isochrones
            Niso = len(iso_mag1_list)
            terminal_output.print_terminal(
                Niso,
                string="Plot {} isochrone(s)",
                style_name='OKGREEN',
                )

            #   Make color map
            cpick = mk_colormap(Niso)

            #   Prepare cycler for the line styles
            linecycler = mk_cycler()

            #   Cycle through iso lines
            for i in range(0,Niso):
                if logAGE:
                    age_value = float(age_list[i][0])
                    age_value = 10**(age_value)/10**9
                    age_value = round(age_value,2)
                else:
                    age_value = round(float(age_list[i][0]),2)
                agestr = str(age_value)+' Gyr'

                #   Plot iso lines
                plt.plot(
                    iso_color_list[i],
                    iso_mag1_list[i],
                    linestyle=next(linecycler),
                    color=cpick.to_rgba(i),
                    linewidth=0.8,
                    label=agestr,
                    )
                isodata.close()

        #   Plot legend
        if IsoLabels:
            plt.legend(
                bbox_to_anchor=(0.,1.02,1.0,0.102),
                loc=3,
                ncol=4,
                mode='expand',
                borderaxespad=0.,
                )

    #   Set ticks and labels
    color = filt_2+'-'+filt_1
    mk_ticks_labels(filt_1, color)

    #   Write plot to disk
    write_cmd(nameOfStarcluster, filename, filt_1, color, filetype,
              'absolut', outdir=outdir)
    plt.close()


def comp_scatter(values_x, values_y, name_x, name_y, string, outdir,
                 oneTOone=True):
    '''
        Make a 2D scatter plot

        Parameters
        ----------
        values_x    : `numpy.ndarray`
            X values to plot

        values_y    : `numpy.ndarray`
            Y values to plot

        name_x      : `string`
            Label for the X axis

        name_y      : `string`
            Label for the Y axis

        string      : `string`
            String characterizing the output file

        outdir      : `string`
            Output directory

        oneTOone   : `boolean`, optional
            If True a 1:1 line will be plotted.
            Default is ``True``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'compare'),
        )


    ###
    #   Make plot
    #
    fig = plt.figure(figsize=(20,9))

    #   Determine boundaries for 1:1 line
    x_min = np.amin(values_x)
    x_max = np.amax(values_x)
    y_min = np.amin(values_y)
    y_max = np.amax(values_y)
    max_plot = np.max([x_max, y_max])
    min_plot = np.min([x_min, y_min])

    #   Plot the data
    plt.scatter(values_x, values_y)

    #   Set X & Y label
    plt.xlabel(name_x)
    plt.ylabel(name_y)

    #   Plot the 1:1 line
    if oneTOone:
        plt.plot(
            [min_plot, max_plot],
            [min_plot, max_plot],
            color='black',
            lw=2,
            )

    #   Save figure
    plt.savefig(
        outdir+'/compare/compare'+str(string)+'.pdf',
        bbox_inches='tight',
        format='pdf',
        )
    plt.close()


def onpick3(event):
    print('---------------------')
    print(dir(event))
    ind = event.ind
    #print('onpick3 scatter:', ind, np.take(x, ind))
    print('onpick3 scatter:', ind)
    print(event.artist)
    print(dir(event.artist))
    print(event.artist.get_label())
    print(event.artist.get_gid())
    #print(event.mouseevent)
    #print(dir(event.mouseevent))
    #print(event.mouseevent.inaxes)
    #print(dir(event.mouseevent.inaxes))
    #print(event.name)
    print('+++++++++++++++++++++')


def click_point(event):
    print('---------------------')
    print(dir(event))
    print(event.button)
    print(event.guiEvent)
    print(event.key)
    print(event.lastevent)
    print(event.name)
    print(event.step)
    print('+++++++++++++++++++++')


def D3_scatter(xs, ys, zs, outdir, color=None, name_x='', name_y='',
               name_z='', string='_3D_', pmra=None, pmde=None,
               display=False):
    '''
        Make a 2D scatter plot

        Parameters
        ----------
        xs           : `list` of `numpy.ndarray`s
            X values

        ys          : `list` of `numpy.ndarray`s
            Y values

        zs          : `list` of `numpy.ndarray`s
            Z values

        outdir      : `string`
            Output directory

        name_x      : `string`, optional
            Label for the X axis
            Default is ````.

        name_y      : `string`, optional
            Label for the Y axis
            Default is ````.

        name_z      : `string`, optional
            Label for the Z axis
            Default is ````.

        string      : `string`, optional
            String characterizing the output file
            Default is ``_3D_``.

        pmra        : `float`, optional
            Literature proper motion in right ascension.
            If not ``None`` the value will be printed to the plot.
            Default is ``None``.

        pmde        : `float`, optional
            Literature proper motion in declination.
            If not ``None`` the value will be printed to the plot.
            Default is ``None``.

        display     : `boolean`, optional
            If ``True`` the 3D plot will be displayed in an interactive
            window. If ``False`` four views of the 3D plot will be saved to
            a file.
            Default is ``False``.
    '''
    #   Switch backend to allow direct display of the plot
    if display:
        plt.switch_backend('TkAgg')

    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'compare'),
        )

    #   Prepare plot
    fig = plt.figure(figsize=(20,15), constrained_layout=True)

    #   Set title
    if display:
        if pmra is not None and pmde is not None:
            fig.suptitle(
                'Proper motion vs. distance: Literature proper motion: '\
                '{:.1f}, {:.1f} - Choose a cluster then close the ' \
                'plot'.format(pmra, pmde),
                fontsize=20,
                )
        else:
            fig.suptitle(
                'Proper motion vs. distance: Literature proper motion: '\
                +'- Choose a cluster then close the plot',
                fontsize=20,
                )
    else:
        if pmra is not None and pmde is not None:
            fig.suptitle(
                'Proper motion vs. distance: Literature proper motion: '\
                '{:.1f}, {:.1f} '.format(pmra, pmde),
                fontsize=20,
                )
        else:
            fig.suptitle(
                'Proper motion vs. distance',
                fontsize=20,
                )

    #   Switch to one subplot for direct display
    if display:
        nsubs = 1
    else:
        nsubs = 4

    #   Loop over all subplots
    for i in range(0,nsubs):
        if display:
            ax = fig.add_subplot(1, 1, i+1, projection='3d')
        else:
            ax = fig.add_subplot(2, 2, i+1, projection='3d')

        #   Change view angle
        ax.view_init(25, 45 + i*90)

        #   Labelling X-Axis
        ax.set_xlabel(name_x)

        #   Labelling Y-Axis
        ax.set_ylabel(name_y)

        #   Labelling Z-Axis
        ax.set_zlabel(name_z)

        #   Set default plot ranges/limits
        default_pm_range = [-20,20]
        default_dist_range = [0,10]

        #   Find suitable plot ranges
        xs_list = list(itertools.chain.from_iterable(xs))
        max_xs = np.max(xs_list)
        min_xs = np.min(xs_list)

        ys_list = list(itertools.chain.from_iterable(ys))
        max_ys = np.max(ys_list)
        min_ys = np.min(ys_list)

        dist_list = list(itertools.chain.from_iterable(zs))
        max_zs = np.max(dist_list)
        min_zs = np.min(dist_list)

        #   Set range: defaults or values from above
        if default_pm_range[0] < min_xs:
            xmin = min_xs
        else:
            xmin = default_pm_range[0]
        if default_pm_range[1] > min_xs:
            xmax = max_xs
        else:
            xmax = default_pm_range[1]
        if default_pm_range[0] < min_ys:
            ymin = min_ys
        else:
            ymin = default_pm_range[0]
        if default_pm_range[1] > min_ys:
            ymax = max_ys
        else:
            ymax = default_pm_range[1]
        if default_dist_range[0] < min_zs:
            zmin = min_zs
        else:
            zmin = default_dist_range[0]
        if default_dist_range[1] > min_zs:
            zmax = max_zs
        else:
            zmax = default_dist_range[1]

        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_zlim([zmin,zmax])

        #   Plot data
        if color is None:
            for i, x in enumerate(xs):
                p = ax.scatter3D(
                    x,
                    ys[i],
                    zs[i],
                    #c=zs[i],
                    cmap='cividis',
                    #cmap='tab20',
                    label='Cluster '+str(i),
                    #picker=True,
                    picker=5,
                    )
                ax.legend()
        else:
            for i, x in enumerate(xs):
                p = ax.scatter3D(
                    x,
                    ys[i],
                    zs[i],
                    c=color[i],
                    cmap='cividis',
                    #cmap='tab20',
                    label='Cluster '+str(i),
                    )
                ax.legend()

    #fig.canvas.mpl_connect('pick_event', onpick3)
    #fig.canvas.mpl_connect('button_press_event',click_point)

    #   Display plot and swicht backend back to default
    if display:
        plt.show()
        #plt.show(block=False)
        #time.sleep(300)
        #print('after sleep')
        plt.close()
        plt.switch_backend('Agg')
    else:
        #   Save image if it is not displayed directly
        plt.savefig(
            outdir+'/compare/pm_vs_distance.pdf',
            bbox_inches='tight',
            format='pdf',
            )
        plt.close()


def scatter(value1, name1, value2, name2, rts, outdir, err1=None, err2=None,
            nameobj='', fit=None):
    '''
        Plot magnitudes

        Parameters
        ----------
        value1      : `numpy.ndarray`
            Magnitudes of filter 1

        name1       : `string`
            Filter 1

        value2      : `numpy.ndarray`
            Magnitudes of filter 2

        name2       : `string`
            Filter 2

        rts         : `string`
            Expression characterizing the plot

        outdir      : `string`
            Output directory

        err1        : `numpy.ndarray' or ``None``, optional
            Error of value 1
            Default is ``None``.

        err2        : `numpy.ndarray' or ``None``, optional
            Error of value 2
            Default is ``None``.

        nameobj     : `string`, optional
            Name of the object
            Default is ````

        fit             : ` astropy.modeling.fitting` instance, optional
            Fit to plot
            Default is ``None``.
    '''
    #   Check output directories
    checks.check_out(
        outdir,
        os.path.join(outdir, 'scatter'),
        )

    #   Plot magnitudes
    fig = plt.figure(figsize=(8,8))

    #   Set title
    fig.suptitle(
        name1+' vs. '+name2+' ('+nameobj+'):',
        fontsize=20,
        )

    #   Plot data
    plt.errorbar(
        value1,
        value2,
        xerr=err1,
        yerr=err2,
        fmt='b.',
        markersize=3,
        capsize=2,
        ecolor='lightgray',
        )

    #   Add grid
    plt.grid(color='0.95')

    #   Plot fit
    if fit is not None:
        value1_sort = np.sort(value1)
        plt.plot(
            value1_sort,
            fit(value1_sort),
            color='r',
            linewidth=3,
            label='Fit',
            )

    #   Set x and y axis label
    plt.ylabel(name2)
    plt.xlabel(name1)

    #   Save plot
    plt.savefig(
        outdir+'/scatter/'+rts+'.pdf',
        bbox_inches='tight',
        format='pdf',
               )
    plt.close()
