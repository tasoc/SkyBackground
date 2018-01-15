#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function fo restimation of sky background in TESS Full Frame Images

Includes a '__main__' for independent test runs on local machines.

..versionadded:: 1.0.0
..versionchanged:: 1.0.2

.. codeauthor:: Carolina Von Essen <cessen@phys.au.dk>
.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
"""
#TODO: Write up more clearly how the code functions
#TODO: Add more in-depth comments

import numpy as np
import matplotlib.pylab as plt
import scipy
import pyfits
import glob

import scipy.ndimage
import scipy.integrate
from PyAstronomy import pyasl



def fit_background(ffi, plots_on = False):
    """
    Estimate the background of a Full Frame Image (FFI) by treating each line of
    data as a spectroscopic spectrum and fitting to the background accordingly.
    This is done in both the x and y directions, and averaged across to create a
    final background.

    Parameters:
        ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

        plots_on (bool): Default False. When True, it will plot an example of
            the method fitting to the first line of data on the first iteration
            of the fitting loop.

    Returns:
        ndarray: Estimated background with the same size as the input image.

    .. codeauthor:: Carolina Von Essen <cessen@phys.au.dk>
    .. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
    """

    # Setting up image dimensions
    ndim = ffi.shape[0]
    nbin = int(ndim/60)                         #Binning into ndim/60 bins
    ndim_vec = np.linspace(0,ndim,ndim)         #Creating a vector of length ndim
    nbin_vec = np.linspace(0,ndim,nbin)         #Creating a vector of length nbin
    int_nbin_vec = np.array(map(int,nbin_vec))  #Converting nbin_vec to integer

    # Setting up filter
    background_map = np.zeros([ndim,ndim])
    hamming_filter = ndim/2+1

    # Calculating data metadata
    data_min = np.min(ffi)
    data_max = np.max(ffi)
    data_mean_img = np.mean(ffi)

    # Preparing arrays for smoothed data
    ffi_smooth_y = np.zeros_like(background_map)
    ffi_smooth_x = np.zeros_like(background_map)

    # Cutting line by line in both x and y
    for i in range(ndim):
        min_vecx = np.zeros(nbin-1)
        min_vecy = np.zeros(nbin-1)
        y_msky = ffi[i,::]      #All X values for Y = i
        x_msky = ffi[::,i]      #All Y values for X = i

        for h in range(nbin-1):     #Get the minimum values in each bin
            min_vecx[h] = np.min(x_msky[int_nbin_vec[h]:int_nbin_vec[h+1]])
            min_vecy[h] = np.min(y_msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

        nbin_vec = nbin_vec[0:nbin-1]   #Adjusting nbin_vec to the same size as min_vec for interpolation

        #Interpolating the minimum values to the scale of ndim
        x_min_vec_int = np.interp(ndim_vec, nbin_vec, min_vecx)
        y_min_vec_int = np.interp(ndim_vec, nbin_vec, min_vecy)

        #Smoothing the interpolation
        x_min_vec_int_smooth = pyasl.smooth(x_min_vec_int, hamming_filter, 'hamming')
        y_min_vec_int_smooth = pyasl.smooth(y_min_vec_int, hamming_filter, 'hamming')

        #Saving the smoothed background line on this level
        ffi_smooth_x[::,i] = x_min_vec_int_smooth
        ffi_smooth_y[i,::] = y_min_vec_int_smooth

        if plots_on:
            if (i == 0):
                #Showing data with continuum estimation from method
                fig, ax = plt.subplots()
                ax.set_ylim(130000., 90000.)
                ax.set_title('Low end of FFI data sliced in y with continuum estimation shown')
                ax.plot(ndim_vec, y_msky, 'r-', label='FFI data')
                ax.plot(ndim_vec, y_min_vec_int, 'b-', label='Binned estimate')
                ax.plot(ndim_vec, y_min_vec_int_smooth, 'k-', label='Smoothed estimate (final)')
                ax.legend(fancybox=True, loc='best')
                plt.show()

    #Smooth across the background estimation using a Gaussian filter
    kern_px = 100
    ffi_smooth_yy = scipy.ndimage.gaussian_filter(ffi_smooth_y, kern_px/(2*np.sqrt(2*np.log(2))))
    ffi_smooth_xx = scipy.ndimage.gaussian_filter(ffi_smooth_x, kern_px/(2*np.sqrt(2*np.log(2))))

    # Taking the average between the two images
    bkg_est = (ffi_smooth_yy + ffi_smooth_xx)/2.

    return bkg_est

if __name__ == '__main__':
    # Set up parameters
    plt.close('all')
    plots_on = True

    # Read in data
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[0]
    sfile = glob.glob('../data/FFI/'+ffi_type+'.fits')[0]
    bgfile = glob.glob('../data/FFI/backgrounds_'+ffi_type+'.fits')[0]

    try:
        hdulist = pyfits.open(sfile)
        bkglist = pyfits.open(bgfile)

    except IOError:
        print('File not located correctly.')
        exit()

    ffi = hdulist[0].data
    bkg = bkglist[0].data

    # Run background estimation
    est_bkg = fit_background(ffi, plots_on)

    # Plot background difference
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_title(ffi_type)

    if plots_on:
        fbf, abf = plt.subplots()
        bgf = abf.imshow(np.log10(est_bkg), cmap='Blues_r', origin='lower')
        fbf.colorbar(bgf, label=r'$log_{10}$(Flux)')
        abf.set_title('Background averaged across evaluations in x and y')

        ftru, atru = plt.subplots()
        btru = atru.imshow(np.log10(bkg), cmap='Blues_r', origin='lower')
        ftru.colorbar(btru, label=r'$log_{10}$(Flux)')
        atru.set_xlabel('Pixel #')
        atru.set_ylabel('Pixel #')
        atru.set_title('True background of simulated data')

    fdiff, adiff = plt.subplots()
    diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
    fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
    adiff.set_title('Estimated bkg - True bkg')

    plt.show('all')
    plt.close('all')
