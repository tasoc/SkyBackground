#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#T'DA1 | Carolina VAN ESSEN  | 2016
#T'DA3 | Oliver James HALL | 2017

import numpy as np
import matplotlib.pylab as plt
import scipy
import pyfits
import glob

import scipy.ndimage
import scipy.integrate
from PyAstronomy import pyasl


'''
Estimate the background of a Full Frame Image (FFI) by treating each line of
data as a spectroscopic spectrum and fitting to the background accordingly.
This is done in both the x and y directions, and averaged across to create a
final background.

Notes: Struggles with crowded fields (simulated cluster FFI)

Includes a '__main__' for independent test runs on local machines.

Parameters
-----------
ffi : float, numpy array
        A single TESS Full Frame Image in the form of a 2D array.

plots_on : default False, boolean
        A boolean parameter. When True, it will plot an example of the method
        fitting to the first line of data on the first iteration of the fitting
        loop.
-----------

Output
-----------
final : float, numpy array
        A background estimate in the same style and shape as the FFI input.

TO DO:
-   Write up more clearly how the code functions.
-   Add more in-depth comments.

'''


def fit_background(ffi, plots_on = False):
    '''Setting up image dimensions'''
    ndim = ffi.shape[0]
    nbin = int(ndim/60)
    ndim_vec = np.linspace(0,ndim,ndim)
    nbin_vec = np.linspace(0,ndim,nbin)
    int_nbin_vec = np.zeros(nbin, dtype=np.int)
    for j in range(nbin):
      int_nbin_vec[j] = int(nbin_vec[j])

    '''Setting up filter'''
    background_map = np.zeros(ndim*ndim).reshape(ndim, ndim)
    hamming_filter = ndim/2+1

    '''Calculating data metadata'''
    data_min = np.min(ffi)
    data_max = np.max(ffi)
    data_mean_img = np.mean(ffi)

    '''Smoothing the data'''
    ffi_smooth_1 = np.zeros(ndim*ndim).reshape(ndim,ndim)
    ffi_smooth_2 = np.zeros(ndim*ndim).reshape(ndim,ndim)

    '''Cutting line by line in y'''
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = ffi[i,::]

        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        ffi_smooth_1[i,::] = min_vec_int_smooth

        if plots_on:
            if (i == 0):
              plt.ylim(90000.,250000.)
              plt.plot(ndim_vec, msky, 'r-')
              plt.show()
              plt.ylim(250000., 90000.)
              plt.plot(ndim_vec, msky, 'r-')
              plt.show()
              plt.ylim(130000., 90000.)
              plt.plot(ndim_vec, msky, 'r-')
              plt.show()
              plt.ylim(130000., 90000.)
              plt.plot(ndim_vec, msky, 'r-')
              plt.plot(ndim_vec, min_vec_int, 'b-')
              plt.plot(ndim_vec, min_vec_int_smooth, 'k-')
              plt.show()

    kern_px = 100
    ffi_smooth_11 = scipy.ndimage.gaussian_filter(ffi_smooth_1, kern_px/(2*np.sqrt(2*np.log(2))))

    '''Repeating the process in x'''
    for i in range(ndim):
        min_vec = np.zeros(nbin)
        msky = ffi[::,i]
        for h in range(nbin-1):
            min_vec[h] = np.min(msky[int_nbin_vec[h]:int_nbin_vec[h+1]])

        nbin_vec = nbin_vec[0:nbin-1] ; min_vec = min_vec[0:nbin-1]
        min_vec_int = np.interp(ndim_vec, nbin_vec, min_vec)

        min_vec_int_smooth = pyasl.smooth(min_vec_int, hamming_filter, 'hamming')

        ffi_smooth_2[::,i] = min_vec_int_smooth

    ffi_smooth_22 = scipy.ndimage.gaussian_filter(ffi_smooth_2, kern_px/(2*np.sqrt(2*np.log(2))))


    '''Taking the average between the two images.'''
    final = (ffi_smooth_11 + ffi_smooth_22)/2.

    return final

if __name__ == '__main__':
    plt.close('all')

    #Load file:
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[2]
    sfile = glob.glob('../data/'+ffi_type+'/simulated/*.fits')[0]
    bgfile = glob.glob('../data/'+ffi_type+'/backgrounds.fits')[0]

    try:
        hdulist = pyfits.open(sfile)
        bkglist = pyfits.open(bgfile)

    except IOError:
        print('File not located correctly.')
        exit()

    ffi = hdulist[0].data
    bkg = bkglist[0].data

    plots_on = False
    est_bkg = fit_background(ffi, plots_on)

    '''Plotting: all'''
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
