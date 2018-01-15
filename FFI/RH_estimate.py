#!/bin/env python
# -*- coding: utf-8 -*-

"""
Function for estimation of sky background in TESS Full Frame Images.

Includes a '__main__' for independent test runs on local machines.

..versionadded:: 1.0.0
..versionchanged:: 1.0.2

Notes: Copied over from TASOC/photometry/backgrounds.py [15/01/18]

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>

"""
#TODO: Use the known locations of bright stars
#TODO: Add testing call function

import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.io.fits as pyfits
from photutils import Background2D, SExtractorBackground
from astropy.stats import SigmaClip


def fit_background(ffi):
    """
    Estimate the background of a Full Frame Image (FFI) using the photutils package.
    This method uses the photoutils Background 2D background estimation using a
    SExtracktorBackground estimator, a 3-sigma clip to the data, and a masking
    of all pixels above 3e5 in flux.

    Parameters:
        ffi (ndarray): A single TESS Full Frame Image in the form of a 2D array.

    Returns:
        ndarray: Estimated background with the same size as the input image.

    .. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
    .. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>
    """

    mask = ~np.isfinite(ffi)
    mask |= (ffi > 3e5)

    # Estimate the background:
    sigma_clip = SigmaClip(sigma=3.0, iters=5) #Sigma clip the data
    bkg_estimator = SExtractorBackground()     #Call background estimator
    bkg = Background2D(ffi, (64, 64),          #Estimate background on sigma clipped data
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=mask,
            exclude_percentile=50)

    return bkg.background


if __name__ == '__main__':
    plt.close('all')

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

    #Run background estimation
    est_bkg = fit_background(ffi)

    #Plot background difference
    '''Plotting: all'''
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_title(ffi_type)

    fdiff, adiff = plt.subplots()
    diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
    fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
    adiff.set_title('Estimated bkg - True bkg')

    fest, aest = plt.subplots()
    est = aest.imshow(np.log10(est_bkg), cmap='Blues_r', origin='lower')
    fest.colorbar(est, label=r'$log_{10}$(Flux)')
    aest.set_title('Background estimated using the RH_estimate method')

    plt.show('all')
    plt.close('all')
