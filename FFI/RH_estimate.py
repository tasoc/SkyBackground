#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#Rasmus HANDBERG
#Oliver James HALL

import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.io.fits as pyfits
from photutils import Background2D, SigmaClip, SExtractorBackground
from astropy.stats import sigma_clip

'''
Estimate the background of a Full Frame Image (FFI) using the 'photutils' package.

Includes a '__main__' for independent test runs on local machines.

Parameters
-----------
ffi : float, numpy array
        A single TESS Full Frame Image in the form of a 2D array.
-----------

Output
-----------
bkg.background : float, numpy array
        A background estimate in the same style and shape as the FFI input.
-----------

TO DO:
-   Write own Sigma Clip not dependent on the package, which doesnt work
    on all machines.

'''

def fit_background(ffi):
	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(ffi)
	mask |= (ffi > 2e5)

	# Estimate the background:
	sigma_clip = SigmaClip(sigma=3.0, iters=5)
	bkg_estimator = SExtractorBackground()
	bkg = Background2D(ffi, (64, 64),
		filter_size=(3, 3),
		sigma_clip=sigma_clip,
		bkg_estimator=bkg_estimator,
		mask=mask)

	return bkg.background


if __name__ == '__main__':
    plt.close('all')

    # Load file:
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[2]
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

    est_bkg = fit_background(ffi)


    '''Plotting: all'''
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_title(ffi_type)

    fdiff, adiff = plt.subplots()
    diff = adiff.imshow(np.log10(est_bkg) - np.log10(bkg), origin='lower')
    fdiff.colorbar(diff, label='Estimated Bkg - True Bkg (both in log10 space)')
    adiff.set_title('Estimated bkg - True bkg')

    plt.show('all')
    plt.close('all')
