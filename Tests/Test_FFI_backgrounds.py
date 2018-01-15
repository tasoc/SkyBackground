#!/bin/env python
# -*- coding: utf-8 -*-

#AUTHORS:
#Oliver James HALL

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy import interpolate

import sys
import glob

sys.path.append('../FFI/')
from CvE_estimate import fit_background as Efit_bkg
# from RH_estimate import fit_background as Rfit_bkg
from OJH_estimate import fit_background as Hfit_bkg


'''
Tests the 3 different types of background estimation for FFIs, and returns
some simple metrics on their quality.
'''
def load_file(ffi_type):
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

    return ffi, bkg

if __name__ == "__main__":
    plt.close('all')

    # Load file:
    ffis = ['ffi_north', 'ffi_south', 'ffi_cluster']
    ffi_type = ffis[0]
    ffi, bkg = load_file(ffi_type)

    CvE = Efit_bkg(ffi)
    OJH = Hfit_bkg(ffi)

    stdCvE = np.std(CvE-bkg)
    stdOJH = np.std(OJH-bkg)

    '''Plotting: all'''
    plt.close('all')
    fig, ax = plt.subplots()
    im = ax.imshow(np.log10(ffi),cmap='Blues_r', origin='lower')
    fig.colorbar(im,label=r'$log_{10}$(Flux)')
    ax.set_title(ffi_type)

    fC, aC = plt.subplots()
    diffC = aC.imshow(np.log10(CvE) - np.log10(bkg), origin='lower')
    fC.colorbar(diffC, label='Estimated Bkg - True Bkg (both in log10 space)')
    aC.set_title('CvE_estimate | Estimated bkg - True bkg')

    fH, aH = plt.subplots()
    diffH = aH.imshow(np.log10(OJH) - np.log10(bkg),origin='lower')
    fH.colorbar(diffH, label=r'$log_{10}$(Flux)')
    aH.set_title('OJH_estimate | Estimated bkg - True bkg')

    fhist, ahist = plt.subplots(2, sharex=True)
    ahist[0].hist((CvE-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))
    ahist[1].hist((OJH-bkg).ravel(),histtype='step',bins=int(np.sqrt(len(bkg.ravel()))))
    ahist[0].set_title('Histogram of residuals for CvE')
    ahist[1].set_title('Histogram of residuals for OJH')
    ahist[1].set_xlabel('Background Estimate - True Background')


    plt.show('all')
    plt.close('all')
